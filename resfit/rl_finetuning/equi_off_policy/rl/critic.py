# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

import math
import time

import torch
# from torch import nn

from escnn import gspaces
from escnn import nn

# Import vmap functionality (PyTorch 2.1+)
from torch.func import functional_call, stack_module_state, vmap
from torch.nn import functional

from resfit.rl_finetuning.config.rlpd import CriticConfig
from resfit.rl_finetuning.off_policy.common_utils import utils
from resfit.rl_finetuning.equi_off_policy.rl.equi_rl_utils import *


class Critic(torch.nn.Module):
    def __init__(self,
                 group,
                 vis_ih_type: nn.FieldType,
                 prop_type:   nn.FieldType,
                 action_type: nn.FieldType,
                 hidden_dim: int,
                 num_layers: int,
                 initialize: bool,
                 cfg: CriticConfig):
        super().__init__()
        self.cfg = cfg
        self.group = group
        self.loss_cfg = cfg.loss

        self.vis_ih_type = vis_ih_type
        self.prop_type   = prop_type
        self.action_type = action_type

        # ---- Visual projection (analog of non-eq cfg.visual_proj_dim) ----
        # Compresses [vis | ih] (≈1120 dims) → vp_n_regular regular_reps,
        # so action+prop have proportional weight at the trunk concat.
        # Pure regular_repr output → FieldNorm is safe here.
        vp_n_regular = getattr(cfg, "visual_proj_n_regular", 16)
        self.vp_type = nn.FieldType(group, vp_n_regular * [group.regular_repr])

        vp_layers = [nn.Linear(vis_ih_type, self.vp_type, initialize=initialize)]
        if cfg.use_layer_norm:
            vp_layers.append(FieldNorm(self.vp_type))
        vp_layers.append(nn.ReLU(self.vp_type))
        self.visual_proj = torch.nn.Sequential(*vp_layers)

        # ---- Trunk: bottleneck on (visual_proj_out + prop + action) ----
        trunk_in_type = self.vp_type + prop_type + action_type
        self.trunk_proj_type = nn.FieldType(group, hidden_dim * [group.regular_repr])

        ip_layers = [nn.Linear(trunk_in_type, self.trunk_proj_type, initialize=initialize)]
        if cfg.use_layer_norm:
            ip_layers.append(FieldNorm(self.trunk_proj_type))
        ip_layers.append(nn.ReLU(self.trunk_proj_type))
        self.input_proj = torch.nn.Sequential(*ip_layers)

        # ---- Q-ensemble: heads see (trunk_out + prop + action) via skip ----
        head_in_type = self.trunk_proj_type + prop_type + action_type

        output_dim = self.loss_cfg.n_bins if self.loss_cfg.type in {"hl_gauss", "c51"} else 1
        head_hidden_type = nn.FieldType(group, hidden_dim * [group.regular_repr])
        head_out_type = nn.FieldType(group, output_dim * [group.trivial_repr])

        orth = getattr(cfg, "orth", False)
        self.q_ensemble = EquiQEnsemble(
            group          = group,
            in_type        = head_in_type,
            hidden_type    = head_hidden_type,
            out_type       = head_out_type,
            initialize     = initialize,
            num_layers     = num_layers,
            num_heads      = cfg.num_q,
            use_layer_norm = cfg.use_layer_norm,
            orth           = orth,
        )

        # ---- Loss objects ----
        if self.loss_cfg.type == "hl_gauss":
            sigma_val = None if self.loss_cfg.sigma < 0 else self.loss_cfg.sigma
            self.hl_loss = HLGaussLoss(
                min_value=self.loss_cfg.v_min,
                max_value=self.loss_cfg.v_max,
                num_bins=self.loss_cfg.n_bins,
                sigma=sigma_val,
            )
        elif self.loss_cfg.type == "c51":
            self.c51_loss = C51Loss(
                v_min=self.loss_cfg.v_min,
                v_max=self.loss_cfg.v_max,
                num_atoms=self.loss_cfg.n_bins,
            )

        # ---- Initialization (orth on the new projections too) ----
        # if orth:
        #     apply_deltaortho_init(self.visual_proj, exclude_final_layer=False)
        #     apply_deltaortho_init(self.input_proj,  exclude_final_layer=False)
            # q_ensemble heads are already orth-init'd inside EquiQEnsemble

    @staticmethod
    def _logits_to_q(probs, support):
        return (probs * support).sum(-1, keepdim=True)

    def forward(self, feat: nn.GeometricTensor, act: torch.Tensor,
                *, return_logits: bool = False):
        # feat layout: [vis | ih | prop],  type = enc_out_type_critic
        n = feat.tensor.shape[1]
        vis_ih_end = self.vis_ih_type.size

        vis_ih = slice_gt(feat, 0,           vis_ih_end, self.vis_ih_type)
        prop   = slice_gt(feat, vis_ih_end,  n,          self.prop_type)
        act_gt = nn.GeometricTensor(act, self.action_type)

        # Stage 1: compress visual stream
        v = self.visual_proj(vis_ih)

        # Stage 2: trunk bottleneck on (v, prop, act)
        trunk_in = cat_gts([v, prop, act_gt])
        z = self.input_proj(trunk_in)

        # Stage 3: skip-reinject prop and act before the Q-heads
        head_in = cat_gts([z, prop, act_gt])
        logits_per_head = self.q_ensemble(head_in)

        return logits_per_head

    # q_value and q_value_for_policy unchanged — they call self.forward()
    def q_value(self, feat, act):
        q_out = self.forward(feat, act)
        num_heads = min(self.cfg.min_q_heads, q_out.shape[0])
        idx = torch.randperm(q_out.shape[0], device=q_out.device)[:num_heads]
        return torch.min(q_out.index_select(0, idx), dim=0).values

    def q_value_for_policy(self, feat, act):
        q_out = self.forward(feat, act)
        if self.cfg.policy_gradient_type == "ensemble_mean":
            return q_out.mean(dim=0)
        if self.cfg.policy_gradient_type == "min_random_pair":
            num_heads = min(self.cfg.min_q_heads, q_out.shape[0])
            idx = torch.randperm(q_out.shape[0], device=q_out.device)[:num_heads]
            return torch.min(q_out.index_select(0, idx), dim=0).values
        if self.cfg.policy_gradient_type == "q1":
            return q_out[0]
        raise ValueError(f"Unknown policy_gradient_type: {self.cfg.policy_gradient_type}")

class EquiQEnsemble(torch.nn.Module):
    def __init__(
        self,
        group, 
        in_type,
        hidden_type,
        out_type,
        initialize,
        num_layers,
        num_heads: int = 2,
        use_layer_norm: bool = True,
        orth: bool = False,
    ):
        super().__init__()

        # When using deltaortho, skip the slow default generalized_he_init
        escnn_init = not orth and initialize

        self.heads = torch.nn.ModuleList([EquiHeadMLP(group=group, 
                                                      in_type=in_type, 
                                                      hidden_type=hidden_type, 
                                                      out_type=out_type, 
                                                      initialize=escnn_init,
                                                      num_layers=num_layers,
                                                      use_layer_norm=use_layer_norm) for _ in range(num_heads)]
                                        )

        # Mirrors SpatialEmbQEnsemble._init_per_head_params(orth):
        # apply deltaortho init to all linear layers in every head
        if orth:
            for head in self.heads:
                apply_deltaortho_init(head.net, exclude_final_layer=False)

    def forward(self, feat):

        q_vals = []
        for head in self.heads:
            q_vals.append(head(feat).tensor)

        ret = torch.stack(q_vals, dim=0)

        return ret


class EquiHeadMLP(torch.nn.Module):
    """Single MLP head for the ensemble."""
 
    def __init__(self,
                 group,
                 in_type,
                 hidden_type,
                 out_type,
                 initialize,
                 num_layers: int = 2,
                 use_layer_norm: bool = True):
        super().__init__()
 
        pooled = nn.FieldType(group, len(hidden_type.representations) * [group.trivial_repr])
 
        # First hidden block: Linear -> (FieldNorm) -> ReLU
        layers = [nn.Linear(in_type, hidden_type, initialize=initialize)]
        if use_layer_norm:
            layers.append(FieldNorm(hidden_type))
        layers.append(nn.ReLU(hidden_type))
 
        # Remaining hidden blocks
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_type, hidden_type, initialize=initialize))
            if use_layer_norm:
                layers.append(FieldNorm(hidden_type))
            layers.append(nn.ReLU(hidden_type))
 
        layers.append(nn.GroupPooling(hidden_type))
        layers.append(nn.Linear(pooled, out_type, initialize=initialize))
        self.net = torch.nn.Sequential(*layers)
 
    def forward(self, feat):
        ret = self.net(feat)
        return ret

