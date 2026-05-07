# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPx-License-Identifier: CC-BY-NC-4.0

import torch

from escnn import gspaces
from escnn import nn

from resfit.rl_finetuning.config.rlpd import ActorConfig
from resfit.rl_finetuning.off_policy.common_utils import utils
from resfit.rl_finetuning.equi_off_policy.rl.equi_rl_utils import *

from einops import rearrange

import time


class Actor(torch.nn.Module):
    def __init__(self,
                 group,
                 vis_ih_type: nn.FieldType,
                 prop_type:   nn.FieldType,
                 action_type: nn.FieldType,
                 action_layout,                 # NEW: per-irrep slice layout
                 hidden_dim: int,
                 action_shape,
                 num_layers: int,
                 initialize: bool,
                 cfg: ActorConfig,
                 residual_actor: bool = True):
        super().__init__()
        assert residual_actor, "Non-residual mode dropped — actor is residual-only"
        self.residual_actor = True
        self.cfg = cfg
        self.group = group

        self.vis_ih_type = vis_ih_type
        self.prop_type   = prop_type
        self.action_type = action_type
        self.action_layout = action_layout

        use_deltaortho = cfg.orth
        escnn_init = not use_deltaortho and initialize
        use_layer_norm = getattr(cfg, "use_layer_norm", False)

        # ---- Visual projection (mirrors Critic) ----
        vp_n_regular = getattr(cfg, "visual_proj_n_regular", 16)
        self.vp_type = nn.FieldType(group, vp_n_regular * [group.regular_repr])

        vp_layers = [nn.Linear(vis_ih_type, self.vp_type, initialize=escnn_init)]
        if use_layer_norm:
            vp_layers.append(FieldNorm(self.vp_type))
        vp_layers.append(nn.ReLU(self.vp_type))
        self.visual_proj = torch.nn.Sequential(*vp_layers)

        # ---- Trunk: bottleneck on (vp + prop + base_action) ----
        trunk_in_type = self.vp_type + prop_type + action_type
        self.trunk_proj_type = nn.FieldType(group, hidden_dim * [group.regular_repr])

        ip_layers = [nn.Linear(trunk_in_type, self.trunk_proj_type, initialize=escnn_init)]
        if use_layer_norm:
            ip_layers.append(FieldNorm(self.trunk_proj_type))
        ip_layers.append(nn.ReLU(self.trunk_proj_type))
        self.input_proj = torch.nn.Sequential(*ip_layers)

        # ---- Policy MLP: trunk_out + prop + base_action via skip injection ----
        pol_in_type     = self.trunk_proj_type + prop_type + action_type
        pol_hidden_type = nn.FieldType(group, hidden_dim * [group.regular_repr])
        pol_out_type    = nn.FieldType(group, action_shape)

        layers = [nn.Linear(pol_in_type, pol_hidden_type, initialize=escnn_init)]
        if use_layer_norm:
            layers.append(FieldNorm(pol_hidden_type))
        layers.append(nn.ReLU(pol_hidden_type, inplace=True))

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(pol_hidden_type, pol_hidden_type, initialize=escnn_init))
            if use_layer_norm:
                layers.append(FieldNorm(pol_hidden_type))
            layers.append(nn.ReLU(pol_hidden_type, inplace=True))

        # Final projection — NO FieldNorm (out_type contains trivial irreps)
        layers.append(nn.Linear(pol_hidden_type, pol_out_type, initialize=escnn_init))
        self.policy = torch.nn.Sequential(*layers)

        # ---- Initialization ----
        if use_deltaortho:
            apply_deltaortho_init(self.visual_proj, exclude_final_layer=False)
            apply_deltaortho_init(self.input_proj,  exclude_final_layer=False)
            apply_deltaortho_init(self.policy,      exclude_final_layer=False)

        if cfg.actor_last_layer_init_scale is not None:
            scale_final_equi_layer(self.policy, cfg.actor_last_layer_init_scale)

    def forward(self, feat: nn.GeometricTensor, std=None):
        """
        Returns:
            If std is None:        scaled_mu                (deterministic mean, no clip)
            If std is a tensor/scalar: clipped noisy action (mu + noise, equivariantly clipped to [-1, 1])
        """
        assert isinstance(feat, nn.GeometricTensor)
        # feat layout: [vis | ih | prop | base_action], type = enc_out_type_actor
        n = feat.tensor.shape[1]
        vis_ih_end = self.vis_ih_type.size
        prop_end   = vis_ih_end + self.prop_type.size

        vis_ih      = slice_gt(feat, 0,          vis_ih_end, self.vis_ih_type)
        prop        = slice_gt(feat, vis_ih_end, prop_end,   self.prop_type)
        base_action = slice_gt(feat, prop_end,   n,          self.action_type)

        v = self.visual_proj(vis_ih)
        trunk_in  = cat_gts([v, prop, base_action])
        z         = self.input_proj(trunk_in)
        policy_in = cat_gts([z, prop, base_action])

        mu: torch.Tensor = self.policy(policy_in).tensor
        scaled_mu = mu * self.cfg.action_scale

        if std is None:
            return scaled_mu

        noise = torch.randn_like(scaled_mu) * std
        return equi_clip(scaled_mu + noise, self.action_layout, bound=1.0)