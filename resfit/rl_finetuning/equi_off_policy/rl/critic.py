from __future__ import annotations

import torch
from escnn import nn

from resfit.rl_finetuning.config.rlpd import CriticConfig
from resfit.rl_finetuning.off_policy.common_utils import utils
from resfit.rl_finetuning.equi_off_policy.rl.equi_rl_utils import *


class EquiHeadMLP(torch.nn.Module):
    def __init__(self,
                 group,
                 in_type,
                 hidden_type,
                 out_type,
                 num_layers: int,
                 use_norms: bool,
                 dropout: float,
                 escnn_init: bool):
        super().__init__()

        layers = []
        
        layers.append(nn.Linear(in_type, hidden_type, initialize=escnn_init))
        # if use_norms:
            # layers.append(nn.IIDBatchNorm1d(hidden_type, track_running_stats=True))
        # layers.append(nn.ReLU(hidden_type, inplace=False))

        # layers.append(nn.Linear(hidden_type, hidden_type, initialize=escnn_init))
        # if use_norms:
            # layers.append(nn.IIDBatchNorm1d(hidden_type, track_running_stats=True))
        # layers.append(nn.ReLU(hidden_type, inplace=False))

        gpool = nn.GroupPooling(hidden_type)
        layers.append(gpool)
        layers.append(nn.Linear(gpool.out_type, out_type, initialize=escnn_init))
        self.net = torch.nn.Sequential(*layers)

        # with torch.no_grad():
        #     # after building the head, scale down the final trivial→trivial layer
        #     final = self.net[-1]  # the nn.Linear after GroupPooling
        #     final.weights.mul_(1e-3)  # escnn stores basis coeffs in .weights
        
    def forward(self, feat):
        return self.net(feat)


class EquiQEnsemble(torch.nn.Module):
    def __init__(self,
                 group,
                 in_type,
                 hidden_type,
                 out_type,
                 num_layers: int,
                 num_heads: int,
                 use_norms: bool,
                 dropout: float,
                 use_orth_init: bool,
                 escnn_init: bool):
        super().__init__()
        
        self.heads = torch.nn.ModuleList([
            EquiHeadMLP(group, in_type, hidden_type, out_type, num_layers, use_norms, dropout, escnn_init) 
            for _ in range(num_heads)
        ])

        # if use_orth_init:
        #     for head in self.heads:
        #         apply_deltaortho_init(head.net, exclude_final_layer=False)

    def forward(self, feat):
        return torch.stack([head(feat).tensor for head in self.heads], dim=0)


class Critic(torch.nn.Module):
    def __init__(self,
                 group,
                 vis_ih_type: nn.FieldType,
                 prop_type:   nn.FieldType,
                 action_type: nn.FieldType,
                 hidden_dim: int,
                 num_layers: int,
                 num_q: int,
                 dropout: float,
                 use_norms: bool,
                 use_orth_init: bool,
                 loss_cfg,
                 min_q_heads: int,
                 policy_gradient_type: str):
        super().__init__()
        self.group = group
        self.vis_ih_type = vis_ih_type
        self.prop_type   = prop_type
        self.action_type = action_type

        self.loss_cfg             = loss_cfg
        self.min_q_heads          = min_q_heads
        self.policy_gradient_type = policy_gradient_type

        escnn_init = True

        # ---- Visual projection ----
        self.vp_type = nn.FieldType(group, hidden_dim * [group.regular_repr])
        vp_layers = [nn.Linear(vis_ih_type, self.vp_type, initialize=escnn_init)]
        # if use_norms:
        #     vp_layers.append(nn.IIDBatchNorm1d(self.vp_type, track_running_stats=True))
        vp_layers.append(nn.ReLU(self.vp_type))
        self.visual_proj = torch.nn.Sequential(*vp_layers)

        # ---- Trunk (visual_proj_out + prop + action) ----
        trunk_in_type        = self.vp_type + prop_type + action_type
        self.trunk_proj_type = nn.FieldType(group, hidden_dim * [group.regular_repr])
        ip_layers = [nn.Linear(trunk_in_type, self.trunk_proj_type, initialize=escnn_init)]
        # if use_norms:
        #     ip_layers.append(nn.IIDBatchNorm1d(self.trunk_proj_type, track_running_stats=True))
        ip_layers.append(nn.ReLU(self.trunk_proj_type))
        self.input_proj = torch.nn.Sequential(*ip_layers)

        # ---- Q-ensemble: heads see (trunk_out + prop + action) via skip ----
        head_in_type     = self.trunk_proj_type + prop_type + action_type
        head_hidden_type = nn.FieldType(group, hidden_dim * [group.regular_repr])

        output_dim    = 1
        head_out_type = nn.FieldType(group, output_dim * [group.trivial_repr])

        self.q_ensemble = EquiQEnsemble(
            group         = group,
            in_type       = head_in_type,
            hidden_type   = head_hidden_type,
            out_type      = head_out_type,
            num_layers    = num_layers,
            num_heads     = num_q,
            use_norms     = use_norms,
            dropout       = dropout,
            use_orth_init = use_orth_init,
            escnn_init    = escnn_init,
        )

        # if use_orth_init:
        #     apply_deltaortho_init(self.visual_proj, exclude_final_layer=False)
        #     apply_deltaortho_init(self.input_proj,  exclude_final_layer=False)

    @staticmethod
    def _logits_to_q(probs, support):
        return (probs * support).sum(-1, keepdim=True)

    def forward(self, feat: nn.GeometricTensor, act: torch.Tensor, *, return_logits: bool = False):
        n          = feat.tensor.shape[1]
        vis_ih_end = self.vis_ih_type.size

        vis_ih = slice_gt(feat, 0,          vis_ih_end, self.vis_ih_type)
        prop   = slice_gt(feat, vis_ih_end, n,          self.prop_type)
        act_gt = nn.GeometricTensor(act, self.action_type)

        v        = self.visual_proj(vis_ih)
        trunk_in = cat_gts([v, prop, act_gt])

        z       = self.input_proj(trunk_in)
        head_in = cat_gts([z, prop, act_gt])

        return self.q_ensemble(head_in)

    def q_value(self, feat, act):
        q_out = self.forward(feat, act)
        num_heads = min(self.min_q_heads, q_out.shape[0])
        idx = torch.randperm(q_out.shape[0], device=q_out.device)[:num_heads]
        return torch.min(q_out.index_select(0, idx), dim=0).values

    def q_value_for_policy(self, feat, act):
        q_out = self.forward(feat, act)
        if self.policy_gradient_type == "ensemble_mean":
            return q_out.mean(dim=0)
        if self.policy_gradient_type == "min_random_pair":
            num_heads = min(self.min_q_heads, q_out.shape[0])
            idx = torch.randperm(q_out.shape[0], device=q_out.device)[:num_heads]
            return torch.min(q_out.index_select(0, idx), dim=0).values
        if self.policy_gradient_type == "q1":
            return q_out[0]
        raise ValueError(f"Unknown policy_gradient_type: {self.policy_gradient_type}")