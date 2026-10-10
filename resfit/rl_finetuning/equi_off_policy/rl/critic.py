from __future__ import annotations

import torch
from escnn import nn

from resfit.rl_finetuning.config.rlpd import CriticConfig
from resfit.rl_finetuning.off_policy.common_utils import utils
from resfit.rl_finetuning.equi_off_policy.rl.equi_rl_utils import *
# After the star import, so nothing it brings in can shadow it.
from resfit.rl_finetuning.equi_off_policy.common_utils.escnn_fixes import GroupPooling


class EquiHeadMLP(torch.nn.Module):
    """One Q head: hidden Linears on regular fields, GroupPooling, Linear to Q.

    GroupPooling (a max over each field's channels) is what makes Q invariant
    rather than equivariant (STANDARDS 2.2). Each hidden Linear is followed by
    `norm`, and every hidden layer but the last by a ReLU: with one hidden layer,
    as every run before 2026-09-29 had, the pooling's max is the only
    nonlinearity. `pooled_norm` (POOLED_NORMS) normalizes the pooled features,
    which are invariant, before the output Linear.
    """

    def __init__(self, in_type, hidden_type, out_type, num_layers: int, norm: str, pooled_norm: str):
        super().__init__()
        assert num_layers >= 1, f"a Q head needs a hidden layer, got num_layers={num_layers}"

        layers = []
        current = in_type
        for i in range(num_layers):
            layers.append(nn.Linear(current, hidden_type, initialize=True))
            layers += hidden_norm(norm, hidden_type)
            if i < num_layers - 1:
                layers.append(nn.ReLU(hidden_type, inplace=False))
            current = hidden_type

        # escnn's GroupPooling, minus its GPU-scalar slicing: escnn_fixes.py.
        gpool = GroupPooling(hidden_type)
        layers.append(gpool)
        layers += pooled_feature_norm(pooled_norm, gpool.out_type)
        layers.append(nn.Linear(gpool.out_type, out_type, initialize=True))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, feat):
        return self.net(feat)


class EquiQEnsemble(torch.nn.Module):
    def __init__(self, in_type, hidden_type, out_type, num_layers: int, num_heads: int, norm: str,
                 pooled_norm: str):
        super().__init__()
        self.heads = torch.nn.ModuleList([
            EquiHeadMLP(in_type, hidden_type, out_type, num_layers, norm, pooled_norm)
            for _ in range(num_heads)
        ])

    def forward(self, feat):
        return torch.stack([head(feat).tensor for head in self.heads], dim=0)


class Critic(torch.nn.Module):
    """The invariant critic: Q(g.s, g.a) = Q(s, a).

    visual projection -> trunk projection -> an ensemble of num_q heads, each
    hidden layer followed by `norm` (HIDDEN_NORMS) and each head's pooled
    features by `pooled_norm` (POOLED_NORMS).
    """

    def __init__(self,
                 group,
                 vis_ih_type: nn.FieldType,
                 prop_type:   nn.FieldType,
                 action_type: nn.FieldType,
                 hidden_dim: int,
                 num_layers: int,
                 num_q: int,
                 dropout: float,
                 norm: str,
                 pooled_norm: str,
                 orth_init: bool,
                 last_layer_init_scale: float | None,
                 use_visual_proj: bool,
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

        # ---- Visual projection; Identity feeds the encoder features straight to the trunk ----
        if use_visual_proj:
            self.vp_type = nn.FieldType(group, hidden_dim * [group.regular_repr])
            self.visual_proj = torch.nn.Sequential(
                nn.Linear(vis_ih_type, self.vp_type, initialize=True),
                *hidden_norm(norm, self.vp_type),
                nn.ReLU(self.vp_type),
            )
        else:
            self.vp_type = vis_ih_type
            self.visual_proj = torch.nn.Identity()

        # ---- Trunk (visual_proj_out + prop + action) ----
        trunk_in_type        = self.vp_type + prop_type + action_type
        self.trunk_proj_type = nn.FieldType(group, hidden_dim * [group.regular_repr])
        self.input_proj = torch.nn.Sequential(
            nn.Linear(trunk_in_type, self.trunk_proj_type, initialize=True),
            *hidden_norm(norm, self.trunk_proj_type),
            nn.ReLU(self.trunk_proj_type),
        )

        # ---- Q-ensemble: heads see (trunk_out + prop + action) via skip ----
        head_in_type     = self.trunk_proj_type + prop_type + action_type
        head_hidden_type = nn.FieldType(group, hidden_dim * [group.regular_repr])
        head_out_type    = nn.FieldType(group, [group.trivial_repr])

        self.q_ensemble = EquiQEnsemble(
            in_type     = head_in_type,
            hidden_type = head_hidden_type,
            out_type    = head_out_type,
            num_layers  = num_layers,
            num_heads   = num_q,
            norm        = norm,
            pooled_norm = pooled_norm,
        )

        # Delta-orthonormal init draws from numpy's global RNG, so turning it on
        # also moves every later draw, including the actor's init.
        if orth_init:
            apply_deltaortho_init(self.visual_proj, exclude_final_layer=False)
            apply_deltaortho_init(self.input_proj,  exclude_final_layer=False)
            for head in self.q_ensemble.heads:
                apply_deltaortho_init(head.net, exclude_final_layer=False)
        if last_layer_init_scale is not None:
            for head in self.q_ensemble.heads:
                scale_final_equi_layer(head.net, last_layer_init_scale)

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
