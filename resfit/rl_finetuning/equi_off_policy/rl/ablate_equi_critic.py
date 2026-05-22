# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

import math
import time

import torch
from torch.func import functional_call, stack_module_state, vmap
from torch.nn import functional

from resfit.rl_finetuning.config.rlpd import CriticConfig
from resfit.rl_finetuning.off_policy.common_utils import utils

# ---------------------------------------------------------------------------
# Non-equivariant MLP head
# ---------------------------------------------------------------------------

class HeadMLP(torch.nn.Module):
    """Single MLP head for the ensemble."""

    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int = 2,
                 use_norms: bool = True,
                 dropout: float = 0.0):
        super().__init__()

        layers = []

        layers.append(torch.nn.Linear(in_dim, hidden_dim))
        # if use_norms:
        #     layers.append(torch.nn.LayerNorm(hidden_dim))
        layers.append(torch.nn.ReLU())  # no inplace — required for vmap

        layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
        # if use_norms:
        #     layers.append(torch.nn.LayerNorm(hidden_dim))
        layers.append(torch.nn.ReLU())  # no inplace — required for vmap

        layers.append(torch.nn.Linear(hidden_dim, output_dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)


class QEnsemble(torch.nn.Module):
    """vmap-parallelised ensemble of HeadMLPs."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        num_heads: int = 2,
        use_norms: bool = True,
        dropout: float = 0.0,
        use_orth_init: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads

        heads = [
            HeadMLP(in_dim, hidden_dim, output_dim, num_layers, use_norms, dropout)
            for _ in range(num_heads)
        ]

        # if use_orth_init:
        #     for head in heads:
        #         head.net.apply(utils.orth_weight_init)

        self.params, self.buffers = stack_module_state(heads)

        self._head_template = HeadMLP(in_dim, hidden_dim, output_dim, num_layers, use_norms, dropout)

        for name, param in self.params.items():
            self.register_parameter(
                f"_vmap_param_{name.replace('.', '_')}",
                torch.nn.Parameter(param),
            )
        for name, buffer in self.buffers.items():
            self.register_buffer(
                f"_vmap_buffer_{name.replace('.', '_')}",
                buffer,
            )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        current_params = {
            name: getattr(self, f"_vmap_param_{name.replace('.', '_')}")
            for name in self.params
        }
        current_buffers = {
            name: getattr(self, f"_vmap_buffer_{name.replace('.', '_')}")
            for name in self.buffers
        }

        def f_one_head(p, b, z):
            return functional_call(self._head_template, (p, b), (z,))

        return vmap(f_one_head, in_dims=(0, 0, None))(
            current_params, current_buffers, feat
        )


# ---------------------------------------------------------------------------
# Non-equivariant Critic with visual-projection layer.
#
# Pipeline:
#   feat = [vis (1024), ih (96), prop (11)]
#                                                       ┌─ prop ─┐
#   [vis, ih] ──► visual_proj ──► v ──► concat(v, prop, act) ──► input_proj ──► z
#                                                       └─ act ──┘
#
#   z, prop, act ──► concat ──► q_ensemble (heads) ──► logits per head
# ---------------------------------------------------------------------------

class Critic(torch.nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 action_dim: int,
                 prop_dim: int,
                 num_layers: int,
                 N: int,
                 
                 loss,
                 num_q,
                 use_norms,
                 use_orth_init,
                 dropout,
                 min_q_heads,
                 policy_gradient_type,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.prop_dim = prop_dim

        output_dim = 1
        hidden_dim_full = hidden_dim * N
        feat_dim = in_dim - action_dim
        self.feat_dim = feat_dim
        self.vis_dim = feat_dim - prop_dim   # everything except prop tail

        self.loss_cfg = loss
        self.num_q = num_q
        self.policy_gradient_type = policy_gradient_type
        self.use_orth_init = use_orth_init
        self.use_norms = use_norms
        self.min_q_heads = min_q_heads
        self.dropout = dropout

        # ---- Visual Projection: agentview + inhand -> visual features ----
        vis_proj_layers = [torch.nn.Linear(self.vis_dim, hidden_dim_full)]
        # if use_norms:
        #     vis_proj_layers.append(torch.nn.LayerNorm(hidden_dim_full))
        vis_proj_layers.append(torch.nn.ReLU())
        self.visual_proj = torch.nn.Sequential(*vis_proj_layers)

        # ---- Trunk: visual features + prop + action -> all features ----
        trunk_in_dim = hidden_dim_full + prop_dim + action_dim
        layers = [torch.nn.Linear(trunk_in_dim, hidden_dim_full)]
        # if use_norms:
        #     layers.append(torch.nn.GroupNorm(hidden_dim_full // N, hidden_dim_full))
        layers.append(torch.nn.ReLU())
        self.input_proj = torch.nn.Sequential(*layers)

        # ---- Heads: all features + prop + action ----
        head_in_dim = hidden_dim_full + prop_dim + action_dim
        self.q_ensemble = QEnsemble(
            in_dim=head_in_dim,
            hidden_dim=hidden_dim_full,
            output_dim=output_dim,
            num_layers=num_layers,
            num_heads=num_q,
            use_norms=use_norms,
            dropout=dropout,
            use_orth_init=use_orth_init,
        )

    @staticmethod
    def _logits_to_q(probs, support):
        return (probs * support).sum(-1, keepdim=True)

    def forward(self, feat, act, *, return_logits=False):
        # Slice prop off the tail of feat; everything before prop is visual.
        # feat layout (from ResObsEnc): [vis (1024), ih (96), prop (11)]
        prop = feat[:, -self.prop_dim:]
        vis  = feat[:, :-self.prop_dim]                    # [B, vis_dim]

        # Compress the visual stream so action+prop have proportional weight
        v = self.visual_proj(vis)                           # [B, visual_proj_dim]

        # Trunk: bottleneck (compressed_vis, prop, act)
        z = self.input_proj(torch.cat([v, prop, act], dim=-1))   # [B, proj_dim]

        # Head skip: re-inject low-dim semantically-important inputs
        z = torch.cat([z, prop, act], dim=-1)                    # [B, proj_dim + prop_dim + action_dim]

        logits_per_head = self.q_ensemble(z)

        return logits_per_head

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