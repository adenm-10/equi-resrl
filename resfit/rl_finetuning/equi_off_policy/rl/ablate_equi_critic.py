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
                 use_layer_norm: bool = True,
                 dropout: float = 0.0):
        super().__init__()

        layers = []
        current_dim = in_dim

        for _ in range(num_layers):
            layers.append(torch.nn.Linear(current_dim, hidden_dim))
            if use_layer_norm:
                layers.append(torch.nn.LayerNorm(hidden_dim))
            if dropout > 0:
                layers.append(torch.nn.Dropout(dropout))
            layers.append(torch.nn.ReLU())  # no inplace — required for vmap
            current_dim = hidden_dim

        layers.append(torch.nn.Linear(current_dim, output_dim))
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
        use_layer_norm: bool = True,
        dropout: float = 0.0,
        orth: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads

        heads = [
            HeadMLP(in_dim, hidden_dim, output_dim, num_layers, use_layer_norm, dropout)
            for _ in range(num_heads)
        ]

        if orth:
            for head in heads:
                head.net.apply(utils.orth_weight_init)

        self.params, self.buffers = stack_module_state(heads)

        self._head_template = HeadMLP(in_dim, hidden_dim, output_dim, num_layers, use_layer_norm, dropout)

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
#
# The visual_proj compresses the 1120-dim visual stream down to
# `visual_proj_dim` (default 128, matching baseline patch dim) before action
# and prop are concatenated. This brings action's share of trunk input from
# ~0.6% (current: 7/1138) up to ~4.8% (with proj_dim=128: 7/146), matching
# the baseline's per-token proportions and removing the initialization
# disadvantage for action signal.
#
# The head-skip concat is unchanged in spirit: [z, prop, act].
# ---------------------------------------------------------------------------

class Critic(torch.nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 action_dim: int,
                 prop_dim: int,
                 num_layers: int,
                 cfg: CriticConfig,
                 N: int = 8):
        super().__init__()
        self.cfg = cfg
        self.loss_cfg = cfg.loss
        self.action_dim = action_dim
        self.prop_dim = prop_dim

        # NOTE on `in_dim`: by convention from the original code, in_dim is
        # passed as `enc_out_dim_critic + action_dim`, but the actual `feat`
        # tensor only has shape [B, enc_out_dim_critic] — action is concatenated
        # separately inside forward(). So feat_dim = in_dim - action_dim, and
        # the visual portion of feat is feat_dim - prop_dim.
        feat_dim = in_dim - action_dim
        self.feat_dim = feat_dim
        self.vis_dim = feat_dim - prop_dim   # everything except prop tail

        # Visual projection dim — knob for action proportion.
        # Default to baseline-parity (128); allow override via cfg.
        self.visual_proj_dim = getattr(cfg, "visual_proj_dim", 128)

        output_dim = self.loss_cfg.n_bins if self.loss_cfg.type in {"hl_gauss", "c51"} else 1
        hidden_dim_full = proj_dim = hidden_dim * N

        # ---- NEW: visual projection ----
        # Compress [vis, ih] from vis_dim (e.g. 1120) → visual_proj_dim (e.g. 128)
        # so action+prop have proportional weight at the trunk concat.
        vis_proj_layers = [torch.nn.Linear(self.vis_dim, self.visual_proj_dim)]
        if cfg.use_layer_norm:
            vis_proj_layers.append(torch.nn.LayerNorm(self.visual_proj_dim))
        vis_proj_layers.append(torch.nn.ReLU())
        self.visual_proj = torch.nn.Sequential(*vis_proj_layers)

        # ---- Trunk: projected_visual + prop + action → bottleneck ----
        trunk_in_dim = self.visual_proj_dim + prop_dim + action_dim
        layers = [torch.nn.Linear(trunk_in_dim, proj_dim)]
        if cfg.use_layer_norm:
            layers.append(torch.nn.GroupNorm(proj_dim // N, proj_dim))
        layers.append(torch.nn.ReLU())
        self.input_proj = torch.nn.Sequential(*layers)

        # ---- Heads see bottleneck + re-concatenated prop + action (skip) ----
        head_in_dim = proj_dim + prop_dim + action_dim
        self.q_ensemble = QEnsemble(
            in_dim=head_in_dim,
            hidden_dim=hidden_dim_full,
            output_dim=output_dim,
            num_layers=num_layers,
            num_heads=cfg.num_q,
            use_layer_norm=cfg.use_layer_norm,
            dropout=getattr(cfg, "dropout", 0.0),
            orth=getattr(cfg, "orth", True),
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