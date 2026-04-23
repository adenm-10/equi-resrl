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
# Loss classes (unchanged — pure PyTorch, no equivariance)
# ---------------------------------------------------------------------------

class HLGaussLoss(torch.nn.Module):
    def __init__(self, min_value: float, max_value: float, num_bins: int, sigma: float | None = None):
        super().__init__()
        if sigma is None:
            sigma = 0.75 * (max_value - min_value) / num_bins

        self.sigma = sigma
        bin_edges = torch.linspace(min_value, max_value, num_bins + 1, dtype=torch.float32)
        self.register_buffer("bin_edges", bin_edges)

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        self.register_buffer("bin_centers", bin_centers)

    def _log1mexp(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.abs(x)
        return torch.where(x < math.log(2), torch.log(-torch.expm1(-x)), torch.log1p(-torch.exp(-x)))

    def _log_sub_exp(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        larger = torch.maximum(x, y)
        smaller = torch.minimum(x, y)
        return larger + self._log1mexp(torch.maximum(larger - smaller, torch.zeros_like(larger)))

    def _log_ndtr(self, x: torch.Tensor) -> torch.Tensor:
        ndtr_vals = torch.zeros_like(x)

        large_pos = x > 6
        if torch.any(large_pos):
            ndtr_vals[large_pos] = torch.log(torch.special.ndtr(x[large_pos]))

        large_neg = x < -6
        if torch.any(large_neg):
            x_neg = x[large_neg]
            ndtr_vals[large_neg] = -x_neg * x_neg / 2 - math.log(math.sqrt(2 * math.pi)) - torch.log(-x_neg)

        moderate = (~large_pos) & (~large_neg)
        if torch.any(moderate):
            x_mod = x[moderate]
            erfc_val = torch.special.erfc(-x_mod / math.sqrt(2))
            ndtr_vals[moderate] = torch.log(0.5 * erfc_val)

        return ndtr_vals

    def _normal_cdf_log_difference(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        is_y_positive = y >= 0
        x_hat = torch.where(is_y_positive, -y, x)
        y_hat = torch.where(is_y_positive, -x, y)

        log_ndtr_x = self._log_ndtr(x_hat)
        log_ndtr_y = self._log_ndtr(y_hat)

        return self._log_sub_exp(log_ndtr_x, log_ndtr_y)

    def _target_to_probs(self, target):
        target = target.unsqueeze(-1)

        norm_factor = math.sqrt(2) * self.sigma
        upper_edges = (self.bin_edges[1:].unsqueeze(0) - target) / norm_factor
        lower_edges = (self.bin_edges[:-1].unsqueeze(0) - target) / norm_factor

        bin_log_probs = self._normal_cdf_log_difference(upper_edges, lower_edges)

        log_z = self._normal_cdf_log_difference(
            (self.bin_edges[-1] - target) / norm_factor, (self.bin_edges[0] - target) / norm_factor
        )

        probs = torch.exp(bin_log_probs - log_z.unsqueeze(-1))

        valid_mask = torch.isfinite(probs).all(dim=-1, keepdim=True)
        uniform_probs = torch.ones_like(probs) / probs.shape[-1]
        return torch.where(valid_mask, probs, uniform_probs)

    def forward(self, logits, target):
        tgt_probs = self._target_to_probs(target.detach())
        log_probs = functional.log_softmax(logits, dim=-1)
        return -(tgt_probs * log_probs).sum(dim=-1).mean()

    def forward_batched(self, logits_batch, target):
        num_heads = logits_batch.shape[0]
        batch_size = target.shape[0]

        tgt_probs_buggy = self._target_to_probs(target.detach())

        log_probs = functional.log_softmax(logits_batch, dim=-1)

        tgt_probs_expanded = tgt_probs_buggy.unsqueeze(0).expand(num_heads, -1, -1, -1)
        log_probs_expanded = log_probs.unsqueeze(2).expand(-1, -1, batch_size, -1)

        product = tgt_probs_expanded * log_probs_expanded
        summed = product.sum(dim=-1)

        losses_per_head = -summed.view(num_heads, -1).mean(dim=-1)
        return losses_per_head.mean()


class C51Loss(torch.nn.Module):
    def __init__(self, v_min: float, v_max: float, num_atoms: int):
        super().__init__()
        self.v_min = v_min
        self.v_max = v_max
        self.num_atoms = num_atoms

        support = torch.linspace(v_min, v_max, num_atoms, dtype=torch.float32)
        self.register_buffer("support", support)

        self.delta_z = (v_max - v_min) / (num_atoms - 1)

    def project_distribution(
        self, next_distribution: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor, gamma: float
    ) -> torch.Tensor:
        batch_size = rewards.size(0)

        target_support = rewards.unsqueeze(1) + gamma * (1 - dones.unsqueeze(1)) * self.support.unsqueeze(0)
        target_support = torch.clamp(target_support, self.v_min, self.v_max)

        b = (target_support - self.v_min) / self.delta_z
        lower = b.floor().long()
        upper = b.ceil().long()

        lower[(upper > 0) * (lower == upper)] -= 1
        upper[(lower < (self.num_atoms - 1)) * (lower == upper)] += 1

        target_distribution = torch.zeros_like(next_distribution)
        offset = (
            torch.linspace(
                0, (batch_size - 1) * self.num_atoms, batch_size, dtype=torch.long, device=target_distribution.device
            )
            .unsqueeze(1)
            .expand(batch_size, self.num_atoms)
        )

        target_distribution.view(-1).index_add_(
            0, (lower + offset).view(-1), (next_distribution * (upper.float() - b)).view(-1)
        )
        target_distribution.view(-1).index_add_(
            0, (upper + offset).view(-1), (next_distribution * (b - lower.float())).view(-1)
        )

        return target_distribution

    def forward(self, current_logits: torch.Tensor, target_distribution: torch.Tensor) -> torch.Tensor:
        current_log_probs = functional.log_softmax(current_logits, dim=-1)
        return -(target_distribution * current_log_probs).sum(dim=-1).mean()

    def logits_to_q_value(self, logits: torch.Tensor) -> torch.Tensor:
        probs = functional.softmax(logits, dim=-1)
        return (probs * self.support).sum(dim=-1, keepdim=True)

    def forward_batched(self, current_logits_batch: torch.Tensor, target_distribution: torch.Tensor) -> torch.Tensor:
        current_log_probs = functional.log_softmax(current_logits_batch, dim=-1)

        num_heads = current_logits_batch.shape[0]
        target_expanded = target_distribution.unsqueeze(0).expand(num_heads, -1, -1)

        losses = -(target_expanded * current_log_probs).sum(dim=-1)
        return losses.mean()


# ---------------------------------------------------------------------------
# Non-equivariant MLP head
#
# Uniform hidden dim throughout (no GroupPooling bottleneck):
#   [Linear(in, hid) → [LayerNorm] → ReLU] × L → Linear(hid, out)
#
# hid = hidden_dim * N  (matches actual channel count of equivariant regular_repr)
#
# Uses ReLU without inplace=True for vmap compatibility.
# ---------------------------------------------------------------------------

class HeadMLP(torch.nn.Module):
    """Single MLP head for the ensemble."""

    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int = 2,
                 use_layer_norm: bool = True):
        super().__init__()

        layers = []
        current_dim = in_dim

        for _ in range(num_layers):
            layers.append(torch.nn.Linear(current_dim, hidden_dim))
            if use_layer_norm:
                layers.append(torch.nn.LayerNorm(hidden_dim))
            layers.append(torch.nn.ReLU())  # no inplace — required for vmap
            current_dim = hidden_dim

        layers.append(torch.nn.Linear(current_dim, output_dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)


class QEnsemble(torch.nn.Module):
    """vmap-parallelised ensemble of HeadMLPs.

    All heads are created independently (full diversity), then their
    parameters are stacked into tensors with a leading ``num_heads``
    dimension.  ``forward`` uses ``torch.vmap`` + ``functional_call``
    to evaluate every head in a single batched matmul — much faster
    than a Python loop for large ensembles (REDQ with 10+ heads).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        num_heads: int = 2,
        use_layer_norm: bool = True,
        orth: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads

        # Build independent heads
        heads = [
            HeadMLP(in_dim, hidden_dim, output_dim, num_layers, use_layer_norm)
            for _ in range(num_heads)
        ]

        # Optional orthogonal init
        if orth:
            for head in heads:
                head.net.apply(utils.orth_weight_init)

        # Stack parameters/buffers into tensors with leading H dimension
        self.params, self.buffers = stack_module_state(heads)

        # Template (structure only, no grad) for functional_call
        self._head_template = HeadMLP(in_dim, hidden_dim, output_dim, num_layers, use_layer_norm)

        # Register so they follow .to(device), .cuda(), state_dict(), etc.
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
        # Reconstruct dicts from registered tensors (proper device)
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

        # [num_heads, B, output_dim]
        return vmap(f_one_head, in_dims=(0, 0, None))(
            current_params, current_buffers, feat
        )


# ---------------------------------------------------------------------------
# Non-equivariant Critic (drop-in replacement for equivariant Critic)
# ---------------------------------------------------------------------------

class Critic(torch.nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 action_dim: int,
                 num_layers: int,
                 cfg: CriticConfig,
                 N: int = 8):
        super().__init__()
        self.cfg = cfg
        self.loss_cfg = cfg.loss

        output_dim = self.loss_cfg.n_bins if self.loss_cfg.type in {"hl_gauss", "c51"} else 1

        hidden_dim_full = hidden_dim * N      # matches regular_repr channel count

        self.q_ensemble = QEnsemble(
            in_dim=in_dim,
            hidden_dim=hidden_dim_full,
            output_dim=output_dim,
            num_layers=num_layers,
            num_heads=cfg.num_q,
            use_layer_norm=cfg.use_layer_norm,
            orth=getattr(cfg, "orth", True),
        )

        # Loss objects ----
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

    @staticmethod
    def _logits_to_q(probs, support):
        return (probs * support).sum(-1, keepdim=True)

    def forward(self, feat, act, *, return_logits: bool = False):
        # feat is a plain tensor (no .tensor unwrap needed)
        feat = torch.cat((feat, act), dim=-1)

        # logits_per_head: [num_q, B, out_dim]
        logits_per_head = self.q_ensemble(feat)

        if self.loss_cfg.type == "hl_gauss":
            q_per_head = self._logits_to_q(
                torch.softmax(logits_per_head, dim=-1),
                self.hl_loss.bin_centers,
            )
            if return_logits:
                return q_per_head, logits_per_head
            return q_per_head

        if self.loss_cfg.type == "c51":
            q_per_head = self.c51_loss.logits_to_q_value(logits_per_head)
            if return_logits:
                return q_per_head, logits_per_head
            return q_per_head

        # MSE case
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