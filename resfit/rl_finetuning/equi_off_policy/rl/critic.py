# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

import math

import torch
# from torch import nn

from escnn import gspaces
from escnn import nn

# Import vmap functionality (PyTorch 2.1+)
from torch.func import functional_call, stack_module_state, vmap
from torch.nn import functional

from resfit.rl_finetuning.config.rlpd import CriticConfig
from resfit.rl_finetuning.off_policy.common_utils import utils


class HLGaussLoss(torch.nn.Module):
    def __init__(self, min_value: float, max_value: float, num_bins: int, sigma: float | None = None):
        super().__init__()
        if sigma is None:
            # Use the recommended sigma from the paper
            sigma = 0.75 * (max_value - min_value) / num_bins

        self.sigma = sigma
        # Create bin edges (num_bins + 1 points)
        bin_edges = torch.linspace(min_value, max_value, num_bins + 1, dtype=torch.float32)
        self.register_buffer("bin_edges", bin_edges)

        # Bin centers for expectation computation
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        self.register_buffer("bin_centers", bin_centers)

    def _log1mexp(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log(1 - exp(-|x|)) in numerically stable way."""
        x = torch.abs(x)
        return torch.where(x < math.log(2), torch.log(-torch.expm1(-x)), torch.log1p(-torch.exp(-x)))

    def _log_sub_exp(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute log(exp(max(x,y)) - exp(min(x,y))) in numerically stable way."""
        larger = torch.maximum(x, y)
        smaller = torch.minimum(x, y)
        return larger + self._log1mexp(torch.maximum(larger - smaller, torch.zeros_like(larger)))

    def _log_ndtr(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log(Φ(x)) where Φ is standard normal CDF, numerically stable."""
        # For x > 6, use asymptotic expansion
        # For x < -6, use log(Φ(x)) ≈ -x²/2 - log(√(2π)) - log(-x)
        # For -6 ≤ x ≤ 6, use log(0.5 * (1 + erf(x/√2)))

        ndtr_vals = torch.zeros_like(x)

        # For large positive x (> 6): log(Φ(x)) ≈ log(1 - Φ(-x)) ≈ log(1 - e^(-x²/2 - log(√(2π)) + log(-x)))
        large_pos = x > 6
        if torch.any(large_pos):
            ndtr_vals[large_pos] = torch.log(torch.special.ndtr(x[large_pos]))

        # For large negative x (< -6): log(Φ(x)) ≈ -x²/2 - log(√(2π)) - log(-x)
        large_neg = x < -6
        if torch.any(large_neg):
            x_neg = x[large_neg]
            ndtr_vals[large_neg] = -x_neg * x_neg / 2 - math.log(math.sqrt(2 * math.pi)) - torch.log(-x_neg)

        # For moderate x (-6 ≤ x ≤ 6): use direct computation
        moderate = (~large_pos) & (~large_neg)
        if torch.any(moderate):
            x_mod = x[moderate]
            # Use erfc for better precision when x is negative
            erfc_val = torch.special.erfc(-x_mod / math.sqrt(2))
            ndtr_vals[moderate] = torch.log(0.5 * erfc_val)

        return ndtr_vals

    def _normal_cdf_log_difference(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute log(Φ(x) - Φ(y)) where Φ is standard normal CDF."""
        # When x >= y >= 0, compute log(Φ(-y) - Φ(-x)) for better precision
        is_y_positive = y >= 0
        x_hat = torch.where(is_y_positive, -y, x)
        y_hat = torch.where(is_y_positive, -x, y)

        log_ndtr_x = self._log_ndtr(x_hat)
        log_ndtr_y = self._log_ndtr(y_hat)

        return self._log_sub_exp(log_ndtr_x, log_ndtr_y)

    def _target_to_probs(self, target):
        target = target.unsqueeze(-1)  # [batch] -> [batch, 1]

        # Compute normalized distances to bin edges
        norm_factor = math.sqrt(2) * self.sigma
        upper_edges = (self.bin_edges[1:].unsqueeze(0) - target) / norm_factor
        lower_edges = (self.bin_edges[:-1].unsqueeze(0) - target) / norm_factor

        # Compute log probabilities for each bin
        bin_log_probs = self._normal_cdf_log_difference(upper_edges, lower_edges)

        # Compute normalization factor (log of total probability mass)
        log_z = self._normal_cdf_log_difference(
            (self.bin_edges[-1] - target) / norm_factor, (self.bin_edges[0] - target) / norm_factor
        )

        # Convert to probabilities
        probs = torch.exp(bin_log_probs - log_z.unsqueeze(-1))

        # Safety check: if anything goes wrong, fall back to uniform
        valid_mask = torch.isfinite(probs).all(dim=-1, keepdim=True)
        uniform_probs = torch.ones_like(probs) / probs.shape[-1]
        return torch.where(valid_mask, probs, uniform_probs)

    def forward(self, logits, target):
        tgt_probs = self._target_to_probs(target.detach())
        # Manual cross-entropy for soft labels: -sum(p * log_softmax(logits))
        log_probs = functional.log_softmax(logits, dim=-1)
        return -(tgt_probs * log_probs).sum(dim=-1).mean()

    def forward_batched(self, logits_batch, target):
        """
        Compute HLGaussLoss for multiple critic heads in a batched manner.
        This exactly replicates the original (buggy) behavior but in a vectorized way.

        Args:
            logits_batch: [num_heads, batch_size, num_bins] - logits from all critic heads
            target: [batch_size] - target values

        Returns:
            loss: scalar - average loss across all heads and batch
        """
        num_heads = logits_batch.shape[0]
        batch_size = target.shape[0]

        # Get the buggy target probabilities (same for all heads)
        tgt_probs_buggy = self._target_to_probs(target.detach())  # [batch_size, batch_size, num_bins]

        # Compute log softmax for all heads
        log_probs = functional.log_softmax(logits_batch, dim=-1)  # [num_heads, batch_size, num_bins]

        # Expand tgt_probs_buggy to all heads: [B, B, bins] -> [heads, B, B, bins]
        tgt_probs_expanded = tgt_probs_buggy.unsqueeze(0).expand(num_heads, -1, -1, -1)

        # Expand log_probs to match: [num_heads, batch_size, num_bins] -> [num_heads, batch_size, batch_size, num_bins]
        log_probs_expanded = log_probs.unsqueeze(2).expand(-1, -1, batch_size, -1)

        # Multiply: [num_heads, batch_size, batch_size, num_bins]
        product = tgt_probs_expanded * log_probs_expanded

        # Sum over bins: [num_heads, batch_size, batch_size]
        summed = product.sum(dim=-1)

        # Mean over all dimensions except heads, then mean over heads
        # This matches the original: -summed_h.mean() for each head, then average across heads
        losses_per_head = -summed.view(num_heads, -1).mean(dim=-1)  # [num_heads]

        return losses_per_head.mean()


class C51Loss(torch.nn.Module):
    def __init__(self, v_min: float, v_max: float, num_atoms: int):
        super().__init__()
        self.v_min = v_min
        self.v_max = v_max
        self.num_atoms = num_atoms

        # Create support for value distribution
        support = torch.linspace(v_min, v_max, num_atoms, dtype=torch.float32)
        self.register_buffer("support", support)

        # Delta z for projection
        self.delta_z = (v_max - v_min) / (num_atoms - 1)

    def project_distribution(
        self, next_distribution: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor, gamma: float
    ) -> torch.Tensor:
        """Project Bellman update onto categorical support."""
        batch_size = rewards.size(0)

        # Compute target values for each atom: r + gamma * (1 - done) * support
        target_support = rewards.unsqueeze(1) + gamma * (1 - dones.unsqueeze(1)) * self.support.unsqueeze(0)

        # Clamp target support to valid range
        target_support = torch.clamp(target_support, self.v_min, self.v_max)

        # Compute indices and interpolation weights for projection
        b = (target_support - self.v_min) / self.delta_z
        lower = b.floor().long()  # Lower bound indices
        upper = b.ceil().long()  # Upper bound indices

        # Handle edge cases
        lower[(upper > 0) * (lower == upper)] -= 1
        upper[(lower < (self.num_atoms - 1)) * (lower == upper)] += 1

        # Project probabilities onto support
        target_distribution = torch.zeros_like(next_distribution)
        offset = (
            torch.linspace(
                0, (batch_size - 1) * self.num_atoms, batch_size, dtype=torch.long, device=target_distribution.device
            )
            .unsqueeze(1)
            .expand(batch_size, self.num_atoms)
        )

        # Lower bound projection
        target_distribution.view(-1).index_add_(
            0, (lower + offset).view(-1), (next_distribution * (upper.float() - b)).view(-1)
        )
        # Upper bound projection
        target_distribution.view(-1).index_add_(
            0, (upper + offset).view(-1), (next_distribution * (b - lower.float())).view(-1)
        )

        return target_distribution

    def forward(self, current_logits: torch.Tensor, target_distribution: torch.Tensor) -> torch.Tensor:
        """Compute C51 loss using cross-entropy between current and target distributions."""
        # Convert logits to log probabilities
        current_log_probs = functional.log_softmax(current_logits, dim=-1)

        # Compute cross-entropy loss
        return -(target_distribution * current_log_probs).sum(dim=-1).mean()

    def logits_to_q_value(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert categorical logits to Q-value using expectation over support."""
        probs = functional.softmax(logits, dim=-1)
        return (probs * self.support).sum(dim=-1, keepdim=True)

    def forward_batched(self, current_logits_batch: torch.Tensor, target_distribution: torch.Tensor) -> torch.Tensor:
        """
        Compute C51 loss for multiple critic heads in a batched manner.

        Args:
            current_logits_batch: [num_heads, batch_size, num_atoms] - logits from all critic heads
            target_distribution: [batch_size, num_atoms] - target distribution (same for all heads)

        Returns:
            loss: scalar - average loss across all heads and batch
        """
        # Convert logits to log probabilities for all heads
        current_log_probs = functional.log_softmax(current_logits_batch, dim=-1)

        # Expand target distribution to match number of heads
        num_heads = current_logits_batch.shape[0]
        target_expanded = target_distribution.unsqueeze(0).expand(num_heads, -1, -1)

        # Compute cross-entropy loss for all heads at once
        losses = -(target_expanded * current_log_probs).sum(dim=-1)  # [num_heads, batch_size]
        return losses.mean()  # Average over all heads and batch


class Critic(torch.nn.Module):
    def __init__(self, 
                 group,
                 in_type, 
                 hidden_dim,
                 action_dim, 
                 num_layers,
                 initialize,
                 cfg: CriticConfig):
        super().__init__()
        self.cfg = cfg
        self.group = group
        self.loss_cfg = cfg.loss
        self.in_type = in_type

        output_dim = self.loss_cfg.n_bins if self.loss_cfg.type in {"hl_gauss", "c51"} else 1

        # Number of Q-heads (ensemble size)
        num_q = getattr(cfg, "num_q", 2)

        head_hidden_type = nn.FieldType(self.group, hidden_dim * [self.group.regular_repr])
        head_out_type = nn.FieldType(self.group, output_dim * [self.group.trivial_repr])

        self.q_ensemble = EquiQEnsemble(group          = self.group, 
                                        in_type        = in_type,
                                        hidden_type    = head_hidden_type,
                                        out_type       = head_out_type,
                                        initialize     = initialize,
                                        num_layers     = num_layers,
                                        num_heads      = cfg.num_q,
                                        use_layer_norm = cfg.use_layer_norm,
                                    )

        # Loss objects (single instance) ----
        if self.loss_cfg.type == "hl_gauss":
            # Use None if sigma is -1.0 (auto-compute), otherwise use the specified value
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
        # (B,K) * (K,) → (B,)
        return (probs * support).sum(-1, keepdim=True)  # E[z]

    def forward(self, feat, act, *, return_logits: bool = False):
        feat = torch.cat((feat.tensor, act), dim=-1)
        feat = nn.GeometricTensor(feat, self.in_type)

        # logits_per_head: [num_q, B, out_dim] where out_dim is either 1 or K (bins)
        logits_per_head = self.q_ensemble(feat)

        if self.loss_cfg.type == "hl_gauss":
            # expectation over bin centers → scalar Q, done per head
            q_per_head = self._logits_to_q(
                torch.softmax(logits_per_head, dim=-1),
                self.hl_loss.bin_centers,
            )  # [num_q, B, 1]

            if return_logits:
                return q_per_head, logits_per_head
            return q_per_head

        if self.loss_cfg.type == "c51":
            # expectation over categorical support → scalar Q, done per head
            q_per_head = self.c51_loss.logits_to_q_value(logits_per_head)  # [num_q, B, 1]

            if return_logits:
                return q_per_head, logits_per_head
            return q_per_head

        # MSE case: outputs are already scalars per head → [num_q, B, 1]
        return logits_per_head

    def q_value(self, feat, act):
        """
        Returns the Q-value for a given feature, property, and action.
        I.e., gets all Q-values, subsets them, and returns the min.
        Used for critic loss computation (uses configurable min_q_heads).
        """
        # Get the Q-values for all heads
        q_out = self.forward(feat, act)

        # Take min over random subset of heads (configurable via min_q_heads)
        num_heads = min(self.cfg.min_q_heads, q_out.shape[0])
        idx = torch.randperm(q_out.shape[0], device=q_out.device)[:num_heads]
        return torch.min(q_out.index_select(0, idx), dim=0).values

    def q_value_for_policy(self, feat, act):
        """
        Returns the Q-value for policy gradient computation.
        Supports different policy gradient types:
        - "ensemble_mean": mean over all heads (standard RED-Q approach)
        - "min_random_pair": min of configurable number of random heads (conservative variant)
        - "q1": just use q1 from ensemble (standard TD3 approach)
        """
        # Get the Q-values for all heads
        q_out = self.forward(feat, act)

        if self.cfg.policy_gradient_type == "ensemble_mean":
            # Take mean over all heads (standard RED-Q approach)
            return q_out.mean(dim=0)
        if self.cfg.policy_gradient_type == "min_random_pair":
            # Take min over random subset of heads (configurable via min_q_heads)
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
    ):
        super().__init__()
        self.heads = torch.nn.ModuleList([EquiHeadMLP(group=group, 
                                                      in_type=in_type, 
                                                      hidden_type=hidden_type, 
                                                      out_type=out_type, 
                                                      initialize=initialize,
                                                      num_layers=num_layers,
                                                      use_layer_norm=use_layer_norm) for _ in range(num_heads)]
                                        )

    def forward(self, feat):
        q_vals = []
        for head in self.heads:
            q_vals.append(head(feat).tensor)

        return torch.stack(q_vals, dim=0)

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
        layers = [nn.Linear(in_type, hidden_type, initialize=initialize),
                  nn.ReLU(hidden_type, inplace=True)]
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_type, hidden_type, initialize=initialize))
            layers.append(nn.ReLU(hidden_type, inplace=True))
        
        layers.append(nn.GroupPooling(hidden_type))
        layers.append(nn.Linear(pooled, out_type, initialize=initialize))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, feat):
        return self.net(feat)
