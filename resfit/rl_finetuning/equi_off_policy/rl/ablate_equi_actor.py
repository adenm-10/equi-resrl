# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0

import torch

from resfit.rl_finetuning.config.rlpd import ActorConfig
from resfit.rl_finetuning.off_policy.common_utils import utils


class Actor(torch.nn.Module):
    """Non-equivariant actor (mirrors equivariant Actor).

    Architecture:
        [Linear → (LayerNorm) → (Dropout) → ReLU] × L → Linear → Tanh → out
        where hidden width = hidden_dim * N

    Optional features (all controlled via cfg with safe defaults):
        - LayerNorm:  cfg.use_layer_norm  (default False)
        - Dropout:    cfg.dropout          (default 0.0 = off)
        - Orth init:  cfg.orth             (default False)
        - Final layer scale: cfg.actor_last_layer_init_scale (default None = skip)
    """

    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 action_dim: int,
                 num_layers: int,
                 cfg: ActorConfig,
                 N: int = 8,
                 residual_actor: bool = False):
        super().__init__()

        self.residual_actor = residual_actor
        self.cfg = cfg

        hid_full = hidden_dim * N
        use_layer_norm = getattr(cfg, "use_layer_norm", False)
        dropout = getattr(cfg, "dropout", 0.0)

        # ---- Build policy MLP ----
        layers = []
        current_dim = in_dim

        for _ in range(num_layers):
            layers.append(torch.nn.Linear(current_dim, hid_full))
            if use_layer_norm:
                layers.append(torch.nn.LayerNorm(hid_full))
            if dropout > 0:
                layers.append(torch.nn.Dropout(dropout))
            layers.append(torch.nn.ReLU(inplace=True))
            current_dim = hid_full

        layers.append(torch.nn.Linear(current_dim, action_dim))
        layers.append(torch.nn.Tanh())
        self.policy = torch.nn.Sequential(*layers)

        # ---- Initialization ----
        self._initialize_weights(cfg)

        # for module in reversed(list(self.policy.modules())):
        #     if isinstance(module, torch.nn.Linear):
        #         print(f"Final layer weight norm: {module.weight.data.norm():.8f}")
        #         print(f"Final layer bias norm: {module.bias.data.norm():.8f}")
        #         break

        # assert False

    def _initialize_weights(self, cfg: ActorConfig):
        """Apply weight initialization.

        - cfg.orth (bool):  orthogonal init for all layers
        - cfg.actor_last_layer_init_scale (float | None):
              if set, rescale the final Linear layer's weights by this factor.
              Small values (e.g. 0.01) ensure near-zero initial actions,
              preventing early saturation in residual RL.
        """
        orth = getattr(cfg, "orth", False)

        if orth:
            self.policy.apply(utils.orth_weight_init)

        # Optionally scale down the final layer for small initial actions
        last_layer_scale = getattr(cfg, "actor_last_layer_init_scale", None)
        if last_layer_scale is not None:
            # Walk backwards to find the last Linear
            for module in reversed(list(self.policy.modules())):
                if isinstance(module, torch.nn.Linear):
                    module.weight.data.mul_(last_layer_scale)
                    if module.bias is not None:
                        module.bias.data.mul_(last_layer_scale)
                    break

    def forward(self, feat, std):
        mu = self.policy(feat)
        
        # if not hasattr(self, '_debug_printed'):
        # print(f"ACTOR FORWARD DEBUG: mu.abs().mean()={mu.abs().mean():.8f}, "
        #     f"std={std}, action_scale={self.cfg.action_scale}")
        #     # self._debug_printed = True
        # assert False

        scaled_mu = mu * self.cfg.action_scale
        action_dist = utils.TruncatedNormal(scaled_mu, std)
        return action_dist