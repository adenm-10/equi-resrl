# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import torch

from resfit.rl_finetuning.config.rlpd import ActorConfig
from resfit.rl_finetuning.off_policy.common_utils import utils


class Actor(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, action_dim, prop_dim,
                 num_layers, cfg: ActorConfig, N: int = 8,
                 residual_actor: bool = True):
        super().__init__()
        assert residual_actor, "Non-residual mode dropped — actor is residual-only"
        self.residual_actor = True
        self.cfg = cfg
        self.action_dim = action_dim
        self.prop_dim = prop_dim

        # in_dim = vis + ih + prop + base_action
        self.vis_ih_dim = in_dim - prop_dim - action_dim

        hid_full = proj_dim = hidden_dim * N
        use_layer_norm = getattr(cfg, "use_layer_norm", False)
        dropout = getattr(cfg, "dropout", 0.0)

        # ---- Visual projection ----
        self.visual_proj_dim = getattr(cfg, "visual_proj_dim", 128)
        vp_layers = [torch.nn.Linear(self.vis_ih_dim, self.visual_proj_dim)]
        if use_layer_norm:
            vp_layers.append(torch.nn.LayerNorm(self.visual_proj_dim))
        vp_layers.append(torch.nn.ReLU())
        self.visual_proj = torch.nn.Sequential(*vp_layers)

        # ---- Trunk: bottleneck (v, prop, base_action) ----
        trunk_in_dim = self.visual_proj_dim + prop_dim + action_dim
        proj_layers = [torch.nn.Linear(trunk_in_dim, proj_dim)]
        if use_layer_norm:
            proj_layers.append(torch.nn.LayerNorm(proj_dim))
        proj_layers.append(torch.nn.ReLU())
        self.input_proj = torch.nn.Sequential(*proj_layers)

        # ---- Policy MLP with skip injection ----
        skip_dim = prop_dim + action_dim   # always-residual
        policy_in_dim = proj_dim + skip_dim

        layers = []
        current_dim = policy_in_dim
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

        self._initialize_weights(cfg)

    def _initialize_weights(self, cfg: ActorConfig):
        if getattr(cfg, "orth", False):
            self.visual_proj.apply(utils.orth_weight_init)
            self.input_proj.apply(utils.orth_weight_init)
            self.policy.apply(utils.orth_weight_init)

        last_layer_scale = getattr(cfg, "actor_last_layer_init_scale", None)
        if last_layer_scale is not None:
            for module in reversed(list(self.policy.modules())):
                if isinstance(module, torch.nn.Linear):
                    module.weight.data.mul_(last_layer_scale)
                    if module.bias is not None:
                        module.bias.data.mul_(last_layer_scale)
                    break

    def forward(self, feat, std):
        # feat layout: [vis_ih | prop | base_action]
        vis_ih      = feat[:, :self.vis_ih_dim]
        prop        = feat[:, self.vis_ih_dim : self.vis_ih_dim + self.prop_dim]
        base_action = feat[:, self.vis_ih_dim + self.prop_dim :]

        v         = self.visual_proj(vis_ih)
        trunk_in  = torch.cat([v, prop, base_action], dim=-1)
        z         = self.input_proj(trunk_in)
        policy_in = torch.cat([z, prop, base_action], dim=-1)

        mu = self.policy(policy_in)
        scaled_mu = mu * self.cfg.action_scale
        return utils.TruncatedNormal(scaled_mu, std)