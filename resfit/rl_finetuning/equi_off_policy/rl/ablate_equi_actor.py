# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import torch

from resfit.rl_finetuning.config.rlpd import ActorConfig
from resfit.rl_finetuning.off_policy.common_utils import utils


class Actor(torch.nn.Module):
    def __init__(self, 
                 in_dim, 
                 hidden_dim, 
                 action_dim, 
                 prop_dim,
                 N,
                 dropout,
                 use_norms,
                 use_orth_init,
                 num_layers,
                 action_scale,
                 last_layer_scale,
                 residual_actor: bool = True):
        super().__init__()
        assert residual_actor, "Non-residual mode dropped — actor is residual-only"
        self.residual_actor = True
        self.action_dim = action_dim
        self.prop_dim = prop_dim

        # in_dim = vis + ih + prop + base_action
        self.vis_ih_dim = in_dim - prop_dim - action_dim

        hidden_dim_full = proj_dim = hidden_dim * N
        self.dropout = dropout
        self.use_norms = use_norms
        self.action_scale = action_scale
        self.use_orth_init = use_orth_init
        self.last_layer_scale = last_layer_scale

        # ---- Visual Projection: agentview + inhand -> visual features ----
        vp_layers = [torch.nn.Linear(self.vis_ih_dim, hidden_dim_full)]
        if use_norms:
            vp_layers.append(torch.nn.LayerNorm(hidden_dim_full))
        vp_layers.append(torch.nn.ReLU())
        self.visual_proj = torch.nn.Sequential(*vp_layers)

        # ---- Trunk: visual features + prop + action -> all features ----
        trunk_in_dim = hidden_dim_full + prop_dim + action_dim
        proj_layers = [torch.nn.Linear(trunk_in_dim, hidden_dim_full)]
        if use_norms:
            proj_layers.append(torch.nn.LayerNorm(hidden_dim_full))
        proj_layers.append(torch.nn.ReLU())
        self.input_proj = torch.nn.Sequential(*proj_layers)

        # ---- Head MLP with skip injection ----
        policy_in_dim = hidden_dim_full + prop_dim + action_dim

        layers = []
        current_dim = policy_in_dim
        for _ in range(num_layers):
            layers.append(torch.nn.Linear(current_dim, hidden_dim_full))
            if use_norms:
                layers.append(torch.nn.LayerNorm(hidden_dim_full))
            if dropout > 0:
                layers.append(torch.nn.Dropout(dropout))
            layers.append(torch.nn.ReLU(inplace=True))
            current_dim = hidden_dim_full
        layers.append(torch.nn.Linear(current_dim, action_dim))
        layers.append(torch.nn.Tanh())
        self.policy = torch.nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        if self.use_orth_init:
            self.visual_proj.apply(utils.orth_weight_init)
            self.input_proj.apply(utils.orth_weight_init)
            self.policy.apply(utils.orth_weight_init)

        if self.last_layer_scale is not None:
            for module in reversed(list(self.policy.modules())):
                if isinstance(module, torch.nn.Linear):
                    module.weight.data.mul_(self.last_layer_scale)
                    if module.bias is not None:
                        module.bias.data.mul_(self.last_layer_scale)
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
        scaled_mu = mu * self.action_scale
        return utils.TruncatedNormal(scaled_mu, std)