# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPx-License-Identifier: CC-BY-NC-4.0

import torch

from escnn import gspaces
from escnn import nn

from resfit.rl_finetuning.config.rlpd import ActorConfig
from resfit.rl_finetuning.off_policy.common_utils import utils
from resfit.rl_finetuning.equi_off_policy.common_utils.utils import (
    apply_deltaortho_init,
    scale_final_equi_layer,
)

from einops import rearrange

import time


class Actor(torch.nn.Module):
    def __init__(self, 
                 group,
                 in_type, 
                 hidden_dim,
                 action_shape,
                 num_layers,
                 initialize,
                 cfg: ActorConfig, 
                 residual_actor: bool = False):
        super().__init__()

        self.residual_actor = residual_actor
        self.cfg = cfg

        # ----------------------------------------------------------------
        # When using deltaortho init, skip the (slow) default
        # generalized_he_init by passing initialize=False to escnn.nn.Linear,
        # then apply deltaorthonormal_init manually afterward.
        # ----------------------------------------------------------------
        use_deltaortho = cfg.orth
        escnn_init = not use_deltaortho and initialize

        # Create policy network
        pol_in_type = in_type
        pol_hidden_type = nn.FieldType(group, hidden_dim * [group.regular_repr])
        pol_out_type = nn.FieldType(group, action_shape)

        layers = [nn.Linear(pol_in_type, pol_hidden_type, initialize=escnn_init),
                  nn.ReLU(pol_hidden_type, inplace=True)]
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(pol_hidden_type, pol_hidden_type, initialize=escnn_init))
            layers.append(nn.ReLU(pol_hidden_type, inplace=True))
        
        layers.append(nn.Linear(pol_hidden_type, pol_out_type, initialize=escnn_init))
        self.policy = torch.nn.Sequential(*layers)

        # ---- Initialization (mirrors non-equivariant Actor._initialize_weights) ----
        if use_deltaortho:
            # Deltaortho on all policy layers (intermediate + final)
            # Mirrors: orth_weight_init on compress + apply_initialization_to_network on policy
            apply_deltaortho_init(self.policy, exclude_final_layer=False)

            # Then scale the final layer down for near-zero initial residuals
            # Mirrors: utils.initialize_layer_weights(final_layer, distribution, scale)
            if cfg.actor_last_layer_init_scale is not None:
                scale_final_equi_layer(self.policy, cfg.actor_last_layer_init_scale)

    def forward(self, feat, std):

        assert isinstance(feat, nn.GeometricTensor), "Passed in non-Geometric Tensor to Equivariant Actor"

        mu: torch.Tensor = self.policy(feat).tensor

        # Scale the mean by action_scale
        # NOTE: std is already in environment action space (more interpretable)
        scaled_mu = mu * self.cfg.action_scale

        # Create distribution with scaled mean but environment-scale std
        action_dist = utils.TruncatedNormal(scaled_mu, std)

        return action_dist  # noqa: RET504