# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPx-License-Identifier: CC-BY-NC-4.0

import torch
# from torch import nn

from escnn import gspaces
from escnn import nn

from resfit.rl_finetuning.config.rlpd import ActorConfig
from resfit.rl_finetuning.off_policy.common_utils import utils

from einops import rearrange


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

        # Create policy network
        pol_in_type = in_type
        pol_hidden_type = nn.FieldType(group, hidden_dim * [group.regular_repr])
        pol_out_type = nn.FieldType(group, action_shape)

        layers = [nn.Linear(pol_in_type, pol_hidden_type, initialize=initialize),
                  nn.ReLU(pol_hidden_type, inplace=True)]
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(pol_hidden_type, pol_hidden_type, initialize=initialize))
            layers.append(nn.ReLU(pol_hidden_type, inplace=True))
        
        layers.append(nn.Linear(pol_hidden_type, pol_out_type, initialize=initialize))
        self.policy = torch.nn.Sequential(*layers)

    def forward(self, feat, std):
        assert isinstance(feat, nn.GeometricTensor), "Passed in non-Geometric Tensor to Equivariant Actor"

        mu: torch.Tensor = self.policy(feat).tensor

        # Scale the mean by action_scale
        # NOTE: std is already in environment action space (more interpretable)
        scaled_mu = mu * self.cfg.action_scale

        # Create distribution with scaled mean but environment-scale std
        action_dist = utils.TruncatedNormal(scaled_mu, std)

        return action_dist  # noqa: RET504
