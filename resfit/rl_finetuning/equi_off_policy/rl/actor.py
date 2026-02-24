# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPx-License-Identifier: CC-BY-NC-4.0

import torch
# from torch import nn

from escnn import gspaces
from escnn import nn

from resfit.rl_finetuning.config.rlpd import ActorConfig
from resfit.rl_finetuning.off_policy.common_utils import utils

from einops import rearrange


# def build_fc(in_dim, hidden_dim, action_dim, num_layer, layer_norm, dropout, use_layer_norm=True):
#     dims = [in_dim]
#     dims.extend([hidden_dim for _ in range(num_layer)])

#     layers = []
#     for i in range(len(dims) - 1):
#         layers.append(nn.Linear(dims[i], dims[i + 1]))
#         if use_layer_norm and layer_norm == 1:
#             layers.append(nn.LayerNorm(dims[i + 1]))
#         if use_layer_norm and layer_norm == 2 and (i == num_layer - 1):
#             layers.append(nn.LayerNorm(dims[i + 1]))
#         layers.append(nn.Dropout(dropout))
#         layers.append(nn.ReLU())

#     layers.append(nn.Linear(dims[-1], action_dim))
#     layers.append(nn.Tanh())
#     return nn.Sequential(*layers)


# class SpatialEmb(nn.Module):
#     def __init__(self, num_patch, patch_dim, prop_dim, proj_dim, dropout, use_layer_norm=True):
#         super().__init__()

#         # if fuse_patch:
#         proj_in_dim = num_patch + prop_dim
#         num_proj = patch_dim

#         self.patch_dim = patch_dim
#         self.prop_dim = prop_dim

#         layers = [nn.Linear(proj_in_dim, proj_dim)]
#         if use_layer_norm:
#             layers.append(nn.LayerNorm(proj_dim))
#         layers.append(nn.ReLU(inplace=True))

#         self.input_proj = nn.Sequential(*layers)
#         self.weight = nn.Parameter(torch.zeros(1, num_proj, proj_dim))
#         self.dropout = nn.Dropout(dropout)
#         nn.init.normal_(self.weight)

#     def extra_repr(self) -> str:
#         return f"weight: nn.Parameter ({self.weight.size()})"

#     def forward(self, feat: torch.Tensor, prop: torch.Tensor):
#         feat = feat.transpose(1, 2)

#         if self.prop_dim > 0:
#             repeated_prop = prop.unsqueeze(1).repeat(1, feat.size(1), 1)
#             feat = torch.cat((feat, repeated_prop), dim=-1)

#         y = self.input_proj(feat)
#         z = (self.weight * y).sum(1)
#         z = self.dropout(z)
#         return z  # noqa: RET504


class Actor(nn.Module):
    def __init__(self, 
                 group,
                 in_type, 
                 hidden_dim,
                 action_shape,
                 cfg: ActorConfig, 
                 residual_actor: bool = False):
        super().__init__()

        self.residual_actor = residual_actor
        self.group = group
        self.cfg = cfg

        # if residual_actor:
        #     # The residual actor takes the base action as input alongside the state
        #     self.prop_dim += action_dim

        # obs features + act + prop -> features
        self.compress = torch.nn.Sequential(
                                nn.Linear(
                                    in_type,
                                    nn.FieldType(group, hidden_dim * [self.group.regular_repr])),
                                nn.ReLU(nn.FieldType(group, hidden_dim * [self.group.regular_repr]))
                           )

        # Create policy network
        self.pol_in_type = nn.FieldType(group, hidden_dim * [self.group.regular_repr])
        self.pol_hidden_type = nn.FieldType(group, hidden_dim * [self.group.trivial_repr])
        self.pol_out_type = nn.FieldType(group, action_shape)
        self.policy = self.build_equi_policy(num_layer=cfg.num_layers)

        # Apply weight initialization
        # self._initialize_weights(cfg)

    # def _initialize_weights(self, cfg: ActorConfig):
    #     """Apply weight initialization to all networks."""
    #     # Determine initialization distributions
    #     intermediate_init = cfg.actor_intermediate_layer_init_distribution
    #     if cfg.orth and intermediate_init == "default":
    #         intermediate_init = "orthogonal"

    #     # Initialize compression layers
    #     if cfg.orth:
    #         # Backward compatibility: use existing orthogonal initialization
    #         self.compress.apply(utils.orth_weight_init)
    #     else:
    #         # Use the specified distribution for compression layers
    #         utils.apply_initialization_to_network(self.compress, intermediate_init)

    #     # Initialize policy network intermediate layers (exclude final layer)
    #     utils.apply_initialization_to_network(self.policy, intermediate_init, exclude_final_layer=True)

    #     # Initialize final layer with specific configuration if provided
    #     if cfg.actor_last_layer_init_scale is not None:
    #         final_layer = None
    #         for module in reversed(list(self.policy.modules())):
    #             if isinstance(module, nn.Linear):
    #                 final_layer = module
    #                 break

    #         if final_layer is not None:
    #             utils.initialize_layer_weights(
    #                 final_layer,
    #                 cfg.actor_last_layer_init_distribution,
    #                 cfg.actor_last_layer_init_scale,
    #             )

    # def forward(self, obs: dict[str, torch.Tensor], std: float):
    #     if isinstance(self.compress, SpatialEmb):
    #         assert not self.residual_actor, "Not implemented"
    #         feat = self.compress.forward(obs["feat"], obs["observation.state"])
    #     else:
    #         feat = obs["feat"].flatten(1, -1)
    #         feat = self.compress(feat)

    #     all_input = [feat]
    #     if self.prop_dim > 0:
    #         prop = obs["observation.state"]
    #         all_input.append(prop)
    #         if self.residual_actor:
    #             # The residual actor takes the base action as input alongside the state
    #             all_input.append(obs["observation.base_action"])

    #     policy_input = torch.cat(all_input, dim=-1)

    #     mu: torch.Tensor = self.policy(policy_input)

    #     # Scale the mean by action_scale
    #     # NOTE: std is alreay in environment action space (more interpretable)
    #     scaled_mu = mu * self.cfg.action_scale

    #     # Create distribution with scaled mean but environment-scale std
    #     action_dist = utils.TruncatedNormal(scaled_mu, std)

    #     return action_dist  # noqa: RET504

    def get6DRotation(self, quat):
        # data is in xyzw, but rotation transformer takes wxyz
        return self.quaternion_to_sixd.forward(quat[:, [3, 0, 1, 2]]) 

    def build_equi_policy(self, num_layers, dropout=False, use_layer_norm=False):

        # TODO:
        # - Include LayerNorm?
        # - Include Dropout?
        layers = [nn.Linear(self.pol_in_type, self.pol_hidden_type), 
                  nn.ReLU(self.pol_hidden_type)]
        for i in range(num_layer):
            layers.append(nn.Linear(self.pol_hidden_type, self.pol_hidden_type))
            layers.append(nn.ReLU(self.pol_hidden_type))

        layers.append(nn.Linear(self.pol_hidden_type, self.pol_out_type))
        # layers.append(torch.nn.Tanh())
        return nn.Sequential(*layers)
        
    def forward(self, feat, std):
        assert isinstance(feat, nn.GeometricTensor), "Passed in non-Geometric Tensor to Equivariant Actor"

        feat = self.compress(feat)
        mu: torch.Tensor = self.policy(feat).tensor

        # Scale the mean by action_scale
        # NOTE: std is already in environment action space (more interpretable)
        scaled_mu = mu * self.cfg.action_scale

        # Create distribution with scaled mean but environment-scale std
        action_dist = utils.TruncatedNormal(scaled_mu, std)

        return action_dist  # noqa: RET504
