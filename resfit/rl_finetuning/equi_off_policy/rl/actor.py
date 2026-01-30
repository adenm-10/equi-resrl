# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPx-License-Identifier: CC-BY-NC-4.0

import torch
# from torch import nn

from escnn import gspaces
from escnn import nn

from resfit.rl_finetuning.config.rlpd import ActorConfig
from resfit.rl_finetuning.off_policy.common_utils import utils

from einops import rearrange


def build_fc(in_dim, hidden_dim, action_dim, num_layer, layer_norm, dropout, use_layer_norm=True):
    dims = [in_dim]
    dims.extend([hidden_dim for _ in range(num_layer)])

    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if use_layer_norm and layer_norm == 1:
            layers.append(nn.LayerNorm(dims[i + 1]))
        if use_layer_norm and layer_norm == 2 and (i == num_layer - 1):
            layers.append(nn.LayerNorm(dims[i + 1]))
        layers.append(nn.Dropout(dropout))
        layers.append(nn.ReLU())

    layers.append(nn.Linear(dims[-1], action_dim))
    layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class SpatialEmb(nn.Module):
    def __init__(self, num_patch, patch_dim, prop_dim, proj_dim, dropout, use_layer_norm=True):
        super().__init__()

        # if fuse_patch:
        proj_in_dim = num_patch + prop_dim
        num_proj = patch_dim

        self.patch_dim = patch_dim
        self.prop_dim = prop_dim

        layers = [nn.Linear(proj_in_dim, proj_dim)]
        if use_layer_norm:
            layers.append(nn.LayerNorm(proj_dim))
        layers.append(nn.ReLU(inplace=True))

        self.input_proj = nn.Sequential(*layers)
        self.weight = nn.Parameter(torch.zeros(1, num_proj, proj_dim))
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.weight)

    def extra_repr(self) -> str:
        return f"weight: nn.Parameter ({self.weight.size()})"

    def forward(self, feat: torch.Tensor, prop: torch.Tensor):
        feat = feat.transpose(1, 2)

        if self.prop_dim > 0:
            repeated_prop = prop.unsqueeze(1).repeat(1, feat.size(1), 1)
            feat = torch.cat((feat, repeated_prop), dim=-1)

        y = self.input_proj(feat)
        z = (self.weight * y).sum(1)
        z = self.dropout(z)
        return z  # noqa: RET504


class Actor(nn.Module):
    def __init__(self, 
                 repr_dim, 
                 patch_repr_dim, 
                 cfg: ActorConfig, 
                 N=8,
                 obs_shape=(3, 84, 84),
                 crop_shape=(76, 76),
                 residual_actor: bool = False):
        super().__init__()

        self.residual_actor = residual_actor
        self.group = gspaces.rot2dOnR2(N=N)
        self.cfg = cfg
        obs_channel = obs_shape[0]

        if residual_actor:
            # The residual actor takes the base action as input alongside the state
            self.prop_dim += action_dim

        if cfg.spatial_emb > 0:
            pass
            # assert cfg.spatial_emb > 1, "this is the dimension"
            # self.compress = SpatialEmb(
            #     num_patch=repr_dim // patch_repr_dim,
            #     patch_dim=patch_repr_dim,
            #     prop_dim=self.prop_dim,
            #     proj_dim=cfg.spatial_emb,
            #     dropout=cfg.dropout,
            #     use_layer_norm=cfg.use_layer_norm,
            # )
            # policy_in_dim = cfg.spatial_emb
        else:
            pass
            # layers = [nn.Linear(repr_dim, cfg.feature_dim)]
            # if cfg.use_layer_norm:
            #     layers.append(nn.LayerNorm(cfg.feature_dim))
            # layers.extend([nn.Dropout(cfg.dropout), nn.ReLU()])
            # self.compress = nn.Sequential(*layers)

        self.action_shape = 2 * [self.group.irrep(1)]               # action_xy, action_rx_ry
                          + 3 * [self.group.trivial_repr],          # action_z, action_rz, action_gripper
        self.prop_shape   = 4 * [self.group.irrep(1)]               # pos_xy + 3 rotation columns
                          + 3 * [self.group.trivial_repr]           # pos_z, ee_q (2)

        self.feat_type = nn.FieldType(
                self.group,
                n_hidden * [self.group.regular_repr] # agentview img
                + prop_shape                         # proprioceptive state
                + self.action_shape                       # base action 
            )
        
        self.quaternion_to_sixd = RotationTransformer('quaternion', 'rotation_6d')
        
        # TODO: handle null
        self.crop_randomizer = dmvc.CropRandomizer(
            input_shape=obs_shape,
            crop_height=crop_shape[0],
            crop_width=crop_shape[1],
        )

        # Create policy network
        self.policy = build_equi_policy(
            cfg.hidden_dim,
            action_dim,
            num_layer=cfg.num_layers,
            layer_norm=1,
            dropout=cfg.dropout,
            use_layer_norm=cfg.use_layer_norm,
        )

        # Apply weight initialization
        self._initialize_weights(cfg)

    def _initialize_weights(self, cfg: ActorConfig):
        """Apply weight initialization to all networks."""
        # Determine initialization distributions
        intermediate_init = cfg.actor_intermediate_layer_init_distribution
        if cfg.orth and intermediate_init == "default":
            intermediate_init = "orthogonal"

        # Initialize compression layers
        if cfg.orth:
            # Backward compatibility: use existing orthogonal initialization
            self.compress.apply(utils.orth_weight_init)
        else:
            # Use the specified distribution for compression layers
            utils.apply_initialization_to_network(self.compress, intermediate_init)

        # Initialize policy network intermediate layers (exclude final layer)
        utils.apply_initialization_to_network(self.policy, intermediate_init, exclude_final_layer=True)

        # Initialize final layer with specific configuration if provided
        if cfg.actor_last_layer_init_scale is not None:
            final_layer = None
            for module in reversed(list(self.policy.modules())):
                if isinstance(module, nn.Linear):
                    final_layer = module
                    break

            if final_layer is not None:
                utils.initialize_layer_weights(
                    final_layer,
                    cfg.actor_last_layer_init_distribution,
                    cfg.actor_last_layer_init_scale,
                )

    def forward_reg(self, obs: dict[str, torch.Tensor], std: float):
        if isinstance(self.compress, SpatialEmb):
            assert not self.residual_actor, "Not implemented"
            feat = self.compress.forward(obs["feat"], obs["observation.state"])
        else:
            feat = obs["feat"].flatten(1, -1)
            feat = self.compress(feat)

        all_input = [feat]
        if self.prop_dim > 0:
            prop = obs["observation.state"]
            all_input.append(prop)
            if self.residual_actor:
                # The residual actor takes the base action as input alongside the state
                all_input.append(obs["observation.base_action"])

        policy_input = torch.cat(all_input, dim=-1)

        mu: torch.Tensor = self.policy(policy_input)

        # Scale the mean by action_scale
        # NOTE: std is alreay in environment action space (more interpretable)
        scaled_mu = mu * self.cfg.action_scale

        # Create distribution with scaled mean but environment-scale std
        action_dist = utils.TruncatedNormal(scaled_mu, std)

        return action_dist  # noqa: RET504

    def get6DRotation(self, quat):
        # data is in xyzw, but rotation transformer takes wxyz
        return self.quaternion_to_sixd.forward(quat[:, [3, 0, 1, 2]]) 

    def build_equi_policy(self, in_dim, hidden_dim, action_dim, num_layer, layer_norm, dropout, use_layer_norm=True):

        # TODO:
        # - Include LayerNorm?
        # - Include Dropout?

        in_type     = self.feat_type
        hidden_type = nn.FieldType(self.group, hidden_dim * [self.group.regular_repr])
        out_type    = nn.FieldType(self.group, action_dim * self.action_shape)

        layers = [nn.Linear(self.feat_type, hidden_type), 
                  nn.ReLU(hidden_type)]
        for i in range(num_layer):
            layers.append(nn.Linear(hidden_type, hidden_type))
            layers.append(nn.ReLU(hidden_type))

        layers.append(nn.Linear(hidden_type, out_type))
        layers.append(torch.nn.Tanh())
        return nn.Sequential(*layers)
        
    def forward(self, nobs):

        feat = nobs["feat"]
        ee_pos = nobs["robot0_eef_pos"]
        ee_quat = nobs["robot0_eef_quat"]
        ee_q = nobs["robot0_gripper_qpos"]
        base_action = nobs["observation.base_action"]
        # ih = nobs["robot0_eye_in_hand_image"]

        ee_pos = rearrange(ee_pos, "b t d -> (b t) d")
        ee_quat = rearrange(ee_quat, "b t d -> (b t) d")
        ee_q = rearrange(ee_q, "b t d -> (b t) d")
        base_action = rearrange(ee_q, "b t d -> (b t) d")
        # ih = rearrange(ih, "b t c h w -> (b t) c h w")

        # According to Claude:
        """
        Action dim: 7

        ┌─────────────────────────────────────────────────────────┐
        │  Index │  Name       │  Description                    │
        ├────────┼─────────────┼─────────────────────────────────┤
        │  0     │  x         │  End-effector X position delta  │
        │  1     │  y         │  End-effector Y position delta  │
        │  2     │  z         │  End-effector Z position delta  │
        │  3     │  rx        │  End-effector X rotation delta  │
        │  4     │  ry        │  End-effector Y rotation delta  │
        │  5     │  rz        │  End-effector Z rotation delta  │
        │  6     │  gripper    │  Gripper command (-1=open, 1=close) │
        └─────────────────────────────────────────────────────────┘

        Action bounds: [-1, 1] for all dimensions
        """

        ee_rot = self.get6DRotation(ee_quat)
        # ih = self.crop_randomizer(ih)

        pos_xy = ee_pos[:, 0:2]
        pos_z = ee_pos[:, 2:3]
        # ih_out = self.enc_ih(ih)
        
        action_xy      = base_action[:, 0:2]  # x, y
        action_z       = base_action[:, 2:3]  # z
        action_rx_ry   = base_action[:, 3:5]  # rx, ry
        action_rz      = base_action[:, 5:6]  # rz
        action_gripper = base_action[:, 6:7]  # gripper
        
        features = torch.cat(
            [
                # Compressed Observation Features
                feat,

                # Proprioceptive Data
                pos_xy,
                ee_rot[:, 0:1],  # R[0][0]
                ee_rot[:, 3:4],  # R[1][0]
                ee_rot[:, 1:2],  # R[0][1]
                ee_rot[:, 4:5],  # R[1][1]
                ee_rot[:, 2:3],  # R[0][2]
                ee_rot[:, 5:6],  # R[1][2]
                pos_z,
                ee_q,
                
                # Base action components
                action_xy[:, 0:1],    # x
                action_xy[:, 1:2],    # y
                action_rx_ry[:, 0:1], # rx
                action_rx_ry[:, 1:2], # ry
                action_z,             # z
                action_rz,            # rz
                action_gripper,       # gripper
            ],
            dim=1
        )

        features = nn.GeometricTensor(features, self.feat_type.in_type)
        mu: torch.Tensor = self.policy(features)

        # Scale the mean by action_scale
        # NOTE: std is alreay in environment action space (more interpretable)
        scaled_mu = mu * self.cfg.action_scale

        # Create distribution with scaled mean but environment-scale std
        action_dist = utils.TruncatedNormal(scaled_mu, std)

        return action_dist  # noqa: RET504
