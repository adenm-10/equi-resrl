import torch
from torchvision import models as vision_models

from escnn import gspaces, nn
from escnn.group import CyclicGroup
from einops import rearrange

import resfit.rl_finetuning.equi_off_policy.common_utils.crop_randomizer as dmvc
from   resfit.rl_finetuning.equi_off_policy.common_utils.module_attr_mixin import ModuleAttrMixin
from   resfit.rl_finetuning.equi_off_policy.common_utils.rotation_transformer import RotationTransformer
from   resfit.rl_finetuning.equi_off_policy.networks.equi_encoder import EquivariantResEncoder76Cyclic
from   robomimic.models.base_nets import SpatialSoftmax

class Identity(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
    
class InHandEncoder(torch.nn.Module):
    def __init__(self, out_size):
        super().__init__()
        net = vision_models.resnet18(norm_layer=Identity)
        self.resnet = torch.nn.Sequential(*(list(net.children())[:-2]))
        self.spatial_softmax = SpatialSoftmax([512, 3, 3], num_kp=out_size//2)

    def forward(self, ih):
        batch_size = ih.shape[0]
        return self.spatial_softmax(self.resnet(ih)).reshape(batch_size, -1)

# VitEncoder
class EquivariantResObsEnc(ModuleAttrMixin):
    def __init__(
        self,
        obs_shape=(3, 84, 84),
        crop_shape=(76, 76),
        n_hidden=128,
        N=8,
        initialize=True,
    ):
        super().__init__()
        obs_channel = obs_shape[0]
        self.n_hidden = n_hidden
        self.N = N

        # TODO: Why use CyclicGroup explicitly?
        # self.group = gspaces.no_base_space(CyclicGroup(self.N))
        self.group = gspaces.rot2dOnR2(N=N)

        self.action_shape = 2 * [self.group.irrep(1)]               # action_xy, action_rx_ry
                          + 3 * [self.group.trivial_repr],          # action_z, action_rz, action_gripper
        self.prop_shape   = 4 * [self.group.irrep(1)]               # pos_xy + 3 rotation columns
                          + 3 * [self.group.trivial_repr]           # pos_z, ee_q (2)
        
        self.full_feat_shape = self.group,
                               self.n_hidden   * [self.group.regular_repr] # agentview img
                               + self.n_hidden * [self.group.trivial_repr] # ih
                               + self.prop_shape                           # proprioceptive state
                               + self.action_shape                         # base action 
        
        self.out_type = nn.FieldType(self.group, self.n_hidden * [self.group.regular_repr])
        self.enc_obs = EquivariantEncoder128(obs_channel, self.n_hidden, initialize)
        self.enc_ih = InHandEncoder(self.n_hidden).to(self.device)
        
        self.quaternion_to_sixd = RotationTransformer('quaternion', 'rotation_6d')

        self.gTgc = torch.Tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

        self.crop_randomizer = dmvc.CropRandomizer(
            input_shape=obs_shape,
            crop_height=crop_shape[0],
            crop_width=crop_shape[1],
        )

    def get6DRotation(self, quat):
        # data is in xyzw, but rotation transformer takes wxyz
        return self.quaternion_to_sixd.forward(quat[:, [3, 0, 1, 2]]) 
        
    def forward(self, nobs, action):

        obs = nobs["agentview_image"]
        ih = nobs["robot0_eye_in_hand_image"]
        ee_pos = nobs["robot0_eef_pos"]
        ee_quat = nobs["robot0_eef_quat"]
        ee_q = nobs["robot0_gripper_qpos"]
        base_action = nobs["observation.base_action"]

        batch_size, t = obs.shape[0], obs.shape[1]

        obs = rearrange(obs, "b t c h w -> (b t) c h w")
        ih = rearrange(ih, "b t c h w -> (b t) c h w")
        ee_pos = rearrange(ee_pos, "b t d -> (b t) d")
        ee_quat = rearrange(ee_quat, "b t d -> (b t) d")
        ee_q = rearrange(ee_q, "b t d -> (b t) d")
        base_action = rearrange(ee_q, "b t d -> (b t) d")
        ih = rearrange(ih, "b t c h w -> (b t) c h w")

        # According to Claude:
        """
        Action dim: 7

        ┌─────────────────────────────────────────────────────────┐
        │  Index │  Name       │  Description                    │
        ├────────┼─────────────┼─────────────────────────────────┤
        │  0     │  x          │  End-effector X position delta  │
        │  1     │  y          │  End-effector Y position delta  │
        │  2     │  z          │  End-effector Z position delta  │
        │  3     │  rx         │  End-effector X rotation delta  │
        │  4     │  ry         │  End-effector Y rotation delta  │
        │  5     │  rz         │  End-effector Z rotation delta  │
        │  6     │  gripper    │  Gripper command (-1=open, 1=close) │
        └─────────────────────────────────────────────────────────┘

        Action bounds: [-1, 1] for all dimensions
        """

        ee_rot = self.get6DRotation(ee_quat)
        ih = self.crop_randomizer(ih)

        enc_out = self.enc_obs(obs).tensor.reshape(batch_size * t, -1)  # b d
        ih_out = self.enc_ih(ih)

        pos_xy = ee_pos[:, 0:2]
        pos_z = ee_pos[:, 2:3]
        
        action_xy      = base_action[:, 0:2]  # x, y
        action_z       = base_action[:, 2:3]  # z
        action_rx_ry   = base_action[:, 3:5]  # rx, ry
        action_rz      = base_action[:, 5:6]  # rz
        action_gripper = base_action[:, 6:7]  # gripper
        
        features = torch.cat(
            [
                # Compressed Observation Features
                enc_out,
                ih_out,

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

        features = rearrange(features, "(b t) d -> b t d", b=batch_size)
        return nn.GeometricTensor(features, self.out_type)