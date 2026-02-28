import torch
from torchvision import models as vision_models

from escnn import gspaces, nn
from escnn.group import CyclicGroup
from einops import rearrange

import resfit.rl_finetuning.equi_off_policy.common_utils.crop_randomizer as dmvc
from   resfit.rl_finetuning.equi_off_policy.networks.equi_encoder import EquivariantEncoder128
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

def quaternion_to_rotation_6d(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (w,x,y,z) to 6D rotation representation."""
    q = q / torch.norm(q, dim=-1, keepdim=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    matrix = torch.stack([
        1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y),
        2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x),
        2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)
    return matrix[..., :2, :].clone().reshape(*matrix.shape[:-2], 6)

# VitEncoder
class EquivariantResObsEnc(ModuleAttrMixin):
    def __init__(
        self,
        group,
        obs_shape=(3, 84, 84),
        crop_shape=(76, 76),
        N=8,
        n_hidden=128,
        initialize=True,
    ):
        super().__init__()
        self.obs_channel = obs_shape[0]
        self.n_hidden = n_hidden
        self.N = N
        self.group = group

        self.prop_shape   = 4 * [self.group.irrep(1)] \
                          + 3 * [self.group.trivial_repr] # pos_xy + 3 rotation columns \ pos_z, ee_q (2)
        self.action_shape = 2 * [self.group.irrep(1)] \
                          + 3 * [self.group.trivial_repr] # action_xy, action_rx_ry \ action_z, action_rz, action_gripper

        self.enc_in_type  = nn.FieldType(
                                self.group,
                                n_hidden * [self.group.regular_repr] # agentview
                                + n_hidden * [self.group.trivial_repr] # ih
                                + self.prop_shape
                                + self.action_shape
                            )
        self.enc_out_type = nn.FieldType(self.group, self.n_hidden * [self.group.regular_repr])
        self.enc_out      = nn.Linear(self.enc_in_type, self.enc_out_type)

        # TODO: Equivariant parameters
        self.enc_obs = EquivariantResEncoder76Cyclic(self.obs_channel, self.n_hidden, initialize, self.N)
        self.enc_ih = InHandEncoder(self.n_hidden).to(self.device)
        # self.enc_obs = EquivariantEncoder128(group       = self.group,
        #                                      obs_channel = self.obs_channel, 
        #                                      n_out       = 32,              # can be n_hidden?  
        #                                      initialize  = initialize,
        #                                      N           = N)
        
        # self.quaternion_to_sixd = RotationTransformer('quaternion', 'rotation_6d')


        self.gTgc = torch.Tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

        self.crop_randomizer = dmvc.CropRandomizer(
            input_shape=obs_shape,
            crop_height=crop_shape[0],
            crop_width=crop_shape[1],
        )

    def get6DRotation(self, quat):
        # data is in xyzw, but rotation transformer takes wxyz
        # return self.quaternion_to_sixd.forward(quat[:, [3, 0, 1, 2]]) 
        return quaternion_to_rotation_6d(quat[:, [3, 0, 1, 2]])
        
    def forward(self, nobs):

        obs = nobs["observation.images.agentview"]
        ih = nobs["observation.images.robot0_eye_in_hand"]
        base_action = nobs["observation.base_action"]
        state = nobs["observation.state"]

        batch_size = obs.shape[0]

        # In Hand Image Embeddings
        ih = self.crop_randomizer(ih)
        ih = ih.float() / 255.0
        ih_out = self.enc_ih(ih)

        # Agent View Image Embeddings
        obs = self.crop_randomizer(obs)
        obs = obs.float() / 255.0
        enc_out = self.enc_obs(obs).tensor.reshape(batch_size, -1)  # b d

        # End Effector Pose 
        ee_pos = state[..., :3]
        ee_quat = state[..., 3:7]
        ee_q = state[..., 7:9]
        ee_rot = self.get6DRotation(ee_quat)

        # Proprioceptive Data
        pos_xy = ee_pos[:, 0:2]
        pos_z = ee_pos[:, 2:3]
        
        # Action Data
        action_xy      = base_action[:, 0:2]  # x, y
        action_z       = base_action[:, 2:3]  # z
        action_rx_ry   = base_action[:, 3:5]  # rx, ry
        action_rz      = base_action[:, 5:6]  # rz
        action_gripper = base_action[:, 6:7]  # gripper

        # Complete Feature Construction
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
                # action_xy[:, 0:1],    # x
                # action_xy[:, 1:2],    # y
                # action_rx_ry[:, 0:1], # rx
                # action_rx_ry[:, 1:2], # ry
                action_xy,      # x
                action_rx_ry,   # rx
                action_z,       # z
                action_rz,      # rz
                action_gripper, # gripper
            ],
            dim=1
        )

        # Feature Processing
        features = nn.GeometricTensor(features, self.enc_in_type)
        return self.enc_out(features)