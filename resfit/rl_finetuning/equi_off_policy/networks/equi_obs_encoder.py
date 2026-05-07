import torch
import numpy as np
from torchvision import models as vision_models

from escnn import gspaces, nn
from escnn.group import CyclicGroup
from einops import rearrange

import resfit.rl_finetuning.equi_off_policy.common_utils.crop_randomizer as dmvc
# from   resfit.rl_finetuning.equi_off_policy.networks.equi_encoder import EquivariantEncoder128
from   resfit.rl_finetuning.equi_off_policy.common_utils.module_attr_mixin import ModuleAttrMixin
from   resfit.rl_finetuning.equi_off_policy.common_utils.rotation_transformer import RotationTransformer
from   resfit.rl_finetuning.equi_off_policy.networks.equi_encoder import EquivariantResEncoder76Cyclic
from   resfit.rl_finetuning.equi_off_policy.common_utils.field_norm import FieldNorm
from   robomimic.models.base_nets import SpatialSoftmax

from resfit.rl_finetuning.equi_off_policy.networks.equi_normalizer import LinearNormalizer
from resfit.rl_finetuning.equi_off_policy.networks.ablate_equi_encoder import ResEncoder76InHand


class Identity(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

# class InHandEncoder(torch.nn.Module):
#     def __init__(self, out_size):
#         super().__init__()
#         net = vision_models.resnet18(norm_layer=lambda c: torch.nn.GroupNorm(min(32, c), c))
#         self.resnet = torch.nn.Sequential(*(list(net.children())[:-2]))
#         self.spatial_softmax = SpatialSoftmax([512, 3, 3], num_kp=out_size // 2)

#     def forward(self, ih):
#         batch_size = ih.shape[0]
#         return self.spatial_softmax(self.resnet(ih)).reshape(batch_size, -1)


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


class EquivariantResObsEnc(ModuleAttrMixin):
    def __init__(
        self,
        group,
        obs_shape=(3, 84, 84),
        crop_shape=(76, 76),
        N=8,
        n_hidden=128,
        initialize=True,
        ih_n_out=96,
        ih_channels=(16, 32, 64, 96),
        ih_blocks_per_stage=1,
        absolute_actions=False,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        self.obs_channel = obs_shape[0]
        self.n_hidden = n_hidden
        self.N = N
        self.group = group
        self.absolute_actions = absolute_actions
        self.use_layer_norm = use_layer_norm

        self.prop_shape = (
            4 * [self.group.irrep(1)]               # pos_xy + 3 rotation col pairs
            + 3 * [self.group.trivial_repr]         # pos_z, ee_q(2)
        )

        self.action_shape = [
            self.group.irrep(1),       # action_xy       (dims 0–1)
            self.group.trivial_repr,   # action_z        (dim 2)
            self.group.irrep(1),       # action_rx_ry    (dims 3–4)
            self.group.trivial_repr,   # action_rz       (dim 5)
            self.group.trivial_repr,   # action_gripper  (dim 6)
        ]

        # ---- FieldTypes for downstream modules (slicing + skip injection) ----
        self.vis_type    = nn.FieldType(self.group, n_hidden  * [self.group.regular_repr])
        self.ih_type     = nn.FieldType(self.group, ih_n_out  * [self.group.trivial_repr])
        self.prop_type   = nn.FieldType(self.group, self.prop_shape)
        self.action_type = nn.FieldType(self.group, self.action_shape)
        self.vis_ih_type = self.vis_type + self.ih_type   # FieldType supports + as rep concat

        # Flat dims for slicing GeometricTensors
        self.vis_dim    = self.vis_type.size      # n_hidden * N
        self.ih_dim     = self.ih_type.size       # ih_n_out
        self.prop_dim   = self.prop_type.size     # 11
        self.action_dim = self.action_type.size   # 7
        self.vis_ih_dim = self.vis_ih_type.size

        self.enc_out_type_critic = nn.FieldType(
            self.group,
            n_hidden * [self.group.regular_repr]    # agentview
            + ih_n_out * [self.group.trivial_repr]  # ih
            + self.prop_shape
        )

        self.enc_out_type_actor = nn.FieldType(
            self.group,
            n_hidden * [self.group.regular_repr]    # agentview
            + ih_n_out * [self.group.trivial_repr]  # ih
            + self.prop_shape
            + self.action_shape
        )

        self.enc_obs = EquivariantResEncoder76Cyclic(
            obs_channel=self.obs_channel, 
            n_out=n_hidden, 
            N=N
        )

        self.enc_ih  = ResEncoder76InHand(
            obs_channel=self.obs_channel,
            channels=ih_channels,
            n_out=ih_n_out,
            blocks_per_stage=ih_blocks_per_stage,
        )

        self.gTgc = torch.Tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        self.crop_randomizer = dmvc.CropRandomizer(
            input_shape=obs_shape,
            crop_height=crop_shape[0],
            crop_width=crop_shape[1],
        )

        self.action_layout = self._build_action_layout(self.action_shape, self.group)

    # As a static method on the class
    @staticmethod
    def _build_action_layout(action_shape, group):
        layout = []
        pos = 0
        for rep in action_shape:
            size = rep.size
            if rep == group.trivial_repr:
                kind = "trivial"
            elif rep == group.irrep(1):
                kind = "irrep1"
            else:
                raise ValueError(f"Unsupported rep in action_shape: {rep}")
            layout.append((slice(pos, pos + size), kind))
            pos += size
        return layout

    def get6DRotation(self, quat):
        return quaternion_to_rotation_6d(quat[:, [3, 0, 1, 2]])

    def set_normalizer(self,
                       normalizer: LinearNormalizer,
                       robot_base_xy: torch.Tensor=None):
        """
        Store the pre-built LinearNormalizer and robot base position.

        LinearNormalizer inherits from DictOfTensorMixin(nn.Module), so storing
        it as self.normalizer makes it a submodule — it follows .to(device),
        .cuda(), and appears in state_dict() automatically.
        """
        self.normalizer = normalizer
        # self.register_buffer("_robot_base_xy", robot_base_xy.clone().float())
        self._normalizers_ready = True

        for name in ["pos_xy", "pos_z", "ee_rot", "ee_q"]:
            s = self.normalizer[name].params_dict["scale"]
            o = self.normalizer[name].params_dict["offset"]
            inp = self.normalizer[name].params_dict["input_stats"]
            print(f"{name}: scale={s.detach().numpy()}, offset={o.detach().numpy()}")
            print(f"  input_stats: min={inp['min'].detach().numpy()}, max={inp['max'].detach().numpy()}")


    def forward(self, nobs):
        assert self._normalizers_ready, "Call set_normalizer() before forward()"

        obs = nobs["observation.images.agentview"]
        ih  = nobs["observation.images.robot0_eye_in_hand"]
        base_action = nobs["observation.base_action"]
        state = nobs["observation.state"]

        batch_size = obs.shape[0]

        # ---- Images (inline: uint8 [0,255] → float [-1,1]) ----
        obs = obs.float() / 255.0 * 2.0 - 1.0
        ih  = ih.float()  / 255.0 * 2.0 - 1.0

        ih     = self.crop_randomizer(ih)
        ih_out = self.enc_ih(ih)

        obs     = self.crop_randomizer(obs)
        enc_out = self.enc_obs(obs).tensor.reshape(batch_size, -1)

        # ---- Proprioception ----
        ee_pos  = state[...,  :3]
        ee_quat = state[..., 3:7]
        ee_q    = state[..., 7:9]

        pos_xy = self.normalizer["pos_xy"].normalize(ee_pos[:, 0:2])
        pos_z  = self.normalizer["pos_z"].normalize(ee_pos[:, 2:3])
        ee_rot = self.normalizer["ee_rot"].normalize(self.get6DRotation(ee_quat))
        ee_q   = self.normalizer["ee_q"].normalize(ee_q)

        # ---- Action ----
        action_xy      = self.normalizer["action_xy"].normalize(base_action[:, 0:2])
        action_z       = self.normalizer["action_z"].normalize(base_action[:, 2:3])
        action_rx_ry   = self.normalizer["action_rx_ry"].normalize(base_action[:, 3:5])
        action_rz      = self.normalizer["action_rz"].normalize(base_action[:, 5:6])
        action_gripper = self.normalizer["action_gripper"].normalize(base_action[:, 6:7])

        # ---- Concatenate into equivariant feature vector ----
        critic_features = torch.cat(
            [
                enc_out,
                ih_out,

                # Proprioception (irrep(1) pairs interleaved per rotation matrix column)
                pos_xy,
                ee_rot[:, 0:1], ee_rot[:, 3:4],  # col 0: (R00, R10)
                ee_rot[:, 1:2], ee_rot[:, 4:5],  # col 1: (R01, R11)
                ee_rot[:, 2:3], ee_rot[:, 5:6],  # col 2: (R02, R12)
                pos_z,
                ee_q,
            ],
            dim=1,
        )
        actor_features = torch.empty_like(critic_features).copy_(critic_features)
        critic_features = nn.GeometricTensor(critic_features, self.enc_out_type_critic)

        actor_features = torch.cat(
            [
                actor_features,

                action_xy,        # 0:2  → irrep(1)
                action_z,         # 2:3  → trivial
                action_rx_ry,     # 3:5  → irrep(1)
                action_rz,        # 5:6  → trivial
                action_gripper,   # 6:7  → trivial
            ],
            dim=1,
        )
        actor_features = nn.GeometricTensor(actor_features, self.enc_out_type_actor)

        return actor_features, critic_features