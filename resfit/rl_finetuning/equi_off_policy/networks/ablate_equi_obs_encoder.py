import torch
import numpy as np
from torchvision import models as vision_models

import resfit.rl_finetuning.equi_off_policy.common_utils.crop_randomizer as dmvc
from   resfit.rl_finetuning.equi_off_policy.common_utils.module_attr_mixin import ModuleAttrMixin
from   robomimic.models.base_nets import SpatialSoftmax

from resfit.rl_finetuning.equi_off_policy.common_utils.equi_normalizer_util import LinearNormalizer
from resfit.rl_finetuning.equi_off_policy.networks.ablate_equi_encoder import ResBlock, ResEncoder76


# ---------------------------------------------------------------------------
# Helpers (unchanged from equivariant version)
# ---------------------------------------------------------------------------

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
        self.spatial_softmax = SpatialSoftmax([512, 3, 3], num_kp=out_size // 2)

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

# ---------------------------------------------------------------------------
# Non-equivariant ResObsEnc (mirrors EquivariantResObsEnc)
#
# Input dimension breakdown (N=8, n_hidden=128):
#   agentview encoder  : n_hidden * N       = 1024
#   in-hand encoder    : n_hidden           = 128
#   proprioception     : 4*2 + 3*1          = 11
#       pos_xy(2), ee_rot(6), pos_z(1), ee_q(2)
#   action             : 2*2 + 3*1          = 7
#       action_xy(2), action_rx_ry(2), action_z(1), action_rz(1), action_gripper(1)
#   ─────────────────────────────────────────────
#   total              :                      1170
#
# Output dimension: n_hidden * N = 1024
# ---------------------------------------------------------------------------

class ResObsEnc(ModuleAttrMixin):
    def __init__(
        self,
        obs_shape=(3, 84, 84),
        crop_shape=(76, 76),
        N=8,
        n_hidden=128,
        absolute_actions=False,
    ):
        super().__init__()
        self.obs_channel = obs_shape[0]
        self.n_hidden = n_hidden
        self.N = N
        self.absolute_actions = absolute_actions

        # ---- Dimension bookkeeping (replaces escnn field types) ----
        # Proprioception: pos_xy(2) + ee_rot(6) + pos_z(1) + ee_q(2)
        self.prop_dim = 11
        # Action: action_xy(2) + action_rx_ry(2) + action_z(1) + action_rz(1) + action_gripper(1)
        self.action_dim = 7

        enc_in_dim = (
            n_hidden * N          # agentview visual features
            + n_hidden            # in-hand visual features
            + self.prop_dim
            + self.action_dim
        )
        enc_out_dim = n_hidden * N

        self.enc_out = torch.nn.Linear(enc_in_dim, enc_out_dim)

        self.enc_obs = ResEncoder76(self.obs_channel, n_hidden, N)
        self.enc_ih = InHandEncoder(n_hidden).to(self.device)

        self.gTgc = torch.Tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

        self.crop_randomizer = dmvc.CropRandomizer(
            input_shape=obs_shape,
            crop_height=crop_shape[0],
            crop_width=crop_shape[1],
        )

    def get6DRotation(self, quat):
        return quaternion_to_rotation_6d(quat[:, [3, 0, 1, 2]])

    def set_normalizer(self,
                       normalizer: LinearNormalizer,
                       robot_base_xy: torch.Tensor = None):
        """
        Store the pre-built LinearNormalizer and robot base position.

        LinearNormalizer inherits from DictOfTensorMixin(nn.Module), so storing
        it as self.normalizer makes it a submodule — it follows .to(device),
        .cuda(), and appears in state_dict() automatically.
        """
        self.normalizer = normalizer
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
        enc_out = self.enc_obs(obs).reshape(batch_size, -1)

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

        # ---- Concatenate feature vector ----
        # Same ordering as the equivariant version for reproducibility,
        # including the interleaved rotation column layout.
        features = torch.cat(
            [
                enc_out,
                ih_out,

                # Proprioception (same interleaved rotation layout as equivariant)
                pos_xy,
                ee_rot[:, 0:1], ee_rot[:, 3:4],  # col 0: (R00, R10)
                ee_rot[:, 1:2], ee_rot[:, 4:5],  # col 1: (R01, R11)
                ee_rot[:, 2:3], ee_rot[:, 5:6],  # col 2: (R02, R12)
                pos_z,
                ee_q,

                # Action
                action_xy,
                action_rx_ry,
                action_z,
                action_rz,
                action_gripper,
            ],
            dim=1,
        )

        enc_out = self.enc_out(features)

        return enc_out