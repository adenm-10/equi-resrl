import torch
import numpy as np

from escnn import gspaces, nn as enn
from escnn.group import CyclicGroup

import resfit.rl_finetuning.equi_off_policy.common_utils.crop_randomizer as dmvc
from resfit.rl_finetuning.equi_off_policy.common_utils.module_attr_mixin import ModuleAttrMixin
from resfit.rl_finetuning.equi_off_policy.networks.equi_encoder import EquivariantResEncoder76Cyclic
from resfit.rl_finetuning.equi_off_policy.networks.ablate_equi_encoder import (
    ResEncoder76,
    ResEncoder76InHand,
)
from resfit.rl_finetuning.equi_off_policy.networks.equi_normalizer import LinearNormalizer


def quaternion_to_rotation_6d(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (w,x,y,z) to 6D rotation representation."""
    q = q / torch.norm(q, dim=-1, keepdim=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    matrix = torch.stack([
        1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y),
            2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x),
            2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)
    return matrix[..., :2, :].clone().reshape(*matrix.shape[:-2], 6)


class ResObsEnc(ModuleAttrMixin):
    """
    Unified observation encoder for SO(2)/C_N-equivariant and scalar (ablation) paths.

    equivariant=True:
      - Agentview: EquivariantResEncoder76Cyclic
      - Returns features wrapped in escnn.nn.GeometricTensor

    equivariant=False:
      - Agentview: ResEncoder76 (scalar)
      - All FieldType / action_layout / action_shape attributes are present but set to None
      - Returns plain torch.Tensor features

    The in-hand encoder (ResEncoder76InHand) is scalar in both modes — the in-hand camera
    frame rotates with the gripper, so world-frame SO(2) equivariance does not apply there.

    Integer dims (prop_dim, action_dim, enc_out_dim_critic, enc_out_dim_actor) are
    exposed unconditionally in both modes.
    """

    def __init__(
        self,
        obs_shape=(3, 84, 84),
        crop_shape=(76, 76),
        N=8,
        n_hidden=128,
        ih_channels=(16, 32, 64, 128),
        ih_blocks_per_stage=1,
        use_norms=True,
        *,
        equivariant: bool = True,
        group=None,            # auto-built from N if None and equivariant=True
        initialize: bool = True,  # ignored when equivariant=False
    ):
        super().__init__()

        if not equivariant and group is not None:
            raise ValueError("`group` should be None when equivariant=False")

        self.equivariant = equivariant
        self.obs_channel = obs_shape[0]
        self.n_hidden = n_hidden
        self.N = N

        self.register_buffer(
            "robot_base_xy",
            torch.zeros(2, dtype=torch.float32),
        )

        # ----- Common dims and modules -----
        self.prop_dim = 11
        self.action_dim = 7
        self._normalizers_ready = False

        self.crop_randomizer = dmvc.CropRandomizer(
            input_shape=obs_shape,
            crop_height=crop_shape[0],
            crop_width=crop_shape[1],
        )

        # In-hand encoder is scalar in both modes
        self.enc_ih = ResEncoder76InHand(
            obs_channel=self.obs_channel,
            channels=ih_channels,
            n_out=n_hidden,
            blocks_per_stage=ih_blocks_per_stage,
        )

        # ----- Branch on equivariance -----
        if equivariant:
            self.group = group if group is not None else gspaces.no_base_space(CyclicGroup(N))

            # Representations
            self.prop_shape = (
                4 * [self.group.irrep(1)]        # pos_xy + 3 rotation column pairs
                + 3 * [self.group.trivial_repr]  # pos_z, ee_q(2)
            )
            self.action_shape = [
                self.group.irrep(1),       # action_xy       (0:2)
                self.group.trivial_repr,   # action_z        (2:3)
                self.group.irrep(1),       # action_rx_ry    (3:5)
                self.group.trivial_repr,   # action_rz       (5:6)
                self.group.trivial_repr,   # action_gripper  (6:7)
            ]

            # FieldTypes
            self.vis_type    = enn.FieldType(self.group, n_hidden * [self.group.regular_repr])
            self.ih_type     = enn.FieldType(self.group, n_hidden * [self.group.trivial_repr])
            self.prop_type   = enn.FieldType(self.group, self.prop_shape)
            self.action_type = enn.FieldType(self.group, self.action_shape)
            self.vis_ih_type = self.vis_type + self.ih_type

            self.enc_out_type_critic = enn.FieldType(
                self.group,
                n_hidden * [self.group.regular_repr]
                + n_hidden * [self.group.trivial_repr]
                + self.prop_shape,
            )
            self.enc_out_type_actor = enn.FieldType(
                self.group,
                n_hidden * [self.group.regular_repr]
                + n_hidden * [self.group.trivial_repr]
                + self.prop_shape
                + self.action_shape,
            )

            self.enc_obs = EquivariantResEncoder76Cyclic(
                obs_channel=self.obs_channel,
                n_out=n_hidden,
                N=N,
                initialize=initialize,
                use_norms=use_norms,   # NOTE: encoder kwarg is singular
            )

            self.action_layout = self._build_action_layout(self.action_shape, self.group)

            # Integer dims derived from FieldType.size — guarantees parity with scalar branch
            self.enc_out_dim_critic = self.enc_out_type_critic.size
            self.enc_out_dim_actor  = self.enc_out_type_actor.size
        else:
            self.group = None

            # Set all equivariant attributes to None for grep-ability / safe getattr
            self.prop_shape  = None
            self.action_shape = None
            self.vis_type    = None
            self.ih_type     = None
            self.prop_type   = None
            self.action_type = None
            self.vis_ih_type = None
            self.enc_out_type_critic = None
            self.enc_out_type_actor  = None
            self.action_layout = None

            self.enc_obs = ResEncoder76(
                obs_channel=self.obs_channel,
                n_out=n_hidden,
                N=N,
                use_norms=use_norms,
            )

            self.enc_out_dim_critic = n_hidden * N + n_hidden + self.prop_dim
            self.enc_out_dim_actor  = self.enc_out_dim_critic + self.action_dim

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
        # return quaternion_to_rotation_6d(quat) # i believe its already stored in w, x, y, z from looking at dataset stats

    def set_normalizer(
        self,
        normalizer: LinearNormalizer,
        robot_base_xy: torch.Tensor | np.ndarray | list | None = None,
        verbose: bool = False,
    ):
        """Attach the LinearNormalizer; optionally set robot_base_xy used to
        center pos_xy at runtime so the irrep(1) normalizer is applied about
        the rotation center."""
        self.normalizer = normalizer

        if robot_base_xy is not None:
            base = (
                torch.as_tensor(robot_base_xy, dtype=torch.float32)
                .reshape(2)
                .to(self.robot_base_xy.device)
            )
            self.robot_base_xy.copy_(base)
            print(f"[ResObsEnc] robot_base_xy set to {base.tolist()}")
        else:
            print("[ResObsEnc] robot_base_xy left at (0, 0) — pos_xy will NOT be centered")

        self._normalizers_ready = True

        keys = [
            "pos_xy", "pos_z", "ee_rot", "ee_q",
            "action_xy", "action_z", "action_rx_ry", "action_rz", "action_gripper",
        ]
        for name in keys:
            params = self.normalizer[name].params_dict
            s = params["scale"]
            o = params["offset"]
            print(f"{name}: scale={s.detach().cpu().numpy()}, offset={o.detach().cpu().numpy()}")
            if verbose and "input_stats" in params:
                inp = params["input_stats"]
                print(f"  input_stats: min={inp['min'].detach().cpu().numpy()}, "
                      f"max={inp['max'].detach().cpu().numpy()}")

    def forward(self, nobs):
        assert self._normalizers_ready, "Call set_normalizer() before forward()"

        obs         = nobs["observation.images.agentview"]
        ih          = nobs["observation.images.robot0_eye_in_hand"]
        base_action = nobs["observation.base_action"]
        state       = nobs["observation.state"]

        batch_size = obs.shape[0]

        # ----- Images (uint8 [0,255] → float [-1,1]) -----
        obs = obs.float() / 255.0 * 2.0 - 1.0
        ih  = ih.float()  / 255.0 * 2.0 - 1.0

        ih     = self.crop_randomizer(ih)
        ih_out = self.enc_ih(ih)

        obs     = self.crop_randomizer(obs)
        enc_raw = self.enc_obs(obs)
        enc_out = (enc_raw.tensor if self.equivariant else enc_raw).reshape(batch_size, -1)

        # ---- Proprioception ----
        ee_pos  = state[...,  :3]
        ee_quat = state[..., 3:7]
        ee_q    = state[..., 7:9]

        pos_xy_centered = ee_pos[:, 0:2] - self.robot_base_xy  # (B, 2), buffer broadcasts
        pos_xy = self.normalizer["pos_xy"].normalize(pos_xy_centered)
        pos_z  = self.normalizer["pos_z"].normalize(ee_pos[:, 2:3])
        ee_rot = self.normalizer["ee_rot"].normalize(self.get6DRotation(ee_quat))
        ee_q   = self.normalizer["ee_q"].normalize(ee_q)

        # ---- Action ----
        action_xy      = self.normalizer["action_xy"].normalize(base_action[:, 0:2])
        action_z       = self.normalizer["action_z"].normalize(base_action[:, 2:3])
        action_rx_ry   = self.normalizer["action_rx_ry"].normalize(base_action[:, 3:5])
        action_rz      = self.normalizer["action_rz"].normalize(base_action[:, 5:6])
        action_gripper = self.normalizer["action_gripper"].normalize(base_action[:, 6:7])

        # ----- Concatenate feature vector -----
        critic_features = torch.cat(
            [
                enc_out,
                ih_out,
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

        # Wrap in GeometricTensor for equivariant path
        if self.equivariant:
            critic_features = enn.GeometricTensor(critic_features, self.enc_out_type_critic)
            actor_features  = enn.GeometricTensor(actor_features,  self.enc_out_type_actor)

        return actor_features, critic_features