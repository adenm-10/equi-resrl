import torch
import numpy as np

from escnn import gspaces, nn as enn
from escnn.group import CyclicGroup

import resfit.rl_finetuning.equi_off_policy.common_utils.crop_randomizer as dmvc
from resfit.rl_finetuning.config.rlpd import VitEncoderConfig
from resfit.rl_finetuning.equi_off_policy.common_utils.module_attr_mixin import ModuleAttrMixin
from resfit.rl_finetuning.equi_off_policy.networks.equi_encoder import EquivariantResEncoder76Cyclic
from resfit.rl_finetuning.equi_off_policy.networks.ablate_equi_encoder import ResEncoder76InHand
from resfit.rl_finetuning.equi_off_policy.networks.equi_normalizer import (
    ACTION_FIELDS,
    PROP_FIELDS,
    LinearNormalizer,
    arm_key,
)
from resfit.rl_finetuning.equi_off_policy.rl.ablated import plain
from resfit.rl_finetuning.off_policy.common_utils import RandomShiftsAug
from resfit.rl_finetuning.off_policy.networks.encoder import VitEncoder

ENCODERS = ("equivariant", "ablated", "resfit")  # equivariance.encoder_family


def arm_action_shape(group, hand_dof: int) -> list:
    """One arm's action, as representations of `group`."""
    return [
        group.irrep(1),       # action_xy      (0:2)
        group.trivial_repr,   # action_z       (2:3)
        group.irrep(1),       # action_rx_ry   (3:5)
        group.trivial_repr,   # action_rz      (5:6)
    ] + hand_dof * [group.trivial_repr]  # action_hand (6:6+H)


def build_action_layout(action_shape, group) -> list[tuple[slice, str]]:
    """(slice, kind) per representation, kind "irrep1" or "trivial": what equi_tanh and equi_clip
    read. ResFiT's agent builds its layout here too, under agent.actor.action_bound=equivariant."""
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
    The observation encoder, in one of three families (``encoder``, equivariance.encoder_family):

      "equivariant": the agentview camera through EquivariantResEncoder76Cyclic, after a 76x76
        crop, random in train mode and centred in eval mode. Returns features wrapped in
        escnn.nn.GeometricTensor.
      "ablated": that encoder built exactly as the equivariant family builds it, then exported to
        plain torch layers (rl/ablated.py), so it starts as the same function. Returns tensors of
        the same layout.
      "resfit": ResFiT's encoder, as off_policy/rl/q_agent.py builds and runs it: a ViT per camera
        (``vit_cfg``, agent.vit), ResFiT's random shifts (pad 4) in train mode, and ResFiT's image
        preprocessing (÷255 for uint8, then −0.5). Returns tensors whose image part is the ViTs'
        tokens, 81 per camera of 128 numbers each, flattened in ResFiT's order.

    In the first two, the in-hand encoders (ResEncoder76InHand, one per arm) are scalar — an
    in-hand camera frame rotates with its gripper, so world-frame SO(2) equivariance does
    not apply there. ResFiT's ViTs read the wrist cameras as they read the agentview one.

    ``n_arms`` generalises the single-arm layout to bimanual tasks. Per arm the state is
    ``[eef_pos 3, eef_quat 4, gripper_qpos G]`` and the action is ``[delta_pose 6, hand H]``,
    laid out as contiguous per-arm blocks. See docs/EQUIVARIANCE.md for the representations.
    Every family normalizes them alike, per field, so the three differ in their image features
    alone: [images | proprioception | base action].

    The field types, action_layout and the integer dims (prop_dim, action_dim,
    enc_out_dim_critic, enc_out_dim_actor) are built in every family. Behind the ResFiT encoder
    the image features are declared trivial fields, invariant, which is how an ablated critic is
    built behind it: as the equivariant critic already treats the wrist camera's features.

    ``ih_out_channel`` is each wrist camera's feature width. It is required, so no
    caller keeps its old coupling to ``n_hidden`` by accident: before 2026-09-29
    the wrist width was ``n_hidden``, i.e. equivariance.enc_degree_channel.
    ``encoder_norm`` and
    ``encoder_orth_init`` configure the equivariant agentview encoder, and so the ablated one.

    Images arrive in two formats: act() gets the env's float images in [0, 1]
    (dexmg.py), and the update gets the replay buffer's uint8 copies in [0, 255]
    (utils/dtype.py). Both map to the same floats in [-1, 1]. Until 2026-10-02
    both were divided by 255, which left act()'s images in [-1, -0.992];
    ``float_images_as_uint8`` reproduces that (EXPERIMENTS.md, 2026-10-02
    CORRECTION). ResFiT's encoder converts by dtype, as its own agent does.

    ``base_action_input=False`` is vanilla RL's encoder (base_policy=None): it reads
    no ``observation.base_action``, so the actor's features are the critic's.

    The interface, in the first two families (EXPERIMENTS.md, 2026-10-23 CODE). ``ih_channels`` are
    the wrist CNN's stage widths; ``enc_out_channel`` the agentview encoder's output fields, from its
    last conv, which is ``n_hidden`` unless set. ``critic_window`` and ``critic_window_ih`` give the
    critic a second, wider look at the images: a 9x9 conv over each encoder's 9x9 map, before its last
    max pool, to ``critic_window`` regular fields (equivariant, as the encoder's own last conv) and to
    ``critic_window_ih`` channels per wrist (plain, as the wrist encoder), each then a ReLU. Their
    outputs join the critic's features alone, [images | window | proprioception]; the actor's stay
    [images | proprioception | base action]. They are built with the random state forked, so every
    other weight at a seed is the one built without them.
    """

    def __init__(
        self,
        obs_shape=(3, 84, 84),
        crop_shape=(76, 76),
        N=8,
        n_hidden=128,
        ih_channels=(16, 32, 64, 128),
        ih_blocks_per_stage=1,
        *,
        encoder: str = "equivariant",
        vit_cfg: VitEncoderConfig | None = None,  # the ResFiT encoder's, agent.vit
        group=None,            # auto-built from N if None
        initialize: bool = True,
        n_arms: int = 1,
        gripper_dim: int = 2,
        hand_dof: int = 1,
        agentview_key: str = "observation.images.agentview",
        wrist_keys: tuple[str, ...] = ("observation.images.robot0_eye_in_hand",),
        ih_out_channel: int,
        encoder_norm: str = "none",
        encoder_orth_init: bool = False,
        float_images_as_uint8: bool = False,
        base_action_input: bool = True,
        enc_out_channel: int | None = None,
        critic_window: int = 0,
        critic_window_ih: int = 0,
    ):
        super().__init__()

        if encoder not in ENCODERS:
            raise ValueError(f"unknown encoder {encoder!r}; expected one of {ENCODERS}")
        if (encoder == "resfit") != (vit_cfg is not None):
            raise ValueError("vit_cfg configures the ResFiT encoder, and only it")
        if encoder == "resfit" and (enc_out_channel is not None or critic_window or critic_window_ih):
            raise ValueError("enc_out_channel and the critic's window configure the equivariant and ablated "
                             "encoders: the ResFiT encoder has no 9x9 map or last conv of its own")

        wrist_keys = tuple(wrist_keys)
        if len(wrist_keys) != n_arms:
            raise ValueError(
                f"expected one wrist camera per arm: n_arms={n_arms}, wrist_keys={wrist_keys}"
            )

        self.encoder = encoder
        self.obs_channel = obs_shape[0]
        self.n_hidden = n_hidden
        self.ih_out_channel = ih_out_channel
        self.N = N
        self.float_images_as_uint8 = float_images_as_uint8
        self.base_action_input = base_action_input

        self.register_buffer(
            "robot_base_xy",
            torch.zeros(2, dtype=torch.float32),
        )

        # ----- Common dims and modules -----
        self.n_arms = n_arms
        self.gripper_dim = gripper_dim
        self.hand_dof = hand_dof
        self.agentview_key = agentview_key
        self.wrist_keys = wrist_keys

        # Per-arm block widths in the raw observation and action vectors.
        self.state_stride  = 7 + gripper_dim   # eef_pos 3, eef_quat 4, gripper_qpos
        self.action_stride = 6 + hand_dof      # delta pose 6, hand

        # state_dim is the raw observation.state width; prop_dim is the encoded
        # width, which is larger because quat(4) is expanded to a 6D rotation.
        self.state_dim  = n_arms * self.state_stride
        self.prop_dim   = n_arms * (9 + gripper_dim)  # pos_xy 2, rot6d 6, pos_z 1, gripper
        self.action_dim = n_arms * self.action_stride
        self._normalizers_ready = False

        self.group = group if group is not None else gspaces.no_base_space(CyclicGroup(N))

        # Representations, one contiguous block per arm
        arm_prop_shape = (
            4 * [self.group.irrep(1)]                        # pos_xy + 3 rotation column pairs
            + (1 + gripper_dim) * [self.group.trivial_repr]  # pos_z, gripper_qpos
        )
        self.prop_shape   = n_arms * arm_prop_shape
        self.action_shape = n_arms * arm_action_shape(self.group, hand_dof)
        # The action field each entry of action_layout belongs to, as arm_key names it
        # (action_rx_ry_0, ...), so a field can be addressed by name.
        arm_action_fields = [*ACTION_FIELDS[:-1], *hand_dof * [ACTION_FIELDS[-1]]]  # as arm_action_shape
        self.action_fields = [arm_key(f, arm) for arm in range(n_arms) for f in arm_action_fields]
        self.prop_type   = enn.FieldType(self.group, self.prop_shape)
        self.action_type = enn.FieldType(self.group, self.action_shape)
        self.action_layout = build_action_layout(self.action_shape, self.group)

        if encoder == "resfit":
            # ResFiT's _build_encoders: a ViT per camera, agentview then the wrists, as rl_cameras
            # orders them, and its augmentation (off_policy/rl/q_agent.py).
            self.camera_keys = (agentview_key, *wrist_keys)
            self.vits = torch.nn.ModuleList([VitEncoder(obs_shape, vit_cfg) for _ in self.camera_keys])
            self.aug = RandomShiftsAug(pad=4)
            self.num_patch = len(self.vits) * self.vits[0].num_patch  # ResFiT's critic reads them as tokens
            self.patch_dim = self.vits[0].patch_repr_dim
            # The tokens, declared invariant (the class docstring); there are no separate wrist features.
            self.vis_type = self.ih_type = None
            self.vis_ih_type = enn.FieldType(self.group, self.num_patch * self.patch_dim * [self.group.trivial_repr])
            self.window_type = None
        else:
            self.crop_randomizer = dmvc.CropRandomizer(
                input_shape=obs_shape,
                crop_height=crop_shape[0],
                crop_width=crop_shape[1],
            )

            # In-hand encoders are scalar, one per arm
            self.enc_ih = torch.nn.ModuleList([
                ResEncoder76InHand(
                    obs_channel=self.obs_channel,
                    channels=tuple(ih_channels),
                    n_out=ih_out_channel,
                    blocks_per_stage=ih_blocks_per_stage,
                )
                for _ in range(n_arms)
            ])

            out_fields = n_hidden if enc_out_channel is None else enc_out_channel
            self.vis_type    = enn.FieldType(self.group, out_fields * [self.group.regular_repr])
            self.ih_type     = enn.FieldType(self.group, n_arms * ih_out_channel * [self.group.trivial_repr])
            self.vis_ih_type = self.vis_type + self.ih_type

            # The ablated encoder is the equivariant one, built as it is, then exported.
            enc_obs = EquivariantResEncoder76Cyclic(
                obs_channel=self.obs_channel,
                n_out=n_hidden,
                N=N,
                initialize=initialize,
                norm=encoder_norm,
                orth_init=encoder_orth_init,
                out_fields=out_fields,
            )
            self.obs_map_end = enc_obs.map_end
            self.enc_obs = enc_obs if encoder == "equivariant" else plain(enc_obs.conv)
            self._build_window(enc_obs, ih_channels[-1], critic_window, critic_window_ih, initialize)

        # The critic's [images | window | proprioception], and the actor's [images | proprioception |
        # base action]; vanilla RL's has no base action, so without a window it is the critic's.
        self.enc_out_type_critic = (self.vis_ih_type + self.prop_type if self.window_type is None
                                    else self.vis_ih_type + self.window_type + self.prop_type)
        self.enc_out_type_actor = self.vis_ih_type + self.prop_type
        if base_action_input:
            self.enc_out_type_actor = self.enc_out_type_actor + self.action_type
        self.enc_out_dim_critic = self.enc_out_type_critic.size
        self.enc_out_dim_actor  = self.enc_out_type_actor.size

    def _build_window(self, enc_obs, ih_map_channels: int, fields: int, channels: int, initialize: bool):
        """The critic's window (the class docstring): window_obs over the agentview encoder's 9x9 map
        and window_ih over each wrist's, None where its width is 0; window_type, their features' type,
        None without either. Exported to plain layers in the ablated family. Built with torch's and
        numpy's random state forked, so it moves no other weight at a seed."""
        self.window_obs = self.window_ih = self.window_type = None
        if not (fields or channels):
            return
        reps = fields * [self.group.regular_repr] + self.n_arms * channels * [self.group.trivial_repr]
        self.window_type = enn.FieldType(self.group, reps)
        np_state = np.random.get_state()
        with torch.random.fork_rng(devices=[]):
            if fields:
                out_type = enn.FieldType(enc_obs.group, fields * [enc_obs.group.regular_repr])
                window_obs = torch.nn.Sequential(
                    enn.R2Conv(enc_obs.map_type, out_type, kernel_size=9, padding=0, bias=False,
                               initialize=initialize),
                    enn.ReLU(out_type),
                )
                self.window_obs = window_obs if self.encoder == "equivariant" else plain(window_obs)
            if channels:
                self.window_ih = torch.nn.ModuleList([
                    torch.nn.Sequential(torch.nn.Conv2d(ih_map_channels, channels, kernel_size=9, bias=False),
                                        torch.nn.ReLU())
                    for _ in range(self.n_arms)
                ])
        np.random.set_state(np_state)

    def get6DRotation(self, quat):
        """robosuite stores eef_quat as (x, y, z, w); reorder to (w, x, y, z).

        By slicing: indexing with a Python list, quat[:, [3, 0, 1, 2]], copies the index to the
        GPU, and that copy waits for all the image-encoder work queued before it.
        """
        return quaternion_to_rotation_6d(torch.cat([quat[:, 3:4], quat[:, 0:3]], dim=1))

    def _image_to_signed_unit(self, img):
        """uint8 [0, 255] or float [0, 1] → float [-1, 1]. See the class docstring.

        Reads the dtype, which is metadata: no host sync."""
        if img.dtype == torch.uint8 or self.float_images_as_uint8:
            img = img.float() / 255.0
        return img.float() * 2.0 - 1.0

    def set_normalizer(
        self,
        normalizer: LinearNormalizer,
        robot_base_xy: torch.Tensor | np.ndarray | list | None = None,
        verbose: bool = False,
    ):
        """Attach the LinearNormalizer; optionally set robot_base_xy, the rotation
        center subtracted from every arm's pos_xy at runtime so the irrep(1)
        normalizer is applied about that center."""
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

        for arm in range(self.n_arms):
            for field in PROP_FIELDS + ACTION_FIELDS:
                name = arm_key(field, arm)
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

        obs         = nobs[self.agentview_key]
        base_action = nobs["observation.base_action"] if self.base_action_input else None
        state       = nobs["observation.state"]

        batch_size = obs.shape[0]

        window = []
        if self.encoder == "resfit":
            images = [self._vit_tokens(nobs)]
        else:
            # ----- Images (uint8 [0,255] or float [0,1] → float [-1,1]) -----
            # In-hand cameras are cropped before agentview; CropRandomizer draws a random
            # offset per call in train mode, so this order is part of the trained behaviour.
            # Each encoder runs in two slices, around its 9x9 map, which compute what it does whole.
            ih_out, ih_maps = [], []
            for key, enc_ih in zip(self.wrist_keys, self.enc_ih):
                ih = self._image_to_signed_unit(nobs[key])
                out, ih_map = enc_ih.forward_map(self.crop_randomizer(ih))
                ih_out.append(out)
                ih_maps.append(ih_map)

            obs     = self._image_to_signed_unit(obs)
            obs     = self.crop_randomizer(obs)
            if self.encoder == "equivariant":
                enc_raw, obs_map = self.enc_obs.forward_map(obs)
            else:
                obs_map = self.enc_obs[:self.obs_map_end](obs)
                enc_raw = self.enc_obs[self.obs_map_end:](obs_map)
            enc_out = (enc_raw.tensor if self.encoder == "equivariant" else enc_raw).reshape(batch_size, -1)
            images  = [enc_out, *ih_out]

            if self.window_obs is not None:
                w = self.window_obs(obs_map)
                window.append((w.tensor if self.encoder == "equivariant" else w).reshape(batch_size, -1))
            if self.window_ih is not None:
                window += [win(ih_map).flatten(1) for win, ih_map in zip(self.window_ih, ih_maps)]

        # ---- Proprioception and action, one contiguous block per arm ----
        prop, action = [], []
        for arm in range(self.n_arms):
            s = state[:, arm * self.state_stride : (arm + 1) * self.state_stride]

            pos_xy_centered = s[:, 0:2] - self.robot_base_xy  # (B, 2), buffer broadcasts
            pos_xy = self.normalizer[arm_key("pos_xy", arm)].normalize(pos_xy_centered)
            pos_z  = self.normalizer[arm_key("pos_z", arm)].normalize(s[:, 2:3])
            ee_rot = self.normalizer[arm_key("ee_rot", arm)].normalize(self.get6DRotation(s[:, 3:7]))
            ee_q   = self.normalizer[arm_key("ee_q", arm)].normalize(s[:, 7:self.state_stride])

            prop += [
                pos_xy,
                ee_rot[:, 0:1], ee_rot[:, 3:4],  # col 0: (R00, R10)
                ee_rot[:, 1:2], ee_rot[:, 4:5],  # col 1: (R01, R11)
                ee_rot[:, 2:3], ee_rot[:, 5:6],  # col 2: (R02, R12)
                pos_z,
                ee_q,
            ]
            if not self.base_action_input:
                continue
            a = base_action[:, arm * self.action_stride : (arm + 1) * self.action_stride]
            action += [
                self.normalizer[arm_key("action_xy", arm)].normalize(a[:, 0:2]),
                self.normalizer[arm_key("action_z", arm)].normalize(a[:, 2:3]),
                self.normalizer[arm_key("action_rx_ry", arm)].normalize(a[:, 3:5]),
                self.normalizer[arm_key("action_rz", arm)].normalize(a[:, 5:6]),
                self.normalizer[arm_key("action_hand", arm)].normalize(a[:, 6:self.action_stride]),
            ]

        # ----- Concatenate feature vectors; vanilla RL's actor, without a window, has the critic's -----
        critic_features = torch.cat([*images, *window, *prop], dim=1)
        actor_features  = torch.cat([*images, *prop, *action], dim=1) if action or window else critic_features

        # Wrap in GeometricTensor for the equivariant family
        if self.encoder == "equivariant":
            critic_features = enn.GeometricTensor(critic_features, self.enc_out_type_critic)
            actor_features  = enn.GeometricTensor(actor_features,  self.enc_out_type_actor)

        return actor_features, critic_features

    def _vit_tokens(self, nobs) -> torch.Tensor:
        """Every camera's ViT tokens, [B, cameras * 81 * 128]: ResFiT's _encode, and in it
        VitEncoder.forward less its two torch.cuda.synchronize() and its `obs.max() > 5` rescale,
        each a host sync. The images reach that rescale in [0, 1], so it never applies, and the ViT
        reads them less 0.5, as there. ResFiT's update encodes with its random shifts, and its act()
        without them: here train mode and eval mode."""
        tokens = []
        for key, vit in zip(self.camera_keys, self.vits):
            img = nobs[key]
            img = img.float().div_(255.0) if img.dtype == torch.uint8 else img.float()
            if self.training:
                img = self.aug(img)
            tokens.append(vit.vit(img - 0.5))  # [B, 81, 128]
        # Along the patches, as ResFiT's _encode joins its cameras; its actor flattens them so.
        return torch.cat(tokens, dim=1).flatten(1)
