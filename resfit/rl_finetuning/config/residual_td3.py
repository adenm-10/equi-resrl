# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore

from resfit.rl_finetuning.config.rlpd import ActorConfig, QAgentConfig, RLPDAlgoConfig, RLPDDexmgConfig


@dataclass
class OfflineDataConfig:
    name: str = "ankile/robomimic-mh-can-image"
    num_episodes: int | None = 100
    # Offline data action labeling options
    use_base_policy_for_base_actions: bool = True
    # Normalization safeguards
    min_action_range: float = 1e-1  # Minimum range for any action dimension to prevent normalization blow-up
    min_state_std: float = 1e-1  # Minimum std for any state dimension to prevent normalization blow-up


@dataclass
class WandBConfig:
    project: str = "robomimic-can-residual-td3"
    mode: str = "online"
    entity: str | None = None
    notes: str | None = None
    continue_run_id: str | None = None
    name: str | None = None
    group: str | None = None


@dataclass
class BasePolicyConfig:
    wandb_id: str = "TODO"
    wt_type: str = "best"
    wt_version: str = "latest"


@dataclass
class EquivarianceConfig:
    use_equivariant_model: bool = True

    N: int = 8
    enc_degree_channel: int = 32 # power of 8 for agent view encoder, ideally 128
    # Width of each wrist camera's feature vector (scalar channels). Until
    # 2026-09-29 it was tied to enc_degree_channel; 32 matches the default.
    ih_out_channel: int = 32
    initialize: bool = True

    # Per-arm observation layout. State is [eef_pos 3, eef_quat 4, gripper_qpos G]
    # and action is [delta_pose 6, hand H] per arm; one wrist camera per arm.
    n_arms: int = 1
    gripper_dim: int = 2
    hand_dof: int = 1
    # Read only by the scalar ablation; the equivariant networks use the three
    # fields below.
    use_norms: bool = True
    # Delta-orthonormal init in the actor; escnn's default, which the critic and
    # encoder keep unless their fields below say otherwise, is He.
    use_orth_init: bool = True

    # Normalization after each hidden Linear of the actor and critic: "none",
    # "iid_bn" (escnn IIDBatchNorm1d), "layer_norm" (RegularLayerNorm, which
    # normalizes by all channels at once, as ResFiT's LayerNorm does) or
    # "field_norm" (FieldNorm, each field by its own statistics). The encoder
    # takes "none" or "inner_bn" (escnn InnerBatchNorm after every conv).
    # critic_norm and encoder_norm default to what every run before 2026-09-29
    # had. actor_norm defaults to "none" from 2026-09-30: every run before then
    # had "iid_bn", no run ever tested it against "none", and the SO(2)
    # equivariant RL code this project follows has no normalization anywhere.
    # EXPERIMENTS.md, 2026-09-30 CODE.
    actor_norm: str = "none"
    critic_norm: str = "none"
    encoder_norm: str = "none"
    critic_orth_init: bool = False
    encoder_orth_init: bool = False
    # Scales each critic head's final Linear at init; None keeps escnn's init.
    critic_last_layer_init_scale: float | None = None
    # False skips the actor's and critic's visual projection: the trunk takes
    # the encoder's features directly.
    use_visual_proj: bool = True
    # True bounds the actor's mean by action_scale, as ResFiT's Tanh does:
    # equi_tanh on each irrep(1) pair's norm and each trivial component.
    residual_squash: bool = False

    # True reproduces every equivariant run before 2026-09-29: the target critic
    # sat in eval mode, where escnn layers compute with the weights expanded at
    # construction, so it never saw its soft updates. EXPERIMENTS.md, 2026-09-29
    # CORRECTION.
    frozen_target_critic: bool = False

    # True reproduces every equivariant and scalar-ablation run before
    # 2026-10-02: the env's float images in [0, 1] were divided by 255 as if they
    # were uint8, so act() saw near-blank images while the update saw the
    # buffer's uint8 copies at full contrast. EXPERIMENTS.md, 2026-10-02
    # CORRECTION.
    float_images_as_uint8: bool = False

    num_actor_layers: int = 2
    actor_degree_channel: int = 128
    
    # Hidden layers in each critic head, before GroupPooling. Logged as 2 until
    # 2026-09-29, while the field was dead and every critic had 1.
    num_critic_layers: int = 1
    critic_degree_channel: int = 128



@dataclass
class ResidualTD3AlgoConfig(RLPDAlgoConfig):
    # ------------------------------------------------------------------
    # Critic warmup phase ----------------------------------------------
    # ------------------------------------------------------------------
    # Number of critic-only updates before training the actor
    critic_warmup_steps: int = 10_000

    # ------------------------------------------------------------------
    # Random action exploration -----------------------------------------
    # ------------------------------------------------------------------
    # Scale for random action noise during initial exploration phase
    # Actions are sampled as: rand_actions = torch.rand(...) * 2 * random_action_noise_scale - random_action_noise_scale
    random_action_noise_scale: float = 0.2  # Default: uniform in [-1, 1]

    # Whether to use base policy + noise (True) or pure uniform noise (False) during warmup
    # Note: Environment wrapper always applies base_action + residual_action
    # True: residual_action = noise (resulting in base_action + noise)
    # False: residual_action = pure_random - base_action (resulting in pure_random)
    use_base_policy_for_warmup: bool = True

    # ------------------------------------------------------------------
    # Standard deviation schedule -------------------------------------------
    # ------------------------------------------------------------------
    stddev_max: float = 0.05
    stddev_min: float = 0.05
    stddev_step: int = 300_000

    # Progressive clipping schedule for the residual actions
    # I.e., starts clipping linearly from 0 to action scale over progressive_clipping_steps steps
    progressive_clipping_steps: int = 0


# -----------------------------------------------------------------------------
# Top-level experiment config --------------------------------------------------
# -----------------------------------------------------------------------------
@dataclass
class ResidualTD3DexmgConfig(RLPDDexmgConfig):
    actor_name: str | None = None  # Inferred from base policy config

    # ------------------------------------------------------------------
    # Algorithm & optimisation
    # ------------------------------------------------------------------
    algo: ResidualTD3AlgoConfig = field(default_factory=ResidualTD3AlgoConfig)

    # ------------------------------------------------------------------
    # Network architectures
    # ------------------------------------------------------------------
    agent: QAgentConfig = field(
        default_factory=lambda: QAgentConfig(
            actor_lr=1e-6, # default
            # actor_lr=1e-5,
            critic_lr=1e-4,
            critic_target_tau=0.005,
            actor=ActorConfig(
                action_scale=0.1,
                actor_last_layer_init_scale=0.0,  # imp for residual
                # actor_last_layer_init_scale=1e-4,  # imp for residual
            ),
        )
    )

    # ------------------------------------------------------------------
    # Offline dataset
    # ------------------------------------------------------------------
    offline_data: OfflineDataConfig | None = field(default_factory=OfflineDataConfig)

    # ------------------------------------------------------------------
    # Base policy
    # ------------------------------------------------------------------
    base_policy: BasePolicyConfig = field(default_factory=BasePolicyConfig)

    # ------------------------------------------------------------------
    # Weights & Biases logging
    # ------------------------------------------------------------------
    wandb: WandBConfig = field(default_factory=WandBConfig)

    # ------------------------------------------------------------------
    # Logging / checkpointing
    # ------------------------------------------------------------------
    eval_interval_every_steps: int = 10_000

    # Whether to run an evaluation pass before training begins (at step 0)
    eval_first: bool = True

    # False keeps both replay buffers in memory only. Existing caches are still read.
    write_buffer_caches: bool = True

    # Equivariant agent or not
    equivariance: EquivarianceConfig | None = None


@dataclass
class ResidualTD3CanConfig(ResidualTD3DexmgConfig):
    task: str = "Can"

    offline_data: OfflineDataConfig = field(
        default_factory=lambda: OfflineDataConfig(
            name="ankile/robomimic-mh-can-image",
            num_episodes=300,
        )
    )

    base_policy: BasePolicyConfig = field(
        default_factory=lambda: BasePolicyConfig(
            wandb_id="robomimic-can-bc/xhjdl8a7",
        )
    )

    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="robomimic-can-residual-td3"))


@dataclass
class ResidualTD3SquareConfig(ResidualTD3DexmgConfig):
    task: str = "Square"

    offline_data: OfflineDataConfig = field(
        default_factory=lambda: OfflineDataConfig(
            name="ankile/robomimic-mh-square-image",
            num_episodes=300,
        )
    )

    base_policy: BasePolicyConfig = field(
        default_factory=lambda: BasePolicyConfig(
            wandb_id="robomimic-square-bc/3hzs5bz1",
        )
    )

    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="robomimic-square-residual-td3"))


@dataclass
class ResidualEquiTD3CanConfig(ResidualTD3DexmgConfig):
    task: str = "Can"

    offline_data: OfflineDataConfig = field(
        default_factory=lambda: OfflineDataConfig(
            name="ankile/robomimic-mh-can-image",
            num_episodes=300,
        )
    )

    base_policy: BasePolicyConfig = field(
        default_factory=lambda: BasePolicyConfig(
            wandb_id="robomimic-can-bc/xhjdl8a7",
        )
    )

    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="robomimic-can-residual-td3"))

    equivariance: EquivarianceConfig = field(default_factory=lambda: EquivarianceConfig(N=8))


@dataclass
class ResidualEquiTD3SquareConfig(ResidualTD3DexmgConfig):
    task: str = "Square"

    offline_data: OfflineDataConfig = field(
        default_factory=lambda: OfflineDataConfig(
            name="ankile/robomimic-mh-square-image",
            num_episodes=300,
        )
    )

    base_policy: BasePolicyConfig = field(
        default_factory=lambda: BasePolicyConfig(
            wandb_id="robomimic-square-bc/3hzs5bz1",
        )
    )

    # Every Square equivariant run goes to one project and one group, whatever the experiment;
    # each run's own name says what it is. The group name is the one the Can reproduction
    # established. Before 2026-09-30 submit.sh set these per experiment instead.
    wandb: WandBConfig = field(
        default_factory=lambda: WandBConfig(project="robomimic-square-final", group="equi-residual-rl")
    )

    equivariance: EquivarianceConfig = field(default_factory=lambda: EquivarianceConfig(N=8))


@dataclass
class ResidualTD3BoxCleanConfig(ResidualTD3DexmgConfig):
    task: str = "TwoArmBoxCleanup"

    rl_camera: list[str] = field(
        default_factory=lambda: [
            "observation.images.agentview",
            "observation.images.robot0_eye_in_hand",
            "observation.images.robot1_eye_in_hand",
        ]
    )

    algo: ResidualTD3AlgoConfig = field(
        default_factory=lambda: ResidualTD3AlgoConfig(
            total_timesteps=500_000,
        )
    )

    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="dexmg-box-clean-residual-td3"))

    offline_data: OfflineDataConfig = field(
        default_factory=lambda: OfflineDataConfig(
            name="ankile/dexmg-two-arm-box-cleanup",
            num_episodes=1_000,
        )
    )
    # ACT, chunk 20, step 24k of 50k (eval 0.74, an outlier draw; the policy's
    # rate is ~0.53-0.61). Pinned to the version, not `latest`, so the config
    # names the weights. Backed up in ~/equi-resrl-preserve/run_xg0w2r0z_best:v7.
    base_policy: BasePolicyConfig = field(
        default_factory=lambda: BasePolicyConfig(
            wandb_id="dexmg-boxcleanup-bc/xg0w2r0z",
            wt_type="best",
            wt_version="v7",
        )
    )


@dataclass
class ResidualEquiTD3BoxCleanConfig(ResidualTD3BoxCleanConfig):
    """Equivariant residual TD3 on the two-arm BoxCleanup task.

    C8 about the midpoint of the two robot bases. The task holds object yaw
    fixed, so this is a geometric prior, not an exact task symmetry — see
    docs/EQUIVARIANCE.md."""

    wandb: WandBConfig = field(
        default_factory=lambda: WandBConfig(project="dexmg-box-clean-equi-residual-td3")
    )

    equivariance: EquivarianceConfig = field(
        default_factory=lambda: EquivarianceConfig(
            N=8, n_arms=2, gripper_dim=12, hand_dof=6,
        )
    )


@dataclass
class ResidualTD3CoffeeConfig(ResidualTD3BoxCleanConfig):
    task: str = "TwoArmCoffee"

    rl_camera: list[str] = field(
        default_factory=lambda: [
            "observation.images.agentview",
            "observation.images.robot0_eye_in_left_hand",
            "observation.images.robot0_eye_in_right_hand",
        ]
    )

    algo: ResidualTD3AlgoConfig = field(
        default_factory=lambda: ResidualTD3AlgoConfig(
            total_timesteps=500_000,
        )
    )

    wandb: WandBConfig = field(
        default_factory=lambda: WandBConfig(project="dexmg-coffee-residual-td3", notes="all cameras")
    )

    offline_data: OfflineDataConfig = field(
        default_factory=lambda: OfflineDataConfig(
            name="ankile/dexmg-two-arm-coffee",
            num_episodes=1_000,
        )
    )
    base_policy: BasePolicyConfig = field(
        default_factory=lambda: BasePolicyConfig(
            # wandb_id="dexmg-bc/6qgo2g5k",
            wandb_id="dexmg-bc/tjsr8xad",
            wt_type="best",
            wt_version="latest",
        )
    )

@dataclass
class ResidualTD3TwoArmCanSortConfig(ResidualTD3BoxCleanConfig):
    task: str = "TwoArmCanSortRandom"

    rl_camera: list[str] = field(
        default_factory=lambda: [
            "observation.images.frontview",
            "observation.images.robot0_eye_in_left_hand",
            "observation.images.robot0_eye_in_right_hand",
        ]
    )

    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="dexmg-cansort-residual-td3"))

    offline_data: OfflineDataConfig = field(
        default_factory=lambda: OfflineDataConfig(
            name="ankile/dexmg-two-arm-can-sort-random",
            num_episodes=1_000,
        )
    )
    base_policy: BasePolicyConfig = field(
        default_factory=lambda: BasePolicyConfig(
            wandb_id="TODO",
            wt_type="best",
            wt_version="latest",
        )
    )


# -----------------------------------------------------------------------------
# Register with Hydra
# -----------------------------------------------------------------------------
cs = ConfigStore.instance()
cs.store(name="residual_td3_dexmg_config", node=ResidualTD3DexmgConfig)
cs.store(name="residual_td3_can_config", node=ResidualTD3CanConfig)
cs.store(name="residual_td3_square_config", node=ResidualTD3SquareConfig)
cs.store(name="residual_td3_box_clean_config", node=ResidualTD3BoxCleanConfig)
cs.store(name="residual_td3_coffee_config", node=ResidualTD3CoffeeConfig)
cs.store(name="residual_td3_two_arm_cansort_config", node=ResidualTD3TwoArmCanSortConfig)

cs.store(name="residual_equi_td3_can_config",    node=ResidualEquiTD3CanConfig)
cs.store(name="residual_equi_td3_square_config", node=ResidualEquiTD3SquareConfig)
cs.store(name="residual_equi_td3_box_clean_config", node=ResidualEquiTD3BoxCleanConfig)
