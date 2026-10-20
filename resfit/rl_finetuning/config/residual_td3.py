# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore

from resfit.rl_finetuning.config.rlpd import (
    ActorConfig,
    CriticConfig,
    QAgentConfig,
    RLPDAlgoConfig,
    RLPDDexmgConfig,
)


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
    # The family of each of the agent's three networks (equi_off_policy/rl/q_agent.py), one of:
    #   "equivariant": the escnn network, as in every run before these fields existed;
    #   "ablated": that network built exactly as the equivariant one, then exported to plain torch
    #     layers (rl/ablated.py), so it starts as the same function at the same seed and only its
    #     updates are unconstrained;
    #   "resfit": ResFiT's own ViT, actor or critic (off_policy/), wrapped to read this agent's
    #     features and to bound its actions as this agent does (rl/resfit_heads.py).
    # The agent refuses a mix it cannot build, and any field below that no network of the mix
    # reads (q_agent.py, check_families). They replace use_equivariant_model, which picked the
    # scalar ablation, and use_norms, which only that read. EXPERIMENTS.md, 2026-10-20 CODE.
    encoder_family: str = "equivariant"
    actor_family: str = "equivariant"
    critic_family: str = "equivariant"

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
    # Normalization of each critic head's pooled features, after GroupPooling and before its
    # output Linear: "none", or "layer_norm" (torch's LayerNorm; the features are invariant,
    # so any norm is allowed there). It bounds Q whatever the input, as RLPD's LayerNorm
    # before its critic's last layer does. EXPERIMENTS.md, 2026-10-10 CODE.
    critic_pooled_norm: str = "none"
    critic_orth_init: bool = False
    encoder_orth_init: bool = False
    # Scales each critic head's final Linear at init; None keeps escnn's init.
    critic_last_layer_init_scale: float | None = None
    # False skips the actor's and critic's visual projection: the trunk takes
    # the encoder's features directly.
    use_visual_proj: bool = True
    # True bounds the actor's mean by action_scale, as ResFiT's Tanh does:
    # equi_tanh on each irrep(1) pair's norm and each trivial component. In
    # vanilla RL (base_policy=None) that mean is the whole action.
    residual_squash: bool = False
    # The residual's bound per action field, in env units, keyed by ACTION_FIELDS name
    # (equi_normalizer.py) and applied to every arm alike, e.g. {action_rx_ry: 0.0}. A field not
    # named keeps agent.actor.action_scale, so the empty default is every run before 2026-10-12.
    # Each field gets one number, so an irrep(1) pair is scaled as a whole and the actor stays
    # equivariant; a bound per component of a pair would not be (EQUIVARIANCE.md). It bounds the
    # actor's mean; exploration and target noise are added after it, unchanged. Residual runs only.
    # Set a field with +equivariance.residual_field_bound.<field>=<bound>: the + adds a key to the
    # mapping. EXPERIMENTS.md, 2026-10-12 PRE-REGISTERED, "the equivariant residual on Square with
    # no tilt".
    residual_field_bound: dict[str, float] = field(default_factory=dict)

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
    # None runs vanilla RL: no base policy, and the actor outputs the whole
    # action (TunedRLPDConfig). train_residual_td3.py branches on it as
    # residual_run.
    base_policy: BasePolicyConfig | None = field(default_factory=BasePolicyConfig)

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
    # False fills both replay buffers in the run itself and never loads a cache, from disk or the
    # Hub. A key does not record everything that filled its cache: not the normalization path
    # (TODO B7, S9), and before 2026-10-07 not mujoco, whose 3.8.1 rendered other frames. Needs
    # write_buffer_caches=false, or the run would write over the cache it did not read.
    read_buffer_caches: bool = True

    # True passes every camera frame the env returns through the demos' video codec, so rollouts,
    # evals and the online buffer see env frames coded as the demo frames are
    # (resfit/dexmg/environments/demo_codec.py). False, as in every run before 2026-10-08, gives
    # the renderer's frames as they are.
    encode_env_frames_like_demos: bool = False

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
# Tuned RLPD: ResFiT's off-policy RL with no base policy ----------------------
# -----------------------------------------------------------------------------
# The fields only a residual run reads, at the values that turn them off. A vanilla run
# (base_policy=None) must leave each one off: the trainer would ignore it, and the run's
# recorded config would describe a base policy it never had (STANDARDS 1.2).
RESIDUAL_ONLY_FIELDS_OFF = {
    "offline_data.use_base_policy_for_base_actions": False,  # labels the demos with base actions
    "algo.use_base_policy_for_warmup": False,                # warm-up acts as base policy + noise
    "algo.progressive_clipping_steps": 0,                    # ramps the residual in from zero
}


def residual_only_fields_set(cfg) -> list[str]:
    """The residual-only fields `cfg` turns on, as "path=value". train_residual_td3.py refuses
    a vanilla run (base_policy=None) unless this is empty."""
    on = []
    for path, off in RESIDUAL_ONLY_FIELDS_OFF.items():
        value = cfg
        for name in path.split("."):
            value = getattr(value, name)
        if value != off:
            on.append(f"{path}={value}")
    return on


def _vanilla_agent(num_q: int) -> QAgentConfig:
    """The agent half of the vanilla column: upstream's RLPD agent, written out."""
    return QAgentConfig(
        actor_lr=1e-4,
        critic_lr=1e-4,
        critic_target_tau=0.01,
        critic=CriticConfig(num_q=num_q),
        actor=ActorConfig(action_scale=1.0, actor_last_layer_init_scale=None),
    )


@dataclass
class TunedRLPDConfig(ResidualTD3DexmgConfig):
    """The ResFiT paper's "Tuned RLPD": ResFiT's off-policy RL, with no base policy.

    A residual config made vanilla the way the paper made ResFiT vanilla. Every field in which
    upstream's residual config and its RLPD config (config/rlpd.py) differ takes the RLPD
    value; every other field stays as the residual config has it. The actor then learns the
    whole action from random exploration and the demos, as in upstream's train_rlpd_dexmg.py,
    whose training steps this trainer runs (docs/ARCHITECTURE.md, "Residual or vanilla").

    Ten critic heads are ResFiT's design as the paper ran it (EXPERIMENTS.md, 2026-09-29
    CORRECTION). The equivariant subclasses keep their residual reference's two.
    """

    base_policy: BasePolicyConfig | None = None

    algo: ResidualTD3AlgoConfig = field(
        default_factory=lambda: ResidualTD3AlgoConfig(
            critic_warmup_steps=0,
            random_action_noise_scale=1.0,  # warm-up actions U(-1, 1)
            use_base_policy_for_warmup=False,
            stddev_max=0.1,
            stddev_min=0.1,
        )
    )

    agent: QAgentConfig = field(default_factory=lambda: _vanilla_agent(num_q=10))

    # Each run builds its buffers in memory: a vanilla buffer holds no base action, and the
    # cache key does not record which normalization path filled a buffer (TODO B7).
    write_buffer_caches: bool = False


@dataclass
class TunedRLPDSquareConfig(TunedRLPDConfig):
    task: str = "Square"

    offline_data: OfflineDataConfig = field(
        default_factory=lambda: OfflineDataConfig(
            name="ankile/robomimic-mh-square-image",
            num_episodes=300,
            use_base_policy_for_base_actions=False,
        )
    )

    wandb: WandBConfig = field(
        default_factory=lambda: WandBConfig(project="robomimic-square-final", group="tuned-rlpd")
    )


@dataclass
class TunedRLPDCanConfig(TunedRLPDConfig):
    task: str = "Can"

    offline_data: OfflineDataConfig = field(
        default_factory=lambda: OfflineDataConfig(
            name="ankile/robomimic-mh-can-image",
            num_episodes=300,
            use_base_policy_for_base_actions=False,
        )
    )

    wandb: WandBConfig = field(
        default_factory=lambda: WandBConfig(project="robomimic-can-final", group="tuned-rlpd")
    )


@dataclass
class EquiTunedRLPDSquareConfig(TunedRLPDSquareConfig):
    """Tuned RLPD on the equivariant stack. These three fields are its whole difference from
    the ResFiT arm's config; tests/test_sweep_arms.py checks that for the arms as launched."""

    agent: QAgentConfig = field(default_factory=lambda: _vanilla_agent(num_q=2))

    wandb: WandBConfig = field(
        default_factory=lambda: WandBConfig(project="robomimic-square-final", group="equi-tuned-rlpd")
    )

    equivariance: EquivarianceConfig = field(default_factory=lambda: EquivarianceConfig(N=8))


@dataclass
class EquiTunedRLPDCanConfig(TunedRLPDCanConfig):
    """As EquiTunedRLPDSquareConfig, on Can."""

    agent: QAgentConfig = field(default_factory=lambda: _vanilla_agent(num_q=2))

    wandb: WandBConfig = field(
        default_factory=lambda: WandBConfig(project="robomimic-can-final", group="equi-tuned-rlpd")
    )

    equivariance: EquivarianceConfig = field(default_factory=lambda: EquivarianceConfig(N=8))


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

cs.store(name="tuned_rlpd_square_config", node=TunedRLPDSquareConfig)
cs.store(name="tuned_rlpd_can_config", node=TunedRLPDCanConfig)
cs.store(name="equi_tuned_rlpd_square_config", node=EquiTunedRLPDSquareConfig)
cs.store(name="equi_tuned_rlpd_can_config", node=EquiTunedRLPDCanConfig)
