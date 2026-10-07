"""Tier 1: the Hydra configs resolve, and the reproduction config is what we think.

No GPU, seconds. Guards against launching a 31-hour run against a config that
silently differs from what the experiment log claims.
"""

from __future__ import annotations

import pytest
from hydra.core.config_store import ConfigStore

import resfit.rl_finetuning.config.residual_td3 as rtd3

# Registered names that must exist. Mirrors the ConfigStore block at the bottom
# of config/residual_td3.py.
EXPECTED_NAMES = {
    "residual_td3_dexmg_config",
    "residual_td3_can_config",
    "residual_td3_square_config",
    "residual_td3_box_clean_config",
    "residual_td3_coffee_config",
    "residual_td3_two_arm_cansort_config",
    "residual_equi_td3_can_config",
    "residual_equi_td3_square_config",
    "residual_equi_td3_box_clean_config",
    "tuned_rlpd_square_config",
    "tuned_rlpd_can_config",
    "equi_tuned_rlpd_square_config",
    "equi_tuned_rlpd_can_config",
}
# Vanilla RL, no base policy (TunedRLPDConfig). Every other name is a residual config.
TUNED_RLPD_NAMES = {n for n in EXPECTED_NAMES if "tuned_rlpd" in n}
RESIDUAL_NAMES = EXPECTED_NAMES - TUNED_RLPD_NAMES

# The reproduction target. Values pinned here are the ones the experiment log
# records; a change to any of them invalidates comparability with prior runs
# (STANDARDS.md rule 1.6) and should fail this test until the log is updated.
REPRO_CONFIG = "residual_equi_td3_can_config"


def _node(name):
    repo = ConfigStore.instance().repo
    key = f"{name}.yaml"
    assert key in repo, f"{name} not registered in the ConfigStore"
    return repo[key].node


# The configs that build the equivariant agent, residual or vanilla.
EQUIVARIANT_NAMES = {n for n in EXPECTED_NAMES if _node(n).equivariance is not None}


def test_all_expected_configs_registered():
    repo = ConfigStore.instance().repo
    missing = {n for n in EXPECTED_NAMES if f"{n}.yaml" not in repo}
    assert not missing, f"unregistered configs: {sorted(missing)}"


@pytest.mark.parametrize("name", sorted(EXPECTED_NAMES))
def test_config_instantiates(name):
    node = _node(name)
    assert node is not None


def test_repro_config_is_equivariant_can():
    cfg = _node(REPRO_CONFIG)
    assert cfg.task == "Can"
    assert cfg.equivariance is not None, (
        "equivariance block is None, so train_residual_td3.py would build the "
        "BASELINE agent, not the equivariant one"
    )
    assert cfg.equivariance.use_equivariant_model is True
    assert cfg.equivariance.N == 8


def test_repro_config_pinned_hyperparameters():
    """The values the experiment log records for the reproduction."""
    cfg = _node(REPRO_CONFIG)
    assert cfg.offline_data.name == "ankile/robomimic-mh-can-image"
    assert cfg.offline_data.num_episodes == 300
    assert cfg.base_policy.wandb_id == "robomimic-can-bc/xhjdl8a7"
    # n_step and gamma are part of the replay-buffer cache key. Changing either
    # invalidates the 5 GB offline and 16 GB online Can caches.
    assert cfg.algo.n_step == 3
    assert cfg.algo.gamma == 0.99
    assert cfg.algo.buffer_size == 200_000
    assert cfg.algo.learning_starts == 10_000
    assert cfg.algo.critic_warmup_steps == 10_000
    assert cfg.algo.total_timesteps == 300_000
    assert cfg.algo.num_updates_per_iteration == 4
    assert cfg.algo.batch_size == 256
    assert cfg.algo.offline_fraction == 0.5
    assert cfg.algo.random_action_noise_scale == 0.2
    assert cfg.agent.actor.action_scale == 0.1     # NOT overridden; the reference run used 0.1
    assert cfg.agent.actor_lr == 1e-6
    assert cfg.agent.critic_lr == 1e-4
    assert cfg.algo.stddev_max == 0.05   # reference run used the default, not 0.025
    assert cfg.algo.stddev_min == 0.05
    # 16 at 7dae4925, raised to 32 in caf83f3. submit.sh does not override it,
    # so the reproduction runs at 32. See the assumption note in EXPERIMENTS.md.
    assert cfg.equivariance.enc_degree_channel == 32
    assert cfg.equivariance.actor_degree_channel == 128
    assert cfg.equivariance.critic_degree_channel == 128


def test_residual_starts_at_exactly_zero():
    """The step-0 evaluation must measure the base BC policy alone.

    EXPERIMENTS.md reads every run's step-0 number as the base policy's score.
    That is only true while the residual's final layer is initialised to zero.
    """
    cfg = _node(REPRO_CONFIG)
    assert cfg.agent.actor.actor_last_layer_init_scale == 0.0
    assert cfg.eval_first is True


def test_equivariant_configs_declare_their_arm_layout():
    """n_arms / gripper_dim / hand_dof drive every slice in ResObsEnc.

    Wrong values do not crash, they mis-slice state and action into the wrong
    representations. These are measured from the envs, so pin them.
    """
    single = {"n_arms": 1, "gripper_dim": 2, "hand_dof": 1}
    two = {"n_arms": 2, "gripper_dim": 12, "hand_dof": 6}
    for name, expected, cameras in (
        ("residual_equi_td3_can_config", single, 2),
        ("residual_equi_td3_square_config", single, 2),
        ("residual_equi_td3_box_clean_config", two, 3),
    ):
        cfg = _node(name)
        for field, value in expected.items():
            assert getattr(cfg.equivariance, field) == value, f"{name}.{field}"
        assert len(cfg.rl_camera) == cameras, (
            f"{name} needs agentview plus one wrist camera per arm"
        )


def test_boxcleanup_base_policy_is_pinned():
    """Both BoxCleanup configs load the backed-up ACT checkpoint, by version.

    `latest` would let a new upload change the base policy under an unchanged
    config. preflight checks the same run, so the two must agree.
    """
    import preflight

    for name in ("residual_td3_box_clean_config", "residual_equi_td3_box_clean_config"):
        base = _node(name).base_policy
        assert base.wandb_id == preflight.TASKS["TwoArmBoxCleanup"]["bc"], name
        assert (base.wt_type, base.wt_version) == ("best", "v7"), name


# Buffer cache keys measured on boce-WS-01. The torchrl and tensordict versions are part of both
# keys, and mujoco's of the online key, so these are stack-specific by design.
CACHE_STACK = ("0.7.0", "0.7.0", "3.3.2")


def _cache_stack() -> tuple[str, str, str]:
    from importlib.metadata import version

    import tensordict
    import torchrl

    return torchrl.__version__, tensordict.__version__, version("mujoco")


# The online keys moved on 2026-10-07, on purpose: they record mujoco, since the online buffer
# holds rendered frames. Before, at mujoco 3.8.1's frames, they were 43637c10 (Can), 66c46c51
# (Square) and 5709219c (TwoArmBoxCleanup). The offline keys did not move.
CACHE_HASHES = {
    "Can": ("8edf618e", "48638eb8"),
    "Square": ("1d31ed9b", "1683475f"),
    "TwoArmBoxCleanup": ("514502d0", "dca5e179"),
}


def test_buffer_cache_keys_are_stable():
    """Refactoring preflight.TASKS must not move an existing cache key.

    Square's 6.4 GB offline buffer is addressed by its hash; moving it orphans the cache and forces
    a rebuild. (Square's old online buffer, 66c46c51, holds frames rendered by mujoco 3.8.1, and
    the key that recorded mujoco orphaned it on purpose.)
    """
    import preflight

    if _cache_stack() != CACHE_STACK:
        pytest.skip(f"cache keys pinned to torchrl/tensordict/mujoco {CACHE_STACK}")

    for task, expected in CACHE_HASHES.items():
        assert preflight.cache_hashes(task) == expected, f"{task} cache key moved"


def test_dead_config_fields_are_documented():
    """Fields that exist but do nothing must stay listed in the provenance table.

    If someone wires one up, this test fails and points them at the doc to
    update -- which keeps ARCHITECTURE.md honest. See STANDARDS.md rule 1.2.
    """
    import pathlib

    doc = (pathlib.Path(__file__).resolve().parent.parent
           / "docs" / "ARCHITECTURE.md").read_text()
    for field in ("num_actor_layers", "num_critic_layers", "use_norms", "use_orth_init"):
        assert field in doc, f"{field} missing from the provenance table"
    assert "DEAD" in doc, "provenance table no longer marks any field DEAD"


@pytest.mark.parametrize("name", sorted(RESIDUAL_NAMES))
def test_every_residual_config_writes_caches_by_default(name):
    """write_buffer_caches is new; True reproduces every earlier run."""
    assert _node(name).write_buffer_caches is True


@pytest.mark.parametrize("name", sorted(EQUIVARIANT_NAMES))
def test_target_critic_is_not_frozen_by_default(name):
    """frozen_target_critic=True reproduces the bug in every equivariant run
    before 2026-09-29. It exists for comparison runs only."""
    assert _node(name).equivariance.frozen_target_critic is False


@pytest.mark.parametrize("name", sorted(EQUIVARIANT_NAMES))
def test_act_time_images_are_not_scaled_as_uint8_by_default(name):
    """float_images_as_uint8=True reproduces the bug in every equivariant and
    scalar-ablation run before 2026-10-02. It exists for comparison runs only."""
    assert _node(name).equivariance.float_images_as_uint8 is False


@pytest.mark.parametrize("name", sorted(EQUIVARIANT_NAMES))
def test_equivariant_defaults_build_the_reference_network(name):
    """Every field added for the 2026-09-29 sweep defaults to what earlier runs had,
    except actor_norm.

    num_critic_layers was dead and logged as 2; every critic had 1 hidden layer
    per head, which is now its wired default. actor_norm was "iid_bn" in every
    run before 2026-09-30 and defaults to "none" from then on (EXPERIMENTS.md,
    2026-09-30 CODE).
    """
    eq = _node(name).equivariance
    assert (eq.actor_norm, eq.critic_norm, eq.encoder_norm) == ("none", "none", "none")
    assert eq.critic_orth_init is False and eq.encoder_orth_init is False
    assert eq.critic_last_layer_init_scale is None
    assert eq.use_visual_proj is True and eq.residual_squash is False
    assert eq.num_critic_layers == 1
    assert eq.ih_out_channel == eq.enc_degree_channel == 32


# ---------------------------------------------------------------------------
# Tuned RLPD: vanilla RL, no base policy (TunedRLPDConfig)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(TUNED_RLPD_NAMES))
def test_tuned_rlpd_is_the_residual_design_made_vanilla(name):
    """Every field in which upstream ResFiT's residual and RLPD configs differ takes its RLPD
    value; every field the two share keeps it (docs/plans/2026-10-02-tuned-rlpd.md, rule 1)."""
    cfg = _node(name)
    assert cfg.base_policy is None
    assert rtd3.residual_only_fields_set(cfg) == []
    # Made vanilla
    assert cfg.agent.actor_lr == 1e-4
    assert cfg.agent.critic_target_tau == 0.01
    assert cfg.agent.actor.action_scale == 1.0
    assert cfg.agent.actor.actor_last_layer_init_scale is None
    assert cfg.algo.critic_warmup_steps == 0
    assert cfg.algo.random_action_noise_scale == 1.0
    assert (cfg.algo.stddev_max, cfg.algo.stddev_min) == (0.1, 0.1)
    # Computed when the dataclass is built, so a CLI override of stddev_max would not reach it.
    assert cfg.algo.stddev_schedule == "linear(0.1,0.1,300000)"
    assert cfg.offline_data.use_base_policy_for_base_actions is False
    assert cfg.write_buffer_caches is False
    # Held fixed, as both of ResFiT's configs have them
    assert cfg.agent.critic_lr == 1e-4
    assert (cfg.agent.critic_grad_clip_norm, cfg.agent.actor_grad_clip_norm) == (1.0, 1.0)
    assert cfg.agent.stddev_clip == 0.3
    assert (cfg.agent.critic.min_q_heads, cfg.agent.critic.policy_gradient_type) == (2, "ensemble_mean")
    assert (cfg.algo.n_step, cfg.algo.gamma) == (3, 0.99)
    assert (cfg.algo.num_updates_per_iteration, cfg.algo.actor_updates_per_iteration) == (4, 1)
    assert (cfg.algo.batch_size, cfg.algo.offline_fraction) == (256, 0.5)
    assert (cfg.algo.buffer_size, cfg.algo.learning_starts) == (200_000, 10_000)
    assert cfg.offline_data.num_episodes == 300
    assert (cfg.eval_num_episodes, cfg.eval_interval_every_steps) == (50, 10_000)


@pytest.mark.parametrize("task", ["square", "can"])
def test_tuned_rlpd_stacks(task):
    """ResFiT's arm keeps its 10 critic heads and has no equivariance; the equivariant arm
    keeps its reference's 2. The design the equivariant arm launches with is submit.sh's."""
    resfit, equi = _node(f"tuned_rlpd_{task}_config"), _node(f"equi_tuned_rlpd_{task}_config")
    project = f"robomimic-{task}-final"
    assert resfit.equivariance is None and resfit.agent.critic.num_q == 10
    assert equi.equivariance.N == 8 and equi.agent.critic.num_q == 2
    assert (resfit.wandb.project, resfit.wandb.group) == (project, "tuned-rlpd")
    assert (equi.wandb.project, equi.wandb.group) == (project, "equi-tuned-rlpd")
    assert resfit.task == equi.task == task.capitalize()
    assert resfit.offline_data.name == equi.offline_data.name == f"ankile/robomimic-mh-{task}-image"


def test_residual_only_fields_are_named():
    """train_residual_td3.py refuses a vanilla run that turns one on."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.structured(rtd3.TunedRLPDSquareConfig)
    assert rtd3.residual_only_fields_set(cfg) == []
    cfg.algo.use_base_policy_for_warmup = True
    cfg.offline_data.use_base_policy_for_base_actions = True
    assert rtd3.residual_only_fields_set(cfg) == [
        "offline_data.use_base_policy_for_base_actions=True",
        "algo.use_base_policy_for_warmup=True",
    ]
    # So base_policy=None on a residual config fails at startup instead of running mislabeled.
    for name in RESIDUAL_NAMES:
        assert rtd3.residual_only_fields_set(_node(name)), name


# The tuned-RLPD keys by (task, stack). train_residual_td3.py printed the first three in the
# 2026-10-04 debug runs; Can's equivariant key comes from the same code. They add fields that no
# residual key has, so the two can never coincide.
# Online keys before mujoco joined them (2026-10-07): 6e94049f, 8e67e7b9, 3718c816, 163d26c0.
VANILLA_CACHE_HASHES = {
    ("Square", "resfit"): ("a2d779c4", "e7739da8"),
    ("Square", "equi"): ("8b214742", "099fa3d6"),
    ("Can", "resfit"): ("83506f61", "9a511563"),
    ("Can", "equi"): ("e83a0b31", "f3250e68"),
}


def test_tuned_rlpd_cache_keys_are_their_own():
    import preflight

    if _cache_stack() != CACHE_STACK:
        pytest.skip(f"cache keys pinned to torchrl/tensordict/mujoco {CACHE_STACK}")

    for (task, stack), expected in VANILLA_CACHE_HASHES.items():
        assert preflight.cache_hashes(task, "tuned-rlpd", stack) == expected, (task, stack)
    residual = set().union(*CACHE_HASHES.values())
    assert not residual & set().union(*VANILLA_CACHE_HASHES.values())
