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
}

# The reproduction target. Values pinned here are the ones the experiment log
# records; a change to any of them invalidates comparability with prior runs
# (STANDARDS.md rule 1.6) and should fail this test until the log is updated.
REPRO_CONFIG = "residual_equi_td3_can_config"


def _node(name):
    repo = ConfigStore.instance().repo
    key = f"{name}.yaml"
    assert key in repo, f"{name} not registered in the ConfigStore"
    return repo[key].node


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
