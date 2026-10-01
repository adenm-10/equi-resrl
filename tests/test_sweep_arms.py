"""The Square ablations' arms, exactly as submit.sh launches them.

A queued arm that fails at launch loses its GPU slot. So every arm in
submit.sh's case table is checked here first:

  Tier 1  its overrides compose through Hydra, name valid options, set no key
          twice, and keep the screen's shared settings.
  Tier 2  its agent builds on the GPU and runs one TD3 update, with finite
          losses.

The arm list itself is pinned to the pre-registration (EXPERIMENTS.md,
2026-09-30 PRE-REGISTERED, as amended 2026-10-01: every arm has the actor's
batch norm, and target-fix is the reference again).
"""

from __future__ import annotations

import math
import os
import pathlib
import re
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
SUBMIT = REPO / "submit.sh"

# Each change an arm can make, by the name it has in the arm names. An arm's name is its
# changes joined by "-", in this order; target-fix, the reference, makes none.
CHANGES = {
    "critic-fn": "equivariance.critic_norm=field_norm",
    "clipq": "agent.clip_q_target_to_reward_range=true",
    "squash": "equivariance.residual_squash=true",
    "depth": "equivariance.num_critic_layers=2",
}
PREREGISTERED = {
    "target-fix", "critic-fn", "clipq",
    "squash", "depth", "critic-fn-squash", "critic-fn-depth", "clipq-squash", "clipq-depth",
    "critic-fn-clipq-squash", "critic-fn-clipq-depth",
}
# What every arm's launch sets, besides its own changes.
SHARED = {"--config-name", "algo.prefetch_batches", "algo.total_timesteps",
          "agent.actor.actor_last_layer_init_scale", "equivariance.actor_norm",
          "wandb.name", "wandb.notes", "seed", "debug"}


def _arms() -> list[str]:
    """The arm names in submit.sh's case table, in order."""
    text = SUBMIT.read_text()
    table = text[text.index('case "$ARM" in'):text.index("\nesac")]
    return re.findall(r"^ {4}([a-z0-9][a-z0-9-]*)\)", table, flags=re.M)


ARMS = _arms()


def _overrides(arm: str, tmp_path: pathlib.Path) -> list[str]:
    """What submit.sh would launch `arm` with, read from its dry run.

    PYTHON=true passes its interpreter check: the dry run exits before using it.
    """
    (tmp_path / "offline_buffer_cache").mkdir(exist_ok=True)
    env = {**os.environ, "DRY_RUN": "1", "SKIP_PREFLIGHT": "1", "PYTHON": "true",
           "CACHE_DIR": str(tmp_path)}
    out = subprocess.run([str(SUBMIT), arm, "1"], env=env, capture_output=True,
                         text=True, check=True).stdout
    return [line.strip() for line in out.splitlines() if line.startswith("    ")]


def _compose(overrides: list[str]):
    from hydra import compose, initialize

    import resfit.rl_finetuning.config.residual_td3  # noqa: F401  registers the configs

    flag, config_name = overrides[0].split("=", 1)
    assert flag == "--config-name"
    with initialize(version_base=None, config_path=None):
        return compose(config_name=config_name, overrides=overrides[1:])


def test_case_table_is_the_preregistered_arms():
    assert len(ARMS) == len(set(ARMS)), "an arm is listed twice"
    assert set(ARMS) == PREREGISTERED


@pytest.mark.parametrize("arm", ARMS)
def test_arm_composes(arm, tmp_path):
    from resfit.rl_finetuning.equi_off_policy.networks.equi_encoder import ENCODER_NORMS
    from resfit.rl_finetuning.equi_off_policy.rl.equi_rl_utils import HIDDEN_NORMS

    overrides = _overrides(arm, tmp_path)
    keys = [o.split("=", 1)[0] for o in overrides]
    assert len(keys) == len(set(keys)), f"{arm} sets a key twice: {keys}"

    cfg = _compose(overrides)
    assert cfg.task == "Square"
    assert cfg.algo.total_timesteps == 150_000
    assert cfg.seed == 1
    # One project and group for every Square equivariant run; the name tells runs apart.
    assert (cfg.wandb.project, cfg.wandb.group) == ("robomimic-square-final", "equi-residual-rl")
    assert cfg.wandb.name == f"equi-residual-rl-actor-bn-{arm}-1"
    assert "," not in cfg.wandb.notes, "hydra splits on commas"
    eq = cfg.equivariance
    assert eq.actor_norm == "iid_bn", "every arm has the actor's batch norm (2026-10-01 AMENDMENT)"
    assert eq.critic_norm in HIDDEN_NORMS and eq.encoder_norm in ENCODER_NORMS
    assert eq.frozen_target_critic is False, "every arm has the target-critic fix"
    assert cfg.agent.actor.actor_last_layer_init_scale == 1e-4


def test_target_fix_is_the_reference_config(tmp_path):
    """The arm every other arm is judged against overrides nothing else."""
    overrides = _overrides("target-fix", tmp_path)
    assert {o.split("=", 1)[0] for o in overrides} == SHARED


@pytest.mark.parametrize("arm", ARMS)
def test_arm_name_lists_its_changes(arm, tmp_path):
    """critic-fn-clipq-squash makes exactly those three changes, and nothing else."""
    parts = [name for name in CHANGES if name in arm]
    assert arm == "target-fix" or "-".join(parts) == arm, f"{arm} is not a join of {list(CHANGES)}"
    own = {o for o in _overrides(arm, tmp_path) if o.split("=", 1)[0] not in SHARED}
    assert own == {CHANGES[name] for name in parts}


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("arm", ARMS)
def test_arm_builds_and_updates(arm, tmp_path, device):
    import torch
    from omegaconf import open_dict

    from resfit.rl_finetuning.equi_off_policy.networks.equi_normalizer import (
        build_equivariant_normalizer,
    )
    from resfit.rl_finetuning.equi_off_policy.rl.q_agent import QAgent
    from tools.agent_fingerprint import (
        ACTION_DIM, CAMERAS, OBS_SHAPE, ROTATION_CENTRE, STATE_DIM, STDDEV, _stats, fixed_batch,
    )

    cfg = _compose(_overrides(arm, tmp_path))
    with open_dict(cfg):
        cfg.agent.device = str(device)
    agent = QAgent(
        obs_shape=OBS_SHAPE, prop_shape=(STATE_DIM,), action_dim=ACTION_DIM,
        rl_cameras=list(CAMERAS), agent_cfg=cfg.agent, equi_cfg=cfg.equivariance,
        residual_actor=True, equivariant=True,
    )
    normalizer = build_equivariant_normalizer(
        stats={"observation.state": _stats(STATE_DIM), "action": _stats(ACTION_DIM)},
        robot_base_xy=torch.tensor(ROTATION_CENTRE),
    )
    agent.enc.set_normalizer(normalizer, robot_base_xy=list(ROTATION_CENTRE))
    agent.to(device)

    agent.train(True)
    metrics = agent.update(fixed_batch().to(device), stddev=STDDEV, update_actor=True)
    for key in ("train/critic_loss", "train/actor_loss_base"):
        assert math.isfinite(metrics[key]), f"{arm}: {key} = {metrics[key]}"

    agent.train(False)
    with torch.no_grad():
        action = agent.act(fixed_batch()["obs"].to(device), eval_mode=True, cpu=False)
    assert torch.isfinite(action).all(), f"{arm}: non-finite eval action"
