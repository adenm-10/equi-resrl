"""The arms submit.sh launches, exactly as it launches them.

A queued arm that fails at launch loses its GPU slot. So every arm in
submit.sh's case table is checked here first:

  Tier 1  its overrides compose through Hydra, name valid options, set no key
          twice, and are the pre-registered ones.
  Tier 2  its agent builds on the GPU and runs one TD3 update, with finite
          losses.

The arm list is pinned to the pre-registration (EXPERIMENTS.md, 2026-10-02
PRE-REGISTERED, "Square: the reference with the image fix, seeds 1 and 2"):
one arm, image-fix, which is the parked ablations' reference target-fix at
the commit that fixes act()'s image scaling.
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

PREREGISTERED = {"image-fix"}
# target-fix's overrides as ccff539's submit.sh launched them (2cchirbo, seed 2), less
# the run's name and notes. 0rehtuoq, seed 1, ran the same config at ce96bb5, launched
# to 150k, where iid_bn was the default. image-fix must launch exactly these: only the
# code under them changed.
TARGET_FIX = {
    "--config-name=residual_equi_td3_square_config",
    "algo.prefetch_batches=4",
    "algo.total_timesteps=100000",
    "agent.actor.actor_last_layer_init_scale=1e-4",
    "equivariance.actor_norm=iid_bn",
    "seed=1",
    "debug=false",
}
NAMING = ("wandb.name", "wandb.notes")


def case_table_arms(script: str) -> list[str]:
    """The arm names in a submit.sh's case table, in order."""
    table = script[script.index('case "$ARM" in'):script.index("\nesac")]
    return re.findall(r"^ {4}([a-z0-9][a-z0-9-]*)\)", table, flags=re.M)


ARMS = case_table_arms(SUBMIT.read_text())


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
def test_arm_launches_target_fix_overrides(arm, tmp_path):
    overrides = _overrides(arm, tmp_path)
    keys = [o.split("=", 1)[0] for o in overrides]
    assert len(keys) == len(set(keys)), f"{arm} sets a key twice: {keys}"
    assert {o for o in overrides if o.split("=", 1)[0] not in NAMING} == TARGET_FIX


@pytest.mark.parametrize("arm", ARMS)
def test_arm_composes(arm, tmp_path):
    cfg = _compose(_overrides(arm, tmp_path))
    assert cfg.task == "Square"
    assert cfg.algo.total_timesteps == 100_000
    assert cfg.seed == 1
    # One project and group for every Square equivariant run; the name tells runs apart.
    assert (cfg.wandb.project, cfg.wandb.group) == ("robomimic-square-final", "equi-residual-rl")
    assert cfg.wandb.name == f"equi-residual-rl-actor-bn-{arm}-1"
    assert cfg.wandb.notes == f"square-image-fix-1002/{arm}/seed1"
    assert "," not in cfg.wandb.notes, "hydra splits on commas"
    eq = cfg.equivariance
    assert eq.actor_norm == "iid_bn"
    assert eq.frozen_target_critic is False, "the target-critic fix"
    assert eq.float_images_as_uint8 is False, "the image-scaling fix"
    assert cfg.agent.actor.actor_last_layer_init_scale == 1e-4


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
