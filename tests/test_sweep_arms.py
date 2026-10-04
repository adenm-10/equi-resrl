"""The arms submit.sh launches, exactly as it launches them.

A queued arm that fails at launch loses its GPU slot. So every arm in
submit.sh's case table is checked here first:

  Tier 1  its overrides compose through Hydra, name valid options, set no key
          twice, and are the pre-registered ones; a task's two arms differ in
          their stack only; and an arm still pending refuses to launch.
  Tier 2  its agent builds on the GPU and runs one TD3 update, with finite
          losses, on a batch with no base action.

The arm list is pinned to the pre-registration (EXPERIMENTS.md, 2026-10-04
PRE-REGISTERED, "tuned RLPD on Can and Square"): four arms of vanilla RL with
no base policy, ResFiT's stack and the equivariant one on each task. The
equivariant arms are pending Aden's design decision, which follows the ResFiT
arms' results (2026-10-04); their overrides here are the plan's provisional
design.
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

PREREGISTERED = {"tuned-rlpd-square", "tuned-rlpd-can", "equi-tuned-rlpd-square", "equi-tuned-rlpd-can"}
PENDING = {"equi-tuned-rlpd-square", "equi-tuned-rlpd-can"}
PROJECT = {"Square": "robomimic-square-final", "Can": "robomimic-can-final"}

# What each arm launches, less the run's name and notes.
COMMON = {"algo.total_timesteps=150000", "seed=1", "debug=false"}
PROVISIONAL_EQUI = {"equivariance.actor_norm=iid_bn", "equivariance.residual_squash=true"}
LAUNCHED = {
    "tuned-rlpd-square": {"--config-name=tuned_rlpd_square_config", *COMMON},
    "tuned-rlpd-can": {"--config-name=tuned_rlpd_can_config", *COMMON},
    "equi-tuned-rlpd-square": {"--config-name=equi_tuned_rlpd_square_config", *COMMON, *PROVISIONAL_EQUI},
    "equi-tuned-rlpd-can": {"--config-name=equi_tuned_rlpd_can_config", *COMMON, *PROVISIONAL_EQUI},
}
NAMING = ("wandb.name", "wandb.notes")
# The fields in which a task's two arms may differ: the stack, and the run's naming.
STACK_FIELDS = ("equivariance", "agent.critic.num_q", "wandb.group", *NAMING)


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


def _flat(cfg) -> dict[str, object]:
    """A composed config as dotted path -> value."""
    from omegaconf import OmegaConf

    out = {}

    def walk(prefix, node):
        if isinstance(node, dict):
            for key, value in node.items():
                walk(f"{prefix}.{key}" if prefix else key, value)
        else:
            out[prefix] = node

    walk("", OmegaConf.to_container(cfg, resolve=True))
    return out


def test_case_table_is_the_preregistered_arms():
    assert len(ARMS) == len(set(ARMS)), "an arm is listed twice"
    assert set(ARMS) == PREREGISTERED


@pytest.mark.parametrize("arm", ARMS)
def test_arm_launches_its_preregistered_overrides(arm, tmp_path):
    overrides = _overrides(arm, tmp_path)
    keys = [o.split("=", 1)[0] for o in overrides]
    assert len(keys) == len(set(keys)), f"{arm} sets a key twice: {keys}"
    assert {o for o in overrides if o.split("=", 1)[0] not in NAMING} == LAUNCHED[arm]


@pytest.mark.parametrize("arm", ARMS)
def test_arm_composes(arm, tmp_path):
    cfg = _compose(_overrides(arm, tmp_path))
    task = "Square" if arm.endswith("square") else "Can"
    equivariant = arm.startswith("equi-")
    assert cfg.task == task
    assert cfg.base_policy is None, "vanilla RL"
    assert cfg.write_buffer_caches is False
    assert cfg.algo.total_timesteps == 150_000
    assert cfg.seed == 1
    assert (cfg.wandb.project, cfg.wandb.group) == (
        PROJECT[task], "equi-tuned-rlpd" if equivariant else "tuned-rlpd")
    assert cfg.wandb.name == f"{arm}-1"
    assert cfg.wandb.notes == f"tuned-rlpd-1002/{arm}/seed1"
    assert "," not in cfg.wandb.notes, "hydra splits on commas"
    assert cfg.agent.actor.actor_last_layer_init_scale is None
    if equivariant:
        assert cfg.agent.critic.num_q == 2, "the equivariant reference's heads"
        eq = cfg.equivariance
        assert (eq.actor_norm, eq.residual_squash) == ("iid_bn", True), "the provisional design"
        assert eq.frozen_target_critic is False, "the target-critic fix"
        assert eq.float_images_as_uint8 is False, "the image-scaling fix"
    else:
        assert cfg.agent.critic.num_q == 10, "ResFiT's heads, as the paper ran it"
        assert cfg.equivariance is None


@pytest.mark.parametrize("task", ["square", "can"])
def test_a_tasks_two_arms_differ_in_their_stack_only(task, tmp_path):
    resfit = _flat(_compose(_overrides(f"tuned-rlpd-{task}", tmp_path)))
    equi = _flat(_compose(_overrides(f"equi-tuned-rlpd-{task}", tmp_path)))
    missing = object()
    differing = {k for k in resfit.keys() | equi.keys() if resfit.get(k, missing) != equi.get(k, missing)}
    # equivariance is None on ResFiT's arm, so that path exists on one side only.
    assert {"agent.critic.num_q", "wandb.group", "equivariance"} <= differing
    outside = sorted(k for k in differing if not k.startswith(STACK_FIELDS))
    assert not outside, f"the {task} arms also differ in {outside}"


@pytest.mark.parametrize("arm", ARMS)
def test_only_the_decided_arms_launch(arm, tmp_path):
    """A pending arm stops before anything runs. PYTHON=true stands in for the trainer."""
    (tmp_path / "offline_buffer_cache").mkdir()
    env = {**os.environ, "SKIP_PREFLIGHT": "1", "PYTHON": "true", "ALLOW_NO_TMUX": "1",
           "CACHE_DIR": str(tmp_path)}
    env.pop("DRY_RUN", None)
    proc = subprocess.run([str(SUBMIT), arm, "1"], env=env, capture_output=True, text=True)
    if arm in PENDING:
        assert proc.returncode == 3 and "pending" in proc.stderr
    else:
        assert proc.returncode == 0 and f"Launching {arm}-1" in proc.stdout


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("arm", ARMS)
def test_arm_builds_and_updates(arm, tmp_path, device):
    import torch
    from omegaconf import open_dict

    from tools.agent_fingerprint import (
        ACTION_DIM, BATCH, CAMERAS, OBS_SHAPE, ROTATION_CENTRE, STATE_DIM, STDDEV, _stats, fixed_batch,
    )

    cfg = _compose(_overrides(arm, tmp_path))
    with open_dict(cfg):
        cfg.agent.device = str(device)
    shapes = dict(obs_shape=OBS_SHAPE, prop_shape=(STATE_DIM,), action_dim=ACTION_DIM,
                  rl_cameras=list(CAMERAS), residual_actor=False)
    if cfg.equivariance is None:
        from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent

        agent = QAgent(cfg=cfg.agent, **shapes)
    else:
        from resfit.rl_finetuning.equi_off_policy.networks.equi_normalizer import (
            build_equivariant_normalizer,
        )
        from resfit.rl_finetuning.equi_off_policy.rl.q_agent import QAgent

        agent = QAgent(agent_cfg=cfg.agent, equi_cfg=cfg.equivariance, equivariant=True, **shapes)
        normalizer = build_equivariant_normalizer(
            stats={"observation.state": _stats(STATE_DIM), "action": _stats(ACTION_DIM)},
            robot_base_xy=torch.tensor(ROTATION_CENTRE),
        )
        agent.enc.set_normalizer(normalizer, robot_base_xy=list(ROTATION_CENTRE))
        agent.to(device)

    agent.train(True)
    batch = fixed_batch(base_action=False).to(device)
    metrics = agent.update(batch, stddev=STDDEV, update_actor=True)
    for key in ("train/critic_loss", "train/actor_loss_base"):
        assert math.isfinite(metrics[key]), f"{arm}: {key} = {metrics[key]}"

    agent.train(False)
    with torch.no_grad():
        action = agent.act(fixed_batch(base_action=False)["obs"].to(device), eval_mode=True, cpu=False)
    assert action.shape == (BATCH, ACTION_DIM)
    assert torch.isfinite(action).all(), f"{arm}: non-finite eval action"
