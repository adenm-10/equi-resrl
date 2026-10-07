"""The arms submit.sh launches, exactly as it launches them.

A queued arm that fails at launch loses its GPU slot. So every arm in
submit.sh's case table is checked here first:

  Tier 1  its overrides compose through train_residual_td3's Hydra configs, name
          valid options, set no key twice, and are the pre-registered ones; each
          arm differs from its reference run's config in its declared change
          only; and every arm launches.
  Tier 2  its agent builds on the GPU and runs one TD3 update, with finite
          losses, on a batch with no base action, passing the demo half to the
          actor's BC term when the arm turns it on, as the trainer does.

The arm list is pinned to the pre-registration (EXPERIMENTS.md, 2026-10-07
PRE-REGISTERED, "tuned RLPD on Can: the BC term trains the encoder"): one arm of
vanilla RL with no base policy on Can, 30k steps, against tuned-rlpd-can-bc
(kc8z7vsf, at ed47957): tuned_rlpd_can_config with agent.bc_loss_coef=0.025.
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

SEED_GROUP = "tuned-rlpd-can-bc-encoder-1007"

# What each arm launches, less the run's name and notes.
COMMON = {"algo.total_timesteps=30000", "seed=1", "debug=false"}
OURS = "--config-name=tuned_rlpd_can_config"
# The reference run's change from tuned_rlpd_can_config: tuned-rlpd-can-bc, kc8z7vsf.
REFERENCE = {"agent.bc_loss_coef=0.025"}
CHANGES = {
    "tuned-rlpd-can-bc-encoder": {"agent.bc_backprop_encoder=true"},
}
LAUNCHED = {arm: {OURS, *REFERENCE, *change, *COMMON} for arm, change in CHANGES.items()}
NAMING = ("wandb.name", "wandb.notes")
# Each change, as the composed config's fields it moves.
MOVED = {
    "agent.bc_loss_coef=0.025": ("agent.bc_loss_coef", 0.025),
    "agent.bc_backprop_encoder=true": ("agent.bc_backprop_encoder", True),
}

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
    assert set(ARMS) == set(LAUNCHED)


@pytest.mark.parametrize("arm", ARMS)
def test_arm_launches_its_preregistered_overrides(arm, tmp_path):
    overrides = _overrides(arm, tmp_path)
    keys = [o.split("=", 1)[0] for o in overrides]
    assert len(keys) == len(set(keys)), f"{arm} sets a key twice: {keys}"
    assert {o for o in overrides if o.split("=", 1)[0] not in NAMING} == LAUNCHED[arm]


@pytest.mark.parametrize("arm", ARMS)
def test_arm_composes(arm, tmp_path):
    cfg = _compose(_overrides(arm, tmp_path))
    assert cfg.task == "Can"
    assert cfg.algo.total_timesteps == 30_000
    assert cfg.seed == 1
    assert cfg.wandb.name == f"{arm}-1"
    assert cfg.wandb.notes == f"{SEED_GROUP}/{arm}/seed1"
    assert "," not in cfg.wandb.notes, "hydra splits on commas"
    assert cfg.agent.critic.num_q == 10, "ResFiT's heads, as upstream's code has them"
    assert cfg.wandb.project == "robomimic-can-final"
    assert cfg.base_policy is None, "vanilla RL"
    assert cfg.equivariance is None, "ResFiT's stack"
    assert cfg.write_buffer_caches is False
    assert cfg.wandb.group == "tuned-rlpd"
    for change in REFERENCE | CHANGES[arm]:
        path, value = MOVED[change]
        node = cfg
        for name in path.split("."):
            node = getattr(node, name)
        assert node == value, f"{arm}: {path} = {node}"


@pytest.mark.parametrize("arm", ARMS)
def test_an_arm_differs_from_its_reference_in_its_change_only(arm, tmp_path):
    """The reference is tuned-rlpd-can-bc's config: tuned_rlpd_can_config with the BC term."""
    reference = _flat(_compose([OURS, *sorted(REFERENCE), *sorted(COMMON)]))
    changed = _flat(_compose(_overrides(arm, tmp_path)))
    differing = {k for k in reference.keys() | changed.keys() if reference.get(k) != changed.get(k)}
    assert differing - set(NAMING) == {MOVED[c][0] for c in CHANGES[arm]}
    assert reference["agent.bc_backprop_encoder"] is False, "upstream's default, as kc8z7vsf ran"
    assert reference["agent.critic_target_tau"] == 0.01, "rlpd.py's value, as kc8z7vsf ran"


@pytest.mark.parametrize("arm", ARMS)
def test_every_arm_launches(arm, tmp_path):
    """PYTHON=true stands in for the trainer."""
    (tmp_path / "offline_buffer_cache").mkdir()
    env = {**os.environ, "SKIP_PREFLIGHT": "1", "PYTHON": "true", "ALLOW_NO_TMUX": "1",
           "CACHE_DIR": str(tmp_path)}
    env.pop("DRY_RUN", None)
    proc = subprocess.run([str(SUBMIT), arm, "1"], env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert f"Launching {arm}-1" in proc.stdout


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("arm", ARMS)
def test_arm_builds_and_updates(arm, tmp_path, device):
    import torch
    from omegaconf import open_dict

    from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
    from tools.agent_fingerprint import ACTION_DIM, BATCH, CAMERAS, OBS_SHAPE, STATE_DIM, STDDEV, fixed_batch

    cfg = _compose(_overrides(arm, tmp_path))
    with open_dict(cfg):
        cfg.agent.device = str(device)
    agent = QAgent(obs_shape=OBS_SHAPE, prop_shape=(STATE_DIM,), action_dim=ACTION_DIM,
                   rl_cameras=list(CAMERAS), cfg=cfg.agent, residual_actor=False)

    agent.train(True)
    batch = fixed_batch(base_action=False).to(device)
    bc_batch = fixed_batch(base_action=False)[2:].to(device) if cfg.agent.bc_loss_coef > 0 else None
    metrics = agent.update(batch, stddev=STDDEV, update_actor=True, bc_batch=bc_batch, ref_agent=agent)
    keys = ["train/critic_loss", "train/actor_loss_base"] + (["rft/bc_loss"] if bc_batch is not None else [])
    for key in keys:
        assert math.isfinite(metrics[key]), f"{arm}: {key} = {metrics[key]}"

    agent.train(False)
    with torch.no_grad():
        action = agent.act(fixed_batch(base_action=False)["obs"].to(device), eval_mode=True, cpu=False)
    assert action.shape == (BATCH, ACTION_DIM)
    assert torch.isfinite(action).all(), f"{arm}: non-finite eval action"
