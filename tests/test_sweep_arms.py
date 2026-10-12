"""The arms submit.sh launches, exactly as it launches them.

A queued arm that fails at launch loses its GPU slot. So every arm in
submit.sh's case table is checked here first:

  Tier 1  its overrides are the pre-registered ones and set no key twice; it
          composes through train_residual_td3's Hydra configs to the anchor's
          config (oj5k0jqb's launch, resfit-residual-square 1 at 9cf207d) but for
          its one change; the anchor is what oj5k0jqb launched with, and composes
          today to oj5k0jqb's recorded config but for fields added since at their
          no-op defaults; the re-rendered demos leave the action scaler as it was;
          the dataset each arm loads is there; the smoke mode is the arm, briefly;
          and every arm launches.
  Tier 2  its agent builds on the GPU and runs one TD3 update, with finite
          losses; at the same seed it starts from the anchor's weights, bit for
          bit, but for any norm it drops; noln has no LayerNorm in its actor or
          critic and keeps its ViTs'. A test here keeps on record that the
          fixed-env baselines' FieldNorm critic is blind to every trivial input
          (2026-10-08 FINDING).

The arm list is pinned to the pre-registration (EXPERIMENTS.md, 2026-10-12
PRE-REGISTERED, "ResFiT's residual on Square, given the equivariant stack's
differences one at a time; the equivariant residual given ResFiT's layout").
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import re
import subprocess

import pytest

from resfit.lerobot.utils.load_policy import RUNS_ROOT

REPO = pathlib.Path(__file__).resolve().parent.parent
SUBMIT = REPO / "submit.sh"
LEROBOT_HOME = pathlib.Path.home() / ".cache/huggingface/lerobot"

SEED_GROUP = "resfit-equi-diff-1012"

RERENDERED = "resfit-residual-square-rerendered"
NOLN = "resfit-residual-square-noln"
CONFIG = "residual_td3_square_config"
DEMOS = "ankile/robomimic-mh-square-image"  # the config's own, the original coded demos
RERENDERED_DEMOS = "local/robomimic-mh-square-rerendered"  # every equivariant run's

# What every arm launches, less the run's name, notes, project and group.
COMMON = {"algo.total_timesteps=150000", "read_buffer_caches=false", "write_buffer_caches=false",
          "seed=1", "debug=false"}
# oj5k0jqb's launch at 9cf207d, the anchor every arm is one change from.
ANCHOR = {f"--config-name={CONFIG}", "agent.critic.num_q=10", "base_policy.wt_version=v9", *COMMON}
LAUNCHED = {
    RERENDERED: ANCHOR | {f"offline_data.name={RERENDERED_DEMOS}"},
    NOLN: ANCHOR | {"agent.actor.use_layer_norm=false", "agent.critic.use_layer_norm=false"},
}
# Each arm's changes from the anchor, as composed paths, and its values there.
FROM_ANCHOR = {
    RERENDERED: {"offline_data.name"},
    NOLN: {"agent.actor.use_layer_norm", "agent.critic.use_layer_norm"},
}
ARM_VALUES = {
    RERENDERED: {"offline_data.name": RERENDERED_DEMOS},
    NOLN: {"agent.actor.use_layer_norm": False, "agent.critic.use_layer_norm": False},
}
# The demos each arm loads.
ARM_DEMOS = {RERENDERED: RERENDERED_DEMOS, NOLN: DEMOS}
# The fixed-env baselines' equi-tuned-rlpd-can (eyyi99q4) as 9cf207d launched it, less naming.
BASELINES = {
    "equi-tuned-rlpd-can": ["--config-name=equi_tuned_rlpd_can_config", "equivariance.residual_squash=true",
                            "agent.critic.num_q=10", "equivariance.num_critic_layers=2",
                            "equivariance.critic_norm=field_norm", "equivariance.actor_norm=none",
                            "algo.total_timesteps=150000", "read_buffer_caches=false",
                            "write_buffer_caches=false", "seed=1", "debug=false"],
}
NAMING = ("wandb.name", "wandb.notes", "wandb.project", "wandb.group")
WANDB = ("robomimic-square-final", "residual-rl")
# Under CACHE_DIR, where every run package lives: a launch worktree has no outputs/.
OJ5 = RUNS_ROOT / "robomimic-square-final/oj5k0jqb"

# The composed config every arm must have: oj5k0jqb's, as its recorded config holds it.
SHARED = {
    "task": "Square", "offline_data.name": DEMOS, "algo.total_timesteps": 150_000,
    "read_buffer_caches": False, "write_buffer_caches": False, "algo.n_step": 3, "algo.gamma": 0.99,
    "algo.batch_size": 256, "algo.offline_fraction": 0.5, "algo.num_updates_per_iteration": 4,
    "algo.actor_updates_per_iteration": 1, "algo.buffer_size": 200_000, "algo.learning_starts": 10_000,
    "algo.critic_warmup_steps": 10_000, "num_envs": 1, "eval_first": True,
    "eval_interval_every_steps": 10_000, "eval_num_episodes": 50, "save_video": True, "seed": 1,
    "debug": False, "wandb.mode": "online", "encode_env_frames_like_demos": False, "equivariance": None,
    "base_policy.wt_version": "v9", "offline_data.use_base_policy_for_base_actions": True,
    "offline_data.min_state_std": 0.1, "agent.actor_lr": 1e-6, "agent.critic_lr": 1e-4,
    "agent.critic_target_tau": 0.005, "algo.stddev_min": 0.05, "algo.stddev_max": 0.05,
    "agent.stddev_clip": 0.3, "agent.actor.action_scale": 0.1,
    "agent.actor.actor_last_layer_init_scale": 0.0, "algo.random_action_noise_scale": 0.2,
    "algo.use_base_policy_for_warmup": True, "algo.actor_lr_warmup_steps": 0, "agent.bc_loss_coef": 0.0,
    "agent.critic.num_q": 10, "agent.critic.min_q_heads": 2,
    "agent.critic.policy_gradient_type": "ensemble_mean", "agent.critic_grad_clip_norm": 1.0,
}
# Keys a run writes into its own recorded config, which no composition holds.
RUNTIME_STAMPED = {"actor_name"}


def case_table_arms(script: str) -> list[str]:
    """The arm names in a submit.sh's case table, in order."""
    table = script[script.index('case "$ARM" in'):script.index("\nesac")]
    return re.findall(r"^ {4}([a-z0-9][a-z0-9-]*)\)", table, flags=re.M)


ARMS = case_table_arms(SUBMIT.read_text())
EQUI_ARMS = [arm for arm in ARMS if arm.startswith("equi-")]


def _overrides(arm: str, tmp_path: pathlib.Path, **env_extra) -> list[str]:
    """What submit.sh would launch `arm` with, read from its dry run.

    PYTHON=true passes its interpreter check: the dry run exits before using it.
    """
    (tmp_path / "offline_buffer_cache").mkdir(exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k != "SMOKE"}
    env.update({"DRY_RUN": "1", "SKIP_PREFLIGHT": "1", "PYTHON": "true", "CACHE_DIR": str(tmp_path),
                **env_extra})
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
def test_arm_composes_to_its_preregistered_recipe(arm, tmp_path):
    flat = _flat(_compose(_overrides(arm, tmp_path)))
    expected = {**SHARED, **ARM_VALUES[arm], "wandb.project": WANDB[0], "wandb.group": WANDB[1],
                "wandb.name": f"{arm}-1", "wandb.notes": f"{SEED_GROUP}/{arm}/seed1"}
    for path, value in expected.items():
        assert flat.get(path, "absent") == value, f"{arm}: {path} = {flat.get(path, 'absent')}"
    assert "," not in flat["wandb.notes"], "hydra splits on commas"


def _differing(a: dict, b: dict) -> set[str]:
    return {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)} - set(NAMING)


@pytest.mark.parametrize("arm", ARMS)
def test_arm_is_the_anchor_but_for_its_change(arm, tmp_path):
    ours = _flat(_compose(_overrides(arm, tmp_path)))
    anchor = _flat(_compose(sorted(ANCHOR, key=lambda o: not o.startswith("--config-name"))))
    assert _differing(ours, anchor) == FROM_ANCHOR[arm]


@pytest.mark.skipif(not OJ5.exists(), reason="oj5k0jqb's run package is not on this machine")
def test_the_anchor_is_what_oj5k0jqb_launched_with():
    argv = json.loads((OJ5 / "manifest.json").read_text())["argv"][1:]
    launched = {o for o in argv if o.split("=", 1)[0] not in (*NAMING, "hydra.run.dir")}
    assert launched == ANCHOR


def _number(value):
    """wandb records some floats as strings ('1e-06'); compare them as the numbers they are."""
    if isinstance(value, str) and re.fullmatch(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?", value):
        return float(value)
    return value


def _flat_recorded(path: pathlib.Path) -> dict[str, object]:
    """A run's recorded config.yaml as dotted path -> value, less wandb's own keys, the versions the
    run stamped in, and its naming."""
    import yaml

    record = yaml.safe_load(path.read_text())
    values = {k: v["value"] for k, v in record.items() if not k.startswith("_") and k != "versions"}
    out = {}

    def walk(prefix, node):
        if isinstance(node, dict):
            for key, value in node.items():
                walk(f"{prefix}.{key}" if prefix else key, value)
        else:
            out[prefix] = _number(node)

    walk("", values)
    return {k: v for k, v in out.items() if not k.startswith("wandb.")}


@pytest.mark.skipif(not OJ5.exists(), reason="oj5k0jqb's run package is not on this machine")
def test_the_anchor_composes_to_oj5k0jqbs_recorded_config():
    """So the anchor at this commit is the config oj5k0jqb ran with. A field added since 9cf207d
    is allowed only at the default that leaves every earlier run as it was."""
    recorded = _flat_recorded(OJ5 / "wandb_logs/config.yaml")
    ours = {k: _number(v) for k, v in _flat(_compose(sorted(ANCHOR, key=lambda o: not o.startswith("--config-name")))).items()
            if not k.startswith("wandb.")}
    added = {k: ours[k] for k in ours.keys() - recorded.keys()}
    assert added == {"encode_env_frames_like_demos": False}
    changed = {k for k in ours.keys() & recorded.keys() if ours[k] != recorded[k]}
    assert changed <= RUNTIME_STAMPED, {k: (recorded[k], ours[k]) for k in changed}
    assert recorded.keys() - ours.keys() <= RUNTIME_STAMPED


def test_the_rerendered_demos_leave_the_action_scaler_as_it_was():
    """The re-rendered demos keep every demo's actions, so the action statistics and the ActionScaler
    built from them are the original's bit for bit: the arm's change reaches the frames and the
    state, not the residual's units. The state is observed one physics substep late, as the env
    observes it, so its statistics move a little (at most 4.2e-4 on a min or max)."""
    import torch
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

    from resfit.rl_finetuning.utils.normalization import ActionScaler

    original, rerendered = (LeRobotDatasetMetadata(name).stats for name in (DEMOS, RERENDERED_DEMOS))
    for key in ("min", "max", "mean", "std"):
        assert torch.equal(torch.as_tensor(original["action"][key]), torch.as_tensor(rerendered["action"][key])), key
    old, new = (ActionScaler.from_dataset_stats(action_stats=stats["action"], action_scale=0.1,
                                                min_range_per_dim=0.1).limits
                for stats in (original, rerendered))
    assert torch.equal(old.min, new.min) and torch.equal(old.max, new.max)
    for key in ("min", "max"):
        drift = (torch.as_tensor(original["observation.state"][key]) -
                 torch.as_tensor(rerendered["observation.state"][key])).abs().max().item()
        assert 0 < drift < 5e-4, (key, drift)


@pytest.mark.parametrize("arm", ARMS)
def test_the_dataset_each_arm_loads_is_built(arm):
    """A dataset the trainer cannot find is a launch that dies at startup."""
    root = LEROBOT_HOME / ARM_DEMOS[arm]
    assert (root / "meta/info.json").exists() and (root / "meta/episodes_stats.jsonl").exists(), root
    if ARM_DEMOS[arm] == RERENDERED_DEMOS:
        assert (root / "meta/rerender.json").exists(), root


@pytest.mark.parametrize("arm", ARMS)
def test_smoke_mode_is_the_arm_briefly(arm, tmp_path):
    real = _overrides(arm, tmp_path)
    smoke = _overrides(arm, tmp_path, SMOKE="1")
    keys = [o.split("=", 1)[0] for o in smoke]
    assert len(keys) == len(set(keys)), f"{arm} smoke sets a key twice: {keys}"
    brief = {"algo.total_timesteps=1000", "algo.learning_starts=300", "eval_first=false",
             "eval_interval_every_steps=100000", "wandb.mode=offline",
             "wandb.project=equi-resrl-smoke", "wandb.group=smoke", f"wandb.name=smoke-{arm}-1",
             f"wandb.notes=smoke/{SEED_GROUP}/{arm}/seed1"}
    replaced = {o.split("=", 1)[0] for o in brief}
    assert set(smoke) == {o for o in real if o.split("=", 1)[0] not in replaced} | brief
    assert _compose(smoke).algo.total_timesteps == 1000


@pytest.mark.parametrize("arm", ARMS)
def test_every_arm_launches(arm, tmp_path):
    """PYTHON=true stands in for the trainer."""
    (tmp_path / "offline_buffer_cache").mkdir()
    env = {**os.environ, "SKIP_PREFLIGHT": "1", "PYTHON": "true", "ALLOW_NO_TMUX": "1",
           "CACHE_DIR": str(tmp_path)}
    env.pop("DRY_RUN", None)
    env.pop("SMOKE", None)
    proc = subprocess.run([str(SUBMIT), arm, "1"], env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert f"Launching {arm}-1" in proc.stdout


# ---------------------------------------------------------------------------
# Tier 2: each arm's agent, as its composed config builds it
# ---------------------------------------------------------------------------


def _agent(cfg, device):
    """The agent the trainer builds from `cfg`, on the reference batch's shapes."""
    import torch
    from omegaconf import open_dict

    from tools.agent_fingerprint import ACTION_DIM, CAMERAS, OBS_SHAPE, ROTATION_CENTRE, STATE_DIM, _stats

    with open_dict(cfg):
        cfg.agent.device = str(device)
    shapes = dict(obs_shape=OBS_SHAPE, prop_shape=(STATE_DIM,), action_dim=ACTION_DIM,
                  rl_cameras=list(CAMERAS), residual_actor=cfg.base_policy is not None)
    if cfg.equivariance is None:
        from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent

        return QAgent(cfg=cfg.agent, **shapes)
    from resfit.rl_finetuning.equi_off_policy.networks.equi_normalizer import build_equivariant_normalizer
    from resfit.rl_finetuning.equi_off_policy.rl.q_agent import QAgent

    agent = QAgent(agent_cfg=cfg.agent, equi_cfg=cfg.equivariance, equivariant=True, **shapes)
    normalizer = build_equivariant_normalizer(
        stats={"observation.state": _stats(STATE_DIM), "action": _stats(ACTION_DIM)},
        robot_base_xy=torch.tensor(ROTATION_CENTRE),
    )
    agent.enc.set_normalizer(normalizer, robot_base_xy=list(ROTATION_CENTRE))
    return agent.to(device)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("arm", ARMS)
def test_arm_builds_and_updates(arm, tmp_path, device):
    import torch

    from tools.agent_fingerprint import ACTION_DIM, BATCH, STDDEV, fixed_batch

    cfg = _compose(_overrides(arm, tmp_path))
    residual = cfg.base_policy is not None
    agent = _agent(cfg, device)
    agent.train(True)
    metrics = agent.update(fixed_batch(base_action=residual).to(device), stddev=STDDEV, update_actor=True)
    for key in ("train/critic_loss", "train/actor_loss_base"):
        assert math.isfinite(metrics[key]), f"{arm}: {key} = {metrics[key]}"

    agent.train(False)
    with torch.no_grad():
        action = agent.act(fixed_batch(base_action=residual)["obs"].to(device), eval_mode=True, cpu=False)
    assert action.shape == (BATCH, ACTION_DIM)
    assert torch.isfinite(action).all(), f"{arm}: non-finite eval action"




@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("training", [False, True], ids=["eval", "train"])
@pytest.mark.parametrize("arm", EQUI_ARMS)
def test_equivariant_arm_commutes_with_the_group(arm, training, tmp_path, device):
    """The arm's own networks, at its sizes (32 encoder fields, 128 hidden): pi(g.s) = g.pi(s), and
    Q(g.s, g.a) = Q(s, a). A zero-initialized actor outputs zero, which commutes trivially; its
    architecture is its pair's, checked there."""
    import torch
    from escnn import nn as enn

    from tests import group_action as ga
    from tests.conftest import N, ONE_ARM, TOL
    from tests.test_equivariance import _assert_actor_equivariant, _feat, _rel_err

    cfg = _compose(_overrides(arm, tmp_path))
    assert cfg.equivariance.N == N
    agent = _agent(cfg, device)
    agent.train(training)  # QAgent.train returns None
    eq, spec = cfg.equivariance, ONE_ARM
    critic_layout = ga.enc_out_layout_critic(eq.enc_degree_channel, eq.ih_out_channel,
                                             spec["n_arms"], spec["gripper_dim"])
    if cfg.base_policy is None:
        actor_type, actor_layout = agent.enc.enc_out_type_critic, critic_layout
    else:
        actor_type = agent.enc.enc_out_type_actor
        actor_layout = ga.enc_out_layout_actor(eq.enc_degree_channel, eq.ih_out_channel,
                                               spec["n_arms"], spec["gripper_dim"], spec["hand_dof"])
    if cfg.agent.actor.actor_last_layer_init_scale != 0.0:
        _assert_actor_equivariant(agent.actor, actor_type, actor_layout, spec, device, arm, seed=21)

    act_layout = ga.action_layout(spec["n_arms"], spec["hand_dof"])
    feat = _feat(None, critic_layout, device, seed=22)
    act = _feat(None, act_layout, device, seed=23)
    with torch.no_grad():
        q = agent.critic(enn.GeometricTensor(feat, agent.enc.enc_out_type_critic), act)
        assert q.shape[0] == cfg.agent.critic.num_q
        for k in range(1, N):
            got = agent.critic(enn.GeometricTensor(ga.act_on_layout(feat, critic_layout, k, N),
                                                   agent.enc.enc_out_type_critic),
                               ga.act_on_layout(act, act_layout, k, N))
            err = _rel_err(got, q)
            assert err < TOL, f"{arm}: critic not invariant at k={k}: {err:.2e}"


# Changes to one input of the reference batch, each large against that input's range. Every input
# reaches the critic and the actor; the trivial-type ones are the wrist camera, the arm's height,
# the gripper opening, and the z, yaw and gripper actions.
SEES = 1e-4  # every input moves Q by at least 1.1e-3 without FieldNorm, a trivial one by at most
             # 6.7e-6 with it (2026-10-08, on this batch at init)
TRIVIAL_INPUTS = ("wrist image", "state z", "state gripper", "action z", "action yaw", "action gripper")


def _input_changes():
    from tools.agent_fingerprint import CAMERAS

    def image(cam):
        return lambda obs, act: obs.__setitem__(cam, obs[cam].new_zeros(obs[cam].shape))

    def state(i, by):
        return lambda obs, act: obs.__setitem__("observation.state", obs["observation.state"].index_add(
            1, obs["observation.state"].new_tensor([i], dtype=int),
            obs["observation.state"].new_full((obs["observation.state"].shape[0], 1), by)))

    def action(i):
        return lambda obs, act: act.__setitem__((slice(None), i), act[:, i] + 0.5)

    return {
        "front image": image(CAMERAS[0]), "wrist image": image(CAMERAS[1]),
        "state x": state(0, 0.1), "state z": state(2, 0.2), "state gripper": state(7, 0.02),
        "action x": action(0), "action rx": action(3), "action z": action(2),
        "action yaw": action(5), "action gripper": action(6),
    }


def _output_changes(cfg, device) -> dict[str, tuple[float, float]]:
    """Per input change, the largest change in Q (mean over heads) and in the actor's mean action.

    A residual actor's last layer starts scaled down (1e-4), and its mean is squashed and multiplied
    by its action scale (0.1); each is a fixed map on the output that shrinks every input's change
    alike. What reaches the actor is a property of its layers, so it is built at escnn's own init and
    read before the squash and the scale. (Read after them, at escnn's init, the residual's change
    with the gripper was 6e-5: the layers' 7.5e-4, which grows with the gripper's change, squashed at
    |mu| up to about 3 and scaled by 0.1; 2026-10-09.)
    """
    import torch
    from omegaconf import open_dict

    from tools.agent_fingerprint import fixed_batch

    residual = cfg.base_policy is not None
    with open_dict(cfg):
        cfg.agent.actor.actor_last_layer_init_scale = None
        cfg.agent.actor.action_scale = 1.0
        cfg.equivariance.residual_squash = False
    torch.manual_seed(0)
    agent = _agent(cfg, device)
    agent.train(False)  # QAgent.train returns None

    def outputs(change=None):
        batch = fixed_batch(base_action=residual).to(device)
        obs = {k: v.clone() for k, v in batch["obs"].items()}
        act = batch["action"].clone()
        if change is not None:
            change(obs, act)
        with torch.no_grad():
            actor_feat, critic_feat = agent.enc(obs)
            return agent.critic(critic_feat, act).mean(dim=0), agent.actor(actor_feat, std=None)

    q0, pi0 = outputs()
    changes = {}
    for name, change in _input_changes().items():
        q, pi = outputs(change)
        changes[name] = ((q - q0).abs().max().item(), (pi - pi0).abs().max().item())
    return changes


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("arm", EQUI_ARMS)
def test_every_input_can_move_the_critic_and_the_actor(arm, tmp_path, device):
    """Invariance tests pass on a network that ignores an input, so this checks what they cannot."""
    changes = _output_changes(_compose(_overrides(arm, tmp_path)), device)
    blind_q = {k: f"{dq:.1e}" for k, (dq, _) in changes.items() if dq <= SEES}
    assert not blind_q, f"{arm}: Q does not move with {blind_q}"
    blind_pi = {k: f"{dpi:.1e}" for k, (_, dpi) in changes.items()
                if not k.startswith("action") and dpi <= SEES}
    assert not blind_pi, f"{arm}: the actor does not move with {blind_pi}"


@pytest.mark.gpu
@pytest.mark.parametrize("arm", EQUI_ARMS)
def test_each_arm_has_no_norm_and_five_one_layer_heads(arm, tmp_path, device):
    """6ng62tpm's design: no norm in the encoder, the actor or the critic, and five critic heads of
    one hidden layer each (a hidden Linear, then the output Linear after pooling)."""
    from escnn import nn as enn

    from resfit.rl_finetuning.equi_off_policy.rl.equi_rl_utils import (
        FieldNorm,
        RegularInnerBatchNorm,
        RegularLayerNorm,
        TrivialLayerNorm,
    )

    agent = _agent(_compose(_overrides(arm, tmp_path)), device)
    norms = (enn.IIDBatchNorm1d, RegularInnerBatchNorm, FieldNorm, RegularLayerNorm, TrivialLayerNorm)
    for part in (agent.enc, agent.actor, agent.critic):
        assert not any(isinstance(m, norms) for m in part.modules()), f"{arm}: a norm in {type(part).__name__}"
    heads = agent.critic.q_ensemble.heads
    assert len(heads) == 5
    assert all(sum(isinstance(m, enn.Linear) for m in head.modules()) == 2 for head in heads)


def _seeded_agent(cfg, device, seed: int = 1):
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return _agent(cfg, device)


def _weights_in_order(module, drop_layer_norms: bool) -> list:
    """Every entry of the module's state, in order, less LayerNorms' if asked. Dropping a LayerNorm
    renumbers the Sequentials after it, so names cannot be matched; the order can. The critic's
    ensemble also keeps its heads' parameters stacked, as `_vmap_param_net_<i>_*` for layer i of
    its head template, so those of a template LayerNorm are dropped too."""
    import re

    import torch

    owners = dict(module.named_modules())
    out = []
    for name, tensor in module.state_dict().items():
        owner = owners.get(name.rsplit(".", 1)[0]) if "." in name else module
        stacked = re.search(r"(.*)_vmap_param_net_(\d+)_\w+$", name)
        if stacked:
            ensemble = owners[stacked.group(1).rstrip(".")]
            owner = ensemble._head_template.net[int(stacked.group(2))]
        if drop_layer_norms and isinstance(owner, torch.nn.LayerNorm):
            continue
        out.append(tensor)
    return out


@pytest.mark.gpu
@pytest.mark.parametrize("arm", ARMS)
def test_each_arm_starts_from_the_anchors_weights(arm, tmp_path, device):
    """At the same seed, the arm's initial weights are the anchor's bit for bit: rerendered changes
    the data alone, and noln drops LayerNorms, which draw no random number at init, so every other
    weight is drawn as the anchor's. The episodes still differ, since `seed` does not reach the
    envs."""
    import torch

    anchor = _seeded_agent(_compose(sorted(ANCHOR, key=lambda o: not o.startswith("--config-name"))), device)
    ours = _seeded_agent(_compose(_overrides(arm, tmp_path)), device)
    if arm == RERENDERED:
        theirs, mine = anchor.state_dict(), ours.state_dict()
        assert theirs.keys() == mine.keys()
        for key, value in mine.items():
            assert torch.equal(value, theirs[key]), f"{arm}: {key} differs from the anchor's"
        return
    # noln: the anchor's actor and critic less their LayerNorms; the encoders keep theirs.
    for part in ("actor", "critic"):
        theirs = _weights_in_order(getattr(anchor, part), drop_layer_norms=True)
        mine = _weights_in_order(getattr(ours, part), drop_layer_norms=False)
        assert len(theirs) == len(mine), (part, len(theirs), len(mine))
        for i, (a, b) in enumerate(zip(theirs, mine)):
            assert a.shape == b.shape and torch.equal(a, b), f"{arm}: {part} tensor {i} differs"
    theirs, mine = anchor.encoders.state_dict(), ours.encoders.state_dict()
    assert theirs.keys() == mine.keys() and all(torch.equal(mine[k], theirs[k]) for k in mine)


@pytest.mark.gpu
def test_noln_has_no_layernorm_in_its_actor_or_critic_and_its_vits_keep_theirs(tmp_path, device):
    """Where ResFiT's use_layer_norm puts LayerNorm: after the actor's compression and each of its
    hidden layers, and after the critic's input projection and each head's hidden layers. The ViT
    encoders' own (two per transformer layer and a final one) are another switch, noln-all's."""
    import torch

    def count(module):
        return sum(isinstance(m, torch.nn.LayerNorm) for m in module.modules())

    anchor = _agent(_compose(sorted(ANCHOR, key=lambda o: not o.startswith("--config-name"))), device)
    ours = _agent(_compose(_overrides(NOLN, tmp_path)), device)
    assert count(anchor.actor) > 0 and count(anchor.critic) > 0
    assert count(ours.actor) == count(ours.critic) == 0
    assert count(ours.encoders) == count(anchor.encoders) == 2 * 3


@pytest.mark.gpu
@pytest.mark.slow
def test_the_field_norm_critic_these_arms_replace_is_blind_to_every_trivial_input(device):
    """eyyi99q4's critic: FieldNorm subtracts each regular field's trivial part, which is all a
    trivial-type input contributes to it (2026-10-08 FINDING). The rotating inputs still reach Q."""
    changes = _output_changes(_compose(BASELINES["equi-tuned-rlpd-can"]), device)
    for name, (dq, _) in changes.items():
        if name in TRIVIAL_INPUTS:
            assert dq < SEES, f"field norm's Q moved by {dq:.1e} with {name}"
        else:
            assert dq > SEES, f"field norm's Q did not move with {name}: {dq:.1e}"
