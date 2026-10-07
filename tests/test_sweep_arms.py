"""The arms submit.sh launches, exactly as it launches them.

A queued arm that fails at launch loses its GPU slot. So every arm in
submit.sh's case table is checked here first:

  Tier 1  its overrides are the pre-registered ones, set no key twice, and
          compose through train_residual_td3's Hydra configs to the
          pre-registered recipe; within a task, the two stacks' arms differ in
          the stack only; the zero-init arm differs from its pair in the init
          only; the smoke mode is the arm, briefly; and every arm launches.
  Tier 2  its agent builds on the GPU and runs one TD3 update, with finite
          losses, on the batch its method trains on; each equivariant arm's
          actor is equivariant and its critic invariant, at the arm's own
          sizes; and the zero-init arm's residual starts at exactly zero and
          leaves it.

The arm list is pinned to the pre-registration (EXPERIMENTS.md, 2026-10-07
PRE-REGISTERED, "the fixed-env baselines"): tuned RLPD on Can and residual RL on
Square, each in ResFiT's stack and the equivariant one at Aden's design, plus
the equivariant residual with its last layer initialized to zero.
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

SEED_GROUP = "fixed-env-baselines-1007"

# What each arm launches, less the run's name, notes, project and group.
COMMON = {"algo.total_timesteps=150000", "read_buffer_caches=false", "write_buffer_caches=false",
          "seed=1", "debug=false"}
DESIGN = {"equivariance.residual_squash=true", "agent.critic.num_q=10",
          "equivariance.num_critic_layers=2", "equivariance.critic_norm=field_norm",
          "equivariance.actor_norm=none"}
PIN = "base_policy.wt_version=v9"
LAUNCHED = {
    "resfit-tuned-rlpd-can": {"--config-name=tuned_rlpd_can_config", *COMMON},
    "equi-tuned-rlpd-can": {"--config-name=equi_tuned_rlpd_can_config", *DESIGN, *COMMON},
    "resfit-residual-square": {"--config-name=residual_td3_square_config", "agent.critic.num_q=10",
                               PIN, *COMMON},
    "equi-residual-square": {"--config-name=residual_equi_td3_square_config", *DESIGN,
                             "agent.actor.actor_last_layer_init_scale=1e-4", PIN, *COMMON},
    "equi-residual-square-zero-init": {"--config-name=residual_equi_td3_square_config", *DESIGN,
                                       "agent.actor.actor_last_layer_init_scale=0.0", PIN, *COMMON},
}
NAMING = ("wandb.name", "wandb.notes", "wandb.project", "wandb.group")
# Where each kind of run has gone before (Aden, 2026-10-07).
WANDB = {
    "resfit-tuned-rlpd-can": ("robomimic-can-final", "tuned-rlpd"),
    "equi-tuned-rlpd-can": ("robomimic-can-final", "tuned-rlpd"),
    "resfit-residual-square": ("robomimic-square-final", "residual-rl"),
    "equi-residual-square": ("robomimic-square-final", "equi-residual-rl"),
    "equi-residual-square-zero-init": ("robomimic-square-final", "equi-residual-rl"),
}
TASK = {arm: "Can" if arm.endswith("-can") else "Square" for arm in LAUNCHED}

# The composed config every arm must have, then each method's recipe: the pre-registration's numbers.
SHARED = {
    "algo.total_timesteps": 150_000, "read_buffer_caches": False, "write_buffer_caches": False,
    "agent.critic.num_q": 10, "algo.n_step": 3, "algo.gamma": 0.99, "algo.batch_size": 256,
    "algo.offline_fraction": 0.5, "algo.num_updates_per_iteration": 4, "algo.buffer_size": 200_000,
    "algo.learning_starts": 10_000, "num_envs": 1, "eval_first": True,
    "eval_interval_every_steps": 10_000, "eval_num_episodes": 50, "seed": 1, "debug": False,
    "wandb.mode": "online",
}
RECIPE = {
    "tuned-rlpd": {
        "base_policy": None, "algo.critic_warmup_steps": 0, "agent.actor_lr": 1e-4,
        "agent.critic_target_tau": 0.01, "algo.stddev_min": 0.1, "algo.stddev_max": 0.1,
        "agent.actor.action_scale": 1.0, "algo.random_action_noise_scale": 1.0,
        "algo.use_base_policy_for_warmup": False,
        "agent.actor.actor_last_layer_init_scale": None,
    },
    "residual": {
        "base_policy.wandb_id": "robomimic-square-bc/3hzs5bz1", "base_policy.wt_type": "best",
        "base_policy.wt_version": "v9", "algo.critic_warmup_steps": 10_000,
        "agent.actor_lr": 1e-6, "agent.critic_target_tau": 0.005, "algo.stddev_min": 0.05,
        "algo.stddev_max": 0.05, "agent.actor.action_scale": 0.1,
        "algo.random_action_noise_scale": 0.2, "algo.use_base_policy_for_warmup": True,
        "offline_data.use_base_policy_for_base_actions": True,
    },
}
INIT = {"resfit-residual-square": 0.0, "equi-residual-square": 1e-4,
        "equi-residual-square-zero-init": 0.0}
EQUI_DESIGN = {
    "equivariance.N": 8, "equivariance.use_equivariant_model": True,
    "equivariance.residual_squash": True, "equivariance.num_critic_layers": 2,
    "equivariance.critic_norm": "field_norm", "equivariance.actor_norm": "none",
    "equivariance.actor_degree_channel": 128, "equivariance.critic_degree_channel": 128,
    "equivariance.frozen_target_critic": False, "equivariance.float_images_as_uint8": False,
}
# Within a task the stacks share everything but these (ARCHITECTURE.md, "The two stacks side by
# side"), and the residual arms also their last layer's init.
STACK = "equivariance"
PAIRS = [("resfit-tuned-rlpd-can", "equi-tuned-rlpd-can"),
         ("resfit-residual-square", "equi-residual-square"),
         ("resfit-residual-square", "equi-residual-square-zero-init")]


def case_table_arms(script: str) -> list[str]:
    """The arm names in a submit.sh's case table, in order."""
    table = script[script.index('case "$ARM" in'):script.index("\nesac")]
    return re.findall(r"^ {4}([a-z0-9][a-z0-9-]*)\)", table, flags=re.M)


ARMS = case_table_arms(SUBMIT.read_text())


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
    method = "tuned-rlpd" if "tuned-rlpd" in arm else "residual"
    expected = {**SHARED, **RECIPE[method], "task": TASK[arm],
                "wandb.project": WANDB[arm][0], "wandb.group": WANDB[arm][1],
                "wandb.name": f"{arm}-1", "wandb.notes": f"{SEED_GROUP}/{arm}/seed1",
                **({"agent.actor.actor_last_layer_init_scale": INIT[arm]} if arm in INIT else {})}
    if arm.startswith("equi-"):
        expected.update(EQUI_DESIGN)
    else:
        expected["equivariance"] = None
    for path, value in expected.items():
        assert flat.get(path, "absent") == value, f"{arm}: {path} = {flat.get(path, 'absent')}"
    assert flat["offline_data.name"] == f"ankile/robomimic-mh-{TASK[arm].lower()}-image"
    assert "," not in flat["wandb.notes"], "hydra splits on commas"


@pytest.mark.parametrize("pair", PAIRS, ids=[b for _, b in PAIRS])
def test_a_tasks_arms_differ_in_the_stack_only(pair, tmp_path):
    resfit, equi = (_flat(_compose(_overrides(arm, tmp_path))) for arm in pair)
    differing = {k for k in resfit.keys() | equi.keys() if resfit.get(k) != equi.get(k)}
    allowed = {k for k in differing if k == STACK or k.startswith(STACK + ".")}
    allowed |= {"wandb.name", "wandb.notes", "wandb.project", "wandb.group"}
    if pair[1] in INIT and INIT[pair[0]] != INIT[pair[1]]:
        allowed.add("agent.actor.actor_last_layer_init_scale")
    assert differing - allowed == set(), f"{pair}: the recipe differs in {sorted(differing - allowed)}"


def test_zero_init_differs_from_its_pair_in_the_init_only(tmp_path):
    ours, zero = (_flat(_compose(_overrides(arm, tmp_path)))
                  for arm in ("equi-residual-square", "equi-residual-square-zero-init"))
    differing = {k for k in ours.keys() | zero.keys() if ours.get(k) != zero.get(k)}
    assert differing == {"agent.actor.actor_last_layer_init_scale", "wandb.name", "wandb.notes"}


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


EQUI_ARMS = [arm for arm in ARMS if arm.startswith("equi-")]


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


@pytest.mark.gpu
@pytest.mark.slow
def test_zero_init_residual_starts_at_zero_and_leaves_it(tmp_path, device):
    """Its step-0 action is exactly the base policy's, and the first actor step moves it.

    The other equivariant residual arm's actor has the same modules and shapes, and only that
    arm's equivariance is checked on a nonzero output."""
    import torch

    from tools.agent_fingerprint import STDDEV, fixed_batch

    zero = _agent(_compose(_overrides("equi-residual-square-zero-init", tmp_path)), device)
    ours = _agent(_compose(_overrides("equi-residual-square", tmp_path)), device)
    assert {k: v.shape for k, v in zero.state_dict().items()} == \
           {k: v.shape for k, v in ours.state_dict().items()}

    def residual(agent):
        batch = fixed_batch().to(device)
        keys = [*agent.rl_cameras, "observation.state", "observation.base_action"]
        obs = {k: batch["obs"][k] for k in keys}
        agent.train(False)
        with torch.no_grad():
            feat, _ = agent.enc(obs)
            out = agent.actor(feat, std=None)
        agent.train(True)
        return out

    assert torch.equal(residual(zero), torch.zeros_like(residual(zero)))
    for k in range(4):  # the fourth updates the actor, as in training
        metrics = zero.update(fixed_batch(weights=True).to(device), stddev=STDDEV, update_actor=k == 3)
    moved = residual(zero)
    assert torch.isfinite(moved).all() and moved.abs().max().item() > 0, "the residual did not move"
    assert math.isfinite(metrics["train/actor_loss_base"])
