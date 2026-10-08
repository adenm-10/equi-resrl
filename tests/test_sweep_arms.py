"""The arms submit.sh launches, exactly as it launches them.

A queued arm that fails at launch loses its GPU slot. So every arm in
submit.sh's case table is checked here first:

  Tier 1  its overrides are the pre-registered ones, set no key twice, and
          compose through train_residual_td3's Hydra configs to the
          pre-registered recipe; each arm differs from the fixed-env baselines'
          equi-tuned-rlpd-can (eyyi99q4) in the critic's norm only, and from
          ResFiT's tuned RLPD in the stack only; the smoke mode is the arm,
          briefly; and every arm launches.
  Tier 2  its agent builds on the GPU and runs one TD3 update, with finite
          losses, on the batch its method trains on; its actor is equivariant
          and its critic invariant, at the arm's own sizes; and every input its
          critic and actor are given can move their outputs. The FieldNorm
          critic it replaces could not (2026-10-08 FINDING), and a test here
          keeps that on record.

The arm list is pinned to the pre-registration (EXPERIMENTS.md, 2026-10-08
PRE-REGISTERED, "equivariant tuned RLPD on Can without FieldNorm"): tuned RLPD
on Can in the equivariant stack at Aden's design, with no norm in the critic,
and with RegularLayerNorm in it.
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

SEED_GROUP = "critic-norm-can-1008"

# What each arm launches, less the run's name, notes, project and group.
COMMON = {"algo.total_timesteps=150000", "read_buffer_caches=false", "write_buffer_caches=false",
          "seed=1", "debug=false"}
DESIGN = {"equivariance.residual_squash=true", "agent.critic.num_q=10",
          "equivariance.num_critic_layers=2", "equivariance.actor_norm=none"}
CRITIC_NORM = {"equi-tuned-rlpd-can-nonorm": "none", "equi-tuned-rlpd-can-layernorm": "layer_norm"}
LAUNCHED = {
    arm: {"--config-name=equi_tuned_rlpd_can_config", *DESIGN, f"equivariance.critic_norm={norm}",
          *COMMON}
    for arm, norm in CRITIC_NORM.items()
}
# The fixed-env baselines' tuned-RLPD arms as 9cf207d launched them (eyyi99q4, d047it6c), less naming.
BASELINES = {
    "equi-tuned-rlpd-can": ["--config-name=equi_tuned_rlpd_can_config", *sorted(DESIGN),
                            "equivariance.critic_norm=field_norm", *sorted(COMMON)],
    "resfit-tuned-rlpd-can": ["--config-name=tuned_rlpd_can_config", *sorted(COMMON)],
}
NAMING = ("wandb.name", "wandb.notes", "wandb.project", "wandb.group")
# Where the tuned-RLPD arms have gone before (Aden, 2026-10-07).
WANDB = ("robomimic-can-final", "tuned-rlpd")

# The composed config every arm must have: the fixed-env baselines' tuned-RLPD recipe, and the
# equivariant design with its critic norm replaced.
SHARED = {
    "algo.total_timesteps": 150_000, "read_buffer_caches": False, "write_buffer_caches": False,
    "agent.critic.num_q": 10, "algo.n_step": 3, "algo.gamma": 0.99, "algo.batch_size": 256,
    "algo.offline_fraction": 0.5, "algo.num_updates_per_iteration": 4, "algo.buffer_size": 200_000,
    "algo.learning_starts": 10_000, "num_envs": 1, "eval_first": True,
    "eval_interval_every_steps": 10_000, "eval_num_episodes": 50, "seed": 1, "debug": False,
    "wandb.mode": "online", "task": "Can", "offline_data.name": "ankile/robomimic-mh-can-image",
}
RECIPE = {
    "base_policy": None, "algo.critic_warmup_steps": 0, "agent.actor_lr": 1e-4,
    "agent.critic_target_tau": 0.01, "algo.stddev_min": 0.1, "algo.stddev_max": 0.1,
    "agent.actor.action_scale": 1.0, "algo.random_action_noise_scale": 1.0,
    "algo.use_base_policy_for_warmup": False, "agent.actor.actor_last_layer_init_scale": None,
}
EQUI_DESIGN = {
    "equivariance.N": 8, "equivariance.use_equivariant_model": True,
    "equivariance.residual_squash": True, "equivariance.num_critic_layers": 2,
    "equivariance.actor_norm": "none", "equivariance.actor_degree_channel": 128,
    "equivariance.critic_degree_channel": 128, "equivariance.frozen_target_critic": False,
    "equivariance.float_images_as_uint8": False,
}
STACK = "equivariance"


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
    expected = {**SHARED, **RECIPE, **EQUI_DESIGN, "equivariance.critic_norm": CRITIC_NORM[arm],
                "wandb.project": WANDB[0], "wandb.group": WANDB[1],
                "wandb.name": f"{arm}-1", "wandb.notes": f"{SEED_GROUP}/{arm}/seed1"}
    for path, value in expected.items():
        assert flat.get(path, "absent") == value, f"{arm}: {path} = {flat.get(path, 'absent')}"
    assert "," not in flat["wandb.notes"], "hydra splits on commas"


def _differing(a: dict, b: dict) -> set[str]:
    return {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)} - set(NAMING)


@pytest.mark.parametrize("arm", ARMS)
def test_arm_differs_from_the_baselines_equivariant_arm_in_the_critic_norm_only(arm, tmp_path):
    """eyyi99q4's launch, with only its critic's FieldNorm replaced."""
    ours = _flat(_compose(_overrides(arm, tmp_path)))
    theirs = _flat(_compose(BASELINES["equi-tuned-rlpd-can"]))
    assert _differing(ours, theirs) == {"equivariance.critic_norm"}


@pytest.mark.parametrize("arm", ARMS)
def test_arm_differs_from_resfits_tuned_rlpd_in_the_stack_only(arm, tmp_path):
    ours = _flat(_compose(_overrides(arm, tmp_path)))
    resfit = _flat(_compose(BASELINES["resfit-tuned-rlpd-can"]))
    outside = {k for k in _differing(ours, resfit) if k != STACK and not k.startswith(STACK + ".")}
    assert outside == set(), f"{arm}: the recipe differs from ResFiT's in {sorted(outside)}"


def test_the_two_arms_differ_in_the_critic_norm_only(tmp_path):
    a, b = (_flat(_compose(_overrides(arm, tmp_path))) for arm in ARMS)
    assert _differing(a, b) == {"equivariance.critic_norm"}


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


EQUI_ARMS = ARMS  # every arm is equivariant


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
    """Per input change, the largest change in Q (mean over heads) and in the actor's mean action."""
    import torch

    from tools.agent_fingerprint import fixed_batch

    torch.manual_seed(0)
    agent = _agent(cfg, device)
    agent.train(False)  # QAgent.train returns None

    def outputs(change=None):
        batch = fixed_batch(base_action=False).to(device)
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
@pytest.mark.parametrize("arm", ARMS)
def test_every_input_can_move_the_critic_and_the_actor(arm, tmp_path, device):
    """Invariance tests pass on a network that ignores an input, so this checks what they cannot."""
    changes = _output_changes(_compose(_overrides(arm, tmp_path)), device)
    blind_q = {k: f"{dq:.1e}" for k, (dq, _) in changes.items() if dq <= SEES}
    assert not blind_q, f"{arm}: Q does not move with {blind_q}"
    blind_pi = {k: f"{dpi:.1e}" for k, (_, dpi) in changes.items()
                if not k.startswith("action") and dpi <= SEES}
    assert not blind_pi, f"{arm}: the actor does not move with {blind_pi}"


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
