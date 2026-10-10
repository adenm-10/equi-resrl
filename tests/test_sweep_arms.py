"""The arms submit.sh launches, exactly as it launches them.

A queued arm that fails at launch loses its GPU slot. So every arm in
submit.sh's case table is checked here first:

  Tier 1  its overrides are the pre-registered ones and set no key twice; it
          composes through train_residual_td3's Hydra configs to the anchor's
          config (vc5o977f's launch, equi-tuned-rlpd-square-rerendered 1 at
          eb90a0c) but for its one change and, for the two normed arms, the
          scale of each critic head's last Linear at init; the anchor is what
          vc5o977f launched with; the dataset is built; the smoke mode is the
          arm, briefly; and every arm launches.
  Tier 2  its agent builds on the GPU and runs one TD3 update, with finite
          losses; its actor is equivariant and its critic invariant, at the
          arm's own sizes; every input its critic and actor are given can move
          their outputs (the FieldNorm critic of the fixed-env baselines could
          not, 2026-10-08 FINDING, and a test here keeps that on record); the
          normed arms' critics start as spread as the anchor's, on the
          pre-registered measurement; and each norm sits where it is meant to.

The arm list is pinned to the pre-registration (EXPERIMENTS.md, 2026-10-10
PRE-REGISTERED, "stabilizing equivariant tuned RLPD on Square").
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

SEED_GROUP = "square-trlpd-stab-1010"

CLIPQ = "equi-tuned-rlpd-square-clipq"
POOLED_LN = "equi-tuned-rlpd-square-pooledln"
LN = "equi-tuned-rlpd-square-ln"
TEN_BY_TWO = "equi-tuned-rlpd-square-10x2"
CONFIG = "equi_tuned_rlpd_square_config"
DEMOS = "local/robomimic-mh-square-rerendered"
# Each normed arm's critic heads' last Linear at init, so that its critic starts as spread as the
# anchor's: tools/critic_start.py, real batch, 20 seeds (the pre-registration's table).
POOLED_LN_SCALE = 0.8835
LN_SCALE = 0.8168

# What every arm launches, less the run's name, notes, project and group.
COMMON = {"algo.total_timesteps=300000", "read_buffer_caches=false", "write_buffer_caches=false",
          "seed=1", "debug=false"}
# vc5o977f's launch at eb90a0c, the anchor every arm is one change from.
ANCHOR = {f"--config-name={CONFIG}", "equivariance.residual_squash=true", "agent.critic.num_q=5",
          f"offline_data.name={DEMOS}", *COMMON}
LAUNCHED = {
    CLIPQ: ANCHOR | {"agent.clip_q_target_to_reward_range=true"},
    POOLED_LN: ANCHOR | {"equivariance.critic_pooled_norm=layer_norm",
                         f"equivariance.critic_last_layer_init_scale={POOLED_LN_SCALE}"},
    LN: ANCHOR | {"equivariance.actor_norm=layer_norm", "equivariance.critic_norm=layer_norm",
                  f"equivariance.critic_last_layer_init_scale={LN_SCALE}"},
    TEN_BY_TWO: (ANCHOR - {"agent.critic.num_q=5"}) | {"agent.critic.num_q=10",
                                                         "equivariance.num_critic_layers=2"},
}
# Each arm's changes from the anchor, as composed paths.
FROM_ANCHOR = {
    CLIPQ: {"agent.clip_q_target_to_reward_range"},
    POOLED_LN: {"equivariance.critic_pooled_norm", "equivariance.critic_last_layer_init_scale"},
    LN: {"equivariance.actor_norm", "equivariance.critic_norm", "equivariance.critic_last_layer_init_scale"},
    TEN_BY_TWO: {"agent.critic.num_q", "equivariance.num_critic_layers"},
}
# The fixed-env baselines' equi-tuned-rlpd-can (eyyi99q4) as 9cf207d launched it, less naming.
BASELINES = {
    "equi-tuned-rlpd-can": ["--config-name=equi_tuned_rlpd_can_config", "equivariance.residual_squash=true",
                            "agent.critic.num_q=10", "equivariance.num_critic_layers=2",
                            "equivariance.critic_norm=field_norm", "equivariance.actor_norm=none",
                            "algo.total_timesteps=150000", "read_buffer_caches=false",
                            "write_buffer_caches=false", "seed=1", "debug=false"],
}
NAMING = ("wandb.name", "wandb.notes", "wandb.project", "wandb.group")
WANDB = ("robomimic-square-final", "tuned-rlpd")
# Under CACHE_DIR, where every run package lives: a launch worktree has no outputs/.
VC5O977F = RUNS_ROOT / "robomimic-square-final/vc5o977f/manifest.json"

# The composed config every arm must have.
SHARED = {
    "task": "Square", "offline_data.name": DEMOS, "algo.total_timesteps": 300_000,
    "read_buffer_caches": False, "write_buffer_caches": False, "algo.n_step": 3, "algo.gamma": 0.99,
    "algo.batch_size": 256, "algo.offline_fraction": 0.5, "algo.num_updates_per_iteration": 4,
    "algo.buffer_size": 200_000, "algo.learning_starts": 10_000, "num_envs": 1, "eval_first": True,
    "eval_interval_every_steps": 10_000, "eval_num_episodes": 50, "seed": 1, "debug": False,
    "wandb.mode": "online", "encode_env_frames_like_demos": False,
    "equivariance.N": 8, "equivariance.use_equivariant_model": True, "equivariance.residual_squash": True,
    "equivariance.encoder_norm": "none", "equivariance.frozen_target_critic": False,
    "equivariance.float_images_as_uint8": False,
    "base_policy": None, "offline_data.use_base_policy_for_base_actions": False,
    "algo.critic_warmup_steps": 0, "agent.actor_lr": 1e-4, "agent.critic_target_tau": 0.01,
    "algo.stddev_min": 0.1, "algo.stddev_max": 0.1, "agent.actor.action_scale": 1.0,
    "algo.random_action_noise_scale": 1.0, "algo.use_base_policy_for_warmup": False,
    "agent.actor.actor_last_layer_init_scale": None, "agent.critic.min_q_heads": 2,
    "agent.critic.policy_gradient_type": "ensemble_mean", "agent.critic_grad_clip_norm": 1.0,
}


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
    expected = {**SHARED, "wandb.project": WANDB[0], "wandb.group": WANDB[1],
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


@pytest.mark.skipif(not VC5O977F.exists(), reason="vc5o977f's run package is not on this machine")
def test_the_anchor_is_what_vc5o977f_launched_with():
    argv = json.loads(VC5O977F.read_text())["argv"][1:]
    launched = {o for o in argv if o.split("=", 1)[0] not in (*NAMING, "hydra.run.dir")}
    assert launched == ANCHOR


@pytest.mark.parametrize("arm", ARMS)
def test_the_dataset_each_arm_loads_is_built(arm):
    """A local dataset the trainer cannot find is a launch that dies at startup."""
    root = LEROBOT_HOME / DEMOS
    assert (root / "meta/info.json").exists() and (root / "meta/rerender.json").exists(), root


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
@pytest.mark.slow
@pytest.mark.parametrize("arm", [POOLED_LN, LN])
def test_the_normed_arms_critics_start_as_spread_as_the_anchors(arm, tmp_path, device):
    """The pre-registered measurement (tools/critic_start.py, real batch, 20 seeds): with its last
    Linear scaled, each normed arm's heads start as spread as the anchor's. Scaling the last Linear
    by s scales every head's Q by s, so on the same batch and seeds the match is exact but for the
    scale's rounding."""
    from tools.critic_start import measure, real_batch

    batch = real_batch(DEMOS, 256)
    anchor = measure(_compose(sorted(ANCHOR, key=lambda o: not o.startswith("--config-name"))), batch,
                     str(device), 20)
    ours = measure(_compose(_overrides(arm, tmp_path)), batch, str(device), 20)
    assert abs(ours["spread"] / anchor["spread"] - 1) < 0.01, (ours, anchor)


@pytest.mark.gpu
@pytest.mark.parametrize("arm", ARMS)
def test_each_arms_norms_sit_where_they_are_meant_to(arm, tmp_path, device):
    """pooledln: one TrivialLayerNorm per head, after its pooling, and no other norm. ln:
    RegularLayerNorm after every hidden Linear of the actor (visual projection, trunk, hidden
    layer) and the critic (visual projection, trunk, each head's one), where ResFiT's
    use_layer_norm puts LayerNorm, and none after pooling. clipq and 10x2: no norm anywhere, and
    10x2's ten heads have two hidden Linears each."""
    from escnn import nn as enn

    from resfit.rl_finetuning.equi_off_policy.rl.equi_rl_utils import (
        FieldNorm,
        RegularInnerBatchNorm,
        RegularLayerNorm,
        TrivialLayerNorm,
    )

    agent = _agent(_compose(_overrides(arm, tmp_path)), device)

    def count(module, *classes):
        return sum(isinstance(m, classes) for m in module.modules())

    others = (enn.IIDBatchNorm1d, RegularInnerBatchNorm, FieldNorm)
    heads = agent.critic.q_ensemble.heads
    assert count(agent.critic, *others) == count(agent.actor, *others) == count(agent.enc, *others) == 0
    expected = {  # (critic RegularLayerNorm, actor RegularLayerNorm, TrivialLayerNorm, heads, Linears per head)
        CLIPQ: (0, 0, 0, 5, 2), POOLED_LN: (0, 0, 5, 5, 2), LN: (2 + 5, 3, 0, 5, 2), TEN_BY_TWO: (0, 0, 0, 10, 3),
    }[arm]
    got = (count(agent.critic, RegularLayerNorm), count(agent.actor, RegularLayerNorm),
           count(agent.critic, TrivialLayerNorm), len(heads), count(heads[0], enn.Linear))
    assert got == expected, f"{arm}: {got}"
    if arm == POOLED_LN:
        assert all(isinstance(head.net[-2], TrivialLayerNorm) for head in heads)


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
