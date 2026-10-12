"""agent.actor.action_bound: ResFiT's residual given the equivariant stack's action limits.

`box` is ResFiT's own, and every ResFiT run before 2026-10-12 ran it: the residual squashed per
axis to ±action_scale in the scaler's units, the sum the critic scores clamped per axis to [-1, 1]
in those units, and the executed action clamped to the demos' range widened by action_scale.
`equivariant` is resfit-residual-square-limits' (EXPERIMENTS.md, 2026-10-12 PRE-REGISTERED): the
equivariant stack's three limits, in env units.

Tier 2, on the GPU (ResFiT's ViT encoder synchronizes CUDA in its forward):
  - `box` leaves ResFiT's agent and scaler as they were: parameters, the eval action, the critic's
    diagnostics and one TD3 update, against goldens captured before the field existed;
  - under `equivariant`, the residual's mean is inside a circle of radius action_scale on xy and
    on the tilt pair (rx, ry), and within ±action_scale on z, yaw and gripper, all in env units;
  - the critic scores the sum clipped as equi_clip clips it in env units, which commutes with
    rotating both pairs;
  - the executed action is the sum, unclamped, for robosuite to clip to its box, as the
    equivariant stack's IdentityScaler leaves it;
  - at one seed both bounds start from the same weights;
  - the field is refused where it means nothing.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import pathlib
import subprocess
import sys

import pytest
import torch

from tools.agent_fingerprint import (
    ACTION_DIM,
    CAMERAS,
    OBS_SHAPE,
    STATE_DIM,
    STDDEV,
    UPDATE_SEED,
    _flat_params,
    _reduce,
    _seed,
    _stats,
    fixed_batch,
)

REPO = pathlib.Path(__file__).resolve().parent.parent
PARTS = ("encoders", "actor", "critic")
# Not the configs' 0.0, at which the actor outputs exact zeros and its forward would pin nothing.
INIT_SCALE = 0.1
RTOL = 1e-6
PAIRS = {"xy": slice(0, 2), "tilt": slice(3, 5)}  # the irrep(1) pairs of one arm's action
SCALARS = [2, 5, 6]                               # z, yaw, gripper

# Captured on 2026-10-12 from the tree at 8ce1642, before action_bound existed, on both of this
# machine's GPUs (identical in every value), by running this file. The tree with the field gives
# the same values, exactly. If these move, ResFiT's agent changed: fix the change, or record it in
# docs/EXPERIMENTS.md.
GOLDEN = {
    "agent": {
        "params": {"encoders": [779.4234960767983, -14.0625382150271],
                   "actor": [2173.1806874263743, 34.2209361776182],
                   "critic": [23508.895337534723, -471.1228556662111]},
        "act": [-0.9900615755468607, -0.48791498113294696],
        "diagnostics": [0.7282002568244934, 0.6145258545875549, -0.03159486502408981,
                        -0.11577405035495758, -0.039512425661087036, -0.09900036454200745],
        "update": {"critic_loss": 0.4870557188987732, "actor_loss_base": -0.19023537635803223,
                   "encoders_delta": [-0.08286876801335774, -0.06596509794965014],
                   "actor_delta": [0.07525264423838962, 0.0004131924422519442],
                   "critic_delta": [-5.983681928855809, 0.47111084640869344]},
    },
    "scaler": {"scale": [-12.883776843547821, -2.504741446652548],
               "unscale": [-22.309037446975708, -4.9678202064484305]},
}


def _anchor_cfg(device, action_bound: str = "box"):
    """oj5k0jqb's agent (resfit-residual-square 1), as resfit-equi-diff-1012's arms compose it."""
    from hydra import compose, initialize

    import resfit.rl_finetuning.config.residual_td3  # noqa: F401  registers the configs

    with initialize(version_base=None, config_path=None):
        cfg = compose(config_name="residual_td3_square_config",
                      overrides=["agent.critic.num_q=10", "base_policy.wt_version=v9",
                                 f"agent.actor.action_bound={action_bound}"])
    cfg.agent.device = str(device)
    cfg.agent.actor.actor_last_layer_init_scale = INIT_SCALE
    return cfg


def _off_centre_stats(n: int) -> dict:
    """Demo statistics whose middle is off zero on every axis, as Square's are (rx [-0.44, 0.22]), so
    that a test of the env-unit bound would see a missing or wrong offset. _stats is centred."""
    a = torch.arange(n, dtype=torch.float32).numpy()
    return {"min": -0.4 + 0.02 * a, "max": 0.6 + 0.12 * a}  # middles 0.1 to 0.52


def _scaler(cfg, device, stats=None):
    """The trainer's scaler, on stand-in demo statistics spread per axis."""
    from resfit.rl_finetuning.utils.normalization import ActionScaler

    stats = _stats(ACTION_DIM) if stats is None else stats
    return ActionScaler.from_dataset_stats(stats, action_scale=cfg.agent.actor.action_scale,
                                           min_range_per_dim=cfg.offline_data.min_action_range, device=device,
                                           clamp_unscaled=cfg.agent.actor.action_bound == "box")


def _agent(cfg, device, seed: int = 0, stats=None, **kwargs):
    from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent

    kwargs = {"action_scaler": _scaler(cfg, device, stats), "residual_actor": True, **kwargs}
    _seed(seed)
    with contextlib.redirect_stdout(sys.stderr):  # QAgent prints its modules
        agent = QAgent(obs_shape=OBS_SHAPE, prop_shape=(STATE_DIM,), action_dim=ACTION_DIM,
                       rl_cameras=list(CAMERAS), cfg=cfg.agent, **kwargs)
    return agent.to(device)


def _reduce_cpu(t: torch.Tensor) -> list[float]:
    return _reduce(t.cpu())


def _resfit_fingerprint(agent, device) -> dict:
    """Parameters; the eval action and the critic's diagnostics, which combine the residual with
    the base action as the actor loss does; and what one update does. Leaves the agent in train
    mode, with one update applied."""
    before = {part: _flat_params(getattr(agent, part)) for part in PARTS}
    fp = {"params": {part: _reduce_cpu(before[part]) for part in PARTS}}

    agent.train(False)
    with torch.no_grad():
        fp["act"] = _reduce_cpu(agent.act(fixed_batch()["obs"].to(device), eval_mode=True))
        diagnostics = agent.critic_diagnostics(fixed_batch().to(device), num_online=2)
    fp["diagnostics"] = [diagnostics[k] for k in sorted(diagnostics)]

    agent.train(True)
    _seed(UPDATE_SEED)
    metrics = agent.update(fixed_batch().to(device), stddev=STDDEV, update_actor=True)
    fp["update"] = {
        "critic_loss": metrics["train/critic_loss"],
        "actor_loss_base": metrics["train/actor_loss_base"],
        **{f"{part}_delta": _reduce_cpu(_flat_params(getattr(agent, part)) - before[part]) for part in PARTS},
    }
    return fp


def _scaler_fingerprint(scaler, device) -> dict:
    """scale and unscale on actions inside and outside the widened demo range."""
    g = torch.Generator().manual_seed(3)
    action = (torch.randn(64, ACTION_DIM, generator=g) * 1.5).to(device)
    return {"scale": _reduce_cpu(scaler.scale(action)), "unscale": _reduce_cpu(scaler.unscale(action))}


def _flatten(fp) -> list[float]:
    if isinstance(fp, dict):
        return [x for key in sorted(fp) for x in _flatten(fp[key])]
    if isinstance(fp, (list, tuple)):
        return [x for item in fp for x in _flatten(item)]
    return [float(fp)]


def _env_units(agent, action: torch.Tensor) -> torch.Tensor:
    """The agent's units to the env's, as ActionScaler.unscale maps them, without its clamp."""
    return agent.action_mid + action * agent.action_half_range


def _rotate(env_action: torch.Tensor, theta: float) -> torch.Tensor:
    """Both irrep(1) pairs rotated by theta, the scalars left: the group's action on one arm's action."""
    c, s = math.cos(theta), math.sin(theta)
    out = env_action.clone()
    for sl in PAIRS.values():
        x, y = env_action[..., sl.start], env_action[..., sl.start + 1]
        out[..., sl.start], out[..., sl.start + 1] = c * x - s * y, s * x + c * y
    return out


# ---------------------------------------------------------------------------
# box: ResFiT's agent and scaler as they were
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.slow
def test_box_leaves_resfits_agent_and_scaler_as_they_were():
    """Measured in its own process, in PyTorch's deterministic mode, which is global."""
    env = {**os.environ, "PYTHONPATH": f"{REPO}:{os.environ.get('PYTHONPATH', '')}"}
    out = subprocess.run([sys.executable, str(pathlib.Path(__file__))], cwd=REPO, env=env,
                         capture_output=True, text=True, check=True).stdout
    measured = json.loads(out.strip().splitlines()[-1])
    assert _flatten(measured) == pytest.approx(_flatten(GOLDEN), rel=RTOL)


@pytest.mark.gpu
def test_box_combines_as_resfit_clamped(device):
    agent = _agent(_anchor_cfg(device), device)
    g = torch.Generator().manual_seed(4)
    base, residual = (torch.randn(2, 256, ACTION_DIM, generator=g) * 1.2).to(device)
    assert torch.equal(agent.combine_actions(base, residual), torch.clamp(base + residual, -1.0, 1.0))


# ---------------------------------------------------------------------------
# equivariant: the equivariant stack's three limits, in env units
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_the_residuals_mean_is_inside_the_equivariant_stacks_bound_in_env_units(device):
    """With a last layer large enough to saturate the squash, so the bound is reached, not just
    respected: equi_tanh in env units, scaled by action_scale (0.1), as the equivariant actor."""
    from resfit.rl_finetuning.equi_off_policy.rl.equi_rl_utils import equi_tanh

    cfg = _anchor_cfg(device, "equivariant")
    agent = _agent(cfg, device, stats=_off_centre_stats(ACTION_DIM))
    last = agent.actor.policy[-1]
    assert isinstance(last, torch.nn.Linear), "the squash is in forward, not a final Tanh"
    with torch.no_grad():
        last.weight.normal_(0.0, 1.0)
    agent.train(False)

    scale = cfg.agent.actor.action_scale
    obs = fixed_batch()["obs"].to(device)
    with torch.no_grad():
        residual = agent.act(obs, eval_mode=True, cpu=False)
        residual_env = residual * agent.action_half_range  # a residual has no offset
        for name, sl in PAIRS.items():
            norm = residual_env[:, sl].norm(dim=-1)
            assert (norm <= scale * (1 + 1e-5)).all(), f"{name}: {norm.max():.4f} > {scale}"
            assert (norm >= 0.9 * scale).any(), f"{name}: the bound is never approached ({norm.max():.4f})"
        assert (residual_env[:, SCALARS].abs() <= scale * (1 + 1e-5)).all()

        # And it is exactly the equivariant actor's map, applied to this actor's pre-squash output.
        obs = dict(obs.items())
        obs["feat"] = agent._encode(obs, augment=False)
        feat = agent.actor.compress(obs["feat"].flatten(1, -1))
        pre = agent.actor.policy(torch.cat([feat, obs["observation.state"], obs["observation.base_action"]], -1))
        torch.testing.assert_close(residual_env, scale * equi_tanh(pre, agent.action_layout))


@pytest.mark.gpu
def test_the_critic_scores_the_sum_clipped_as_equi_clip_clips_it_in_env_units(device):
    """In env units: each pair inside the unit circle and each scalar within ±1; a sum already inside
    is left as it is, a pair outside keeps its direction; and the clip commutes with rotating
    both pairs, so the critic's view of the bound is equivariant."""
    agent = _agent(_anchor_cfg(device, "equivariant"), device, stats=_off_centre_stats(ACTION_DIM))
    assert (agent.action_mid.abs() > 0.05).all()
    g = torch.Generator().manual_seed(5)
    base, residual = (torch.randn(2, 512, ACTION_DIM, generator=g) * 1.2).to(device)
    combined = agent.combine_actions(base, residual)
    env_sum, env_out = _env_units(agent, base + residual), _env_units(agent, combined)

    for name, sl in PAIRS.items():
        before, after = env_sum[:, sl], env_out[:, sl]
        inside = before.norm(dim=-1) <= 1.0
        assert inside.any() and (~inside).any(), f"{name}: the batch does not cross the bound"
        torch.testing.assert_close(after[inside], before[inside])
        torch.testing.assert_close(after[~inside].norm(dim=-1), torch.ones_like(after[~inside, 0]))
        torch.testing.assert_close(after[~inside], before[~inside] / before[~inside].norm(dim=-1, keepdim=True))
    torch.testing.assert_close(env_out[:, SCALARS], env_sum[:, SCALARS].clamp(-1.0, 1.0))

    # Outside the widened demo range in the agent's units, which box would have clamped.
    assert (combined.abs() > 1.0).any()

    for theta in (math.pi / 4, 2.0):
        rotated_sum = _rotate(env_sum, theta)
        rotated_in_agent_units = (rotated_sum - agent.action_mid) / agent.action_half_range
        out = _env_units(agent, agent.combine_actions(rotated_in_agent_units, torch.zeros_like(base)))
        torch.testing.assert_close(out, _rotate(env_out, theta), atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
def test_the_executed_action_is_the_sum_unclamped_for_the_env_to_clip_to_its_box(device):
    """What the env wrapper executes: unscale(base + residual). Unclamped, as the equivariant stack's
    IdentityScaler leaves it; robosuite clips it to [-1, 1], its controllers' input range."""
    stats = _off_centre_stats(ACTION_DIM)
    box = _scaler(_anchor_cfg(device), device, stats)
    ours = _scaler(_anchor_cfg(device, "equivariant"), device, stats)
    g = torch.Generator().manual_seed(6)
    action = (torch.randn(256, ACTION_DIM, generator=g) * 2).to(device)
    mid = (ours.limits.max + ours.limits.min) / 2
    half = (ours.limits.max - ours.limits.min) / 2
    torch.testing.assert_close(ours.unscale(action), mid + action * half)
    torch.testing.assert_close(box.unscale(action), mid + action.clamp(-1.0, 1.0) * half)
    assert not torch.allclose(box.unscale(action), ours.unscale(action))
    # Both scale as ResFiT does: the base action reaches the agent clamped to the demos' range.
    torch.testing.assert_close(ours.scale(action), box.scale(action))
    assert ours.to("cpu").clamp_unscaled is False

    import robosuite

    # The installed robosuite's, which a launch worktree runs too.
    config = pathlib.Path(robosuite.__file__).parent / "controllers/config/robots/default_panda.json"
    controllers = json.loads(config.read_text())
    arm = controllers["body_parts"]["arms"]["right"]
    assert (arm["type"], arm["input_min"], arm["input_max"]) == ("OSC_POSE", -1, 1)


@pytest.mark.gpu
def test_both_bounds_start_from_the_same_weights(device):
    """The bound draws no random number and adds no weight, so at one seed the networks are the
    same: the change is the bound alone."""
    box = _agent(_anchor_cfg(device), device, seed=1).state_dict()
    ours = _agent(_anchor_cfg(device, "equivariant"), device, seed=1).state_dict()
    assert box.keys() == ours.keys()
    for key, value in ours.items():
        assert torch.equal(value, box[key]), key


@pytest.mark.gpu
def test_the_field_is_refused_where_it_means_nothing(device):
    from resfit.rl_finetuning.utils.normalization import ActionScaler

    cfg = _anchor_cfg(device, "equivariant")
    with pytest.raises(ValueError, match="bounds a residual"):
        _agent(cfg, device, residual_actor=False)
    with pytest.raises(ValueError, match="needs the run's ActionScaler"):
        _agent(cfg, device, action_scaler=None)
    clamping = ActionScaler.from_dataset_stats(_stats(ACTION_DIM), action_scale=0.1, device=device)
    with pytest.raises(ValueError, match="clamp_unscaled=False"):
        _agent(cfg, device, action_scaler=clamping)
    cfg.agent.actor.action_bound = "disk"
    with pytest.raises(ValueError, match="'box' or 'equivariant'"):
        _agent(cfg, device)

    from tools.agent_fingerprint import build_reference_agent

    with pytest.raises(ValueError, match="ResFiT's agent's"):
        build_reference_agent(str(device), overrides={"agent.actor.action_bound": "equivariant"})


if __name__ == "__main__":
    # Prints the goldens for the tree on PYTHONPATH, in PyTorch's deterministic mode, so that two
    # trees measured on one GPU can be compared exactly.
    from tools.agent_fingerprint import _deterministic_cuda

    _deterministic_cuda()
    device = torch.device("cuda")
    cfg = _anchor_cfg(device)
    print(json.dumps({"agent": _resfit_fingerprint(_agent(cfg, device), device),
                      "scaler": _scaler_fingerprint(_scaler(cfg, device), device)}))
