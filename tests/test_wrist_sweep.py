"""The wrist sweep's agent options (EXPERIMENTS.md, 2026-11-01 CODE; docs/plans/2026-10-31-wrist-sweep.md): the
residual without the agentview camera, ResFiT's critic over flat features, the gripper's xy dropped, the hand's
heading frame (rl/heading_frame.py) and the C8 wrist encoder.

- split_cameras reads the cameras by name, and the agent refuses each mix it cannot build and each new field no
  network of the mix reads.
- The flat critic is off_policy's Critic built with one token and fuse_patch=0, bit for bit.
- Without the agentview camera the equivariant agent (NA-eee) is still equivariant under the world's turn about
  the base, at every element of C8, checked from the physics: the state's xy turned about the base, its rotation
  turned, the base action's xy and rx_ry pairs turned, the wrist image unchanged.
- In the heading frame (G-cnn, G-c8), a turn of the scene and the hand together about the hand, by angles that
  are not C8's, leaves every network input unchanged, turns the actor's world-frame residual with it, and leaves
  Q unchanged, in eval and train mode.
- The C8 wrist encoder's output permutes as its regular fields declare under the crop's 90-degree turns.

The rotation and the turns are written here from the physics, not imported from the code under test.
"""

from __future__ import annotations

import contextlib
import math
import sys

import pytest
import torch

import tests.group_action as ga

WRIST = ("observation.images.robot0_eye_in_hand",)
BOTH = ("observation.images.agentview", *WRIST)
BASE_XY = (-0.56, 0.0)  # tools/agent_fingerprint.py's ROTATION_CENTRE, the reference agent's
FLAT = {"equivariance.critic_tokens": "flat", "agent.critic.fuse_patch": 0}
RESFIT_NETS = {"agent.critic.num_q": 10, "equivariance.actor_family": "resfit", "equivariance.critic_family": "resfit"}
# The arms' agents at the reference config (tools/agent_fingerprint.py), each with its arm's options.
ARMS = {
    "na-eee": (WRIST, {"equivariance.actor_norm": "layer_norm", "equivariance.critic_norm": "layer_norm",
                       "agent.critic.num_q": 10, "equivariance.num_critic_layers": 2,
                       "equivariance.actor_degree_channel": 448}),
    "p2": (BOTH, {**RESFIT_NETS, **FLAT, "equivariance.encoder_family": "resfit", "equivariance.actor_norm": "none"}),
    "w-cnn": (WRIST, {**RESFIT_NETS, **FLAT, "equivariance.actor_norm": "none", "equivariance.ih_out_channel": 256,
                      "equivariance.gripper_xy_input": False}),
}
ARMS["g-cnn"] = (WRIST, {**ARMS["w-cnn"][1], "equivariance.action_frame": "heading"})
ARMS["g-c8"] = (WRIST, {**{k: v for k, v in ARMS["g-cnn"][1].items() if k != "equivariance.ih_out_channel"},
                        "equivariance.wrist_encoder": "c8"})


def _agent(arm: str, device="cpu"):
    from tools.agent_fingerprint import build_reference_agent

    cameras, overrides = ARMS[arm]
    return build_reference_agent(str(device), overrides=overrides, cameras=cameras)


def _batch(arm: str, device="cpu"):
    from tools.agent_fingerprint import fixed_batch

    return fixed_batch(cameras=ARMS[arm][0]).to(device)


def _turn(v: torch.Tensor, theta: float) -> torch.Tensor:
    """2-vectors [..., 2] turned counter-clockwise by theta."""
    c, s = math.cos(theta), math.sin(theta)
    return torch.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1]], dim=-1)


def _turned_quat(q: torch.Tensor, theta: float) -> torch.Tensor:
    """robosuite's (x, y, z, w) quaternion of R_z(theta) R: the product (cos theta/2, 0, 0, sin theta/2) q."""
    hw, hz = math.cos(theta / 2), math.sin(theta / 2)
    x, y, z, w = q.unbind(-1)
    return torch.stack([hw * x - hz * y, hw * y + hz * x, hw * z + hz * w, hw * w - hz * z], dim=-1)


def _turned_action(a: torch.Tensor, theta: float) -> torch.Tensor:
    """One arm's action with its xy (0:2) and rx_ry (3:5) pairs turned; z, rz and the hand kept."""
    return torch.cat([_turn(a[:, 0:2], theta), a[:, 2:3], _turn(a[:, 3:5], theta), a[:, 5:]], dim=1)


def _turned_obs(obs: dict, theta: float, about_base: bool) -> dict:
    """The scene and the hand turned together by theta about the base (the network's group) or about the hand
    (the heading frame's): the hand's rotation and the base action turn; about the base its xy turns too, about
    the hand it stays. The wrist camera turns with the hand, so its image is unchanged."""
    out = dict(obs)
    s = obs["observation.state"].clone()
    if about_base:
        base = torch.tensor(BASE_XY, dtype=s.dtype, device=s.device)
        s[:, 0:2] = _turn(s[:, 0:2] - base, theta) + base
    s[:, 3:7] = _turned_quat(s[:, 3:7], theta)
    out["observation.state"] = s
    out["observation.base_action"] = _turned_action(obs["observation.base_action"], theta)
    return out


def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def _features(agent, obs, training: bool, seed: int = 5):
    """The encoder's features, as tensors; the crop is drawn from `seed` in train mode."""
    from resfit.rl_finetuning.equi_off_policy.rl.equi_rl_utils import feature_tensor

    agent.train(training)
    torch.manual_seed(seed)
    actor_feat, critic_feat = agent.enc(dict(obs))
    return feature_tensor(actor_feat), feature_tensor(critic_feat)


# ---------------------------------------------------------------------------
# Every arm builds and updates, and the agent refuses what it cannot read
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arm", list(ARMS))
def test_each_arm_builds_and_updates(arm):
    """One update of each arm's agent on a batch of its cameras alone, and act() on one observation."""
    with contextlib.redirect_stdout(sys.stderr):
        agent = _agent(arm)
    agent.train(True)
    metrics = agent.update(_batch(arm), stddev=0.05, update_actor=True, bc_batch=None, ref_agent=agent)
    assert math.isfinite(metrics["train/critic_loss"])
    agent.train(False)
    with torch.no_grad():
        action = agent.act(dict(_batch(arm)["obs"][:1]), eval_mode=True)
    assert action.shape == (1, 7)


def test_cameras_are_read_by_name():
    from resfit.rl_finetuning.equi_off_policy.rl.q_agent import split_cameras

    assert split_cameras(list(BOTH), 1) == (BOTH[0], WRIST)
    assert split_cameras(list(WRIST), 1) == (None, WRIST)
    for bad in ([WRIST[0], BOTH[0]], [BOTH[0]], [BOTH[0], "observation.images.sideview", WRIST[0]]):
        with pytest.raises(ValueError, match="one wrist camera per arm"):
            split_cameras(bad, 1)


# Each refusal: (cameras, config paths, the message it raises).
REFUSALS = {
    "spatial critic behind another encoder": (WRIST, {**RESFIT_NETS, "agent.critic.fuse_patch": 0}, "critic_tokens=flat"),
    "flat critic with fuse_patch 1": (WRIST, {**RESFIT_NETS, "equivariance.critic_tokens": "flat"}, "fuse_patch=0"),
    "unknown critic_tokens": (BOTH, {"equivariance.critic_tokens": "patches"}, "expected one of"),
    "ablated encoder without agentview": (WRIST, {"equivariance.encoder_family": "ablated",
                                                  "equivariance.actor_family": "ablated",
                                                  "equivariance.critic_family": "ablated"}, "agentview"),
    "c8 wrist with agentview": (BOTH, {"equivariance.wrist_encoder": "c8"}, "without the agentview"),
    "unknown wrist_encoder": (WRIST, {"equivariance.wrist_encoder": "vit"}, "expected one of"),
    "heading with the equivariant actor": (WRIST, {"equivariance.action_frame": "heading",
                                                   "equivariance.gripper_xy_input": False}, "ResFiT's actor"),
    "heading with agentview": (BOTH, {**RESFIT_NETS, **FLAT, "equivariance.encoder_family": "resfit",
                                      "equivariance.action_frame": "heading",
                                      "equivariance.gripper_xy_input": False}, "no agentview"),
    "heading with gripper xy": (WRIST, {**RESFIT_NETS, **FLAT, "equivariance.action_frame": "heading"},
                                "gripper_xy_input=false"),
    "unknown action_frame": (WRIST, {"equivariance.action_frame": "gripper"}, "expected one of"),
}


@pytest.mark.parametrize("case", list(REFUSALS))
def test_a_mix_the_agent_cannot_build_is_refused(case):
    from tools.agent_fingerprint import build_reference_agent

    cameras, overrides, message = REFUSALS[case]
    with pytest.raises(ValueError, match=message), contextlib.redirect_stdout(sys.stderr):
        build_reference_agent("cpu", overrides=overrides, cameras=cameras)


# Each field no network reads without the agentview camera (or with the eq critic, critic_tokens), off its
# default, on the arm that does not read it.
UNREAD = {"enc_degree_channel": ("na-eee", 16), "encoder_norm": ("na-eee", "inner_bn"),
          "encoder_orth_init": ("na-eee", True), "initialize": ("na-eee", False),
          "enc_out_channel": ("w-cnn", 64), "critic_tokens": ("na-eee", "flat"),
          "ih_out_channel": ("g-c8", 64), "ih_channels": ("g-c8", [32, 64, 128, 256])}


@pytest.mark.parametrize("field", list(UNREAD))
def test_a_field_no_network_reads_is_refused_off_its_default(field):
    from tools.agent_fingerprint import build_reference_agent

    arm, value = UNREAD[field]
    cameras, overrides = ARMS[arm]
    with pytest.raises(ValueError, match=rf"reads equivariance\.\['{field}'\]"), \
            contextlib.redirect_stdout(sys.stderr):
        build_reference_agent("cpu", overrides={**overrides, f"equivariance.{field}": value}, cameras=cameras)


# ---------------------------------------------------------------------------
# The flat critic is ResFiT's own
# ---------------------------------------------------------------------------


def test_the_flat_critic_is_resfits_own_with_one_token():
    from resfit.rl_finetuning.config.rlpd import CriticConfig
    from resfit.rl_finetuning.equi_off_policy.rl.resfit_heads import ResFiTCritic
    from resfit.rl_finetuning.off_policy.rl.critic import Critic

    width, prop, act_dim = 256, 9, 7
    cfg = CriticConfig(num_q=10, fuse_patch=0)
    torch.manual_seed(0)
    ours = ResFiTCritic(1, width, prop, act_dim, cfg)
    torch.manual_seed(0)
    theirs = Critic(repr_dim=width, patch_repr_dim=width, prop_dim=prop, action_dim=act_dim, cfg=cfg)
    mine, its = ours.net.state_dict(), theirs.state_dict()
    assert mine.keys() == its.keys() and all(torch.equal(mine[k], its[k]) for k in mine)
    # Its trunk projects [features | action | state] whole: one Linear of all of them.
    assert theirs.q_ensemble.input_proj[0].in_features == width + act_dim + prop

    g = torch.Generator().manual_seed(1)
    feat, state = torch.randn(4, width, generator=g), torch.randn(4, prop, generator=g)
    act = torch.rand(4, act_dim, generator=g) * 2 - 1
    with torch.no_grad():
        assert torch.equal(ours(torch.cat([feat, state], dim=1), act), theirs(feat.unsqueeze(1), state, act))


# ---------------------------------------------------------------------------
# NA-eee: equivariant without the agentview camera
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("training", [False, True], ids=["eval", "train"])
@pytest.mark.parametrize("k", range(1, 8))
def test_the_agent_without_agentview_is_equivariant_about_the_base(k, training):
    """Every input turned about the base by 2 pi k / 8, the wrist image as it was: the actor's residual turns
    with them and Q is unchanged, for the action turned with them."""
    theta = 2 * math.pi * k / 8
    with contextlib.redirect_stdout(sys.stderr):
        agent = _agent("na-eee")
    obs, act = dict(_batch("na-eee")["obs"]), _batch("na-eee")["action"]
    turned = _turned_obs(obs, theta, about_base=True)
    with torch.no_grad():
        af, cf = _features(agent, obs, training)
        af_t, cf_t = _features(agent, turned, training)
        from escnn.nn import GeometricTensor
        wrap = lambda x, t: GeometricTensor(x, t)  # noqa: E731
        a = agent.actor(wrap(af, agent.enc.enc_out_type_actor), std=None)
        a_t = agent.actor(wrap(af_t, agent.enc.enc_out_type_actor), std=None)
        q = agent.critic(wrap(cf, agent.enc.enc_out_type_critic), act)
        q_t = agent.critic(wrap(cf_t, agent.enc.enc_out_type_critic), _turned_action(act, theta))
    assert a.abs().max() > 0
    assert _rel_err(a_t, _turned_action(a, theta)) < 1e-4
    assert _rel_err(q_t, q) < 1e-4


# ---------------------------------------------------------------------------
# G-cnn and G-c8: the heading frame
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("training", [False, True], ids=["eval", "train"])
@pytest.mark.parametrize("theta", [0.3, 1.0, -2.2])
@pytest.mark.parametrize("arm", ["g-cnn", "g-c8"])
def test_a_turn_about_the_hand_changes_nothing_in_the_heading_frame(arm, theta, training):
    with contextlib.redirect_stdout(sys.stderr):
        agent = _agent(arm)
    obs, act = dict(_batch(arm)["obs"]), _batch(arm)["action"]
    turned = _turned_obs(obs, theta, about_base=False)
    with torch.no_grad():
        af, cf = _features(agent, obs, training)
        af_t, cf_t = _features(agent, turned, training)
        # Every input but (cos psi, sin psi), which the wrappers read, is unchanged.
        assert (af_t[:, :-2] - af[:, :-2]).abs().max() < 1e-5
        assert (cf_t[:, :-2] - cf[:, :-2]).abs().max() < 1e-5
        assert (af_t[:, -2:] - _turn(af[:, -2:], theta)).abs().max() < 1e-5
        a, a_t = agent.actor(af, std=None), agent.actor(af_t, std=None)
        q, q_t = agent.critic(cf, act), agent.critic(cf_t, _turned_action(act, theta))
    assert a.abs().max() > 0
    assert (a_t - _turned_action(a, theta)).abs().max() < 1e-6
    assert (q_t - q).abs().max() < 1e-5


def test_the_heading_frame_turns_back_what_it_turns():
    """The heading-frame helpers against the matrices: R_psi^T R, and a turn and its inverse."""
    from resfit.rl_finetuning.equi_off_policy.networks.obs_encoder import quaternion_to_rotation_6d
    from resfit.rl_finetuning.equi_off_policy.rl.heading_frame import heading_of, turn_action, turn_rot6d

    g = torch.Generator().manual_seed(3)
    q = torch.randn(6, 4, generator=g)
    rot6d = quaternion_to_rotation_6d(q)
    c, s = heading_of(rot6d)
    psi = torch.atan2(s, c).squeeze(1)
    r_psi_t = torch.stack([torch.stack([psi.cos(), psi.sin()], -1), torch.stack([-psi.sin(), psi.cos()], -1)], -2)
    rows = rot6d.reshape(6, 2, 3)
    expected = torch.cat([r_psi_t @ rows[:, :, :2], r_psi_t @ rows[:, :, 2:]], dim=-1).reshape(6, 6)
    assert (turn_rot6d(rot6d, c, s) - expected).abs().max() < 1e-6
    a = torch.randn(6, 7, generator=g)
    assert (turn_action(turn_action(a, c, s), c, -s) - a).abs().max() < 1e-6


# ---------------------------------------------------------------------------
# The C8 wrist encoder and the proprioception without xy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [2, 4, 6])
def test_the_c8_wrist_encoder_permutes_its_fields_under_right_angle_turns(k):
    with contextlib.redirect_stdout(sys.stderr):
        enc = _agent("g-c8").enc
    enc.train(False)
    x = torch.randn(4, 3, 76, 76, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        out = enc.enc_ih[0](x).tensor.reshape(4, -1, 8)
        out_t = enc.enc_ih[0](ga.rotate_image(x, k, 8)).tensor.reshape(4, -1, 8)
    assert _rel_err(out_t, ga.act_regular(out, k, 8)) < 1e-4


def test_without_gripper_xy_nothing_reads_the_gripper_position():
    with contextlib.redirect_stdout(sys.stderr):
        agent = _agent("w-cnn")
    assert agent.enc.prop_dim == 9  # rot6d 6, z 1, opening 2
    obs = dict(_batch("w-cnn")["obs"])
    moved = dict(obs)
    moved["observation.state"] = obs["observation.state"].clone()
    moved["observation.state"][:, 0:2] += 0.3
    with torch.no_grad():
        for f, f_moved in zip(_features(agent, obs, False), _features(agent, moved, False)):
            assert torch.equal(f, f_moved)
