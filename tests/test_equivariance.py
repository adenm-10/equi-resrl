"""Tier 2: the network actually commutes with the group action.

test_layouts.py checks the declared representations against the physics. This
file checks the *implementation* against those declarations:

  * actor is EQUIVARIANT   pi(g.s) == g.pi(s)
  * critic is INVARIANT    Q(g.s, g.a) == Q(s, a)

Those are different assertions, and conflating them is the mistake that would
hide a broken GroupPooling head. See docs/EQUIVARIANCE.md.

Everything runs in eval mode. Random cropping is a random translation and does
not commute with rotation, so the vision encoder is tested directly rather than
through ResObsEnc.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests import group_action as ga
from tests.conftest import BATCH, CROP, ENC_HIDDEN, N, OBS_SHAPE, TOL


def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = b.abs().max().clamp_min(1e-12)
    return ((a - b).abs().max() / denom).item()


def _feat(obs_enc, layout, device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = ga.layout_width(layout, N)
    return torch.randn(BATCH, w, generator=g).to(device)


def _critic_layout(spec):
    return ga.enc_out_layout_critic(ENC_HIDDEN, spec["n_arms"], spec["gripper_dim"])


def _actor_layout(spec):
    return ga.enc_out_layout_actor(
        ENC_HIDDEN, spec["n_arms"], spec["gripper_dim"], spec["hand_dof"])


def _action_layout(spec):
    return ga.action_layout(spec["n_arms"], spec["hand_dof"])


# ---------------------------------------------------------------------------
# Vision encoder
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize("k", [k for k in range(N) if ga.is_right_angle(k, N)])
def test_vision_encoder_equivariant_at_right_angles(vision_encoder, device, k):
    """At 90-degree multiples, image rotation is exact, so this is a hard assertion."""
    g = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randn(BATCH, OBS_SHAPE[0], CROP, CROP, generator=g).to(device)

    with torch.no_grad():
        out_then_act = ga.act_on_layout(
            vision_encoder(x).tensor.reshape(BATCH, -1),
            [("regular", ENC_HIDDEN)], k, N,
        )
        act_then_out = vision_encoder(ga.rotate_image(x, k, N)).tensor.reshape(BATCH, -1)

    err = _rel_err(act_then_out, out_then_act)
    assert err < 1e-3, (
        f"vision encoder not equivariant at k={k} ({360*k/N:.0f} deg): "
        f"relative error {err:.2e}"
    )


@pytest.mark.gpu
def test_vision_encoder_error_at_diagonal_angles(vision_encoder, device, capsys):
    """At 45-degree multiples, image rotation needs interpolation.

    Reported, not asserted at the tight tolerance: the error here is a property
    of resampling a pixel grid, not a bug in the network. Recorded so the
    magnitude is a known quantity rather than a surprise.
    """
    g = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randn(BATCH, OBS_SHAPE[0], CROP, CROP, generator=g).to(device)
    errs = {}
    with torch.no_grad():
        base = vision_encoder(x).tensor.reshape(BATCH, -1)
        for k in range(N):
            if ga.is_right_angle(k, N):
                continue
            expected = ga.act_on_layout(base, [("regular", ENC_HIDDEN)], k, N)
            got = vision_encoder(ga.rotate_image(x, k, N)).tensor.reshape(BATCH, -1)
            errs[k] = _rel_err(got, expected)

    with capsys.disabled():
        pretty = ", ".join(f"k={k}: {v:.3f}" for k, v in sorted(errs.items()))
        print(f"\n    [diagonal-angle equivariance error] {pretty}")

    # A loose sanity bound only. If this trips, resampling is not the explanation.
    worst = max(errs.values())
    assert worst < 2.0, f"diagonal-angle error implausibly large: {worst:.2f}"


# ---------------------------------------------------------------------------
# Critic: invariance
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize("k", range(N))
def test_critic_is_invariant(critic, obs_enc, arm_spec, device, k):
    """Q(g.s, g.a) == Q(s, a). A value is a number; it has no direction.

    This is the assertion that certifies the GroupPooling head. A critic that
    trains fine and is not invariant is indistinguishable without this test.
    """
    from escnn import nn as enn

    feat_layout = _critic_layout(arm_spec)
    act_layout = _action_layout(arm_spec)
    feat = _feat(obs_enc, feat_layout, device, seed=1)
    act = _feat(obs_enc, act_layout, device, seed=2)

    with torch.no_grad():
        q_plain = critic(enn.GeometricTensor(feat, obs_enc.enc_out_type_critic), act)
        q_rot = critic(
            enn.GeometricTensor(
                ga.act_on_layout(feat, feat_layout, k, N), obs_enc.enc_out_type_critic),
            ga.act_on_layout(act, act_layout, k, N),
        )

    err = _rel_err(q_rot, q_plain)
    assert err < TOL, (
        f"critic not invariant at k={k}: relative error {err:.2e}. "
        "Check the GroupPooling placement in EquiHeadMLP -- it is what makes "
        "the Q output invariant rather than equivariant."
    )


@pytest.mark.gpu
def test_critic_q_depends_on_action(critic, obs_enc, arm_spec, device):
    """An invariant critic that ignores its action entirely is also 'invariant'.

    This guards against a degenerate pass: the residual-saturation failure mode
    in this project showed dQ_da around 2e-4, i.e. a critic with essentially no
    action dependence. An invariance test alone would not notice.
    """
    from escnn import nn as enn

    feat = _feat(obs_enc, _critic_layout(arm_spec), device, seed=3)
    gt = enn.GeometricTensor(feat, obs_enc.enc_out_type_critic)
    a1 = _feat(obs_enc, _action_layout(arm_spec), device, seed=4)
    a2 = _feat(obs_enc, _action_layout(arm_spec), device, seed=5)

    with torch.no_grad():
        q1, q2 = critic(gt, a1), critic(gt, a2)
    spread = (q1 - q2).abs().max().item()
    assert spread > 1e-6, (
        f"critic output is identical for two different actions (spread {spread:.2e}); "
        "it is ignoring the action, so the invariance test above is vacuous"
    )


# ---------------------------------------------------------------------------
# Actor: equivariance
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize("k", range(N))
def test_actor_is_equivariant(actor, obs_enc, arm_spec, device, k):
    """pi(g.s) == g.pi(s), on the deterministic path with action_scale applied."""
    from escnn import nn as enn

    feat_layout = _actor_layout(arm_spec)
    feat = _feat(obs_enc, feat_layout, device, seed=6)

    with torch.no_grad():
        mu = actor(enn.GeometricTensor(feat, obs_enc.enc_out_type_actor), std=None)
        mu_then_act = ga.act_on_layout(mu, _action_layout(arm_spec), k, N)
        act_then_mu = actor(
            enn.GeometricTensor(
                ga.act_on_layout(feat, feat_layout, k, N), obs_enc.enc_out_type_actor),
            std=None,
        )

    err = _rel_err(act_then_mu, mu_then_act)
    assert err < TOL, f"actor not equivariant at k={k}: relative error {err:.2e}"


@pytest.mark.gpu
def test_actor_output_is_nonzero(actor, obs_enc, arm_spec, device):
    """Guards against the trivial pass.

    With actor_last_layer_init_scale=0.0 -- which is what the reproduction runs
    with -- the actor outputs exactly zero and every equivariance test passes
    for free. The fixture deliberately uses None instead; this asserts it.
    """
    from escnn import nn as enn

    feat = _feat(obs_enc, _actor_layout(arm_spec), device, seed=7)
    with torch.no_grad():
        mu = actor(enn.GeometricTensor(feat, obs_enc.enc_out_type_actor), std=None)
    assert mu.abs().max().item() > 1e-8, (
        "actor output is all zeros, so the equivariance tests are vacuous"
    )


@pytest.mark.gpu
def test_actor_mean_is_scaled_not_bounded_by_action_scale(actor, obs_enc, arm_spec, device, capsys):
    """action_scale multiplies the mean; it does not bound it.

    The squash after the final Linear is commented out in actor.py, so |mu| is
    not <= 1 and the mean can exceed action_scale. What bounds the executed
    action is equi_clip at bound=1.0 on the std is not None path.
    """
    from escnn import nn as enn

    gt = enn.GeometricTensor(_feat(obs_enc, _actor_layout(arm_spec), device, seed=8),
                             obs_enc.enc_out_type_actor)
    with torch.no_grad():
        mu = actor(gt, std=None)       # unclipped mean
        executed = actor(gt, std=0.0)  # eval path: scaled, then equi_clip

    with capsys.disabled():
        print(f"\n    [actor |mu| / action_scale] {(mu.abs().max() / actor.action_scale):.3f}")

    for sl, kind in obs_enc.action_layout:
        chunk = executed[..., sl]
        worst = chunk.abs().max() if kind == "trivial" else chunk.norm(dim=-1).max()
        assert worst.item() <= 1.0 + 1e-5, (
            f"{kind} block {sl} exceeds the equi_clip bound on the eval path"
        )


# ---------------------------------------------------------------------------
# equi_clip
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize("k", range(N))
def test_equi_clip_commutes_with_the_group(obs_enc, arm_spec, device, k):
    """clip(g.a) == g.clip(a).

    Clamping each dimension onto a box is not rotationally symmetric; equi_clip
    projects irrep(1) pairs onto a disk instead. Its docstring claims this
    commutes -- this makes the claim checkable.
    """
    from resfit.rl_finetuning.equi_off_policy.rl.equi_rl_utils import equi_clip

    g = torch.Generator(device="cpu").manual_seed(9)
    # Scale up so a good fraction of samples actually hit the bound
    act = (torch.randn(64, obs_enc.action_type.size, generator=g) * 2.0).to(device)
    layout = obs_enc.action_layout

    act_layout = _action_layout(arm_spec)
    clipped_then_rot = ga.act_on_layout(
        equi_clip(act, layout, bound=1.0), act_layout, k, N)
    rot_then_clipped = equi_clip(
        ga.act_on_layout(act, act_layout, k, N), layout, bound=1.0)

    err = _rel_err(rot_then_clipped, clipped_then_rot)
    assert err < TOL, f"equi_clip does not commute with the group at k={k}: {err:.2e}"


@pytest.mark.gpu
def test_equi_clip_actually_clips(obs_enc, device):
    from resfit.rl_finetuning.equi_off_policy.rl.equi_rl_utils import equi_clip

    g = torch.Generator(device="cpu").manual_seed(10)
    act = (torch.randn(256, obs_enc.action_type.size, generator=g) * 5.0).to(device)
    out = equi_clip(act, obs_enc.action_layout, bound=1.0)

    for sl, kind in obs_enc.action_layout:
        chunk = out[..., sl]
        if kind == "trivial":
            assert chunk.abs().max().item() <= 1.0 + 1e-5, f"trivial block {sl} not clamped"
        else:
            norms = chunk.norm(dim=-1)
            assert norms.max().item() <= 1.0 + 1e-5, f"irrep1 block {sl} exceeds unit disk"
    assert (act.abs() > 1.0).any(), "test input never exceeded the bound; clipping untested"
