"""Tier 2: the declared FieldTypes match the physics, and widths line up.

Two separate things are checked here, and the split is the point:

  1. The representation *assignment* the code declares equals the one derived
     independently from the physics in tests/group_action.py. This is the
     "names match the math" check (STANDARDS.md rule 2.1). If the interleaved
     rotation-column slicing in obs_encoder ever drifts out of sync with
     prop_shape, the shapes still line up, the network still trains, and the
     symmetry is silently false. Only this test catches that.

  2. Tensor widths agree at every GeometricTensor boundary, so a mismatch fails
     at construction instead of after 41 hours (STANDARDS.md rule 2.5). 43 of
     197 failed runs in this project died with escnn types in a traceback.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests import group_action as ga
from tests.conftest import ENC_HIDDEN, IH_OUT, N


def _layouts(spec):
    """The documented layouts for a per-arm spec, from tests/group_action.py."""
    n, g, h = spec["n_arms"], spec["gripper_dim"], spec["hand_dof"]
    return {
        "action": ga.action_layout(n, h),
        "prop": ga.prop_layout(n, g),
        "vis_ih": ga.vis_ih_layout(ENC_HIDDEN, IH_OUT, n),
        "critic": ga.enc_out_layout_critic(ENC_HIDDEN, IH_OUT, n, g),
        "actor": ga.enc_out_layout_actor(ENC_HIDDEN, IH_OUT, n, g, h),
    }


# ---------------------------------------------------------------------------
# 1. Declared representations vs. the physics
# ---------------------------------------------------------------------------


def _independent_matrix(layout, k, n):
    """Block-diagonal group action matrix built from the physics layout.

    act_on_layout treats the last dimension as the vector axis, so feeding it an
    identity matrix transforms each *row*. Row i comes out as (R e_i), giving
    M[i, j] = R[j, i], i.e. M == R.T. Transpose back to compare against escnn's
    representation matrices.
    """
    width = ga.layout_width(layout, n)
    eye = torch.eye(width, dtype=torch.float64)
    return ga.act_on_layout(eye, layout, k, n).numpy().T


@pytest.mark.gpu
@pytest.mark.parametrize("k", range(N))
def test_action_representation_matches_physics(obs_enc, arm_spec, k):
    """The action FieldType must act as: rotate xy, hold z, rotate rx/ry, hold rz, hold gripper."""
    escnn_mat = obs_enc.action_type.representation(
        list(obs_enc.group.fibergroup.elements)[k]
    )
    ours = _independent_matrix(_layouts(arm_spec)["action"], k, N)
    np.testing.assert_allclose(escnn_mat, ours, atol=1e-6, err_msg=(
        f"action_type does not act as docs/EQUIVARIANCE.md says at k={k}. "
        "Either the FieldType or the documented layout is wrong."
    ))


@pytest.mark.gpu
@pytest.mark.parametrize("k", range(N))
def test_prop_representation_matches_physics(obs_enc, arm_spec, k):
    """Proprioception: 4 rotating xy pairs then 3 invariants."""
    escnn_mat = obs_enc.prop_type.representation(
        list(obs_enc.group.fibergroup.elements)[k]
    )
    ours = _independent_matrix(_layouts(arm_spec)["prop"], k, N)
    np.testing.assert_allclose(escnn_mat, ours, atol=1e-6, err_msg=(
        f"prop_type does not act as docs/EQUIVARIANCE.md says at k={k}"
    ))


@pytest.mark.gpu
@pytest.mark.parametrize("k", range(N))
def test_encoder_output_representation_matches_physics(obs_enc, arm_spec, k):
    """The full critic and actor feature layouts, end to end."""
    els = list(obs_enc.group.fibergroup.elements)
    lay = _layouts(arm_spec)
    for name, ft, layout in (
        ("enc_out_type_critic", obs_enc.enc_out_type_critic, lay["critic"]),
        ("enc_out_type_actor", obs_enc.enc_out_type_actor, lay["actor"]),
    ):
        np.testing.assert_allclose(
            ft.representation(els[k]), _independent_matrix(layout, k, N),
            atol=1e-6, err_msg=f"{name} disagrees with the documented layout at k={k}",
        )


# ---------------------------------------------------------------------------
# 2. Widths
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_field_type_widths(obs_enc, arm_spec):
    lay = _layouts(arm_spec)
    assert obs_enc.prop_type.size == ga.layout_width(lay["prop"], N)
    assert obs_enc.action_type.size == ga.layout_width(lay["action"], N)
    assert obs_enc.prop_dim == obs_enc.prop_type.size
    assert obs_enc.action_dim == obs_enc.action_type.size
    assert obs_enc.vis_ih_type.size == ga.layout_width(lay["vis_ih"], N)
    assert obs_enc.enc_out_type_critic.size == ga.layout_width(lay["critic"], N)
    assert obs_enc.enc_out_type_actor.size == ga.layout_width(lay["actor"], N)
    # The actor feature vector is the critic's plus one action block
    assert (obs_enc.enc_out_type_actor.size
            == obs_enc.enc_out_type_critic.size + obs_enc.action_type.size)


@pytest.mark.gpu
def test_action_layout_covers_action_exactly_once(obs_enc):
    """equi_clip iterates action_layout; an uncovered slice would be left unclipped."""
    covered = torch.zeros(obs_enc.action_type.size, dtype=torch.int32)
    for sl, kind in obs_enc.action_layout:
        assert kind in ("trivial", "irrep1"), f"unknown kind {kind!r}"
        covered[sl] += 1
    assert covered.min().item() == 1, f"gap in action_layout: {covered.tolist()}"
    assert covered.max().item() == 1, f"overlap in action_layout: {covered.tolist()}"


@pytest.mark.gpu
def test_action_fields_name_each_layout_entry_and_match_the_logged_fields(obs_enc, arm_spec):
    """obs_enc.action_fields names the field of each action_layout entry, which
    equivariance.residual_field_bound addresses, and field_metrics.action_fields logs the same
    fields over the same components."""
    from resfit.rl_finetuning.utils.field_metrics import action_fields

    assert len(obs_enc.action_fields) == len(obs_enc.action_layout)
    logged = action_fields(obs_enc.action_dim, arm_spec["n_arms"], arm_spec["hand_dof"])
    assert [f for f, _, _ in logged] == list(dict.fromkeys(obs_enc.action_fields))
    for field, sl, components in logged:
        entries = [s for (s, _), name in zip(obs_enc.action_layout, obs_enc.action_fields) if name == field]
        assert (sl.start, sl.stop) == (entries[0].start, entries[-1].stop) and len(components) == sl.stop - sl.start


@pytest.mark.gpu
def test_critic_trunk_widths(critic, obs_enc):
    """The critic slices its input by these widths; they must sum correctly."""
    assert critic.vis_ih_type.size == obs_enc.vis_ih_type.size
    assert critic.prop_type.size == obs_enc.prop_type.size
    assert critic.action_type.size == obs_enc.action_type.size
    assert (critic.vis_ih_type.size + critic.prop_type.size
            == obs_enc.enc_out_type_critic.size)


@pytest.mark.gpu
def test_actor_trunk_widths(actor, obs_enc):
    assert (actor.vis_ih_type.size + actor.prop_type.size + actor.action_type.size
            == obs_enc.enc_out_type_actor.size)


@pytest.mark.gpu
@pytest.mark.parametrize("task,state_dim,action_dim", [
    ("Can/Square", 9, 7),
    ("TwoArmBoxCleanup", 38, 24),
])
def test_raw_widths_match_the_environments(task, state_dim, action_dim):
    """state_dim is what the env emits; prop_dim is wider because quat becomes 6D.

    QAgent asserts encoder-vs-env on state_dim, so confusing the two would fail
    every run at construction. These numbers were measured from the envs.
    """
    from tests.conftest import ONE_ARM, TWO_ARM
    from resfit.rl_finetuning.equi_off_policy.networks.obs_encoder import ResObsEnc
    from tests.conftest import wrist_keys

    spec = ONE_ARM if task == "Can/Square" else TWO_ARM
    enc = ResObsEnc(
        N=N, n_hidden=ENC_HIDDEN, ih_out_channel=IH_OUT, equivariant=True, initialize=False,
        wrist_keys=wrist_keys(spec["n_arms"]), **spec,
    )
    assert enc.state_dim == state_dim
    assert enc.action_dim == action_dim
    assert enc.prop_dim == state_dim + 2 * spec["n_arms"]


def test_ih_out_channel_sets_the_wrist_width():
    """The wrist feature width is its own field; the agentview width does not move."""
    from tests.conftest import TWO_ARM, wrist_keys
    from resfit.rl_finetuning.equi_off_policy.networks.obs_encoder import ResObsEnc

    widths = {}
    for ih_out in (8, IH_OUT):
        enc = ResObsEnc(N=N, n_hidden=ENC_HIDDEN, ih_out_channel=ih_out, equivariant=True,
                        initialize=False, wrist_keys=wrist_keys(2), **TWO_ARM)
        assert enc.ih_type.size == 2 * ih_out
        widths[ih_out] = enc.vis_type.size
    assert widths[8] == widths[IH_OUT] == ENC_HIDDEN * N


# ---------------------------------------------------------------------------
# 3. Vanilla RL: no base action anywhere
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_vanilla_encoder_gives_the_actor_the_critics_features(device, arm_spec):
    """base_action_input=False, vanilla RL's encoder, reads no base action, so observations
    without one go through, and the actor's features are exactly the critic's."""
    from resfit.rl_finetuning.equi_off_policy.networks.equi_normalizer import (
        build_equivariant_normalizer,
    )
    from resfit.rl_finetuning.equi_off_policy.networks.obs_encoder import ResObsEnc
    from tests.conftest import BATCH, CROP, OBS_SHAPE, wrist_keys
    from tools.agent_fingerprint import ROTATION_CENTRE, _stats

    keys = ("observation.images.agentview", *wrist_keys(arm_spec["n_arms"]))
    enc = ResObsEnc(
        obs_shape=OBS_SHAPE, crop_shape=(CROP, CROP), N=N, n_hidden=ENC_HIDDEN, ih_out_channel=IH_OUT,
        equivariant=True, initialize=True, wrist_keys=keys[1:], base_action_input=False, **arm_spec,
    )
    assert enc.enc_out_type_actor == enc.enc_out_type_critic
    assert enc.enc_out_dim_actor == enc.enc_out_dim_critic == ga.layout_width(_layouts(arm_spec)["critic"], N)

    enc.set_normalizer(
        build_equivariant_normalizer(
            stats={"observation.state": _stats(enc.state_dim), "action": _stats(enc.action_dim)},
            robot_base_xy=torch.tensor(ROTATION_CENTRE), **arm_spec,
        ),
        robot_base_xy=list(ROTATION_CENTRE),
    )
    enc = enc.to(device).eval()
    g = torch.Generator().manual_seed(0)
    obs = {key: torch.randint(0, 256, (BATCH, *OBS_SHAPE), generator=g, dtype=torch.uint8) for key in keys}
    obs["observation.state"] = torch.randn(BATCH, enc.state_dim, generator=g)
    with torch.no_grad():
        actor_feat, critic_feat = enc({k: v.to(device) for k, v in obs.items()})
    assert actor_feat.type == critic_feat.type
    assert torch.equal(actor_feat.tensor, critic_feat.tensor)


@pytest.mark.gpu
def test_vanilla_actor_reads_no_base_action(obs_enc, device):
    """Its trunk and policy MLP each take one action block fewer than the residual actor's,
    and it refuses features that still carry one."""
    from escnn import nn as enn

    from tests.conftest import BATCH, build_actor

    residual = build_actor(obs_enc, device)
    vanilla = build_actor(obs_enc, device, residual_actor=False)
    action = obs_enc.action_type.size
    assert vanilla.input_proj[0].in_type.size == residual.input_proj[0].in_type.size - action
    assert vanilla.policy[0].in_type.size == residual.policy[0].in_type.size - action
    assert vanilla.vis_ih_type.size + vanilla.prop_type.size == obs_enc.enc_out_type_critic.size

    feat = torch.randn(BATCH, obs_enc.enc_out_type_actor.size, device=device)
    with pytest.raises(AssertionError, match="vanilla actor reads"):
        vanilla(enn.GeometricTensor(feat, obs_enc.enc_out_type_actor), std=None)


def test_the_scalar_ablation_is_residual_only():
    """Its actor reads a base action, so it refuses vanilla RL (residual_actor=False) before
    building anything."""
    from resfit.rl_finetuning.config.residual_td3 import ResidualEquiTD3SquareConfig
    from resfit.rl_finetuning.equi_off_policy.rl.q_agent import QAgent
    from tools.agent_fingerprint import ACTION_DIM, CAMERAS, OBS_SHAPE, STATE_DIM

    cfg = ResidualEquiTD3SquareConfig()
    cfg.agent.device = "cpu"
    with pytest.raises(ValueError, match="residual-only"):
        QAgent(obs_shape=OBS_SHAPE, prop_shape=(STATE_DIM,), action_dim=ACTION_DIM,
               rl_cameras=list(CAMERAS), agent_cfg=cfg.agent, equi_cfg=cfg.equivariance,
               residual_actor=False, equivariant=False)
