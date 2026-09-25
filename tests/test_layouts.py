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
from tests.conftest import ENC_HIDDEN, N


def _layouts(spec):
    """The documented layouts for a per-arm spec, from tests/group_action.py."""
    n, g, h = spec["n_arms"], spec["gripper_dim"], spec["hand_dof"]
    return {
        "action": ga.action_layout(n, h),
        "prop": ga.prop_layout(n, g),
        "vis_ih": ga.vis_ih_layout(ENC_HIDDEN, n),
        "critic": ga.enc_out_layout_critic(ENC_HIDDEN, n, g),
        "actor": ga.enc_out_layout_actor(ENC_HIDDEN, n, g, h),
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
        N=N, n_hidden=ENC_HIDDEN, equivariant=True, initialize=False,
        wrist_keys=wrist_keys(spec["n_arms"]), **spec,
    )
    assert enc.state_dim == state_dim
    assert enc.action_dim == action_dim
    assert enc.prop_dim == state_dim + 2 * spec["n_arms"]
