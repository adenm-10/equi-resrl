"""The equivariant update makes no host sync, and the fixes that got it there change nothing.

A host sync is any operation that makes the CPU wait until the GPU has finished all the work
queued so far: `.item()`, `.cpu()`, a copy from ordinary host memory, a GPU scalar used as a
slice bound. Before 2026-09-30 one environment step's learning made 487 of them.
docs/ARCHITECTURE.md, "Why the update used to stall", has the breakdown; docs/STANDARDS.md rule
6.5 is the rule this file enforces.

  Tier 1  escnn is the release escnn_fixes.py was written against; its basis expansions hash
          by identity; the project's GroupPooling gives exactly escnn's output; the random
          crops are the windows at the offsets the CPU generator draws; get6DRotation keeps
          its layout.
  Tier 2  For every config in tools/agent_fingerprint.py's CANDIDATES, for the two-arm
          layout, and for each mix of network families arch-swap-1021 runs, an update without
          the actor, one with it, and an act() make no host sync.

That the agent still trains bit for bit as before is checked elsewhere: within this tree by
tests/test_regression_agent.py, and across trees by `tools/agent_fingerprint.py --deep`.
"""

from __future__ import annotations

import pytest
import torch

from tests.conftest import BATCH, N, wrist_keys
from tools.agent_fingerprint import (
    CANDIDATES,
    INIT_SCALE,
    OBS_SHAPE,
    ROTATION_CENTRE,
    STDDEV,
    _stats,
    build_reference_agent,
    fixed_batch,
)

# target-fix and enc-inner keep the actor's IIDBatchNorm1d, the actor_norm default before
# 2026-09-30, which build_reference_agent builds unless a candidate overrides it. escnn's forward
# asserts on its own running statistics, which syncs 6 times per actor forward. It is not fixed
# (TODO H8), so those candidates are left out of the census.
SYNC_FREE = [name for name, overrides in CANDIDATES.items()
             if overrides.get("equivariance.actor_norm", "iid_bn") != "iid_bn"]

# The mixes of network families (equivariance.*_family; 2026-10-20 CODE) other than the equivariant
# stack, at their arms' designs: arch-wide's for the ablated stack, and ResFiT's 10 critic heads of 2
# layers with RegularLayerNorm for an equivariant or ablated critic beside ResFiT's actor. ResFiT's
# actor reads none of the equivariant actor's fields, so they stay at their defaults.
_HEADS = {"agent.critic.num_q": 10, "equivariance.num_critic_layers": 2, "equivariance.critic_norm": "layer_norm"}


def _families(encoder, actor, critic):
    return {"equivariance.encoder_family": encoder, "equivariance.actor_family": actor,
            "equivariance.critic_family": critic}


FAMILY_MIXES = {
    "aaa": {**_HEADS, "equivariance.actor_norm": "layer_norm", "equivariance.actor_degree_channel": 448,
            **_families("ablated", "ablated", "ablated")},
    "ere": {**_HEADS, "equivariance.actor_norm": "none", **_families("equivariant", "resfit", "equivariant")},
    "ara": {**_HEADS, "equivariance.actor_norm": "none", **_families("ablated", "resfit", "ablated")},
    "rrr": {"agent.critic.num_q": 10, "equivariance.actor_norm": "none", **_families("resfit", "resfit", "resfit")},
    "rra": {**_HEADS, "equivariance.actor_norm": "none", **_families("resfit", "resfit", "ablated")},
}


# ---------------------------------------------------------------------------
# Tier 1: the fixes are what they claim to be
# ---------------------------------------------------------------------------


def test_escnn_is_the_release_the_fixes_were_written_for():
    from importlib.metadata import version

    from resfit.rl_finetuning.equi_off_policy.common_utils.escnn_fixes import ESCNN_VERSION
    assert version("escnn") == ESCNN_VERSION, (
        "escnn_fixes.py relies on escnn internals. Re-run this file against the new release, "
        "then update ESCNN_VERSION.")


def test_basis_expansions_hash_by_identity():
    from escnn.nn.modules.basismanager.basisexpansion_blocks import BlocksBasisExpansion
    from escnn.nn.modules.basismanager.basisexpansion_singleblock import (
        SingleBlockBasisExpansion,
    )

    # The fix applies when the agent's modules are imported, whoever imports them first.
    import resfit.rl_finetuning.equi_off_policy.rl.q_agent  # noqa: F401
    for cls in (BlocksBasisExpansion, SingleBlockBasisExpansion):
        assert cls.__hash__ is object.__hash__ and cls.__eq__ is object.__eq__, cls


def _field_type(layout: str):
    """A field type for the pooling tests: regular fields as in the critic's heads, regular
    then trivial fields (two contiguous blocks), or the two interleaved (not contiguous)."""
    from escnn import gspaces
    from escnn import nn as enn
    from escnn.group import CyclicGroup

    group = gspaces.no_base_space(CyclicGroup(N))
    regular, trivial = group.regular_repr, group.trivial_repr
    reprs = {
        "regular": 5 * [regular],
        "regular+trivial": 3 * [regular] + 2 * [trivial],
        "interleaved": [regular, trivial, regular],
    }[layout]
    return enn.FieldType(group, reprs)


@pytest.mark.parametrize("layout", ["regular", "regular+trivial"])
def test_group_pooling_matches_escnn(layout, device):
    from escnn import nn as enn

    from resfit.rl_finetuning.equi_off_policy.common_utils.escnn_fixes import GroupPooling

    in_type = _field_type(layout)
    ours, escnns = GroupPooling(in_type).to(device), enn.GroupPooling(in_type).to(device)
    assert ours.out_type == escnns.out_type

    g = torch.Generator().manual_seed(0)
    x = enn.GeometricTensor(torch.randn(BATCH, in_type.size, generator=g).to(device), in_type)
    torch.testing.assert_close(ours(x).tensor, escnns(x).tensor, rtol=0, atol=0)

    # Same buffers, so a checkpoint saved with either loads into the other.
    theirs = escnns.state_dict()
    assert list(ours.state_dict()) == list(theirs)
    for name, value in ours.state_dict().items():
        assert torch.equal(value, theirs[name]), name


def test_group_pooling_refuses_fields_it_cannot_slice():
    from resfit.rl_finetuning.equi_off_policy.common_utils.escnn_fixes import GroupPooling

    with pytest.raises(ValueError, match="contiguous"):
        GroupPooling(_field_type("interleaved"))


def test_random_crops_are_the_windows_at_the_drawn_offsets(device):
    """The crop offsets are drawn from torch's CPU generator, in the same calls and order as
    before 2026-09-30, and each crop is the window of its image at that offset."""
    from resfit.rl_finetuning.equi_off_policy.common_utils.crop_randomizer import (
        sample_random_image_crops,
    )

    height, width, crop = OBS_SHAPE[1], OBS_SHAPE[2], 76
    g = torch.Generator().manual_seed(0)
    images = torch.randn(BATCH, 3, height, width, generator=g).to(device)

    torch.manual_seed(3)
    crops, offsets = sample_random_image_crops(images, crop, crop, num_crops=1)
    torch.manual_seed(3)
    expected_h = ((height - crop) * torch.rand(BATCH, 1)).long()
    expected_w = ((width - crop) * torch.rand(BATCH, 1)).long()

    assert offsets.device == images.device
    assert torch.equal(offsets.cpu(), torch.stack((expected_h, expected_w), dim=-1))
    for b in range(BATCH):
        h, w = expected_h[b, 0].item(), expected_w[b, 0].item()
        assert torch.equal(crops[b, 0], images[b, :, h:h + crop, w:w + crop]), b


def test_rotation_keeps_its_layout(obs_enc, device):
    """get6DRotation reorders robosuite's (x, y, z, w) quaternion by slicing. The result must be
    what indexing with [3, 0, 1, 2], as before 2026-09-30, gave."""
    from resfit.rl_finetuning.equi_off_policy.networks.obs_encoder import (
        quaternion_to_rotation_6d,
    )

    quat = torch.randn(BATCH, 4, generator=torch.Generator().manual_seed(0)).to(device)
    quat = quat / quat.norm(dim=-1, keepdim=True)
    expected = quaternion_to_rotation_6d(quat[:, [3, 0, 1, 2]])
    torch.testing.assert_close(obs_enc.get6DRotation(quat), expected, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Tier 2: no host syncs
# ---------------------------------------------------------------------------


def _two_arm_agent(device):
    """The agent on TwoArmBoxCleanup at the default config, and its batch layout."""
    from resfit.rl_finetuning.config.residual_td3 import ResidualEquiTD3BoxCleanConfig
    from resfit.rl_finetuning.equi_off_policy.networks.equi_normalizer import (
        build_equivariant_normalizer,
    )
    from resfit.rl_finetuning.equi_off_policy.rl.q_agent import QAgent

    cfg = ResidualEquiTD3BoxCleanConfig()
    cfg.agent.device = str(device)
    cfg.agent.actor.actor_last_layer_init_scale = INIT_SCALE
    eq = cfg.equivariance
    layout = dict(
        cameras=("observation.images.agentview", *wrist_keys(eq.n_arms)),
        state_dim=eq.n_arms * (3 + 4 + eq.gripper_dim),   # [eef_pos, eef_quat, gripper_qpos]
        action_dim=eq.n_arms * (6 + eq.hand_dof),         # [delta_pose, hand]
    )
    agent = QAgent(
        obs_shape=OBS_SHAPE, prop_shape=(layout["state_dim"],), action_dim=layout["action_dim"],
        rl_cameras=list(layout["cameras"]), agent_cfg=cfg.agent, equi_cfg=eq,
        residual_actor=True,
    )
    normalizer = build_equivariant_normalizer(
        stats={"observation.state": _stats(layout["state_dim"]),
               "action": _stats(layout["action_dim"])},
        robot_base_xy=torch.tensor(ROTATION_CENTRE),  # also BoxCleanup's (env_probes/)
        n_arms=eq.n_arms, gripper_dim=eq.gripper_dim, hand_dof=eq.hand_dof,
    )
    agent.enc.set_normalizer(normalizer, robot_base_xy=list(ROTATION_CENTRE))
    agent.to(device)
    return agent, layout


def _assert_no_host_sync(agent, make_batch):
    """What the trainer does between two log steps: updates without metrics, with and without
    the actor, and act() between them. torch raises at the first host sync."""
    from resfit.rl_finetuning.off_policy.common_utils import utils

    def cycle(critic_batch, actor_batch, act_batch):
        agent.update(critic_batch, stddev=STDDEV, update_actor=False, compute_metrics=False)
        agent.update(actor_batch, stddev=STDDEV, update_actor=True, compute_metrics=False)
        with torch.no_grad(), utils.eval_mode(agent):
            agent.act(act_batch["obs"], eval_mode=False, stddev=STDDEV, cpu=False)

    agent.train(True)
    # The batches are built on the GPU before the check: the trainer's come from the replay
    # buffer by an asynchronous copy, but these come from ordinary host memory, which syncs.
    # The first cycle is outside the check too: cuDNN and the allocator set up there.
    cycle(make_batch(), make_batch(), make_batch())
    batches = (make_batch(), make_batch(), make_batch())
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        cycle(*batches)
    finally:
        torch.cuda.set_sync_debug_mode("default")


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("candidate", SYNC_FREE)
def test_candidate_update_makes_no_host_sync(candidate, device):
    agent = build_reference_agent(str(device), overrides=CANDIDATES[candidate])
    _assert_no_host_sync(agent, lambda: fixed_batch(weights=True).to(device))


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("mix", list(FAMILY_MIXES))
def test_family_mix_update_makes_no_host_sync(mix, device):
    """The ablated layers, ResFiT's actor, critic and ViTs, the random shifts, and the ViTs' forward
    without VitEncoder.forward's own syncs (networks/obs_encoder.py, _vit_tokens)."""
    agent = build_reference_agent(str(device), overrides=FAMILY_MIXES[mix])
    _assert_no_host_sync(agent, lambda: fixed_batch(weights=True).to(device))


@pytest.mark.gpu
@pytest.mark.slow
def test_encoder_inner_bn_makes_no_host_sync(device):
    """enc-inner is out of the census for its actor's IIDBatchNorm1d; its encoder norm is not."""
    overrides = {**CANDIDATES["enc-inner"], "equivariance.actor_norm": "none"}
    agent = build_reference_agent(str(device), overrides=overrides)
    _assert_no_host_sync(agent, lambda: fixed_batch(weights=True).to(device))


@pytest.mark.gpu
@pytest.mark.slow
def test_two_arm_update_makes_no_host_sync(device):
    agent, layout = _two_arm_agent(device)
    _assert_no_host_sync(agent, lambda: fixed_batch(weights=True, **layout).to(device))


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("squash", [False, True], ids=["unbounded", "squash"])
def test_vanilla_update_makes_no_host_sync(device, squash):
    """Vanilla RL's agent (base_policy=None), on batches with no base action, at the actor_norm
    default. The equivariant tuned-RLPD arm's provisional iid_bn syncs, as every iid_bn actor does."""
    overrides = {"equivariance.actor_norm": "none", "equivariance.residual_squash": squash,
                 "agent.actor.action_scale": 1.0, "agent.actor.actor_last_layer_init_scale": None}
    agent = build_reference_agent(str(device), overrides=overrides, residual_actor=False)
    _assert_no_host_sync(agent, lambda: fixed_batch(weights=True, base_action=False).to(device))
