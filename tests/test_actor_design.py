"""Tier 2: the actor's design (equivariance.actor_design; EXPERIMENTS.md, 2026-10-26 CODE).

ResFiTShapedActor is the equivariant actor wired as ResFiT's actor is (off_policy/rl/actor.py): the image
features compressed to actor_feature_fields regular fields, then [compressed | state | base action] through
two hidden layers of actor_degree_channel fields, each hidden layer followed by its norm and a ReLU, then
a Linear to the action representation. These check, at actor-resfit-shape's config and in isolation:

  * its sizes are ResFiT's actor's in channels: 128, 1,024, 1,024, then the 7 action numbers, and the
    state and base action reach it once, at its first hidden layer;
  * it starts at zero with the last layer scaled to 0, as every residual actor here does;
  * it is equivariant, pi(g.s) = g.pi(s), with and without its norm and its squash, in eval and train;
  * the agent refuses the design where it cannot be built, and its width where nothing reads it.

Actor, the default design, is pinned by tests/test_regression_agent.py and built by every other test.
tests/test_sweep_arms.py checks the arm itself: its launch, its families, its encoder and critic eee's.
"""

from __future__ import annotations

import pytest
import torch

from tests.conftest import ACTION_SCALE, ACTOR_HIDDEN
from tests.test_equivariance import _actor_layout, _assert_actor_equivariant

# actor-resfit-shape's networks (submit.sh): arch-swap-eee's, with the actor's design ResFiT's. Its rate
# (3e-6) moves nothing built.
ARM = {"equivariance.actor_norm": "layer_norm", "equivariance.critic_norm": "layer_norm",
       "agent.critic.num_q": 10, "equivariance.num_critic_layers": 2,
       "agent.actor.actor_last_layer_init_scale": 0.0, "equivariance.actor_design": "resfit"}


def _arm_agent(**overrides):
    from tools.agent_fingerprint import build_reference_agent

    return build_reference_agent(overrides={**ARM, **overrides})


def _shaped(obs_enc, device, **kw):
    from resfit.rl_finetuning.equi_off_policy.rl.actor import ResFiTShapedActor

    opts = dict(group=obs_enc.group, vis_ih_type=obs_enc.vis_ih_type, prop_type=obs_enc.prop_type,
                action_type=obs_enc.action_type, action_layout=obs_enc.action_layout, feature_fields=2,
                hidden_dim=ACTOR_HIDDEN, action_shape=obs_enc.action_shape, norm="none", use_orth_init=True,
                last_layer_scale=None, action_scale=ACTION_SCALE, residual_squash=False)
    opts.update(kw)
    return ResFiTShapedActor(**opts).to(device)


def test_the_arms_actor_has_resfits_widths_and_reads_the_state_and_base_action_once():
    from escnn import nn as enn

    from resfit.rl_finetuning.equi_off_policy.rl.actor import ResFiTShapedActor
    from resfit.rl_finetuning.equi_off_policy.rl.equi_rl_utils import RegularLayerNorm

    actor = _arm_agent().actor
    assert type(actor) is ResFiTShapedActor
    linears = [m for m in [*actor.compress, *actor.policy] if isinstance(m, enn.Linear)]
    assert [m.out_type.size for m in linears] == [128, 1024, 1024, 7]
    assert all(r.name == "regular" for m in linears[:3] for r in m.out_type.representations)
    assert linears[0].in_type == actor.vis_ih_type
    # The state and the base action, at the first hidden layer alone.
    assert linears[1].in_type == actor.compress_type + actor.prop_type + actor.action_type
    assert linears[2].in_type == linears[1].out_type and linears[3].in_type == linears[2].out_type
    kinds = [type(m).__name__ for m in [*actor.compress, *actor.policy]]
    assert kinds == ["Linear", "RegularLayerNorm", "ReLU"] * 3 + ["Linear"]
    assert sum(isinstance(m, RegularLayerNorm) for m in actor.modules()) == 3
    # 156,083 free parameters at Square's one arm: the weights tied across the 8 rotations.
    assert sum(p.numel() for p in actor.parameters()) == 156_083


def test_the_wide_arms_hidden_layer_has_resfits_free_parameters():
    """actor-resfit-shape-wide: 362 fields per hidden layer, so its hidden-to-hidden Linear has ResFiT's
    1024 x 1024 + 1024 = 1,049,600 free parameters to 0.1%; 1,110,569 in all, against ResFiT's 1,248,647,
    its bottleneck and first hidden layer staying as narrow as the design has them."""
    from escnn import nn as enn

    actor = _arm_agent(**{"equivariance.actor_degree_channel": 362}).actor
    hidden = [m for m in actor.policy if isinstance(m, enn.Linear)][1]
    assert hidden.out_type.size == 362 * 8
    assert abs(sum(p.numel() for p in hidden.parameters()) - 1_049_600) / 1_049_600 < 1e-3
    assert sum(p.numel() for p in actor.parameters()) == 1_110_569


def test_the_arms_actor_starts_at_zero():
    from escnn import nn as enn

    actor = _arm_agent().actor
    feat_type = actor.vis_ih_type + actor.prop_type + actor.action_type
    with torch.no_grad():
        mu = actor(enn.GeometricTensor(torch.randn(4, feat_type.size), feat_type))
    assert torch.equal(mu, torch.zeros_like(mu))


@pytest.mark.gpu
@pytest.mark.parametrize("training", [False, True], ids=["eval", "train"])
@pytest.mark.parametrize("option", [{}, {"norm": "layer_norm"}, {"residual_squash": True},
                                    {"norm": "layer_norm", "residual_squash": True}],
                         ids=["plain", "layer-norm", "squash", "layer-norm+squash"])
def test_the_resfit_shaped_actor_is_equivariant(obs_enc, arm_spec, device, option, training):
    actor = _shaped(obs_enc, device, **option).train(training)
    _assert_actor_equivariant(actor, obs_enc.enc_out_type_actor, _actor_layout(arm_spec), arm_spec, device,
                              f"resfit-shaped {option}", seed=31)


@pytest.mark.parametrize("overrides, message", [
    ({"equivariance.actor_design": "wide"}, "expected one of"),
    ({"equivariance.actor_family": "ablated"}, "no export of ResFiTShapedActor"),
    ({"equivariance.actor_family": "resfit"}, "actor_design"),
    ({"equivariance.actor_design": "ours", "equivariance.actor_feature_fields": 32}, "actor_feature_fields"),
], ids=["unknown", "ablated", "resfit-family", "unread-width"])
def test_the_agent_refuses_the_design_where_it_cannot_be_built(overrides, message):
    with pytest.raises(ValueError, match=message):
        _arm_agent(**overrides)


def test_the_agent_refuses_the_design_in_vanilla_rl():
    from tools.agent_fingerprint import build_reference_agent

    with pytest.raises(ValueError, match="residual-only"):
        build_reference_agent(residual_actor=False, actor_design="resfit")
