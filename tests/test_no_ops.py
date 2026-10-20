"""Tier 2: pin down which config fields do nothing, and what the live ones reach.

The provenance table in docs/ARCHITECTURE.md says which EquivarianceConfig
fields are accepted, logged to wandb and ignored. That claim came from reading
code; these tests make it executable. A run launched with `num_actor_layers=3`
had a 1-layer actor, and the log can say so because a test here proves it.

On 2026-09-29 the dead `num_critic_layers` was wired up, and `use_norms`, which
reached two of the actor's three norm sites and the encoder's conv biases, was
replaced on the equivariant path by `actor_norm`, `critic_norm` and
`encoder_norm`. The tests that pinned the old behaviour now pin the new one.
The defaults still build the networks every earlier run had, bit for bit:
tests/test_regression_agent.py checks that.
"""

from __future__ import annotations

import pytest
import torch

from tests.conftest import ENC_HIDDEN, IH_OUT, N
from tests.conftest import build_actor as _build_actor
from tests.conftest import build_critic as _build_critic


def _state_signature(module) -> list[tuple[str, tuple, float]]:
    """Shape + checksum per parameter, enough to detect any architecture change."""
    out = []
    for name, p in sorted(module.state_dict().items()):
        out.append((name, tuple(p.shape), float(p.double().sum())))
    return out


def _shapes_only(module) -> list[tuple[str, tuple]]:
    return [(n, s) for n, s, _ in _state_signature(module)]


def _count(module, cls) -> int:
    return sum(isinstance(m, cls) for m in module.modules())


def _build_obs_enc(device, **kw):
    from resfit.rl_finetuning.equi_off_policy.networks.obs_encoder import ResObsEnc
    opts = dict(obs_shape=(3, 84, 84), crop_shape=(76, 76), N=N,
                n_hidden=ENC_HIDDEN, ih_out_channel=IH_OUT, encoder="equivariant", initialize=True)
    opts.update(kw)
    return ResObsEnc(**opts).to(device)


def _norm_class(kind):
    from escnn import nn as enn

    from resfit.rl_finetuning.equi_off_policy.rl.equi_rl_utils import (
        FieldNorm,
        RegularInnerBatchNorm,
        RegularLayerNorm,
    )
    return {"iid_bn": enn.IIDBatchNorm1d, "inner_bn": RegularInnerBatchNorm,
            "layer_norm": RegularLayerNorm, "field_norm": FieldNorm}[kind]


# ---------------------------------------------------------------------------
# Still dead: architecture must be identical
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize("value", [1, 2, 3, 4])
def test_num_actor_layers_is_dead(obs_enc, device, value):
    """The policy head is hardcoded to 1 hidden layer."""
    ref = _shapes_only(_build_actor(obs_enc, device, num_layers=2))
    got = _shapes_only(_build_actor(obs_enc, device, num_layers=value))
    assert got == ref, (
        f"num_actor_layers={value} changed the actor architecture. Run a3e3zylp was "
        "launched with num_actor_layers=3 on the assumption it did nothing; if that "
        "changed, EXPERIMENTS.md needs revisiting."
    )


@pytest.mark.gpu
@pytest.mark.parametrize("value", [0.0, 0.1, 0.5])
def test_dropout_is_dead_in_both(obs_enc, device, value):
    assert _shapes_only(_build_critic(obs_enc, device, dropout=value)) == \
           _shapes_only(_build_critic(obs_enc, device, dropout=0.0))
    assert _shapes_only(_build_actor(obs_enc, device, dropout=value)) == \
           _shapes_only(_build_actor(obs_enc, device, dropout=0.0))


@pytest.mark.gpu
def test_initialize_is_dead_in_actor_and_critic(obs_enc, device):
    """Both hardcode escnn's init; only the encoder honours the flag."""
    torch.manual_seed(0)
    c_on = _state_signature(_build_critic(obs_enc, device))
    torch.manual_seed(0)
    c_off = _state_signature(_build_critic(obs_enc, device))
    assert c_on == c_off, "critic construction is not deterministic under a fixed seed"


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_use_orth_init_is_live_in_the_actor(obs_enc, device):
    torch.manual_seed(0)
    a = _state_signature(_build_actor(obs_enc, device, use_orth_init=True))
    torch.manual_seed(0)
    b = _state_signature(_build_actor(obs_enc, device, use_orth_init=False))
    assert a != b, (
        "use_orth_init no longer affects the actor. It was live at caf83f3; if it "
        "has gone dead, something was lost."
    )


@pytest.mark.gpu
def test_critic_orth_init_and_last_layer_scale_change_values_only(obs_enc, device):
    torch.manual_seed(0)
    ref = _state_signature(_build_critic(obs_enc, device))
    for kw in (dict(orth_init=True), dict(last_layer_init_scale=1e-3)):
        torch.manual_seed(0)
        got = _state_signature(_build_critic(obs_enc, device, **kw))
        assert [s[:2] for s in got] == [s[:2] for s in ref], f"{kw} changed the architecture"
        assert got != ref, f"{kw} changed no weight"


@pytest.mark.gpu
def test_encoder_orth_init_changes_values_only(device):
    torch.manual_seed(0)
    ref = _state_signature(_build_obs_enc(device).enc_obs)
    torch.manual_seed(0)
    got = _state_signature(_build_obs_enc(device, encoder_orth_init=True).enc_obs)
    assert [s[:2] for s in got] == [s[:2] for s in ref]
    assert got != ref


# ---------------------------------------------------------------------------
# Normalization: which sites each field reaches
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize("kind", ["iid_bn", "inner_bn", "layer_norm", "field_norm"])
def test_actor_norm_reaches_all_three_sites(obs_enc, device, kind):
    """Visual projection, trunk and the policy's hidden layer.

    "iid_bn" is what every run before 2026-09-30 had: use_norms=True put batch
    norm at the first two sites, and the third was unconditional. The default
    is now "none".
    """
    actor = _build_actor(obs_enc, device, norm=kind)
    assert _count(actor, _norm_class(kind)) == 3
    assert _shapes_only(_build_actor(obs_enc, device, norm="none")) != _shapes_only(actor)


@pytest.mark.gpu
def test_actor_without_norm_has_none(obs_enc, device):
    from escnn import nn as enn
    assert _count(_build_actor(obs_enc, device, norm="none"), enn.IIDBatchNorm1d) == 0


@pytest.mark.gpu
@pytest.mark.parametrize("kind", ["iid_bn", "inner_bn", "layer_norm", "field_norm"])
@pytest.mark.parametrize("num_layers", [1, 2])
def test_critic_norm_reaches_every_hidden_layer(obs_enc, device, kind, num_layers):
    """Visual projection, trunk, and each hidden layer of each head, never after pooling."""
    critic = _build_critic(obs_enc, device, norm=kind, num_layers=num_layers, num_q=3)
    assert _count(critic, _norm_class(kind)) == 2 + 3 * num_layers


@pytest.mark.gpu
@pytest.mark.parametrize("num_layers", [1, 2])
def test_critic_pooled_norm_sits_between_each_heads_pooling_and_output(obs_enc, device, num_layers):
    """critic_pooled_norm="layer_norm" adds one TrivialLayerNorm per head, right after its
    GroupPooling and right before its output Linear; "none" adds nothing, so the default critic
    keeps every parameter it had."""
    from escnn import nn as enn

    from resfit.rl_finetuning.equi_off_policy.common_utils.escnn_fixes import GroupPooling
    from resfit.rl_finetuning.equi_off_policy.rl.equi_rl_utils import TrivialLayerNorm

    critic = _build_critic(obs_enc, device, pooled_norm="layer_norm", num_layers=num_layers, num_q=3)
    assert _count(critic, TrivialLayerNorm) == 3
    for head in critic.q_ensemble.heads:
        assert isinstance(head.net[-3], GroupPooling)
        assert isinstance(head.net[-2], TrivialLayerNorm)
        assert isinstance(head.net[-1], enn.Linear)
    plain = _build_critic(obs_enc, device, num_layers=num_layers, num_q=3)
    assert _count(plain, TrivialLayerNorm) == 0
    assert _shapes_only(plain) != _shapes_only(critic)
    with pytest.raises(ValueError, match="unknown pooled-feature norm"):
        _build_critic(obs_enc, device, pooled_norm="field_norm")


def test_clip_q_target_clamps_the_equivariant_critics_target():
    """agent.clip_q_target_to_reward_range clamps the equivariant critic's TD target to [0, 1],
    the range of every correct target under the sparse 0/1 reward (EXPERIMENTS.md, 2026-10-10
    MEASUREMENT); off, the target follows the target critic. With the target critic made to say
    100, the clamped targets are all exactly 1 and the unclamped ones far above it."""
    from tools.agent_fingerprint import STDDEV, build_reference_agent, fixed_batch

    targets = {}
    for clip in (False, True):
        agent = build_reference_agent(overrides={"equivariance.actor_norm": "none",
                                                 "agent.clip_q_target_to_reward_range": clip})
        with torch.no_grad():
            for head in agent.critic_target.q_ensemble.heads:
                head.net[-1].bias.fill_(100.0)
        metrics = agent.update(fixed_batch(weights=True), stddev=STDDEV, update_actor=False,
                               compute_metrics=True)
        targets[clip] = metrics["_target_q"]
    assert torch.equal(targets[True], torch.ones_like(targets[True]))
    assert targets[False].max().item() > 50


@pytest.mark.gpu
def test_reference_critic_has_no_norm(obs_enc, device):
    from escnn import nn as enn

    from resfit.rl_finetuning.equi_off_policy.rl.equi_rl_utils import (
        FieldNorm,
        RegularInnerBatchNorm,
        RegularLayerNorm,
    )
    critic = _build_critic(obs_enc, device)
    for cls in (enn.IIDBatchNorm1d, RegularInnerBatchNorm, RegularLayerNorm, FieldNorm):
        assert _count(critic, cls) == 0


@pytest.mark.gpu
def test_encoder_is_bias_free_with_or_without_norm(device):
    """Every agentview conv is bias-free, as in every run so far.

    Before 2026-09-29 the last conv's bias was `not use_norms`, so use_norms=True
    gave a bias-free layer with no normalization after it (STANDARDS 1.3). The
    bias is now always off; use_norms stopped reaching this encoder then, and was
    removed with the scalar ablation on 2026-10-20.
    """
    from escnn import nn as enn
    for kw in (dict(), dict(encoder_norm="inner_bn")):
        enc = _build_obs_enc(device, **kw).enc_obs
        convs = [m for m in enc.modules() if isinstance(m, enn.R2Conv)]
        assert convs and all(m.bias is None for m in convs), kw


@pytest.mark.gpu
def test_encoder_norm_follows_every_conv(device):
    from escnn import nn as enn

    from resfit.rl_finetuning.equi_off_policy.rl.equi_rl_utils import RegularInnerBatchNorm
    enc = _build_obs_enc(device, encoder_norm="inner_bn").enc_obs
    convs = sum(isinstance(m, enn.R2Conv) for m in enc.modules())
    assert _count(enc, RegularInnerBatchNorm) == convs
    assert _count(enc, enn.InnerBatchNorm) == 0, "escnn's syncs; the encoder uses the sync-free one"
    assert _count(_build_obs_enc(device).enc_obs, RegularInnerBatchNorm) == 0


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize("num_layers", [1, 2, 3])
def test_num_critic_layers_sets_the_head_depth(obs_enc, device, num_layers):
    """Hidden Linears per head, then GroupPooling and the output Linear.

    1 is the head every run before 2026-09-29 had, whatever the logged value.
    The pooling is the project's subclass of escnn's (escnn_fixes.py), so it
    is compared by exact type.
    """
    from escnn import nn as enn

    from resfit.rl_finetuning.equi_off_policy.common_utils.escnn_fixes import GroupPooling
    head = _build_critic(obs_enc, device, num_layers=num_layers).q_ensemble.heads[0].net
    kinds = [type(m) for m in head]
    assert kinds.count(enn.Linear) == num_layers + 1
    assert kinds.count(enn.ReLU) == num_layers - 1
    assert kinds[-2:] == [GroupPooling, enn.Linear]


@pytest.mark.gpu
def test_visual_proj_off_removes_both_projections(obs_enc, device):
    for build in (_build_actor, _build_critic):
        on, off = build(obs_enc, device), build(obs_enc, device, use_visual_proj=False)
        assert isinstance(off.visual_proj, torch.nn.Identity)
        assert not isinstance(on.visual_proj, torch.nn.Identity)
        assert off.input_proj[0].in_type.size < on.input_proj[0].in_type.size


# ---------------------------------------------------------------------------
# equivariance.residual_field_bound (2026-10-12)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_residual_field_bound_scales_its_own_field_and_adds_no_parameter(obs_enc, arm_spec, device):
    """The empty mapping is every earlier run's actor, bit for bit. A field's bound replaces
    action_scale on that field's components, every arm's, and nowhere else. The bound is a buffer
    outside the state dict, so a checkpoint from before it existed still loads."""
    from escnn import nn as enn

    from tests.conftest import ACTION_SCALE
    from tests.test_equivariance import _actor_layout, _feat

    names = obs_enc.action_fields

    ref = _build_actor(obs_enc, device).eval()

    def like_ref(**kw):
        """An actor with ref's weights. The load is strict, so the bound added no state-dict entry."""
        actor = _build_actor(obs_enc, device, **kw).eval()
        actor.load_state_dict(ref.state_dict())
        return actor

    same = like_ref(field_bound={}, action_fields=names)
    no_tilt = like_ref(field_bound={"action_rx_ry": 0.0}, action_fields=names)
    rescaled = like_ref(field_bound={"action_rx_ry": 0.05, "action_xy": 0.0}, action_fields=names)
    assert "bound" not in no_tilt.state_dict() and _shapes_only(no_tilt) == _shapes_only(ref)

    feat = enn.GeometricTensor(_feat(None, _actor_layout(arm_spec), device, seed=5), obs_enc.enc_out_type_actor)
    with torch.no_grad():
        mu_ref, mu_same, mu_no_tilt, mu_rescaled = (a(feat, std=None) for a in (ref, same, no_tilt, rescaled))
    assert torch.equal(mu_same, mu_ref)

    def columns(field):
        return [i for (sl, _), name in zip(obs_enc.action_layout, names) if name.rsplit("_", 1)[0] == field
                for i in range(sl.start, sl.stop)]

    tilt, xy = columns("action_rx_ry"), columns("action_xy")
    assert len(tilt) == 2 * arm_spec["n_arms"]
    rest = [i for i in range(mu_ref.shape[1]) if i not in tilt]
    assert mu_ref[:, tilt].abs().min() > 0
    assert not mu_no_tilt[:, tilt].any()
    assert torch.equal(mu_no_tilt[:, rest], mu_ref[:, rest])
    torch.testing.assert_close(mu_rescaled[:, tilt], mu_ref[:, tilt] * (0.05 / ACTION_SCALE))
    assert not mu_rescaled[:, xy].any()


@pytest.mark.gpu
def test_residual_field_bound_refuses_an_unknown_field_and_a_vanilla_actor(obs_enc, device):
    with pytest.raises(ValueError, match="unknown fields"):
        _build_actor(obs_enc, device, field_bound={"action_tilt": 0.0}, action_fields=obs_enc.action_fields)
    with pytest.raises(ValueError, match="bounds a residual"):
        _build_actor(obs_enc, device, residual_actor=False, field_bound={"action_rx_ry": 0.0},
                     action_fields=obs_enc.action_fields)
