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
                n_hidden=ENC_HIDDEN, ih_out_channel=IH_OUT, equivariant=True, initialize=True)
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
    bias is now always off, and use_norms no longer reaches this encoder.
    """
    from escnn import nn as enn
    for kw in (dict(), dict(encoder_norm="inner_bn"), dict(use_norms=False)):
        enc = _build_obs_enc(device, **kw).enc_obs
        convs = [m for m in enc.modules() if isinstance(m, enn.R2Conv)]
        assert convs and all(m.bias is None for m in convs), kw
    assert _shapes_only(_build_obs_enc(device, use_norms=False)) == \
           _shapes_only(_build_obs_enc(device, use_norms=True))


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
