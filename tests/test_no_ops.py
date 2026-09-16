"""Tier 2: pin down which config fields actually do nothing.

The provenance table in docs/ARCHITECTURE.md claims several EquivarianceConfig
fields are accepted, logged to wandb, and ignored. That claim came from reading
code. These tests make it executable, which does two jobs:

  1. Experiment records can be read correctly *now* -- a run launched with
     `num_actor_layers=3` had a 1-layer actor, and the log says so because this
     test proves it.
  2. They are the regression baseline for TODO P3.1. When the dead fields get
     wired up after the reproduction, these tests flip from passing to failing,
     which is the signal that behaviour changed on purpose.

`use_norms` gets a four-way treatment because it is NOT a clean no-op: the norm
layers are commented out but the flag still drives `bias=not use_norms` in the
encoder. A flag that half-survives a commenting-out is more dangerous than one
fully dead, because it looks inert and is not. See STANDARDS.md rule 1.3.
"""

from __future__ import annotations

import pytest
import torch

from tests.conftest import ACTION_SCALE, ACTOR_HIDDEN, CRITIC_HIDDEN, ENC_HIDDEN, N


def _state_signature(module) -> list[tuple[str, tuple, float]]:
    """Shape + checksum per parameter, enough to detect any architecture change."""
    out = []
    for name, p in sorted(module.state_dict().items()):
        out.append((name, tuple(p.shape), float(p.double().sum())))
    return out


def _shapes_only(module) -> list[tuple[str, tuple]]:
    return [(n, s) for n, s, _ in _state_signature(module)]


def _build_obs_enc(device, **kw):
    from resfit.rl_finetuning.equi_off_policy.networks.obs_encoder import ResObsEnc
    opts = dict(obs_shape=(3, 84, 84), crop_shape=(76, 76), N=N,
                n_hidden=ENC_HIDDEN, equivariant=True, initialize=True)
    opts.update(kw)
    return ResObsEnc(**opts).to(device)


def _build_critic(obs_enc, device, **kw):
    from resfit.rl_finetuning.config.rlpd import CriticLossCfg
    from resfit.rl_finetuning.equi_off_policy.rl.critic import Critic
    opts = dict(group=obs_enc.group, vis_ih_type=obs_enc.vis_ih_type,
                prop_type=obs_enc.prop_type, action_type=obs_enc.action_type,
                hidden_dim=CRITIC_HIDDEN, num_layers=2, num_q=2, dropout=0.0,
                use_norms=True, use_orth_init=True, loss_cfg=CriticLossCfg(),
                min_q_heads=2, policy_gradient_type="ensemble_mean")
    opts.update(kw)
    return Critic(**opts).to(device)


def _build_actor(obs_enc, device, **kw):
    from resfit.rl_finetuning.equi_off_policy.rl.actor import Actor
    opts = dict(group=obs_enc.group, vis_ih_type=obs_enc.vis_ih_type,
                prop_type=obs_enc.prop_type, action_type=obs_enc.action_type,
                action_layout=obs_enc.action_layout, hidden_dim=ACTOR_HIDDEN,
                action_shape=obs_enc.action_shape, num_layers=2, dropout=0.0,
                use_norms=True, use_orth_init=True, last_layer_scale=None,
                action_scale=ACTION_SCALE, residual_actor=True)
    opts.update(kw)
    return Actor(**opts).to(device)


# ---------------------------------------------------------------------------
# Clean no-ops: architecture must be identical
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize("value", [1, 2, 3, 4])
def test_num_critic_layers_is_dead(obs_enc, device, value):
    """critic.py:30-33 is commented out, so the head is fixed regardless."""
    ref = _shapes_only(_build_critic(obs_enc, device, num_layers=2))
    got = _shapes_only(_build_critic(obs_enc, device, num_layers=value))
    assert got == ref, (
        f"num_critic_layers={value} changed the critic architecture. If this was "
        "intentional (TODO P3.1), update the provenance table in ARCHITECTURE.md "
        "and this test together."
    )


@pytest.mark.gpu
@pytest.mark.parametrize("value", [1, 2, 3, 4])
def test_num_actor_layers_is_dead(obs_enc, device, value):
    """actor.py:75-84 is commented out; the policy head is hardcoded to 1 hidden layer."""
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
def test_use_orth_init_is_dead_in_the_critic(obs_enc, device):
    """critic.py:68-70 and 140-142 are commented out.

    Compares checksums, not just shapes: an init change moves values, not shapes.
    """
    torch.manual_seed(0)
    a = _state_signature(_build_critic(obs_enc, device, use_orth_init=True))
    torch.manual_seed(0)
    b = _state_signature(_build_critic(obs_enc, device, use_orth_init=False))
    assert a == b, "use_orth_init now affects the critic; update ARCHITECTURE.md"


@pytest.mark.gpu
def test_use_orth_init_is_live_in_the_actor(obs_enc, device):
    """The counterpart: the actor DOES honour it, via _initialize_weights."""
    torch.manual_seed(0)
    a = _state_signature(_build_actor(obs_enc, device, use_orth_init=True))
    torch.manual_seed(0)
    b = _state_signature(_build_actor(obs_enc, device, use_orth_init=False))
    assert a != b, (
        "use_orth_init no longer affects the actor. It was live at caf83f3; if it "
        "has gone dead, something was lost."
    )


# ---------------------------------------------------------------------------
# use_norms: the four-way split
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_use_norms_is_dead_in_the_critic(obs_enc, device):
    assert _shapes_only(_build_critic(obs_enc, device, use_norms=True)) == \
           _shapes_only(_build_critic(obs_enc, device, use_norms=False)), \
        "use_norms now affects the critic; update ARCHITECTURE.md and P3.1"


@pytest.mark.gpu
def test_use_norms_is_live_in_the_actor(obs_enc, device):
    """Two of the actor's three norm sites are conditional, so shapes differ."""
    on = _shapes_only(_build_actor(obs_enc, device, use_norms=True))
    off = _shapes_only(_build_actor(obs_enc, device, use_norms=False))
    assert on != off, "use_norms no longer affects the actor"


@pytest.mark.gpu
def test_actor_has_one_unconditional_norm(obs_enc, device):
    """actor.py:70 adds IIDBatchNorm1d regardless of use_norms.

    Documented so nobody 'fixes' the config field without noticing that one
    site ignores it. Run qe2by47h was launched with use_norms=False and still
    had this layer.
    """
    from escnn import nn as enn
    a = _build_actor(obs_enc, device, use_norms=False)
    norms = [m for m in a.policy.modules() if isinstance(m, enn.IIDBatchNorm1d)]
    assert len(norms) == 1, (
        f"expected exactly 1 unconditional batch norm in the actor policy head, "
        f"found {len(norms)}"
    )


@pytest.mark.gpu
def test_use_norms_changes_the_encoder_via_bias_coupling(device):
    """The subtle one.

    Every InnerBatchNorm in equi_encoder.py is commented out, but line 125 still
    sets `bias=not use_norms`. So use_norms=True gives a bias-free layer with no
    normalisation following it -- almost certainly not what was intended, and
    invisible if you only grep for the norm layers.
    """
    on = _shapes_only(_build_obs_enc(device, use_norms=True))
    off = _shapes_only(_build_obs_enc(device, use_norms=False))
    assert on != off, (
        "use_norms no longer changes the encoder. Either the bias coupling at "
        "equi_encoder.py:125 was removed or the norms were restored; either way "
        "ARCHITECTURE.md's provenance table needs updating."
    )


@pytest.mark.gpu
def test_initialize_is_dead_in_actor_and_critic(obs_enc, device):
    """Both hardcode `escnn_init = True`; only the encoder honours the flag."""
    torch.manual_seed(0)
    c_on = _state_signature(_build_critic(obs_enc, device))
    torch.manual_seed(0)
    c_off = _state_signature(_build_critic(obs_enc, device))
    assert c_on == c_off, "critic construction is not deterministic under a fixed seed"
