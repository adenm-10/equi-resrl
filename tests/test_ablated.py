"""Tier 2: the ablated family (equivariance.*_family=ablated; rl/ablated.py) is the equivariant
network exported to plain torch layers, and the negative control the equivariance suite lacked
(tests/README.md, "Known gaps").

At init0's and arch-wide's configs, at seed 2, the ablated stack starts as the equivariant stack's
function: the same random state after construction, the same encoder features, actor means and Q
values, and the same first critic step. Its parameters are exactly the equivariant layers' expanded
weights. After a few updates at the arm's own rates it is no longer equivariant, while the
equivariant stack still is: so the equivariance tests can fail, and an ablated arm is unconstrained
from its first update. A layer with no plain form refuses to export.
"""

from __future__ import annotations

import contextlib
import os

import numpy as np
import pytest
import torch

from tests.conftest import N, TOL

FAMILIES = ("encoder_family", "actor_family", "critic_family")


def _launches():
    from tests.test_sweep_arms import ARCH_WIDE, IFACE_ARMS, INIT0_LONG, LAUNCHED, _launch_order

    return {"init0": _launch_order(LAUNCHED[INIT0_LONG]), "arch-wide": _launch_order(LAUNCHED[ARCH_WIDE]),
            **{arm: _launch_order(LAUNCHED[arm]) for arm in IFACE_ARMS if arm != "iface-eee"}}


def _build(launch, family: str, device, **paths):
    """The agent the trainer builds from `launch` with every network of `family`, at seed 2, and
    torch's and numpy's random states just after."""
    from omegaconf import open_dict

    from tests.test_sweep_arms import _compose, _seeded_agent

    cfg = _compose(launch)
    with open_dict(cfg):
        for name in FAMILIES:
            cfg.equivariance[name] = family
        for path, value in paths.items():
            node = cfg
            *parents, leaf = path.split(".")
            for parent in parents:
                node = node[parent]
            node[leaf] = value
    with contextlib.redirect_stdout(open(os.devnull, "w")):  # QAgent prints its modules
        agent = _seeded_agent(cfg, device, seed=2)
    return agent, (torch.get_rng_state(), np.random.get_state()[1].copy())


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("config", ["init0", "arch-wide", "iface-wide-all", "iface-wide-out", "iface-window"])
def test_ablated_starts_as_the_equivariant_function(config, device, no_tf32):
    """Exporting draws no random number, so torch and numpy are where the equivariant build leaves
    them, and every update after draws alike. The actor's last layer is at escnn's own init here: at
    zero, both actors output zeros, which would compare nothing. Measured on the GPU in full
    precision (2026-10-20): every output exactly equal. On the CPU the probe found up to 7e-7 in Q
    (outputs/diagnostics/arch-swap-probe-1020/), where a LayerNorm's fused scale and shift round
    otherwise, hence the tolerance."""
    from tools.agent_fingerprint import STDDEV, UPDATE_SEED, _seed, fixed_batch

    launch = _launches()[config]
    unzeroed = {"agent.actor.actor_last_layer_init_scale": None}
    equi, equi_rng = _build(launch, "equivariant", device, **unzeroed)
    ablated, ablated_rng = _build(launch, "ablated", device, **unzeroed)
    assert torch.equal(equi_rng[0], ablated_rng[0]) and np.array_equal(equi_rng[1], ablated_rng[1])

    equi.train(False)
    ablated.train(False)
    batch = fixed_batch(base_action=True).to(device)
    with torch.no_grad():
        (ea, ec), (aa, ac) = equi.enc(dict(batch["obs"])), ablated.enc(dict(batch["obs"]))
        assert torch.equal(ea.tensor, aa) and torch.equal(ec.tensor, ac)
        mu = equi.actor(ea, std=None)
        assert mu.abs().max() > 1e-3, "a zero mean would compare nothing"
        torch.testing.assert_close(ablated.actor(aa, std=None), mu, rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(ablated.critic(ac, batch["action"]), equi.critic(ec, batch["action"]),
                                   rtol=1e-5, atol=1e-6)

    losses = []
    for agent in (equi, ablated):
        agent.train(True)
        _seed(UPDATE_SEED)
        losses.append(agent.update(fixed_batch(base_action=True).to(device), stddev=STDDEV,
                                   update_actor=False)["train/critic_loss"])
    assert losses[1] == pytest.approx(losses[0], rel=1e-6), losses


def _expanded_size(module) -> int:
    """The number of weights the equivariant `module` computes with: each escnn layer's expanded
    weight and bias, each RegularLayerNorm's scale and shift per channel, and every plain parameter
    (the wrist CNN, TrivialLayerNorm)."""
    from escnn import nn as enn

    from resfit.rl_finetuning.equi_off_policy.rl.equi_rl_utils import RegularLayerNorm

    total, inside = 0, set()
    for m in module.modules():
        if isinstance(m, enn.Linear):
            total += m.out_type.size * m.in_type.size + (m.out_type.size if m.bias is not None else 0)
        elif isinstance(m, enn.R2Conv):
            k = m.kernel_size
            total += m.out_type.size * m.in_type.size * k * k + (m.out_type.size if m.bias is not None else 0)
        elif isinstance(m, RegularLayerNorm):
            total += 2 * m.in_type.size
        else:
            continue
        inside |= {id(p) for p in m.parameters()}
    return total + sum(p.numel() for p in module.parameters() if id(p) not in inside)


@pytest.mark.gpu
@pytest.mark.slow
def test_ablated_has_exactly_the_equivariant_layers_expanded_weights(device):
    """At arch-wide (2026-10-20): the agentview encoder 3.382M dense weights against escnn's 0.284M
    coefficients, the actor 26.9M against 3.36M, the critic 22.6M against 2.82M."""
    launch = _launches()["arch-wide"]
    equi, _ = _build(launch, "equivariant", device)
    ablated, _ = _build(launch, "ablated", device)
    counts = {}
    for part in ("enc", "actor", "critic"):
        dense = sum(p.numel() for p in getattr(ablated, part).parameters())
        assert dense == _expanded_size(getattr(equi, part)), part
        counts[part] = dense
    assert counts == {"enc": 3_725_436, "actor": 26_908_679, "critic": 22_586_634}, counts


@pytest.mark.gpu
@pytest.mark.slow
def test_ablated_is_not_equivariant_after_a_few_updates_and_the_equivariant_is(device, no_tf32):
    """The negative control: four updates at arch-wide's rates (the actor's 1e-6, the critic's
    1e-4), then the actor and critic on random features at every group element. Measured
    (2026-10-20): the equivariant actor's error 2.6e-7 and its critic's 5.0e-7; the ablated actor's
    2.3 (its last layer starts at zero, so its whole output is unconstrained) and its critic's 0.41."""
    from escnn import nn as enn

    from tests import group_action as ga
    from tests.test_equivariance import _rel_err
    from tools.agent_fingerprint import STDDEV, UPDATE_SEED, _seed, fixed_batch

    errors = {}
    for family in ("equivariant", "ablated"):
        agent, _ = _build(_launches()["arch-wide"], family, device)
        agent.train(True)
        _seed(UPDATE_SEED)
        for _ in range(4):
            agent.update(fixed_batch(base_action=True).to(device), stddev=STDDEV, update_actor=True)
        agent.train(False)

        eq = agent.equi_cfg
        actor_layout = ga.enc_out_layout_actor(eq.enc_degree_channel, eq.ih_out_channel, 1, 2, 1)
        critic_layout = ga.enc_out_layout_critic(eq.enc_degree_channel, eq.ih_out_channel, 1, 2)
        act_layout = ga.action_layout(1, 1)
        g = torch.Generator().manual_seed(21)
        feat_a = torch.randn(4, ga.layout_width(actor_layout, N), generator=g).to(device)
        feat_c = torch.randn(4, ga.layout_width(critic_layout, N), generator=g).to(device)
        act = (0.3 * torch.randn(4, 7, generator=g)).to(device)

        def as_input(x, field_type, family=family):
            return enn.GeometricTensor(x, field_type) if family == "equivariant" else x

        with torch.no_grad():
            mu = agent.actor(as_input(feat_a, agent.enc.enc_out_type_actor), std=None)
            q = agent.critic(as_input(feat_c, agent.enc.enc_out_type_critic), act)
            assert mu.abs().max() > 0 and q.abs().max() > 0
            errors[family] = (
                max(_rel_err(agent.actor(as_input(ga.act_on_layout(feat_a, actor_layout, k, N),
                                                  agent.enc.enc_out_type_actor), std=None),
                             ga.act_on_layout(mu, act_layout, k, N)) for k in range(1, N)),
                max(_rel_err(agent.critic(as_input(ga.act_on_layout(feat_c, critic_layout, k, N),
                                                   agent.enc.enc_out_type_critic),
                                          ga.act_on_layout(act, act_layout, k, N)), q) for k in range(1, N)),
            )
    assert max(errors["equivariant"]) < TOL, errors
    assert min(errors["ablated"]) > 100 * TOL, errors


@pytest.mark.gpu
@pytest.mark.parametrize("part, norm", [("actor", "iid_bn"), ("actor", "inner_bn"), ("actor", "field_norm"),
                                        ("critic", "iid_bn"), ("critic", "field_norm"), ("encoder", "inner_bn")])
def test_a_layer_with_no_plain_form_refuses_to_export(obs_enc, part, norm, device):
    """The batch norms and FieldNorm compute with statistics of their own: no plain layer computes the
    same, so the ablated family refuses them at construction rather than build something else."""
    from tests.conftest import CROP, ENC_HIDDEN, IH_OUT, OBS_SHAPE, build_actor, build_critic, wrist_keys

    from resfit.rl_finetuning.equi_off_policy.networks.obs_encoder import ResObsEnc
    from resfit.rl_finetuning.equi_off_policy.rl.ablated import AblatedActor, AblatedCritic

    with pytest.raises(TypeError, match="has no plain form"):
        if part == "actor":
            AblatedActor(build_actor(obs_enc, device, norm=norm))
        elif part == "critic":
            AblatedCritic(build_critic(obs_enc, device, norm=norm))
        else:
            ResObsEnc(obs_shape=OBS_SHAPE, crop_shape=(CROP, CROP), N=N, n_hidden=ENC_HIDDEN,
                      ih_out_channel=IH_OUT, encoder="ablated", encoder_norm=norm, wrist_keys=wrist_keys(1))


@pytest.mark.gpu
def test_the_ablated_critic_reads_the_window_as_the_equivariant_one(device, no_tf32):
    """enc-iface-1023's window (2026-10-23): its projection starts at zero, so the start test above
    cannot see the exported projection's path. Moved off zero before the export, the two critics give
    the same Q, the same change in Q with the window, and the same window share."""
    from escnn import nn as enn

    from resfit.rl_finetuning.equi_off_policy.networks.obs_encoder import ResObsEnc
    from resfit.rl_finetuning.equi_off_policy.rl.ablated import AblatedCritic
    from tests.conftest import BATCH, ENC_HIDDEN, IH_OUT, build_critic

    torch.manual_seed(0)
    enc = ResObsEnc(N=N, n_hidden=ENC_HIDDEN, ih_out_channel=IH_OUT, encoder="equivariant",
                    critic_window=4, critic_window_ih=8).to(device)
    critic = build_critic(enc, device, window_type=enc.window_type, norm="layer_norm", num_layers=2)
    with torch.no_grad():
        critic.window_proj.weights.normal_()
    ablated = AblatedCritic(critic).to(device).eval()
    critic.eval()
    g = torch.Generator().manual_seed(1)
    feat = torch.randn(BATCH, enc.enc_out_dim_critic, generator=g).to(device)
    act = torch.randn(BATCH, enc.action_dim, generator=g).to(device)
    moved = feat.clone()
    images = enc.vis_ih_type.size
    moved[:, images:images + enc.window_type.size] += 1.0
    with torch.no_grad():
        q, qa = critic(enn.GeometricTensor(feat, enc.enc_out_type_critic), act), ablated(feat, act)
        q2, qa2 = critic(enn.GeometricTensor(moved, enc.enc_out_type_critic), act), ablated(moved, act)
        torch.testing.assert_close(qa, q, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(qa2, q2, rtol=1e-5, atol=1e-6)
        assert (q2 - q).abs().max() > 1e-4
        share = critic.window_share(enn.GeometricTensor(feat, enc.enc_out_type_critic))
        torch.testing.assert_close(ablated.window_share(feat), share, rtol=1e-5, atol=1e-7)
