"""Tier 1: pin the whole equivariant agent at the Square reference config.

tests/test_regression_single_arm.py pins the encoder alone. This pins what a
run actually trains: parameters at construction, the eval-mode actor and
critic outputs, and one TD3 update of encoder, critic and actor, all measured
by tools/agent_fingerprint.py.

The Square screen's arms run at a newer commit than its reference seeds
(j04yoeui at 2611adb, 88i0lpc8 at b8e1de7). This test passing is what makes
the two comparable, in place of the freeze on equi_off_policy/.

The reference has batch norm in the actor, as every run before 2026-09-30 had.
A second golden pins the same agent at the actor_norm default from then on,
"none". This module also checks that skipping the metrics of an update that
will not be logged (QAgent.update's compute_metrics) changes nothing. Those
builds are on CPU, so they belong here: see the ordering note in conftest.py.
"""

from __future__ import annotations

import pytest
import torch

from tools.agent_fingerprint import (
    CANDIDATES,
    DEEP_UPDATES,
    MODULES,
    OPTIMIZERS,
    STDDEV,
    UPDATE_SEED,
    _leaves,
    _seed,
    build_reference_agent,
    compare_deep,
    fingerprint,
    fixed_batch,
    flatten,
)

# Captured at f3e6315 on 2026-09-27. `tools/agent_fingerprint.py --compare`
# found the b8e1de7 tree identical in all 22 values, exactly. If these move,
# agent behaviour changed: fix the change, or record it in docs/EXPERIMENTS.md.
GOLDEN = {
    "params": {
        "enc": [18.81238356381943, -27.68563702049253],
        "actor": [362.26152028833076, -20.68517458581647],
        "critic": [131.18920593832468, -42.17377269175855],
    },
    "forward": {
        "actor_feat": [1182.7750322069514, 37.444658355090624],
        "critic_feat": [1186.0099155451644, -88.07476794682242],
        "actor": [-1.3796153105261055e-05, -2.2013957315048377e-05],
        "critic": [4.699684351682663, 0.9916826388925024],
    },
    "update": {
        "critic_loss": 0.28064101934432983,
        "actor_loss_base": -0.6309514045715332,
        "enc_delta": [-1.7308517712621025, -0.007743229964953467],
        "actor_delta": [0.00047839740414601373, 0.0002474898355390846],
        "critic_delta": [0.26120524681743973, 0.04731270367694712],
    },
}
# The same agent with actor_norm="none". Captured on 2026-09-30 at the commit that
# made it the default; 1e33b8f gives the same 22 values, exactly.
GOLDEN_ACTOR_NONE = {
    "params": {
        "enc": [18.81238356381943, -27.68563702049253],
        "actor": [-21.73847971166923, 45.70448979202473],
        "critic": [131.18920593832468, -42.17377269175855],
    },
    "forward": {
        "actor_feat": [1182.7750322069514, 37.444658355090624],
        "critic_feat": [1186.0099155451644, -88.07476794682242],
        "actor": [-1.3796414187972061e-05, -2.2014305518294043e-05],
        "critic": [4.699684351682663, 0.9916826388925024],
    },
    "update": {
        "critic_loss": 0.2806415557861328,
        "actor_loss_base": -0.6309517025947571,
        "enc_delta": [-1.7308486234004405, -0.00774491479315881],
        "actor_delta": [0.0008693588376185168, 2.473117622748681e-05],
        "critic_delta": [0.2612024769986192, 0.04731162629106193],
    },
}
RTOL = 1e-6


@pytest.fixture(scope="module")
def measured():
    return fingerprint(build_reference_agent())


def test_agent_fingerprint_unchanged(measured):
    assert flatten(measured) == pytest.approx(flatten(GOLDEN), rel=RTOL)


def test_default_actor_fingerprint_unchanged():
    """The agent the runs from 2026-09-30 on train: the reference, minus the
    actor's batch norm."""
    measured = fingerprint(build_reference_agent(actor_norm="none"))
    assert flatten(measured) == pytest.approx(flatten(GOLDEN_ACTOR_NONE), rel=RTOL)


def _train(candidate: str, metrics_on_every_update: bool) -> dict:
    """The deep check's 8 updates, computing metrics on every update or on the
    last only, as the trainer does. Returns every tensor the agent holds, and
    the last update's metrics."""
    agent = build_reference_agent(overrides=CANDIDATES[candidate])
    agent.train(True)
    _seed(UPDATE_SEED)
    for k in range(DEEP_UPDATES):
        metrics = agent.update(
            fixed_batch(weights=True), stddev=STDDEV, update_actor=(k + 1) % 4 == 0,
            compute_metrics=metrics_on_every_update or k == DEEP_UPDATES - 1)
    out = _leaves("state_dict", agent.state_dict())
    for name in OPTIMIZERS:
        out.update(_leaves(name, getattr(agent, name).state_dict()))
    out.update(_leaves("metrics", metrics))
    return out


@pytest.mark.parametrize("candidate", ["target-fix", "all-four"])
def test_skipping_metrics_changes_nothing(candidate):
    """compute_metrics=False skips only values that are logged, bit for bit.

    target-fix has the actor's batch norm, whose running statistics any
    skipped forward pass would leave behind. all-four has every change of the
    2026-09-30 schedule at once, so every branch the metrics sit in runs."""
    assert compare_deep(_train(candidate, True), _train(candidate, False)) == []


def _train_diagnosed(candidate: str, diagnose: bool) -> dict:
    """The deep check's 8 updates, with or without the trainer's log-step critic diagnostics after
    each: QAgent.critic_diagnostics in eval mode, on the batch the update just used."""
    from resfit.rl_finetuning.off_policy.common_utils import utils

    agent = build_reference_agent(overrides=CANDIDATES[candidate])
    agent.train(True)
    _seed(UPDATE_SEED)
    for k in range(DEEP_UPDATES):
        batch = fixed_batch(weights=True)
        metrics = agent.update(batch, stddev=STDDEV, update_actor=(k + 1) % 4 == 0)
        if diagnose:
            with utils.eval_mode(agent):
                agent.critic_diagnostics(batch, num_online=2)
    out = _leaves("state_dict", agent.state_dict())
    for name in OPTIMIZERS:
        out.update(_leaves(name, getattr(agent, name).state_dict()))
    out.update(_leaves("metrics", metrics))
    return out


@pytest.mark.parametrize("candidate", ["target-fix", "all-four"])
def test_critic_diagnostics_change_no_update(candidate):
    """Logging the critic's Q by buffer leaves the equivariant agent's training bit for bit alone.

    The trainer calls it at every log step from 2026-10-07, in eval mode. An escnn layer expands
    its weights when it enters eval mode, and a batch norm must not update its running statistics
    there (target-fix has the actor's). all-four has the field-norm critic, two critic layers and
    the squash, the design of the 2026-10-07 baselines."""
    assert compare_deep(_train_diagnosed(candidate, True), _train_diagnosed(candidate, False)) == []


def test_fingerprint_is_not_vacuous(measured):
    """A zero actor, a zero loss, or a no-op update would pin nothing."""
    assert measured["forward"]["actor"][1] != 0.0, "actor output is zero"
    assert measured["update"]["critic_loss"] > 0.0
    for name in MODULES:
        assert any(v != 0.0 for v in measured["update"][f"{name}_delta"]), (
            f"one update left {name} unchanged")


@pytest.mark.parametrize("frozen", [False, True], ids=["fixed", "frozen"])
def test_target_critic_computes_with_its_soft_updated_weights(frozen):
    """The bug in every equivariant run before 2026-09-29, and its fix.

    The target critic sat in eval mode, where an escnn layer computes with the
    weights it expanded at .train(False), so soft updates moved its parameters
    and nothing else. With tau=1 one update copies the critic into the target:
    the two must now agree, and frozen_target_critic=True must still give the
    target's initial values. The fingerprint above cannot see this, because
    an update reads the target before it soft-updates it.
    """
    agent = build_reference_agent(frozen_target_critic=frozen)
    agent.agent_cfg.critic_target_tau = 1.0
    batch = fixed_batch()

    def q(critic):
        with torch.no_grad():
            return critic(feat, batch["action"])

    agent.train(False)
    with torch.no_grad():
        _, feat = agent.enc(batch["obs"])
    target_before = q(agent.critic_target)

    agent.train(True)
    agent.update(fixed_batch(), stddev=STDDEV, update_actor=False)

    agent.train(False)  # rebuilds the online critic's weights from its parameters
    online_after = q(agent.critic)
    assert not torch.allclose(online_after, target_before), "the update left the critic unchanged"
    expected = target_before if frozen else online_after
    torch.testing.assert_close(q(agent.critic_target), expected, rtol=0, atol=0)
