"""Tier 1: pin the whole equivariant agent at the Square reference config.

tests/test_regression_single_arm.py pins the encoder alone. This pins what a
run actually trains: parameters at construction, the eval-mode actor and
critic outputs, and one TD3 update of encoder, critic and actor, all measured
by tools/agent_fingerprint.py.

The Square screen's arms run at a newer commit than its reference seeds
(j04yoeui at 2611adb, 88i0lpc8 at b8e1de7). This test passing is what makes
the two comparable, in place of the freeze on equi_off_policy/.
"""

from __future__ import annotations

import pytest
import torch

from tools.agent_fingerprint import (
    MODULES,
    STDDEV,
    build_reference_agent,
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
RTOL = 1e-6


@pytest.fixture(scope="module")
def measured():
    return fingerprint(build_reference_agent())


def test_agent_fingerprint_unchanged(measured):
    assert flatten(measured) == pytest.approx(flatten(GOLDEN), rel=RTOL)


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
