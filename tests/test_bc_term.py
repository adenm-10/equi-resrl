"""The BC term on the actor, as train_residual_td3.py passes it since 2026-10-06 (TODO B14).

With agent.bc_loss_coef > 0, the trainer hands each actor update the batch's demo half, and
QAgent.update_actor_rft adds bc_loss_coef x sum over axes of (pi(s) - a_demo)^2, averaged over the
demo rows, to the actor's loss. Upstream's train_rlpd_dexmg.py does the same. At 0 the trainer
passes no demo half, and the actor takes QAgent.update_actor, which tests/test_regression_agent.py
pins bit for bit.

The term must pull the actor toward the demos' actions, and the update must still log the
critic's own slope at the policy's action: the tuned-RLPD diagnostics read it
(EXPERIMENTS.md, 2026-10-06 RESULTS, the screen).
"""

from __future__ import annotations

import math

import pytest
import torch

NUM_ONLINE = 2  # of tools/agent_fingerprint.BATCH = 4 rows; the rest are the demo half


def _agent(bc_loss_coef: float, device, bc_backprop_encoder: bool = False):
    from resfit.rl_finetuning.config.residual_td3 import TunedRLPDCanConfig
    from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
    from tools.agent_fingerprint import ACTION_DIM, CAMERAS, OBS_SHAPE, STATE_DIM

    cfg = TunedRLPDCanConfig().agent
    cfg.device = str(device)
    cfg.bc_loss_coef = bc_loss_coef
    cfg.bc_backprop_encoder = bc_backprop_encoder
    torch.manual_seed(0)
    agent = QAgent(obs_shape=OBS_SHAPE, prop_shape=(STATE_DIM,), action_dim=ACTION_DIM,
                   rl_cameras=list(CAMERAS), cfg=cfg, residual_actor=False)
    return agent.to(device)


def _batches(device):
    """The trainer's batch and, separately sampled as the trainer's offline_batch is, its demo half."""
    from tools.agent_fingerprint import fixed_batch

    return (fixed_batch(base_action=False).to(device),
            fixed_batch(base_action=False)[NUM_ONLINE:].to(device))


def _demo_gap(agent, device) -> float:
    """Mean |pi(s) - a_demo| per axis on the demo rows, unaugmented."""
    from resfit.rl_finetuning.off_policy.common_utils import utils

    _, demo = _batches(device)
    with torch.no_grad(), utils.eval_mode(agent):
        obs = {k: demo["obs"][k] for k in [*agent.rl_cameras, "observation.state"]}
        obs["feat"] = agent._encode(obs, augment=False)
        return (agent.actor(obs, 0.0).mean - demo["action"]).abs().mean().item()


@pytest.mark.gpu
def test_bc_update_logs_both_terms(device):
    from tools.agent_fingerprint import STDDEV

    agent = _agent(0.025, device)
    agent.train(True)
    batch, demo = _batches(device)
    metrics = agent.update(batch, STDDEV, update_actor=True, bc_batch=demo, ref_agent=agent)
    assert math.isfinite(metrics["train/actor_loss_base"])
    assert metrics["rft/bc_loss"] > 0
    assert metrics["rft/ratio"] == 1, "bc_loss_dynamic is off"
    assert metrics["debug/dQ_da_mean_abs"] > 0


@pytest.mark.gpu
def test_bc_term_pulls_the_actor_toward_the_demo_actions(device):
    """The same updates with and without the term: only the term closes the gap.

    On this batch the plain actor drifts away from the demo actions (about 0.50 to 0.86 per axis
    in 100 updates, as the tuned-RLPD runs' actors do), and the term's actor moves toward them
    (to about 0.37)."""
    from tools.agent_fingerprint import STDDEV

    gaps = {}
    for coef in (0.0, 0.025):
        agent = _agent(coef, device)
        before = _demo_gap(agent, device)
        agent.train(True)
        for _ in range(100):
            batch, demo = _batches(device)
            agent.update(batch, STDDEV, update_actor=True,
                         bc_batch=demo if coef > 0 else None, ref_agent=agent)
        gaps[coef] = (before, _demo_gap(agent, device))
    (start, plain), (start_bc, with_bc) = gaps[0.0], gaps[0.025]
    assert start == pytest.approx(start_bc), "both agents start from the same weights"
    assert with_bc < 0.9 * start, gaps
    assert plain > start, gaps


@pytest.mark.gpu
@pytest.mark.parametrize("backprop", [False, True], ids=["detached", "backprop"])
def test_bc_step_trains_the_encoder_only_when_asked(backprop, device):
    """agent.bc_backprop_encoder: the actor's BC step moves the encoder iff it is set.

    The step's Q term never does: QAgent.update hands the actor detached features, as here.
    So with the flag set, the encoder learns from the critic's TD loss and the BC loss, and
    never from the actor's Q loss.
    """
    from tools.agent_fingerprint import STDDEV

    agent = _agent(0.025, device, bc_backprop_encoder=backprop)
    agent.train(True)
    batch, demo = _batches(device)
    obs = batch["obs"]
    obs["feat"] = agent._encode(obs, augment=True).detach()
    before = [p.detach().clone() for p in agent.encoders.parameters()]
    metrics = agent.update_actor_rft(obs, STDDEV, demo, agent)
    moved = any(not torch.equal(b, p) for b, p in zip(before, agent.encoders.parameters()))
    assert moved == backprop
    assert ("train/encoder_grad_norm" in metrics) == backprop
