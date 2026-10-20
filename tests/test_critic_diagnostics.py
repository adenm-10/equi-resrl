"""Both agents' critic diagnostics, and the ResFiT agent's dQ_da log.

critic_diagnostics splits the critic's Q by buffer for the trainer's log: on the online rows and
on the demo rows, Q of the data's action and Q of the policy's. It was added for the tuned-RLPD
screen (EXPERIMENTS.md, 2026-10-06), whose control run tests the reading that the critic values
the policy's saturated actions on demo states above the demos' own. It must report what it says
and change nothing a run computes: no parameter, no random draw.

debug/dQ_da_* read 0 at every step of every ResFiT run before 2026-10-06, because the action it
reads is not a leaf tensor and its gradient was never kept (TODO B13).
"""

from __future__ import annotations

import pytest
import torch

NUM_ONLINE = 2  # of tools/agent_fingerprint.BATCH = 4 rows


def _agent(residual: bool, device):
    from resfit.rl_finetuning.config.residual_td3 import ResidualTD3CanConfig, TunedRLPDCanConfig
    from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
    from tools.agent_fingerprint import ACTION_DIM, CAMERAS, OBS_SHAPE, STATE_DIM

    cfg = (ResidualTD3CanConfig() if residual else TunedRLPDCanConfig()).agent
    cfg.device = str(device)
    torch.manual_seed(0)
    agent = QAgent(obs_shape=OBS_SHAPE, prop_shape=(STATE_DIM,), action_dim=ACTION_DIM,
                   rl_cameras=list(CAMERAS), cfg=cfg, residual_actor=residual)
    return agent.to(device)


def _rng_states(device):
    states = [torch.get_rng_state()]
    if device.type == "cuda":
        states.append(torch.cuda.get_rng_state(device))
    return states


@pytest.mark.gpu
@pytest.mark.parametrize("residual", [False, True], ids=["vanilla", "residual"])
def test_critic_diagnostics_reports_q_by_buffer(residual, device):
    from resfit.rl_finetuning.off_policy.common_utils import utils
    from tools.agent_fingerprint import CAMERAS, fixed_batch

    agent = _agent(residual, device)
    batch = fixed_batch(base_action=residual).to(device)
    with utils.eval_mode(agent):
        metrics = agent.critic_diagnostics(batch, NUM_ONLINE)

        # The same quantities, computed here from the agent's parts.
        obs = {k: batch["obs"][k] for k in [*CAMERAS, "observation.state"]}
        if residual:
            obs["observation.base_action"] = batch["obs"]["observation.base_action"]
        with torch.no_grad():
            obs["feat"] = agent._encode(obs, augment=False)
            policy = agent.actor(obs, 0.0).mean
            if residual:
                policy = torch.clamp(obs["observation.base_action"] + policy, -1.0, 1.0)
            heads = agent.critic.cfg.num_q

            def q(action):
                per_head = agent.critic(obs["feat"], obs["observation.state"], action)
                assert per_head.shape[0] == heads
                return per_head.mean(dim=0).squeeze(-1)

            q_data, q_policy = q(batch["action"]), q(policy)
            gap = (policy - batch["action"]).abs().mean(dim=-1)

    expected = {}
    for name, rows in {"online": slice(0, NUM_ONLINE), "demo": slice(NUM_ONLINE, None)}.items():
        expected[f"debug/q_data_{name}"] = q_data[rows].mean().item()
        expected[f"debug/q_policy_{name}"] = q_policy[rows].mean().item()
        expected[f"debug/policy_data_gap_{name}"] = gap[rows].mean().item()
    assert metrics.keys() == expected.keys()
    for key, value in expected.items():
        assert metrics[key] == pytest.approx(value, abs=1e-6), key


@pytest.mark.gpu
def test_critic_diagnostics_skips_an_empty_demo_half(device):
    """An online-only batch (offline_fraction=0) has no demo rows to report."""
    from resfit.rl_finetuning.off_policy.common_utils import utils
    from tools.agent_fingerprint import BATCH, fixed_batch

    agent = _agent(False, device)
    with utils.eval_mode(agent):
        metrics = agent.critic_diagnostics(fixed_batch(base_action=False).to(device), BATCH)
    assert sorted(metrics) == ["debug/policy_data_gap_online", "debug/q_data_online", "debug/q_policy_online"]


@pytest.mark.gpu
@pytest.mark.parametrize("residual", [False, True], ids=["vanilla", "residual"])
def test_critic_diagnostics_changes_nothing(residual, device):
    """No parameter moves and no random number is drawn, so logging leaves a run's numbers alone."""
    from resfit.rl_finetuning.off_policy.common_utils import utils
    from tools.agent_fingerprint import fixed_batch

    agent = _agent(residual, device)
    batch = fixed_batch(base_action=residual).to(device)
    before = {k: v.clone() for k, v in agent.state_dict().items()}
    rng = _rng_states(device)
    with utils.eval_mode(agent):
        agent.critic_diagnostics(batch, NUM_ONLINE)
    assert agent.training, "eval_mode restores train mode"
    for got, want in zip(_rng_states(device), rng):
        assert torch.equal(got, want), "a random number was drawn"
    for key, value in agent.state_dict().items():
        assert torch.equal(value, before[key]), f"{key} changed"


@pytest.mark.gpu
@pytest.mark.parametrize("residual", [False, True], ids=["vanilla", "residual"])
def test_dq_da_is_logged(residual, device):
    from tools.agent_fingerprint import STDDEV, fixed_batch

    agent = _agent(residual, device)
    agent.train(True)
    metrics = agent.update(fixed_batch(base_action=residual).to(device), STDDEV, update_actor=True)
    assert metrics["debug/dQ_da_norm"] > 0
    assert metrics["debug/dQ_da_mean_abs"] > 0


# ---------------------------------------------------------------------------
# The equivariant agent's, from 2026-10-07
# ---------------------------------------------------------------------------

# The equivariant design of the 2026-10-07 baselines (TODO B15): the squash, 10 heads, two critic
# layers with field norm, no actor norm.
EQUI_DESIGN = {
    "equivariance.residual_squash": True,
    "agent.critic.num_q": 10,
    "equivariance.num_critic_layers": 2,
    "equivariance.critic_norm": "field_norm",
    "equivariance.actor_norm": "none",
}


def _equi_agent(residual: bool, device):
    from tools.agent_fingerprint import build_reference_agent

    return build_reference_agent(device=str(device), overrides=EQUI_DESIGN, residual_actor=residual)


@pytest.mark.gpu
@pytest.mark.parametrize("residual", [False, True], ids=["vanilla", "residual"])
def test_equivariant_critic_diagnostics_report_q_by_buffer(residual, device):
    from resfit.rl_finetuning.equi_off_policy.rl.equi_rl_utils import equi_clip
    from resfit.rl_finetuning.off_policy.common_utils import utils
    from tools.agent_fingerprint import CAMERAS, fixed_batch

    agent = _equi_agent(residual, device)
    batch = fixed_batch(base_action=residual).to(device)
    with utils.eval_mode(agent):
        metrics = agent.critic_diagnostics(batch, NUM_ONLINE)

        keys = [*CAMERAS, "observation.state"] + (["observation.base_action"] if residual else [])
        obs = {k: batch["obs"][k] for k in keys}
        with torch.no_grad():
            obs["actor_feat"], obs["critic_feat"] = agent.enc(obs)
            layout = agent.actor.action_layout
            policy = equi_clip(agent.actor(obs["actor_feat"], std=None), layout, bound=1.0)
            # The action the actor loss scores, by its std=0 path: the same, bit for bit.
            assert torch.equal(policy, agent._act_default(obs=obs, eval_mode=True, stddev=0.0,
                                                          use_target=False))
            if residual:
                policy = equi_clip(obs["observation.base_action"] + policy, layout, bound=1.0)

            def q(action):
                per_head = agent.critic(obs["critic_feat"], action)
                assert per_head.shape[0] == EQUI_DESIGN["agent.critic.num_q"]
                return per_head.squeeze(-1).mean(dim=0)

            q_data, q_policy = q(batch["action"]), q(policy)
            gap = (policy - batch["action"]).abs().mean(dim=-1)

    expected = {}
    for name, rows in {"online": slice(0, NUM_ONLINE), "demo": slice(NUM_ONLINE, None)}.items():
        expected[f"debug/q_data_{name}"] = q_data[rows].mean().item()
        expected[f"debug/q_policy_{name}"] = q_policy[rows].mean().item()
        expected[f"debug/policy_data_gap_{name}"] = gap[rows].mean().item()
    assert metrics.keys() == expected.keys()
    for key, value in expected.items():
        assert metrics[key] == pytest.approx(value, abs=1e-6), key


@pytest.mark.gpu
def test_equivariant_critic_diagnostics_skip_an_empty_demo_half(device):
    from resfit.rl_finetuning.off_policy.common_utils import utils
    from tools.agent_fingerprint import BATCH, fixed_batch

    agent = _equi_agent(False, device)
    with utils.eval_mode(agent):
        metrics = agent.critic_diagnostics(fixed_batch(base_action=False).to(device), BATCH)
    assert sorted(metrics) == ["debug/policy_data_gap_online", "debug/q_data_online", "debug/q_policy_online"]


@pytest.mark.gpu
@pytest.mark.parametrize("residual", [False, True], ids=["vanilla", "residual"])
def test_equivariant_critic_diagnostics_change_nothing(residual, device):
    """No parameter moves and no random number is drawn. The actor's std=0 path would draw one:
    tests/test_regression_agent.py shows the updates after it would then differ."""
    from resfit.rl_finetuning.off_policy.common_utils import utils
    from tools.agent_fingerprint import fixed_batch

    agent = _equi_agent(residual, device)
    agent.train(True)
    batch = fixed_batch(base_action=residual).to(device)
    before = {k: v.clone() for k, v in agent.state_dict().items()}
    rng = _rng_states(device)
    with utils.eval_mode(agent):
        agent.critic_diagnostics(batch, NUM_ONLINE)
    assert agent.training, "eval_mode restores train mode"
    for got, want in zip(_rng_states(device), rng):
        assert torch.equal(got, want), "a random number was drawn"
    after = agent.state_dict()
    assert after.keys() == before.keys()
    for key, value in after.items():
        assert torch.equal(value, before[key]), f"{key} changed"
