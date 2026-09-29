"""Tier 1: the base-rate script's stub agent drives the trainer's evaluator.

eval_base_policy.py measures base rates by running run_dexmg_evaluation with
ZeroResidualAgent in place of QAgent. The risk is the stub drifting from what
the evaluator calls, so this runs the real evaluator on a scripted env.
"""

from __future__ import annotations

import torch

from resfit.rl_finetuning.scripts.eval_base_policy import ZeroResidualAgent
from resfit.rl_finetuning.utils.evaluate_dexmg import run_dexmg_evaluation

NUM_ENVS = 2
STATE_DIM = 9
ACTION_DIM = 7


class ScriptedEnv:
    """A wrapped env whose episodes end as scripted: per step, per env,
    None (still running), "success" (terminated, reward 1) or "fail" (timeout)."""

    num_envs = NUM_ENVS

    def __init__(self, script):
        self.script = list(script)
        self.actions = []

    def _obs(self):
        return {
            "observation.state": torch.zeros(NUM_ENVS, STATE_DIM),
            "observation.base_action": torch.zeros(NUM_ENVS, ACTION_DIM),
        }

    def reset(self, **kwargs):
        return self._obs(), {}

    def step(self, actions):
        self.actions.append(actions.clone())
        outcomes = self.script.pop(0)
        reward = torch.tensor([1.0 if o == "success" else 0.0 for o in outcomes])
        terminated = torch.tensor([o == "success" for o in outcomes])
        truncated = torch.tensor([o == "fail" for o in outcomes])
        return self._obs(), reward, terminated, truncated, {}


def test_stub_runs_the_evaluator_and_counts_like_training():
    # Three episodes wanted. Env 1's second episode is still running when the
    # third finishes, so it is dropped even though it would have succeeded.
    env = ScriptedEnv([
        ("success", None),
        (None, "fail"),
        ("success", None),
        (None, "success"),
    ])
    metrics, video = run_dexmg_evaluation(
        env=env, agent=ZeroResidualAgent(action_dim=ACTION_DIM), num_episodes=3,
        equivariant=True, save_video=False, save_q_plots=False,
    )

    assert metrics["eval/success_rate"] == 2 / 3
    assert video is None
    assert len(env.script) == 1, "evaluator should stop at the third finished episode"
    for a in env.actions:
        assert a.shape == (NUM_ENVS, ACTION_DIM)
        assert torch.count_nonzero(a) == 0, "the residual must be exactly zero"


def test_stub_act_matches_qagent_act():
    agent = ZeroResidualAgent(action_dim=ACTION_DIM)
    obs = {"observation.state": torch.zeros(NUM_ENVS, STATE_DIM)}
    assert agent.act(obs, eval_mode=True, stddev=0.0, cpu=True).shape == (NUM_ENVS, ACTION_DIM)
    assert agent.critic.q_value(None, torch.zeros(NUM_ENVS, ACTION_DIM)).shape == (NUM_ENVS, 1)
