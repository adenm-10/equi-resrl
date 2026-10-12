"""Tier 1: the base-rate script's stub agent drives the trainer's evaluator.

eval_base_policy.py measures base rates by running run_dexmg_evaluation with
ZeroResidualAgent in place of QAgent. The risk is the stub drifting from what
the evaluator calls, so this runs the real evaluator on a scripted env.
"""

from __future__ import annotations

import pytest
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


class ScriptedTiltEnv(ScriptedEnv):
    """ScriptedEnv whose step() returns scripted gripper orientations: per step, per env, a
    (axis, degrees) rotation from the start, as the end-effector quaternion (x, y, z, w)."""

    def __init__(self, script, rotations):
        super().__init__(script)
        self.rotations = list(rotations)
        self._state = torch.zeros(NUM_ENVS, STATE_DIM)
        self._state[:, 6] = 1.0  # w: the identity rotation

    def _obs(self):
        return {"observation.state": self._state.clone(),
                "observation.base_action": torch.zeros(NUM_ENVS, ACTION_DIM)}

    def step(self, actions):
        for env, (axis, degrees) in enumerate(self.rotations.pop(0)):
            half = torch.deg2rad(torch.tensor(degrees / 2.0))
            q = torch.zeros(4)
            q[{"x": 0, "z": 2}[axis]] = torch.sin(half)
            q[3] = torch.cos(half)
            self._state[env, 3:7] = q
        return super().step(actions)


def test_the_evaluator_reports_each_episodes_largest_tilt_and_not_its_yaw():
    """The gripper's axis is measured against its axis at the episode's first step, on every
    observation the agent acts on; a yaw leaves the axis where it is; an env that has finished starts
    its next episode afresh (the envs auto-reset in the done step)."""
    identity = ("x", 0.0)
    env = ScriptedTiltEnv(
        script=[(None, None), (None, None), ("fail", "success"), ("success", None)],
        rotations=[(("x", 20.0), ("z", 90.0)), (("x", 40.0), ("z", 90.0)), (identity, identity),
                   (identity, identity)],
    )
    metrics, _ = run_dexmg_evaluation(
        env=env, agent=ZeroResidualAgent(action_dim=ACTION_DIM), num_episodes=3,
        equivariant=True, save_video=False, save_q_plots=False,
    )
    # Episodes: env 0 tilted to 40 degrees, env 1 only yawed (0), env 0 again from upright (0).
    assert metrics["eval/tilt_max_deg_mean"] == pytest.approx(40.0 / 3, abs=1e-3)
    assert metrics["eval/tilt_over_30deg_frac"] == pytest.approx(1 / 3)
