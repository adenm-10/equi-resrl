"""Tier 1: when BasePolicyVecEnvWrapper resets the base policy.

ACT keeps one action queue per env, so only the envs whose episode ended are
reset. Diffusion's queues are batched, and every Can and Square run reset all
of them on any termination. Until 2026-09-27 the ACT branch raised NameError
on the first successful episode (TODO B3), so no ACT run got past one.
"""

from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from resfit.rl_finetuning.equi_off_policy.networks.equi_normalizer import (
    IdentityScaler,
    IdentityStandardizer,
)
from resfit.rl_finetuning.wrappers.residual_env_wrapper import BasePolicyVecEnvWrapper

NUM_ENVS = 3
STATE_DIM = 9
ACTION_DIM = 7
IMAGE_KEY = "observation.images.agentview"
NONE = (False, False, False)


class FakeVecEnv:
    """Steps through a script of per-env (terminated, truncated) flags."""

    def __init__(self, script):
        self.script = list(script)
        self.observation_space = gym.spaces.Dict({
            "observation.state": gym.spaces.Box(-np.inf, np.inf, (NUM_ENVS, STATE_DIM), np.float32),
            IMAGE_KEY: gym.spaces.Box(0, 255, (NUM_ENVS, 3, 84, 84), np.uint8),
        })
        self.action_space = gym.spaces.Box(-1, 1, (NUM_ENVS, ACTION_DIM), np.float32)

    def _obs(self) -> dict[str, torch.Tensor]:
        return {
            "observation.state": torch.zeros(NUM_ENVS, STATE_DIM),
            IMAGE_KEY: torch.zeros(NUM_ENVS, 3, 84, 84, dtype=torch.uint8),
        }

    def reset(self, **kwargs):
        return self._obs(), {}

    def step(self, action):
        terminated, truncated = self.script.pop(0)
        return self._obs(), torch.zeros(NUM_ENVS), torch.tensor(terminated), torch.tensor(truncated), {}


class FakePolicy:
    """Records each reset: None for all envs, otherwise the env indices."""

    def __init__(self, name: str):
        self.config = SimpleNamespace(name=name, image_features={IMAGE_KEY: None})
        self.calls: list[list[int] | None] = []

    def select_action(self, obs):
        return torch.zeros(NUM_ENVS, ACTION_DIM)

    def reset(self, env_ids=None):
        self.calls.append(None if env_ids is None else [int(i) for i in env_ids])


def _reset_calls(policy_name: str, script) -> list[list[int] | None]:
    """Reset the wrapper, then step it once per script entry."""
    policy = FakePolicy(policy_name)
    env = BasePolicyVecEnvWrapper(FakeVecEnv(script), policy, IdentityScaler(), IdentityStandardizer())
    env.reset()
    for _ in script:
        env.step(torch.zeros(NUM_ENVS, ACTION_DIM))
    return policy.calls


def test_act_resets_only_the_terminated_envs():
    script = [((False, True, False), NONE), ((True, False, True), NONE)]
    # The leading None is wrapper.reset().
    assert _reset_calls("act", script) == [None, [1], [0, 2]]


def test_diffusion_resets_all_envs_on_any_termination():
    assert _reset_calls("diffusion", [((False, True, False), NONE)]) == [None, None]


@pytest.mark.parametrize("name", ["act", "diffusion"])
def test_no_reset_without_termination(name):
    assert _reset_calls(name, [(NONE, NONE)]) == [None]


@pytest.mark.parametrize("name", ["act", "diffusion"])
def test_truncation_does_not_reset_the_base_policy(name):
    """Pins a known limitation rather than endorsing it: after a timeout the
    base policy carries its old chunk into the next episode. Changing this
    changes what every Can and Square run did."""
    assert _reset_calls(name, [(NONE, (True, False, False))]) == [None]
