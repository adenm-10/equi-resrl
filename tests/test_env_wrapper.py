"""Tier 1: the env wrappers.

When BasePolicyVecEnvWrapper resets the base policy: ACT keeps one action
queue per env, so only the envs whose episode ended are reset. Diffusion's
queues are batched, and every Can and Square run reset all of them on any
termination. Until 2026-09-27 the ACT branch raised NameError on the first
successful episode (TODO B3), so no ACT run got past one.

NormalizedVecEnvWrapper, vanilla RL's (base_policy=None), must step exactly
as the residual wrapper does when the base action normalizes to zero.
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
from resfit.rl_finetuning.utils.normalization import ActionScaler, StateStandardizer
from resfit.rl_finetuning.wrappers.normalized_env_wrapper import NormalizedVecEnvWrapper
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


# ---------------------------------------------------------------------------
# NormalizedVecEnvWrapper: the residual wrapper less its base policy
# ---------------------------------------------------------------------------

# Dyadic action bounds, so that scaling the middle of the range gives exactly zero.
ACTION_MIN = torch.tensor([-1.0, -3.0, 0.0, -2.0, -0.5, -4.0, 0.0])
ACTION_MAX = torch.tensor([3.0, 1.0, 2.0, 2.0, 1.5, 4.0, 1.0])


class RecordingVecEnv(FakeVecEnv):
    """FakeVecEnv with states, images and rewards that differ per step, which
    records every action it is given."""

    def __init__(self, script):
        super().__init__(script)
        self.actions: list[torch.Tensor] = []
        self.t = 0

    def _obs(self) -> dict[str, torch.Tensor]:
        self.t += 1
        g = torch.Generator().manual_seed(self.t)
        return {
            "observation.state": torch.randn(NUM_ENVS, STATE_DIM, generator=g) * 3 + 1,
            IMAGE_KEY: torch.randint(0, 256, (NUM_ENVS, 3, 84, 84), generator=g, dtype=torch.uint8),
        }

    def step(self, action):
        self.actions.append(action.clone())
        obs, _, terminated, truncated, info = super().step(action)
        return obs, torch.arange(NUM_ENVS, dtype=torch.float32) * self.t, terminated, truncated, info


class MidRangePolicy(FakePolicy):
    """A base policy whose action normalizes to exactly zero: the middle of the range."""

    def __init__(self, scaler: ActionScaler):
        super().__init__("diffusion")
        self.action = scaler.unscale(torch.zeros(NUM_ENVS, ACTION_DIM))

    def select_action(self, obs):
        return self.action.clone()


def test_vanilla_wrapper_steps_as_the_residual_wrapper_with_a_zero_base_action():
    # The vanilla runs' scaler: dataset bounds, no expansion (train_residual_td3.py).
    scaler = ActionScaler(ACTION_MIN, ACTION_MAX, action_scale=0.0)
    standardizer = StateStandardizer(torch.linspace(-1, 1, STATE_DIM), torch.linspace(0.5, 2, STATE_DIM))
    script = [(NONE, NONE), ((False, True, False), NONE), (NONE, (True, False, False)), (NONE, NONE)]
    residual_env, vanilla_env = RecordingVecEnv(script), RecordingVecEnv(script)
    residual = BasePolicyVecEnvWrapper(residual_env, MidRangePolicy(scaler), scaler, standardizer)
    vanilla = NormalizedVecEnvWrapper(vanilla_env, scaler, standardizer)

    obs_r, _ = residual.reset()
    obs_v, _ = vanilla.reset()
    assert torch.equal(residual._last_base_naction, torch.zeros(NUM_ENVS, ACTION_DIM))
    g = torch.Generator().manual_seed(0)
    for step in range(len(script)):
        assert set(obs_r) == set(obs_v) | {"observation.base_action"}
        for key in obs_v:
            assert torch.equal(obs_r[key], obs_v[key]), (step, key)
        # Past [-1, 1] too, so the unscale's clamp takes part.
        action = torch.rand(NUM_ENVS, ACTION_DIM, generator=g) * 2.4 - 1.2
        obs_r, reward_r, term_r, trunc_r, info_r = residual.step(action)
        obs_v, reward_v, term_v, trunc_v, info_v = vanilla.step(action)
        assert torch.equal(residual_env.actions[-1], vanilla_env.actions[-1]), "the env got another action"
        assert torch.equal(info_r["scaled_action"], info_v["scaled_action"]), "the buffer gets another action"
        assert torch.equal(info_v["scaled_action"], action)
        for got, expected in ((reward_v, reward_r), (term_v, term_r), (trunc_v, trunc_r)):
            assert torch.equal(got, expected)


def test_vanilla_wrapper_exposes_the_env():
    """What the trainer and the evaluator read through it."""
    fake = FakeVecEnv([])
    env = NormalizedVecEnvWrapper(fake, IdentityScaler(), IdentityStandardizer())
    assert env.vec_env is fake
    assert env.action_space is fake.action_space
    state = env.observation_space["observation.state"]
    assert state.shape == (NUM_ENVS, STATE_DIM) and np.isinf(state.low).all() and np.isinf(state.high).all()
    assert env.observation_space[IMAGE_KEY] is fake.observation_space[IMAGE_KEY]
    fake.render = lambda: "frames"
    assert env.render() == "frames"
