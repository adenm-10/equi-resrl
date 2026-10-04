"""
Environment wrapper for vanilla RL (base_policy=None): BasePolicyVecEnvWrapper without the base
policy.

The actor outputs the whole action, in the normalized space. This wrapper unscales it for the
environment, standardizes the state, and records the action in info["scaled_action"], which the
trainer stores in the replay buffer. Upstream's train_rlpd_dexmg.py does the same with its local
NormalizedEnvWrapper. tests/test_env_wrapper.py checks that it steps exactly as the residual
wrapper does with a zero base action.

Like the residual wrapper, it leaves info["final_obs"] unstandardized. The trainer stores that
observation only as the next observation of a transition marked done, and a done transition is
never bootstrapped from, timeouts included.
"""

import gymnasium as gym
import numpy as np
import torch


class NormalizedVecEnvWrapper:
    """Normalizes a vectorized environment's states and actions, as BasePolicyVecEnvWrapper does."""

    def __init__(self, vec_env, action_scaler, state_standardizer):
        """
        Args:
            vec_env: Vectorized environment from create_vectorized_env
            action_scaler: ActionScaler, or IdentityScaler on the equivariant path
            state_standardizer: StateStandardizer, or IdentityStandardizer on the equivariant path
        """
        assert action_scaler is not None, "action_scaler is required for consistent normalization"
        assert state_standardizer is not None, "state_standardizer is required for consistent normalization"

        self.vec_env = vec_env
        self.action_scaler = action_scaler
        self.state_standardizer = state_standardizer

        # The spaces the residual wrapper exposes, less its base action: a standardized state
        # is unbounded.
        self.action_space = vec_env.action_space
        self.observation_space = gym.spaces.Dict(
            {
                key: gym.spaces.Box(low=-np.inf, high=np.inf, shape=space.shape, dtype=space.dtype)
                if key == "observation.state"
                else space
                for key, space in vec_env.observation_space.spaces.items()
            }
        )

    def reset(self, **kwargs) -> tuple[dict[str, torch.Tensor], dict]:
        raw_obs, info = self.vec_env.reset(**kwargs)
        return self._standardize(raw_obs), info

    def step(
        self, naction: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Step with the actor's normalized action; info["scaled_action"] is that action."""
        raw_obs, reward, terminated, truncated, info = self.vec_env.step(self.action_scaler.unscale(naction))
        info["scaled_action"] = naction
        return self._standardize(raw_obs), reward, terminated, truncated, info

    def _standardize(self, raw_obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        obs = raw_obs.copy()
        obs["observation.state"] = self.state_standardizer.standardize(obs["observation.state"])
        return obs

    def render(self):
        """Pass through to underlying environment."""
        return self.vec_env.render()

    def close(self):
        """Close the environment."""
        return self.vec_env.close()

    def __getattr__(self, name: str):
        """Delegate unknown attributes to the underlying vectorized environment."""
        return getattr(self.vec_env, name)
