# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Environment wrapper that includes a base policy, enabling residual RL training
to be done in a standard way without explicit base policy handling in the training loop.

This wrapper assumes:
- The environment is already vectorized (batched)
- All inputs/outputs are torch tensors
- The environment comes from create_vectorized_env
"""

import gymnasium as gym
import numpy as np
import torch

from resfit.dexmg.environments.dexmg import VectorizedEnvWrapper
from resfit.lerobot.policies.act.modeling_act import ACTPolicy


class BasePolicyVecEnvWrapper:
    """
    Wraps a vectorized environment with a base policy to enable standard RL training of residual policies.

    This wrapper:
    1. Takes raw observations from the vectorized environment
    2. Passes them through the base policy to get base actions
    3. Augments observations with base actions for the residual policy
    4. Combines base + residual actions before stepping the environment
    5. Returns augmented observations that include base actions

    Assumes the environment is already vectorized and works with torch tensors.
    """

    def __init__(
        self,
        vec_env: VectorizedEnvWrapper,
        base_policy: ACTPolicy,
        action_scaler,
        state_standardizer,
    ):
        """
        Args:
            vec_env: Vectorized environment from create_vectorized_env
            base_policy: Base policy (e.g., ACTPolicy) to augment with residual actions
            action_scaler: ActionScaler object for scaling/unscaling actions (REQUIRED)
            state_standardizer: StateStandardizer object for standardizing states (REQUIRED)
        """
        assert action_scaler is not None, "action_scaler is required for consistent normalization"
        assert state_standardizer is not None, "state_standardizer is required for consistent normalization"

        self.vec_env = vec_env
        self.base_policy = base_policy
        self.action_scaler = action_scaler
        self.state_standardizer = state_standardizer

        # Get action dimension from the environment
        self.action_dim = vec_env.action_space.shape[-1]

        # Store image keys from base policy config
        self.image_keys = list(base_policy.config.image_features.keys())

        # Create modified observation space that includes base actions
        self._setup_observation_space()

    def _setup_observation_space(self):
        """Setup observation space to include base actions in the state."""

        # Get original observation space
        orig_obs_space = self.vec_env.observation_space

        # Copy the action space
        self.action_space = self.vec_env.action_space

        # Create new observation space with augmented state
        obs_spaces = {}
        for key, space in orig_obs_space.spaces.items():
            if key == "observation.state":
                # Augment state dimension with base actions
                orig_shape = list(space.shape)
                new_shape = orig_shape.copy()
                # new_shape[-1] += self.action_dim  # Not anymore
                obs_spaces[key] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=tuple(new_shape), dtype=space.dtype)
            else:
                # Keep other observations unchanged
                obs_spaces[key] = space

        self.observation_space = gym.spaces.Dict(obs_spaces)

    def reset(self, **kwargs) -> tuple[dict[str, torch.Tensor], dict]:
        """Reset environment and base policy."""
        # Reset the underlying vectorized environment
        raw_obs, info = self.vec_env.reset(**kwargs)

        # Reset base policy
        self.base_policy.reset()

        # Get base action from the base policy
        with torch.no_grad():
            base_action = self.base_policy.select_action(raw_obs)

        base_naction = self.action_scaler.scale(base_action)

        # Augment observations with base action and apply state standardization
        augmented_obs = self._augment_obs(raw_obs, base_naction)

        # Store for later use in step
        self._last_base_naction = base_naction

        return augmented_obs, info

    def step(
        self, residual_naction: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Step the environment with residual action.

        Args:
            residual_action: The residual action from the residual policy

        Returns:
            augmented_obs: Observations augmented with base actions
            reward: Reward tensor
            terminated: Terminated tensor
            truncated: Truncated tensor
            info: Info dict
        """
        # Combine base and residual actions
        # Residual action is already scaled inside the Actor class
        # To ensure that we can use the same exploration for all dimensions,
        # we use the normalized actions as the action space
        # The normalized base action is stored as [-1, 1] in the replay buffer
        # and the residual action is predicted as action_scale * [-1, 1]
        combined_naction = self._last_base_naction + residual_naction

        # Unscale back to original action space for environment execution
        env_action = self.action_scaler.unscale(combined_naction)

        # Step the underlying vectorized environment
        raw_obs, reward, terminated, truncated, info = self.vec_env.step(env_action)

        # Store the scaled action for replay buffer (already computed above)
        info["scaled_action"] = combined_naction

        # Get next base action from the base policy
        with torch.no_grad():
            base_action = self.base_policy.select_action(raw_obs)

        base_naction = self.action_scaler.scale(base_action)

        # Handle policy reset for terminated environments
        if terminated.any():
            reset_ids = torch.where(terminated)[0]
            if   self.base_policy.config.name == 'diffusion': self.base_policy.reset()
            elif self.base_policy.config.name == 'act':       self.base_policy.reset(env_ids=terminated_envs)
            # self.base_policy.reset(env_ids=reset_ids)

        # Augment observations with base action and apply state standardization
        augmented_obs = self._augment_obs(raw_obs, base_naction)

        # Handle final_obs in info dict to ensure consistent shapes
        if "final_obs" in info:
            info = self._process_final_obs_in_info(info, combined_naction.device)

        # Store for next step
        self._last_base_naction = base_naction

        return augmented_obs, reward, terminated, truncated, info

    def _augment_obs(self, raw_obs: dict[str, torch.Tensor], base_naction: torch.Tensor) -> dict[str, torch.Tensor]:
        """Augment observations with base actions."""

        # New way to do this is to just add the base action to the state under its own key
        augmented_obs = raw_obs.copy()
        augmented_obs["observation.base_action"] = base_naction
        augmented_obs["observation.state"] = self.state_standardizer.standardize(augmented_obs["observation.state"])

        return augmented_obs

    def _process_final_obs_in_info(self, info: dict, device: torch.device) -> dict:
        """Pad final_obs state with zeros to match augmented observation format."""
        if "final_obs" not in info or info["final_obs"] is None:
            return info

        for final_obs_dict in info["final_obs"]:
            if final_obs_dict is not None and "observation.state" in final_obs_dict:
                # Pad with zeros (no action taken at terminal state)
                final_obs_dict["observation.base_action"] = torch.zeros(
                    self.action_dim, device=device, dtype=torch.float32
                )

        return info

    def render(self):
        """Pass through to underlying environment."""
        return self.vec_env.render()

    def close(self):
        """Close the environment."""
        return self.vec_env.close()

    # Pass through any other attributes/methods to the underlying environment
    def __getattr__(self, name: str):
        """Delegate unknown attributes to the underlying vectorized environment."""
        return getattr(self.vec_env, name)

    def validate_normalizers(self):
        """Verify that normalizers behave as expected (identity or otherwise)."""
        print("\n" + "="*60)
        print("WRAPPER NORMALIZER VALIDATION")
        print("="*60)
        
        # 1. Reset and capture raw vs augmented state
        raw_obs, _ = self.vec_env.reset()
        self.base_policy.reset()
        
        with torch.no_grad():
            base_action = self.base_policy.select_action(raw_obs)
        
        raw_state = raw_obs["observation.state"].clone()
        scaled_base = self.action_scaler.scale(base_action)
        standardized_state = self.state_standardizer.standardize(raw_state.clone())
        
        state_diff = (standardized_state - raw_state).abs().max().item()
        print(f"  State standardizer max diff from raw: {state_diff:.8f}")
        
        # 2. Scale → unscale round-trip
        unscaled_base = self.action_scaler.unscale(scaled_base)
        roundtrip_diff = (unscaled_base - base_action).abs().max().item()
        print(f"  Action scale→unscale round-trip diff: {roundtrip_diff:.8f}")
        
        # 3. Zero-residual test (most important)
        augmented_obs, _ = self.reset()
        base_naction = self._last_base_naction.clone()
        
        zero_residual = torch.zeros_like(base_naction)
        _, _, _, _, info = self.step(zero_residual)
        
        combined = info["scaled_action"]
        zero_diff = (combined - base_naction).abs().max().item()
        print(f"  Zero residual → combined == base? max diff: {zero_diff:.8f}")
        
        if zero_diff > 1e-6:
            print(f"  ⚠️  FAILED: zero residual should give base action exactly")
        else:
            print(f"  ✅ Zero residual gives base action")
        
        # 4. Check what env actually received
        # The env should have received unscale(base_naction + 0) = unscale(scale(base_action)) = base_action
        env_action = self.action_scaler.unscale(base_naction)
        env_vs_raw = (env_action - base_action[:base_naction.shape[0]]).abs().max().item()
        print(f"  Env received action vs raw base policy output diff: {env_vs_raw:.8f}")
        
        # 5. Print ranges for sanity
        print(f"\n  Raw state range: [{raw_state.min():.4f}, {raw_state.max():.4f}]")
        print(f"  Standardized state range: [{standardized_state.min():.4f}, {standardized_state.max():.4f}]")
        print(f"  Raw base action range: [{base_action.min():.4f}, {base_action.max():.4f}]")
        print(f"  Scaled base action range: [{scaled_base.min():.4f}, {scaled_base.max():.4f}]")
        
        print("="*60 + "\n")
        
        # Re-reset for clean state
        return self.reset()
