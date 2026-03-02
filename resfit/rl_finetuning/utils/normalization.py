# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Action scaling utilities for residual RL training.

This module provides a clean interface for min-max scaling of actions to [-1, 1] range,
with proper safeguards against numerical instabilities.
"""

from __future__ import annotations
from resfit.dexmg.environments.dexmg import create_vectorized_env
from typing import NamedTuple
import torch
import numpy as np

class ActionScaler:
    """
    Handles min-max scaling of actions to [-1, 1] range for residual RL training.

    This class encapsulates all the logic for:
    - Computing action limits from dataset statistics
    - Applying safeguards against numerical instabilities
    - Scaling actions to [-1, 1] and unscaling back to original range
    - Handling device placement automatically
    """

    class Limits(NamedTuple):
        """Action limits for scaling."""

        min: torch.Tensor
        max: torch.Tensor

    def __init__(
        self,
        action_min: torch.Tensor,
        action_max: torch.Tensor,
        action_scale: float = 1.0,
        min_range_per_dim: float = 1e-1,
        device: torch.device | str = "cpu",
    ):
        """
        Initialize the action scaler.

        Args:
            action_min: Minimum values from dataset statistics
            action_max: Maximum values from dataset statistics
            action_scale: Scale factor to expand the action range (1 + action_scale)
            min_range_per_dim: Minimum range per dimension to prevent normalization blow-up
            device: Device to place tensors on
        """
        self.device = torch.device(device)
        self.action_scale = action_scale
        self.min_range_per_dim = min_range_per_dim

        # Move inputs to device
        action_min = action_min.to(self.device)
        action_max = action_max.to(self.device)

        # Compute action center and half-range
        action_mid = (action_min + action_max) / 2
        action_half_range = (action_max - action_min) / 2

        # Apply safeguard: ensure minimum range to prevent normalization blow-up
        min_half_range = torch.tensor(min_range_per_dim / 2, device=self.device)
        action_half_range = torch.maximum(action_half_range, min_half_range)

        # Expand the range by action_scale factor
        expanded_half_range = action_half_range * (1 + action_scale)

        # Store the final limits
        self._limits = self.Limits(
            min=action_mid - expanded_half_range,
            max=action_mid + expanded_half_range,
        )

        # Precompute range for efficiency (with safeguard)
        self._range = self._limits.max - self._limits.min
        self._range = torch.maximum(self._range, torch.tensor(1e-8, device=self.device))

        print("ActionScaler initialized:")
        print(f"  Original range: [{action_min.min():.4f}, {action_max.max():.4f}]")
        print(f"  Expanded range: [{self._limits.min.min():.4f}, {self._limits.max.max():.4f}]")
        print(f"  Action scale factor: {action_scale}")

    @property
    def limits(self) -> Limits:
        """Get the action limits."""
        return self._limits

    def scale(self, action: torch.Tensor) -> torch.Tensor:
        """
        Scale action to [-1, 1] range.

        Args:
            action: Action tensor to scale

        Returns:
            Scaled action in [-1, 1] range
        """
        # Move limits to same device as input action
        action_min = self._limits.min.to(action.device)
        action_max = self._limits.max.to(action.device)
        range_vals = self._range.to(action.device)

        # Clamp input to prevent extreme values
        action_clamped = torch.clamp(action, action_min, action_max)

        # Scale to [-1, 1]
        return 2.0 * (action_clamped - action_min) / range_vals - 1.0

    def unscale(self, scaled_action: torch.Tensor) -> torch.Tensor:
        """
        Unscale action from [-1, 1] back to original range.

        Args:
            scaled_action: Scaled action in [-1, 1] range

        Returns:
            Unscaled action in original range
        """
        # Move limits to same device as input action
        action_min = self._limits.min.to(scaled_action.device)
        action_max = self._limits.max.to(scaled_action.device)

        # Clamp to [-1, 1] to prevent extreme unscaled values
        scaled_clamped = torch.clamp(scaled_action, -1.0, 1.0)

        # Unscale back to original range
        return action_min + (scaled_clamped + 1.0) * (action_max - action_min) / 2.0

    def to(self, device: torch.device | str) -> ActionScaler:
        """Move scaler to a different device."""
        new_scaler = ActionScaler.__new__(ActionScaler)
        new_scaler.device = torch.device(device)
        new_scaler.action_scale = self.action_scale
        new_scaler.min_range_per_dim = self.min_range_per_dim
        new_scaler._limits = self.Limits(
            min=self._limits.min.to(device),
            max=self._limits.max.to(device),
        )
        new_scaler._range = self._range.to(device)
        return new_scaler

    @classmethod
    def from_dataset_stats(
        cls,
        action_stats: dict,
        action_scale: float = 1.0,
        min_range_per_dim: float = 1e-1,
        device: torch.device | str = "cpu",
    ) -> ActionScaler:
        """
        Create ActionScaler from dataset statistics.

        Args:
            action_stats: Dictionary with 'min' and 'max' keys containing action statistics
            action_scale: Scale factor to expand the action range
            min_range_per_dim: Minimum range per dimension to prevent normalization blow-up
            device: Device to place tensors on

        Returns:
            Configured ActionScaler instance
        """
        action_min = torch.tensor(action_stats["min"], dtype=torch.float32)
        action_max = torch.tensor(action_stats["max"], dtype=torch.float32)

        return cls(
            action_min=action_min,
            action_max=action_max,
            action_scale=action_scale,
            min_range_per_dim=min_range_per_dim,
            device=device,
        )


class StateStandardizer:
    """
    Handles standardization of states to mean=0, std=1.

    This class provides a clean interface for state normalization with safeguards.
    """

    def __init__(
        self,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        min_std: float = 1e-1,
        device: torch.device | str = "cpu",
    ):
        """
        Initialize the state standardizer.

        Args:
            state_mean: Mean values from dataset statistics
            state_std: Standard deviation values from dataset statistics
            min_std: Minimum std to prevent normalization blow-up
            device: Device to place tensors on
        """
        self.device = torch.device(device)
        self.min_std = min_std

        # Move to device and apply safeguards
        self._mean = state_mean.to(self.device)
        self._std = torch.maximum(state_std.to(self.device), torch.tensor(min_std, device=self.device))

        print("StateStandardizer initialized:")
        print(f"  Mean range: [{self._mean.min():.4f}, {self._mean.max():.4f}]")
        print(f"  Std range: [{self._std.min():.6f}, {self._std.max():.6f}]")

    def standardize(self, state: torch.Tensor) -> torch.Tensor:
        """
        Standardize state to mean=0, std=1.

        Args:
            state: State tensor to standardize

        Returns:
            Standardized state
        """
        # Move stats to same device as input state
        mean = self._mean.to(state.device)
        std = self._std.to(state.device)

        # Additional runtime safeguard
        std_safe = torch.maximum(std, torch.tensor(1e-8, device=state.device))

        return (state - mean) / std_safe

    def to(self, device: torch.device | str) -> StateStandardizer:
        """Move standardizer to a different device."""
        new_standardizer = StateStandardizer.__new__(StateStandardizer)
        new_standardizer.device = torch.device(device)
        new_standardizer.min_std = self.min_std
        new_standardizer._mean = self._mean.to(device)
        new_standardizer._std = self._std.to(device)
        return new_standardizer

    @classmethod
    def from_dataset_stats(
        cls,
        state_stats: dict,
        min_std: float = 1e-1,
        device: torch.device | str = "cpu",
    ) -> StateStandardizer:
        """
        Create StateStandardizer from dataset statistics.

        Args:
            state_stats: Dictionary with 'mean' and 'std' keys containing state statistics
            min_std: Minimum std to prevent normalization blow-up
            device: Device to place tensors on

        Returns:
            Configured StateStandardizer instance
        """
        state_mean = torch.tensor(state_stats["mean"], dtype=torch.float32)
        state_std = torch.tensor(state_stats["std"], dtype=torch.float32)

        return cls(
            state_mean=state_mean,
            state_std=state_std,
            min_std=min_std,
            device=device,
        )


class EquivariantActionScaler:
    """
    Min-max action scaler to [-1, 1] that respects equivariant structure.

    For irrep(1) pairs the scale and offset are tied so that the linear
    map  a ↦ 2(a - a_min)/(a_max - a_min) - 1  commutes with the group
    action (planar rotation for SO(2) / discrete rotation for C_N).

    Expected single-arm action layout: [a_xy(2), a_z(1), a_rxry(2), a_rz(1), a_grip(1)]
      - dims 0,1  (a_xy):   irrep(1) pair  → shared scale & center
      - dim  2     (a_z):   trivial         → per-dim
      - dims 3,4  (a_rxry): irrep(1) pair  → shared scale & center
      - dim  5     (a_rz):  trivial         → per-dim
      - dim  6     (a_grip):trivial         → per-dim
    """

    class Limits(NamedTuple):
        min: torch.Tensor
        max: torch.Tensor

    def __init__(
        self,
        action_min: torch.Tensor,
        action_max: torch.Tensor,
        equivariant_pairs: list[list[int]],
        action_scale: float = 1.0,
        min_range_per_dim: float = 1e-1,
        device: torch.device | str = "cpu",
    ):
        """
        Args:
            action_min: Per-dim minimums from dataset statistics.
            action_max: Per-dim maximums from dataset statistics.
            equivariant_pairs: Groups of indices that must share the same
                               scale factor, e.g. [[0,1], [3,4]].
            action_scale: Expansion factor — final range is (1 + action_scale)
                          times the (possibly shared) data range.
            min_range_per_dim: Floor on half-range to prevent blow-up.
            device: Target device.
        """
        self.device = torch.device(device)
        self.action_scale = action_scale
        self.min_range_per_dim = min_range_per_dim
        self.equivariant_pairs = equivariant_pairs

        action_min = action_min.to(self.device)
        action_max = action_max.to(self.device)

        # ---- per-dim midpoint and half-range ----
        mid = (action_min + action_max) / 2
        half = (action_max - action_min) / 2

        # ---- tie equivariant groups ----
        equi_mid = mid.clone()
        equi_half = half.clone()

        for group in equivariant_pairs:
            # Shared midpoint  → group mean  (preserves no preferred direction)
            shared_mid = mid[group].mean()
            # Shared half-range → group max   (covers all dims, uniform scale)
            shared_half = half[group].max()
            for idx in group:
                equi_mid[idx] = shared_mid
                equi_half[idx] = shared_half

        # ---- floor on half-range ----
        min_half = torch.tensor(min_range_per_dim / 2, device=self.device)
        equi_half = torch.maximum(equi_half, min_half)

        # ---- expand range ----
        expanded_half = equi_half * (1 + action_scale)

        # ---- store limits ----
        self._limits = self.Limits(
            min=equi_mid - expanded_half,
            max=equi_mid + expanded_half,
        )
        self._range = self._limits.max - self._limits.min
        self._range = torch.maximum(self._range, torch.tensor(1e-8, device=self.device))

        # ---- diagnostics ----
        print("EquivariantActionScaler initialized:")
        print(f"  Original  mid:       {mid.cpu().numpy()}")
        print(f"  Equi      mid:       {equi_mid.cpu().numpy()}")
        print(f"  Original  half:      {half.cpu().numpy()}")
        print(f"  Equi      half:      {equi_half.cpu().numpy()}")
        print(f"  Expanded  half:      {expanded_half.cpu().numpy()}")
        print(f"  Final range:         {self._range.cpu().numpy()}")
        print(f"  Equivariant groups:  {equivariant_pairs}")
        print(f"  Action scale factor: {action_scale}")

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------
    @property
    def limits(self) -> Limits:
        return self._limits

    # ------------------------------------------------------------------
    # scale / unscale
    # ------------------------------------------------------------------
    def scale(self, action: torch.Tensor) -> torch.Tensor:
        """Map from original action space → [-1, 1]."""
        lo = self._limits.min.to(action.device)
        hi = self._limits.max.to(action.device)
        r = self._range.to(action.device)
        return 2.0 * (torch.clamp(action, lo, hi) - lo) / r - 1.0

    def unscale(self, scaled_action: torch.Tensor) -> torch.Tensor:
        """Map from [-1, 1] → original action space."""
        lo = self._limits.min.to(scaled_action.device)
        hi = self._limits.max.to(scaled_action.device)
        clamped = torch.clamp(scaled_action, -1.0, 1.0)
        return lo + (clamped + 1.0) * (hi - lo) / 2.0

    # ------------------------------------------------------------------
    # device movement
    # ------------------------------------------------------------------
    def to(self, device: torch.device | str) -> EquivariantActionScaler:
        new = EquivariantActionScaler.__new__(EquivariantActionScaler)
        new.device = torch.device(device)
        new.action_scale = self.action_scale
        new.min_range_per_dim = self.min_range_per_dim
        new.equivariant_pairs = self.equivariant_pairs
        new._limits = self.Limits(
            min=self._limits.min.to(device),
            max=self._limits.max.to(device),
        )
        new._range = self._range.to(device)
        return new

    # ------------------------------------------------------------------
    # convenience constructor
    # ------------------------------------------------------------------
    @classmethod
    def from_dataset_stats(
        cls,
        action_stats: dict,
        equivariant_pairs: list[list[int]] | None = None,
        action_scale: float = 1.0,
        min_range_per_dim: float = 1e-1,
        device: torch.device | str = "cpu",
    ) -> EquivariantActionScaler:
        """
        Create from dataset statistics dict with 'min' / 'max' keys.

        Args:
            action_stats: {'min': array-like, 'max': array-like}
            equivariant_pairs: Index groups sharing scale.
                Default for single-arm 7-dim action
                [a_xy(2), a_z(1), a_rxry(2), a_rz(1), a_grip(1)]:
                [[0,1], [3,4]]
            action_scale: Range expansion factor.
            min_range_per_dim: Half-range floor.
            device: Target device.
        """
        if equivariant_pairs is None:
            equivariant_pairs = [
                [0, 1],  # action_xy:   irrep(1)
                [3, 4],  # action_rxry: irrep(1)
            ]

        action_min = torch.tensor(action_stats["min"], dtype=torch.float32)
        action_max = torch.tensor(action_stats["max"], dtype=torch.float32)

        return cls(
            action_min=action_min,
            action_max=action_max,
            equivariant_pairs=equivariant_pairs,
            action_scale=action_scale,
            min_range_per_dim=min_range_per_dim,
            device=device,
        )


class EquivariantStateStandardizer:
    """
    State standardizer that respects equivariant structure and supports
    passthrough (identity) dimensions that should not be normalized.

    Three categories of dimensions:
      - equivariant groups: shared mean & std within each group
      - passthrough dims:   identity transform (no centering, no scaling)
      - everything else:    standard per-dim Gaussian normalization
    """

    def __init__(
        self,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        equivariant_pairs: list[list[int]],
        passthrough_dims: list[int] | None = None,
        min_std: float = 1e-1,
        device: torch.device | str = "cpu",
        robot_base_xy=None,
    ):
        """
        Args:
            state_mean: Per-dim means from dataset statistics.
            state_std: Per-dim stds from dataset statistics.
            equivariant_pairs: Groups of indices that must share the same
                               std and mean, e.g. [[0, 1]] for pos_xy.
            passthrough_dims: Indices that should NOT be standardized
                              (identity transform). e.g. [3, 4, 5, 6] for
                              quaternion dims that get converted to 6D rotation
                              downstream.
            min_std: Floor on std to prevent blow-up.
            device: Target device.
        """
        self.device = torch.device(device)
        self.min_std = min_std
        self.equivariant_pairs = equivariant_pairs
        self.passthrough_dims = set(passthrough_dims or [])
        self.robot_base_xy = robot_base_xy or torch.tensor([-0.5, -0.1])

        # --- validate no overlap between equivariant and passthrough ---
        equi_dims = {idx for group in equivariant_pairs for idx in group}
        overlap = equi_dims & self.passthrough_dims
        assert len(overlap) == 0, (
            f"Dims {overlap} appear in both equivariant_pairs and "
            f"passthrough_dims — a dim cannot be both."
        )

        state_mean = state_mean.to(self.device)
        state_std = state_std.to(self.device)

        # --- apply std floor ---
        safe_std = torch.maximum(
            state_std,
            torch.tensor(min_std, device=self.device),
        )

        # --- build effective mean and std ---
        eff_mean = state_mean.clone()
        eff_std = safe_std.clone()

        # Equivariant groups: tie mean (group avg) and std (group max)
        for group in equivariant_pairs:
                    shared_std = safe_std[group].max()
                    for idx in group:
                        eff_mean[idx] = 0.0      # base subtraction handles centering
                        eff_std[idx] = shared_std

        # Passthrough dims: identity transform → mean=0, std=1
        for idx in self.passthrough_dims:
            eff_mean[idx] = 0.0
            eff_std[idx] = 1.0

        self._mean = eff_mean
        self._std = eff_std

        # --- keep originals for diagnostics ---
        self._raw_mean = state_mean
        self._raw_std = state_std

        # --- diagnostics ---
        ndim = len(state_mean)
        trivial_dims = sorted(
            set(range(ndim)) - equi_dims - self.passthrough_dims
        )
        print("EquivariantStateStandardizer initialized:")
        print(f"  State dim:          {ndim}")
        print(f"  Equivariant groups: {equivariant_pairs}")
        print(f"  Passthrough dims:   {sorted(self.passthrough_dims)}")
        print(f"  Trivial dims:       {trivial_dims}")
        print(f"  Raw mean:           {state_mean.cpu().numpy()}")
        print(f"  Raw std:            {state_std.cpu().numpy()}")
        print(f"  Effective mean:     {eff_mean.cpu().numpy()}")
        print(f"  Effective std:      {eff_std.cpu().numpy()}")

    # ------------------------------------------------------------------
    # standardize / unstandardize
    # ------------------------------------------------------------------
    def standardize(self, state: torch.Tensor) -> torch.Tensor:
        state = state.clone()
        print(f"  [DEBUG standardize] input xy: {state[..., 0:2].cpu().numpy().flatten()[:2]}")
        state[..., 0:2] -= self.robot_base_xy.to(state.device)
        mean = self._mean.to(state.device)
        std = self._std.to(state.device)
        return (state - mean) / std

    def unstandardize(self, state: torch.Tensor) -> torch.Tensor:
        """Inverse of standardize."""
        mean = self._mean.to(state.device)
        std = self._std.to(state.device)
        # state * 1 + 0 = state for passthrough dims, as desired
        return state * std + mean

    # ------------------------------------------------------------------
    # device movement
    # ------------------------------------------------------------------
    def to(self, device: torch.device | str) -> EquivariantStateStandardizer:
        new = EquivariantStateStandardizer.__new__(EquivariantStateStandardizer)
        new.device = torch.device(device)
        new.min_std = self.min_std
        new.equivariant_pairs = self.equivariant_pairs
        new.passthrough_dims = self.passthrough_dims
        new._mean = self._mean.to(device)
        new._std = self._std.to(device)
        new._raw_mean = self._raw_mean.to(device)
        new._raw_std = self._raw_std.to(device)
        new.robot_base_xy = self.robot_base_xy.to(device)
        return new

    # ------------------------------------------------------------------
    # convenience constructor
    # ------------------------------------------------------------------
    @classmethod
    def from_dataset_stats(
        cls,
        state_stats: dict,
        equivariant_pairs: list[list[int]] | None = None,
        passthrough_dims: list[int] | None = None,
        min_std: float = 1e-1,
        device: torch.device | str = "cpu",
        robot_base_xy=None,
    ) -> EquivariantStateStandardizer:
        """
        Create from dataset statistics dict with 'mean' / 'std' keys.

        Defaults match single-arm Panda layout:
          [ee_pos(3), ee_quat(4), ee_q(2)]

        Args:
            state_stats: {'mean': array-like, 'std': array-like}
            equivariant_pairs: Default [[0, 1]] for pos_xy irrep(1).
            passthrough_dims: Default [3, 4, 5, 6] for quaternion.
            min_std: Std floor.
            device: Target device.
        """
        if equivariant_pairs is None:
            equivariant_pairs = [
                [0, 1],  # pos_xy: irrep(1)
            ]

        if passthrough_dims is None:
            passthrough_dims = [3, 4, 5, 6]  # ee_quat: converted to 6D downstream

        state_mean = torch.tensor(state_stats["mean"], dtype=torch.float32)
        state_std = torch.tensor(state_stats["std"], dtype=torch.float32)

        return cls(
            state_mean=state_mean,
            state_std=state_std,
            equivariant_pairs=equivariant_pairs,
            passthrough_dims=passthrough_dims,
            min_std=min_std,
            device=device,
            robot_base_xy=robot_base_xy,
        )

def detect_robot_base_xy(env_name: str, video_key: str, device: str = "cpu") -> torch.Tensor:
    """Probe a single env instance to find the robot base XY position from MuJoCo."""
    _probe_env = create_vectorized_env(
        env_name=env_name,
        num_envs=1,
        device=device,
        video_key=video_key,
        debug=True,
    )

    # Dig into wrappers to find the sim
    _e = _probe_env.envs[0] if hasattr(_probe_env, 'envs') else _probe_env
    for _ in range(10):
        if hasattr(_e, 'sim'):
            sim = _e.sim
            # Look for the robot base body
            for name in sim.model.body_names:
                if name.endswith('_base') and 'robot' in name:
                    bid = sim.model.body_name2id(name)
                    pos = sim.data.body_xpos[bid]
                    quat = sim.data.body_xquat[bid]
                    print(f"Detected robot base: {name}, pos={pos}, quat={quat}")

                    # Warn if base orientation isn't identity
                    if abs(quat[0] - 1.0) > 1e-3 or np.linalg.norm(quat[1:]) > 1e-3:
                        print(f"WARNING: robot base quat is not identity: {quat}")
                        print("  Equivariant pos_xy subtraction assumes identity base orientation.")

                    _probe_env.close()
                    return torch.tensor(pos[:2], dtype=torch.float32)

        if hasattr(_e, 'env'):
            _e = _e.env
        elif hasattr(_e, 'unwrapped'):
            _e = _e.unwrapped
        else:
            break

    _probe_env.close()
    raise RuntimeError(
        f"Could not find robot base body in {env_name}. "
        f"Available bodies: {list(sim.model.body_names) if 'sim' in dir(_e) else 'unknown'}"
    )