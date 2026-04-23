"""
Build an equivariant-aware LinearNormalizer from LeRobot dataset stats.

Uses the same normalize_util functions as equi_diffpo so the normalizer
matches what RobomimicReplayImageSymDataset.get_normalizer() produces.
"""

import numpy as np
import torch

from resfit.rl_finetuning.equi_off_policy.common_utils.equi_normalizer_util import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
    get_range_symmetric_normalizer_from_stat,
    get_range_normalizer_from_stat,
    get_identity_normalizer_from_stat,
)


def _slice_stat(stat: dict, dims) -> dict:
    """Slice a stat dict (min/max/mean/std) to specific dimensions, return numpy."""
    out = {}
    for k in ("min", "max", "mean", "std"):
        v = stat[k]
        if isinstance(v, torch.Tensor):
            v = v.cpu().numpy()
        v = np.array(v, dtype=np.float32).copy()
        out[k] = v[dims]
    return out


def _shift_stat(stat: dict, shift: np.ndarray) -> dict:
    """Subtract a constant from min/max/mean (for robot_base_xy centering)."""
    return {
        "min": stat["min"] - shift,
        "max": stat["max"] - shift,
        "mean": stat["mean"] - shift,
        "std": stat["std"].copy(),  # std is shift-invariant
    }


def _dummy_stat(ndim: int) -> dict:
    """Dummy stats for identity normalizer fields."""
    return {
        "min": -np.ones(ndim, dtype=np.float32),
        "max":  np.ones(ndim, dtype=np.float32),
        "mean": np.zeros(ndim, dtype=np.float32),
        "std":  np.ones(ndim, dtype=np.float32),
    }


def build_equivariant_normalizer(
    stats: dict,
    min_range: float = 0.1,  # matches min_action_range default
) -> LinearNormalizer:
    """
    Build a LinearNormalizer with per-field equivariant-safe normalization.

    State layout (9-dim):
        [0:2]  pos_xy      — irrep(1), symmetric normalization
        [2:3]  pos_z       — trivial, range normalization
        [3:7]  ee_quat     — converted to 6D rot in encoder, identity here
        [7:9]  ee_q        — trivial, range normalization

    Action layout (7-dim):
        [0:2]  action_xy   — irrep(1), symmetric normalization
        [2:3]  action_z    — trivial, range normalization
        [3:5]  action_rx_ry — identity
        [5:6]  action_rz   — identity
        [6:7]  action_gripper — trivial, range normalization

    for full dataset description:
     - https://huggingface.co/datasets/ankile/robomimic-mh-can-image
    """
    # base_xy_np = robot_base_xy.cpu().numpy().astype(np.float32)

    state_stat = {k: (v.cpu().numpy().astype(np.float32) if isinstance(v, torch.Tensor)
                      else np.array(v, dtype=np.float32))
                  for k, v in stats["observation.state"].items()}
    action_stat = {k: (v.cpu().numpy().astype(np.float32) if isinstance(v, torch.Tensor)
                       else np.array(v, dtype=np.float32))
                   for k, v in stats["action"].items()}
    
    print(f"state_stat = {state_stat}")
    print(f"\n\nstats[observation.state] = {stats['observation.state']}")

    normalizer = LinearNormalizer()

    # ==== Proprioception ====

    # pos_xy: symmetric (equivariant irrep(1)), centered by robot_base_xy
    # pos_xy_stat = _shift_stat(_slice_stat(state_stat, slice(0, 2)), base_xy_np)
    # normalizer["pos_xy"] = get_range_symmetric_normalizer_from_stat(pos_xy_stat)

    pos_xy_stat = _slice_stat(state_stat, slice(0, 2))
    normalizer["pos_xy"] = get_range_symmetric_normalizer_from_stat(pos_xy_stat)

    # pos_z: range → [-1, 1]
    normalizer["pos_z"] = get_range_normalizer_from_stat(_slice_stat(state_stat, slice(2, 3)))

    # ee_rot: identity (quat → 6D rotation happens in encoder, no data-dependent scaling)
    normalizer["ee_rot"] = get_identity_normalizer_from_stat(_dummy_stat(6))

    # ee_q: range → [-1, 1]
    normalizer["ee_q"] = get_range_normalizer_from_stat(_slice_stat(state_stat, slice(7, 9)))

    # ==== Action ====

    # if absolute_actions:
    #     action_xy_stat = {
    #         "min": state_stat["min"][:2] + action_stat["min"][:2] - base_xy_np,
    #         "max": state_stat["max"][:2] + action_stat["max"][:2] - base_xy_np,
    #         "mean": state_stat["mean"][:2] + action_stat["mean"][:2] - base_xy_np,
    #         "std": action_stat["std"][:2].copy(),
    #     }
    # else:
    #     action_xy_stat = _slice_stat(action_stat, slice(0, 2))
    # normalizer["action_xy"] = get_range_symmetric_normalizer_from_stat(action_xy_stat)

    action_xy_stat = _slice_stat(action_stat, slice(0, 2))
    normalizer["action_xy"] = get_range_symmetric_normalizer_from_stat(action_xy_stat)

    # action_z: range
    normalizer["action_z"] = get_range_normalizer_from_stat(_slice_stat(action_stat, slice(2, 3)))

    # action_rx_ry: identity
    normalizer["action_rx_ry"] = get_identity_normalizer_from_stat(_dummy_stat(2))

    # action_rz: identity
    normalizer["action_rz"] = get_identity_normalizer_from_stat(_dummy_stat(1))

    # action_gripper: range
    normalizer["action_gripper"] = get_range_normalizer_from_stat(_slice_stat(action_stat, slice(6, 7)))

    # NOTE: Images are NOT included here.
    # get_image_range_normalizer() expects [0,1] input (scale=2, offset=-1).
    # Your replay buffer stores uint8 [0,255], so handle images inline in
    # the encoder: obs.float() / 255.0 * 2.0 - 1.0

    return normalizer

def detect_robot_base_xy(env_name: str, video_key: str, device: str = "cpu") -> torch.Tensor:
    """Probe a single env instance to find the robot base XY position from MuJoCo."""
    from resfit.dexmg.environments.dexmg import create_vectorized_env
    _probe_env = create_vectorized_env(
        env_name=env_name,
        num_envs=1,
        device=device,
        video_key=video_key,
        debug=True,
    )

    _e = _probe_env.envs[0] if hasattr(_probe_env, 'envs') else _probe_env
    for _ in range(10):
        if hasattr(_e, 'sim'):
            sim = _e.sim
            for name in sim.model.body_names:
                if name.endswith('_base') and 'robot' in name:
                    bid = sim.model.body_name2id(name)
                    pos = sim.data.body_xpos[bid]
                    quat = sim.data.body_xquat[bid]
                    print(f"Detected robot base: {name}, pos={pos}, quat={quat}")

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

class IdentityScaler():
    @staticmethod
    def scale(action):
        return action
    
    @staticmethod
    def unscale(action):
        return action
    
class IdentityStandardizer():
    @staticmethod
    def standardize(data):
        return data
    
    @staticmethod
    def unstandardize(data):
        return data