"""
Build an equivariant-aware LinearNormalizer from LeRobot dataset stats.

Uses the same normalize_util functions as equi_diffpo so the normalizer
matches what RobomimicReplayImageSymDataset.get_normalizer() produces.
"""

import numpy as np
import torch
import torch.nn as nn

import zarr
import unittest
from typing import Union, Dict

from resfit.rl_finetuning.equi_off_policy.rl.equi_rl_utils import dict_apply, DictOfTensorMixin

#################################
# Utils
#################################

class LinearNormalizer(DictOfTensorMixin):
    avaliable_modes = ['limits', 'gaussian']
    
    @torch.no_grad()
    def fit(self,
        data: Union[Dict, torch.Tensor, np.ndarray, zarr.Array],
        last_n_dims=1,
        dtype=torch.float32,
        mode='limits',
        output_max=1.,
        output_min=-1.,
        range_eps=1e-4,
        fit_offset=True):
        if isinstance(data, dict):
            for key, value in data.items():
                self.params_dict[key] =  _fit(value, 
                    last_n_dims=last_n_dims,
                    dtype=dtype,
                    mode=mode,
                    output_max=output_max,
                    output_min=output_min,
                    range_eps=range_eps,
                    fit_offset=fit_offset)
        else:
            self.params_dict['_default'] = _fit(data, 
                    last_n_dims=last_n_dims,
                    dtype=dtype,
                    mode=mode,
                    output_max=output_max,
                    output_min=output_min,
                    range_eps=range_eps,
                    fit_offset=fit_offset)
    
    def __call__(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self.normalize(x)
    
    def __getitem__(self, key: str):
        return SingleFieldLinearNormalizer(self.params_dict[key])

    def __setitem__(self, key: str , value: 'SingleFieldLinearNormalizer'):
        self.params_dict[key] = value.params_dict

    def _normalize_impl(self, x, forward=True):
        if isinstance(x, dict):
            result = dict()
            for key, value in x.items():
                params = self.params_dict[key]
                result[key] = _normalize(value, params, forward=forward)
            return result
        else:
            if '_default' not in self.params_dict:
                raise RuntimeError("Not initialized")
            params = self.params_dict['_default']
            return _normalize(x, params, forward=forward)

    def normalize(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self._normalize_impl(x, forward=True)

    def unnormalize(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self._normalize_impl(x, forward=False)

    def get_input_stats(self) -> Dict:
        if len(self.params_dict) == 0:
            raise RuntimeError("Not initialized")
        if len(self.params_dict) == 1 and '_default' in self.params_dict:
            return self.params_dict['_default']['input_stats']
        
        result = dict()
        for key, value in self.params_dict.items():
            if key != '_default':
                result[key] = value['input_stats']
        return result


    def get_output_stats(self, key='_default'):
        input_stats = self.get_input_stats()
        if 'min' in input_stats:
            # no dict
            return dict_apply(input_stats, self.normalize)
        
        result = dict()
        for key, group in input_stats.items():
            this_dict = dict()
            for name, value in group.items():
                this_dict[name] = self.normalize({key:value})[key]
            result[key] = this_dict
        return result


class SingleFieldLinearNormalizer(DictOfTensorMixin):
    avaliable_modes = ['limits', 'gaussian']
    
    @torch.no_grad()
    def fit(self,
            data: Union[torch.Tensor, np.ndarray, zarr.Array],
            last_n_dims=1,
            dtype=torch.float32,
            mode='limits',
            output_max=1.,
            output_min=-1.,
            range_eps=1e-4,
            fit_offset=True):
        self.params_dict = _fit(data, 
            last_n_dims=last_n_dims,
            dtype=dtype,
            mode=mode,
            output_max=output_max,
            output_min=output_min,
            range_eps=range_eps,
            fit_offset=fit_offset)
    
    @classmethod
    def create_fit(cls, data: Union[torch.Tensor, np.ndarray, zarr.Array], **kwargs):
        obj = cls()
        obj.fit(data, **kwargs)
        return obj
    
    @classmethod
    def create_manual(cls, 
            scale: Union[torch.Tensor, np.ndarray], 
            offset: Union[torch.Tensor, np.ndarray],
            input_stats_dict: Dict[str, Union[torch.Tensor, np.ndarray]]):
        def to_tensor(x):
            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(x)
            x = x.flatten()
            return x
        
        # check
        for x in [offset] + list(input_stats_dict.values()):
            assert x.shape == scale.shape
            assert x.dtype == scale.dtype
        
        params_dict = nn.ParameterDict({
            'scale': to_tensor(scale),
            'offset': to_tensor(offset),
            'input_stats': nn.ParameterDict(
                dict_apply(input_stats_dict, to_tensor))
        })
        return cls(params_dict)

    @classmethod
    def create_identity(cls, dtype=torch.float32):
        scale = torch.tensor([1], dtype=dtype)
        offset = torch.tensor([0], dtype=dtype)
        input_stats_dict = {
            'min': torch.tensor([-1], dtype=dtype),
            'max': torch.tensor([1], dtype=dtype),
            'mean': torch.tensor([0], dtype=dtype),
            'std': torch.tensor([1], dtype=dtype)
        }
        return cls.create_manual(scale, offset, input_stats_dict)

    def normalize(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return _normalize(x, self.params_dict, forward=True)

    def unnormalize(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return _normalize(x, self.params_dict, forward=False)

    def get_input_stats(self):
        return self.params_dict['input_stats']

    def get_output_stats(self):
        return dict_apply(self.params_dict['input_stats'], self.normalize)

    def __call__(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self.normalize(x)



def _fit(data: Union[torch.Tensor, np.ndarray, zarr.Array],
        last_n_dims=1,
        dtype=torch.float32,
        mode='limits',
        output_max=1.,
        output_min=-1.,
        range_eps=1e-4,
        fit_offset=True):
    assert mode in ['limits', 'gaussian']
    assert last_n_dims >= 0
    assert output_max > output_min

    # convert data to torch and type
    if isinstance(data, zarr.Array):
        data = data[:]
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    if dtype is not None:
        data = data.type(dtype)

    # convert shape
    dim = 1
    if last_n_dims > 0:
        dim = np.prod(data.shape[-last_n_dims:])
    data = data.reshape(-1,dim)

    # compute input stats min max mean std
    input_min, _ = data.min(axis=0)
    input_max, _ = data.max(axis=0)
    input_mean = data.mean(axis=0)
    input_std = data.std(axis=0)

    # compute scale and offset
    if mode == 'limits':
        if fit_offset:
            # unit scale
            input_range = input_max - input_min
            ignore_dim = input_range < range_eps
            input_range[ignore_dim] = output_max - output_min
            scale = (output_max - output_min) / input_range
            offset = output_min - scale * input_min
            offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]
            # ignore dims scaled to mean of output max and min
        else:
            # use this when data is pre-zero-centered.
            assert output_max > 0
            assert output_min < 0
            # unit abs
            output_abs = min(abs(output_min), abs(output_max))
            input_abs = torch.maximum(torch.abs(input_min), torch.abs(input_max))
            ignore_dim = input_abs < range_eps
            input_abs[ignore_dim] = output_abs
            # don't scale constant channels 
            scale = output_abs / input_abs
            offset = torch.zeros_like(input_mean)
    elif mode == 'gaussian':
        ignore_dim = input_std < range_eps
        scale = input_std.clone()
        scale[ignore_dim] = 1
        scale = 1 / scale

        if fit_offset:
            offset = - input_mean * scale
        else:
            offset = torch.zeros_like(input_mean)
    
    # save
    this_params = nn.ParameterDict({
        'scale': scale,
        'offset': offset,
        'input_stats': nn.ParameterDict({
            'min': input_min,
            'max': input_max,
            'mean': input_mean,
            'std': input_std
        })
    })
    for p in this_params.parameters():
        p.requires_grad_(False)
    return this_params


def _normalize(x, params, forward=True):
    assert 'scale' in params
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    scale = params['scale']
    offset = params['offset']
    x = x.to(device=scale.device, dtype=scale.dtype)
    src_shape = x.shape
    x = x.reshape(-1, scale.shape[0])
    if forward:
        x = x * scale + offset
    else:
        x = (x - offset) / scale
    x = x.reshape(src_shape)
    return x

def get_range_normalizer_from_stat(stat, output_max=1, output_min=-1, range_eps=1e-7):
    # -1, 1 normalization
    input_max = stat['max']
    input_min = stat['min']
    input_range = input_max - input_min
    ignore_dim = input_range < range_eps
    input_range[ignore_dim] = output_max - output_min
    scale = (output_max - output_min) / input_range
    offset = output_min - scale * input_min
    offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]

    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )

def get_range_symmetric_normalizer_from_stat(stat, output_max=1, output_min=-1, range_eps=1e-7):
    # -1, 1 normalization
    input_max = stat['max']
    input_min = stat['min']
    abs_max = np.max([np.abs(stat['max'][:2]), np.abs(stat['min'][:2])])
    input_max[:2] = abs_max
    input_min[:2] = -abs_max
    input_range = input_max - input_min
    ignore_dim = input_range < range_eps
    input_range[ignore_dim] = output_max - output_min
    scale = (output_max - output_min) / input_range
    offset = output_min - scale * input_min
    offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]

    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )

def get_image_range_normalizer():
    scale = np.array([2], dtype=np.float32)
    offset = np.array([-1], dtype=np.float32)
    stat = {
        'min': np.array([0], dtype=np.float32),
        'max': np.array([1], dtype=np.float32),
        'mean': np.array([0.5], dtype=np.float32),
        'std': np.array([np.sqrt(1/12)], dtype=np.float32)
    }
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )

def get_identity_normalizer_from_stat(stat):
    scale = np.ones_like(stat['min'])
    offset = np.zeros_like(stat['min'])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
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