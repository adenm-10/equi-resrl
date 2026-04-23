# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import re

import numpy as np
import torch
from torch import distributions as pyd
from torch import nn
from torch.distributions.utils import _standard_normal
from torchvision import transforms

from typing import Dict, Callable, List

def get_rescale_transform(target_size):
    return transforms.Resize(
        target_size,
        interpolation=transforms.InterpolationMode.BICUBIC,
        antialias=True,
    )


def concat_obs(curr_idx, obses, obs_stack) -> torch.Tensor:
    """
    cat obs as [obses[curr_idx], obses[curr_idx-1], ... obs[curr_odx-obs_stack+1]]
    """
    vals = []
    for offset in range(obs_stack):
        if curr_idx - offset >= 0:
            val = obses[curr_idx - offset]
            if not isinstance(val, torch.Tensor):
                val = torch.from_numpy(val)
            vals.append(val)
        else:
            vals.append(torch.zeros_like(vals[-1]))
    return torch.cat(vals, dim=0)


class eval_mode:  # noqa: N801
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def to_torch(xs, device):
    return tuple(torch.as_tensor(x, device=device) for x in xs)


"""Equivariant initialization utilities.

Drop-in replacement for the non-equivariant `utils.orth_weight_init` /
`utils.apply_initialization_to_network` helpers, operating in the
equivariant basis-coefficient space instead of raw weight matrices.
"""

import torch
from escnn import nn as enn
from escnn.nn.init import deltaorthonormal_init


def apply_deltaortho_init(network, *, exclude_final_layer: bool = False):
    """Apply deltaorthonormal_init to every escnn.nn.Linear in *network*.

    Args:
        network: an nn.Module (or nn.Sequential) containing escnn.nn.Linear layers.
        exclude_final_layer: if True, skip the last escnn.nn.Linear found
                             (useful when the final layer gets its own scaling).
    """
    linears = [m for m in network.modules() if isinstance(m, enn.Linear)]

    if not linears:
        return

    targets = linears[:-1] if exclude_final_layer else linears

    for module in targets:
        deltaorthonormal_init(module.weights.data, module.basisexpansion)
        if module.bias is not None:
            module.bias.data.zero_()


def scale_final_equi_layer(network, scale: float):
    """Scale the weights (and bias) of the last escnn.nn.Linear in *network*."""
    for module in reversed(list(network.modules())):
        if isinstance(module, enn.Linear):
            with torch.no_grad():
                module.weights.mul_(scale)
                if module.bias is not None:
                    module.bias.mul_(scale)
            return

def clip_action_norm(action, max_norm):
    assert max_norm > 0
    assert action.dim() == 2 and action.size(1) == 7

    ee_action = action[:, :6]
    gripper_action = action[:, 6:]

    ee_action_norm = ee_action.norm(dim=1).unsqueeze(1)
    ee_action = ee_action / ee_action_norm
    assert (ee_action.norm(dim=1).min() - 1).abs() <= 1e-5
    scale = ee_action_norm.clamp(max=max_norm)
    ee_action = ee_action * scale
    action = torch.cat([ee_action, gripper_action], dim=1)
    return action  # noqa: RET504


class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6, max_action_norm: float = -1):
        if isinstance(scale, float):
            scale = torch.ones_like(loc) * scale

        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps
        self.max_action_norm = max_action_norm

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x  # noqa: RET504

    def sample(self, clip=None, sample_shape=None):
        if sample_shape is None:
            sample_shape = torch.Size()
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        x = self._clamp(x)
        if self.max_action_norm > 0:
            x = clip_action_norm(x, self.max_action_norm)
        return x


def schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r"linear\((.+),(.+),(.+)\)", schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r"step_linear\((.+),(.+),(.+),(.+),(.+)\)", schdl)
        if match:
            init, final1, duration1, final2, duration2 = [float(g) for g in match.groups()]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
            return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)

# ==========================
# For Normalizer
# ==========================

def dict_apply(
        x: Dict[str, torch.Tensor], 
        func: Callable[[torch.Tensor], torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result

class DictOfTensorMixin(nn.Module):
    def __init__(self, params_dict=None):
        super().__init__()
        if params_dict is None:
            params_dict = nn.ParameterDict()
        self.params_dict = params_dict

    @property
    def device(self):
        return next(iter(self.parameters())).device

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        def dfs_add(dest, keys, value: torch.Tensor):
            if len(keys) == 1:
                dest[keys[0]] = value
                return

            if keys[0] not in dest:
                dest[keys[0]] = nn.ParameterDict()
            dfs_add(dest[keys[0]], keys[1:], value)

        def load_dict(state_dict, prefix):
            out_dict = nn.ParameterDict()
            for key, value in state_dict.items():
                value: torch.Tensor
                if key.startswith(prefix):
                    param_keys = key[len(prefix):].split('.')[1:]
                    # if len(param_keys) == 0:
                    #     import pdb; pdb.set_trace()
                    dfs_add(out_dict, param_keys, value.clone())
            return out_dict

        self.params_dict = load_dict(state_dict, prefix + 'params_dict')
        self.params_dict.requires_grad_(False)
        return 
    
