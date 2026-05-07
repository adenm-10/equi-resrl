# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import re

import numpy as np
import torch
from torch import nn
from torch.distributions.utils import _standard_normal
from torchvision import transforms

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