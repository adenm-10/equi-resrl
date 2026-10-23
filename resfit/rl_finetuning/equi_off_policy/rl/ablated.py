"""The ablated family (equivariance.{encoder,actor,critic}_family=ablated): an equivariant network,
built exactly as the equivariant family builds it, then exported to plain torch layers.

At the same seed an ablated network starts as its equivariant twin's function: the encoder bit for
bit, the actor and critic to float rounding, where a LayerNorm's scale and shift are fused into it
(at arch-wide: 0, 1.5e-8 and 7e-7; outputs/diagnostics/arch-swap-probe-1020/). Only its updates
differ. escnn trains a few coefficients per layer and expands them into weights tied by the group;
here every expanded weight is a parameter and moves on its own. Its forward mirrors the equivariant
one's, the same slicing, concatenation, bound and noise, on plain tensors. tests/test_ablated.py
checks both: equal at the start, no longer equivariant after a few updates.

Exporting draws no random number. torch's Linear and Conv2d draw their initial weights when built,
before escnn's export overwrites them, so each export runs under fork_rng. Every network built after
an ablated one, and every update, then draws what it would after its equivariant twin.
"""

from __future__ import annotations

import copy

import torch
from escnn import nn as enn

from resfit.rl_finetuning.equi_off_policy.common_utils.escnn_fixes import GroupPooling
from resfit.rl_finetuning.equi_off_policy.networks.equi_encoder import EquiResBlock
from resfit.rl_finetuning.equi_off_policy.rl.actor import Actor as EquiActor
from resfit.rl_finetuning.equi_off_policy.rl.critic import Critic as EquiCritic
from resfit.rl_finetuning.equi_off_policy.rl.equi_rl_utils import (
    RegularLayerNorm,
    TrivialLayerNorm,
    bounded_residual,
    feature_tensor,
)


class BlockMax(torch.nn.Module):
    """GroupPooling on plain tensors: the max over each regular field's channels."""

    def __init__(self, n_fields: int, field_size: int):
        super().__init__()
        self.n_fields, self.field_size = n_fields, field_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.shape[0], self.n_fields, self.field_size).max(dim=2).values


class PlainResBlock(torch.nn.Module):
    """EquiResBlock on plain tensors: relu(norm(conv2(relu(norm(conv1(x))))) + skip(x))."""

    def __init__(self, block: EquiResBlock):
        super().__init__()
        self.conv1, self.bn1 = plain(block.conv1), plain(block.bn1)
        self.conv2, self.bn2 = plain(block.conv2), plain(block.bn2)
        self.skip = plain(block.skip)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + self.skip(x))


def plain(module: torch.nn.Module) -> torch.nn.Module:
    """`module` as plain torch layers computing the same function from its current weights.

    escnn's Linear and R2Conv become their dense layers (escnn's own export); RegularLayerNorm a
    LayerNorm with each field's scale and shift on each of its channels; TrivialLayerNorm its own
    LayerNorm; GroupPooling BlockMax; escnn's ReLU and max pool torch's; EquiResBlock PlainResBlock;
    a Sequential or an Identity stays one. Anything else raises, the batch norms and FieldNorm
    among them: none has a plain form that computes the same.
    """
    if isinstance(module, torch.nn.Sequential):
        return torch.nn.Sequential(*(plain(m) for m in module))
    if isinstance(module, torch.nn.Identity):
        return torch.nn.Identity()
    if isinstance(module, (enn.Linear, enn.R2Conv)):
        # escnn shares each sampled basis between all the layers that use it, so where an escnn
        # network is already on the GPU in this process (a test session), a new layer's basis is
        # there too: the layer is expanded there. In a fresh process, as the trainer's, it is the CPU.
        module.to(next(b.device for name, b in module.named_buffers() if name.endswith("sampled_basis")))
        # torch's layers are built on the CPU, so its RNG is the one their initialization draws from.
        with torch.random.fork_rng(devices=[]):
            return module.export()
    if isinstance(module, RegularLayerNorm):
        norm = torch.nn.LayerNorm(module.in_type.size, eps=module.eps)
        with torch.no_grad():
            norm.weight.copy_(module.weight.repeat_interleave(module.field_size, dim=0).flatten())
            norm.bias.copy_(module.bias.repeat_interleave(module.field_size, dim=0).flatten())
        return norm
    if isinstance(module, TrivialLayerNorm):
        return copy.deepcopy(module.norm)
    if isinstance(module, GroupPooling):
        reps = module.in_type.representations
        assert all(r.name == "regular" for r in reps), f"BlockMax pools regular fields, got {module.in_type}"
        return BlockMax(len(reps), reps[0].size)
    if isinstance(module, enn.ReLU):
        return torch.nn.ReLU()
    if isinstance(module, enn.PointwiseMaxPool2D):
        return torch.nn.MaxPool2d(module.kernel_size, module.stride, module.padding, module.dilation,
                                  ceil_mode=module.ceil_mode)
    if isinstance(module, EquiResBlock):
        return PlainResBlock(module)
    raise TypeError(f"{type(module).__name__} has no plain form, so the ablated family cannot build it; "
                    f"leave the norm that adds it at its default")


class AblatedActor(torch.nn.Module):
    """The equivariant actor `actor`, its layers exported: visual projection, trunk and policy, each
    reading proprioception and the base action beside its input, then the same bound and noise.
    Residual only: the agent refuses this family in vanilla RL."""

    def __init__(self, actor: EquiActor):
        super().__init__()
        assert actor.residual_actor, "the ablated actor is residual-only"
        self.action_layout = actor.action_layout
        self.residual_squash = actor.residual_squash
        self.vis_ih_end = actor.vis_ih_type.size
        self.prop_end = self.vis_ih_end + actor.prop_type.size
        self.register_buffer("bound", actor.bound.clone(), persistent=False)  # as the equivariant actor's
        self.visual_proj = plain(actor.visual_proj)
        self.input_proj = plain(actor.input_proj)
        self.policy = plain(actor.policy)

    def forward(self, feat, std=None) -> torch.Tensor:
        x = feature_tensor(feat)
        skip = [x[:, self.vis_ih_end:self.prop_end], x[:, self.prop_end:]]  # proprioception, base action
        z = self.input_proj(torch.cat([self.visual_proj(x[:, :self.vis_ih_end]), *skip], dim=1))
        mu = self.policy(torch.cat([z, *skip], dim=1))
        return bounded_residual(mu, self.action_layout, self.bound, self.residual_squash, std)


class AblatedCritic(torch.nn.Module):
    """The equivariant critic `critic`, its layers exported: visual projection (and its window, if it
    has one), trunk, then each head, each reading proprioception and the action beside its input; Q
    per head, [num_q, B, 1]. Its q_value and q_value_for_policy are the equivariant critic's, which
    read only forward."""

    q_value = EquiCritic.q_value
    q_value_for_policy = EquiCritic.q_value_for_policy

    def __init__(self, critic: EquiCritic):
        super().__init__()
        self.min_q_heads = critic.min_q_heads
        self.policy_gradient_type = critic.policy_gradient_type
        self.vis_ih_end = critic.vis_ih_type.size
        self.window_end = self.vis_ih_end + (0 if critic.window_type is None else critic.window_type.size)
        self.visual_proj = plain(critic.visual_proj)
        self.window_proj = None if critic.window_proj is None else plain(critic.window_proj)
        self.input_proj = plain(critic.input_proj)
        self.heads = torch.nn.ModuleList([plain(head.net) for head in critic.q_ensemble.heads])

    def _visual(self, x: torch.Tensor):
        """As the equivariant critic's _visual, on plain tensors."""
        if self.window_proj is None:
            return self.visual_proj(x[:, :self.vis_ih_end]), None
        window = self.window_proj(x[:, self.vis_ih_end:self.window_end])
        images = self.visual_proj[0](x[:, :self.vis_ih_end])
        return self.visual_proj[1:](images + window), (window, images)

    def window_share(self, feat) -> torch.Tensor:
        """As the equivariant critic's."""
        _, (window, images) = self._visual(feature_tensor(feat))
        return (window.norm(dim=1) / images.norm(dim=1).clamp_min(1e-12)).mean()

    def forward(self, feat, act: torch.Tensor) -> torch.Tensor:
        x = feature_tensor(feat)
        prop = x[:, self.window_end:]
        v, _ = self._visual(x)
        z = self.input_proj(torch.cat([v, prop, act], dim=1))
        head_in = torch.cat([z, prop, act], dim=1)
        return torch.stack([head(head_in) for head in self.heads], dim=0)
