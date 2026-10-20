"""The ResFiT family's actor and critic (equivariance.{actor,critic}_family=resfit): ResFiT's own
classes from off_policy/rl/, wrapped to read this agent's features and to bound as this agent does.
ResFiT's encoder, a ViT per camera, is ResObsEnc(encoder="resfit") (networks/obs_encoder.py).

Each wrapper builds ResFiT's own class, so its layers and their initialization are ResFiT's. What
the wrappers change, and nothing else:
- the actor reads [image features | state | base action], ResFiT's order, as one tensor, and its
  final Tanh is replaced by this agent's bound and noise (bounded_residual): equi_tanh times the
  per-field bound, as every actor family has it;
- the critic reads [tokens | state] as one tensor, and unflattens the tokens for ResFiT's spatial
  trunk, which reads them as ResFiT's _encode gives them, [B, patches, 128].
The state is this agent's, 11 numbers per arm normalized per field, where ResFiT's agent has a
z-score of the raw 9; and the actions are in env units, where ResFiT's are in its scaler's units.
tests/test_resfit_heads.py checks both wrappers against ResFiT's own modules.
"""

from __future__ import annotations

import torch

from resfit.rl_finetuning.config.rlpd import ActorConfig, CriticConfig
from resfit.rl_finetuning.equi_off_policy.rl.equi_rl_utils import bounded_residual, feature_tensor
from resfit.rl_finetuning.off_policy.rl.actor import Actor
from resfit.rl_finetuning.off_policy.rl.critic import Critic


class ResFiTActor(torch.nn.Module):
    """ResFiT's residual actor: compress the image features to agent.actor.feature_dim (128, with
    LayerNorm), then agent.actor.num_layers hidden layers of hidden_dim (2 x 1024, with LayerNorm)
    over [compressed | state | base action]. `bound` is the per-field bound (field_bounds) and
    `residual_squash` equivariance.residual_squash, both as the equivariant actor reads them."""

    def __init__(self, repr_dim: int, prop_dim: int, action_dim: int, cfg: ActorConfig, *, action_layout,
                 bound: torch.Tensor, residual_squash: bool):
        super().__init__()
        if cfg.spatial_emb > 0:
            raise ValueError("agent.actor.spatial_emb: ResFiT's actor has its spatial embedding only "
                             "without a base action, and this actor is residual")
        # Its second argument, the width of one patch, is read by that spatial embedding alone.
        actor = Actor(repr_dim, repr_dim, prop_dim, action_dim, cfg, residual_actor=True)
        assert isinstance(actor.policy[-1], torch.nn.Tanh), "agent.actor.action_bound=box ends ResFiT's actor in a Tanh"
        self.compress = actor.compress
        self.policy = actor.policy[:-1]  # less its Tanh: the bound is this agent's
        self.repr_dim = repr_dim
        self.action_layout = action_layout
        self.residual_squash = residual_squash
        self.register_buffer("bound", bound, persistent=False)  # as the equivariant actor's

    def forward(self, feat, std=None) -> torch.Tensor:
        x = feature_tensor(feat)
        mu = self.policy(torch.cat([self.compress(x[:, :self.repr_dim]), x[:, self.repr_dim:]], dim=1))
        return bounded_residual(mu, self.action_layout, self.bound, self.residual_squash, std)


class ResFiTCritic(torch.nn.Module):
    """ResFiT's critic: a spatial-embedding trunk over the tokens, state and action, then
    agent.critic.num_q heads of agent.critic.num_layers hidden layers (10 x 2 x 1024 at the arms'
    settings, with LayerNorm). Q per head, [num_q, B, 1]."""

    def __init__(self, num_patch: int, patch_dim: int, prop_dim: int, action_dim: int, cfg: CriticConfig):
        super().__init__()
        self.num_patch, self.patch_dim = num_patch, patch_dim
        self.net = Critic(repr_dim=num_patch * patch_dim, patch_repr_dim=patch_dim, prop_dim=prop_dim,
                          action_dim=action_dim, cfg=cfg)

    def _tokens_and_state(self, feat) -> tuple[torch.Tensor, torch.Tensor]:
        x = feature_tensor(feat)
        n = self.num_patch * self.patch_dim
        return x[:, :n].unflatten(1, (self.num_patch, self.patch_dim)), x[:, n:]

    def forward(self, feat, act: torch.Tensor) -> torch.Tensor:
        return self.net(*self._tokens_and_state(feat), act)

    def q_value(self, feat, act: torch.Tensor) -> torch.Tensor:
        return self.net.q_value(*self._tokens_and_state(feat), act)

    def q_value_for_policy(self, feat, act: torch.Tensor) -> torch.Tensor:
        return self.net.q_value_for_policy(*self._tokens_and_state(feat), act)
