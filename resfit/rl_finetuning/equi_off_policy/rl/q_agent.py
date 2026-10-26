# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

import copy
from contextlib import contextmanager

import torch

from escnn import gspaces, nn
from escnn.group import CyclicGroup

from resfit.rl_finetuning.config.rlpd import QAgentConfig
from resfit.rl_finetuning.config.residual_td3 import EquivarianceConfig
from resfit.rl_finetuning.off_policy import common_utils
from resfit.rl_finetuning.off_policy.common_utils import utils

from resfit.rl_finetuning.equi_off_policy.networks.obs_encoder import ENCODERS, ResObsEnc
from resfit.rl_finetuning.equi_off_policy.rl.ablated import AblatedActor, AblatedCritic
from resfit.rl_finetuning.equi_off_policy.rl.actor import Actor as EquiActor, ResFiTShapedActor
from resfit.rl_finetuning.equi_off_policy.rl.critic import Critic as EquiCritic
from resfit.rl_finetuning.equi_off_policy.rl.equi_rl_utils import equi_clip, field_bounds
from resfit.rl_finetuning.equi_off_policy.rl.resfit_heads import ResFiTActor, ResFiTCritic


# The EquivarianceConfig fields that only the equivariant family's network of each kind reads, and so
# the ablated one, which is built by it. ResFiT's networks read agent.vit, agent.actor and
# agent.critic instead, so behind them these do nothing. The critic's window is read by both the
# encoder, which computes it, and the critic, which reads it.
WINDOW_FIELDS = ("critic_window", "critic_window_ih")
ENCODER_FIELDS = ("encoder_norm", "encoder_orth_init", "initialize", "enc_degree_channel", "ih_out_channel",
                  "float_images_as_uint8", "ih_channels", "enc_out_channel", *WINDOW_FIELDS)
ACTOR_FIELDS = ("actor_norm", "actor_degree_channel", "use_orth_init", "actor_design", "actor_feature_fields")
ACTOR_DESIGNS = ("ours", "resfit")
CRITIC_FIELDS = ("critic_norm", "critic_pooled_norm", "num_critic_layers", "critic_degree_channel",
                 "critic_orth_init", "critic_last_layer_init_scale", *WINDOW_FIELDS)


def check_families(equi_cfg: EquivarianceConfig, agent_cfg: QAgentConfig, residual_actor: bool) -> None:
    """Refuse, before anything is built, a mix of network families this agent cannot build, and any
    field that no network of the mix reads: a run's recorded config must describe the networks that
    trained it (STANDARDS 1.2). equivariance.residual_squash and residual_field_bound are read by
    every actor family, and use_visual_proj by the equivariant and ablated actor and critic."""
    encoder, actor, critic = equi_cfg.encoder_family, equi_cfg.actor_family, equi_cfg.critic_family
    for part, family in (("encoder", encoder), ("actor", actor), ("critic", critic)):
        if family not in ENCODERS:
            raise ValueError(f"equivariance.{part}_family={family!r}: expected one of {ENCODERS}")
    if encoder != "equivariant" and "equivariant" in (actor, critic):
        raise ValueError("an equivariant actor or critic needs the equivariant encoder: behind any other, "
                         "its image features would have to be declared invariant")
    if critic == "resfit" and encoder != "resfit":
        raise ValueError("ResFiT's critic reads the ViTs' spatial tokens: it needs encoder_family=resfit")
    if not residual_actor and (encoder, actor, critic) != ("equivariant",) * 3:
        raise ValueError("the ablated and ResFiT families are residual-only: vanilla RL keeps the "
                         "equivariant networks")
    if equi_cfg.actor_design not in ACTOR_DESIGNS:
        raise ValueError(f"equivariance.actor_design={equi_cfg.actor_design!r}: expected one of {ACTOR_DESIGNS}")
    if equi_cfg.actor_design == "resfit" and actor == "ablated":
        raise ValueError("equivariance.actor_design=resfit: the ablated actor exports Actor's layers, "
                         "\"ours\"; it has no export of ResFiTShapedActor")
    if equi_cfg.actor_design == "resfit" and not residual_actor:
        raise ValueError("equivariance.actor_design=resfit is residual-only, as ResFiT's actor is here")
    if critic == "resfit" and agent_cfg.critic.loss.type != "mse":
        raise ValueError(f"agent.critic.loss.type={agent_cfg.critic.loss.type!r}: this agent's critic loss "
                         f"is a mean squared error on Q, so ResFiT's critic must output Q itself")
    unread = []
    if encoder == "resfit":
        unread += ENCODER_FIELDS
    if actor == "resfit":
        unread += ACTOR_FIELDS
    if critic == "resfit":
        unread += CRITIC_FIELDS
    if actor != "resfit" and equi_cfg.actor_design == "ours":
        unread.append("actor_feature_fields")  # ResFiTShapedActor's alone
    unread = list(dict.fromkeys(unread))  # the window's fields are in two lists
    # use_visual_proj is read by Actor and by the equivariant and ablated critic, not by
    # ResFiTShapedActor or ResFiT's networks.
    if (actor == "resfit" or equi_cfg.actor_design == "resfit") and critic == "resfit":
        unread.append("use_visual_proj")
    if critic != "equivariant":
        unread.append("frozen_target_critic")  # a plain critic computes the same in either mode
    defaults = EquivarianceConfig()
    changed = [f for f in unread if getattr(equi_cfg, f) != getattr(defaults, f)]
    if changed:
        raise ValueError(f"no network of this agent ({encoder} encoder, {actor} actor, {critic} critic) reads "
                         f"equivariance.{changed}; leave them at their defaults")


def build_encoder(obs_shape, rl_cameras: list[str], agent_cfg: QAgentConfig, equi_cfg: EquivarianceConfig,
                  residual_actor: bool) -> ResObsEnc:
    """The agent's observation encoder, as QAgent builds it, on the CPU. tools/encoder_probe.py builds
    a run's encoder alone through here."""
    return ResObsEnc(                              # was: EquivariantResObsEnc
        obs_shape=obs_shape,
        N=equi_cfg.N,
        n_hidden=equi_cfg.enc_degree_channel,
        encoder=equi_cfg.encoder_family,
        vit_cfg=agent_cfg.vit if equi_cfg.encoder_family == "resfit" else None,
        group=gspaces.no_base_space(CyclicGroup(equi_cfg.N)),
        initialize=equi_cfg.initialize,
        encoder_norm=equi_cfg.encoder_norm,
        encoder_orth_init=equi_cfg.encoder_orth_init,
        float_images_as_uint8=equi_cfg.float_images_as_uint8,
        # Per-arm observation layout, the same in every family. Vanilla RL
        # (residual_actor=False) has no base action to encode.
        n_arms=equi_cfg.n_arms,
        gripper_dim=equi_cfg.gripper_dim,
        hand_dof=equi_cfg.hand_dof,
        agentview_key=rl_cameras[0],
        wrist_keys=tuple(rl_cameras[1:]),
        ih_out_channel=equi_cfg.ih_out_channel,
        base_action_input=residual_actor,
        ih_channels=tuple(equi_cfg.ih_channels),
        enc_out_channel=equi_cfg.enc_out_channel,
        critic_window=equi_cfg.critic_window,
        critic_window_ih=equi_cfg.critic_window_ih,
    )


class QAgent(torch.nn.Module):
    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        prop_shape: tuple[int],
        action_dim: int,
        rl_cameras: list[str] | str,
        agent_cfg: QAgentConfig,
        equi_cfg: EquivarianceConfig,
        residual_actor: bool = False,
    ):
        """Initialize the Q-agent.

        Parameters
        ----------
        obs_shape : tuple[int, int, int]
            Shape (C, H, W) for a single camera image.
        prop_shape : tuple[int]
            Shape of the proprioceptive observation vector.
        action_dim : int
            Number of action dimensions.
        rl_cameras : list[str] | str
            Camera name(s) used by the RL policy.
        agent_cfg : QAgentConfig
            Hyper-parameter configuration.
        equi_cfg : EquivarianceConfig
            The networks' configuration: each network's family (encoder_family, actor_family,
            critic_family), and the equivariant and ablated networks' sizes and options.
        residual_actor : bool
            Whether the actor outputs a residual on top of a base action.

        Every family is trained and bounded alike: the trainer, the normalizer, the action's bound
        and noise, the optimizers and the critic's targets are the same whatever the networks. The
        networks are built encoder, then critic, then actor, so at one seed two mixes that differ
        in one network start with the same weights in the others.
        """
        super().__init__()
        if isinstance(rl_cameras, str):
            rl_cameras = [rl_cameras]
        assert len(rl_cameras) > 0, "At least one camera must be provided"
        assert len(rl_cameras) == 1 + equi_cfg.n_arms, (
            f"expected agentview plus one wrist camera per arm: "
            f"n_arms={equi_cfg.n_arms}, rl_cameras={rl_cameras}"
        )

        self.rl_cameras = rl_cameras
        self.agent_cfg = agent_cfg
        self.residual_actor = residual_actor
        self.equi_cfg = equi_cfg
        if agent_cfg.actor.action_bound != "box":
            raise ValueError("agent.actor.action_bound is ResFiT's agent's: this agent bounds its actions by "
                             "its own equi_tanh and equi_clip, in every family")
        if agent_cfg.actor.noise_units != "scaled":
            raise ValueError("agent.actor.noise_units is ResFiT's agent's: this agent's noise is in env units "
                             "already, as its actions are")
        check_families(equi_cfg, agent_cfg, residual_actor)

        use_orth_init = self.equi_cfg.use_orth_init

        critic_hidden_dim = self.equi_cfg.critic_degree_channel
        critic_num_layers = self.equi_cfg.num_critic_layers
        critic_dropout = self.agent_cfg.critic.dropout
        loss = self.agent_cfg.critic.loss
        num_q = self.agent_cfg.critic.num_q
        min_q_heads = self.agent_cfg.critic.min_q_heads
        policy_gradient_type = self.agent_cfg.critic.policy_gradient_type

        actor_hidden_dim = self.equi_cfg.actor_degree_channel
        actor_num_layers = self.equi_cfg.num_actor_layers
        action_scale = self.agent_cfg.actor.action_scale
        actor_dropout = self.agent_cfg.actor.dropout
        last_layer_scale = self.agent_cfg.actor.actor_last_layer_init_scale

        self.enc = build_encoder(obs_shape, rl_cameras, agent_cfg, equi_cfg, residual_actor).to(self.agent_cfg.device)
        # The group is built in every family: the action layout, which bounds every actor, is its.
        self.group = self.enc.group

        # The critic, then the actor: the equivariant network, or it exported (ablated), or ResFiT's.
        if self.equi_cfg.critic_family == "resfit":
            self.critic = ResFiTCritic(self.enc.num_patch, self.enc.patch_dim, self.enc.prop_dim,
                                       self.enc.action_dim, self.agent_cfg.critic)
        else:
            critic = EquiCritic(
                group=self.group,
                vis_ih_type=self.enc.vis_ih_type,
                prop_type=self.enc.prop_type,
                action_type=self.enc.action_type,
                hidden_dim=critic_hidden_dim,
                num_layers=critic_num_layers,
                loss_cfg=loss,
                num_q=num_q,
                norm=self.equi_cfg.critic_norm,
                pooled_norm=self.equi_cfg.critic_pooled_norm,
                orth_init=self.equi_cfg.critic_orth_init,
                last_layer_init_scale=self.equi_cfg.critic_last_layer_init_scale,
                use_visual_proj=self.equi_cfg.use_visual_proj,
                dropout=critic_dropout,
                min_q_heads=min_q_heads,
                policy_gradient_type=policy_gradient_type,
                window_type=self.enc.window_type,
            )
            self.critic = critic if self.equi_cfg.critic_family == "equivariant" else AblatedCritic(critic)

        if self.equi_cfg.actor_family == "resfit":  # check_families: then actor_design is "ours"
            bound = field_bounds(self.enc.action_type.size, action_scale, self.equi_cfg.residual_field_bound,
                                 self.enc.action_layout, self.enc.action_fields, residual_actor)
            self.actor = ResFiTActor(self.enc.vis_ih_type.size, self.enc.prop_dim, self.enc.action_dim,
                                     self.agent_cfg.actor, action_layout=self.enc.action_layout, bound=bound,
                                     residual_squash=self.equi_cfg.residual_squash)
        elif self.equi_cfg.actor_design == "resfit":  # the equivariant family (check_families)
            self.actor = ResFiTShapedActor(
                group=self.group,
                vis_ih_type=self.enc.vis_ih_type,
                prop_type=self.enc.prop_type,
                action_type=self.enc.action_type,
                action_layout=self.enc.action_layout,
                feature_fields=self.equi_cfg.actor_feature_fields,
                hidden_dim=actor_hidden_dim,
                action_shape=self.enc.action_shape,
                norm=self.equi_cfg.actor_norm,
                use_orth_init=use_orth_init,
                last_layer_scale=last_layer_scale,
                action_scale=action_scale,
                residual_squash=self.equi_cfg.residual_squash,
                field_bound=self.equi_cfg.residual_field_bound,
                action_fields=self.enc.action_fields,
            )
        else:
            actor = EquiActor(
                group=self.group,
                vis_ih_type=self.enc.vis_ih_type,
                prop_type=self.enc.prop_type,
                action_type=self.enc.action_type,
                action_layout=self.enc.action_layout,
                hidden_dim=actor_hidden_dim,
                action_shape=self.enc.action_shape,
                num_layers=actor_num_layers,
                dropout=actor_dropout,
                norm=self.equi_cfg.actor_norm,
                use_orth_init=use_orth_init,
                action_scale=action_scale,
                last_layer_scale=last_layer_scale,
                use_visual_proj=self.equi_cfg.use_visual_proj,
                residual_squash=self.equi_cfg.residual_squash,
                residual_actor=residual_actor,
                field_bound=self.equi_cfg.residual_field_bound,
                action_fields=self.enc.action_fields,
            )
            self.actor = actor if self.equi_cfg.actor_family == "equivariant" else AblatedActor(actor)

        # prop_shape and action_dim come from the env; the encoder derives its own
        # from equi_cfg. A mismatch would mis-slice every batch without erroring.
        assert self.enc.state_dim == prop_shape[0], (
            f"encoder expects a {self.enc.state_dim}-dim observation.state but the env "
            f"gives {prop_shape[0]}; check equivariance.n_arms / gripper_dim"
        )
        assert self.enc.action_dim == action_dim, (
            f"encoder action is {self.enc.action_dim}-dim but the env gives "
            f"{action_dim}; check equivariance.n_arms / hand_dof"
        )

        self.to(self.agent_cfg.device)

        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)

        self._enc_params = list(self.enc.parameters())
        self._critic_params = list(self.critic.parameters())

        print(common_utils.wrap_ruler("encoder weights"))
        print(self.enc)
        common_utils.count_parameters(self.enc)

        print(common_utils.wrap_ruler("critic weights"))
        print(self.critic)
        common_utils.count_parameters(self.critic)

        print(common_utils.wrap_ruler("actor weights"))
        print(self.actor)
        common_utils.count_parameters(self.actor)

        # optimizers
        # Freeze encoder parameters if requested
        if getattr(self.agent_cfg, "freeze_encoder", False):
            for param in self.enc.parameters():
                param.requires_grad = False
            print("🧊 Encoder parameters frozen - no gradient updates will be performed")

        # Create optimizers (PyTorch will ignore frozen parameters)
        self.encoder_opt = torch.optim.AdamW(self.enc.parameters(), lr=self.agent_cfg.critic_lr)
        self.critic_opt = torch.optim.AdamW(self.critic.parameters(), lr=self.agent_cfg.critic_lr)
        self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=self.agent_cfg.actor_lr)

        # LR schedulers for warmup (if warmup is enabled)
        self.encoder_scheduler = None
        self.critic_scheduler = None
        self.actor_scheduler = None

        if self.agent_cfg.lr_warmup_steps > 0:
            # LinearLR scheduler that linearly ramps from start_factor to 1.0 over total_iters steps
            # Note: start_factor must be > 0 for LinearLR scheduler
            warmup_start = self.agent_cfg.lr_warmup_start

            # Calculate start factors for each optimizer
            critic_start_factor = warmup_start / self.agent_cfg.critic_lr if self.agent_cfg.critic_lr > 0 else 1e-8
            critic_start_factor = max(critic_start_factor, 1e-8)

            actor_start_factor = warmup_start / self.agent_cfg.actor_lr if self.agent_cfg.actor_lr > 0 else 1e-8
            actor_start_factor = max(actor_start_factor, 1e-8)

            # Create schedulers with appropriate start factors
            self.encoder_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.encoder_opt, start_factor=critic_start_factor, total_iters=self.agent_cfg.lr_warmup_steps
            )
            self.critic_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.critic_opt, start_factor=critic_start_factor, total_iters=self.agent_cfg.lr_warmup_steps
            )
            self.actor_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.actor_opt, start_factor=actor_start_factor, total_iters=self.agent_cfg.lr_warmup_steps
            )

        # data augmentation
        self.aug = common_utils.RandomShiftsAug(pad=4)

        self.bc_policies: list[torch.nn.Module] = []
        # to log rl vs bc during evaluation
        self.stats: common_utils.MultiCounter | None = None

        # The equivariant target critic runs in train mode, as the actor target
        # does. In eval mode an escnn layer computes with the weights it expanded
        # at .train(False), so the target would never see a soft update;
        # equivariance.frozen_target_critic reproduces that. In train mode a
        # batch-norm critic normalizes each target batch with its own statistics.
        # The ablated and ResFiT critics compute the same in either mode, and their
        # targets stay in eval mode, as ResFiT's agent keeps its own.
        self._critic_target_training = (self.equi_cfg.critic_family == "equivariant"
                                        and not self.equi_cfg.frozen_target_critic)
        self.critic_target.train(self._critic_target_training)
        self.train(True)

    def combine_actions(self, base_action: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """The action the env receives: the base action plus the residual, each irrep(1) pair
        clipped by its length (equi_clip). The actor loss, the TD target, critic_diagnostics and the
        evaluator's Q all combine through here, in every family."""
        return equi_clip(base_action + residual, self.actor.action_layout, bound=1.0)

    def _detach_feat(self, feat):
        """The equivariant encoder's features are a GeometricTensor, the others' a tensor."""
        if isinstance(feat, nn.GeometricTensor):
            return nn.GeometricTensor(feat.tensor.detach(), feat.type)
        return feat.detach()

    def add_bc_policy(self, bc_policy):
        bc_policy.train(False)
        self.bc_policies.append(bc_policy)

    def set_stats(self, stats):
        self.stats = stats

    def train(self, training=True):
        self.training = training
        self.enc.train(training)
        self.actor.train(training)
        self.critic.train(training)

        assert self.critic_target.training == self._critic_target_training
        for bc_policy in self.bc_policies:
            assert not bc_policy.training

    def _min_over_random_two(self, q_values: torch.Tensor) -> torch.Tensor:
        """Compute min over a random subset of 2 heads from q_values [K, B, 1]."""
        assert q_values.dim() == 3 and q_values.size(-1) == 1
        num_heads = q_values.size(0)
        if num_heads <= 2:
            return torch.min(q_values[0], q_values[1])
        # Sample two unique head indices uniformly at random
        idx = torch.randperm(num_heads, device=q_values.device)[:2]
        subset = q_values.index_select(dim=0, index=idx)  # [2, B, 1]
        return subset.min(dim=0).values

    @contextmanager
    def override_act_method(self, override_method: str):
        original_method = self.agent_cfg.act_method
        assert original_method != override_method

        self.agent_cfg.act_method = override_method
        yield

        self.agent_cfg.act_method = original_method

    def _maybe_unsqueeze_(self, obs):
        should_unsqueeze = False
        if obs[self.rl_cameras[0]].dim() == 3:
            should_unsqueeze = True

        if should_unsqueeze:
            for k, v in obs.items():
                obs[k] = v.unsqueeze(0)
        return should_unsqueeze

    def act(self, obs: dict[str, torch.Tensor], *, eval_mode=False, stddev=0.0, cpu=True) -> torch.Tensor:
        """This function takes tensor and returns actions in tensor"""
        assert not self.training
        assert not self.actor.training
        assert not self.enc.training

        # Make a shallow copy of the observation dict
        obs = copy.copy(obs)
        unsqueezed = self._maybe_unsqueeze_(obs)

        assert "actor_feat" not in obs
        obs["actor_feat"], obs["critic_feat"] = self.enc(obs)

        action = self._act_default(
            obs=obs,
            eval_mode=eval_mode,
            stddev=stddev,
            use_target=False,
        )

        if unsqueezed:
            action = action.squeeze(0)

        action = action.detach()
        if cpu:
            action = action.cpu()
        return action

    def _act_default(self, *, obs, eval_mode, stddev, use_target):
        """Every actor family's forward takes (features, std) and bounds as the equivariant actor
        does (bounded_residual). The noise is clipped with the action, by equi_clip, so
        agent.stddev_clip is not read."""
        if eval_mode and not use_target:
            assert not self.training
        actor = self.actor_target if use_target else self.actor
        if eval_mode:
            return actor.forward(obs["actor_feat"], std=0)
        return actor.forward(obs["actor_feat"], std=stddev)

    def update_critic(
        self,
        obs: dict[str, torch.Tensor],
        action: torch.Tensor,
        reward: torch.Tensor,
        discount: torch.Tensor,
        next_obs: dict[str, torch.Tensor],
        stddev: float,
        importance_weights: torch.Tensor | None = None,
        *,
        compute_metrics: bool = True,
    ):
        """One critic and encoder step. `compute_metrics` is explained in `update`."""
        with torch.no_grad():
            # use train mode as we use actor dropout
            assert self.actor_target.training

            # Predict next residual action and form the combined next action
            # Use target_action_noise config to control whether to add noise to target actions
            next_residual_action = self._act_default(
                obs=next_obs,
                eval_mode=not self.agent_cfg.target_action_noise,  # Disable noise if target_action_noise=False
                stddev=stddev,
                use_target=True,
            )

            if self.residual_actor:
                next_action = self.combine_actions(
                    next_obs["observation.base_action"], next_residual_action
                )
            else:
                next_action = next_residual_action

            # Compute target Q using min over a random subset of 2 heads
            target_all = self.critic_target.q_value(next_obs["critic_feat"], next_action)
            target_q_min = target_all.squeeze(-1)  # [B]
            target_q = (reward + (discount * target_q_min)).detach()

        if self.agent_cfg.clip_q_target_to_reward_range:
            target_q = torch.clamp(target_q, min=0, max=1)  # Sparse rewards are in {0, 1}

        q_all = self.critic(obs["critic_feat"], action).squeeze(-1)  # [K,B]
        # Compute TD errors for prioritized experience replay (before taking mean)
        td_errors = torch.abs(q_all - target_q.unsqueeze(0)).mean(dim=0)  # [B] - mean across heads

        # Apply importance sampling weights if provided. The replay buffers always provide them;
        # with uniform sampling they are all 1.
        if importance_weights is not None:
            # Weight the squared TD errors by importance sampling weights
            weighted_td_errors = td_errors**2 * importance_weights
            critic_loss = weighted_td_errors.mean()
        else:
            # Mean squared error across heads and batch (uniform sampling)
            critic_loss = (td_errors**2).mean()

        # Prioritized replay reads the TD errors after every update, so they are always returned.
        # They stay on the GPU: copying them to the CPU would make the CPU wait for the update.
        metrics = {"_td_errors": td_errors.detach()}

        if compute_metrics:
            metrics["train/critic_qt"] = target_q.mean().item()
            metrics["train/critic_loss"] = critic_loss.item()
            # Store target_q for the Q-value histogram
            metrics["_target_q"] = target_q.detach().cpu()
            # Log importance sampling weights for monitoring PER behavior
            if importance_weights is not None:
                metrics["train/importance_weights_mean"] = importance_weights.mean().item()
                metrics["train/importance_weights_std"] = importance_weights.std().item()
                metrics["train/importance_weights_min"] = importance_weights.min().item()
                metrics["train/importance_weights_max"] = importance_weights.max().item()

        # Zero gradients
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)

        critic_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self._enc_params, self.agent_cfg.critic_grad_clip_norm)
        torch.nn.utils.clip_grad_norm_(self._critic_params, self.agent_cfg.critic_grad_clip_norm)

        if compute_metrics:
            # Per-submodule encoder gradient norms, logging only. With max_norm=inf,
            # clip_grad_norm_ measures the norm and scales the gradients by exactly 1.0, so
            # skipping it on updates that are not logged changes nothing.
            for child_name, child_module in self.enc.named_children():
                params_with_grad = [p for p in child_module.parameters() if p.grad is not None]
                if params_with_grad:
                    sub_grad_norm = torch.nn.utils.clip_grad_norm_(params_with_grad, float("inf"))
                    metrics[f"debug/enc_{child_name}_grad_norm"] = sub_grad_norm.item()

        self.encoder_opt.step()
        self.critic_opt.step()

        return metrics

    def _compute_actor_loss(self, obs: dict[str, torch.Tensor], stddev: float):
        assert "critic_feat" in obs, "safety check"

        action_pred: torch.Tensor = self._act_default(
            obs=obs,
            eval_mode=False,
            stddev=0.0,
            use_target=False,
        )

        # Add L2 regularization on action magnitude (before we add the residual action to the base)
        action_l2_penalty = self.agent_cfg.actor.action_l2_reg_weight * torch.mean(torch.sum(action_pred**2, dim=-1))

        if self.residual_actor:
            combined_action = self.combine_actions(
                obs["observation.base_action"], action_pred
            )
        else:
            combined_action = action_pred

        combined_action.retain_grad()
        q = self.critic.q_value_for_policy(obs["critic_feat"], combined_action)
        actor_loss_base = -q.mean()

        actor_loss_total = actor_loss_base + action_l2_penalty
        return actor_loss_total, actor_loss_base, combined_action, action_pred, action_l2_penalty

    def _compute_actor_bc_loss(self, batch, *, backprop_encoder):
        assert not self.residual_actor, "Not implemented"
        obs: dict[str, torch.Tensor] = batch["obs"]

        assert "feat" not in obs
        obs["actor_feat"], _ = self.enc(obs)

        if not backprop_encoder:
            obs["actor_feat"] = self._detach_feat(obs["actor_feat"])

        pred_action = self._act_default(
            obs=obs,
            eval_mode=False,
            stddev=0,
            use_target=False,
        )
        action: torch.Tensor = batch["action"]
        loss = torch.nn.functional.mse_loss(pred_action, action, reduction="none")
        loss = loss.sum(1).mean(0)
        return loss  # noqa: RET504

    def update_actor(self, obs: dict[str, torch.Tensor], stddev: float, *, compute_metrics: bool = True):
        """One actor step. `compute_metrics` is explained in `update`."""
        metrics = {}

        (
            actor_loss_total,
            actor_loss_base,
            combined_action,
            action_pred,
            action_l2_penalty,
        ) = self._compute_actor_loss(obs, stddev)

        if compute_metrics:
            metrics["train/actor_loss_base"] = actor_loss_base.item()
            metrics["train/actor_loss_total"] = actor_loss_total.item()
            # Store residual actions for logging (the actual residual component we want to monitor)
            metrics["_actions"] = action_pred.detach().cpu()
            # Also store combined actions if needed for other purposes
            metrics["_combined_actions"] = combined_action.detach().cpu()

            # Log L2 regularization penalty if applied
            if self.agent_cfg.actor.action_l2_reg_weight > 0:
                metrics["train/actor_l2_penalty"] = action_l2_penalty.item()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss_total.backward()

        # ---- Diagnostic logging ----

        if compute_metrics:
            # 1) ∇_a Q: does the critic provide a gradient signal w.r.t. actions?
            #    If this is near zero, the critic landscape is flat — either the
            #    critic isn't seeing the residual or ensemble pessimism is canceling gradients.
            if combined_action.grad is not None:
                metrics["debug/dQ_da_norm"] = combined_action.grad.norm().item()
                metrics["debug/dQ_da_mean_abs"] = combined_action.grad.abs().mean().item()
                # The per-component gradient, for the trainer's per-field logging (field_metrics.py).
                metrics["_dQ_da"] = combined_action.grad.detach().cpu()
            else:
                metrics["debug/dQ_da_norm"] = 0.0

            # 2) Residual action magnitude: is the actor outputting anything?
            with torch.no_grad():
                metrics["debug/residual_norm"] = action_pred.norm(dim=-1).mean().item()
                metrics["debug/residual_mean_abs"] = action_pred.abs().mean().item()
                metrics["debug/residual_max_abs"] = action_pred.abs().max().item()

                metrics["debug/combined_act_norm"] = combined_action.norm(dim=-1).mean().item()
                metrics["debug/combined_act_mean_abs"] = combined_action.abs().mean().item()
                metrics["debug/combined_act_max_abs"] = combined_action.abs().max().item()

        # 3) Q sensitivity: does the critic value differ between combined and base-only?
        #    If these are nearly equal, the critic is blind to the residual.
        if self.residual_actor:
            with torch.no_grad():
                # Run on every actor update, logged or not. With critic_norm="iid_bn" this
                # forward updates the critic's running statistics, and with
                # policy_gradient_type="min_random_pair" it draws random numbers, so skipping it
                # would change the run. It costs one critic forward.
                q_base_only = self.critic.q_value_for_policy(obs["critic_feat"], obs["observation.base_action"])
                if compute_metrics:
                    metrics["debug/q_combined_mean"] = (-actor_loss_base).item()  # same as q.mean()
                    metrics["debug/q_base_only_mean"] = q_base_only.mean().item()
                    metrics["debug/q_advantage"] = metrics["debug/q_combined_mean"] - metrics["debug/q_base_only_mean"]
                    metrics["debug/q_advantage_ratio"] = metrics["debug/q_advantage"] / metrics["debug/q_combined_mean"]

        # Gradient clipping
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.agent_cfg.actor_grad_clip_norm)
        if compute_metrics:
            metrics["train/actor_grad_norm"] = actor_grad_norm.item()

        self.actor_opt.step()

        return metrics

    def update_actor_rft(
        self,
        obs: dict[str, torch.Tensor],
        stddev: float,
        bc_batch,
        ref_agent: QAgent,
    ):
        metrics = {}

        # Compute actor loss and get the actions used (single actor call)
        (
            actor_loss_total,
            actor_loss_base,
            combined_action,
            action_pred,
            action_l2_penalty,
        ) = self._compute_actor_loss(obs, stddev)

        metrics["train/actor_loss_base"] = actor_loss_base.item()
        metrics["train/actor_loss_total"] = actor_loss_total.item()
        # Store residual actions for logging (the actual residual component we want to monitor)
        metrics["_actions"] = action_pred.detach().cpu()
        # Also store combined actions if needed for other purposes
        metrics["_combined_actions"] = combined_action.detach().cpu()

        # Log L2 regularization penalty if applied
        if self.agent_cfg.actor.action_l2_reg_weight > 0:
            metrics["train/actor_l2_penalty"] = action_l2_penalty.item()

        # Use config option to control whether BC loss updates encoder
        bc_backprop_encoder = self.agent_cfg.bc_backprop_encoder
        bc_loss = self._compute_actor_bc_loss(bc_batch, backprop_encoder=bc_backprop_encoder)
        assert actor_loss_total.size() == bc_loss.size()

        ratio = 1
        if self.agent_cfg.bc_loss_dynamic:
            with torch.no_grad(), utils.eval_mode(self, ref_agent):
                assert ref_agent.cfg.act_method == "rl"

                # temporarily change to rl since we want to regularize actor not hybrid
                act_method = self.agent_cfg.act_method
                self.agent_cfg.act_method = "rl"

                ref_bc_obs = bc_batch.obs.copy()  # shallow copy
                ref_action = ref_agent.act(ref_bc_obs, eval_mode=True, cpu=False)

                # we first get the ref_action and then pop the feature
                # then we get the curr_action so that the obs["feat"] is the current feature
                # which can be used for computing q-values

                # TODO: Do I need to encode this first?
                bc_obs = bc_batch.obs
                curr_action = self.act(bc_obs, eval_mode=True, cpu=False)

                # curr_q = self.critic.q_value_for_policy(bc_obs["feat"], bc_obs["observation.state"], curr_action)
                # ref_q = self.critic.q_value_for_policy(bc_obs["feat"], bc_obs["observation.state"], ref_action)

                curr_q = self.critic.q_value_for_policy(bc_obs["critic_feat"], curr_action)
                ref_q  = self.critic.q_value_for_policy(bc_obs["critic_feat"], ref_action)

                ratio = (ref_q > curr_q).float().mean().item()

                # recover to original act_method
                self.agent_cfg.act_method = act_method

        loss = actor_loss_total + (self.agent_cfg.bc_loss_coef * ratio * bc_loss).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        # Conditionally update encoder along with actor if BC loss should backprop
        if bc_backprop_encoder:
            self.encoder_opt.zero_grad(set_to_none=True)

        loss.backward()

        # Gradient clipping
        metrics["train/actor_grad_norm"] = torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.agent_cfg.actor_grad_clip_norm
        ).item()

        if bc_backprop_encoder:
            metrics["train/encoder_grad_norm"] = torch.nn.utils.clip_grad_norm_(
                self.enc.parameters(), self.agent_cfg.actor_grad_clip_norm
            ).item()

        if bc_backprop_encoder:
            self.encoder_opt.step()
        self.actor_opt.step()

        metrics["rft/bc_loss"] = bc_loss.mean().item()
        metrics["rft/ratio"] = ratio
        return metrics

    def update(
        self,
        batch,
        stddev,
        update_actor,
        bc_batch=None,
        ref_agent: QAgent | None = None,
        *,
        compute_metrics: bool = True,
    ):
        """One TD3 update: critic and encoder, then the actor if `update_actor`.

        `compute_metrics=False` skips every value that is only logged. Each one is a `.item()`
        or `.cpu()`, which makes the CPU wait until the GPU has finished all the work queued so
        far, and the trainer reads the metrics of only the last update before each logging
        step: one update in 400 at the defaults. The update itself is the same either way, bit
        for bit (tests/test_regression_agent.py). The one returned value that is not a metric,
        "_td_errors", is always there, for prioritized replay.
        """
        obs: dict[str, torch.Tensor] = batch["obs"]
        action: torch.Tensor = batch["action"]
        reward: torch.Tensor = batch[("next", "reward")]
        discount: torch.Tensor = batch["gamma"]
        next_nonterminal: torch.Tensor = batch["nonterminal"]
        next_obs: dict[str, torch.Tensor] = batch[("next", "obs")]

        # To not bootstrap on terminal states we zero out the discount factor for terminal next states
        effective_discount = discount * next_nonterminal

        assert "feat" not in obs
        obs["actor_feat"], obs["critic_feat"] = self.enc(obs)

        with torch.no_grad():
            next_obs["actor_feat"], next_obs["critic_feat"] = self.enc(next_obs)

        metrics = {}
        if compute_metrics:
            metrics["data/batch_R"] = reward.mean().item()

        # Extract importance sampling weights if available (for prioritized experience replay)
        importance_weights = batch.get("_weight", None)

        critic_metric = self.update_critic(
            obs=obs,
            action=action,
            reward=reward,
            discount=effective_discount,
            next_obs=next_obs,
            stddev=stddev,
            importance_weights=importance_weights,
            compute_metrics=compute_metrics,
        )

        utils.soft_update_params(self.critic, self.critic_target, self.agent_cfg.critic_target_tau)
        metrics.update(critic_metric)

        if not update_actor:
            return metrics

        # NOTE: actor loss does not backprop into the encoder
        obs["actor_feat"] = self._detach_feat(obs["actor_feat"])
        obs["critic_feat"] = self._detach_feat(obs["critic_feat"])

        if bc_batch is None:
            actor_metric = self.update_actor(obs, stddev, compute_metrics=compute_metrics)
        else:
            assert ref_agent is not None
            actor_metric = self.update_actor_rft(obs, stddev, bc_batch, ref_agent)

        utils.soft_update_params(self.actor, self.actor_target, self.agent_cfg.critic_target_tau)
        metrics.update(actor_metric)

        return metrics

    @torch.no_grad()
    def critic_diagnostics(self, batch, num_online: int) -> dict[str, float]:
        """What the critic thinks of the data's actions and of the policy's, on each buffer.

        Mirrors the ResFiT agent's method of the same name (off_policy/rl/q_agent.py), and logs
        the same keys. Where they differ, this agent's own update decides:
        - a residual policy's action is combined with the base action by `combine_actions`,
          the disk clip its actor loss uses, where ResFiT's clamps each axis;
        - actions are in this stack's units, which `IdentityScaler` leaves as the env's, so
          `policy_data_gap` is not on the ResFiT stack's normalized scale;
        - the policy's action is computed without the equivariant actor's `std=0` path, which
          draws `torch.randn` and scales it by 0: the same action, but a moved RNG would change
          every update after it (tests/test_regression_agent.py).

        The batch is the trainer's: online rows [:num_online], then demo rows. Q is the mean over
        heads; the policy's action is the actor's mean, clipped as its actor loss clips it. Logging
        only: the caller puts the agent in eval mode, where the crop is centred and nothing draws a
        random number, so logging does not change what a run computes. The features are computed
        afresh, as `update` writes its own into the batch's observations.
        """
        assert not self.training
        keys = [*self.rl_cameras, "observation.state"]
        if self.residual_actor:
            keys.append("observation.base_action")
        obs = {k: batch["obs"][k] for k in keys}
        obs["actor_feat"], obs["critic_feat"] = self.enc(obs)

        data_action = batch["action"]
        policy_action = equi_clip(self.actor(obs["actor_feat"], std=None), self.actor.action_layout, bound=1.0)
        if self.residual_actor:
            policy_action = self.combine_actions(obs["observation.base_action"], policy_action)

        def q(action):
            return self.critic(obs["critic_feat"], action).squeeze(-1).mean(dim=0)

        metrics = {}
        if getattr(self.critic, "window_proj", None) is not None:
            metrics["debug/critic_window_share"] = self.critic.window_share(obs["critic_feat"]).item()

        q_data, q_policy = q(data_action), q(policy_action)
        gap = (policy_action - data_action).abs().mean(dim=-1)
        halves = {"online": slice(0, num_online), "demo": slice(num_online, None)}
        for name, rows in halves.items():
            if q_data[rows].numel() == 0:
                continue
            metrics[f"debug/q_data_{name}"] = q_data[rows].mean().item()
            metrics[f"debug/q_policy_{name}"] = q_policy[rows].mean().item()
            metrics[f"debug/policy_data_gap_{name}"] = gap[rows].mean().item()
        return metrics

    def step_lr_schedulers(self):
        """Step the learning rate schedulers for warmup."""
        if self.encoder_scheduler is not None:
            self.encoder_scheduler.step()
        if self.critic_scheduler is not None:
            self.critic_scheduler.step()
        if self.actor_scheduler is not None:
            self.actor_scheduler.step()
