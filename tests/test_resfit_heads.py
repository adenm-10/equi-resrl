"""Tier 2: the ResFiT family (equivariance.*_family=resfit) is ResFiT's own networks, in this agent
(rl/resfit_heads.py, and the ResFiT encoder of networks/obs_encoder.py), and the agent refuses every
mix of families and every field it would not read (q_agent.py, check_families).

- The encoder's tokens are ResFiT's own _encode's on the same ViT weights, bit for bit, from the
  buffer's uint8 images and the env's float ones: unshifted in eval mode, and shifted as ResFiT
  shifts them at the same seed in train mode. At one seed the agent's ViTs are ResFiT's own agent's.
- The critic gives off_policy's Critic's outputs on the same weights and inputs.
- The actor has off_policy's Actor's parameters at the same seed, bit for bit, less its Tanh, and
  its mean is this agent's bound applied to the same pre-activation, where ResFiT's is its Tanh.
- Each refusal raises, and the field it names at its default does not; vanilla RL's is in
  test_layouts.py.
"""

from __future__ import annotations

import contextlib
import os

import pytest
import torch

PATCHES, PATCH_DIM, PROP_DIM, ACTION_DIM = 2 * 81, 128, 11, 7  # two cameras' ViT tokens; Square's layout


def _mix(encoder: str, actor: str, critic: str) -> list[str]:
    """eaqorqki's launch, the init0 anchor, with ResFiT's 10 critic heads and these families."""
    from tests.test_sweep_arms import EQUI_ANCHOR, _launch_order

    return _launch_order((EQUI_ANCHOR - {"agent.critic.num_q=5"}) | {
        "agent.critic.num_q=10", f"equivariance.encoder_family={encoder}",
        f"equivariance.actor_family={actor}", f"equivariance.critic_family={critic}"})


def _seeded(launch, device, seed: int = 2):
    from tests.test_sweep_arms import _compose, _seeded_agent

    with contextlib.redirect_stdout(open(os.devnull, "w")):  # the agents print their modules
        return _seeded_agent(_compose(launch), device, seed=seed)


def _resfit_agent(device, seed: int = 2):
    """ResFiT's own agent, as oj5k0jqb's launch builds it."""
    from tests.test_sweep_arms import RESFIT_ANCHOR, _launch_order

    return _seeded(_launch_order(RESFIT_ANCHOR), device, seed)


def _images(batch, cameras, fmt: str):
    obs = dict(batch["obs"])
    if fmt == "float":  # as the env gives them (dexmg.py)
        for cam in cameras:
            obs[cam] = obs[cam].float() / 255.0
    return obs


@pytest.mark.gpu
@pytest.mark.parametrize("fmt", ["uint8", "float"])
def test_the_resfit_encoder_is_resfits_own(fmt, device):
    """The same weights at the same seed, since both agents build their ViTs first, one per camera in
    camera order; and the same tokens from them, in eval mode, where ResFiT's act() encodes with
    augment=False."""
    from tools.agent_fingerprint import fixed_batch

    ours = _seeded(_mix("resfit", "resfit", "resfit"), device)
    theirs = _resfit_agent(device)
    mine, its = ours.enc.vits.state_dict(), theirs.encoders.state_dict()
    assert mine.keys() == its.keys() and all(torch.equal(mine[k], its[k]) for k in mine)

    ours.train(False)
    theirs.train(False)
    obs = _images(fixed_batch(base_action=True).to(device), ours.rl_cameras, fmt)
    with torch.no_grad():
        tokens = ours.enc._vit_tokens(obs)
        assert tokens.shape == (obs[ours.rl_cameras[0]].shape[0], PATCHES * PATCH_DIM)
        assert torch.equal(tokens, theirs._encode(dict(obs), augment=False).flatten(1))
        actor_feat, critic_feat = ours.enc(obs)
    assert torch.equal(critic_feat[:, :PATCHES * PATCH_DIM], tokens)
    assert critic_feat.shape[1] == PATCHES * PATCH_DIM + PROP_DIM
    assert actor_feat.shape[1] == PATCHES * PATCH_DIM + PROP_DIM + ACTION_DIM


@pytest.mark.gpu
def test_the_resfit_encoder_shifts_its_images_in_train_mode_alone(device):
    """ResFiT's update encodes with augment=True: the same random shifts, at the same seed, in train
    mode; another seed shifts otherwise; eval mode shifts nothing."""
    from tools.agent_fingerprint import fixed_batch

    ours = _seeded(_mix("resfit", "resfit", "resfit"), device)
    theirs = _resfit_agent(device)
    obs = dict(fixed_batch(base_action=True).to(device)["obs"])
    with torch.no_grad():
        ours.train(True)
        torch.manual_seed(5)
        shifted = ours.enc._vit_tokens(obs)
        torch.manual_seed(5)
        assert torch.equal(shifted, theirs._encode(dict(obs), augment=True).flatten(1))
        torch.manual_seed(6)
        assert not torch.equal(shifted, ours.enc._vit_tokens(obs))
        ours.train(False)
        torch.manual_seed(7)
        still = ours.enc._vit_tokens(obs)
        torch.manual_seed(8)
        assert torch.equal(still, ours.enc._vit_tokens(obs))


def test_the_resfit_critic_is_resfits_own():
    """The wrapper holds off_policy's Critic, built at the same seed, and passes it the tokens as
    ResFiT's _encode gives them, [B, patches, 128], beside the state and the action."""
    from resfit.rl_finetuning.config.rlpd import CriticConfig
    from resfit.rl_finetuning.equi_off_policy.rl.resfit_heads import ResFiTCritic
    from resfit.rl_finetuning.off_policy.rl.critic import Critic

    cfg = CriticConfig(num_q=10)
    torch.manual_seed(0)
    ours = ResFiTCritic(PATCHES, PATCH_DIM, PROP_DIM, ACTION_DIM, cfg)
    torch.manual_seed(0)
    theirs = Critic(repr_dim=PATCHES * PATCH_DIM, patch_repr_dim=PATCH_DIM, prop_dim=PROP_DIM,
                    action_dim=ACTION_DIM, cfg=cfg)
    mine, its = ours.net.state_dict(), theirs.state_dict()
    assert mine.keys() == its.keys() and all(torch.equal(mine[k], its[k]) for k in mine)

    g = torch.Generator().manual_seed(1)
    tokens = torch.randn(4, PATCHES, PATCH_DIM, generator=g)
    state = torch.randn(4, PROP_DIM, generator=g)
    act = torch.rand(4, ACTION_DIM, generator=g) * 2 - 1
    feat = torch.cat([tokens.flatten(1), state], dim=1)
    with torch.no_grad():
        q = ours(feat, act)
        assert q.shape == (10, 4, 1)
        assert torch.equal(q, theirs(tokens, state, act))
        assert torch.equal(ours.q_value_for_policy(feat, act), theirs.q_value_for_policy(tokens, state, act))
        torch.manual_seed(3)
        q_min = ours.q_value(feat, act)
        torch.manual_seed(3)
        assert torch.equal(q_min, theirs.q_value(tokens, state, act))


def test_the_resfit_actor_is_resfits_own_but_for_its_bound():
    """Built by off_policy's Actor at the same seed, so its compression and policy are ResFiT's bit for
    bit, its last layer at agent.actor.actor_last_layer_init_scale. ResFiT's mean is tanh of the policy's
    pre-activation times action_scale; this one's is equi_tanh of the same pre-activation times the
    per-field bound, which differs on the xy and tilt pairs alone: a disk, not a box."""
    from resfit.rl_finetuning.config.rlpd import ActorConfig
    from resfit.rl_finetuning.equi_off_policy.networks.obs_encoder import arm_action_shape, build_action_layout
    from resfit.rl_finetuning.equi_off_policy.rl.equi_rl_utils import equi_tanh
    from resfit.rl_finetuning.equi_off_policy.rl.resfit_heads import ResFiTActor
    from resfit.rl_finetuning.off_policy.rl.actor import Actor
    from escnn import gspaces
    from escnn.group import CyclicGroup

    group = gspaces.no_base_space(CyclicGroup(8))
    layout = build_action_layout(arm_action_shape(group, hand_dof=1), group)
    cfg = ActorConfig(action_scale=0.1, actor_last_layer_init_scale=None)  # nonzero, so the mean compares
    repr_dim = PATCHES * PATCH_DIM
    torch.manual_seed(0)
    ours = ResFiTActor(repr_dim, PROP_DIM, ACTION_DIM, cfg, action_layout=layout,
                       bound=torch.full((ACTION_DIM,), 0.1), residual_squash=True)
    torch.manual_seed(0)
    theirs = Actor(repr_dim, repr_dim, PROP_DIM, ACTION_DIM, cfg, residual_actor=True)
    assert isinstance(theirs.policy[-1], torch.nn.Tanh) and len(ours.policy) == len(theirs.policy) - 1
    mine = {**{f"compress.{k}": v for k, v in ours.compress.state_dict().items()},
            **{f"policy.{k}": v for k, v in ours.policy.state_dict().items()}}
    its = theirs.state_dict()
    assert mine.keys() == its.keys() and all(torch.equal(mine[k], its[k]) for k in mine)

    g = torch.Generator().manual_seed(1)
    obs = {"feat": torch.randn(4, PATCHES, PATCH_DIM, generator=g),
           "observation.state": torch.randn(4, PROP_DIM, generator=g),
           "observation.base_action": torch.rand(4, ACTION_DIM, generator=g) * 2 - 1}
    feat = torch.cat([obs["feat"].flatten(1), obs["observation.state"], obs["observation.base_action"]], dim=1)
    with torch.no_grad():
        pre = theirs.policy[:-1](torch.cat([theirs.compress(obs["feat"].flatten(1)), obs["observation.state"],
                                            obs["observation.base_action"]], dim=-1))
        assert torch.equal(theirs(obs, 0.0).mean, torch.tanh(pre) * 0.1)
        mu = ours(feat, std=None)
        assert torch.equal(mu, equi_tanh(pre, layout) * 0.1)
        assert pre.abs().max() > 0.1, "a near-zero pre-activation would not tell a disk from a box"


# ---------------------------------------------------------------------------
# The refusals (q_agent.py, check_families), before anything is built
# ---------------------------------------------------------------------------

MIX_REFUSALS = {
    # an equivariant head behind another encoder
    "equivariant actor, ablated encoder": (("ablated", "equivariant", "ablated"), {}, "needs the equivariant encoder"),
    "equivariant critic, resfit encoder": (("resfit", "resfit", "equivariant"), {}, "needs the equivariant encoder"),
    # ResFiT's critic behind another encoder
    "resfit critic, equivariant encoder": (("equivariant", "equivariant", "resfit"), {}, "encoder_family=resfit"),
    "resfit critic, ablated encoder": (("ablated", "resfit", "resfit"), {}, "encoder_family=resfit"),
    # ResFiT's critic with a distributional loss
    "resfit critic, hl_gauss": (("resfit", "resfit", "resfit"), {"agent.critic.loss.type": "hl_gauss"},
                                "mean squared error"),
    "unknown family": (("equivariant", "scalar", "equivariant"), {}, "expected one of"),
}

# Each field no network of the mix reads, off its default; at its default the mix is accepted.
UNREAD = {
    "encoder_norm": ("resfit", "inner_bn"), "encoder_orth_init": ("resfit", True),
    "initialize": ("resfit", False), "enc_degree_channel": ("resfit", 16), "ih_out_channel": ("resfit", 16),
    "float_images_as_uint8": ("resfit", True),
    "actor_norm": ("ere", "layer_norm"), "actor_degree_channel": ("ere", 448), "use_orth_init": ("ere", False),
    "critic_norm": ("resfit", "layer_norm"), "critic_pooled_norm": ("resfit", "layer_norm"),
    "num_critic_layers": ("resfit", 2), "critic_degree_channel": ("resfit", 64),
    "critic_orth_init": ("resfit", True), "critic_last_layer_init_scale": ("resfit", 0.5),
    "use_visual_proj": ("resfit", False),
    "frozen_target_critic": ("ablated", True),
}
STACKS = {"resfit": ("resfit", "resfit", "resfit"), "ablated": ("ablated", "ablated", "ablated"),
          "ere": ("equivariant", "resfit", "equivariant")}


def _cfg(families, **paths):
    from resfit.rl_finetuning.config.residual_td3 import ResidualEquiTD3SquareConfig

    cfg = ResidualEquiTD3SquareConfig()
    cfg.agent.device = "cpu"
    for part, family in zip(("encoder", "actor", "critic"), families):
        setattr(cfg.equivariance, f"{part}_family", family)
    for path, value in paths.items():
        node = cfg
        *parents, leaf = path.split(".")
        for parent in parents:
            node = getattr(node, parent)
        setattr(node, leaf, value)
    return cfg


def _agent_from(cfg):
    from resfit.rl_finetuning.equi_off_policy.rl.q_agent import QAgent
    from tools.agent_fingerprint import CAMERAS, OBS_SHAPE, STATE_DIM

    return QAgent(obs_shape=OBS_SHAPE, prop_shape=(STATE_DIM,), action_dim=ACTION_DIM, rl_cameras=list(CAMERAS),
                  agent_cfg=cfg.agent, equi_cfg=cfg.equivariance, residual_actor=True)


@pytest.mark.parametrize("case", list(MIX_REFUSALS))
def test_a_mix_the_agent_cannot_build_is_refused(case):
    families, paths, message = MIX_REFUSALS[case]
    with pytest.raises(ValueError, match=message):
        _agent_from(_cfg(families, **paths))


@pytest.mark.parametrize("field", list(UNREAD))
def test_a_field_no_network_reads_is_refused_off_its_default(field):
    """And check_families accepts the same mix with the field at its default, so the refusal is that
    field's. The ablated family reads every equivariant field, frozen_target_critic aside, so the
    same value is accepted there."""
    from resfit.rl_finetuning.equi_off_policy.rl.q_agent import check_families

    stack, value = UNREAD[field]
    with pytest.raises(ValueError, match=rf"reads equivariance\.\['{field}'\]"):
        _agent_from(_cfg(STACKS[stack], **{f"equivariance.{field}": value}))
    accepted = _cfg(STACKS[stack])
    check_families(accepted.equivariance, accepted.agent, residual_actor=True)
    if field != "frozen_target_critic":
        read = _cfg(STACKS["ablated"], **{f"equivariance.{field}": value})
        check_families(read.equivariance, read.agent, residual_actor=True)
