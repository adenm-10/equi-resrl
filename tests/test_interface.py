"""Tier 2: the encoder's interface to the critic (equivariance.ih_channels, enc_out_channel,
critic_window, critic_window_ih; EXPERIMENTS.md, 2026-10-23 CODE), at enc-iface-1023's arms.

  Tier 1  The encoders in two slices, around their 9x9 maps, compute what they compute whole, bit
          for bit, equivariant and exported: every family's forward runs that way from this commit.
          The critic refuses a window without its visual projection, and the ResFiT encoder refuses
          the fields that configure the others (check_families' refusals are
          tests/test_resfit_heads.py's).
  Tier 2  Each arm hands its critic and its actor the numbers the plan lists (288, 576, 1,280,
          1,280 to the critic; 288, 576, 1,280, 288 to the actor), its encoder's outputs as wide.
          iface-window starts as iface-eee: at seed 2 the same random state after construction,
          every tensor they share equal bit for bit, the same features and Q exactly, the same first
          critic loss, a window share of 0; after two updates every window weight has moved and the
          share is above 0. A saved model from before the interface's commit loads into iface-eee.
"""

from __future__ import annotations

import contextlib
import os

import numpy as np
import pytest
import torch

from tests.conftest import N

ARMS = ("iface-eee", "iface-wide-all", "iface-wide-out", "iface-window")
# (critic, actor): the image numbers each network reads per frame
SIZES = {"iface-eee": (288, 288), "iface-wide-all": (576, 576), "iface-wide-out": (1280, 1280),
         "iface-window": (1280, 288)}
WINDOW_KEYS = ("enc.window_obs.", "enc.window_ih.", "critic.window_proj.", "critic_target.window_proj.")


# ---------------------------------------------------------------------------
# Tier 1
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("norm", ["none", "inner_bn"])
def test_the_agentview_encoder_in_two_slices_is_the_encoder_whole(norm, device):
    """forward_map runs conv[:map_end] then conv[map_end:]; the map is the 9x9 one, n_out fields,
    and the output has out_fields. Exported to plain layers (the ablated family), sliced the same.
    On the session's device: escnn shares its bases between layers, wherever the first was built."""
    from resfit.rl_finetuning.equi_off_policy.networks.equi_encoder import EquivariantResEncoder76Cyclic
    from resfit.rl_finetuning.equi_off_policy.rl.ablated import plain

    torch.manual_seed(0)
    for out_fields in (None, 24):
        enc = EquivariantResEncoder76Cyclic(obs_channel=3, n_out=16, N=N, norm=norm,
                                            out_fields=out_fields).to(device).eval()
        x = torch.randn(2, 3, 76, 76, device=device)
        with torch.no_grad():
            whole = enc(x).tensor
            out, fmap = enc.forward_map(x)
            assert torch.equal(out.tensor, whole)
            assert fmap.tensor.shape == (2, 16 * N, 9, 9)
            assert whole.shape == (2, (out_fields or 16) * N, 1, 1)
            if norm == "none":
                flat = plain(enc.conv)
                assert torch.equal(flat[enc.map_end:](flat[:enc.map_end](x)), flat(x))
                assert torch.equal(flat(x), whole)


def test_the_wrist_encoder_in_two_slices_is_the_encoder_whole(device):
    from resfit.rl_finetuning.equi_off_policy.networks.ablate_equi_encoder import ResEncoder76InHand

    torch.manual_seed(0)
    for channels in ((16, 32, 64, 128), (32, 64, 128, 256)):
        enc = ResEncoder76InHand(obs_channel=3, channels=channels, n_out=32).to(device).eval()
        x = torch.randn(2, 3, 76, 76, device=device)
        with torch.no_grad():
            out, fmap = enc.forward_map(x)
            assert torch.equal(out, enc(x)) and out.shape == (2, 32)
            assert fmap.shape == (2, channels[-1], 9, 9) and enc.map_channels == channels[-1]


def test_a_window_needs_the_visual_projection(device):
    from resfit.rl_finetuning.equi_off_policy.networks.obs_encoder import ResObsEnc
    from tests.conftest import build_critic

    enc = ResObsEnc(N=N, n_hidden=16, ih_out_channel=16, encoder="equivariant", critic_window=2,
                    critic_window_ih=4).to(device)
    with pytest.raises(ValueError, match="use_visual_proj"):
        build_critic(enc, device, window_type=enc.window_type, use_visual_proj=False)


@pytest.mark.parametrize("field", [{"enc_out_channel": 64}, {"critic_window": 8}, {"critic_window_ih": 8}],
                         ids=lambda f: next(iter(f)))
def test_the_resfit_encoder_refuses_the_other_encoders_fields(field):
    from resfit.rl_finetuning.config.rlpd import VitEncoderConfig
    from resfit.rl_finetuning.equi_off_policy.networks.obs_encoder import ResObsEnc

    with pytest.raises(ValueError, match="equivariant and ablated"):
        ResObsEnc(N=N, ih_out_channel=32, encoder="resfit", vit_cfg=VitEncoderConfig(), **field)


# ---------------------------------------------------------------------------
# Tier 2: the arms
# ---------------------------------------------------------------------------


def _arm_agent(arm, tmp_path, device, seed=2):
    """The agent the trainer builds for `arm` at `seed`, and torch's and numpy's random states after."""
    from tests.test_sweep_arms import _compose, _overrides, _seeded_agent

    with contextlib.redirect_stdout(open(os.devnull, "w")):  # QAgent prints its modules
        agent = _seeded_agent(_compose(_overrides(arm, tmp_path)), device, seed=seed)
    return agent, (torch.get_rng_state(), np.random.get_state()[1].copy())


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("arm", ARMS)
def test_each_arm_hands_its_critic_and_actor_the_listed_numbers(arm, tmp_path, device):
    from tools.agent_fingerprint import fixed_batch

    agent, _ = _arm_agent(arm, tmp_path, device)
    enc = agent.enc
    window = 0 if enc.window_type is None else enc.window_type.size
    assert (enc.vis_ih_type.size + window, enc.vis_ih_type.size) == SIZES[arm]
    agent.train(False)
    with torch.no_grad():
        actor_feat, critic_feat = enc(dict(fixed_batch().to(device)["obs"]))
    prop, action = enc.prop_type.size, enc.action_type.size
    assert (critic_feat.tensor.shape[1] - prop, actor_feat.tensor.shape[1] - prop - action) == SIZES[arm]


@pytest.mark.gpu
@pytest.mark.slow
def test_the_window_arm_starts_as_the_anchor(tmp_path, device, no_tf32):
    """The window is built with the random state forked and its projection starts at zero and draws
    nothing, so at seed 2 iface-window is iface-eee's agent plus weights that change nothing yet."""
    from tools.agent_fingerprint import STDDEV, UPDATE_SEED, _seed, fixed_batch

    eee, eee_rng = _arm_agent("iface-eee", tmp_path, device)
    win, win_rng = _arm_agent("iface-window", tmp_path, device)
    assert torch.equal(eee_rng[0], win_rng[0]) and np.array_equal(eee_rng[1], win_rng[1])

    theirs, mine = eee.state_dict(), win.state_dict()
    assert theirs.keys() <= mine.keys()
    extra = mine.keys() - theirs.keys()
    assert extra and all(k.startswith(WINDOW_KEYS) for k in extra), sorted(extra)
    for key, value in theirs.items():
        assert torch.equal(mine[key], value), f"iface-window: {key} differs from iface-eee's"
    assert win.critic.window_proj.weights.abs().max() == 0

    eee.train(False)
    win.train(False)
    batch = fixed_batch().to(device)
    with torch.no_grad():
        (ea, ec), (wa, wc) = eee.enc(dict(batch["obs"])), win.enc(dict(batch["obs"]))
        assert torch.equal(ea.tensor, wa.tensor)
        images, prop = eee.enc.vis_ih_type.size, eee.enc.prop_type.size
        assert torch.equal(wc.tensor[:, :images], ec.tensor[:, :images])
        assert torch.equal(wc.tensor[:, -prop:], ec.tensor[:, -prop:])
        assert wc.tensor[:, images:-prop].abs().max() > 0, "the window's features are all zero"
        assert torch.equal(win.critic(wc, batch["action"]), eee.critic(ec, batch["action"]))
        assert win.critic.window_share(wc).item() == 0.0
        metrics = win.critic_diagnostics(batch, num_online=2)
        assert metrics["debug/critic_window_share"] == 0.0

    losses = []
    for agent in (eee, win):
        agent.train(True)
        _seed(UPDATE_SEED)
        losses.append(agent.update(fixed_batch().to(device), stddev=STDDEV, update_actor=False)["train/critic_loss"])
    assert losses[0] == losses[1], losses

    before = {k: v.clone() for k, v in win.state_dict().items() if k.startswith(WINDOW_KEYS[:3])
              and not k.endswith("sampled_basis")}
    _seed(UPDATE_SEED + 1)
    win.update(fixed_batch().to(device), stddev=STDDEV, update_actor=False)
    after = win.state_dict()
    still = [k for k, v in before.items() if torch.equal(after[k], v)]
    assert not still, f"window weights that did not move in two updates: {still}"
    win.train(False)
    with torch.no_grad():
        assert win.critic.window_share(win.enc(dict(batch["obs"]))[1]).item() > 0


@pytest.mark.gpu
@pytest.mark.slow
def test_a_saved_model_from_before_the_interface_loads_into_the_anchor(tmp_path, device):
    """wk45zp27 (arch-swap-eee 2, from aec1bdc) saved its best model before map_end and the window
    existed; neither moved a module, so it loads into iface-eee's agent strictly."""
    from resfit.lerobot.utils.load_policy import RUNS_ROOT
    from resfit.rl_finetuning.utils.checkpoint import load_residual_model

    path = RUNS_ROOT / "robomimic-square-final/wk45zp27/wandb_best/best_model.pt"
    if not path.exists():
        pytest.skip(f"{path} is not on this machine")
    agent, _ = _arm_agent("iface-eee", tmp_path, device)
    saved = load_residual_model(agent, path)
    assert saved["global_step"] == 70_000
