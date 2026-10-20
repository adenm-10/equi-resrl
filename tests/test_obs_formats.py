"""Tier 2: the agent acts on the observations the env produces, not only on buffer batches.

The env hands act() float images in [0, 1] (dexmg.py). The replay buffer stores
uint8 copies in [0, 255] (utils/dtype.py), and every update, golden and
fingerprint feeds those. Until 2026-10-02 the encoder divided both by 255, so
every equivariant run acted on near-blank images and trained on full-contrast
ones. No test fed it a float image, so none could see that (EXPERIMENTS.md,
2026-10-02 CORRECTION).

Here the same frames go in both formats, made as the env and the buffer make
them, through each encoder family: the equivariant and ablated encoders' scaling
to [-1, 1], and the ResFiT encoder's own (÷255 for uint8, then −0.5).
equivariance.float_images_as_uint8 must still reproduce the old scaling in the
two encoders that read it, and must leave the buffer format, which is all the
update ever sees, alone.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from resfit.rl_finetuning.utils.dtype import to_uint8
from tests.conftest import CROP, ENC_HIDDEN, IH_OUT, N, OBS_SHAPE, ONE_ARM, wrist_keys
from tools.agent_fingerprint import ACTION_DIM, ROTATION_CENTRE, STATE_DIM, _stats

CAMERAS = ("observation.images.agentview", *wrist_keys(1))
FRAMES = 4


def _observations(device) -> tuple[dict, dict]:
    """The same observations as the env returns them and as the buffer stores them."""
    g = torch.Generator().manual_seed(0)
    rendered = {cam: torch.randint(0, 256, (FRAMES, *OBS_SHAPE), generator=g, dtype=torch.uint8)
                for cam in CAMERAS}
    env = {cam: torch.from_numpy(img.numpy().astype(np.float32) / 255.0)  # dexmg.py
           for cam, img in rendered.items()}
    env["observation.state"] = torch.randn(FRAMES, STATE_DIM, generator=g)
    env["observation.base_action"] = torch.rand(FRAMES, ACTION_DIM, generator=g) * 2 - 1
    buffer = to_uint8(dict(env), list(CAMERAS))  # what the trainer stores
    assert all(torch.equal(buffer[cam], rendered[cam]) for cam in CAMERAS)
    return ({k: v.to(device) for k, v in env.items()},
            {k: v.to(device) for k, v in buffer.items()})


def _encoder(device, encoder: str, float_images_as_uint8: bool):
    from resfit.rl_finetuning.config.rlpd import VitEncoderConfig
    from resfit.rl_finetuning.equi_off_policy.networks.equi_normalizer import (
        build_equivariant_normalizer,
    )
    from resfit.rl_finetuning.equi_off_policy.networks.obs_encoder import ResObsEnc

    torch.manual_seed(0)
    np.random.seed(0)
    enc = ResObsEnc(
        obs_shape=OBS_SHAPE, crop_shape=(CROP, CROP), N=N, n_hidden=ENC_HIDDEN,
        ih_out_channel=IH_OUT, encoder=encoder, initialize=True,
        vit_cfg=VitEncoderConfig() if encoder == "resfit" else None,
        wrist_keys=wrist_keys(1), float_images_as_uint8=float_images_as_uint8, **ONE_ARM,
    )
    enc.set_normalizer(
        build_equivariant_normalizer(
            stats={"observation.state": _stats(STATE_DIM), "action": _stats(ACTION_DIM)},
            robot_base_xy=torch.tensor(ROTATION_CENTRE),
        ),
        robot_base_xy=list(ROTATION_CENTRE),
    )
    return enc.to(device).eval()


def _features(enc, obs) -> list[torch.Tensor]:
    with torch.no_grad():
        out = enc(dict(obs))
    return [f.tensor if hasattr(f, "tensor") else f for f in out]


def _rel_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / b.norm()).item()


@pytest.mark.gpu
@pytest.mark.parametrize("encoder", ["equivariant", "ablated", "resfit"])
def test_encoder_reads_env_and_buffer_images_alike(device, encoder, no_tf32):
    """Up to float rounding: on CUDA, dividing by 255 multiplies by its reciprocal,
    so the two formats can differ in the last bit. TF32 would round that up to
    about 1e-3."""
    env, buffer = _observations(device)
    enc = _encoder(device, encoder, float_images_as_uint8=False)
    for got, expected in zip(_features(enc, env), _features(enc, buffer)):
        torch.testing.assert_close(got, expected)


@pytest.mark.gpu
@pytest.mark.parametrize("encoder", ["equivariant", "ablated"])
def test_float_images_as_uint8_reproduces_the_old_scaling(device, encoder, no_tf32):
    """The flag changes what act() sees and nothing the update sees. The ResFiT encoder does not
    read it (q_agent.py refuses it there)."""
    env, buffer = _observations(device)
    fixed = _encoder(device, encoder, float_images_as_uint8=False)
    old = _encoder(device, encoder, float_images_as_uint8=True)
    for got, expected in zip(_features(old, buffer), _features(fixed, buffer)):
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
    # The image features lead the critic's feature vector; proprioception and the
    # base action, the same in both formats, follow them.
    images = slice(0, old.enc_out_dim_critic - old.prop_dim)
    _, critic_env = _features(old, env)
    _, critic_buffer = _features(old, buffer)
    rel = _rel_diff(critic_env[:, images], critic_buffer[:, images])
    assert rel > 0.1, f"the flag no longer reproduces the bug: {rel:.2e}"


@pytest.mark.gpu
@pytest.mark.parametrize("float_images_as_uint8", [False, True], ids=["fixed", "old"])
def test_act_is_the_same_on_env_and_buffer_observations(device, float_images_as_uint8, no_tf32):
    """The whole agent at the Square reference config, in eval mode, as the trainer acts."""
    from tools.agent_fingerprint import build_reference_agent

    agent = build_reference_agent(str(device), float_images_as_uint8=float_images_as_uint8)
    agent.train(False)
    env, buffer = _observations(device)
    with torch.no_grad():
        a_env = agent.act(env, eval_mode=True, cpu=False)
        a_buffer = agent.act(buffer, eval_mode=True, cpu=False)
    rel = _rel_diff(a_env, a_buffer)
    if float_images_as_uint8:
        assert rel > 0.1, f"the old scaling should change the action: {rel:.2e}"
    else:
        assert rel < 1e-4, f"act() differs between the env's and the buffer's images: {rel:.2e}"


@pytest.mark.gpu
def test_vanilla_act_is_the_same_on_env_and_buffer_observations(device, no_tf32):
    """Vanilla RL's agent (base_policy=None) acts on the env's observations, which carry no base
    action, as it acts on the buffer's."""
    from tools.agent_fingerprint import build_reference_agent

    agent = build_reference_agent(str(device), residual_actor=False, overrides={
        "agent.actor.action_scale": 1.0, "agent.actor.actor_last_layer_init_scale": None})
    agent.train(False)
    env, buffer = _observations(device)
    for obs in (env, buffer):
        del obs["observation.base_action"]
    with torch.no_grad():
        a_env = agent.act(env, eval_mode=True, cpu=False)
        a_buffer = agent.act(buffer, eval_mode=True, cpu=False)
    assert a_env.abs().max().item() > 1e-3, "a zero action would pass trivially"
    rel = _rel_diff(a_env, a_buffer)
    assert rel < 1e-4, f"act() differs between the env's and the buffer's images: {rel:.2e}"
