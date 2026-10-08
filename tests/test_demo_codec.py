"""`encode_env_frames_like_demos`: env frames through the demos' video codec.

The codec's settings are the ones that wrote the demos, its decode gives the dataset loader's
frames, the env wrapper codes its frames exactly when asked and leaves them bit-identical when
not, the switch moves the online cache key and no other, and coded env frames are as
indistinguishable from demo frames, to six per-frame statistics, as the demos' own first frames
are from each other (tools/codec_check.py, which renders, run in a subprocess as the gate runs
tools/render_check.py).
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import av
import numpy as np
import pytest

from resfit.dexmg.environments import demo_codec
from tools import codec_check

REPO = Path(__file__).resolve().parents[1]
CAN = "ankile/robomimic-mh-can-image"
CAN_ROOT = Path.home() / ".cache/huggingface/lerobot" / CAN
needs_can = pytest.mark.skipif(not CAN_ROOT.exists(), reason=f"{CAN} is not in the Hugging Face cache")


def _smooth_frame(seed: int) -> np.ndarray:
    """An 84x84 RGB frame with structure at several scales, as a render has."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:84, 0:84] / 84.0
    base = np.stack([np.sin(6 * xx + c) * np.cos(4 * yy - c) for c in (0.0, 1.0, 2.0)], axis=-1)
    return np.clip(127 + 90 * base + rng.normal(0, 12, base.shape), 0, 255).astype(np.uint8)


@needs_can
def test_the_settings_are_the_ones_that_wrote_the_demos():
    from lerobot.common.datasets.video_utils import encode_video_frames

    defaults = {k: p.default for k, p in inspect.signature(encode_video_frames).parameters.items()}
    assert (defaults["vcodec"], defaults["pix_fmt"], defaults["g"], defaults["crf"]) == (
        demo_codec.VCODEC, demo_codec.PIX_FMT, demo_codec.GOP, demo_codec.CRF)

    info = json.loads((CAN_ROOT / "meta/info.json").read_text())
    assert info["fps"] == demo_codec.FPS
    for cam in codec_check.CAMERAS:
        video = info["features"][cam]["info"]
        assert (video["video.codec"], video["video.pix_fmt"]) == ("av1", demo_codec.PIX_FMT), cam
        # A keyframe every GOP frames, as encode_like_demos's two-frame clip has.
        with av.open(str(CAN_ROOT / f"videos/chunk-000/{cam}/episode_000000.mp4")) as f:
            keys = [p.is_keyframe for p in f.demux(video=0) if p.size][:20]
        assert keys == [i % demo_codec.GOP == 0 for i in range(len(keys))], (cam, keys)


@needs_can
def test_the_decode_gives_the_dataset_loaders_frames():
    """encode_like_demos decodes as below; the trainer's demo frames are the loader's, through to_uint8."""
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(CAN)
    for cam in codec_check.CAMERAS:
        with av.open(str(CAN_ROOT / f"videos/chunk-000/{cam}/episode_000000.mp4")) as f:
            decoded = [fr.to_ndarray(format="rgb24") for _, fr in zip(range(3), f.decode(video=0))]
        for t, frame in enumerate(decoded):
            loader = (ds[t][cam].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            np.testing.assert_array_equal(frame, loader, err_msg=f"{cam} frame {t}")


def test_the_codec_keeps_shape_and_dtype_and_is_deterministic():
    frame = _smooth_frame(0)
    coded = demo_codec.encode_like_demos(frame)
    assert coded.shape == frame.shape and coded.dtype == np.uint8
    np.testing.assert_array_equal(coded, demo_codec.encode_like_demos(frame))
    assert 0 < np.abs(coded.astype(int) - frame.astype(int)).mean() < 10


def test_the_codec_removes_colour_detail_as_yuv420p_does():
    frames = np.stack([_smooth_frame(s) for s in range(6)])
    coded = np.stack([demo_codec.encode_like_demos(f) for f in frames])
    raw, out = (codec_check.frame_stats(x.transpose(0, 3, 1, 2)) for x in (frames, coded))
    for s in ("cb_detail", "cr_detail"):
        assert (out[s] < 0.7 * raw[s]).all(), s


def _wrapper(encode: bool):
    """The env wrapper's observation processing, without a simulator behind it."""
    from resfit.dexmg.environments.dexmg import RobosuiteGymWrapper

    w = object.__new__(RobosuiteGymWrapper)
    w.env_name = "PickPlaceCan"
    w.expected_image_keys = ["agentview_image", "robot0_eye_in_hand_image"]
    w.encode_frames_like_demos = encode
    return w


def test_the_env_wrapper_codes_its_frames_exactly_when_asked():
    obs = {"agentview_image": _smooth_frame(1), "robot0_eye_in_hand_image": _smooth_frame(2)}
    for encode in (False, True):
        out = _wrapper(encode)._process_obs(obs)
        for key, img in obs.items():
            seen = out[f"observation.images.{key.replace('_image', '')}"]
            source = demo_codec.encode_like_demos(img) if encode else img
            # Off is the code before the switch existed, bit for bit.
            np.testing.assert_array_equal(seen, np.transpose(source.astype(np.float32) / 255.0, (2, 0, 1)))
    coded = _wrapper(True)._process_obs(obs)["observation.images.agentview"]
    assert not np.array_equal(coded, _wrapper(False)._process_obs(obs)["observation.images.agentview"])


def test_the_switch_moves_the_online_cache_key_and_no_other():
    import preflight

    for method, stack in (("residual", "equi"), ("tuned-rlpd", "equi"), ("tuned-rlpd", "resfit")):
        off_key, on_key = preflight.cache_hashes("Can", method, stack)
        off_coded, on_coded = preflight.cache_hashes("Can", method, stack, encode_env_frames_like_demos=True)
        assert off_coded == off_key and on_coded != on_key, (method, stack)


def test_the_trainer_threads_the_switch_into_both_envs_and_the_key():
    """Both env factories (training and eval) go through get_envs, which passes the switch on."""
    src = (REPO / "resfit/rl_finetuning/scripts/train_residual_td3.py").read_text()
    assert src.count("encode_frames_like_demos=cfg.encode_env_frames_like_demos") == 1
    assert src.count("env = get_envs(") == 2  # env and eval_env
    assert 'online_cache_meta["encode_env_frames_like_demos"] = True' in src


def test_the_separation_arithmetic():
    a = np.array([0.1, 0.2, 0.3])
    assert codec_check.auc(a, a + 1.0) == 1.0
    assert codec_check.auc(a + 1.0, a) == 0.0
    assert codec_check.auc(a, a) == 0.5
    frames = np.stack([_smooth_frame(s) for s in range(4)]).transpose(0, 3, 1, 2)
    assert set(codec_check.separation(frames, frames).values()) == {0.5}
    np.testing.assert_array_equal(codec_check.to_uint8(frames.astype(np.float32) / 255.0), frames)


@needs_can
@pytest.mark.slow
def test_coded_env_frames_cannot_be_told_from_demo_frames():
    """Rendered: raw env frames are told from the demos by colour on the front camera; coded, by nothing.

    Bounds: at 30 frames a side, the ROC area of two sets from one distribution has a standard
    error of about 0.075, so [0.15, 0.85] is 4.5 of them. Measured coded: 0.39 to 0.67.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "tools.codec_check", "--task", "Can", "--frames", "30", "--json"],
        cwd=REPO, capture_output=True, text=True, check=False,
        env={**os.environ, "PYTHONPATH": f"{REPO}:{os.environ.get('PYTHONPATH', '')}"},
    )
    found = [json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{")]
    assert len(found) == 1, proc.stderr[-2000:]
    r = found[0]
    front = "observation.images.agentview"
    for s in ("cb_detail", "cr_detail", "mean_R", "mean_G", "mean_B"):
        assert r["raw"][front][s] >= 0.95, (s, r["raw"][front])  # the cue is there to remove
    for cam, by_stat in r["coded"].items():
        for s, area in by_stat.items():
            assert 0.15 <= area <= 0.85, (cam, s, area)
