"""Can a per-frame statistic tell env frames from demo frames, before and after the demos' codec?

    python -m tools.codec_check --task Can [--frames 40] [--rerendered] [--json]

The demos reach the agent through a lossy video codec and the env's frames do not. On Can's front
camera that leaves every env frame with about twice the demos' colour detail and an even colour
offset of about 0.005 (EXPERIMENTS.md, 2026-10-08 FINDING). `encode_env_frames_like_demos` passes
the env's frames through the demos' codec (resfit/dexmg/environments/demo_codec.py). This
measures, on both cameras, how well each of six statistics separates `frames` demo first frames
from `frames` env reset frames, raw and through the codec, as the area under the ROC curve: 0.5
when the statistic cannot tell them apart, 0 or 1 when it always can.

The statistics: the detail in luma (y) and in the two colour differences (cb = B - y, cr = R - y),
each the power left after subtracting a 2x2 box blur; and the mean of each colour channel.

Measured on 2026-10-08, Can, 40 frames (the demos' first frames are keyframes, as the codec's are):

    front camera   y     cb    cr    R     G     B
    raw            0.65  1.00  1.00  1.00  1.00  1.00
    through codec  0.45  0.50  0.47  0.50  0.60  0.62

The wrist camera's raw separation is weaker (cb 0.98, cr 0.76, colour means about 0.55), and the
codec takes cb to 0.66. Its y detail separates at about 0.65 either way: the env's reset poses are
not the demos' start poses, which the wrist camera sees close up.

`--rerendered` takes the demo first frames from the re-rendered datasets (tools/rerender_demos.py),
whose frames come from our renderer losslessly, instead of the source datasets.

Demo frames come through LeRobotDataset and env frames through create_vectorized_env, the paths
training uses. The env frames are coded here by demo_codec.encode_like_demos; that the env wrapper
codes them the same way is tests/test_demo_codec.py's.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy import ndimage

CAMERAS = ("observation.images.agentview", "observation.images.robot0_eye_in_hand")
DATASETS = {"Can": "ankile/robomimic-mh-can-image", "Square": "ankile/robomimic-mh-square-image"}
FRAMES = 40
STATS = ("y_detail", "cb_detail", "cr_detail", "mean_R", "mean_G", "mean_B")


def frame_stats(frames: np.ndarray) -> dict[str, np.ndarray]:
    """Per frame, each of STATS. frames: [n, 3, H, W] uint8."""
    assert frames.dtype == np.uint8 and frames.ndim == 4 and frames.shape[1] == 3, (frames.dtype, frames.shape)
    x = frames.astype(np.float32) / 255.0
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    out = {}
    for name, z in (("y", y), ("cb", b - y), ("cr", r - y)):
        out[f"{name}_detail"] = ((z - ndimage.uniform_filter(z, size=(1, 2, 2))) ** 2).mean(axis=(1, 2))
    for i, c in enumerate("RGB"):
        out[f"mean_{c}"] = x[:, i].mean(axis=(1, 2))
    return out


def auc(demo: np.ndarray, env: np.ndarray) -> float:
    """P(a demo value < an env value), ties counting half: the ROC area of the statistic."""
    return float((demo[:, None] < env[None, :]).mean() + 0.5 * (demo[:, None] == env[None, :]).mean())


def separation(demo: np.ndarray, env: np.ndarray) -> dict[str, float]:
    """Each statistic's ROC area between two sets of frames, [n, 3, H, W] uint8."""
    d, e = frame_stats(demo), frame_stats(env)
    return {s: auc(d[s], e[s]) for s in STATS}


def to_uint8(frames) -> np.ndarray:
    """[0, 1] floats that came from uint8, back to those bytes exactly."""
    x = np.asarray(frames, dtype=np.float32)
    assert 0.0 <= x.min() and x.max() <= 1.0, "frames not in [0, 1]"
    return np.round(x * 255.0).astype(np.uint8)


def demo_dataset(task: str, rerendered: bool = False) -> str:
    from tools.rerender_demos import SOURCES

    return SOURCES[task][3] if rerendered else DATASETS[task]


def demo_first_frames(task: str, n: int, rerendered: bool = False) -> dict[str, np.ndarray]:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(demo_dataset(task, rerendered))
    starts = ds.episode_data_index["from"]
    episodes = np.linspace(0, ds.meta.total_episodes - 1, n).astype(int)
    items = [ds[int(starts[ep])] for ep in episodes]
    return {cam: to_uint8(np.stack([it[cam].numpy() for it in items])) for cam in CAMERAS}


def env_reset_frames(task: str, n: int) -> dict[str, np.ndarray]:
    os.environ["MUJOCO_GL"] = "egl"  # as the trainer sets it, before robosuite is imported
    from resfit.dexmg.environments.dexmg import create_vectorized_env

    venv = create_vectorized_env(env_name=task, num_envs=1, debug=True)
    try:
        frames = {cam: [] for cam in CAMERAS}
        for _ in range(n):
            obs, _ = venv.reset()
            for cam in CAMERAS:
                frames[cam].append(to_uint8(obs[cam].cpu().numpy()[0]))
        return {cam: np.stack(f) for cam, f in frames.items()}
    finally:
        venv.close()


def measure(task: str, frames: int = FRAMES, rerendered: bool = False) -> dict:
    from resfit.dexmg.environments.demo_codec import encode_like_demos

    demo, env = demo_first_frames(task, frames, rerendered), env_reset_frames(task, frames)
    out = {"task": task, "demos": demo_dataset(task, rerendered), "frames": frames, "raw": {}, "coded": {}}
    for cam in CAMERAS:
        coded = np.stack([encode_like_demos(f.transpose(1, 2, 0)).transpose(2, 0, 1) for f in env[cam]])
        out["raw"][cam] = separation(demo[cam], env[cam])
        out["coded"][cam] = separation(demo[cam], coded)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", action="append", choices=sorted(DATASETS), required=True)
    ap.add_argument("--frames", type=int, default=FRAMES)
    ap.add_argument("--rerendered", action="store_true", help="demo frames from the re-rendered datasets")
    ap.add_argument("--json", action="store_true", help="one JSON line per task")
    args = ap.parse_args()
    for task in args.task:
        r = measure(task, args.frames, args.rerendered)
        if args.json:
            print(json.dumps(r))
            continue
        print(f"{task}, {args.frames} first frames of {r['demos']} vs {args.frames} env resets: ROC area per statistic")
        for cam in CAMERAS:
            print(f"  {cam}")
            for version in ("raw", "coded"):
                print(f"    {version:6s} " + "  ".join(f"{s} {r[version][cam][s]:.2f}" for s in STATS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
