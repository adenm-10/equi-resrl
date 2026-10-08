"""Does the env render the scene the demos show? Demo first frames against env reset frames.

    python -m tools.render_check --task Can --task Square [--frames 40] [--repeats 1] [--rerendered] [--json]

mujoco 3.8.1, in the `residual` env from 2026-05-21 to 2026-10-07, draws robosuite's floor as dark
planks where the demos show light grey tiles. A critic can then tell every demo frame from every
online frame by the floor alone (EXPERIMENTS.md, 2026-10-07 FINDING). This measures that gap on
the front camera, which the floor shows in, through the paths training uses: the demos through
LeRobotDataset, the env through create_vectorized_env.

The statistic is the mean absolute difference, per pixel and channel in [0, 1], between the
average of `frames` demo first frames, spread across the dataset, and the average of `frames` env
reset frames. Its bar is 2.5 times the same statistic between the even and the odd demos of that
set, what two sets of demo first frames differ by. The demo side is fixed, so the bar is too; env
resets are unseeded, so the gap varies between calls (`--repeats` measures by how much).

Why 2.5 and not 2, measured on 2026-10-07 at 40 frames (10 repeats on 3.3.2, 3 on 3.8.1):

    task    within demos  bar     demo vs env, mujoco 3.3.2   demo vs env, mujoco 3.8.1
    Can     0.0050        0.0125  0.0103 +- 0.0003            0.098
    Square  0.0052        0.0130  0.0086 +- 0.0005            0.0166 +- 0.0007

Even the matching render differs from the demos by about 2x the within-demo gap, through a small
even colour offset (about 0.005 per channel, on static pixels too) that both mujoco versions
share; the demos were stored as video. So a 2x bar fails a correct Can env at random. 2.5x clears
the matching render by 7 sd on Can and 9 on Square, and fails 3.8.1 by 5 sd on Square, where its
floor is a thin strip.

`--rerendered` takes the demo first frames from the re-rendered datasets (tools/rerender_demos.py),
rendered by our env and stored losslessly, instead of the source datasets the gate checks against.

preflight.py runs this as a subprocess and fails the gate when a task's gap is above its bar.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

CAMERA = "observation.images.agentview"
DATASETS = {"Can": "ankile/robomimic-mh-can-image", "Square": "ankile/robomimic-mh-square-image"}
FRAMES = 40
BAR_FACTOR = 2.5


def mean_frame_gap(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean |average of a - average of b| over pixels and channels. a, b: [n, C, H, W] in [0, 1]."""
    return (a.float().mean(0) - b.float().mean(0)).abs().mean().item()


def verdict(demo: torch.Tensor, env: torch.Tensor) -> dict:
    """The gap between demo and env frames, against twice the gap between two halves of the demos."""
    for name, x in (("demo", demo), ("env", env)):
        assert x.dtype.is_floating_point and 0.0 <= x.min() and x.max() <= 1.0, f"{name} frames not in [0, 1]"
    within = mean_frame_gap(demo[0::2], demo[1::2])
    across = mean_frame_gap(demo, env)
    return {"within_demos": within, "demo_vs_env": across, "bar": BAR_FACTOR * within,
            "passed": across <= BAR_FACTOR * within}


def demo_dataset(task: str, rerendered: bool = False) -> str:
    from tools.rerender_demos import SOURCES

    return SOURCES[task][3] if rerendered else DATASETS[task]


def demo_first_frames(task: str, n: int, rerendered: bool = False) -> torch.Tensor:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(demo_dataset(task, rerendered))
    starts = ds.episode_data_index["from"]
    episodes = np.linspace(0, ds.meta.total_episodes - 1, n).astype(int)
    return torch.stack([ds[int(starts[ep])][CAMERA].float() for ep in episodes])


def env_reset_frames(task: str, n: int, shape: torch.Size) -> torch.Tensor:
    os.environ["MUJOCO_GL"] = "egl"  # as the trainer sets it, before robosuite is imported
    from resfit.dexmg.environments.dexmg import create_vectorized_env

    venv = create_vectorized_env(env_name=task, num_envs=1, debug=True)
    try:
        frames = []
        for _ in range(n):
            obs, _ = venv.reset()
            frame = torch.as_tensor(obs[CAMERA]).float().reshape(shape)
            frames.append(frame / 255.0 if frame.max() > 1.0 else frame)
        return torch.stack(frames)
    finally:
        venv.close()


def measure(task: str, frames: int = FRAMES, repeats: int = 1, rerendered: bool = False) -> dict:
    import mujoco

    demo = demo_first_frames(task, frames, rerendered)
    runs = [verdict(demo, env_reset_frames(task, frames, demo.shape[1:])) for _ in range(repeats)]
    gaps = [r["demo_vs_env"] for r in runs]
    return {"task": task, "demos": demo_dataset(task, rerendered), "camera": CAMERA, "frames": frames,
            "mujoco": mujoco.__version__,
            "within_demos": runs[0]["within_demos"], "bar": runs[0]["bar"],
            "demo_vs_env": gaps, "passed": all(r["passed"] for r in runs)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", action="append", choices=sorted(DATASETS), required=True)
    ap.add_argument("--frames", type=int, default=FRAMES)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--rerendered", action="store_true", help="demo frames from the re-rendered datasets")
    ap.add_argument("--json", action="store_true", help="print one JSON object per task, last on stdout")
    args = ap.parse_args()

    results = [measure(task, args.frames, args.repeats, args.rerendered) for task in args.task]
    for r in results:
        if args.json:
            print(json.dumps(r))
        else:
            gaps = ", ".join(f"{g:.4f}" for g in r["demo_vs_env"])
            print(f"{r['task']}: demo vs env {gaps}; within demos {r['within_demos']:.4f}, "
                  f"bar {r['bar']:.4f} -> {'PASS' if r['passed'] else 'FAIL'} (mujoco {r['mujoco']})")
    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
