"""MimicGen's Stack_D0 demos, rendered in our env, losslessly, as a LeRobot dataset (TODO B25, step 3).

    python -m tools.build_stack_demos [--gpu 1] [--episodes N] --check-only   # guards and measures, no writing
    python -m tools.build_stack_demos [--gpu 1]                               # build
    python -m tools.build_stack_demos --verify [--episodes 5]                 # a built dataset against fresh renders

Why. Square's demos were re-rendered by our env so that a critic cannot tell a demo frame from an online
one by how it was made (tools/rerender_demos.py; EXPERIMENTS.md, 2026-10-08). Stack has no LeRobot source
to copy columns from: MimicGen's hdf5 (`core/stack_d0.hdf5`, recorded under robosuite 1.4.1) is the only
source, so every column is built here, from it and from our env.

How. robosuite 1.5.1's own `Stack` is Stack_D0's scene, and the demos' states set into it directly (same
11 joints in the same order; only the gripper joints are renamed; EXPERIMENTS.md, 2026-10-29 CODE, Stack
steps 1-2). Each episode is rendered by rerender_demos.render_states: its first frame as reset() observes,
every later one as step() does, one physics substep late. Columns:
  action               the demo's actions, as float32;
  next.done            the demo's `dones`, MimicGen's success flags at its last 1-5 steps, as robomimic's
                       `dones` give Square's (the trainer labels a demo transition's reward with it);
  observation.state    the stepped state our env renders, [eef_pos 3, eef_quat 4 (xyzw), gripper 2];
  observation.images.* both cameras, 84 px, PNG;
  timestamp            t / 20.

Guards, each of which stops the build:
  identity  as rerender_demos: frames and state rendered from a state are those our env's step path
            returned in it;
  layout    the demos' joints are ours, in order, but for the gripper's rename, and the states are
            1 + nq + nv wide;
  pairing   states, actions and dones are one per step;
  placement each demo's first state has both cubes on the table within D0's +-0.08 m;
  replay    the state our env computes from each demo state itself is the demo's own recorded
            observation (MimicGen's are the states', with no substep lag: the stepped one differs from
            it by exactly the lag) to REPLAY_POS_TOL / REPLAY_QUAT_TOL / REPLAY_GRIP_TOL, and the
            stepped one differs from it only by the lag (rerender_demos.stepped_change);
  columns   after writing, every column is the one built, and the action statistics are the actions'.
The output is `local/mimicgen-stack-d0-rerendered` in the LeRobot home, its provenance in
`meta/rerender.json`. Nothing is uploaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

from tools.rerender_demos import (EEF_POS, EEF_QUAT, GRIPPER, IMAGE_FEATURE, GuardFailed, _commit, check_identity,
                                  guard, make_env, render_device, render_states, stepped_change, visible_gpus)

TASK = "Stack"
REPO = "amandlek/mimicgen_datasets"
REVISION = "33016f8a62c02334f929f2913af8fdd2a8a129e1"
FILE = "core/stack_d0.hdf5"
SHA256 = "3b687047"  # the head of the file's sha256, as Hugging Face's LFS hash gives it (checked in full below)
OUT = "local/mimicgen-stack-d0-rerendered"
D0_XY = 0.08
RENAMED = ("gripper0_finger_", "gripper0_right_finger_")  # robosuite 1.4.1 -> 1.5.1
# The first 20 demos replay to 0 m, 1.7e-7 and 0 rad (--check-only, 2026-10-29): float32's rounding, as
# rerender_demos.PROPRIO_TOL for Square's.
REPLAY_POS_TOL = 1e-6   # m
REPLAY_QUAT_TOL = 1e-6  # quaternion distance, up to sign
REPLAY_GRIP_TOL = 1e-6  # rad
STATE_NAMES = ["robot0_eef_pos_0", "robot0_eef_pos_1", "robot0_eef_pos_2", "robot0_eef_quat_0", "robot0_eef_quat_1",
               "robot0_eef_quat_2", "robot0_eef_quat_3", "robot0_gripper_qpos_0", "robot0_gripper_qpos_1"]


def source_file() -> Path:
    """The hdf5 from the Hugging Face cache, checked against its LFS sha256."""
    from huggingface_hub import HfApi, hf_hub_download

    path = Path(hf_hub_download(REPO, FILE, repo_type="dataset", revision=REVISION))
    info = HfApi().get_paths_info(REPO, [FILE], repo_type="dataset", revision=REVISION)[0]
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    digest = h.hexdigest()
    guard(digest == info.lfs.sha256 and digest.startswith(SHA256), f"{path}: sha256 {digest}, LFS {info.lfs.sha256}")
    return path


def demo_keys(f) -> list[str]:
    return sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))


def check_layout(w, xml: str, width: int) -> dict:
    """The demos' joints are ours, in order, but for the gripper's rename; the states are 1 + nq + nv wide."""
    import mujoco

    m = w.env.sim.model._model
    ours = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
    theirs = [n.replace(*RENAMED) for n in re.findall(r'<joint[^>]*?name="([^"]+)"', xml)]
    guard(theirs == ours, f"layout: their joints {theirs} are not ours {ours}")
    guard(width == 1 + m.nq + m.nv, f"layout: states are {width} wide, ours 1 + {m.nq} + {m.nv}")
    return {"joints": ours, "renamed": "->".join(RENAMED), "nq": m.nq, "nv": m.nv}


def check_placement(w, state: np.ndarray) -> float:
    """Both cubes rest on the table within D0's bounds at the episode's first state; the largest |x| or |y|."""
    from tools.rerender_demos import set_state

    set_state(w, state)
    env = w.env
    middle = np.asarray(env.table_offset[:2])
    worst = max(float(np.abs(env.sim.data.body_xpos[b][:2] - middle).max()) for b in (env.cubeA_body_id, env.cubeB_body_id))
    guard(worst <= D0_XY + 1e-6, f"placement: a cube starts {worst:.4f} m from the table's middle")
    return worst


def replay_gap(ours: np.ndarray, recorded: np.ndarray) -> tuple[float, float, float]:
    """Our state against the demo's recorded one: the largest position gap (m), quaternion gap (up to sign)
    and gripper gap (rad)."""
    guard(ours.shape == recorded.shape, f"replay: shapes {ours.shape} and {recorded.shape}")
    pos = float(np.linalg.norm(ours[:, EEF_POS] - recorded[:, EEF_POS], axis=1).max())
    q, r = ours[:, EEF_QUAT], recorded[:, EEF_QUAT]
    quat = float(np.minimum(np.linalg.norm(q - r, axis=1), np.linalg.norm(q + r, axis=1)).max())
    grip = float(np.abs(ours[:, GRIPPER] - recorded[:, GRIPPER]).max())
    return pos, quat, grip


def recorded_state(g) -> np.ndarray:
    o = g["obs"]
    return np.concatenate([o["robot0_eef_pos"][()], o["robot0_eef_quat"][()], o["robot0_gripper_qpos"][()]],
                          axis=1).astype(np.float32)


def render_all(gpu: int, n_episodes: int | None, on_episode=None, *, enforce_replay: bool = True) -> dict:
    """Run every guard and render every episode; on_episode(i, frames, state, actions, dones) receives each."""
    import h5py

    path = source_file()
    w = make_env(TASK, gpu)
    report = {"task": TASK, "identity": check_identity(TASK, gpu, w)}
    with h5py.File(path, "r") as f:
        keys = demo_keys(f)
        first = f[f"data/{keys[0]}"]
        report["layout"] = check_layout(w, first.attrs["model_file"], first["states"].shape[1])
        todo = keys if n_episodes is None else keys[:n_episodes]
        worst_xy, pos_gap, quat_gap, grip_gap, lag_pos, lag_quat, frames, done_differs = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0
        for i, k in enumerate(todo):
            g = f[f"data/{k}"]
            states, actions, dones = g["states"][()], g["actions"][()].astype(np.float32), g["dones"][()].astype(bool)
            guard(len(states) == len(actions) == len(dones), f"pairing: {k} has {len(states)} states, "
                  f"{len(actions)} actions, {len(dones)} dones")
            worst_xy = max(worst_xy, check_placement(w, states[0]))
            ep_frames, stepped, exact = render_states(w, states)
            p, q = stepped_change(stepped, exact)
            lag_pos, lag_quat = max(lag_pos, p), max(lag_quat, q)
            p, q, gr = replay_gap(exact, recorded_state(g))
            pos_gap, quat_gap, grip_gap = max(pos_gap, p), max(quat_gap, q), max(grip_gap, gr)
            if enforce_replay:
                guard(p <= REPLAY_POS_TOL and q <= REPLAY_QUAT_TOL and gr <= REPLAY_GRIP_TOL,
                      f"replay: {k}'s state is {p:.2e} m, {q:.2e} and {gr:.2e} rad from its recorded one "
                      f"(limits {REPLAY_POS_TOL}, {REPLAY_QUAT_TOL}, {REPLAY_GRIP_TOL})")
            done_differs += int(not np.array_equal(dones, g["rewards"][()] == 1))
            frames += len(actions)
            if on_episode is not None:
                on_episode(i, ep_frames, stepped, actions, dones)
    w.close()
    report["source"] = {"repo": REPO, "revision": REVISION, "file": FILE, "path": str(path)}
    report["pairing"] = {"episodes": len(todo), "frames": frames}
    report["placement"] = {"max_abs_xy_m": worst_xy, "limit_m": D0_XY}
    report["replay"] = {"from_state_vs_recorded": {"eef_pos_max_m": pos_gap, "eef_quat_max": quat_gap,
                                                   "gripper_max_rad": grip_gap},
                        "limits": {"pos_m": REPLAY_POS_TOL, "quat": REPLAY_QUAT_TOL, "gripper_rad": REPLAY_GRIP_TOL},
                        "enforced": enforce_replay}
    report["stepped_state"] = {"eef_pos_max_m": lag_pos, "eef_quat_max": lag_quat}
    report["dones_differ_from_rewards"] = done_differs
    return report


def build(gpu: int, root: Path | None, n_episodes: int | None) -> dict:
    import mujoco
    import robosuite
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    features = {
        "action": {"dtype": "float32", "shape": (7,), "names": [f"action_{j}" for j in range(7)]},
        "next.done": {"dtype": "bool", "shape": (1,), "names": ["done"]},
        "observation.state": {"dtype": "float32", "shape": (9,), "names": STATE_NAMES},
        "observation.images.agentview": dict(IMAGE_FEATURE),
        "observation.images.robot0_eye_in_hand": dict(IMAGE_FEATURE),
    }
    out = LeRobotDataset.create(repo_id=OUT, fps=20, features=features, root=root, robot_type="mimicgen",
                                use_videos=False, image_writer_threads=4)
    written = []

    def write(i, frames, state, actions, dones):
        assert out.meta.total_episodes == i, (out.meta.total_episodes, i)
        for t in range(len(actions)):
            frame = {"action": actions[t], "next.done": dones[t:t + 1], "observation.state": state[t]}
            frame.update({c: frames[c][t] for c in frames})
            out.add_frame(frame, task=TASK, timestamp=t / 20.0)
        out.save_episode()
        written.append((actions, dones, state))

    t0 = time.time()
    report = render_all(gpu, n_episodes, on_episode=write)
    out.stop_image_writer()
    report["columns"] = check_columns(out.root, written)
    report.update(output=OUT, root=str(out.root), seconds=round(time.time() - t0),
                  frames_rendered="each episode's first from its state, as reset() observes; every later one as "
                                  "step() observes, body positions one 2 ms physics substep behind the state",
                  versions={"robosuite": robosuite.__version__, "mujoco": mujoco.__version__}, commit=_commit())
    (out.root / "meta" / "rerender.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def check_columns(root: Path, written: list) -> dict:
    """Every column is the one built, episode by episode, and the action statistics are the actions'."""
    import pyarrow.parquet as pq
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(OUT, root=root)
    guard(ds.meta.total_episodes == len(written), f"columns: {ds.meta.total_episodes} episodes, {len(written)} built")
    for ep, (actions, dones, state) in enumerate(written):
        table = pq.read_table(ds.root / ds.meta.get_data_file_path(ep_index=ep))
        got = {c: np.asarray(table.column(c).to_pylist()) for c in ("action", "next.done", "observation.state")}
        guard(np.array_equal(got["action"].astype(np.float32), actions), f"columns: episode {ep}'s actions differ")
        guard(np.array_equal(got["next.done"].reshape(-1).astype(bool), dones), f"columns: episode {ep}'s dones differ")
        guard(np.array_equal(got["observation.state"].astype(np.float32), state), f"columns: episode {ep}'s state differs")
    every = np.concatenate([a for a, _, _ in written])
    stats = ds.meta.stats["action"]
    gap = max(float(np.abs(np.asarray(stats["min"]) - every.min(0)).max()),
              float(np.abs(np.asarray(stats["max"]) - every.max(0)).max()))
    guard(gap <= 1e-6, f"columns: the action statistics are {gap:.2e} from the actions'")
    return {"episodes": len(written), "columns_equal": True, "action_stats_max_abs_diff": gap}


def verify(gpu: int, n_episodes: int, root: Path | None = None) -> dict:
    """A built dataset's stored frames, decoded as the trainer decodes them, and its state, against fresh renders."""
    import h5py
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(OUT, root=root)
    w = make_env(TASK, gpu)
    eps = np.unique(np.linspace(0, ds.meta.total_episodes - 1, n_episodes).astype(int))
    with h5py.File(source_file(), "r") as f:
        keys = demo_keys(f)
        for ep in eps:
            frames, stepped, _ = render_states(w, f[f"data/{keys[ep]}"]["states"][()])
            start = int(ds.episode_data_index["from"][ep])
            for t in range(len(stepped)):
                item = ds[start + t]
                for c in frames:
                    stored = (item[c].numpy() * 255).astype(np.uint8).transpose(1, 2, 0)  # the trainer's to_uint8
                    guard(np.array_equal(stored, frames[c][t]), f"verify: episode {ep} frame {t} {c} differs")
                guard(np.array_equal(item["observation.state"].numpy(), stepped[t]), f"verify: episode {ep} frame {t}'s state differs")
    w.close()
    return {"task": TASK, "episodes": [int(e) for e in eps], "stored_equals_fresh_render": True}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", type=int, default=render_device(), help="EGL render device, as rerender_demos")
    ap.add_argument("--episodes", type=int, default=None, help="only the first N (check-only/build), or N spread (verify)")
    ap.add_argument("--root", type=Path, default=None, help="the dataset here instead of the LeRobot home (tests)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true", help="run the guards and render, write nothing; "
                      "the replay gap is measured, not enforced")
    mode.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if visible_gpus() and args.gpu not in visible_gpus():
        ap.error(f"--gpu {args.gpu} is not in CUDA_VISIBLE_DEVICES")
    try:
        if args.verify:
            report = verify(args.gpu, args.episodes or 5, args.root)
        elif args.check_only:
            report = render_all(args.gpu, args.episodes, enforce_replay=False)
        else:
            report = build(args.gpu, args.root, args.episodes)
    except GuardFailed as exc:
        print(f"GUARD FAILED: {exc}")
        return 3
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
