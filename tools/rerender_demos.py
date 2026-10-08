"""Re-render the robomimic MH demos' camera frames in our env, losslessly, as a LeRobot dataset.

    python -m tools.rerender_demos --task Can [--gpu 1] [--json]               # build
    python -m tools.rerender_demos --task Can --check-only [--episodes 3]    # guards only, no writing
    python -m tools.rerender_demos --task Can --verify [--episodes 5]        # a built dataset against fresh renders
    python -m tools.rerender_demos --task Can --codec-change [--episodes 10] # what the source's video codec changed

Why. The LeRobot datasets every run trains on (`ankile/robomimic-mh-{can,square}-image`) store their
frames as lossy AV1 video and the env's frames are not coded, so a critic can tell every demo frame
from every online frame (EXPERIMENTS.md, 2026-10-08: CODE, `encode_env_frames_like_demos`, and the
NOTE on `gkck6eqp`). Here the demo frames are rendered again by the code that renders the online
ones, and stored as PNG, so the two sides cannot differ by how they were made.

How. robomimic's v1.5 files (`demo_v15.hdf5`, made with robosuite 1.5.0) hold every demo's simulator
state at every step, and the LeRobot frames were rendered from those states. Each state is set into
our env's own scene (`RobosuiteGymWrapper`, as training builds it) and the observation is read
through `_process_obs`, which every online frame passes through. Not into the demos' saved XML:
reloading a scene from XML moves a few front-camera pixels by 1-4 levels (MuJoCo writes numbers at
limited precision; camera quaternions shift by 3e-7), and our scene is the one online frames come
from. The two scenes have the same 13 joints in the same order; ours adds two mocap target bodies
(`left_eef_target`, `right_eef_target`) parked at z = -1 m, which no camera sees.

One physics substep late, as the env observes. robosuite's step() runs 25 MuJoCo substeps of 2 ms
and observes after the last one without recomputing where the bodies are. A MuJoCo step computes
the body positions from the joint angles and then integrates, so after step() the cameras and the
end-effector pose show the joint angles of one substep earlier than the state step() leaves; the
gripper's joint readings, read from the joint angles, are current. reset() recomputes the positions
before it observes. So each episode's first frame is rendered from its state, and every later frame
as step() leaves the state: body positions from the previous substep's joint angles, which MuJoCo's
Euler integrator makes the state's own moved back by dt times its velocity (`as_stepped`). Against
the trainer's own step(), over full episodes of random actions, that gives bit-identical frames
and an equal observation.state at every step; rendered from the state itself instead, about 150
front-camera and 1,300-3,600 wrist-camera pixels differ, and the end-effector pose sits 0.6-0.7 mm
away on average (EXPERIMENTS.md, 2026-10-08 CODE, the re-rendered demos).

Guards, each of which stops the build:
  identity    frames and observation.state rendered from a state are those the trainer's step path
              returned in it, bit for bit: an env stepped only through the wrapper's reset() and
              step(), at its resets and after each of its random actions;
  layout      the demos' joints are ours, in order and at the same qpos/qvel addresses; every body
              of theirs is ours and lands at the same place from the same state; ours adds only
              mocap bodies;
  pairing     LeRobot episode i is robomimic demo_i: same length, and actions exactly equal;
  replay      the observation.state our env computes from each state itself equals the source
              dataset's (to PROPRIO_TOL), and the one stepped from it differs only in the
              end-effector pose, by at most LAG_POS_TOL and LAG_QUAT_TOL;
  columns     after writing, every column but the images and observation.state equals the source
              dataset's, observation.state is the stepped one rendered, and the action statistics
              equal the source's.

Everything but the two image columns and observation.state is copied from the source LeRobot
dataset. The output goes to the LeRobot home as `local/robomimic-mh-{can,square}-rerendered`, with
its provenance in `meta/rerender.json`. Nothing is uploaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROBOMIMIC_REPO = "robomimic/robomimic_datasets"
ROBOMIMIC_REVISION = "74fa018461f479cd9fd15b924a16103012096203"
SOURCES = {
    # task: (source LeRobot dataset, robomimic file, its sha256, output dataset)
    "Can": ("ankile/robomimic-mh-can-image", "v1.5/can/mh/demo_v15.hdf5",
            "6f5a0e96fc375468ae19ccac86a6dee3b4d3862ef0fe37ca2d02e93d9046fe6b",
            "local/robomimic-mh-can-rerendered"),
    "Square": ("ankile/robomimic-mh-square-image", "v1.5/square/mh/demo_v15.hdf5",
               "07e232178d4b526facaf6970ffe23c5d2c407f25fdbd1ef8ba0204d312d05311",
               "local/robomimic-mh-square-rerendered"),
}
PROPRIO_TOL = 1e-6   # every demo replays to 1.2e-7, float32's rounding
BODY_TOL = 1e-5      # body positions under the two scenes: XML precision is about 1e-7
# observation.state of the one-arm tasks: robot0_eef_pos, robot0_eef_quat (xyzw), robot0_gripper_qpos
EEF_POS, EEF_QUAT, GRIPPER = slice(0, 3), slice(3, 7), slice(7, 9)
LAG_POS_TOL = 5e-3   # m in one 2 ms substep; random actions reach 2.0e-3
LAG_QUAT_TOL = 5e-2  # quaternion distance, up to sign, in one substep
IMAGE_FEATURE = {"dtype": "image", "shape": (84, 84, 3), "names": ["height", "width", "channel"]}


class GuardFailed(RuntimeError):
    pass


def guard(ok: bool, what: str) -> None:
    if not ok:
        raise GuardFailed(what)


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

def states_file(task: str) -> Path:
    """The robomimic file, from the Hugging Face cache, checked against its recorded sha256."""
    from huggingface_hub import hf_hub_download

    _, fname, sha, _ = SOURCES[task]
    path = Path(hf_hub_download(ROBOMIMIC_REPO, fname, repo_type="dataset", revision=ROBOMIMIC_REVISION))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    guard(digest == sha, f"{path}: sha256 {digest}, recorded {sha}")
    return path


def source_episodes(repo: str) -> tuple[object, list[dict]]:
    """The source dataset and, per episode, its non-image columns as numpy arrays (no video decoding)."""
    import pyarrow.parquet as pq
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo)
    episodes = []
    for ep in range(ds.meta.total_episodes):
        table = pq.read_table(ds.root / ds.meta.get_data_file_path(ep_index=ep))
        cols = {}
        for name in table.column_names:
            values = table.column(name).to_pylist()
            cols[name] = np.asarray(values, dtype=_np_dtype(ds.meta.features[name]["dtype"]))
        cols["tasks"] = ds.meta.episodes[ep]["tasks"]
        episodes.append(cols)
    return ds, episodes


def _np_dtype(dtype: str):
    return {"float32": np.float32, "float64": np.float64, "int64": np.int64, "bool": bool}[dtype]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def make_env(task: str, gpu: int):
    os.environ["MUJOCO_GL"] = "egl"  # as the trainer sets it, before robosuite is imported
    from resfit.dexmg.environments.dexmg import RobosuiteGymWrapper

    return RobosuiteGymWrapper(env_name=task, render_gpu_device_id=gpu)


def cameras(w) -> list[str]:
    return ["observation.images." + k.replace("_image", "") for k in w.expected_image_keys]


def as_frames(w, processed: dict) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """_process_obs's output as (H, W, 3) uint8 frames and observation.state.

    _process_obs divides robosuite's uint8 by 255 in float32; rounding x * 255 recovers it exactly.
    """
    frames = {c: np.round(processed[c] * 255.0).astype(np.uint8).transpose(1, 2, 0) for c in cameras(w)}
    return frames, processed["observation.state"]


def observe(w) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """The env's frames and observation.state from sim.data as it stands, through _process_obs."""
    return as_frames(w, w._process_obs(w.env._get_observations(force_update=True)))


def set_state(w, state: np.ndarray) -> None:
    """sim.data as reset() leaves it: body positions computed from the state's joint angles."""
    w.env.sim.set_state_from_flattened(state)
    w.env.sim.forward()


def as_stepped(m, d) -> None:
    """Leave MjData `d` as mj_step leaves its state: body positions from the joint angles one
    substep earlier, joint angles and velocities as they are.

    mj_step computes the positions, then moves qpos by dt times the new qvel (Euler), so the
    previous substep's qpos is the current one moved back by dt.
    """
    import mujoco

    assert m.opt.integrator == mujoco.mjtIntegrator.mjINT_EULER, m.opt.integrator
    qpos = d.qpos.copy()
    previous = qpos.copy()
    mujoco.mj_integratePos(m, previous, d.qvel, -m.opt.timestep)
    d.qpos[:] = previous
    mujoco.mj_forward(m, d)
    d.qpos[:] = qpos


def set_state_as_stepped(w, state: np.ndarray) -> None:
    """sim.data as robosuite's step() leaves it in `state`."""
    w.env.sim.set_state_from_flattened(state)
    as_stepped(w.env.sim.model._model, w.env.sim.data._data)


def render_states(w, states: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """For an episode's states, in a freshly reset scene: the frames and observation.state the env
    hands the agent in each (the first as reset() leaves it, every later one as step() does), and
    observation.state computed from each state itself (the replay guard's)."""
    w.env.reset()
    frames = {c: [] for c in cameras(w)}
    stepped, exact = [], []
    for t, s in enumerate(states):
        set_state(w, s)
        f, p = observe(w)
        exact.append(p)
        if t > 0:
            set_state_as_stepped(w, s)
            f, p = observe(w)
        for c in frames:
            frames[c].append(f[c])
        stepped.append(p)
    return {c: np.stack(v) for c, v in frames.items()}, np.stack(stepped), np.stack(exact)


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def check_identity(task: str, gpu: int, w, trials: int = 3, steps: int = 30, seed: int = 0) -> dict:
    """Frames and observation.state rendered from a state equal those the trainer's step path returned
    in it, bit for bit, at each reset and after each random action.

    The online env is touched only through the wrapper's reset() and step(), so robosuite's
    observable timers, which pick the substep an observation is taken at, run as in training.
    """
    online = make_env(task, gpu)
    rng = np.random.default_rng(seed)
    resets = steps_checked = 0
    try:
        for _ in range(trials):
            processed, _ = online.reset()
            for t in range(steps + 1):
                if t > 0:
                    processed, _, terminated, truncated, _ = online.step(rng.uniform(-1, 1, online.env.action_dim))
                want, want_p = as_frames(online, processed)
                state = online.env.sim.get_state().flatten()
                w.env.reset()  # another placement first: the state alone sets what is rendered
                (set_state if t == 0 else set_state_as_stepped)(w, state)
                got, got_p = observe(w)
                where = "at a reset" if t == 0 else f"after step {t}"
                for c in want:
                    n = int((want[c] != got[c]).any(-1).sum())
                    guard(n == 0, f"identity: {c} differs at {n} pixels {where}")
                guard(np.array_equal(want_p, got_p), f"identity: observation.state differs {where}")
                resets, steps_checked = resets + (t == 0), steps_checked + (t > 0)
                if t > 0 and (terminated or truncated):
                    break
    finally:
        online.close()
    return {"resets": resets, "steps": steps_checked, "bit_identical": True,
            "online_path": "RobosuiteGymWrapper.reset/step"}


def check_layout(w, xml: str, sample_states: list[np.ndarray]) -> dict:
    """The demos' scene and ours: same joints at the same addresses, same bodies in the same places."""
    import mujoco

    ours = w.env.sim.model._model
    demo = mujoco.MjModel.from_xml_string(w.env.edit_model_xml(xml))

    def names(m, kind, n):
        return [mujoco.mj_id2name(m, kind, i) for i in range(n)]

    J = mujoco.mjtObj.mjOBJ_JOINT
    guard(names(ours, J, ours.njnt) == names(demo, J, demo.njnt), "layout: joints differ")
    guard((ours.nq, ours.nv) == (demo.nq, demo.nv), "layout: nq/nv differ")
    guard(np.array_equal(ours.jnt_qposadr, demo.jnt_qposadr) and np.array_equal(ours.jnt_dofadr, demo.jnt_dofadr),
          "layout: joint addresses differ")
    B = mujoco.mjtObj.mjOBJ_BODY
    ours_b, demo_b = names(ours, B, ours.nbody), names(demo, B, demo.nbody)
    missing = [b for b in demo_b if b not in ours_b]
    extra = [b for b in ours_b if b not in demo_b]
    guard(not missing, f"layout: demo bodies missing from ours: {missing}")
    not_mocap = [b for b in extra if ours.body_mocapid[ours_b.index(b)] < 0]
    guard(not not_mocap, f"layout: our extra bodies are not mocap bodies: {not_mocap}")

    data = mujoco.MjData(demo)
    worst = 0.0
    for s in sample_states:
        set_state(w, s)
        data.time, data.qpos[:], data.qvel[:] = s[0], s[1:1 + demo.nq], s[1 + demo.nq:1 + demo.nq + demo.nv]
        mujoco.mj_forward(demo, data)
        ours_x = w.env.sim.data._data.xpos
        for i, b in enumerate(demo_b):
            worst = max(worst, float(np.abs(ours_x[ours_b.index(b)] - data.xpos[i]).max()))
    guard(worst < BODY_TOL, f"layout: a body lands {worst:.2e} m apart under the two scenes (limit {BODY_TOL})")
    return {"joints": ours.njnt, "extra_mocap_bodies": extra, "max_body_distance_m": worst}


def stepped_change(stepped: np.ndarray, exact: np.ndarray) -> tuple[float, float]:
    """How far the stepped observation.state is from the one computed from the state itself: the
    largest end-effector position change (m) and quaternion change (up to sign). Guards that nothing
    else changed."""
    guard(stepped.shape == exact.shape and stepped.shape[1] == GRIPPER.stop,
          f"replay: observation.state has shape {stepped.shape}, not [T, {GRIPPER.stop}]")
    guard(np.array_equal(stepped[:, GRIPPER], exact[:, GRIPPER]), "replay: the stepped gripper reading moved")
    pos = float(np.linalg.norm(stepped[:, EEF_POS] - exact[:, EEF_POS], axis=1).max())
    q, r = stepped[:, EEF_QUAT], exact[:, EEF_QUAT]
    quat = float(np.minimum(np.linalg.norm(q - r, axis=1), np.linalg.norm(q + r, axis=1)).max())
    guard(pos <= LAG_POS_TOL, f"replay: the stepped end-effector moved {pos:.2e} m (limit {LAG_POS_TOL})")
    guard(quat <= LAG_QUAT_TOL, f"replay: the stepped end-effector turned {quat:.2e} (limit {LAG_QUAT_TOL})")
    return pos, quat


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def demo_keys(f) -> list[str]:
    return sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))


def render_all(task: str, gpu: int, n_episodes: int | None, on_episode=None) -> dict:
    """Run every guard and render every episode; on_episode(i, frames, state, columns) receives each."""
    import h5py

    repo = SOURCES[task][0]
    w = make_env(task, gpu)
    report = {"task": task, "identity": check_identity(task, gpu, w)}
    ds, episodes = source_episodes(repo)
    with h5py.File(states_file(task), "r") as f:
        keys = demo_keys(f)
        guard(len(keys) == len(episodes), f"pairing: {len(keys)} demos, {len(episodes)} LeRobot episodes")
        first = f[f"data/{keys[0]}"]
        xmls = {f[f"data/{k}"].attrs["model_file"] for k in keys}
        guard(len(xmls) == 1, f"layout: the demos have {len(xmls)} different scenes")
        report["layout"] = check_layout(w, first.attrs["model_file"], [first["states"][0], first["states"][-1]])
        worst, frames_done, pos_lag, quat_lag, pos_sum = 0.0, 0, 0.0, 0.0, 0.0
        todo = range(len(keys) if n_episodes is None else n_episodes)
        for i in todo:
            g = f[f"data/{keys[i]}"]
            states, actions = g["states"][()], g["actions"][()]
            cols = episodes[i]
            guard(len(actions) == len(cols["action"]), f"pairing: {keys[i]} has {len(actions)} steps, episode {i} {len(cols['action'])}")
            guard(np.array_equal(actions.astype(np.float32), cols["action"]), f"pairing: {keys[i]}'s actions are not episode {i}'s")
            frames, stepped, exact = render_states(w, states)
            d = float(np.abs(exact - cols["observation.state"]).max())
            guard(d <= PROPRIO_TOL, f"replay: {keys[i]}'s proprioception is {d:.2e} from episode {i}'s (limit {PROPRIO_TOL})")
            pos, quat = stepped_change(stepped, exact)
            pos_sum += float(np.linalg.norm(stepped[:, EEF_POS] - exact[:, EEF_POS], axis=1).sum())
            worst, frames_done = max(worst, d), frames_done + len(actions)
            pos_lag, quat_lag = max(pos_lag, pos), max(quat_lag, quat)
            if on_episode is not None:
                on_episode(i, frames, stepped, cols)
    report["pairing"] = {"episodes": len(todo), "frames": frames_done, "actions_exact": True}
    report["replay"] = {"max_proprio_diff": worst, "limit": PROPRIO_TOL}
    report["stepped_state"] = {"eef_pos_mean_m": pos_sum / frames_done, "eef_pos_max_m": pos_lag,
                               "eef_quat_max": quat_lag, "gripper_unchanged": True}
    w.close()
    return report


def build(task: str, gpu: int, root: Path | None, n_episodes: int | None) -> dict:
    import mujoco
    import robosuite
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    repo, fname, sha, out_repo = SOURCES[task]
    src = LeRobotDataset(repo)
    features = {k: {**ft, "shape": tuple(ft["shape"])} for k, ft in src.meta.features.items()
                if k in ("action", "next.done", "observation.state")}
    for cam in src.meta.camera_keys:
        features[cam] = dict(IMAGE_FEATURE)
    out = LeRobotDataset.create(repo_id=out_repo, fps=src.fps, features=features, root=root,
                                robot_type=src.meta.robot_type, use_videos=False, image_writer_threads=4)
    written = []

    def write(i, frames, state, cols):
        assert out.meta.total_episodes == i, (out.meta.total_episodes, i)
        task_name = cols["tasks"][0]
        for t in range(len(cols["action"])):
            frame = {"action": cols["action"][t], "next.done": cols["next.done"][t].reshape(1),
                     "observation.state": state[t]}
            frame.update({c: frames[c][t] for c in frames})
            out.add_frame(frame, task=task_name, timestamp=float(cols["timestamp"][t]))
        out.save_episode()
        written.append(state)

    t0 = time.time()
    report = render_all(task, gpu, n_episodes, on_episode=write)
    out.stop_image_writer()
    report["columns"] = check_columns(task, out.root, n_episodes, written)
    report.update(source=repo, output=out_repo, root=str(out.root), seconds=round(time.time() - t0),
                  frames="each episode's first from its state, as reset() observes; every later one as "
                         "step() observes, body positions one 2 ms physics substep behind the state",
                  robomimic={"repo": ROBOMIMIC_REPO, "revision": ROBOMIMIC_REVISION, "file": fname, "sha256": sha},
                  versions={"robosuite": robosuite.__version__, "mujoco": mujoco.__version__},
                  commit=_commit())
    (out.root / "meta" / "rerender.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def check_columns(task: str, root: Path, n_episodes: int | None, states: list[np.ndarray]) -> dict:
    """Every column but the images and observation.state equals the source's, observation.state is
    the one rendered, and the action statistics equal the source's."""
    import pyarrow.parquet as pq
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    src_ds, src_eps = source_episodes(SOURCES[task][0])
    new = LeRobotDataset(SOURCES[task][3], root=root)
    n = len(src_eps) if n_episodes is None else n_episodes
    guard(new.meta.total_episodes == n == len(states), f"columns: {new.meta.total_episodes} episodes written, {n} expected")
    for ep in range(n):
        table = pq.read_table(new.root / new.meta.get_data_file_path(ep_index=ep))
        for name, want in src_eps[ep].items():
            if name == "tasks":
                guard(new.meta.episodes[ep]["tasks"] == want, f"columns: episode {ep}'s tasks differ")
                continue
            got = np.asarray(table.column(name).to_pylist(), dtype=want.dtype)
            if name == "observation.state":
                guard(np.array_equal(got, states[ep]), f"columns: episode {ep}'s observation.state is not the one rendered")
                continue
            guard(np.array_equal(got, want), f"columns: episode {ep}'s {name} differs from the source's")
    stats = {}
    if n_episodes is None:
        for key in ("action", "observation.state"):
            for stat in ("mean", "std", "min", "max"):
                a, b = np.asarray(new.meta.stats[key][stat]), np.asarray(src_ds.meta.stats[key][stat])
                stats[f"{key}.{stat}"] = float(np.abs(a - b).max())
        action = max(v for k, v in stats.items() if k.startswith("action."))
        guard(action <= 1e-6, f"columns: action statistics differ from the source's: {stats}")
    return {"episodes": n, "copied_columns_equal": True, "state_is_rendered": True, "stats_max_abs_diff": stats}


def verify(task: str, gpu: int, n_episodes: int, root: Path | None = None) -> dict:
    """A built dataset's stored frames, decoded as the trainer decodes them, and its observation.state,
    against fresh renders."""
    import h5py
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(SOURCES[task][3], root=root)
    w = make_env(task, gpu)
    eps = np.unique(np.linspace(0, ds.meta.total_episodes - 1, n_episodes).astype(int))
    with h5py.File(states_file(task), "r") as f:
        keys = demo_keys(f)
        for ep in eps:
            frames, stepped, _ = render_states(w, f[f"data/{keys[ep]}"]["states"][()])
            start = int(ds.episode_data_index["from"][ep])
            for t in range(len(stepped)):
                item = ds[start + t]
                for c in frames:
                    stored = (item[c].numpy() * 255).astype(np.uint8).transpose(1, 2, 0)  # the trainer's to_uint8
                    guard(np.array_equal(stored, frames[c][t]), f"verify: episode {ep} frame {t} {c} differs from a fresh render")
                guard(np.array_equal(item["observation.state"].numpy(), stepped[t]),
                      f"verify: episode {ep} frame {t}'s observation.state differs from a fresh render")
    w.close()
    return {"task": task, "episodes": [int(e) for e in eps], "stored_equals_fresh_render": True}


def codec_change(task: str, gpu: int, n_episodes: int) -> dict:
    """What the source dataset's video codec changed: its frames, decoded as the trainer decodes them,
    against frames rendered from the same states as the source's were (from the state itself),
    keyframes (even frames) and P-frames (odd) apart."""
    import h5py
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    src = LeRobotDataset(SOURCES[task][0])
    w = make_env(task, gpu)
    eps = np.linspace(0, src.meta.total_episodes - 1, n_episodes).astype(int)
    diffs = {c: {"keyframes": [], "p_frames": []} for c in cameras(w)}
    with h5py.File(states_file(task), "r") as f:
        keys = demo_keys(f)
        for ep in eps:
            w.env.reset()
            start = int(src.episode_data_index["from"][ep])
            for t, s in enumerate(f[f"data/{keys[ep]}"]["states"][()]):
                set_state(w, s)
                fresh, _ = observe(w)
                item = src[start + t]
                for c in fresh:
                    stored = (item[c].numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
                    d = np.abs(stored.astype(int) - fresh[c].astype(int))
                    diffs[c]["keyframes" if t % 2 == 0 else "p_frames"].append((d.mean(), d.any(-1).mean()))
    w.close()
    out = {"task": task, "episodes": [int(e) for e in eps]}
    for c, by_kind in diffs.items():
        out[c] = {kind: {"mean_abs_levels": float(np.mean([m for m, _ in v])),
                         "pixels_changed": float(np.mean([p for _, p in v])), "frames": len(v)}
                  for kind, v in by_kind.items()}
    return out


def _commit() -> dict:
    repo = Path(__file__).resolve().parents[1]
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "tools/rerender_demos.py", "resfit/"],
                           cwd=repo, capture_output=True, text=True).stdout.split("\n")
    return {"head": head, "dirty": [d for d in dirty if d]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=sorted(SOURCES), required=True)
    ap.add_argument("--gpu", type=int, default=1, help="EGL render device")
    ap.add_argument("--episodes", type=int, default=None,
                    help="only the first N (check-only/build), or N spread (verify, codec-change)")
    ap.add_argument("--root", type=Path, default=None, help="the dataset here instead of the LeRobot home (tests)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true", help="run the guards and render, write nothing")
    mode.add_argument("--verify", action="store_true", help="check a built dataset against fresh renders")
    mode.add_argument("--codec-change", action="store_true", help="measure what the source's codec changed")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    try:
        if args.verify:
            report = verify(args.task, args.gpu, args.episodes or 5, args.root)
        elif args.codec_change:
            report = codec_change(args.task, args.gpu, args.episodes or 10)
        elif args.check_only:
            report = render_all(args.task, args.gpu, args.episodes)
        else:
            report = build(args.task, args.gpu, args.root, args.episodes)
    except GuardFailed as exc:
        print(json.dumps({"task": args.task, "guard_failed": str(exc)}) if args.json else f"GUARD FAILED: {exc}")
        return 3
    print(json.dumps(report) if args.json else json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
