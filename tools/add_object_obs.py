"""Add the task objects' poses, observation.object, to the re-rendered Square demos, as a new dataset.

    python -m tools.add_object_obs [--gpu 1] [--json]                   # build
    python -m tools.add_object_obs --check-only [--episodes 3]          # guards only, no writing

Why. O1 (EXPERIMENTS.md, the 2026-10-27 evening DECISION and O1's CODE entry) gives the residual the
nut's and the peg's poses, in the object-centric paper's shape (resfit/dexmg/environments/object_obs.py).
The env returns them online when the run's object_obs is on; the demo half of each batch needs them too.

How. A copy of `local/robomimic-mh-square-rerendered` (tools/rerender_demos.py) with one column added,
`local/robomimic-mh-square-rerendered-object`. Every file of the source is copied as it is, images
included, so the demo frames are the very bytes eee trained on (a new render could differ by the GPU's
residue). The new column is read from each demo's simulator states (robomimic's demo_v15.hdf5, the file
the source was rendered from), set into our env as the re-render set them, the first as reset() leaves
the state and every later one as step() leaves it (one physics substep behind: rerender_demos.py), and
observed through the wrapper's own _process_obs with object_obs on, which every online observation passes
through. No frame is written; robosuite renders the cameras on each observation, and they are dropped.

Guards, each of which stops the build:
  state     the re-observed observation.state equals the source's at every frame, exactly: the states
            are set as the source's were;
  online    an env touched only through the wrapper's reset() and step(), at its resets and after
            random actions, returns the observation.object that observing its state gives, exactly;
  quat      every quaternion, the nut's and the peg's, has unit length (to QUAT_TOL);
  start     each episode's first nut pose is a pure turn about z, inside robosuite's placement range
            (with START_MARGIN): the column is the nut's, in robosuite's convention;
  peg       the peg's pose is the same at every frame: it is fixed to the table;
  columns   after writing, every source file but the parquets, info.json and episodes_stats.jsonl is
            byte for byte the source's; every source column of every parquet equals the source's; the
            other features and every other feature's statistics are the source's; the new column is
            the one observed; and LeRobot loads it, with its statistics.

Its provenance goes to the new dataset's meta/object_obs.json; meta/rerender.json is the source's.
"""

from __future__ import annotations

import argparse
import filecmp
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from resfit.dexmg.environments.object_obs import OBJECT_DIM, OBJECT_KEY, OBJECT_STRIDE, OBJECTS
from tools import rerender_demos as rr
from tools.rerender_demos import guard

TASK = "Square"
SOURCE = rr.SOURCES[TASK][3]  # local/robomimic-mh-square-rerendered
OUTPUT = SOURCE + "-object"
LEROBOT_HOME = Path.home() / ".cache/huggingface/lerobot"
QUAT_TOL = 1e-5
# robosuite's placement of the square nut (nut_assembly.py, _load_model): x in [-0.115, -0.11] and y in
# [0.11, 0.225] about the table's centre, which is the world's origin in xy; any turn about z.
NUT_X, NUT_Y, START_MARGIN = (-0.115, -0.11), (0.11, 0.225), 0.01
TILT_TOL = 1e-4  # the first quaternion's x and y parts, which a turn about z leaves at 0
CHANGED = ("meta/info.json", "meta/episodes_stats.jsonl")
OBJECT_NAMES = [f"{obj}_{part}_{i}" for obj in OBJECTS for part, n in (("pos", 3), ("quat", 4)) for i in range(n)]


def observe_objects(w, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """observation.object and observation.state the env gives in each of an episode's states, in a
    freshly reset scene: the first as reset() leaves it, every later one as step() does."""
    w.env.reset()
    objects, state = [], []
    for t, s in enumerate(states):
        (rr.set_state if t == 0 else rr.set_state_as_stepped)(w, s)
        processed = w._process_obs(w.env._get_observations(force_update=True))
        objects.append(processed[OBJECT_KEY])
        state.append(processed["observation.state"])
    return np.stack(objects), np.stack(state)


def check_online(w, online, trials: int = 3, steps: int = 30, seed: int = 0) -> dict:
    """`online`, an env touched only through reset() and step(), returns the observation.object that
    observing its state in `w` gives, at each reset and after each random action, exactly. Both envs are
    in this process: the frames are not compared here, so their shared GL context does not matter."""
    rng = np.random.default_rng(seed)
    checked = 0
    for _ in range(trials):
        for t in range(steps + 1):
            if t == 0:
                processed, _ = online.reset()
                ended = False
            else:
                processed, _, terminated, truncated, _ = online.step(rng.uniform(-1, 1, online.env.action_dim))
                ended = terminated or truncated
            state = online.env.sim.get_state().flatten()
            w.env.reset()  # another placement first: the state alone sets what is observed
            (rr.set_state if t == 0 else rr.set_state_as_stepped)(w, state)
            got = w._process_obs(w.env._get_observations(force_update=True))[OBJECT_KEY]
            where = "at a reset" if t == 0 else f"after step {t}"
            guard(np.array_equal(processed[OBJECT_KEY], got),
                  f"online: observation.object differs {where}, by up to {np.abs(processed[OBJECT_KEY] - got).max():.2e}")
            checked += 1
            if ended:
                break
    return {"observations": checked, "exact": True, "online_path": "RobosuiteGymWrapper.reset/step"}


def check_episode(objects: np.ndarray, peg0: np.ndarray | None) -> np.ndarray:
    """The quat, start and peg guards on one episode's column; returns its peg pose."""
    for k, name in enumerate(OBJECTS):
        q = objects[:, k * OBJECT_STRIDE + 3 : (k + 1) * OBJECT_STRIDE]
        worst = float(np.abs(np.linalg.norm(q.astype(np.float64), axis=1) - 1).max())
        guard(worst <= QUAT_TOL, f"quat: a {name} quaternion is {worst:.1e} from unit length")
    x, y, _, qx, qy = objects[0, :5]
    guard(max(abs(qx), abs(qy)) <= TILT_TOL, f"start: the first nut pose is tilted (quaternion x, y = {qx}, {qy})")
    guard(NUT_X[0] - START_MARGIN <= x <= NUT_X[1] + START_MARGIN and NUT_Y[0] - START_MARGIN <= y <= NUT_Y[1] + START_MARGIN,
          f"start: the first nut position ({x:.4f}, {y:.4f}) is outside robosuite's placement range")
    peg = objects[:, OBJECT_STRIDE:]
    peg0 = peg[0] if peg0 is None else peg0
    guard(bool((peg == peg0).all()), "peg: the peg's pose moves")
    return peg0


def source_parquets(root: Path) -> list[Path]:
    return sorted((root / "data").glob("chunk-*/episode_*.parquet"))


def observe_all(gpu: int, n_episodes: int | None, src_root: Path) -> tuple[dict, list[np.ndarray]]:
    """Every guard but columns; the column for each episode."""
    import h5py
    import pyarrow.parquet as pq

    w = rr.make_env(TASK, gpu, object_obs=True)
    online = rr.make_env(TASK, gpu, object_obs=True)
    try:
        report = {"online": check_online(w, online)}
    finally:
        online.close()
    parquets = source_parquets(src_root)
    columns, peg0, first = [], None, []
    with h5py.File(rr.states_file(TASK), "r") as f:
        keys = rr.demo_keys(f)
        guard(len(keys) == len(parquets), f"pairing: {len(keys)} demos, {len(parquets)} source episodes")
        todo = range(len(keys) if n_episodes is None else n_episodes)
        for i in todo:
            table = pq.read_table(parquets[i], columns=["action", "observation.state"])
            g = f[f"data/{keys[i]}"]
            states, actions = g["states"][()], g["actions"][()]
            want_state = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32)
            want_action = np.asarray(table.column("action").to_pylist(), dtype=np.float32)
            guard(len(states) == len(want_state), f"pairing: {keys[i]} has {len(states)} states, episode {i} {len(want_state)}")
            guard(np.array_equal(actions.astype(np.float32), want_action), f"pairing: {keys[i]}'s actions are not episode {i}'s")
            objects, state = observe_objects(w, states)
            guard(np.array_equal(state, want_state), f"state: episode {i}'s observation.state is not the source's")
            peg0 = check_episode(objects, peg0)
            first.append(objects[0, :3])
            columns.append(objects)
    w.close()
    first = np.stack(first)
    report.update(
        episodes=len(columns), frames=int(sum(len(c) for c in columns)), state_exact=True,
        quat_tol=QUAT_TOL, peg_pose=peg0.tolist(),
        first_nut_xy={"x": [float(first[:, 0].min()), float(first[:, 0].max())],
                      "y": [float(first[:, 1].min()), float(first[:, 1].max())]})
    return report, columns


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def _after_state(d: dict, key: str, value) -> dict:
    """`d` with key: value inserted right after observation.state, the order the features take."""
    out = {}
    for k, v in d.items():
        out[k] = v
        if k == "observation.state":
            out[key] = value
    assert key in out, "no observation.state to follow"
    return out


def add_column(table, objects: np.ndarray):
    """The parquet table with OBJECT_KEY after observation.state, its Hugging Face features too."""
    import pyarrow as pa

    values = pa.array(objects.astype(np.float32).ravel(), type=pa.float32())
    column = pa.FixedSizeListArray.from_arrays(values, OBJECT_DIM)
    index = table.schema.get_field_index("observation.state") + 1
    new = table.add_column(index, pa.field(OBJECT_KEY, column.type), column)
    meta = dict(table.schema.metadata or {})
    hf = json.loads(meta[b"huggingface"])
    hf["info"]["features"] = _after_state(hf["info"]["features"], OBJECT_KEY, {
        "feature": {"dtype": "float32", "_type": "Value"}, "length": OBJECT_DIM, "_type": "Sequence"})
    meta[b"huggingface"] = json.dumps(hf).encode()
    return new.replace_schema_metadata(meta)


def episode_stats(objects: np.ndarray) -> dict:
    """LeRobot's own per-episode statistics of a column (compute_stats.get_feature_stats), as listed."""
    from lerobot.common.datasets.compute_stats import get_feature_stats

    return {k: v.tolist() for k, v in get_feature_stats(objects, axis=0, keepdims=False).items()}


def write(src_root: Path, out_root: Path, columns: list[np.ndarray]) -> None:
    import pyarrow.parquet as pq

    shutil.copytree(src_root, out_root)
    for path, objects in zip(source_parquets(out_root), columns, strict=True):
        pq.write_table(add_column(pq.read_table(path), objects), path)
    info_path = out_root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"] = _after_state(info["features"], OBJECT_KEY,
                                    {"dtype": "float32", "shape": [OBJECT_DIM], "names": OBJECT_NAMES})
    info_path.write_text(json.dumps(info, indent=4, ensure_ascii=False) + "\n")
    stats_path = out_root / "meta" / "episodes_stats.jsonl"
    lines = [json.loads(line) for line in stats_path.read_text().splitlines() if line.strip()]
    guard(len(lines) == len(columns), f"columns: {len(lines)} episodes' statistics, {len(columns)} episodes")
    for line, objects in zip(lines, columns):
        line["stats"] = _after_state(line["stats"], OBJECT_KEY, episode_stats(objects))
    stats_path.write_text("".join(json.dumps(line) + "\n" for line in lines))


def check_files(src_root: Path, out_root: Path, columns: list[np.ndarray]) -> None:
    """The columns guard on the files (the module docstring)."""
    import pyarrow.parquet as pq

    src_files = sorted(p.relative_to(src_root) for p in src_root.rglob("*") if p.is_file())
    out_files = sorted(p.relative_to(out_root) for p in out_root.rglob("*") if p.is_file())
    guard(src_files == out_files, "columns: the new dataset's files are not the source's")
    parquets = {p.relative_to(src_root) for p in source_parquets(src_root)}
    for rel in src_files:
        if rel in parquets or str(rel) in CHANGED:
            continue
        guard(filecmp.cmp(src_root / rel, out_root / rel, shallow=False), f"columns: {rel} differs from the source's")
    for rel, objects in zip(sorted(parquets), columns, strict=True):
        src, new = pq.read_table(src_root / rel), pq.read_table(out_root / rel)
        guard(new.column_names == _after_state_list(src.column_names), f"columns: {rel}'s columns are out of order")
        for name in src.column_names:
            guard(new.column(name).equals(src.column(name)), f"columns: {rel}'s {name} differs from the source's")
        got = np.asarray(new.column(OBJECT_KEY).to_pylist(), dtype=np.float32)
        guard(np.array_equal(got, objects), f"columns: {rel}'s {OBJECT_KEY} is not the one observed")
    src_info = json.loads((src_root / "meta" / "info.json").read_text())
    out_info = json.loads((out_root / "meta" / "info.json").read_text())
    guard({k: v for k, v in out_info.items() if k != "features"} == {k: v for k, v in src_info.items() if k != "features"}
          and {k: v for k, v in out_info["features"].items() if k != OBJECT_KEY} == src_info["features"],
          "columns: info.json differs from the source's beyond the new feature")
    for src_line, out_line in zip((src_root / "meta" / "episodes_stats.jsonl").read_text().splitlines(),
                                  (out_root / "meta" / "episodes_stats.jsonl").read_text().splitlines(), strict=True):
        a, b = json.loads(src_line), json.loads(out_line)
        b["stats"].pop(OBJECT_KEY)
        guard(a == b, f"columns: episode {a['episode_index']}'s statistics differ from the source's")


def check_load(src_root: Path, out_root: Path, repo_id: str, columns: list[np.ndarray]) -> dict:
    """The columns guard as the trainer loads the dataset: LeRobot's features, statistics and frames."""
    import torch
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    src_ds = LeRobotDataset(SOURCE, root=src_root)
    ds = LeRobotDataset(repo_id, root=out_root)
    guard(tuple(ds.meta.features[OBJECT_KEY]["shape"]) == (OBJECT_DIM,), "columns: LeRobot reads no object feature")
    for key, stats in src_ds.meta.stats.items():
        for stat, value in stats.items():
            guard(np.array_equal(np.asarray(ds.meta.stats[key][stat]), np.asarray(value)),
                  f"columns: LeRobot's {key} {stat} differs from the source's")
    everything = np.concatenate(columns)
    agg = {s: np.asarray(ds.meta.stats[OBJECT_KEY][s]) for s in ("min", "max")}
    guard(np.allclose(agg["min"], everything.min(0)) and np.allclose(agg["max"], everything.max(0)),
          "columns: LeRobot's object statistics are not the column's")
    starts = ds.episode_data_index["from"]
    for ep in np.unique(np.linspace(0, len(columns) - 1, 5).astype(int)):
        item = ds[int(starts[ep]) + 1]
        guard(torch.equal(item[OBJECT_KEY], torch.from_numpy(columns[ep][1])), f"columns: LeRobot's frame of episode {ep} differs")
    return {"episodes": len(columns), "copied_files_identical": True, "source_columns_equal": True,
            "object_stats": {s: np.asarray(ds.meta.stats[OBJECT_KEY][s]).tolist() for s in ("min", "max", "mean", "std")}}


def _after_state_list(names: list[str]) -> list[str]:
    return list(_after_state(dict.fromkeys(names), OBJECT_KEY, None))


def build(gpu: int, root: Path | None) -> dict:
    import mujoco
    import robosuite

    home = root or LEROBOT_HOME
    src_root, out_root = LEROBOT_HOME / SOURCE, home / OUTPUT
    guard(not out_root.exists(), f"{out_root} exists: remove it first, by hand")
    t0 = time.time()
    report, columns = observe_all(gpu, None, src_root)
    partial = out_root.with_name(out_root.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)  # a build that stopped before its guards passed
    write(src_root, partial, columns)
    check_files(src_root, partial, columns)
    partial.rename(out_root)
    try:
        report["columns"] = check_load(src_root, out_root, OUTPUT, columns)  # its cache is made for out_root
    except rr.GuardFailed:
        out_root.rename(out_root.with_name(out_root.name + ".failed"))
        raise
    report.update(
        task=TASK, source=SOURCE, output=OUTPUT, root=str(out_root), seconds=round(time.time() - t0),
        key=OBJECT_KEY, layout=OBJECT_NAMES,
        definition="resfit/dexmg/environments/object_obs.py: robosuite's SquareNut_pos and SquareNut_quat, "
                   "then the body peg1's position and quaternion (x y z w)",
        observed="each episode's first frame as reset() leaves its state; every later one as step() does",
        robomimic={"repo": rr.ROBOMIMIC_REPO, "revision": rr.ROBOMIMIC_REVISION, "file": rr.SOURCES[TASK][1],
                   "sha256": rr.SOURCES[TASK][2]},
        versions={"robosuite": robosuite.__version__, "mujoco": mujoco.__version__}, commit=rr._commit())
    (out_root / "meta" / "object_obs.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", type=int, default=rr.render_device(),
                    help="EGL render device; by default the trainer's, the first of CUDA_VISIBLE_DEVICES, or 1")
    ap.add_argument("--episodes", type=int, default=None, help="--check-only: the first N episodes only")
    ap.add_argument("--root", type=Path, default=None, help="write under this folder instead of the LeRobot home (tests)")
    ap.add_argument("--check-only", action="store_true", help="run the guards, write nothing")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    if rr.visible_gpus() and args.gpu not in rr.visible_gpus():
        ap.error(f"--gpu {args.gpu} is not in CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    if args.episodes is not None and not args.check_only:
        ap.error("--episodes is for --check-only: the dataset is built whole")
    try:
        if args.check_only:
            report, _ = observe_all(args.gpu, args.episodes, LEROBOT_HOME / SOURCE)
        else:
            report = build(args.gpu, args.root)
    except rr.GuardFailed as exc:
        print(json.dumps({"guard_failed": str(exc)}) if args.json else f"GUARD FAILED: {exc}")
        return 3
    print(json.dumps(report) if args.json else json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
