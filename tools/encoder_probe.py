"""What a frozen encoder's features hold about the nut, read at its 1x1 output, its 3x3 map and its
9x9 map (docs/plans/2026-10-23-encoder-interface.md, section 2).

    python -m tools.encoder_probe targets                    # the nut's pose per sampled demo frame
    python -m tools.encoder_probe probe eee2 [rrr2 ...]      # each encoder's taps, probed
    python -m tools.encoder_probe report                     # one table over every encoder probed

Why. The equivariant critic sees 288 numbers per frame, its encoder's 1x1 outputs, and ResFiT's
sees its ViTs' 20,736 spatial tokens. Inside the equivariant encoder the 9x9 map before the last max
pool is 20,736 numbers too, and the last pool and conv keep 256 of them. If the nut's place relative
to the gripper can be read from the 9x9 map and not from the 1x1 output, the squeeze loses it; if
from neither, the encoder never extracts it. Decodable is not used: a probe shows what is there, not
what the critic reads.

Data. The re-rendered Square demos (`local/robomimic-mh-square-rerendered`), every STRIDE-th frame.
Each frame's targets come from its simulator state in robomimic's `demo_v15.hdf5`, set into the
demos' own MuJoCo scene with no renderer: the first frame of an episode as reset() leaves it, every
later one as step() does (`rerender_demos.as_stepped`), which is how each frame was rendered. The
end-effector position so read must equal the dataset's observation.state, which checks the pairing.

Targets: the nut's position relative to the grip site (x, y, z, m); its yaw relative to the grip
site's (sin, cos); its position relative to the square peg; its position in the world (x, y); and
whether it is lifted, more than LIFT_M above where its episode started.

Probes. Episodes split by index, every fifth held out (60 of 300), so neighbouring frames never
straddle the split; of the other 240, 48 pick the ridge penalty and stop the MLP, and 192 fit. Each
tap is reduced to its principal components on the fitting frames (by its kernel when it is wider
than there are frames), then ridge regression on every component (full size) and on the first k
(equal size, so a wider tap gets no free advantage), and a one-hidden-layer MLP on the first k. R^2 on the
held-out frames per target component, averaged per target; the nut-to-gripper xy error in mm;
the lifted flag's AUROC from its regression score.

Outputs go to outputs/diagnostics/enc-iface/probe/ (gitignored): targets.npz, frames.npy (the
decoded frames, a cache), and <encoder>.json per encoder. CPU only, a few GB of RAM at most.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs" / "diagnostics" / "enc-iface" / "probe"
RUNS = REPO / "outputs" / "runs" / "robomimic-square-final"
DATASET = "local/robomimic-mh-square-rerendered"
CAMERAS = ("observation.images.agentview", "observation.images.robot0_eye_in_hand")
STRIDE = 8
LIFT_M = 0.01
HELD_OUT_EVERY = 5   # episode i is held out when i % 5 == 0
VAL_EVERY = 5        # of the rest, every fifth (by order) picks the penalty and stops the MLP
EQUAL_K = (256, 32)  # the equal-size reductions; a tap narrower than k is used whole
MLP_FULL_MAX = 2304  # the MLP reads at most this many components
LAMBDAS = np.logspace(-5, 2, 15)  # times the mean eigenvalue
TARGETS = {          # name: columns of targets.npz's "y"
    "nut_grip_xyz": slice(0, 3), "nut_grip_yaw": slice(3, 5), "nut_peg_xyz": slice(5, 8),
    "nut_world_xy": slice(8, 10), "lifted": slice(10, 11),
}
NUT_BODY, PEG_BODY, GRIP_SITE = "SquareNut_main", "peg1", "gripper0_right_grip_site"
EEF_TOL = 1e-5       # m: the grip site against the dataset's robot0_eef_pos (float32)

# name: (run id or None for an untrained encoder, the run whose config builds it, seed for an init)
ENCODERS = {
    "eee2": ("wk45zp27", "wk45zp27", None),
    "archwide2": ("h216n857", "h216n857", None),
    "ere2": ("6xh63lfb", "6xh63lfb", None),
    "eee3": ("jr63pxb9", "jr63pxb9", None),
    "E-init": (None, "wk45zp27", 2),
    "rrr2": ("w1g1d1tf", "w1g1d1tf", None),
    "R-init": (None, "w1g1d1tf", 2),
}


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

def yaw_of(xmat: np.ndarray) -> float:
    """The heading of a body's x axis in the world's xy plane, from its row-major rotation matrix."""
    return float(np.arctan2(xmat[3], xmat[0]))


def read_pose(m, d) -> dict[str, np.ndarray]:
    """The nut's, the peg's and the grip site's positions and the nut's and site's yaw, from MjData
    `d` as it stands (positions already computed)."""
    import mujoco

    nut = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, NUT_BODY)
    peg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, PEG_BODY)
    site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, GRIP_SITE)
    return {"nut": d.xpos[nut].copy(), "peg": d.xpos[peg].copy(), "grip": d.site_xpos[site].copy(),
            "nut_yaw": yaw_of(d.xmat[nut]), "grip_yaw": yaw_of(d.site_xmat[site])}


def target_row(pose: dict, rest_z: float) -> np.ndarray:
    rel_yaw = pose["nut_yaw"] - pose["grip_yaw"]
    return np.concatenate([
        pose["nut"] - pose["grip"], [np.sin(rel_yaw), np.cos(rel_yaw)], pose["nut"] - pose["peg"],
        pose["nut"][:2], [float(pose["nut"][2] > rest_z + LIFT_M)],
    ])


def demo_scene(xml: str):
    """The demos' scene, its asset paths made local as robosuite does (edit_model_xml reads no state)."""
    import mujoco
    from robosuite.environments.base import MujocoEnv

    return mujoco.MjModel.from_xml_string(MujocoEnv.edit_model_xml(None, xml))


def set_flat_state(m, d, state: np.ndarray) -> None:
    """robosuite's set_state_from_flattened: [time, qpos, qvel]."""
    d.time, d.qpos[:], d.qvel[:] = state[0], state[1:1 + m.nq], state[1 + m.nq:1 + m.nq + m.nv]


def episode_poses(m, d, states: np.ndarray, frames: list[int]) -> list[dict]:
    """The poses each listed frame was rendered at: frame 0 from its state, as reset() observes,
    every later one as step() leaves it."""
    import mujoco

    from tools.rerender_demos import as_stepped

    poses = []
    for t in frames:
        set_flat_state(m, d, states[t])
        if t == 0:
            mujoco.mj_forward(m, d)
        else:
            as_stepped(m, d)
        poses.append(read_pose(m, d))
    return poses


def sampled_frames(lengths: list[int], stride: int) -> list[tuple[int, int]]:
    return [(ep, t) for ep, n in enumerate(lengths) for t in range(0, n, stride)]


def build_targets(stride: int = STRIDE) -> dict:
    """targets.npz: per sampled frame its episode, frame index and targets; checked against the
    dataset's end-effector positions."""
    import h5py
    import mujoco
    import pyarrow.parquet as pq
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

    from tools.rerender_demos import demo_keys, states_file

    meta = LeRobotDatasetMetadata(DATASET)
    episodes, frames, rows, worst = [], [], [], 0.0
    with h5py.File(states_file("Square"), "r") as f:
        keys = demo_keys(f)
        assert len(keys) == meta.total_episodes, (len(keys), meta.total_episodes)
        m = demo_scene(f[f"data/{keys[0]}"].attrs["model_file"])
        d = mujoco.MjData(m)
        for ep, key in enumerate(keys):
            states = f[f"data/{key}"]["states"][()]
            eef = np.asarray(pq.read_table(meta.root / meta.get_data_file_path(ep_index=ep),
                                           columns=["observation.state"]).column(0).to_pylist())
            assert len(eef) == len(states), f"episode {ep}: {len(eef)} frames, {len(states)} states"
            ts = list(range(0, len(states), stride))
            poses = episode_poses(m, d, states, ts)
            set_flat_state(m, d, states[0])
            mujoco.mj_forward(m, d)
            rest_z = read_pose(m, d)["nut"][2]
            for t, pose in zip(ts, poses):
                worst = max(worst, float(np.abs(pose["grip"] - eef[t, :3]).max()))
                episodes.append(ep), frames.append(t), rows.append(target_row(pose, rest_z))
    assert worst < EEF_TOL, f"the grip site is {worst:.2e} m from the dataset's eef position: pairing broken"
    out = {"episode": np.array(episodes), "frame": np.array(frames), "y": np.array(rows),
           "stride": stride, "eef_max_diff_m": worst}
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "targets.npz", **out)
    print(f"{len(rows)} frames from {len(keys)} episodes; grip site within {worst:.1e} m of the dataset's eef")
    return out


def load_targets() -> dict:
    with np.load(OUT / "targets.npz") as z:
        return {k: z[k] for k in z.files}


def decoded_frames(episodes: np.ndarray, frames: np.ndarray) -> np.ndarray:
    """[n, cameras, 3, 84, 84] uint8 for the sampled frames, decoded once and cached."""
    import io

    import pyarrow.parquet as pq
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from PIL import Image

    path = OUT / "frames.npy"
    if path.exists():
        cached = np.load(path, mmap_mode="r")
        if cached.shape[0] == len(episodes):
            return cached
    meta = LeRobotDatasetMetadata(DATASET)
    images = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=(len(episodes), len(CAMERAS), 3, 84, 84))
    row = 0
    for ep in np.unique(episodes):
        table = pq.read_table(meta.root / meta.get_data_file_path(ep_index=int(ep)), columns=list(CAMERAS))
        for t in frames[episodes == ep]:
            for c, cam in enumerate(CAMERAS):
                png = table.column(cam)[int(t)].as_py()["bytes"]
                images[row, c] = np.asarray(Image.open(io.BytesIO(png)).convert("RGB")).transpose(2, 0, 1)
            row += 1
    images.flush()
    return np.load(path, mmap_mode="r")


# ---------------------------------------------------------------------------
# Encoders and taps
# ---------------------------------------------------------------------------

def load_encoder(name: str):
    """A run's encoder from its best model (its weights at that eval's step), or an untrained one
    built from a run's config at a seed; in eval mode: the centred crop, no random shift."""
    import torch

    from resfit.rl_finetuning.equi_off_policy.rl.q_agent import build_encoder
    from tools.residual_field_ablation import run_config

    run_id, config_run, seed = ENCODERS[name]
    cfg = run_config(RUNS / config_run)
    cameras = [cfg.rl_camera] if isinstance(cfg.rl_camera, str) else list(cfg.rl_camera)
    assert tuple(cameras) == CAMERAS, cameras
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    enc = build_encoder((3, 84, 84), cameras, cfg.agent, cfg.equivariance, residual_actor=True)
    enc.eval()
    info = {"run": run_id, "config_run": config_run, "family": enc.encoder, "init_seed": seed}
    if run_id is not None:
        path = RUNS / run_id / "wandb_best" / "best_model.pt"
        saved = torch.load(path, map_location="cpu", weights_only=False)
        state = {k[len("enc."):]: v for k, v in saved["agent_state_dict"].items()
                 if k.startswith("enc.") and not k.startswith("enc.normalizer.")}
        missing, unexpected = enc.load_state_dict(state, strict=False)
        assert not missing and not unexpected, (missing, unexpected)
        info.update(checkpoint=str(path.relative_to(REPO)), step=saved["global_step"],
                    eval_success=saved["success_rate"])
        del saved
    return enc, info


def taps(enc, images) -> dict[str, "torch.Tensor"]:
    """Every tap of one batch of uint8 images [B, cameras, 3, 84, 84], flattened per frame."""
    import torch

    front, wrist = images[:, 0], images[:, 1]
    out = {}
    if enc.encoder == "resfit":
        for cam, img, vit in zip(("front", "wrist"), (front, wrist), enc.vits):
            tokens = vit.vit(img.float() / 255.0 - 0.5)            # [B, 81, 128], ResFiT's eval path
            out[f"{cam}_tokens"] = tokens.flatten(1)
            out[f"{cam}_mean"] = tokens.mean(dim=1)
        return out
    x = enc.crop_randomizer(enc._image_to_signed_unit(front))
    conv, end = (enc.enc_obs.conv, enc.enc_obs.map_end) if enc.encoder == "equivariant" else (enc.enc_obs, enc.obs_map_end)
    if enc.encoder == "equivariant":
        x = enc.enc_obs._as_input(x)
    m9 = conv[:end](x)
    m3 = conv[end](m9)
    m1 = conv[end + 1:](m3)
    value = (lambda g: g.tensor) if enc.encoder == "equivariant" else (lambda g: g)
    out.update(front_9x9=value(m9).flatten(1), front_3x3=value(m3).flatten(1), front_1x1=value(m1).flatten(1))
    ih = enc.enc_ih[0]
    y = enc.crop_randomizer(enc._image_to_signed_unit(wrist))
    w9 = ih.conv[:ih.map_end](y)
    w3 = ih.conv[ih.map_end](w9)
    out.update(wrist_9x9=w9.flatten(1), wrist_3x3=w3.flatten(1), wrist_1x1=ih(y))
    assert torch.equal(out["wrist_1x1"], ih.out_ln(ih.conv[ih.map_end + 1:](w3).flatten(1)))
    return out


def encoder_features(enc, images: np.ndarray, batch: int = 128) -> dict[str, np.ndarray]:
    """Every tap for every sampled frame, float32 (float16 for taps wider than 4,096)."""
    import torch

    feats = {}
    with torch.no_grad():
        for start in range(0, len(images), batch):
            got = taps(enc, torch.from_numpy(np.ascontiguousarray(images[start:start + batch])))
            for k, v in got.items():
                if k not in feats:
                    dtype = np.float16 if v.shape[1] > 4096 else np.float32
                    feats[k] = np.empty((len(images), v.shape[1]), dtype=dtype)
                feats[k][start:start + len(v)] = v.numpy()
    return feats


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

def episode_split(episodes: np.ndarray) -> dict[str, np.ndarray]:
    """Row indices by episode: test (every fifth episode), and of the rest val (every fifth in order),
    which picks the ridge penalty and stops the MLP, and fit."""
    eps = np.unique(episodes)
    test_eps = eps[eps % HELD_OUT_EVERY == 0]
    val_eps = eps[eps % HELD_OUT_EVERY != 0][::VAL_EVERY]
    rows = np.arange(len(episodes))
    test, val = rows[np.isin(episodes, test_eps)], rows[np.isin(episodes, val_eps)]
    fit = rows[~np.isin(episodes, test_eps) & ~np.isin(episodes, val_eps)]
    return {"fit": fit, "val": val, "test": test}


def principal_scores(x: np.ndarray, fit: np.ndarray, others: list[np.ndarray], tol: float = 1e-9):
    """The tap's principal component scores on the fitting rows and on each of `others`, in
    descending variance, and the eigenvalues; by the d x d covariance when the tap is narrower than
    there are fitting rows, else by the n x n kernel. Components below tol of the largest are
    dropped. The fitting rows' scores are orthogonal, each column's squared norm its eigenvalue."""
    mean = x[fit].astype(np.float64).mean(axis=0).astype(np.float32)
    xf = x[fit].astype(np.float32) - mean
    xo = [x[rows].astype(np.float32) - mean for rows in others]
    if x.shape[1] <= len(fit):
        s, v = np.linalg.eigh((xf.T @ xf).astype(np.float64))
        s, v = s[::-1], v[:, ::-1]
        keep = s > tol * s[0]
        v = v[:, keep].astype(np.float32)
        return xf @ v, [o @ v for o in xo], s[keep]
    k = (xf @ xf.T).astype(np.float64)
    ko = [o @ xf.T for o in xo]
    del xf, xo
    s, u = np.linalg.eigh(k)
    del k
    s, u = s[::-1], u[:, ::-1]
    keep = s > tol * s[0]
    s, u = s[keep], u[:, keep]
    root = np.sqrt(s)
    return (u * root).astype(np.float32), [(o.astype(np.float64) @ (u / root)).astype(np.float32) for o in ko], s


def r2(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Per column, against the column's own mean."""
    return 1.0 - ((y - pred) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)


def ridge(z_fit, s, y_fit, z_val, y_val, z_test, lambdas=LAMBDAS) -> tuple[np.ndarray, np.ndarray]:
    """Ridge on principal scores, whose fitting columns are orthogonal with squared norms s, so each
    penalty's solution is diagonal; the penalty picked per target column on the validation rows.
    Returns the test predictions and the penalties (times the mean eigenvalue)."""
    ym = y_fit.mean(0)
    b = z_fit.T.astype(np.float64) @ (y_fit - ym)                 # [k, targets]
    preds_val = [z_val @ (b / (s + lam * s.mean())[:, None]) + ym for lam in lambdas]
    best = np.stack([r2(y_val, p) for p in preds_val]).argmax(axis=0)
    w = np.stack([b[:, j] / (s + lambdas[best[j]] * s.mean()) for j in range(b.shape[1])], axis=1)
    return z_test @ w + ym, lambdas[best]


def mlp(z_fit, y_fit, z_val, y_val, z_test, hidden: int = 256, epochs: int = 300, patience: int = 20, seed: int = 0):
    """One hidden layer of ReLUs on standardized scores, AdamW, stopped at the validation rows' best loss."""
    import torch

    torch.manual_seed(seed)
    mu, sd = z_fit.mean(0), z_fit.std(0) + 1e-6
    ym, ys = y_fit.mean(0), y_fit.std(0) + 1e-6

    def t(a):
        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))

    xf, yf, xv, yv = t((z_fit - mu) / sd), t((y_fit - ym) / ys), t((z_val - mu) / sd), t((y_val - ym) / ys)
    net = torch.nn.Sequential(torch.nn.Linear(xf.shape[1], hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, yf.shape[1]))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state, bad = np.inf, None, 0
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        net.train()
        for idx in torch.randperm(len(xf), generator=g).split(256):
            loss = torch.nn.functional.mse_loss(net(xf[idx]), yf[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = torch.nn.functional.mse_loss(net(xv), yv).item()
        if v < best - 1e-4:
            best, best_state, bad = v, {k: p.clone() for k, p in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    with torch.no_grad():
        return net(t((z_test - mu) / sd)).numpy() * ys + ym


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    """The chance a random positive outscores a random negative, ties counted half."""
    pos, neg = score[label > 0.5], score[label <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    allv = np.concatenate([pos, neg])
    for v in np.unique(allv):  # average the ranks of ties
        tie = allv == v
        if tie.sum() > 1:
            ranks[tie] = ranks[tie].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def readouts(y_test: np.ndarray, pred: np.ndarray) -> dict:
    """Held-out R^2 per target (the mean over its components) and per component; the nut-to-gripper
    xy R^2 alone, which the plan's bars read; its xy error in mm; the lifted flag's AUROC."""
    per = r2(y_test, pred)
    out = {name: float(per[cols].mean()) for name, cols in TARGETS.items() if name != "lifted"}
    xy = TARGETS["nut_grip_xyz"]
    out["nut_grip_xy"] = float(per[xy][:2].mean())
    out["r2_per_component"] = per.tolist()
    out["nut_grip_xy_err_mm"] = float(np.linalg.norm((y_test[:, xy] - pred[:, xy])[:, :2], axis=1).mean() * 1000)
    lifted = TARGETS["lifted"].start
    out["lifted_auroc"] = auroc(pred[:, lifted], y_test[:, lifted])
    return out


def probe_tap(x: np.ndarray, y: np.ndarray, split: dict) -> dict:
    """Ridge at full size and at each equal size; the MLP at each equal size and on the first
    MLP_FULL_MAX components (every one, if fewer)."""
    fit, val, test = split["fit"], split["val"], split["test"]
    z_fit, (z_val, z_test), s = principal_scores(x, fit, [val, test])
    yf, yv, yt = y[fit], y[val], y[test]
    res = {"width": int(x.shape[1]), "components": int(len(s)),
           "variance_in_first": {str(k): float(s[:k].sum() / s.sum()) for k in EQUAL_K}}
    pred, lam = ridge(z_fit, s, yf, z_val, yv, z_test)
    res["ridge_full"] = {**readouts(yt, pred), "penalties": lam.tolist()}
    for k in EQUAL_K:
        if k < len(s):
            pred, lam = ridge(z_fit[:, :k], s[:k], yf, z_val[:, :k], yv, z_test[:, :k])
            res[f"ridge_{k}"] = {**readouts(yt, pred), "penalties": lam.tolist()}
            res[f"mlp_{k}"] = readouts(yt, mlp(z_fit[:, :k], yf, z_val[:, :k], yv, z_test[:, :k]))
    kf = min(len(s), MLP_FULL_MAX)
    name = "mlp_full" if kf == len(s) else f"mlp_{kf}"
    res[name] = readouts(yt, mlp(z_fit[:, :kf], yf, z_val[:, :kf], yv, z_test[:, :kf]))
    return res


def probe(name: str) -> dict:
    import torch

    out = OUT / f"{name}.json"
    if out.exists():
        sys.exit(f"{out} exists; a measurement is a record, delete it to measure again")
    t = load_targets()
    split = episode_split(t["episode"])
    images = decoded_frames(t["episode"], t["frame"])
    started = time.time()
    enc, info = load_encoder(name)
    feats = encoder_features(enc, images)
    del enc
    print(f"{name}: features in {time.time() - started:.0f}s: " + ", ".join(f"{k} {v.shape[1]}" for k, v in feats.items()),
          flush=True)
    results = {}
    for tap in sorted(feats):
        t0 = time.time()
        results[tap] = probe_tap(feats.pop(tap), t["y"], split)
        r = results[tap]
        print(f"  {tap:13} {r['width']:6}  ridge_full xy-R2 {r['ridge_full']['nut_grip_xyz']:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    record = {"encoder": name, **info, "taps": results, "frames": int(len(t["y"])), "stride": int(t["stride"]),
              "split": {k: int(len(np.unique(t["episode"][v]))) for k, v in split.items()},
              "rows": {k: int(len(v)) for k, v in split.items()}, "threads": torch.get_num_threads(),
              "finished": datetime.now().isoformat(timespec="seconds")}
    out.write_text(json.dumps(record, indent=1))
    print(f"wrote {out}")
    return record


def report() -> None:
    """Per encoder and tap: held-out R^2 of nut-to-gripper xyz and yaw, nut-to-peg, mm error, AUROC."""
    rows = []
    for path in sorted(OUT.glob("*.json")):
        rec = json.loads(path.read_text())
        step = rec.get("step")
        for tap, r in rec["taps"].items():
            for probe_name in [k for k in r if k.startswith(("ridge", "mlp"))]:
                v = r[probe_name]
                rows.append((rec["encoder"], step, tap, r["width"], probe_name, v["nut_grip_xy"], v["nut_grip_xyz"],
                             v["nut_grip_yaw"], v["nut_peg_xyz"], v["nut_world_xy"], v["nut_grip_xy_err_mm"],
                             v["lifted_auroc"]))
    print(f"{'encoder':10} {'step':>6} {'tap':13} {'width':>6} {'probe':10} {'grip_xy':>7} {'grip_xyz':>8} {'yaw':>6} "
          f"{'peg_xyz':>7} {'world':>6} {'xy_mm':>6} {'lift':>5}")
    for r in rows:
        print(f"{r[0]:10} {str(r[1]):>6} {r[2]:13} {r[3]:6} {r[4]:10} {r[5]:7.3f} {r[6]:8.3f} {r[7]:6.3f} "
              f"{r[8]:7.3f} {r[9]:6.3f} {r[10]:6.1f} {r[11]:5.3f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("targets")
    p = sub.add_parser("probe")
    p.add_argument("encoders", nargs="+", choices=sorted(ENCODERS))
    sub.add_parser("report")
    args = parser.parse_args(argv)
    if args.cmd == "targets":
        build_targets()
    elif args.cmd == "probe":
        for name in args.encoders:
            probe(name)
    else:
        report()
    return 0


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    sys.exit(main())
