#!/usr/bin/env python
"""Pre-submission gate. Run this before every training launch.

Three tiers, cheapest first:

  Tier 1  static      ~2 min            imports, configs, git state, env, mujoco's version,
                                        and the render against the demos (EGL, one minute)
  Tier 2  construct   ~2 min, GPU       layouts, equivariance, no-op pins, host syncs,
                                        env image format
  Tier 3  resources   seconds           artifacts, buffer caches, RAM, disk

Exits non-zero if anything blocking fails, so submit.sh can gate on it.

    python preflight.py                            # full gate for the Can reproduction
    python preflight.py --task Square              # the Square config instead
    python preflight.py --concurrent-seeds 2       # planning to run two at once
    python preflight.py --skip-tier2               # when both GPUs are busy
    python preflight.py --task Can --method tuned-rlpd --stack resfit   # vanilla RL, no base policy
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent

# Approximate CPU RAM held per run: the online replay buffer is preallocated at
# algo.buffer_size, and the offline buffer is loaded alongside it.
RAM_PER_RUN_GB = 21

# The mujoco whose render matches the demos (tier1_environment, tier1_render).
MUJOCO = "3.3.2"

PASS, FAIL, WARN, INFO = "PASS", "FAIL", "WARN", "INFO"
_MARK = {PASS: "  ok  ", FAIL: " FAIL ", WARN: " warn ", INFO: " info "}

results: list[tuple[str, str, str]] = []


def record(status: str, name: str, detail: str = "") -> None:
    results.append((status, name, detail))
    print(f"[{_MARK[status]}] {name}" + (f"\n{' ' * 9}{detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m")


# ---------------------------------------------------------------------------
# Tier 1: static
# ---------------------------------------------------------------------------


def tier1_environment() -> None:
    section("Tier 1 - environment")

    if sys.prefix.rstrip("/").endswith("envs/residual"):
        record(PASS, "conda env", f"running in {sys.prefix}")
    else:
        record(FAIL, "conda env",
               f"expected .../envs/residual, got {sys.prefix}. Run: conda activate residual")

    if str(REPO) in sys.path or str(REPO) in os.environ.get("PYTHONPATH", ""):
        record(PASS, "PYTHONPATH", "repo root is importable")
    else:
        record(FAIL, "PYTHONPATH",
               f'repo root not on the path. Run: export PYTHONPATH="{REPO}:$PYTHONPATH"')

    # Versions pinned to z8yoqylh's requirements.txt — the only equivariant run
    # that ever succeeded. These were previously taken from a3e3zylp, which is a
    # run that collapsed to 0.00, so the gate was validating against a failure.
    # A mismatch does not only risk behaviour drift: torchrl and tensordict
    # versions are part of the replay-buffer cache key, so bumping either
    # silently invalidates 21 GB of cached buffers and triggers a rebuild.
    expected = {"torch": "2.6.0", "torchrl": "0.7.0", "tensordict": "0.7.0", "escnn": "1.0.11"}
    for mod, want in expected.items():
        try:
            got = __import__(mod).__version__
        except Exception as exc:
            record(FAIL, f"{mod} import", f"{type(exc).__name__}: {exc}")
            continue
        if got.split("+")[0] == want:
            record(PASS, f"{mod} {got}")
        else:
            record(WARN, f"{mod} version", f"expected {want}, got {got}")

    # mujoco decides what the agent sees: 3.8.1 renders robosuite's floor differently from the
    # demos, and tuned RLPD never learned in it (EXPERIMENTS.md, 2026-10-07 FINDING). 3.3.2
    # matches them. tier1_render measures the match; this keeps the version itself on the gate.
    try:
        import mujoco
        if mujoco.__version__ == MUJOCO:
            record(PASS, f"mujoco {mujoco.__version__}", "the version whose render matches the demos")
        else:
            record(FAIL, "mujoco version", f"expected {MUJOCO}, got {mujoco.__version__}. "
                   f"Run: pip install --no-deps mujoco=={MUJOCO}")
    except Exception as exc:
        record(FAIL, "mujoco import", f"{type(exc).__name__}: {exc}")

    try:
        import torch
        if torch.cuda.is_available():
            names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            record(PASS, "CUDA", f"{len(names)} device(s): {', '.join(names)}")
        else:
            record(FAIL, "CUDA", "torch.cuda.is_available() is False")
    except Exception as exc:
        record(FAIL, "CUDA", str(exc))


def tier1_git() -> None:
    section("Tier 1 - git state")

    def git(*args):
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True,
                              text=True, check=False).stdout.strip()

    head = git("rev-parse", "--short", "HEAD")
    record(INFO, "HEAD", f"{head}  {git('log', '-1', '--format=%s')}")

    dirty = [l for l in git("status", "--porcelain").splitlines()
             if l and not l.startswith("??")]
    if dirty:
        record(WARN, "uncommitted changes",
               f"{len(dirty)} tracked file(s) modified. The commit recorded with the run "
               "will not describe the code that ran:\n" + "\n".join(
                   f"{' ' * 9}  {d}" for d in dirty[:8]))
    else:
        record(PASS, "working tree", "no modified tracked files")

    untracked = [l for l in git("status", "--porcelain").splitlines() if l.startswith("??")]
    py_untracked = [l for l in untracked if l.endswith(".py")]
    if py_untracked:
        record(WARN, "untracked python files",
               "not reproducible from the commit:\n" + "\n".join(
                   f"{' ' * 9}  {d}" for d in py_untracked))
    else:
        record(PASS, "no untracked .py files")


def tier1_compose(overrides: list[str]) -> tuple[bool, dict[str, bool]]:
    """Compose the exact override set through Hydra without training.

    Catches malformed overrides before a launch. Hydra treats a comma in a
    value as a list separator, so `wandb.notes="a, b"` aborts at startup --
    which cost two failed launches before this check existed. Also returns the
    composed run's two cache flags and encode_env_frames_like_demos, which is in
    the online key, for Tier 3; empty without overrides.
    """
    section("Tier 1 - hydra override composition")
    if not overrides:
        record(WARN, "compose", "no overrides passed; skipping (submit.sh passes them)")
        return True, {}
    proc = subprocess.run(
        [sys.executable, "-m", "resfit.rl_finetuning.scripts.train_residual_td3",
         "--cfg", "job", "--resolve", *overrides],
        cwd=REPO, capture_output=True, text=True, check=False,
        env={**os.environ, "PYTHONPATH": f"{REPO}:{os.environ.get('PYTHONPATH', '')}"},
    )
    if proc.returncode == 0:
        seed = next((l.strip() for l in proc.stdout.splitlines()
                     if l.startswith("seed:")), "seed: ?")
        record(PASS, "hydra composes", f"{len(overrides)} override(s); {seed}")
        flags = {m.group(1): m.group(2) == "true" for m in re.finditer(
            r"^((?:read|write)_buffer_caches|encode_env_frames_like_demos): (true|false)$",
            proc.stdout, flags=re.M)}
        return True, flags
    err = (proc.stderr or proc.stdout).strip().splitlines()
    record(FAIL, "hydra composes",
           "\n".join(f"{' ' * 9}  {l}" for l in err[:6]))
    return False, {}


def tier1_render() -> None:
    """Demo first frames against env reset frames, front camera, Can and Square: tools/render_check.

    Both tasks whatever the launch: they share mujoco, and a render that drifts on one is a sign
    for all. About a minute, in a subprocess, so the gate's own process never loads a renderer.
    """
    section("Tier 1 - render matches the demos")
    proc = subprocess.run(
        [sys.executable, "-m", "tools.render_check", "--task", "Can", "--task", "Square", "--json"],
        cwd=REPO, capture_output=True, text=True, check=False,
        env={**os.environ, "PYTHONPATH": f"{REPO}:{os.environ.get('PYTHONPATH', '')}"},
    )
    found = [json.loads(l) for l in proc.stdout.splitlines() if l.startswith("{")]
    if len(found) != 2:
        tail = [l for l in proc.stderr.splitlines() if l.strip()][-3:]
        record(FAIL, "render check", "did not report both tasks:\n" + "\n".join(
            f"{' ' * 9}  {l}" for l in tail))
        return
    for r in found:
        record(PASS if r["passed"] else FAIL, f"render: {r['task']}",
               f"front camera, {r['frames']} frames: demo vs env {r['demo_vs_env'][0]:.4f}, "
               f"bar {r['bar']:.4f} (2.5x the {r['within_demos']:.4f} between halves of the "
               f"demos); mujoco {r['mujoco']}")


def tier1_pytest() -> bool:
    section("Tier 1 - static tests")
    return _run_pytest(
        ["tests/test_imports.py", "tests/test_config.py",
         "tests/test_regression_single_arm.py", "tests/test_regression_agent.py",
         "tests/test_run_package.py", "tests/test_env_wrapper.py",
         "tests/test_eval_base_policy.py", "tests/test_square_verdict.py",
         "tests/test_tuned_rlpd_verdict.py", "tests/test_baseline_verdict.py",
         "tests/test_gpu_queue.py",
         "tests/test_render_check.py", "tests/test_demo_codec.py"],
        "imports + config + regression + run package + env wrappers + base-rate eval + verdicts"
        " + queue runner + render check",
    )


def _run_pytest(paths: list[str], label: str) -> bool:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *paths, "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=REPO, capture_output=True, text=True, check=False,
        env={**os.environ, "PYTHONPATH": f"{REPO}:{os.environ.get('PYTHONPATH', '')}"},
    )
    tail = [l for l in proc.stdout.splitlines() if l.strip()][-1:] or ["no output"]
    if proc.returncode == 0:
        record(PASS, f"pytest {label}", tail[-1])
        return True
    failed = [l for l in proc.stdout.splitlines() if l.startswith("FAILED")]
    record(FAIL, f"pytest {label}",
           tail[-1] + ("\n" + "\n".join(f"{' ' * 9}  {f}" for f in failed[:12]) if failed else ""))
    return False


# ---------------------------------------------------------------------------
# Tier 2: construction and equivariance
# ---------------------------------------------------------------------------


def tier2() -> bool:
    section("Tier 2 - layouts, equivariance, no-op pins, host syncs (GPU)")
    return _run_pytest(
        ["tests/test_layouts.py", "tests/test_equivariance.py", "tests/test_no_ops.py",
         "tests/test_sweep_arms.py", "tests/test_sync_free.py", "tests/test_obs_formats.py",
         "tests/test_critic_diagnostics.py", "tests/test_bc_term.py"],
        "equivariance suite + sweep arms + sync-free update + env image format + critic diagnostics"
        " + BC term",
    )


# ---------------------------------------------------------------------------
# Tier 3: resources the run needs
# ---------------------------------------------------------------------------

# image_keys and num_episodes are part of the buffer cache key, so they live
# per task rather than as module constants. One wrist camera per arm.
_ONE_ARM_CAMS = ["observation.images.agentview", "observation.images.robot0_eye_in_hand"]
_TWO_ARM_CAMS = _ONE_ARM_CAMS + ["observation.images.robot1_eye_in_hand"]

TASKS = {
    "Can": dict(dataset="ankile/robomimic-mh-can-image",
                bc="robomimic-can-bc/xhjdl8a7", horizon=200,
                image_keys=_ONE_ARM_CAMS, num_episodes=300),
    "Square": dict(dataset="ankile/robomimic-mh-square-image",
                   bc="robomimic-square-bc/3hzs5bz1", horizon=300,
                   image_keys=_ONE_ARM_CAMS, num_episodes=300),
    "TwoArmBoxCleanup": dict(dataset="ankile/dexmg-two-arm-box-cleanup",
                             bc="dexmg-boxcleanup-bc/xg0w2r0z", horizon=300,
                             image_keys=_TWO_ARM_CAMS, num_episodes=1_000,
                             frames=230_700),
}

# Frames actually loaded (num_episodes worth), from each dataset's meta/info.json.
TASKS["Can"]["frames"] = 62_756
TASKS["Square"]["frames"] = 80_731

# Square's offline cache is 6.4 GB for 80,731 frames x 2 cameras. The same
# constant predicts Can's recorded 5 GB from 62,756 frames, so it holds twice.
KB_PER_FRAME_CAMERA = 39.6
# Measured: Square's online cache is 16 GB for buffer_size=200k x 2 cameras.
GB_PER_ONLINE_CAMERA = 8.0


def cache_size_estimate(task: str) -> tuple[float, float]:
    """Estimated (offline, online) cache sizes in GB, scaled from Square."""
    t = TASKS[task]
    cams = len(t["image_keys"])
    offline = t["frames"] * cams * KB_PER_FRAME_CAMERA / 1e6
    return offline, GB_PER_ONLINE_CAMERA * cams


def _sha8(d: dict) -> str:
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:8]


def cache_hashes(task: str, method: str = "residual", stack: str = "equi",
                 encode_env_frames_like_demos: bool = False) -> tuple[str, str]:
    """Mirror the cache keys built in train_residual_td3.py.

    Kept in sync by hand. If the hashes reported here do not match what the
    training script prints at startup, this function is stale. A residual key
    does not record the stack (TODO B7, S9); a tuned-RLPD key adds the fields
    that record it. Env frames through the demos' codec add a field to the online key.
    """
    from importlib.metadata import version

    import tensordict
    import torchrl
    t = TASKS[task]
    common = dict(task=task, image_keys=t["image_keys"], n_step=3, gamma=0.99,
                  sampling_strategy="uniform", normalized_actions=True,
                  min_action_range=1e-1, min_state_std=1e-1,
                  torchrl_version=torchrl.__version__,
                  tensordict_version=tensordict.__version__)
    offline = {**common, "dataset_name": t["dataset"], "num_episodes": t["num_episodes"],
               "use_base_policy_for_base_actions": True,
               "base_policy_wandb_id": t["bc"], "batch_size": 128}
    # The online buffer holds rendered frames, so its key records mujoco (2026-10-07).
    online = {**common, "horizon": t["horizon"], "size": 10_000,
              "buffer_size": 200_000, "batch_size": 128,
              "random_action_noise_scale": 0.2, "mujoco_version": version("mujoco")}
    if method == "tuned-rlpd":
        vanilla = {"base_policy": None,
                   "normalization": {"equi": "identity", "resfit": "dataset"}[stack]}
        offline.update(use_base_policy_for_base_actions=False, base_policy_wandb_id=None, **vanilla)
        online.update(random_action_noise_scale=1.0, **vanilla)
    if encode_env_frames_like_demos:
        online["encode_env_frames_like_demos"] = True
    return _sha8(offline), _sha8(online)


def tier3_resources(task: str, seeds: int, require_caches: bool = False,
                    method: str = "residual", stack: str = "equi",
                    caches: dict[str, bool] | None = None) -> None:
    """`caches`: the composed run's read/write_buffer_caches and encode_env_frames_like_demos
    (tier1_compose); config defaults if absent."""
    section(f"Tier 3 - resources for {task} ({method}, {stack} stack)")
    caches = caches or {}
    read = caches.get("read_buffer_caches", True)
    t = TASKS[task]
    # The trainer reads caches and probes under CACHE_DIR, which a launch from a
    # pinned worktree points at the main checkout. Look where it will look.
    cache_root = pathlib.Path(os.environ.get("CACHE_DIR", REPO)).expanduser().resolve()
    miss = FAIL if require_caches else WARN

    # -- base BC policy ----------------------------------------------------
    if method == "residual":
        tier3_bc_policy(t)
    else:
        record(INFO, "BC policy", "not used: tuned RLPD has no base policy")

    # -- buffer caches -----------------------------------------------------
    need_gb = 0.0
    try:
        off_h, on_h = cache_hashes(task, method, stack, caches.get("encode_env_frames_like_demos", False))
        off = cache_root / "offline_buffer_cache" / off_h
        on = cache_root / "online_buffer_cache" / on_h
        off_gb, on_gb = cache_size_estimate(task)
        if not read:
            # The trainer refuses read_buffer_caches=false with write_buffer_caches=true.
            for name, h in (("offline", off_h), ("online", on_h)):
                record(INFO, f"{name} buffer cache {h}",
                       "not read and not written: read_buffer_caches=false fills it in the run")
        elif method == "tuned-rlpd":
            # write_buffer_caches=false (TunedRLPDConfig): nothing is written, hit or miss.
            for name, h, path in (("offline", off_h, off), ("online", on_h, on)):
                record(INFO, f"{name} buffer cache {h}",
                       "hit, will be read" if path.exists() else "none: built in memory, not written")
        else:
            record(PASS if off.exists() else miss, f"offline buffer cache {off_h}",
                   "hit, loads in minutes" if off.exists()
                   else f"MISS: will rebuild from the dataset (slow, writes ~{off_gb:.0f} GB)")
            record(PASS if on.exists() else miss, f"online buffer cache {on_h}",
                   "hit, so the 10k-step env warmup is skipped" if on.exists()
                   else f"MISS: 10k-step env warmup will run, then writes ~{on_gb:.0f} GB")
            need_gb = (0 if off.exists() else off_gb) + (0 if on.exists() else on_gb)
        record(INFO, "cache key assumption",
               "hashes assume default n_step=3 / gamma=0.99 / buffer_size=200k / "
               "learning_starts=10k. Overriding any of those on the CLI changes the "
               "key and forces a full rebuild. Cross-check against the hashes the "
               "training script prints at startup.")
    except Exception as exc:
        record(WARN, "buffer cache hashes", f"could not compute: {exc}")

    # -- env probe ---------------------------------------------------------
    probe = cache_root / "env_probes" / f"{task}.json"
    if stack == "resfit":
        record(INFO, "robot base probe", "not used: the ResFiT stack has no equivariant normalizer")
    elif probe.exists():
        data = json.loads(probe.read_text())
        bases = data["bases"] if "bases" in data else [data["base_xy"]]
        centre = [sum(c) / len(bases) for c in zip(*bases)]
        record(PASS, f"robot base probe {task}.json",
               f"{len(bases)} base(s) {bases}, rotation centre {centre}")
    else:
        record(WARN, "robot base probe missing",
               "will be detected by launching a probe env. If it silently fails, pos_xy is "
               "centred on the world origin and the symmetry is broken (see EQUIVARIANCE.md)")

    tier3_memory_and_disk(need_gb, seeds)


def tier3_bc_policy(t: dict) -> None:
    """The residual run's base policy: cached locally, and resolvable in wandb."""
    cached = list(REPO.glob(f"artifacts/run_{t['bc'].split('/')[-1]}_best*/policy/model.safetensors"))
    if cached:
        record(PASS, "BC policy cached locally", str(cached[0].parent.parent.relative_to(REPO)))
    else:
        record(WARN, "BC policy not in artifacts/", "will download ~1 GB from wandb")

    try:
        import wandb
        api = wandb.Api(timeout=30)
        proj, rid = t["bc"].split("/")
        api.run(f"{api.default_entity}/{proj}/{rid}")
        record(PASS, "BC run resolves in wandb", t["bc"])
    except Exception as exc:
        status = PASS if cached else FAIL
        record(status, "BC run in wandb",
               f"{type(exc).__name__}. " + ("Local cache present, so this is survivable."
                                            if cached else "No local cache either: BLOCKING."))


def tier3_memory_and_disk(need_gb: float, seeds: int) -> None:
    """RAM for `seeds` concurrent runs, and disk for `need_gb` of cache each writes."""
    # -- memory ------------------------------------------------------------
    try:
        # MemAvailable, not free: page cache is reclaimable, and this box keeps
        # tens of GB of buffer-cache pages resident after a run.
        avail = None
        for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                avail = int(line.split()[1]) / 1e6
                break
        if avail is None:
            avail = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1e9
        need = RAM_PER_RUN_GB * seeds
        detail = (f"{avail:.0f} GB available, ~{RAM_PER_RUN_GB} GB per run "
                  f"x {seeds} concurrent = ~{need} GB")
        record(PASS if need < avail * 0.85 else FAIL, "RAM headroom", detail)
    except Exception:
        record(WARN, "RAM headroom", "could not determine")

    free = shutil.disk_usage(REPO).free / 1e9
    need_gb = need_gb * seeds
    if need_gb:
        # Blocking: filling the disk mid-build corrupts the cache and wastes hours.
        detail = (f"{free:.0f} GB free, ~{need_gb:.0f} GB of cache to write "
                  f"= ~{free - need_gb:.0f} GB margin")
        record(PASS if free - need_gb > 10 else FAIL, "disk", detail)
    else:
        record(PASS if free > 40 else WARN, "disk", f"{free:.0f} GB free, no cache to write")


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="Can", choices=sorted(TASKS))
    ap.add_argument("--concurrent-seeds", type=int, default=1,
                    help="how many runs will share this machine at once")
    ap.add_argument("--skip-tier2", action="store_true", help="skip the GPU suite")
    ap.add_argument("--require-caches", action="store_true",
                    help="a buffer-cache miss blocks, instead of warning of a rebuild")
    ap.add_argument("--method", default="residual", choices=("residual", "tuned-rlpd"),
                    help="tuned-rlpd is vanilla RL (base_policy=None): no BC checks, and "
                         "buffers built in memory")
    ap.add_argument("--stack", default="equi", choices=("equi", "resfit"),
                    help="the agent; sets the tuned-RLPD cache keys and whether the probe is read")
    ap.add_argument("--compose", nargs=argparse.REMAINDER, default=[],
                    help="hydra overrides to dry-run compose; must come last")
    args = ap.parse_args()

    print(f"\033[1mPre-submission gate\033[0m  task={args.task}  method={args.method}  "
          f"stack={args.stack}  repo={REPO}")

    tier1_environment()
    tier1_render()
    tier1_git()
    ok, caches = tier1_compose(args.compose)
    ok = tier1_pytest() and ok
    if not args.skip_tier2:
        ok = tier2() and ok
    else:
        record(WARN, "Tier 2 skipped", "--skip-tier2 was passed; equivariance is unverified")
    tier3_resources(args.task, args.concurrent_seeds, args.require_caches, args.method, args.stack,
                    caches)

    fails = [r for r in results if r[0] == FAIL]
    warns = [r for r in results if r[0] == WARN]

    print("\n" + "=" * 72)
    if fails:
        print(f"\033[1;31mNO-GO\033[0m  {len(fails)} blocking failure(s), {len(warns)} warning(s)")
        for _, name, _ in fails:
            print(f"    - {name}")
        print("=" * 72)
        return 1
    print(f"\033[1;32mGO\033[0m  all blocking checks passed"
          + (f", {len(warns)} warning(s) to read first" if warns else ""))
    for _, name, _ in warns:
        print(f"    ? {name}")
    print("=" * 72)
    print("\nReminder: pre-register the run in docs/EXPERIMENTS.md before launching")
    print("(STANDARDS.md rule 5.4). Two runs have already been lost to wandb deletion.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
