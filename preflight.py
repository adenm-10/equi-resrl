#!/usr/bin/env python
"""Pre-submission gate. Run this before every training launch.

Three tiers, cheapest first:

  Tier 1  static      seconds, no GPU   imports, configs, git state, env
  Tier 2  construct   ~2 min, GPU       layouts, equivariance, no-op pins
  Tier 3  resources   seconds           artifacts, buffer caches, RAM, disk

Exits non-zero if anything blocking fails, so submit.sh can gate on it.

    python preflight.py                            # full gate for the Can reproduction
    python preflight.py --task Square              # the Square config instead
    python preflight.py --concurrent-seeds 2       # planning to run two at once
    python preflight.py --skip-tier2               # when both GPUs are busy
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent

# Approximate CPU RAM held per run: the online replay buffer is preallocated at
# algo.buffer_size, and the offline buffer is loaded alongside it.
RAM_PER_RUN_GB = 21

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


def tier1_compose(overrides: list[str]) -> bool:
    """Compose the exact override set through Hydra without training.

    Catches malformed overrides before a launch. Hydra treats a comma in a
    value as a list separator, so `wandb.notes="a, b"` aborts at startup --
    which cost two failed launches before this check existed.
    """
    section("Tier 1 - hydra override composition")
    if not overrides:
        record(WARN, "compose", "no overrides passed; skipping (submit.sh passes them)")
        return True
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
        return True
    err = (proc.stderr or proc.stdout).strip().splitlines()
    record(FAIL, "hydra composes",
           "\n".join(f"{' ' * 9}  {l}" for l in err[:6]))
    return False


def tier1_pytest() -> bool:
    section("Tier 1 - static tests")
    return _run_pytest(
        ["tests/test_imports.py", "tests/test_config.py",
         "tests/test_regression_single_arm.py"],
        "imports + config + single-arm regression",
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
    section("Tier 2 - layouts, equivariance, no-op pins (GPU)")
    return _run_pytest(
        ["tests/test_layouts.py", "tests/test_equivariance.py", "tests/test_no_ops.py"],
        "equivariance suite",
    )


# ---------------------------------------------------------------------------
# Tier 3: resources the run needs
# ---------------------------------------------------------------------------

TASKS = {
    "Can": dict(dataset="ankile/robomimic-mh-can-image",
                bc="robomimic-can-bc/xhjdl8a7", horizon=200),
    "Square": dict(dataset="ankile/robomimic-mh-square-image",
                   bc="robomimic-square-bc/3hzs5bz1", horizon=300),
}

IMAGE_KEYS = ["observation.images.agentview", "observation.images.robot0_eye_in_hand"]


def _sha8(d: dict) -> str:
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:8]


def cache_hashes(task: str) -> tuple[str, str]:
    """Mirror the cache keys built in train_residual_td3.py.

    Kept in sync by hand. If the hashes reported here do not match what the
    training script prints at startup, this function is stale.
    """
    import tensordict
    import torchrl
    t = TASKS[task]
    common = dict(task=task, image_keys=IMAGE_KEYS, n_step=3, gamma=0.99,
                  sampling_strategy="uniform", normalized_actions=True,
                  min_action_range=1e-1, min_state_std=1e-1,
                  torchrl_version=torchrl.__version__,
                  tensordict_version=tensordict.__version__)
    offline = {**common, "dataset_name": t["dataset"], "num_episodes": 300,
               "use_base_policy_for_base_actions": True,
               "base_policy_wandb_id": t["bc"], "batch_size": 128}
    online = {**common, "horizon": t["horizon"], "size": 10_000,
              "buffer_size": 200_000, "batch_size": 128,
              "random_action_noise_scale": 0.2}
    return _sha8(offline), _sha8(online)


def tier3_resources(task: str, seeds: int) -> None:
    section(f"Tier 3 - resources for {task}")
    t = TASKS[task]

    # -- base BC policy ----------------------------------------------------
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

    # -- buffer caches -----------------------------------------------------
    try:
        off_h, on_h = cache_hashes(task)
        off = REPO / "offline_buffer_cache" / off_h
        on = REPO / "online_buffer_cache" / on_h
        record(PASS if off.exists() else WARN, f"offline buffer cache {off_h}",
               "hit, loads in minutes" if off.exists()
               else "MISS: will rebuild from the dataset (slow, and writes ~5 GB)")
        record(PASS if on.exists() else WARN, f"online buffer cache {on_h}",
               "hit, so the 10k-step env warmup is skipped" if on.exists()
               else "MISS: 10k-step env warmup will run, then write ~16 GB")
        record(INFO, "cache key assumption",
               "hashes assume default n_step=3 / gamma=0.99 / buffer_size=200k / "
               "learning_starts=10k. Overriding any of those on the CLI changes the "
               "key and forces a full rebuild. Cross-check against the hashes the "
               "training script prints at startup.")
    except Exception as exc:
        record(WARN, "buffer cache hashes", f"could not compute: {exc}")

    # -- env probe ---------------------------------------------------------
    probe = REPO / "env_probes" / f"{task}.json"
    if probe.exists():
        record(PASS, f"robot base probe {task}.json", json.loads(probe.read_text())["base_xy"].__str__())
    else:
        record(WARN, "robot base probe missing",
               "will be detected by launching a probe env. If it silently fails, pos_xy is "
               "centred on the world origin and the symmetry is broken (see EQUIVARIANCE.md)")

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
    record(PASS if free > 40 else WARN, "disk", f"{free:.0f} GB free")


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="Can", choices=sorted(TASKS))
    ap.add_argument("--concurrent-seeds", type=int, default=1,
                    help="how many runs will share this machine at once")
    ap.add_argument("--skip-tier2", action="store_true", help="skip the GPU suite")
    ap.add_argument("--compose", nargs=argparse.REMAINDER, default=[],
                    help="hydra overrides to dry-run compose; must come last")
    args = ap.parse_args()

    print(f"\033[1mPre-submission gate\033[0m  task={args.task}  repo={REPO}")

    tier1_environment()
    tier1_git()
    ok = tier1_compose(args.compose)
    ok = tier1_pytest() and ok
    if not args.skip_tier2:
        ok = tier2() and ok
    else:
        record(WARN, "Tier 2 skipped", "--skip-tier2 was passed; equivariance is unverified")
    tier3_resources(args.task, args.concurrent_seeds)

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
