"""Enforce the Square ablations' stop rules on running trainers, so no one has to watch them.

    tools/run_watch.py --notes-prefix square-ablations-1001 \
        [--stop-at 100000] [--queue-tops CRITIC_FN_ID CLIPQ_ID COMMIT]

Run it inside tmux, from a pinned worktree, with CACHE_DIR set, beside tools/gpu_queue.sh. Every
minute it finds each running trainer (`python -m resfit.rl_finetuning.scripts.train_residual_td3`)
whose wandb.notes starts with the prefix, reads its evals from its live wandb record, and sends
it SIGINT when either:
  - it has logged its eval at --stop-at and was launched to run longer (the two runs launched at
    150k before the 2026-10-01 amendment moved the stop to 100k); or
  - it has collapsed: three consecutive evals at or below 0.05 (the pre-registered rule).
The trainer exits, then its submit.sh, and the GPU's queue runner launches its next line. Once a
trainer it tracked has exited, a run package left incomplete (by a stop or a crash) is finalized.

--queue-tops resolves the squash and depth arms of the 2026-09-30 table. Once both named runs
have finished, it computes their verdicts with tools/square_verdict.py and queues the two arms at
seed 1 from COMMIT's worktree: squash on GPU 1 ahead of the target-fix line there, depth at the
end of GPU 0's queue. If either verdict is missing it queues nothing and says so.

Every action is logged to CACHE_DIR/outputs/queue/watch.log, and remembered in watch_state.json
beside it, so a restarted watcher does not queue twice.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from square_verdict import (  # noqa: E402
    COLLAPSE_RATE,
    COLLAPSE_RUN,
    PROJECT,
    load_history,
    top_arms,
    verdict,
)

from resfit.lerobot.utils.load_policy import (  # noqa: E402
    RUNS_ROOT,
    _history_from_record,
    finalize_run_package,
)

TRAINER = "resfit.rl_finetuning.scripts.train_residual_td3"
QUEUES = {0: Path.home() / "queues" / "gpu0.txt", 1: Path.home() / "queues" / "gpu1.txt"}
LOG_DIR = RUNS_ROOT.parent / "queue"
POLL_S = 60
ESCALATE_S = 15 * 60        # SIGTERM a trainer still alive this long after its SIGINT
TOPS_TRIES = 5              # queue_tops errors tolerated before leaving it to a human


def log(message: str) -> None:
    line = f"{datetime.now():%F %T} watch {message}"
    print(line, flush=True)
    with open(LOG_DIR / "watch.log", "a") as f:
        f.write(line + "\n")


def _arg(argv: list[str], key: str) -> str | None:
    return next((a.split("=", 1)[1] for a in argv if a.startswith(f"{key}=")), None)


def trainers(notes_prefix: str) -> dict[int, dict]:
    """pid -> cwd and argv of every running trainer in this seed group."""
    out = {}
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = (proc / "cmdline").read_bytes().decode().split("\0")
            if TRAINER not in argv:
                continue
            cwd = os.readlink(proc / "cwd")
        except (OSError, UnicodeDecodeError):
            continue
        if (_arg(argv, "wandb.notes") or "").startswith(notes_prefix):
            out[int(proc.name)] = {"cwd": cwd, "argv": argv}
    return out


def package_of(trainer: dict) -> Path | None:
    """The newest incomplete run package launched from the trainer's worktree with its name."""
    name = f"wandb.name={_arg(trainer['argv'], 'wandb.name')}"
    best = None
    for manifest_path in (RUNS_ROOT / PROJECT).glob("*/manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        argv = manifest.get("argv") or []
        if (manifest.get("completed") or name not in argv
                or not argv[0].startswith(trainer["cwd"] + "/")):
            continue
        if best is None or manifest["started"] > best[0]:
            best = (manifest["started"], manifest_path.parent)
    return best[1] if best else None


def evals_from_record(package: Path) -> list[tuple[int, float]]:
    manifest = json.loads((package / "manifest.json").read_text())
    record = Path(manifest["wandb"]["dir"]).parent / f"run-{package.name}.wandb"
    rows = _history_from_record(record)
    return sorted((int(r["_step"]), float(r["eval/success_rate"])) for r in rows
                  if r.get("eval/success_rate") is not None)


def stop_reason(evals: list[tuple[int, float]], total_steps: int, stop_at: int) -> str | None:
    """Why a trainer with these evals should be stopped now, or None."""
    run = 0
    for step, rate in evals:
        run = run + 1 if rate <= COLLAPSE_RATE else 0
        if run == COLLAPSE_RUN:
            return f"collapsed at {step}: three consecutive evals <= {COLLAPSE_RATE}"
    if total_steps > stop_at and any(step >= stop_at for step, _ in evals):
        return f"its {stop_at} eval is logged and it was launched to {total_steps}"
    return None


def insert_line(queue: Path, line: str, before_arm: str | None) -> None:
    """Put `line` before the first pending line for `before_arm`, or at the end."""
    lines = queue.read_text().splitlines()
    at = len(lines)
    for i, existing in enumerate(lines):
        fields = existing.split()
        if before_arm and len(fields) == 3 and not existing.lstrip().startswith("#") \
                and fields[1] == before_arm:
            at = i
            break
    lines.insert(at, line)
    tmp = queue.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + "\n")
    os.replace(tmp, queue)


def queue_tops(state: dict, critic_fn_id: str, clipq_id: str, commit: str) -> None:
    if state.get("tops_queued"):
        return
    for run_id in (critic_fn_id, clipq_id):
        manifest = json.loads((RUNS_ROOT / PROJECT / run_id / "manifest.json").read_text())
        if not manifest.get("completed"):
            return
    verdicts = {run_id: verdict(load_history(run_id)) for run_id in (critic_fn_id, clipq_id)}
    for run_id, v in verdicts.items():
        log(f"VERDICT {run_id}: advances={v['advances']} ({v['why']})")
    if any(v["advances"] is None for v in verdicts.values()):
        log("TOPS NOT QUEUED: a verdict is missing; squash and depth need a human decision")
        state["tops_queued"] = "missing verdict"
        return
    squash, depth = top_arms(verdicts[critic_fn_id]["advances"], verdicts[clipq_id]["advances"])
    insert_line(QUEUES[1], f"{commit} {squash} 1", before_arm="target-fix")
    insert_line(QUEUES[0], f"{commit} {depth} 1", before_arm=None)
    log(f"QUEUED {squash} on GPU 1 and {depth} on GPU 0, at {commit}")
    for gpu, queue in QUEUES.items():
        pending = [l for l in queue.read_text().splitlines() if l.strip() and not l.startswith("#")]
        log(f"  gpu{gpu} queue now: {pending}")
    state["tops_queued"] = {"squash": squash, "depth": depth,
                            "verdicts": {k: v["advances"] for k, v in verdicts.items()}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--notes-prefix", required=True,
                        help="only trainers whose wandb.notes starts with this")
    parser.add_argument("--stop-at", type=int, default=100_000)
    parser.add_argument("--queue-tops", nargs=3, metavar=("CRITIC_FN_ID", "CLIPQ_ID", "COMMIT"))
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    state_path = LOG_DIR / "watch_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    tracked: dict[int, dict] = {}
    log(f"started (pid {os.getpid()}): prefix {args.notes_prefix}, stop at {args.stop_at}, "
        f"queue tops {args.queue_tops}")
    while True:
        live = trainers(args.notes_prefix)
        for pid, trainer in live.items():
            if pid not in tracked:
                package = package_of(trainer)
                if package is None:
                    continue  # wandb not up yet
                total = int(_arg(trainer["argv"], "algo.total_timesteps") or 0)
                tracked[pid] = {"package": package, "total": total, "signalled": None}
                log(f"TRACK {package.name} (pid {pid}, {_arg(trainer['argv'], 'wandb.name')}, "
                    f"launched to {total})")
            entry = tracked[pid]
            if entry["signalled"] is not None:
                if time.time() - entry["signalled"] > ESCALATE_S:
                    log(f"SIGTERM {entry['package'].name} (pid {pid}): still alive after SIGINT")
                    os.kill(pid, signal.SIGTERM)
                    entry["signalled"] = time.time()
                continue
            try:
                evals = evals_from_record(entry["package"])
            except Exception as exc:  # the live record can be mid-write; try next minute
                log(f"READ FAILED {entry['package'].name}: {type(exc).__name__}: {exc}")
                continue
            reason = stop_reason(evals, entry["total"], args.stop_at)
            if reason:
                log(f"STOP {entry['package'].name} (pid {pid}): {reason}; evals "
                    + " ".join(f"{rate:.2f}" for _, rate in evals))
                os.kill(pid, signal.SIGINT)
                entry["signalled"] = time.time()

        for pid in [p for p in tracked if p not in live]:
            package = tracked.pop(pid)["package"]
            manifest = json.loads((package / "manifest.json").read_text())
            if manifest.get("completed"):
                log(f"EXITED {package.name}: finished, package complete")
            else:
                finalize_run_package(package)
                log(f"EXITED {package.name}: package finalized by the watcher")

        if args.queue_tops and not state.get("tops_queued"):
            try:
                queue_tops(state, *args.queue_tops)
            except Exception as exc:
                failures = state.get("tops_failures", 0) + 1
                state["tops_failures"] = failures
                log(f"TOPS FAILED ({failures} of {TOPS_TRIES}): {type(exc).__name__}: {exc}")
                if failures >= TOPS_TRIES:
                    state["tops_queued"] = f"gave up: {exc}"
                    log("TOPS NOT QUEUED: squash and depth need a human decision")
            state_path.write_text(json.dumps(state, indent=2))
        time.sleep(POLL_S)


if __name__ == "__main__":
    sys.exit(main())
