"""Enforce square-rerendered-1009's stop rules on its running trainers, so no one has to watch them.

    CACHE_DIR=~/projects/equi-resrl python tools/square_rerendered_watch.py   # inside tmux

The rules are EXPERIMENTS.md's 2026-10-09 PRE-REGISTERED entry ("the equivariant residual and tuned
RLPD on Square, on the re-rendered demos") as its AMENDMENT of the same date changes them. Every
minute it reads each of the seed group's trainers' history from its live wandb record and sends it
SIGINT when:
  - the residual has collapsed: three consecutive evals at or below 0.05 (tuned RLPD starts at 0,
    so this rule is the residual's alone);
  - an arm fails its 100k check, once its 100k eval is logged. The residual passes iff its evals
    at 60k-100k (5 x 50 episodes) clear B_sq = 0.582 by one standard error; tuned RLPD passes iff
    an eval to 100k is at least 0.04 or at least 5 training episodes to 100k succeeded;
  - it has converged: from 150k, at each eval, the mean of its last 5 evals does not beat the mean
    of the 5 before by two standard errors.
An arm that passes its 100k check is logged as due its seed 2, which a human launches (GPU and RAM
permitting). A critic value (`debug/q_data_demo` or `debug/q_data_online`) above 1.5 is logged once
per run. Once a tracked trainer has exited, a run package left incomplete is finalized. Every action
goes to CACHE_DIR/outputs/queue/watch.log (tools/run_watch.py's log).
"""

from __future__ import annotations

import json
import math
import os
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_watch import ESCALATE_S, LOG_DIR, POLL_S, _arg, log, package_of, trainers  # noqa: E402

from resfit.lerobot.utils.load_policy import _history_from_record, finalize_run_package  # noqa: E402

NOTES_PREFIX = "square-rerendered-1009/"
RESIDUAL, TUNED = "equi-residual-square-rerendered", "equi-tuned-rlpd-square-rerendered"
EPISODES = 50                       # per eval
COLLAPSE_RATE, COLLAPSE_RUN = 0.05, 3
CHECK_STEP = 100_000
B_SQ, B_SQ_EPISODES = 0.582, 500
RESIDUAL_WINDOW = (60_000, 100_000)
TUNED_MIN_RATE, TUNED_MIN_SUCCESSES = 0.04, 5
CONVERGE_FROM, CONVERGE_EVALS = 150_000, 5
Q_DIVERGED = 1.5


def collapsed(evals: list[tuple[int, float]]) -> str | None:
    run = 0
    for step, rate in evals:
        run = run + 1 if rate <= COLLAPSE_RATE else 0
        if run == COLLAPSE_RUN:
            return f"collapsed at {step}: {COLLAPSE_RUN} consecutive evals <= {COLLAPSE_RATE}"
    return None


def residual_check(evals: list[tuple[int, float]]) -> tuple[bool | None, str]:
    """The residual's 100k check: None if its window does not hold exactly five evals."""
    lo, hi = RESIDUAL_WINDOW
    window = [rate for step, rate in evals if lo <= step <= hi]
    if len(window) != 5:
        return None, f"{len(window)} evals in {lo}-{hi}, not 5: needs a human"
    n = EPISODES * len(window)
    k = round(sum(window) * EPISODES)
    p = k / n
    se = math.sqrt(p * (1 - p) / n + B_SQ * (1 - B_SQ) / B_SQ_EPISODES)
    return p - B_SQ >= se, f"{k} of {n} ({p:.3f}) at {lo}-{hi}, against {B_SQ} + 1 SE = {B_SQ + se:.3f}"


def tuned_check(evals: list[tuple[int, float]], successes: int) -> tuple[bool, str]:
    best = max((rate for step, rate in evals if step <= CHECK_STEP), default=0.0)
    passed = best >= TUNED_MIN_RATE or successes >= TUNED_MIN_SUCCESSES
    return passed, (f"best eval to {CHECK_STEP} {best:.2f} (needs {TUNED_MIN_RATE}), "
                    f"{successes} training successes (needs {TUNED_MIN_SUCCESSES})")


def converged(evals: list[tuple[int, float]]) -> tuple[bool, str]:
    rates = [rate for _, rate in evals]
    last, before = rates[-CONVERGE_EVALS:], rates[-2 * CONVERGE_EVALS:-CONVERGE_EVALS]
    n = EPISODES * CONVERGE_EVALS
    p1, p2 = sum(last) / CONVERGE_EVALS, sum(before) / CONVERGE_EVALS
    se = math.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)
    still_rising = p1 > p2 and p1 - p2 >= 2 * se
    return not still_rising, f"last {CONVERGE_EVALS} {p1:.3f}, the {CONVERGE_EVALS} before {p2:.3f}, 2 SE {2 * se:.3f}"


def decide(arm: str, evals: list[tuple[int, float]], successes: int, passed_check: bool) -> tuple[str | None, str]:
    """("stop" | "passed" | "human" | None, why) for a trainer with this history now."""
    if arm == RESIDUAL and (why := collapsed(evals)):
        return "stop", why
    if not evals or evals[-1][0] < CHECK_STEP:
        return None, ""
    if not passed_check:
        passed, why = residual_check(evals) if arm == RESIDUAL else tuned_check(evals, successes)
        if passed is None:
            return "human", f"100k check: {why}"
        return ("passed" if passed else "stop"), f"100k check {'passed' if passed else 'failed'}: {why}"
    if evals[-1][0] >= CONVERGE_FROM and len(evals) >= 2 * CONVERGE_EVALS:
        done, why = converged(evals)
        if done:
            return "stop", f"converged at {evals[-1][0]}: {why}"
    return None, ""


def history(package: Path) -> tuple[list[tuple[int, float]], list[tuple[int, float]], list[tuple[int, str, float]]]:
    """Evals, training episode returns, and critic values, by step, from the live record."""
    manifest = json.loads((package / "manifest.json").read_text())
    rows = _history_from_record(Path(manifest["wandb"]["dir"]).parent / f"run-{package.name}.wandb")
    evals = sorted((int(r["_step"]), float(r["eval/success_rate"])) for r in rows
                   if r.get("eval/success_rate") is not None)
    returns = [(int(r["_step"]), float(r["training/episode_return"])) for r in rows
               if r.get("training/episode_return") is not None]
    qs = [(int(r["_step"]), key, float(r[key])) for r in rows
          for key in ("debug/q_data_demo", "debug/q_data_online") if r.get(key) is not None]
    return evals, returns, qs


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    tracked: dict[int, dict] = {}
    log(f"started (pid {os.getpid()}): square-rerendered-1009, prefix {NOTES_PREFIX}")
    while True:
        live = trainers(NOTES_PREFIX)
        for pid, trainer in live.items():
            if pid not in tracked:
                package = package_of(trainer)
                if package is None:
                    continue  # wandb not up yet
                arm = _arg(trainer["argv"], "wandb.notes").split("/")[1]
                tracked[pid] = {"package": package, "arm": arm, "signalled": None, "passed": False,
                                "q_flagged": False, "human": False, "last_eval": None}
                log(f"TRACK {package.name} (pid {pid}, {_arg(trainer['argv'], 'wandb.name')})")
            entry = tracked[pid]
            if entry["signalled"] is not None:
                if time.time() - entry["signalled"] > ESCALATE_S:
                    log(f"SIGTERM {entry['package'].name} (pid {pid}): still alive after SIGINT")
                    os.kill(pid, signal.SIGTERM)
                    entry["signalled"] = time.time()
                continue
            try:
                evals, returns, qs = history(entry["package"])
            except Exception as exc:  # the live record can be mid-write; try next minute
                log(f"READ FAILED {entry['package'].name}: {type(exc).__name__}: {exc}")
                continue
            name = f"{entry['package'].name} ({entry['arm']})"
            if not entry["q_flagged"] and (high := [q for q in qs if q[2] > Q_DIVERGED]):
                step, key, value = high[0]
                log(f"FLAG {name}: {key} {value:.2f} at {step}, above {Q_DIVERGED}: report to Aden")
                entry["q_flagged"] = True
            if not evals or evals[-1][0] == entry["last_eval"]:
                continue
            entry["last_eval"] = evals[-1][0]
            successes = sum(ret > 0 for step, ret in returns if step <= CHECK_STEP)
            action, why = decide(entry["arm"], evals, successes, entry["passed"])
            rates = " ".join(f"{rate:.2f}" for _, rate in evals)
            log(f"EVAL {name} at {evals[-1][0]}: {evals[-1][1]:.2f}; training successes to "
                f"{CHECK_STEP} {successes}")
            if action == "passed":
                entry["passed"] = True
                log(f"PASSED {name}: {why}; due its seed 2 (a human launches it); evals {rates}")
            elif action == "human" and not entry["human"]:
                entry["human"] = True
                log(f"HUMAN {name}: {why}; not stopped; evals {rates}")
            elif action == "stop":
                log(f"STOP {name} (pid {pid}): {why}; evals {rates}")
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
        time.sleep(POLL_S)


if __name__ == "__main__":
    sys.exit(main())
