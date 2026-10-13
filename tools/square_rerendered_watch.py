"""Enforce a Square seed group's stop rules on its running trainers, so no one has to watch them.

    CACHE_DIR=~/projects/equi-resrl python tools/square_rerendered_watch.py --group <seed-group>   # inside tmux

Nine seed groups (GROUPS), on the re-rendered Square demos unless said otherwise:
  - square-rerendered-1009: EXPERIMENTS.md's 2026-10-09 PRE-REGISTERED entry ("the equivariant
    residual and tuned RLPD on Square, on the re-rendered demos"), as its AMENDMENTs of 2026-10-09
    and 2026-10-10 change it;
  - square-trlpd-stab-1010: the 2026-10-10 PRE-REGISTERED entry ("stabilizing equivariant tuned
    RLPD on Square"), four tuned-RLPD arms under the same rules;
  - square-residual-start-1011: the 2026-10-11 PRE-REGISTERED entry ("the equivariant residual on
    Square: its actor's last layer at zero, and a 50k critic warm-up"), two residual arms under the
    residual's rules;
  - resfit-rx-1012: the 2026-10-12 PRE-REGISTERED entry ("ResFiT's residual on Square with rx at
    the equivariant stack's range"), one ResFiT residual arm on the original demos, stopped only at
    its collapse: its anchors ran to 150k with no 100k check, and the trainer ends itself there;
  - residual-tilt-1012: the 2026-10-12 PRE-REGISTERED entry ("the equivariant residual on Square
    with no tilt, and a diagnostic run with checkpoints"): notilt to a fixed 150k from that date's
    AMENDMENT, and the diagnostic run, whose evals are logged but which is never stopped (it ends
    at 60k);
  - resfit-equi-diff-1012 and equi-resfit-arch-1013: the 2026-10-12 PRE-REGISTERED entry ("ResFiT's
    residual on Square, given the equivariant stack's differences one at a time; the equivariant
    residual given ResFiT's layout"). ResFiT residual arms (on the original demos but for
    rerendered and combined) and equivariant residual arms. From that date's AMENDMENT ("every
    run to a fixed 150k"), all to a fixed 150k;
  - equi-residual-long-1012: the 2026-10-12 PRE-REGISTERED entry ("the equivariant residual's
    anchors again, to a fixed 150k"): 6ng62tpm's and eaqorqki's launches at seed 2, to a fixed
    150k, and from the 2026-10-13 AMENDMENT 6ng62tpm's at seed 3 too;
  - resfit-ln-place-1013: the 2026-10-13 PRE-REGISTERED entry ("ResFiT's residual on Square with
    LayerNorm off in one place"): three ResFiT residual arms on the original demos, to a fixed
    150k.
An arm to a fixed length (FIXED_LENGTH_ARMS) is stopped only on the divergence stop, or at its
STOP_AT step when its trainer would run on past it.
Every minute it reads each of the group's trainers' history from its live wandb record and sends it
SIGINT when:
  - a residual has collapsed: three consecutive evals at or below 0.05 (tuned RLPD starts at 0,
    so this rule is the residual's alone);
  - its critic has diverged: every logged critic value in the 5,000 steps before an eval is above
    10 (the largest of `debug/q_data_demo`, `q_data_online`, `q_policy_demo` and `q_policy_online`
    in each row; no target can exceed 1, EXPERIMENTS.md 2026-10-10 MEASUREMENT) and its last three
    evals are at or below 0.05;
  - an arm fails its 100k check, once its 100k eval is logged. The residual passes iff its evals
    at 60k-100k (5 x 50 episodes) clear B_sq = 0.582 by one standard error; tuned RLPD passes iff
    an eval to 100k is at least 0.04 or at least 5 training episodes to 100k succeeded;
  - it has converged: from 150k, at each eval, the mean of its last 5 evals does not beat the mean
    of the 5 before by two standard errors.
An arm that passes its 100k check is logged as due its seed 2, which a human launches (GPU and RAM
permitting). A critic value above 1.5 (any of the four) is logged once per run. Once a tracked
trainer has exited, a run package left incomplete is finalized. Every action goes to
CACHE_DIR/outputs/queue/watch.log (tools/run_watch.py's log), each line naming the group.
"""

from __future__ import annotations

import argparse
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

RESIDUAL, TUNED = "equi-residual-square-rerendered", "equi-tuned-rlpd-square-rerendered"
STABILIZERS = ("equi-tuned-rlpd-square-clipq", "equi-tuned-rlpd-square-pooledln",
               "equi-tuned-rlpd-square-ln", "equi-tuned-rlpd-square-10x2")
RESIDUAL_STARTS = ("equi-residual-square-init0", "equi-residual-square-warmup50k")
RESFIT_RX = ("resfit-residual-square-rx",)
NOTILT, DIAG = "equi-residual-square-notilt", "equi-residual-square-diag"
RESFIT_EQUI_DIFF = ("resfit-residual-square-rerendered", "resfit-residual-square-noln",
                    "resfit-residual-square-limits", "resfit-residual-square-noln-all",
                    "resfit-residual-square-combined", "resfit-residual-square-smallcritic",
                    "resfit-residual-square-noise")
EQUI_RESFIT_ARCH = ("equi-residual-square-arch", "equi-residual-square-arch-wide")
EQUI_LONG = ("equi-residual-square-long", "equi-residual-square-init0-long")
RESFIT_LN_PLACE = ("resfit-residual-square-noln-actor", "resfit-residual-square-noln-critic",
                   "resfit-residual-square-noln-vit")
GROUPS = {"square-rerendered-1009": (RESIDUAL, TUNED), "square-trlpd-stab-1010": STABILIZERS,
          "square-residual-start-1011": RESIDUAL_STARTS, "resfit-rx-1012": RESFIT_RX,
          "residual-tilt-1012": (NOTILT, DIAG), "resfit-equi-diff-1012": RESFIT_EQUI_DIFF,
          "equi-resfit-arch-1013": EQUI_RESFIT_ARCH, "equi-residual-long-1012": EQUI_LONG,
          "resfit-ln-place-1013": RESFIT_LN_PLACE}
RESIDUAL_ARMS = {RESIDUAL, *RESIDUAL_STARTS, *RESFIT_RX, NOTILT, DIAG,
                 *RESFIT_EQUI_DIFF, *EQUI_RESFIT_ARCH, *EQUI_LONG,
                 *RESFIT_LN_PLACE}  # every other arm is tuned RLPD
# Stopped only at their collapse (or a diverged critic); they end at their own total_timesteps.
COLLAPSE_ONLY_ARMS = {*RESFIT_RX}
# To a fixed length (2026-10-12 AMENDMENT, Aden): a residual that loses the base policy can win it
# back late (eaqorqki: 0.04 at 50k, 0.30 at 80k), so only the divergence stop applies. Each ends at
# its own total_timesteps, or at its STOP_AT step.
FIXED_LENGTH_ARMS = {NOTILT, *RESFIT_EQUI_DIFF, *EQUI_RESFIT_ARCH, *EQUI_LONG, *RESFIT_LN_PLACE}
STOP_AT = {NOTILT: 150_000}  # launched for 300k
# Tracked and logged, never stopped: a run that must go through its collapse to its own end.
WATCH_ONLY_ARMS = {DIAG}
EPISODES = 50                       # per eval
COLLAPSE_RATE, COLLAPSE_RUN = 0.05, 3
CHECK_STEP = 100_000
B_SQ, B_SQ_EPISODES = 0.582, 500
RESIDUAL_WINDOW = (60_000, 100_000)
TUNED_MIN_RATE, TUNED_MIN_SUCCESSES = 0.04, 5
CONVERGE_FROM, CONVERGE_EVALS = 150_000, 5
Q_DIVERGED = 1.5
Q_KEYS = ("debug/q_data_demo", "debug/q_data_online", "debug/q_policy_demo", "debug/q_policy_online")
RUNAWAY_Q, RUNAWAY_STEPS = 10.0, 5_000


def collapsed(evals: list[tuple[int, float]]) -> str | None:
    run = 0
    for step, rate in evals:
        run = run + 1 if rate <= COLLAPSE_RATE else 0
        if run == COLLAPSE_RUN:
            return f"collapsed at {step}: {COLLAPSE_RUN} consecutive evals <= {COLLAPSE_RATE}"
    return None


def diverged(evals: list[tuple[int, float]], qs: list[tuple[int, str, float]]) -> str | None:
    """The divergence stop, at the latest eval: every row's largest critic value above RUNAWAY_Q
    over the RUNAWAY_STEPS before it, and the last COLLAPSE_RUN evals at or below COLLAPSE_RATE."""
    if len(evals) < COLLAPSE_RUN or any(rate > COLLAPSE_RATE for _, rate in evals[-COLLAPSE_RUN:]):
        return None
    step = evals[-1][0]
    rows: dict[int, float] = {}
    for at, _, value in qs:
        if step - RUNAWAY_STEPS < at <= step:
            rows[at] = max(rows.get(at, -math.inf), value)
    if rows and min(rows.values()) > RUNAWAY_Q:
        return (f"critic diverged by {step}: every critic value over the {RUNAWAY_STEPS} steps before "
                f"above {RUNAWAY_Q:g} (lowest {min(rows.values()):.1f}, {len(rows)} rows), and "
                f"{COLLAPSE_RUN} consecutive evals <= {COLLAPSE_RATE}")
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


def decide(arm: str, evals: list[tuple[int, float]], successes: int, passed_check: bool,
           qs: list[tuple[int, str, float]] = ()) -> tuple[str | None, str]:
    """("stop" | "passed" | "human" | None, why) for a trainer with this history now."""
    if arm in WATCH_ONLY_ARMS:
        return None, ""
    residual = arm in RESIDUAL_ARMS
    if residual and arm not in FIXED_LENGTH_ARMS and (why := collapsed(evals)):
        return "stop", why
    if why := diverged(evals, list(qs)):
        return "stop", why
    if arm in FIXED_LENGTH_ARMS:
        if arm in STOP_AT and evals and evals[-1][0] >= STOP_AT[arm]:
            return "stop", f"reached its fixed length, {STOP_AT[arm]}"
        return None, ""
    if arm in COLLAPSE_ONLY_ARMS:
        return None, ""
    if not evals or evals[-1][0] < CHECK_STEP:
        return None, ""
    if not passed_check:
        passed, why = residual_check(evals) if residual else tuned_check(evals, successes)
        if passed is None:
            return "human", f"100k check: {why}"
        return ("passed" if passed else "stop"), f"100k check {'passed' if passed else 'failed'}: {why}"
    if evals[-1][0] >= CONVERGE_FROM and len(evals) >= 2 * CONVERGE_EVALS:
        done, why = converged(evals)
        if done:
            return "stop", f"converged at {evals[-1][0]} (no longer improving): {why}"
    return None, ""


def history(package: Path) -> tuple[list[tuple[int, float]], list[tuple[int, float]], list[tuple[int, str, float]]]:
    """Evals, training episode returns, and critic values, by step, from the live record."""
    manifest = json.loads((package / "manifest.json").read_text())
    rows = _history_from_record(Path(manifest["wandb"]["dir"]).parent / f"run-{package.name}.wandb")
    evals = sorted((int(r["_step"]), float(r["eval/success_rate"])) for r in rows
                   if r.get("eval/success_rate") is not None)
    returns = [(int(r["_step"]), float(r["training/episode_return"])) for r in rows
               if r.get("training/episode_return") is not None]
    qs = [(int(r["_step"]), key, float(r[key])) for r in rows for key in Q_KEYS if r.get(key) is not None]
    return evals, returns, qs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--group", required=True, choices=sorted(GROUPS))
    group = parser.parse_args().group
    prefix = f"{group}/"

    def say(message: str) -> None:
        log(f"[{group}] {message}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    tracked: dict[int, dict] = {}
    ignored: set[int] = set()
    say(f"started (pid {os.getpid()}): arms {', '.join(GROUPS[group])}")
    while True:
        live = trainers(prefix)
        for pid, trainer in live.items():
            if pid in ignored:
                continue
            if pid not in tracked:
                package = package_of(trainer)
                if package is None:
                    continue  # wandb not up yet
                arm = _arg(trainer["argv"], "wandb.notes").split("/")[1]
                if arm not in GROUPS[group]:
                    say(f"IGNORED pid {pid}: arm {arm!r} is not one of this group's")
                    ignored.add(pid)
                    continue
                tracked[pid] = {"package": package, "arm": arm, "signalled": None, "passed": False,
                                "q_flagged": False, "human": False, "last_eval": None}
                say(f"TRACK {package.name} (pid {pid}, {_arg(trainer['argv'], 'wandb.name')})")
            entry = tracked[pid]
            if entry["signalled"] is not None:
                if time.time() - entry["signalled"] > ESCALATE_S:
                    say(f"SIGTERM {entry['package'].name} (pid {pid}): still alive after SIGINT")
                    os.kill(pid, signal.SIGTERM)
                    entry["signalled"] = time.time()
                continue
            try:
                evals, returns, qs = history(entry["package"])
            except Exception as exc:  # the live record can be mid-write; try next minute
                say(f"READ FAILED {entry['package'].name}: {type(exc).__name__}: {exc}")
                continue
            name = f"{entry['package'].name} ({entry['arm']})"
            if not entry["q_flagged"] and (high := [q for q in qs if q[2] > Q_DIVERGED]):
                step, key, value = high[0]
                say(f"FLAG {name}: {key} {value:.2f} at {step}, above {Q_DIVERGED}: report to Aden")
                entry["q_flagged"] = True
            if not evals or evals[-1][0] == entry["last_eval"]:
                continue
            entry["last_eval"] = evals[-1][0]
            successes = sum(ret > 0 for step, ret in returns if step <= CHECK_STEP)
            action, why = decide(entry["arm"], evals, successes, entry["passed"], qs)
            rates = " ".join(f"{rate:.2f}" for _, rate in evals)
            say(f"EVAL {name} at {evals[-1][0]}: {evals[-1][1]:.2f}; training successes to "
                f"{CHECK_STEP} {successes}")
            if action == "passed":
                entry["passed"] = True
                say(f"PASSED {name}: {why}; due its seed 2 (a human launches it); evals {rates}")
            elif action == "human" and not entry["human"]:
                entry["human"] = True
                say(f"HUMAN {name}: {why}; not stopped; evals {rates}")
            elif action == "stop":
                say(f"STOP {name} (pid {pid}): {why}; evals {rates}")
                os.kill(pid, signal.SIGINT)
                entry["signalled"] = time.time()

        for pid in [p for p in tracked if p not in live]:
            package = tracked.pop(pid)["package"]
            manifest = json.loads((package / "manifest.json").read_text())
            if manifest.get("completed"):
                say(f"EXITED {package.name}: finished, package complete")
            else:
                finalize_run_package(package)
                say(f"EXITED {package.name}: package finalized by the watcher")
        time.sleep(POLL_S)


if __name__ == "__main__":
    sys.exit(main())
