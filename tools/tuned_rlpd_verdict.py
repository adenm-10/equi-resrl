"""The stage-1 verdict of a tuned-RLPD run, by the rule pre-registered on 2026-10-04.

    CACHE_DIR=~/projects/equi-resrl python tools/tuned_rlpd_verdict.py --project robomimic-square-final <id>
    python tools/tuned_rlpd_verdict.py --project robomimic-can-final --json <id>

The rule is pre-registered in docs/EXPERIMENTS.md, "2026-10-04 — PRE-REGISTERED — tuned RLPD on
Can and Square" (seed-group tuned-rlpd-1002). Tuned RLPD is vanilla RL with no base policy, so a
run starts near zero. It passes iff the mean of its five evals at 110k-150k is at least its
task's BC base rate, B_sq = 0.508 (2026-09-29 MEASUREMENT) or B_can = 0.894 (2026-10-04
MEASUREMENT): it has learned at least what behavior cloning learns from the same demos. The
paper's ~0.9 on Square is context, not the bar. There is no collapse rule, since a run that
starts at zero has nothing to collapse from. A run without all five evals has no verdict yet.

Also reported, deciding nothing: the step-0, best and final evals (STANDARDS 5.5), and every
eval. History comes from the run's local wandb record, as for tools/square_verdict.py.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from square_verdict import load_history

WINDOW = tuple(range(110_000, 150_001, 10_000))  # the steps of the five evals the rule averages
BASE_RATES = {  # project: (task, its BC base rate)
    "robomimic-square-final": ("Square", 0.508),  # 2026-09-29 MEASUREMENT
    "robomimic-can-final": ("Can", 0.894),        # 2026-10-04 MEASUREMENT
}


def verdict(rows: list[dict], base_rate: float) -> dict:
    """The rule and the reported numbers, from a run's history rows."""
    evals = sorted((int(r["_step"]), float(r["eval/success_rate"])) for r in rows
                   if r.get("eval/success_rate") is not None)
    rate_at = dict(evals)
    out: dict = {"evals": evals, "base_rate": base_rate, "step0": rate_at.get(0),
                 "best": max(evals, key=lambda e: e[1]) if evals else None,
                 "final": evals[-1] if evals else None, "window_mean": None}
    window = [rate_at[step] for step in WINDOW if step in rate_at]
    if len(window) < len(WINDOW):
        out["passes"] = None
        out["why"] = f"{len(window)} of the {len(WINDOW)} evals at 110k-150k so far"
        return out
    out["window_mean"] = float(np.mean(window))
    out["passes"] = out["window_mean"] >= base_rate
    out["why"] = (f"110k-150k mean {out['window_mean']:.3f} "
                  f"{'>=' if out['passes'] else '<'} base rate {base_rate:.3f}")
    return out


def describe(run_id: str, v: dict) -> str:
    def at(e):
        return "-" if e is None else f"{e[1]:.2f} at {e[0]}"

    step0 = "-" if v["step0"] is None else f"{v['step0']:.2f}"
    verdict_text = {True: "PASSES", False: "does not pass", None: "no verdict"}[v["passes"]]
    return (f"{run_id}: {verdict_text} ({v['why']})\n"
            f"  step 0 {step0}, best {at(v['best'])}, final {at(v['final'])}\n"
            f"  evals " + " ".join(f"{rate:.2f}" for _, rate in v["evals"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--project", required=True, choices=sorted(BASE_RATES),
                        help="the runs' wandb project, which names their task")
    parser.add_argument("runs", nargs="+", help="wandb run ids")
    parser.add_argument("--json", action="store_true", help="print the verdicts as JSON")
    args = parser.parse_args()
    _, base_rate = BASE_RATES[args.project]
    verdicts = {run_id: verdict(load_history(run_id, project=args.project), base_rate)
                for run_id in args.runs}
    if args.json:
        print(json.dumps(verdicts, indent=2))
    else:
        print("\n".join(describe(run_id, v) for run_id, v in verdicts.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
