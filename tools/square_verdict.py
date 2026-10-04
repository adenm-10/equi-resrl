"""The stage-1 verdict of a Square ablation run, by the rule of 2026-10-01.

    CACHE_DIR=~/projects/equi-resrl python tools/square_verdict.py 0rehtuoq konwqk6l
    python tools/square_verdict.py --json konwqk6l

The rule is pre-registered in docs/EXPERIMENTS.md, "2026-10-01 — AMENDMENT — stop at 100k, judge
by the training trend, and three normalization arms". Each finished training episode logs one
`training/episode_return` row at the step it ended (num_envs = 1). The reward is 1 on success
and 0 otherwise, so the return is gamma**steps on success and 0 on failure: success = return > 0.

A run advances iff all three hold:
  (a) the least-squares slope of success on step, over episodes ending in (10k, 100k], is at
      least 2 standard errors above zero;
  (b) the success fraction of episodes ending in (70k, 100k] is above that of episodes ending in
      [0, 10k], the run's own start under the same exploration noise;
  (c) it did not collapse: no three consecutive evals at or below 0.05.
A run with no 100k eval, that has not collapsed, has no verdict yet.

Also reported, deciding nothing: the episode-length slope, the eval mean over 60k-100k, and the
evals. Data after 100k is ignored, so runs that went on to 150k or 300k are judged on their
first 100k.

History comes from the run's local wandb record: the finalized package's copy under
CACHE_DIR/outputs/runs/<project>/<id>/wandb_logs/, else the live record its manifest points to.
Not from the API, whose full-history scan drops the final row.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT = "robomimic-square-final"
STOP = 100_000
TREND_START = 10_000        # (a): episodes ending in (TREND_START, STOP]
LATE_START = 70_000         # (b): episodes ending in (LATE_START, STOP] ...
EARLY_END = 10_000          #      ... against those ending in [0, EARLY_END]
MIN_Z = 2.0
COLLAPSE_RATE, COLLAPSE_RUN = 0.05, 3
EVAL_WINDOW = (60_000, 100_000)
B_SQ = 0.508                # Square's base rate, 2026-09-29 MEASUREMENT


def load_history(run_id: str, runs_root: Path | None = None, project: str = PROJECT) -> list[dict]:
    """Every history row of a run, from its local wandb record."""
    if runs_root is None:
        from resfit.lerobot.utils.load_policy import RUNS_ROOT
        runs_root = RUNS_ROOT
    package = runs_root / project / run_id
    saved = package / "wandb_logs" / "metrics.jsonl"
    if saved.exists():
        return [json.loads(line) for line in saved.read_text().splitlines() if line]
    from resfit.lerobot.utils.load_policy import _history_from_record
    manifest = json.loads((package / "manifest.json").read_text())
    return _history_from_record(Path(manifest["wandb"]["dir"]).parent / f"run-{run_id}.wandb")


def evals_of(rows: list[dict]) -> list[tuple[int, float]]:
    """(step, success rate) of every eval up to STOP, in step order."""
    return sorted((int(r["_step"]), float(r["eval/success_rate"])) for r in rows
                  if r.get("eval/success_rate") is not None and r["_step"] <= STOP)


def collapse_step(evals: list[tuple[int, float]]) -> int | None:
    """The step of the eval that completes the first collapse, or None."""
    run = 0
    for step, rate in evals:
        run = run + 1 if rate <= COLLAPSE_RATE else 0
        if run == COLLAPSE_RUN:
            return step
    return None


def _ols_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least-squares slope of y on x and its standard error, from the residual variance."""
    x_c = x - x.mean()
    sxx = float(x_c @ x_c)
    slope = float(x_c @ (y - y.mean())) / sxx
    residual = y - y.mean() - slope * x_c
    return slope, math.sqrt(float(residual @ residual) / (len(y) - 2) / sxx)


def verdict(rows: list[dict]) -> dict:
    """Rules (a)-(c) and the reported numbers, from a run's history rows."""
    episodes = np.array([(r["_step"], r["training/episode_return"],
                          float(np.mean(r["training/episode_steps"])))
                         for r in rows if r.get("training/episode_return") is not None],
                        dtype=float).reshape(-1, 3)
    step, success, length = episodes[:, 0], (episodes[:, 1] > 0).astype(float), episodes[:, 2]
    evals = evals_of(rows)
    out: dict = {"episodes": int((step <= STOP).sum()), "evals": evals,
                 "collapsed_at": collapse_step(evals)}

    trend = (step > TREND_START) & (step <= STOP)
    if trend.sum() > 2:
        x = step[trend] / 1e5   # slopes are per 100k steps
        out["slope"], out["slope_se"] = _ols_slope(x, success[trend])
        out["length_slope"], out["length_slope_se"] = _ols_slope(x, length[trend])
    early, late = step <= EARLY_END, (step > LATE_START) & (step <= STOP)
    out["early_success"] = float(success[early].mean()) if early.any() else None
    out["late_success"] = float(success[late].mean()) if late.any() else None
    window = [rate for s, rate in evals if EVAL_WINDOW[0] <= s <= EVAL_WINDOW[1]]
    out["eval_mean_60k_100k"] = float(np.mean(window)) if window else None

    finished = any(s == STOP for s, _ in evals)
    if out["collapsed_at"] is not None:
        out["advances"], out["why"] = False, f"collapsed at {out['collapsed_at']}"
    elif not finished or "slope" not in out:
        out["advances"], out["why"] = None, "no 100k eval yet"
    else:
        z = out["slope"] / out["slope_se"]
        rising = z >= MIN_Z
        above_start = (out["late_success"] or 0.0) > (out["early_success"] or 0.0)
        out["advances"] = bool(rising and above_start)
        out["why"] = (f"slope z {z:+.1f} ({'>=' if rising else '<'} {MIN_Z}); late success "
                      f"{out['late_success']:.2f} {'>' if above_start else '<='} start "
                      f"{out['early_success']:.2f}")
    return out


def top_arms(critic_fn_advances: bool, clipq_advances: bool) -> tuple[str, str]:
    """The squash and depth arms on the base the two verdicts give (2026-09-30 table)."""
    base = "-".join(name for name, advanced in (("critic-fn", critic_fn_advances),
                                                ("clipq", clipq_advances)) if advanced)
    return tuple(f"{base}-{top}" if base else top for top in ("squash", "depth"))


def describe(run_id: str, v: dict) -> str:
    slope = "no slope"
    if "slope" in v:
        slope = (f"slope {v['slope']:+.2f} ± {v['slope_se']:.2f} /100k, length "
                 f"{v['length_slope']:+.0f} ± {v['length_slope_se']:.0f}")
    early = "-" if v["early_success"] is None else f"{v['early_success']:.2f}"
    late = "-" if v["late_success"] is None else f"{v['late_success']:.2f}"
    window = "-" if v["eval_mean_60k_100k"] is None else f"{v['eval_mean_60k_100k']:.3f}"
    verdict_text = {True: "ADVANCES", False: "does not advance", None: "no verdict"}[v["advances"]]
    return (f"{run_id}: {verdict_text} ({v['why']})\n"
            f"  {v['episodes']} episodes; {slope}; success start {early}, late {late}\n"
            f"  eval 60k-100k {window} (B_sq {B_SQ}); evals "
            + " ".join(f"{rate:.2f}" for _, rate in v["evals"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="+", help="wandb run ids")
    parser.add_argument("--json", action="store_true", help="print the verdicts as JSON")
    args = parser.parse_args()
    verdicts = {run_id: verdict(load_history(run_id)) for run_id in args.runs}
    if args.json:
        print(json.dumps(verdicts, indent=2))
    else:
        print("\n".join(describe(run_id, v) for run_id, v in verdicts.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
