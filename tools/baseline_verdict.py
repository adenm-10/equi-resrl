"""The verdicts of the fixed-env baselines, by the rule pre-registered on 2026-10-07.

    CACHE_DIR=~/projects/equi-resrl python tools/baseline_verdict.py          # every arm, so far
    CACHE_DIR=~/projects/equi-resrl python tools/baseline_verdict.py \\
        [--b-can PATH] [--b-sq PATH] [resfit-tuned-rlpd-can=<id>,<id> ...] [--json]

With no arms named it finds the seed-group's runs itself, from the notes their run packages were
launched with, and reads the bars from the measurements the pre-registration names. Run it while
the runs train: a live run's history is read from its live wandb record.

The rule is pre-registered in docs/EXPERIMENTS.md, "2026-10-07 — PRE-REGISTERED — the fixed-env
baselines" (seed-group fixed-env-baselines-1007). An arm is judged on both its seeds pooled: p is
the mean of the ten evals at 110k-150k, five per seed, 500 episodes.
  - Tuned RLPD (vanilla RL, on Can) passes iff p >= B_can: it has learned at least what behavior
    cloning learns from the same demos.
  - A residual arm (on Square) passes iff p - B_sq >= 2 sqrt(p(1-p)/500 + B_sq(1-B_sq)/n_B), both
    binomial: it beats the base policy it corrects by two standard errors of the difference.
The bars are the 500-episode measurements of eval_base_policy on mujoco 3.3.2, read from their
JSON records. An arm without both seeds' five window evals, or without its bar, has no verdict
yet; its evals so far are reported.

Also reported, deciding nothing: each run's step-0, best and final evals (STANDARDS 5.5), every
eval, and for the residual arms whether a run collapsed by the rule of 2026-10-01 (three
consecutive evals at or below 0.05), which compares the two inits of the equivariant residual.
History comes from each run's local wandb record, as for tools/square_verdict.py.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from square_verdict import load_history

SEED_GROUP = "fixed-env-baselines-1007"
WINDOW = tuple(range(110_000, 150_001, 10_000))  # the steps of the five evals per seed the rule pools
SEEDS = 2
EPISODES_PER_EVAL = 50
MIN_Z = 2.0
COLLAPSE_RATE, COLLAPSE_RUN = 0.05, 3            # the 2026-10-01 rule, reported only
# arm: (method, wandb project), as submit.sh launches it.
ARMS = {
    "resfit-tuned-rlpd-can": ("tuned-rlpd", "robomimic-can-final"),
    "equi-tuned-rlpd-can": ("tuned-rlpd", "robomimic-can-final"),
    "resfit-residual-square": ("residual", "robomimic-square-final"),
    "equi-residual-square": ("residual", "robomimic-square-final"),
    "equi-residual-square-zero-init": ("residual", "robomimic-square-final"),
}
# The bars' measurements, under outputs/base_rates/ (eval_base_policy's names, base policies pinned).
BARS = {"tuned-rlpd": "Can__xhjdl8a7_best_v7__seed1000.json",
        "residual": "Square__3hzs5bz1_best_v9__seed1000.json"}


def read_bar(path: Path) -> tuple[float, int]:
    """A base-rate record's mean and its episode count."""
    record = json.loads(Path(path).read_text())
    return float(record["mean"]), len(record["rates"]) * int(record["episodes_per_eval"])


def evals_of(rows: list[dict]) -> list[tuple[int, float]]:
    return sorted((int(r["_step"]), float(r["eval/success_rate"])) for r in rows
                  if r.get("eval/success_rate") is not None)


def collapsed(evals: list[tuple[int, float]]) -> bool:
    low = 0
    for _, rate in evals:
        low = low + 1 if rate <= COLLAPSE_RATE else 0
        if low >= COLLAPSE_RUN:
            return True
    return False


def run_summary(rows: list[dict]) -> dict:
    evals = evals_of(rows)
    rate_at = dict(evals)
    return {"evals": evals, "step0": rate_at.get(0),
            "best": max(evals, key=lambda e: e[1]) if evals else None,
            "final": evals[-1] if evals else None,
            "window": [rate_at[s] for s in WINDOW if s in rate_at],
            "collapsed": collapsed(evals)}


def find_runs(seed_group: str, runs_root: Path) -> dict[str, list[str]]:
    """Each arm's runs in the seed-group, by seed, from the notes in their manifests' argv."""
    found: dict[str, list[tuple[int, str]]] = {arm: [] for arm in ARMS}
    for arm, (_, project) in ARMS.items():
        prefix = f"wandb.notes={seed_group}/{arm}/seed"
        for manifest in sorted((runs_root / project).glob("*/manifest.json")):
            for arg in json.loads(manifest.read_text()).get("argv", []):
                if arg.startswith(prefix):
                    found[arm].append((int(arg[len(prefix):]), manifest.parent.name))
    return {arm: [run_id for _, run_id in sorted(runs)] for arm, runs in found.items()}


def arm_verdict(arm: str, runs: dict[str, list[dict]], bar: tuple[float, int] | None) -> dict:
    """The rule and the reported numbers for one arm, from its runs' history rows.

    `bar` is None until its measurement exists."""
    method, _ = ARMS[arm]
    b, n_b = bar if bar is not None else (None, 0)
    out: dict = {"arm": arm, "method": method, "bar": b, "bar_episodes": n_b,
                 "runs": {run_id: run_summary(rows) for run_id, rows in runs.items()},
                 "p": None, "episodes": 0, "margin": None}
    windows = [s["window"] for s in out["runs"].values()]
    have = sum(len(w) for w in windows)
    if len(runs) < SEEDS or any(len(w) < len(WINDOW) for w in windows) or bar is None:
        out["passes"] = None
        out["why"] = (f"{have} of the {SEEDS * len(WINDOW)} evals at 110k-150k so far, "
                      f"{len(runs)} of {SEEDS} seeds" + ("" if bar else "; bar not measured yet"))
        return out
    pooled = [rate for w in windows for rate in w]
    p = float(np.mean(pooled))
    n = EPISODES_PER_EVAL * len(pooled)
    out.update(p=p, episodes=n)
    if method == "tuned-rlpd":
        out["passes"] = p >= b
        out["why"] = f"110k-150k mean {p:.3f} {'>=' if out['passes'] else '<'} B_can {b:.3f}"
    else:
        se = math.sqrt(p * (1 - p) / n + b * (1 - b) / n_b)
        out["margin"] = MIN_Z * se
        out["passes"] = p - b >= out["margin"]
        out["why"] = (f"110k-150k mean {p:.3f}, {p - b:+.3f} on B_sq {b:.3f}, "
                      f"{'>=' if out['passes'] else '<'} the {out['margin']:.3f} of 2 SE")
    return out


def describe(v: dict) -> str:
    def at(e):
        return "-" if e is None else f"{e[1]:.2f} at {e[0] // 1000}k"

    text = {True: "PASSES", False: "does not pass", None: "no verdict yet"}[v["passes"]]
    bar = "" if v["bar"] is None else f"; bar {v['bar']:.3f}"
    lines = [f"{v['arm']}: {text} ({v['why']}){bar}"]
    for run_id, s in v["runs"].items():
        step0 = "-" if s["step0"] is None else f"{s['step0']:.2f}"
        lines.append(f"  {run_id}: step 0 {step0}, best {at(s['best'])}, final {at(s['final'])}"
                     + (", COLLAPSED" if v["method"] == "residual" and s["collapsed"] else ""))
        lines.append("    evals " + " ".join(f"{rate:.2f}" for _, rate in s["evals"]))
    return "\n".join(lines)


def parse_arm(text: str) -> tuple[str, list[str]]:
    arm, _, ids = text.partition("=")
    if arm not in ARMS or not ids:
        raise argparse.ArgumentTypeError(f"expected <arm>=<id>[,<id>], arm one of {sorted(ARMS)}")
    return arm, ids.split(",")


def main() -> int:
    from resfit.lerobot.utils.load_policy import RUNS_ROOT

    base_rates = RUNS_ROOT.parent / "base_rates"
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--b-can", type=Path, default=base_rates / BARS["tuned-rlpd"],
                        help="Can's base-rate JSON")
    parser.add_argument("--b-sq", type=Path, default=base_rates / BARS["residual"],
                        help="Square's base-rate JSON")
    parser.add_argument("arms", nargs="*", type=parse_arm,
                        help="<arm>=<run id>,<run id>; default: the seed-group's runs, found")
    parser.add_argument("--json", action="store_true", help="print the verdicts as JSON")
    args = parser.parse_args()
    bars = {method: read_bar(path) if path.exists() else None
            for method, path in (("tuned-rlpd", args.b_can), ("residual", args.b_sq))}
    arms = args.arms or list(find_runs(SEED_GROUP, RUNS_ROOT).items())
    verdicts = []
    for arm, run_ids in arms:
        method, project = ARMS[arm]
        runs = {run_id: load_history(run_id, project=project) for run_id in run_ids}
        verdicts.append(arm_verdict(arm, runs, bars[method]))
    if args.json:
        print(json.dumps(verdicts, indent=2))
    else:
        print("\n".join(describe(v) for v in verdicts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
