"""Whether each named arm of a seed group holds, by resfit-equi-diff-1012's rule, at its 10k-30k evals.

    python tools/arms_hold.py resfit-equi-diff-1012 1 resfit-residual-square-rerendered \
        resfit-residual-square-noln resfit-residual-square-limits

A ResFiT arm breaks iff its 10k, 20k and 30k evals pooled are at most 39 of 150 (below the
anchors' pooled 0.353 by two standard errors), or three evals in a row are at or below 0.05
(EXPERIMENTS.md, 2026-10-12 PRE-REGISTERED and its AMENDMENT "every run to a fixed 150k"). It holds
otherwise. The queue launches `combined` iff rerendered, noln and limits all hold: exit 0 iff every
arm holds, 1 if any breaks, 2 if any has no run package or no 30k eval yet, which the queue reads as
"skip". The run is found by its notes, `<group>/<arm>/seed<seed>`, in the run packages under
CACHE_DIR; its evals are read from its live wandb record, as the watcher reads them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from square_rerendered_watch import EPISODES, collapsed, history  # noqa: E402

from resfit.lerobot.utils.load_policy import RUNS_ROOT  # noqa: E402

BREAKS_AT = 39  # of the 150 episodes at 10k-30k
READOUT = (10_000, 20_000, 30_000)


def package(group: str, arm: str, seed: str) -> Path | None:
    """The newest run package whose launch carries these notes."""
    notes = f"wandb.notes={group}/{arm}/seed{seed}"
    found = [m.parent for m in RUNS_ROOT.glob("*/*/manifest.json") if notes in json.loads(m.read_text()).get("argv", [])]
    return max(found, key=lambda p: json.loads((p / "manifest.json").read_text())["started"], default=None)


def holds(evals: list[tuple[int, float]]) -> tuple[bool | None, str]:
    """(True | False | None, why): None if the 30k eval is not in yet."""
    by_step = dict(evals)
    if any(step not in by_step for step in READOUT):
        return None, f"no eval at each of {READOUT} yet"
    k = round(sum(by_step[step] for step in READOUT) * EPISODES)
    if why := collapsed(evals):
        return False, f"{k} of 150 at 10k-30k; {why}"
    return k > BREAKS_AT, f"{k} of 150 at 10k-30k ({'holds' if k > BREAKS_AT else 'breaks'}: breaks at {BREAKS_AT} or fewer)"


def main(argv: list[str]) -> int:
    group, seed, *arms = argv
    verdicts = []
    for arm in arms:
        found = package(group, arm, seed)
        if found is None:
            print(f"{arm}: no run package with notes {group}/{arm}/seed{seed}")
            verdicts.append(None)
            continue
        ok, why = holds(history(found)[0])
        print(f"{arm} ({found.name}): {why}")
        verdicts.append(ok)
    if None in verdicts:
        return 2
    return 0 if all(verdicts) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
