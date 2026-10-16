"""Whether an arm of equi-arch-parts-1015 is due its seed 3, by that group's rule as the 2026-10-16
DEVIATION leaves it.

    CACHE_DIR=~/projects/equi-resrl python tools/parts_seed3.py equi-residual-square-actorln [--wait-hours 8]

The rule (EXPERIMENTS.md, 2026-10-15 PRE-REGISTERED): seed 3 of each arm better than init0 at 150k,
the two highest if more than two qualify. Better than init0 is the 110k-150k evals pooled (250
episodes) above oav5gk76's 83 of 250 (0.332) by two standard errors, sqrt(p(1-p)/250 +
0.332*0.668/250): at least 105 of 250. The 2026-10-16 DEVIATION (Aden) leaves arch-wide out, so the
candidates are the four parts alone, each at seed 2. An arm is due iff it is better than init0 and
fewer than two other candidates better than init0 score higher (a tie for second launches both).

Exit 0 iff the arm is due, 1 if not, and 2 if it cannot be decided: a candidate has no run package,
or its run ended without an eval at each of 110k-150k. slot_queue.sh reads anything but 0 as "skip".
With --wait-hours, a candidate whose trainer is still going is waited for, polling every 10 minutes;
past that, 2. The run is found by its notes, as tools/arms_hold.py finds it, and its evals are read
from its live wandb record, as the watcher reads them.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from arms_hold import package  # noqa: E402
from run_watch import trainers  # noqa: E402
from square_rerendered_watch import EPISODES, history  # noqa: E402

GROUP, SEED = "equi-arch-parts-1015", "2"
CANDIDATES = ("equi-residual-square-actorln", "equi-residual-square-criticln",
              "equi-residual-square-heads", "equi-residual-square-actor448")
READOUT = (110_000, 120_000, 130_000, 140_000, 150_000)
INIT0_K, N = 83, EPISODES * len(READOUT)  # oav5gk76 at 110k-150k
MAX_SEED3 = 2
POLL_S = 600


def better_than_init0(k: int) -> bool:
    p, p0 = k / N, INIT0_K / N
    return p - p0 > 2 * math.sqrt(p * (1 - p) / N + p0 * (1 - p0) / N)


def due(arm: str, late: dict[str, int]) -> bool:
    """`late`: each candidate's successes of 250 at 110k-150k."""
    if not better_than_init0(late[arm]):
        return False
    higher = [other for other, k in late.items() if other != arm and better_than_init0(k) and k > late[arm]]
    return len(higher) < MAX_SEED3


def late_successes(arm: str) -> tuple[int | None, str]:
    """(successes at 110k-150k, why), or (None, why) while it cannot be read yet or at all."""
    found = package(GROUP, arm, SEED)
    if found is None:
        return None, "no run package"
    by_step = dict(history(found)[0])
    if any(step not in by_step for step in READOUT):
        return None, f"{found.name}: evals to {max(by_step, default=0)}"
    return round(sum(by_step[step] for step in READOUT) * EPISODES), found.name


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("arm", choices=CANDIDATES)
    parser.add_argument("--wait-hours", type=float, default=0.0)
    args = parser.parse_args(argv)
    deadline = time.time() + 3600 * args.wait_hours
    while True:
        late, waiting, stuck = {}, [], []
        for arm in CANDIDATES:
            k, why = late_successes(arm)
            if k is not None:
                late[arm] = k
                print(f"{arm} ({why}): {k} of {N} at 110k-150k, "
                      f"{'better than init0' if better_than_init0(k) else 'not better than init0'}")
            elif trainers(f"{GROUP}/{arm}/seed{SEED}"):
                waiting.append(f"{arm} ({why}, still going)")
            else:
                stuck.append(f"{arm} ({why}, not running)")
        if stuck:
            print(f"cannot decide: {'; '.join(stuck)}")
            return 2
        if not waiting:
            verdict = due(args.arm, late)
            print(f"{args.arm}: {'due' if verdict else 'not due'} its seed 3")
            return 0 if verdict else 1
        if time.time() >= deadline:
            print(f"cannot decide yet: {'; '.join(waiting)}")
            return 2
        time.sleep(POLL_S)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
