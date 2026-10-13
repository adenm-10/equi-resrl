"""tools/square_rerendered_watch.py applies square-rerendered-1009's rules as pre-registered and amended,
square-trlpd-stab-1010's, which are its tuned-RLPD rules, square-residual-start-1011's, which are
its residual rules, resfit-rx-1012's, which are the residual's collapse stop alone, and
residual-tilt-1012's: notilt to a fixed 150k (2026-10-12 AMENDMENT), and none for the diagnostic run.

Checked on real eval sequences where one exists: oj5k0jqb (ResFiT's residual on Square, 0.800 over
110k-150k), kyv7clfc (the fixed-env baselines' equivariant residual, 0.00 from 10k), and vc5o977f
(equivariant tuned RLPD on Square, whose critic diverged from about 92k), whose whole record is
replayed when its run package is on this machine.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from square_rerendered_watch import (  # noqa: E402
    GROUPS,
    Q_DIVERGED,
    Q_KEYS,
    DIAG,
    NOTILT,
    EQUI_LONG,
    EQUI_RESFIT_ARCH,
    RESFIT_EQUI_DIFF,
    RESFIT_RX,
    RESIDUAL,
    RESIDUAL_STARTS,
    STABILIZERS,
    TUNED,
    decide,
)

from resfit.lerobot.utils.load_policy import RUNS_ROOT  # noqa: E402

VC5O977F = RUNS_ROOT / "robomimic-square-final/vc5o977f"  # under CACHE_DIR: a launch worktree has no outputs/

OJ5K0JQB = [0.50, 0.30, 0.58, 0.54, 0.40, 0.54, 0.56, 0.64, 0.70, 0.64, 0.76, 0.72, 0.84, 0.90, 0.66, 0.88]
KYV7CLFC = [0.60, 0.00, 0.00, 0.02, 0.02, 0.00, 0.00]
S8OPBLSD = [0.68, 0.22, 0.28, 0.20, 0.40, 0.44, 0.32, 0.44, 0.68, 0.54, 0.48, 0.60, 0.68, 0.66, 0.58, 0.62]


def evals(rates):
    return [(10_000 * i, rate) for i, rate in enumerate(rates)]


def test_the_residual_is_stopped_at_its_collapse_and_tuned_rlpd_is_not():
    action, why = decide(RESIDUAL, evals(KYV7CLFC[:4]), 0, False)
    assert action == "stop" and "collapsed at 30000" in why
    assert decide(RESIDUAL, evals(KYV7CLFC[:3]), 0, False)[0] is None
    assert decide(TUNED, evals([0.0] * 10), 0, False)[0] is None


def test_the_residuals_100k_check_is_one_standard_error_over_b_sq():
    # oj5k0jqb's 60k-100k evals are 165 of 250: it passes.
    assert decide(RESIDUAL, evals(OJ5K0JQB[:11]), 0, False)[0] == "passed"
    # 155 of 250 passes, 154 does not; the base policy held (0.58 x 5, 145) fails.
    base = [0.58] * 6
    assert decide(RESIDUAL, evals(base + [0.62, 0.62, 0.62, 0.62, 0.62]), 0, False)[0] == "passed"
    assert decide(RESIDUAL, evals(base + [0.62, 0.62, 0.62, 0.62, 0.60]), 0, False)[0] == "stop"
    assert decide(RESIDUAL, evals([0.58] * 11), 0, False)[0] == "stop"
    # Not before its 100k eval is logged.
    assert decide(RESIDUAL, evals([0.58] * 10), 0, False)[0] is None


def test_a_100k_window_without_five_evals_is_left_to_a_human():
    gappy = [(s, 0.6) for s in (0, 10_000, 60_000, 70_000, 100_000)]
    assert decide(RESIDUAL, gappy, 0, False)[0] == "human"


def test_tuned_rlpds_100k_check_is_an_eval_of_004_or_five_training_successes():
    zeros = [0.0] * 11
    assert decide(TUNED, evals(zeros), 4, False)[0] == "stop"
    assert decide(TUNED, evals(zeros), 5, False)[0] == "passed"
    assert decide(TUNED, evals([0.0] * 5 + [0.04] + [0.0] * 5), 0, False)[0] == "passed"


def test_convergence_from_150k_is_the_last_five_not_beating_the_five_before_by_two_se():
    # oj5k0jqb at 150k: 0.80 against 0.66, 2 SE 0.078; still rising.
    assert decide(RESIDUAL, evals(OJ5K0JQB), 0, True)[0] is None
    flat = [0.70] * 16
    action, why = decide(RESIDUAL, evals(flat), 0, True)
    assert action == "stop" and "converged at 150000" in why
    assert decide(TUNED, evals([0.0] * 16), 5, True)[0] == "stop"
    # Not before 150k.
    assert decide(RESIDUAL, evals([0.70] * 15), 0, True)[0] is None


def _qs(start: int, stop: int, value: float, every: int = 100):
    return [(step, key, value) for step in range(start, stop + 1, every) for key in Q_KEYS]


def test_a_diverged_critic_stops_the_run_once_three_evals_are_at_zero():
    peaked = [0.0, 0.0, 0.06, 0.2, 0.26, 0.28, 0.3, 0.44, 0.5, 0.24]
    runaway = _qs(0, 99_900, 0.5) + _qs(100_000, 130_000, 50.0)
    for arm in (TUNED, *STABILIZERS):
        # 100k and 110k at zero: two evals, not three.
        assert decide(arm, evals(peaked + [0.0, 0.0]), 104, True, runaway)[0] is None
        action, why = decide(arm, evals(peaked + [0.0, 0.0, 0.0]), 104, True, runaway)
        assert action == "stop" and "critic diverged by 120000" in why
    # Every row in the 5k before the eval must be above 10: one row at 9 holds it off.
    dip = [q for q in runaway if q[0] != 118_000] + [(118_000, key, 9.0) for key in Q_KEYS]
    assert decide(TUNED, evals(peaked + [0.0] * 3), 104, True, dip)[0] is None
    # A runaway critic whose evals still score is left running.
    assert decide(TUNED, evals(peaked + [0.0, 0.0, 0.1]), 104, True, runaway)[0] is None
    # The largest of the four values in a row counts: one key above 10 is enough.
    one_key = [(step, Q_KEYS[3], 50.0) for step in range(100_000, 130_001, 100)]
    assert decide(TUNED, evals(peaked + [0.0] * 3), 104, True, one_key)[0] == "stop"


def test_the_stabilizer_arms_follow_tuned_rlpds_rules():
    assert GROUPS["square-trlpd-stab-1010"] == STABILIZERS and len(STABILIZERS) == 4
    for arm in STABILIZERS:
        assert decide(arm, evals([0.0] * 10), 0, False)[0] is None      # never stopped for collapse
        assert decide(arm, evals([0.0] * 11), 4, False)[0] == "stop"    # fails its 100k check
        assert decide(arm, evals([0.0] * 11), 5, False)[0] == "passed"
        assert decide(arm, evals([0.0] * 16), 5, True)[0] == "stop"     # converged


def test_the_residual_start_arms_follow_the_residuals_rules():
    assert GROUPS["square-residual-start-1011"] == RESIDUAL_STARTS and len(RESIDUAL_STARTS) == 2
    for arm in RESIDUAL_STARTS:
        assert decide(arm, evals(KYV7CLFC[:4]), 0, False)[0] == "stop"  # collapsed at 30k
        assert decide(arm, evals(KYV7CLFC[:3]), 0, False)[0] is None
        assert decide(arm, evals(OJ5K0JQB[:11]), 0, False)[0] == "passed"  # 165 of 250
        assert decide(arm, evals([0.58] * 11), 0, False)[0] == "stop"      # the base policy held


def test_the_resfit_rx_arm_is_stopped_only_at_its_collapse():
    """Its anchors ran to 150k with no 100k check: s8opblsd's 60k-100k evals (123 of 250) would fail
    the residual's, and neither anchor converged by the rule's test before 150k."""
    assert GROUPS["resfit-rx-1012"] == RESFIT_RX and len(RESFIT_RX) == 1
    for arm in RESFIT_RX:
        assert decide(arm, evals(KYV7CLFC[:4]), 0, False)[0] == "stop"  # collapsed at 30k
        assert decide(arm, evals(KYV7CLFC[:3]), 0, False)[0] is None
        for rates in (OJ5K0JQB, S8OPBLSD, [0.58] * 16):
            for i in range(1, len(rates) + 1):
                assert decide(arm, evals(rates[:i]), 0, False)[0] is None, (rates[:i])


def test_every_arm_of_the_2026_10_12_amendment_runs_to_a_fixed_150k():
    """Never stopped for collapse (2026-10-12 AMENDMENT, "every run to a fixed 150k"): the ResFiT
    and equivariant arms of resfit-equi-diff-1012 and equi-resfit-arch-1013, the two long reruns,
    and notilt, which is stopped at 150k because it was launched for 300k. The divergence stop
    still applies to each, and the diagnostic run is never stopped."""
    assert GROUPS["resfit-equi-diff-1012"] == RESFIT_EQUI_DIFF and len(RESFIT_EQUI_DIFF) == 7
    assert GROUPS["equi-resfit-arch-1013"] == EQUI_RESFIT_ARCH and len(EQUI_RESFIT_ARCH) == 2
    assert GROUPS["equi-residual-long-1012"] == EQUI_LONG and len(EQUI_LONG) == 2
    assert GROUPS["residual-tilt-1012"] == (NOTILT, DIAG)
    collapsed_for_good = KYV7CLFC + [0.0] * 9   # 16 evals, 0 to 150k
    for arm in (*RESFIT_EQUI_DIFF, *EQUI_RESFIT_ARCH, *EQUI_LONG, NOTILT):
        for rates in (collapsed_for_good, OJ5K0JQB, S8OPBLSD, [0.58] * 16):
            for i in range(1, len(rates) + 1):
                stop_at_150k = arm == NOTILT and i == 16
                assert decide(arm, evals(rates[:i]), 0, False)[0] == ("stop" if stop_at_150k else None), (arm, i)
        runaway = _qs(25_000, 30_000, 50.0)
        assert decide(arm, evals(KYV7CLFC[:4]), 0, False, runaway)[0] == "stop"
    assert "fixed length, 150000" in decide(NOTILT, evals([0.3] * 16), 0, False)[1]
    for rates in (KYV7CLFC, [0.0] * 7, [0.58] * 7):
        for i in range(1, len(rates) + 1):
            assert decide(DIAG, evals(rates[:i]), 0, False)[0] is None


def test_vc5o977f_replayed_is_flagged_at_99100_and_stopped_at_120000():
    rows = [json.loads(line) for line in open(VC5O977F / "wandb_logs/metrics.jsonl")]
    all_evals = sorted((int(r["_step"]), float(r["eval/success_rate"])) for r in rows
                       if r.get("eval/success_rate") is not None)
    all_qs = [(int(r["_step"]), key, float(r[key])) for r in rows for key in Q_KEYS
              if isinstance(r.get(key), (int, float))]
    assert min(step for step, _, value in all_qs if value > Q_DIVERGED) == 99_100
    passed, stops = False, []
    for i in range(1, len(all_evals) + 1):
        upto = all_evals[i - 1][0]
        action, why = decide(TUNED, all_evals[:i], 104, passed, [q for q in all_qs if q[0] <= upto])
        passed = passed or action == "passed"
        if action == "stop":
            stops.append((upto, why))
    assert passed
    assert stops[0][0] == 120_000 and "critic diverged" in stops[0][1]


def test_arms_hold_is_the_preregistered_break_rule():
    """tools/arms_hold.py, which decides combined: an arm breaks at 39 or fewer of 150 at 10k-30k,
    or at three evals in a row at or below 0.05; and cannot be decided before its 30k eval."""
    from arms_hold import holds

    assert holds(evals(OJ5K0JQB[:4])) == (True, holds(evals(OJ5K0JQB[:4]))[1])   # 71 of 150
    assert holds(evals(KYV7CLFC[:4]))[0] is False                                # collapsed
    assert holds(evals([0.6, 0.26, 0.26, 0.26]))[0] is False                     # 39 of 150
    assert holds(evals([0.6, 0.26, 0.26, 0.28]))[0] is True                      # 40 of 150
    assert holds(evals(OJ5K0JQB[:3]))[0] is None
    assert holds(evals([0.6, 0.5, 0.0, 0.0, 0.0]))[0] is False                   # collapsed after 30k
