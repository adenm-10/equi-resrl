"""tools/square_rerendered_watch.py applies square-rerendered-1009's rules as pre-registered and amended.

Checked on real eval sequences where one exists: oj5k0jqb (ResFiT's residual on Square, 0.800 over
110k-150k) and kyv7clfc (the fixed-env baselines' equivariant residual, 0.00 from 10k).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from square_rerendered_watch import RESIDUAL, TUNED, decide  # noqa: E402

OJ5K0JQB = [0.50, 0.30, 0.58, 0.54, 0.40, 0.54, 0.56, 0.64, 0.70, 0.64, 0.76, 0.72, 0.84, 0.90, 0.66, 0.88]
KYV7CLFC = [0.60, 0.00, 0.00, 0.02, 0.02, 0.00, 0.00]


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
