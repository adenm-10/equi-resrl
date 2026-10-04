"""Tier 1: the tuned-RLPD verdict (tools/tuned_rlpd_verdict.py).

The rule is pre-registered in docs/EXPERIMENTS.md, 2026-10-04 (seed-group tuned-rlpd-1002): a run
passes iff the mean of its five evals at 110k-150k is at least its task's BC base rate. Checked
here on runs built to sit on either side of it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from tuned_rlpd_verdict import BASE_RATES, WINDOW, verdict  # noqa: E402

BASE = 0.5


def _rows(late, *, early=0.0, last_step=150_000):
    """A run's history: an eval every 10k up to last_step, at `early` before 110k and at the
    rates in `late` from 110k on, with training-episode rows in between, which carry no eval."""
    rows = []
    for step in range(0, last_step + 1, 10_000):
        rate = early if step < WINDOW[0] else late[WINDOW.index(step)]
        rows.append({"_step": step, "eval/success_rate": rate})
        rows.append({"_step": step + 5_000, "training/episode_return": 0.0})
    return rows


def test_the_rule_is_the_preregistered_one():
    assert WINDOW == (110_000, 120_000, 130_000, 140_000, 150_000)
    assert BASE_RATES == {"robomimic-square-final": ("Square", 0.508),
                          "robomimic-can-final": ("Can", 0.894)}


def test_a_run_at_the_base_rate_passes():
    v = verdict(_rows([0.25, 0.75, 0.5, 0.5, 0.5]), BASE)  # exact in binary
    assert v["window_mean"] == BASE
    assert v["passes"] is True, v["why"]


def test_a_run_just_below_the_base_rate_does_not_pass():
    v = verdict(_rows([0.5, 0.5, 0.5, 0.5, 0.48]), BASE)
    assert v["passes"] is False, v["why"]


def test_only_the_five_late_evals_count():
    """A high start or a high middle does not carry a run; the window alone decides."""
    peak_then_fall = verdict(_rows([0.2] * 5, early=0.9), BASE)
    assert peak_then_fall["passes"] is False
    assert peak_then_fall["best"] == (0, 0.9)
    late_rise = verdict(_rows([0.6] * 5, early=0.0), BASE)
    assert late_rise["passes"] is True


def test_no_verdict_before_the_150k_eval():
    v = verdict(_rows([0.9] * 5, last_step=140_000), BASE)
    assert v["passes"] is None and v["window_mean"] is None
    assert "4 of the 5" in v["why"]


def test_step0_best_and_final_are_reported():
    v = verdict(_rows([0.3, 0.7, 0.5, 0.4, 0.6], early=0.02), BASE)
    assert v["step0"] == 0.02
    assert v["best"] == (120_000, 0.7)
    assert v["final"] == (150_000, 0.6)
    assert len(v["evals"]) == 16
