"""Tier 1: the stage-1 verdict and the watcher's decisions, which run unattended.

tools/square_verdict.py applies the rule pre-registered in docs/EXPERIMENTS.md, 2026-10-01
AMENDMENT, and tools/run_watch.py stops trainers and queues runs from it. Both act without anyone
looking, so their decisions are checked here on runs built to sit on either side of each rule.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from square_verdict import collapse_step, top_arms, verdict  # noqa: E402

GOLDEN = (math.sqrt(5) - 1) / 2


def _run(success_rate, evals, *, episodes=400, last_step=100_000):
    """History rows: `episodes` training episodes evenly spread over [0, last_step], each a
    success with probability success_rate(step), drawn by a fixed low-discrepancy sequence so
    the realized rate tracks it closely; and one eval row per (step, rate)."""
    rows = []
    for i in range(episodes):
        step = int((i + 1) * last_step / episodes)
        success = (i * GOLDEN) % 1 < success_rate(step)
        rows.append({"_step": step, "training/episode_return": 0.99 ** 250 if success else 0,
                     "training/episode_steps": 250 if success else 300})
    rows += [{"_step": step, "eval/success_rate": rate} for step, rate in evals]
    return rows


EVALS = [(10_000 * k, 0.5) for k in range(11)]


def test_a_rising_run_advances():
    v = verdict(_run(lambda s: 0.3 + 0.4 * s / 100_000, EVALS))
    assert v["advances"] is True, v["why"]
    assert v["slope"] == pytest.approx(0.4, abs=0.1)
    assert v["length_slope"] < 0


def test_a_flat_run_does_not_advance():
    v = verdict(_run(lambda s: 0.5, EVALS))
    assert v["advances"] is False
    assert abs(v["slope"]) < 2 * v["slope_se"]


def test_climbing_back_to_the_start_does_not_count():
    """A dip at 10k and a steady climb back to where the run began: the slope alone would pass."""
    v = verdict(_run(lambda s: 0.5 if s <= 10_000 else 0.1 + 0.4 * (s - 10_000) / 90_000, EVALS))
    assert v["slope"] / v["slope_se"] >= 2
    assert v["late_success"] <= v["early_success"]
    assert v["advances"] is False


def test_a_collapse_stops_and_fails():
    evals = [(0, 0.5), (10_000, 0.04), (20_000, 0.0), (30_000, 0.05), (40_000, 0.5)]
    assert collapse_step(evals) == 30_000
    assert collapse_step([(0, 0.5), (10_000, 0.0), (20_000, 0.0), (30_000, 0.06)]) is None
    v = verdict(_run(lambda s: 0.3 + 0.4 * s / 100_000, evals + EVALS[5:]))
    assert v["advances"] is False and "collapsed at 30000" in v["why"]


def test_no_verdict_before_the_100k_eval():
    v = verdict(_run(lambda s: 0.5, EVALS[:10], last_step=95_000))
    assert v["advances"] is None


def test_data_after_100k_is_ignored():
    """A run that went on to 150k is judged on its first 100k only."""
    flat_then_rising = _run(lambda s: 0.5 if s <= 100_000 else 0.9,
                            EVALS + [(110_000, 0.9), (120_000, 0.9)], episodes=600,
                            last_step=150_000)
    assert verdict(flat_then_rising)["advances"] is False
    assert all(step <= 100_000 for step, _ in verdict(flat_then_rising)["evals"])


def test_top_arms_follow_the_2026_09_30_table():
    assert top_arms(False, False) == ("squash", "depth")
    assert top_arms(True, False) == ("critic-fn-squash", "critic-fn-depth")
    assert top_arms(False, True) == ("clipq-squash", "clipq-depth")
    assert top_arms(True, True) == ("critic-fn-clipq-squash", "critic-fn-clipq-depth")


def test_top_arms_are_submit_sh_arms():
    from tests.test_sweep_arms import ARMS
    for flags in ((False, False), (True, False), (False, True), (True, True)):
        assert set(top_arms(*flags)) <= set(ARMS)


def test_watcher_stops_only_long_runs_at_100k_and_any_collapse():
    run_watch = pytest.importorskip("run_watch")
    healthy = [(10_000 * k, 0.5) for k in range(11)]
    assert run_watch.stop_reason(healthy, 150_000, 100_000) is not None
    assert run_watch.stop_reason(healthy, 100_000, 100_000) is None, "it stops by itself"
    assert run_watch.stop_reason(healthy[:10], 150_000, 100_000) is None
    collapsed = [(0, 0.5), (10_000, 0.0), (20_000, 0.02), (30_000, 0.0)]
    assert "collapsed at 30000" in run_watch.stop_reason(collapsed, 100_000, 100_000)


def test_watcher_queues_squash_ahead_of_target_fix(tmp_path):
    run_watch = pytest.importorskip("run_watch")
    queue = tmp_path / "gpu1.txt"
    queue.write_text("# comment\nabc1234 target-fix 2\n")
    run_watch.insert_line(queue, "abc1234 squash 1", before_arm="target-fix")
    assert queue.read_text().splitlines() == ["# comment", "abc1234 squash 1",
                                              "abc1234 target-fix 2"]
    run_watch.insert_line(queue, "abc1234 depth 1", before_arm=None)
    assert queue.read_text().splitlines()[-1] == "abc1234 depth 1"
