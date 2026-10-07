"""Tier 1: the fixed-env baselines' verdict (tools/baseline_verdict.py).

The rule is pre-registered in docs/EXPERIMENTS.md, 2026-10-07 (seed-group fixed-env-baselines-1007):
an arm pools both seeds' five evals at 110k-150k; tuned RLPD passes iff that mean is at least B_can,
a residual arm iff it beats B_sq by two binomial standard errors of the difference. Checked here
on runs built to sit on either side of each.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from baseline_verdict import (  # noqa: E402
    ARMS, COLLAPSE_RATE, COLLAPSE_RUN, EPISODES_PER_EVAL, MIN_Z, SEED_GROUP, SEEDS, WINDOW,
    arm_verdict, collapsed, find_runs, parse_arm, read_bar,
)

BAR = (0.5, 500)


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
    assert (SEEDS, EPISODES_PER_EVAL, MIN_Z) == (2, 50, 2.0)
    assert (COLLAPSE_RATE, COLLAPSE_RUN) == (0.05, 3)
    assert {arm for arm, (method, _) in ARMS.items() if method == "tuned-rlpd"} == \
           {"resfit-tuned-rlpd-can", "equi-tuned-rlpd-can"}


def test_tuned_rlpd_at_the_bar_passes_and_below_it_does_not():
    at = arm_verdict("resfit-tuned-rlpd-can", {"a": _rows([0.5] * 5), "b": _rows([0.5] * 5)}, BAR)
    assert at["p"] == 0.5 and at["episodes"] == 500
    assert at["passes"] is True, at["why"]
    below = arm_verdict("equi-tuned-rlpd-can", {"a": _rows([0.5] * 5), "b": _rows([0.48] * 5)}, BAR)
    assert below["passes"] is False, below["why"]


def test_the_seeds_are_pooled_not_judged_alone():
    """One seed above the bar and one below pass together when their mean is at it."""
    v = arm_verdict("resfit-tuned-rlpd-can", {"a": _rows([0.75] * 5), "b": _rows([0.25] * 5)}, BAR)
    assert v["p"] == 0.5 and v["passes"] is True


def test_a_residual_arm_needs_two_standard_errors():
    se = math.sqrt(0.5 * 0.5 / 500 + 0.5 * 0.5 / 500)  # at p = b = 0.5
    assert MIN_Z * se == pytest.approx(0.0632, abs=1e-4)
    above = arm_verdict("equi-residual-square", {"a": _rows([0.58] * 5), "b": _rows([0.58] * 5)}, BAR)
    assert above["margin"] == pytest.approx(
        MIN_Z * math.sqrt(0.58 * 0.42 / 500 + 0.25 / 500))
    assert above["passes"] is True, above["why"]
    short = arm_verdict("equi-residual-square", {"a": _rows([0.54] * 5), "b": _rows([0.54] * 5)}, BAR)
    assert short["p"] > BAR[0]
    assert short["passes"] is False, short["why"]


def test_no_verdict_until_both_seeds_have_every_window_eval():
    one_seed = arm_verdict("resfit-residual-square", {"a": _rows([0.9] * 5)}, BAR)
    assert one_seed["passes"] is None and "1 of 2 seeds" in one_seed["why"]
    early = arm_verdict("resfit-residual-square",
                        {"a": _rows([0.9] * 5), "b": _rows([0.9] * 3, last_step=130_000)}, BAR)
    assert early["passes"] is None and "8 of the 10" in early["why"]
    assert early["runs"]["b"]["final"] == (130_000, 0.9), "the evals so far are still reported"


def test_collapse_is_three_evals_in_a_row_at_or_below_the_floor():
    assert collapsed([(0, 0.5), (10_000, 0.05), (20_000, 0.0), (30_000, 0.04)])
    assert not collapsed([(0, 0.5), (10_000, 0.05), (20_000, 0.0), (30_000, 0.06), (40_000, 0.0)])
    v = arm_verdict("equi-residual-square-zero-init",
                    {"a": _rows([0.0] * 5, early=0.5), "b": _rows([0.5] * 5, early=0.5)}, BAR)
    assert v["runs"]["a"]["collapsed"] and not v["runs"]["b"]["collapsed"]


def test_the_bar_is_read_from_its_measurement(tmp_path):
    path = tmp_path / "bar.json"
    path.write_text(json.dumps({"mean": 0.6, "rates": [0.6] * 10, "episodes_per_eval": 50}))
    assert read_bar(path) == (0.6, 500)


def test_an_arm_is_named_with_its_runs():
    assert parse_arm("equi-residual-square=ab12cd34,ef56gh78") == \
           ("equi-residual-square", ["ab12cd34", "ef56gh78"])
    with pytest.raises(Exception):
        parse_arm("not-an-arm=ab12cd34")


def test_no_verdict_before_the_bar_is_measured():
    v = arm_verdict("resfit-tuned-rlpd-can", {"a": _rows([0.9] * 5), "b": _rows([0.9] * 5)}, None)
    assert v["passes"] is None and "bar not measured yet" in v["why"]


def test_the_seed_groups_runs_are_found_by_their_notes(tmp_path):
    """By seed, per arm; a smoke run and another seed-group's run are not the arm's, and an arm
    whose name another arm's begins with keeps its own runs."""
    def package(project, run_id, notes):
        (tmp_path / project / run_id).mkdir(parents=True)
        argv = ["train_residual_td3.py", "--config-name=x", f"wandb.notes={notes}", "seed=1"]
        (tmp_path / project / run_id / "manifest.json").write_text(json.dumps({"argv": argv}))

    package("robomimic-square-final", "s2", f"{SEED_GROUP}/equi-residual-square/seed2")
    package("robomimic-square-final", "s1", f"{SEED_GROUP}/equi-residual-square/seed1")
    package("robomimic-square-final", "z1", f"{SEED_GROUP}/equi-residual-square-zero-init/seed1")
    package("robomimic-square-final", "old", "square-image-fix-1002/target-fix/seed1")
    package("equi-resrl-smoke", "sm", f"smoke/{SEED_GROUP}/equi-residual-square/seed1")
    found = find_runs(SEED_GROUP, tmp_path)
    assert found["equi-residual-square"] == ["s1", "s2"]
    assert found["equi-residual-square-zero-init"] == ["z1"]
    assert found["resfit-tuned-rlpd-can"] == []
