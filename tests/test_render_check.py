"""Tier 1: the render check's statistic and bar, on synthetic frames.

tools/render_check.py compares demo first frames with env reset frames on the front camera. The
env-backed measurement runs in preflight itself (it needs the dataset and a renderer); this file
pins the arithmetic it is judged by: the gap, its bar of 2.5x the gap between the even and the
odd demos, and refusal of frames outside [0, 1].
"""

from __future__ import annotations

import pytest
import torch

from tools import render_check as rc


def _frames(n=40, seed=0, shift=0.0):
    g = torch.Generator().manual_seed(seed)
    base = torch.full((n, 3, 16, 16), 0.5)
    noise = 0.02 * torch.rand(n, 3, 16, 16, generator=g)
    frames = base + noise
    frames[:, :, :4] += shift  # a band of "floor" rendered differently
    return frames.clamp(0, 1)


def test_the_gap_is_the_mean_of_the_average_frames_difference():
    a = torch.zeros(2, 3, 2, 2)
    b = torch.zeros(2, 3, 2, 2)
    b[:, 0] = 0.3  # one channel of three differs by 0.3 everywhere
    assert rc.mean_frame_gap(a, b) == pytest.approx(0.1)


def test_frames_from_the_same_distribution_pass():
    v = rc.verdict(_frames(seed=0), _frames(seed=1))
    assert v["passed"]
    assert v["bar"] == pytest.approx(2.5 * v["within_demos"])


def test_a_band_rendered_differently_fails():
    v = rc.verdict(_frames(seed=0), _frames(seed=1, shift=0.1))
    assert not v["passed"]
    assert v["demo_vs_env"] > v["bar"]


def test_the_bar_is_two_and_a_half_times_the_within_demo_gap():
    demo = _frames(seed=0)
    within = rc.mean_frame_gap(demo[0::2], demo[1::2])
    assert rc.verdict(demo, demo)["within_demos"] == pytest.approx(within)
    assert rc.BAR_FACTOR == 2.5


@pytest.mark.parametrize("scale", [255.0, -1.0])
def test_frames_outside_unit_range_are_refused(scale):
    with pytest.raises(AssertionError):
        rc.verdict(_frames() * scale, _frames())
