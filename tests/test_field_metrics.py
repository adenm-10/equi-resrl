"""resfit/rl_finetuning/utils/field_metrics.py: the residual and the critic's slope per action field.

Logging only, so what matters is that each number means what its name says: the fields are the
action's, in the order both stacks use; the online and demo rows are the batch's halves; a signed
mean keeps its sign; and the slope is per sample, not per batch.
"""

from __future__ import annotations

import pytest
import torch

from resfit.rl_finetuning.utils.field_metrics import action_fields, residual_field_metrics


def test_a_one_arm_action_has_five_fields_in_the_envs_order():
    fields = action_fields(7, 1, 1)
    assert [(f, (sl.start, sl.stop), c) for f, sl, c in fields] == [
        ("action_xy_0", (0, 2), ("x_0", "y_0")), ("action_z_0", (2, 3), ("z_0",)),
        ("action_rx_ry_0", (3, 5), ("rx_0", "ry_0")), ("action_rz_0", (5, 6), ("rz_0",)),
        ("action_hand_0", (6, 7), ("hand_0",))]
    # ResFiT's path has no equivariance config: 7 per arm reads as one gripper per arm.
    assert action_fields(7) == fields


def test_two_arms_with_six_dof_hands_follow_each_other():
    fields = action_fields(24, 2, 6)
    assert [f for f, _, _ in fields][5:] == ["action_xy_1", "action_z_1", "action_rx_ry_1", "action_rz_1",
                                             "action_hand_1"]
    assert fields[4][1] == slice(6, 12) and fields[4][2] == tuple(f"hand{j}_0" for j in range(6))
    assert fields[-1][1] == slice(18, 24)


def test_an_action_of_another_shape_is_logged_one_component_per_field():
    assert [(f, sl) for f, sl, _ in action_fields(9)] == [(f"a{i}", slice(i, i + 1)) for i in range(9)]


def test_the_metrics_split_the_rows_keep_signs_and_report_the_slope_per_sample():
    batch, num_online = 8, 4
    residual = torch.zeros(batch, 7)
    residual[:num_online, 3] = 0.05      # online rows: rx pushed one way
    residual[num_online:, 3] = -0.05     # demo rows: the other
    residual[:, 4] = torch.tensor([0.03, -0.03] * 4)  # ry: a correction, mean zero
    slope = torch.zeros(batch, 7)
    slope[:, 0] = 0.02                   # the critic's per-sample dQ/dx
    grad = -slope / batch                # what -q.mean() leaves on the action
    out = residual_field_metrics(residual, grad, action_fields(7), num_online)

    assert out["debug/residual_field/rx_0_mean_online"] == pytest.approx(0.05)
    assert out["debug/residual_field/rx_0_mean_demo"] == pytest.approx(-0.05)
    assert out["debug/residual_field/ry_0_mean_online"] == pytest.approx(0.0)
    assert out["debug/residual_field/action_rx_ry_0_mean_abs_online"] == pytest.approx(0.04)
    assert out["debug/residual_field/action_rx_ry_0_length_online"] == pytest.approx((0.05**2 + 0.03**2) ** 0.5)
    assert out["debug/dQ_da_field/x_0_mean_online"] == pytest.approx(0.02)
    assert out["debug/dQ_da_field/action_xy_0_mean_abs_demo"] == pytest.approx(0.01)
    assert "debug/residual_field/action_z_0_length_online" not in out  # a length only for a pair
    assert len(out) == 2 * (2 * 5 + 2 + 2 * 7)


def test_an_empty_demo_half_is_skipped_and_vanilla_rl_names_the_action():
    out = residual_field_metrics(torch.ones(4, 7), torch.zeros(4, 7), action_fields(7), 4, name="action")
    assert out and all(key.endswith("_online") for key in out)
    assert "debug/action_field/x_0_mean_online" in out
