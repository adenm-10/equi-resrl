"""The residual and the critic's action gradient, per action field (2026-10-12).

The equivariant residual's collapse on Square is carried by a few of its seven components
(EXPERIMENTS.md, 2026-10-12 FINDING), and the pooled `train/residual_l1_magnitude` cannot say
which. These metrics split the residual by action field, and the critic's slope with it, on the
logged update's online and demo rows. Logging only: they read tensors the actor update already
made, and change nothing it computes.

Two readings they are for:
- A component's signed mean near its mean |r| is a push the same way in every state; a signed
  mean near zero with a large mean |r| is a correction that depends on the state.
- The slope is the critic's per-sample dQ/da. The actor loss is -q.mean(), so the gradient on the
  action is -dQ/da / batch: `debug/dQ_da_mean_abs` is 1/batch of the per-sample slope reported here.

Units are each stack's action units: env units on the equivariant path (IdentityScaler), ResFiT's
normalized units on its own.
"""

from __future__ import annotations

import torch

from resfit.rl_finetuning.equi_off_policy.networks.equi_normalizer import ACTION_FIELDS, arm_key

# Each field's components, as the env's per-arm action [delta_pose 6, hand H] orders them.
COMPONENTS = {"action_xy": ("x", "y"), "action_z": ("z",), "action_rx_ry": ("rx", "ry"), "action_rz": ("rz",)}
POSE_DIM = 6


def action_fields(action_dim: int, n_arms: int | None = None, hand_dof: int | None = None
                  ) -> list[tuple[str, slice, tuple[str, ...]]]:
    """(field, slice, component names) for each action field, arm by arm, as both stacks order the
    action: the equivariant encoder's action_shape, and the env's per-arm [delta_pose 6, hand H].

    Without n_arms and hand_dof (ResFiT's path has no equivariance config), an action of 7 per arm
    is read as one gripper per arm. Any other action is logged one component per field.
    """
    if n_arms is None:
        n_arms, hand_dof = (action_dim // (POSE_DIM + 1), 1) if action_dim % (POSE_DIM + 1) == 0 else (0, 0)
    if n_arms * (POSE_DIM + hand_dof) != action_dim:
        return [(f"a{i}", slice(i, i + 1), (f"a{i}",)) for i in range(action_dim)]
    fields, pos = [], 0
    for arm in range(n_arms):
        for field in ACTION_FIELDS:
            if field == "action_hand":
                names = ("hand",) if hand_dof == 1 else tuple(f"hand{j}" for j in range(hand_dof))
            else:
                names = COMPONENTS[field]
            fields.append((arm_key(field, arm), slice(pos, pos + len(names)), tuple(f"{c}_{arm}" for c in names)))
            pos += len(names)
    return fields


def residual_field_metrics(residual: torch.Tensor, grad: torch.Tensor, fields, num_online: int,
                           name: str = "residual") -> dict[str, float]:
    """Per field, on the batch's online rows [:num_online] and demo rows [num_online:]:
    - `debug/<name>_field/<field>_mean_abs_<rows>`, and `_length_` for a pair;
    - `debug/<name>_field/<component>_mean_<rows>`, the signed mean;
    - `debug/dQ_da_field/<field>_mean_abs_<rows>` and `<component>_mean_<rows>`, the per-sample slope.

    `residual` is the actor's output (`_actions`; the whole action for vanilla RL, name="action"),
    `grad` the actor loss's gradient on the combined action (`_dQ_da`), both (batch, action_dim).
    """
    slope = -grad * grad.shape[0]
    out = {}
    for rows_name, rows in (("online", slice(0, num_online)), ("demo", slice(num_online, None))):
        if residual[rows].shape[0] == 0:
            continue
        for field, sl, components in fields:
            r, s = residual[rows, sl], slope[rows, sl]
            out[f"debug/{name}_field/{field}_mean_abs_{rows_name}"] = r.abs().mean().item()
            out[f"debug/dQ_da_field/{field}_mean_abs_{rows_name}"] = s.abs().mean().item()
            if len(components) == 2:
                out[f"debug/{name}_field/{field}_length_{rows_name}"] = r.norm(dim=1).mean().item()
            for j, component in enumerate(components):
                out[f"debug/{name}_field/{component}_mean_{rows_name}"] = r[:, j].mean().item()
                out[f"debug/dQ_da_field/{component}_mean_{rows_name}"] = s[:, j].mean().item()
    return out
