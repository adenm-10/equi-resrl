"""Independent implementation of the C_N group action.

Deliberately written from the physics and from docs/EQUIVARIANCE.md, **not** by
importing the FieldType declarations from the modules under test. If the tests
derived the group action from the same declarations the code uses, a wrong
declaration would pass its own test.

The conventions here were verified against escnn's own representation matrices
(see test_group_action.py), which pins down the two sign conventions that
physics alone does not fix:

  * irrep(1) at element k  ->  counter-clockwise rotation by +2*pi*k/N
  * regular_repr element k ->  torch.roll(v, shifts=+k)
"""

from __future__ import annotations

import math

import torch

# ---------------------------------------------------------------------------
# Per-representation action
# ---------------------------------------------------------------------------


def act_trivial(x: torch.Tensor, k: int, N: int) -> torch.Tensor:
    """Trivial representation: invariant. A height, a gripper width."""
    return x.clone()


def act_irrep1(x: torch.Tensor, k: int, N: int) -> torch.Tensor:
    """irrep(1): a vector in the horizontal plane, rotated by 2*pi*k/N.

    Expects the last dimension to be exactly 2, ordered (x, y).
    """
    assert x.shape[-1] == 2, f"irrep(1) acts on pairs, got width {x.shape[-1]}"
    theta = 2.0 * math.pi * k / N
    c, s = math.cos(theta), math.sin(theta)
    rot = torch.tensor([[c, -s], [s, c]], dtype=x.dtype, device=x.device)
    return x @ rot.T


def act_regular(x: torch.Tensor, k: int, N: int) -> torch.Tensor:
    """regular_repr: one coefficient per group element, cyclically shifted."""
    assert x.shape[-1] == N, f"regular_repr acts on width {N}, got {x.shape[-1]}"
    return torch.roll(x, shifts=k, dims=-1)


_ACTIONS = {"trivial": act_trivial, "irrep1": act_irrep1, "regular": act_regular}
_WIDTHS = {"trivial": 1, "irrep1": 2, "regular": None}  # regular resolved from N


# ---------------------------------------------------------------------------
# Layouts, stated from the physics
# ---------------------------------------------------------------------------
# A layout is a list of (kind, count) pairs read left to right along the last
# tensor dimension. These mirror docs/EQUIVARIANCE.md. test_layouts.py asserts
# they agree with the FieldTypes the modules actually declare -- so a drift
# between doc and code fails a test rather than silently changing the maths.


def action_layout() -> list[tuple[str, int]]:
    """The 7-dim delta end-effector action (OSC_POSE) plus gripper."""
    return [
        ("irrep1", 1),   # 0:2  action_xy
        ("trivial", 1),  # 2:3  action_z
        ("irrep1", 1),   # 3:5  action_rx_ry
        ("trivial", 1),  # 5:6  action_rz
        ("trivial", 1),  # 6:7  action_gripper
    ]


def prop_layout() -> list[tuple[str, int]]:
    """The 11-dim proprioception vector."""
    return [
        ("irrep1", 1),   # ee xy, minus robot base xy
        ("irrep1", 3),   # 3 xy column pairs of the end-effector rotation
        ("trivial", 1),  # ee z
        ("trivial", 2),  # gripper state
    ]


def vis_ih_layout(n_hidden: int) -> list[tuple[str, int]]:
    """Visual features: agentview is regular, in-hand is trivial."""
    return [("regular", n_hidden), ("trivial", n_hidden)]


def enc_out_layout_critic(n_hidden: int) -> list[tuple[str, int]]:
    return vis_ih_layout(n_hidden) + prop_layout()


def enc_out_layout_actor(n_hidden: int) -> list[tuple[str, int]]:
    return enc_out_layout_critic(n_hidden) + action_layout()


# ---------------------------------------------------------------------------
# Composing a layout into a full action
# ---------------------------------------------------------------------------


def layout_width(layout: list[tuple[str, int]], N: int) -> int:
    total = 0
    for kind, count in layout:
        w = N if kind == "regular" else _WIDTHS[kind]
        total += w * count
    return total


def act_on_layout(x: torch.Tensor, layout: list[tuple[str, int]], k: int, N: int) -> torch.Tensor:
    """Apply the action of group element k to a tensor laid out as `layout`.

    x has shape [..., layout_width]. Blocks are transformed independently.
    """
    expected = layout_width(layout, N)
    assert x.shape[-1] == expected, f"expected width {expected}, got {x.shape[-1]}"

    if k % N == 0:
        return x.clone()

    out = torch.empty_like(x)
    pos = 0
    for kind, count in layout:
        w = N if kind == "regular" else _WIDTHS[kind]
        fn = _ACTIONS[kind]
        for _ in range(count):
            out[..., pos : pos + w] = fn(x[..., pos : pos + w], k, N)
            pos += w
    assert pos == expected
    return out


def rotate_image(img: torch.Tensor, k: int, N: int) -> torch.Tensor:
    """In-plane rotation of an image batch [B, C, H, W] by 2*pi*k/N.

    Only exact for multiples of 90 degrees, which is why the encoder
    equivariance test asserts on those and reports the rest as measured error.
    Uses torch.rot90 for right angles and bilinear resampling otherwise.
    """
    assert img.dim() == 4, f"expected [B, C, H, W], got {tuple(img.shape)}"
    k = k % N
    if k == 0:
        return img.clone()

    deg = 360.0 * k / N
    if abs(deg % 90.0) < 1e-6:
        # torch.rot90 rotates counter-clockwise in the (H, W) plane for k>0
        return torch.rot90(img, k=int(round(deg / 90.0)), dims=(-2, -1)).contiguous()

    theta = math.radians(deg)
    c, s = math.cos(theta), math.sin(theta)
    # affine_grid maps output coords -> input coords, so use the inverse rotation
    mat = torch.tensor([[c, s, 0.0], [-s, c, 0.0]], dtype=img.dtype, device=img.device)
    mat = mat.unsqueeze(0).expand(img.shape[0], 2, 3)
    grid = torch.nn.functional.affine_grid(mat, list(img.shape), align_corners=False)
    return torch.nn.functional.grid_sample(
        img, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )


def is_right_angle(k: int, N: int) -> bool:
    return abs((360.0 * k / N) % 90.0) < 1e-6
