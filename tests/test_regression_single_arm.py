"""Tier 1: pin the single-arm encoder's output so a refactor cannot move it.

ResObsEnc is about to be generalised to two arms. Can and Square runs made
before that change are only comparable to later ones if this keeps passing.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.conftest import BATCH, CROP, ENC_HIDDEN, N, OBS_SHAPE

# Captured at b8e1de7 on 2026-09-25 from the unmodified single-arm encoder,
# and verified bit-identical across repeated runs. If these move, behaviour
# changed -- either fix the change or record it in docs/EXPERIMENTS.md.
GOLDEN = {
    "actor": {"width": 162, "sum": 1454.6414267920773, "proj": -184.4838089490517},
    "critic": {"width": 155, "sum": 1450.0525313743274, "proj": 45.627534909416},
}

BASE_XY = [-0.5, -0.1]  # Can, from env_probes/Can.json
RTOL = 1e-6

# Start of the proprioception block, after the regular vis and trivial in-hand
# blocks. Much of the vis block is zeros, so the reordering guard below has to
# swap fields here to exercise anything.
PROP_START = ENC_HIDDEN * N + ENC_HIDDEN


def _stats(n: int) -> dict:
    """Deterministic stand-in for LeRobotDataset.meta.stats, spread per dim."""
    a = np.arange(n, dtype=np.float32)
    return {
        "min": -1.0 - 0.1 * a,
        "max": 1.0 + 0.1 * a,
        "mean": 0.05 * a,
        "std": 1.0 + 0.01 * a,
    }


def _encoder():
    """Single-arm equivariant encoder on CPU, seeded and in eval mode."""
    from resfit.rl_finetuning.equi_off_policy.networks.equi_normalizer import (
        build_equivariant_normalizer,
    )
    from resfit.rl_finetuning.equi_off_policy.networks.obs_encoder import ResObsEnc

    torch.manual_seed(0)
    enc = ResObsEnc(
        obs_shape=OBS_SHAPE, crop_shape=(CROP, CROP), N=N,
        n_hidden=ENC_HIDDEN, equivariant=True, initialize=True,
    )
    normalizer = build_equivariant_normalizer(
        stats={"observation.state": _stats(9), "action": _stats(7)},
        robot_base_xy=torch.tensor(BASE_XY),
    )
    enc.set_normalizer(normalizer, robot_base_xy=BASE_XY)
    enc.eval()
    return enc


def _inputs() -> dict:
    """Fixed observation batch. eval() makes the random crop deterministic."""
    g = torch.Generator().manual_seed(1)
    return {
        "observation.images.agentview": torch.randint(
            0, 256, (BATCH, *OBS_SHAPE), generator=g, dtype=torch.uint8),
        "observation.images.robot0_eye_in_hand": torch.randint(
            0, 256, (BATCH, *OBS_SHAPE), generator=g, dtype=torch.uint8),
        "observation.state": torch.randn(BATCH, 9, generator=g),
        "observation.base_action": torch.randn(BATCH, 7, generator=g),
    }


def _fingerprint(t: torch.Tensor) -> tuple[float, float]:
    """Sum plus a seeded random projection, so reordering fields is detected."""
    flat = t.reshape(-1).double()
    proj = torch.randn(
        flat.numel(), generator=torch.Generator().manual_seed(7), dtype=torch.float64)
    return flat.sum().item(), (flat * proj).sum().item()


@pytest.fixture(scope="module")
def features():
    enc = _encoder()
    with torch.no_grad():
        actor, critic = enc(_inputs())
    return {"actor": actor.tensor, "critic": critic.tensor}


@pytest.mark.parametrize("name", ["actor", "critic"])
def test_feature_width_unchanged(features, name):
    assert features[name].shape == (BATCH, GOLDEN[name]["width"])


@pytest.mark.parametrize("name", ["actor", "critic"])
def test_features_bit_identical(features, name):
    total, proj = _fingerprint(features[name])
    assert total == pytest.approx(GOLDEN[name]["sum"], rel=RTOL)
    assert proj == pytest.approx(GOLDEN[name]["proj"], rel=RTOL)


def test_fingerprint_detects_reordering(features):
    """Stop a vacuous pass: the sum alone cannot see a field swap."""
    original = features["actor"]
    swapped = original.clone()
    i, j = PROP_START, PROP_START + 1
    swapped[:, [i, j]] = swapped[:, [j, i]]

    assert not torch.equal(swapped, original), "swap was a no-op, guard is vacuous"
    assert _fingerprint(swapped)[0] == pytest.approx(_fingerprint(original)[0], rel=RTOL)
    assert _fingerprint(swapped)[1] != pytest.approx(_fingerprint(original)[1], rel=RTOL)
