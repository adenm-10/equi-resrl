"""tools/residual_field_ablation.py: the field test's variants zero exactly the fields they name.

The tool's verdict rests on its masks (EXPERIMENTS.md, 2026-10-12 PRE-REGISTERED, "the field test"),
so these check each variant's mask against the action layout, for one and two arms, and that the
masked agent is the agent but for its residual's masked components. Its end-to-end path, a saved
model through the trainer's builder, is checked on the GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.residual_field_ablation import (  # noqa: E402
    VARIANTS,
    FieldMaskedAgent,
    field_mask,
    run_config,
    zeroed_fields,
)

from resfit.lerobot.utils.load_policy import RUNS_ROOT  # noqa: E402

# One arm's action layout and field names, as the equivariant encoder builds them (test_layouts.py
# pins the encoder's own): xy, z, tilt, yaw, gripper.
ONE_ARM_LAYOUT = [(slice(0, 2), "irrep1"), (slice(2, 3), "trivial"), (slice(3, 5), "irrep1"),
                  (slice(5, 6), "trivial"), (slice(6, 7), "trivial")]
ONE_ARM_FIELDS = ["action_xy_0", "action_z_0", "action_rx_ry_0", "action_rz_0", "action_hand_0"]
TWO_ARM_LAYOUT = [(slice(s.start + o, s.stop + o), k) for o in (0, 7) for s, k in ONE_ARM_LAYOUT]
TWO_ARM_FIELDS = ONE_ARM_FIELDS + [f.replace("_0", "_1") for f in ONE_ARM_FIELDS]


@pytest.mark.parametrize(("variant", "kept"), [
    ("full", [0, 1, 2, 3, 4, 5, 6]),
    ("none", []),
    ("-xy", [2, 3, 4, 5, 6]),
    ("-z", [0, 1, 3, 4, 5, 6]),
    ("-tilt", [0, 1, 2, 5, 6]),
    ("-yaw", [0, 1, 2, 3, 4, 6]),
    ("-gripper", [0, 1, 2, 3, 4, 5]),
    ("tilt-only", [3, 4]),
])
def test_each_variant_keeps_exactly_its_components(variant, kept):
    mask = field_mask(ONE_ARM_LAYOUT, ONE_ARM_FIELDS, zeroed_fields(variant), "cpu")
    assert mask.nonzero().flatten().tolist() == kept
    two = field_mask(TWO_ARM_LAYOUT, TWO_ARM_FIELDS, zeroed_fields(variant), "cpu")
    assert two.nonzero().flatten().tolist() == kept + [i + 7 for i in kept]  # every arm alike


def test_the_variant_list_is_the_preregistered_one_and_an_unknown_variant_fails():
    assert VARIANTS == ("full", "none", "-xy", "-z", "-tilt", "-yaw", "-gripper", "tilt-only")
    with pytest.raises(ValueError, match="unknown variant"):
        zeroed_fields("-rx")


class _Agent:
    residual_actor = True

    def __init__(self):
        self.training = True
        self.critic = object()

    def act(self, obs, *, eval_mode=False, stddev=0.0, cpu=True):
        out = torch.arange(1.0, 15.0).reshape(2, 7)
        return out if not cpu else out.cpu()

    def train(self, mode=True):
        self.training = mode


def test_the_masked_agent_is_the_agent_but_for_its_masked_residual():
    agent = _Agent()
    mask = field_mask(ONE_ARM_LAYOUT, ONE_ARM_FIELDS, zeroed_fields("-tilt"), "cpu")
    masked = FieldMaskedAgent(agent, mask)
    out = masked.act({}, eval_mode=True, cpu=False)
    full = agent.act({}, eval_mode=True, cpu=False)
    assert torch.equal(out[:, [0, 1, 2, 5, 6]], full[:, [0, 1, 2, 5, 6]]) and not out[:, 3:5].any()
    assert masked.critic is agent.critic and masked.residual_actor
    masked.eval()
    assert not agent.training
    masked.train(True)
    assert agent.training


EAQ = RUNS_ROOT / "robomimic-square-final/eaqorqki/models/best_model.pt"


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not EAQ.exists(), reason="eaqorqki's saved model is not here")
def test_a_saved_model_loads_through_the_trainers_builder_and_the_unmasked_agent_acts_as_saved(device):
    """The tool's path: the run's config from its launch argv, the trainer's build_agent, a strict
    load. eaqorqki's checkpoint predates residual_field_bound, which its composed config holds at
    its default. The full variant's act() is the loaded agent's act() exactly; none's is zero."""
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

    from resfit.rl_finetuning.scripts.train_residual_td3 import build_agent
    from resfit.rl_finetuning.utils.checkpoint import load_residual_model
    from tools.agent_fingerprint import fixed_batch

    cfg = run_config(EAQ.parent.parent)
    assert dict(cfg.equivariance.residual_field_bound) == {}
    agent = build_agent(cfg, obs_shape=(3, 84, 84), prop_dim=9, action_dim=7,
                        image_keys=list(cfg.rl_camera), dataset_stats=LeRobotDatasetMetadata(cfg.offline_data.name).stats,
                        device=device)
    load_residual_model(agent, EAQ)
    agent.train(False)

    def obs():  # a fresh batch per call: act() writes features into a TensorDict's shallow copy
        return fixed_batch(base_action=True)["obs"].to(device)

    with torch.no_grad():
        plain = agent.act(obs(), eval_mode=True, cpu=False)
        full = FieldMaskedAgent(agent, field_mask(agent.enc.action_layout, agent.enc.action_fields,
                                                  zeroed_fields("full"), device)).act(obs(), eval_mode=True, cpu=False)
        none = FieldMaskedAgent(agent, field_mask(agent.enc.action_layout, agent.enc.action_fields,
                                                  zeroed_fields("none"), device)).act(obs(), eval_mode=True, cpu=False)
    assert torch.equal(full, plain)
    assert not none.any()
