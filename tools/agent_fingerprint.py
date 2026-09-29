"""Fingerprint the whole equivariant agent at the Square reference config.

Parameters at construction, the eval-mode forward pass, and one TD3 update
(critic and actor) are each reduced to a sum and a seeded random projection,
so two source trees can be checked for identical behaviour:

    python tools/agent_fingerprint.py --out fp_head.json
    PYTHONPATH=~/projects/equi-resrl-square python tools/agent_fingerprint.py --out fp_pinned.json
    python tools/agent_fingerprint.py --compare fp_head.json fp_pinned.json

PYTHONPATH selects the tree that is measured; ``meta.resfit_path`` records it.
Only APIs that already existed at b8e1de7, the commit the Square reference
seeds run from, are used here, so that tree can be measured too. Keep it that
way: do not pass n_arms or any other argument added since.

tests/test_regression_agent.py pins this output.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import sys

import escnn
import numpy as np
import torch
from tensordict import TensorDict

from resfit.rl_finetuning.config.residual_td3 import ResidualEquiTD3SquareConfig
from resfit.rl_finetuning.equi_off_policy.networks.equi_normalizer import (
    build_equivariant_normalizer,
)
from resfit.rl_finetuning.equi_off_policy.rl.q_agent import QAgent

# The equi-square-v1 reference: its config, launched with the init-scale
# override below (submit.sh at b8e1de7). With the config default of 0.0 the
# actor outputs exact zeros and the forward fingerprint would pin nothing.
INIT_SCALE = 1e-4
ROTATION_CENTRE = (-0.56, 0.0)  # env_probes/Square.json
CAMERAS = ("observation.images.agentview", "observation.images.robot0_eye_in_hand")
OBS_SHAPE = (3, 84, 84)
STATE_DIM = 9
ACTION_DIM = 7
BATCH = 4
STDDEV = 0.05                   # algo.stddev_{min,max}
UPDATE_SEED = 2                 # the update samples target noise and critic heads
MODULES = ("enc", "actor", "critic")


def _stats(n: int) -> dict:
    """Deterministic stand-in for LeRobotDataset.meta.stats, spread per dim.

    A copy of tests/test_regression_single_arm.py::_stats. This file must not
    import from tests/, which differs between the trees it measures."""
    a = np.arange(n, dtype=np.float32)
    return {
        "min": -1.0 - 0.1 * a,
        "max": 1.0 + 0.1 * a,
        "mean": 0.05 * a,
        "std": 1.0 + 0.01 * a,
    }


def _seed(seed: int) -> None:
    """Seed every RNG the agent draws from, as the trainer does.

    torch alone is not enough: the actor's delta-orthogonal init samples
    through scipy.stats.ortho_group, which uses numpy's global RNG."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _reduce(t: torch.Tensor) -> list[float]:
    """Sum plus a seeded random projection, so a reordering is detected too."""
    flat = t.detach().reshape(-1).double()
    proj = torch.randn(
        flat.numel(), generator=torch.Generator().manual_seed(7), dtype=torch.float64)
    return [flat.sum().item(), (flat * proj).sum().item()]


def _flat_params(module: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in module.parameters()])


def build_reference_agent(**equivariance) -> QAgent:
    """The equivariant QAgent at the Square reference config, on CPU, seeded.

    Keyword arguments override EquivarianceConfig fields, for tests of one
    option; the fingerprint itself always uses the defaults."""
    cfg = ResidualEquiTD3SquareConfig()
    cfg.agent.device = "cpu"
    cfg.agent.actor.actor_last_layer_init_scale = INIT_SCALE
    for name, value in equivariance.items():
        if not hasattr(cfg.equivariance, name):
            raise AttributeError(f"EquivarianceConfig has no field {name!r}")
        setattr(cfg.equivariance, name, value)

    _seed(0)
    # QAgent prints its modules; keep stdout for results.
    with contextlib.redirect_stdout(sys.stderr):
        agent = QAgent(
            obs_shape=OBS_SHAPE,
            prop_shape=(STATE_DIM,),
            action_dim=ACTION_DIM,
            rl_cameras=list(CAMERAS),
            agent_cfg=cfg.agent,
            equi_cfg=cfg.equivariance,
            residual_actor=True,
            equivariant=True,
        )
    normalizer = build_equivariant_normalizer(
        stats={"observation.state": _stats(STATE_DIM), "action": _stats(ACTION_DIM)},
        robot_base_xy=torch.tensor(ROTATION_CENTRE),
    )
    agent.enc.set_normalizer(normalizer, robot_base_xy=list(ROTATION_CENTRE))
    return agent


def fixed_batch() -> TensorDict:
    """A replay batch shaped like the trainer's, rebuilt identically per call.

    Rebuild it for every use: QAgent.update writes features into the obs dicts.
    """
    g = torch.Generator().manual_seed(1)

    def obs() -> TensorDict:
        images = {
            cam: torch.randint(0, 256, (BATCH, *OBS_SHAPE), generator=g, dtype=torch.uint8)
            for cam in CAMERAS
        }
        return TensorDict(
            {
                **images,
                "observation.state": torch.randn(BATCH, STATE_DIM, generator=g),
                "observation.base_action": torch.rand(BATCH, ACTION_DIM, generator=g) * 2 - 1,
            },
            batch_size=[BATCH],
        )

    return TensorDict(
        {
            "obs": obs(),
            "action": torch.rand(BATCH, ACTION_DIM, generator=g) * 2 - 1,
            "next": TensorDict(
                {"obs": obs(), "reward": torch.tensor([0.0, 1.0, 0.0, 1.0])},
                batch_size=[BATCH],
            ),
            "gamma": torch.full((BATCH,), 0.99**3),  # algo.n_step=3
            "nonterminal": torch.tensor([True, False, True, True]),
        },
        batch_size=[BATCH],
    )


def fingerprint(agent: QAgent) -> dict:
    """Parameters, eval-mode outputs, and what one update does to both.

    Leaves the agent in train mode, with one update applied."""
    before = {name: _flat_params(getattr(agent, name)) for name in MODULES}
    fp = {"params": {name: _reduce(before[name]) for name in MODULES}}

    agent.train(False)
    batch = fixed_batch()
    with torch.no_grad():
        feat_actor, feat_critic = agent.enc(batch["obs"])
        fp["forward"] = {
            "actor_feat": _reduce(feat_actor.tensor),
            "critic_feat": _reduce(feat_critic.tensor),
            "actor": _reduce(agent.actor(feat_actor, std=None)),
            "critic": _reduce(agent.critic(feat_critic, batch["action"])),
        }

    agent.train(True)
    _seed(UPDATE_SEED)
    metrics = agent.update(
        fixed_batch(), stddev=STDDEV, update_actor=True, bc_batch=None, ref_agent=agent)
    fp["update"] = {
        "critic_loss": metrics["train/critic_loss"],
        "actor_loss_base": metrics["train/actor_loss_base"],
        # The step each module took. Deltas keep full precision where the
        # parameters themselves would hide a 1e-6 actor step.
        **{f"{name}_delta": _reduce(_flat_params(getattr(agent, name)) - before[name])
           for name in MODULES},
    }

    fp["meta"] = {
        # resfit is a namespace package, so name a module file instead.
        "resfit_path": sys.modules[QAgent.__module__].__file__,
        "torch": torch.__version__,
        "escnn": getattr(escnn, "__version__", "unknown"),
    }
    return fp


def flatten(fp: dict) -> dict[str, float]:
    """Every fingerprint value by path, excluding meta."""
    out = {}
    for group, entries in fp.items():
        if group == "meta":
            continue
        for key, value in entries.items():
            values = value if isinstance(value, list) else [value]
            for i, v in enumerate(values):
                out[f"{group}/{key}/{i}"] = v
    return out


def compare(a: dict, b: dict) -> list[str]:
    """Paths whose values are not exactly equal, or present in one only."""
    fa, fb = flatten(a), flatten(b)
    return sorted(k for k in fa.keys() | fb.keys() if fa.get(k) != fb.get(k))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--out", help="measure the tree on PYTHONPATH and write JSON here")
    mode.add_argument("--compare", nargs=2, metavar=("A", "B"),
                      help="exit non-zero unless the two fingerprints are identical")
    args = parser.parse_args()

    if args.out:
        fp = fingerprint(build_reference_agent())
        with open(args.out, "w") as f:
            json.dump(fp, f, indent=2)
        print(f"wrote {args.out} (resfit from {fp['meta']['resfit_path']})")
        return 0

    fps = []
    for path in args.compare:
        with open(path) as f:
            fps.append(json.load(f))
        print(f"{path}: resfit from {fps[-1]['meta']['resfit_path']}")
    differing = compare(*fps)
    if differing:
        fa, fb = flatten(fps[0]), flatten(fps[1])
        for k in differing:
            print(f"DIFFERS {k}: {fa.get(k)!r} vs {fb.get(k)!r}")
        return 1
    print(f"identical: {len(flatten(fps[0]))} values")
    return 0


if __name__ == "__main__":
    sys.exit(main())
