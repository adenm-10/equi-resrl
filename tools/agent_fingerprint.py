"""Fingerprint the whole equivariant agent at the Square reference config.

Parameters at construction, the eval-mode forward pass, and one TD3 update
(critic and actor) are each reduced to a sum and a seeded random projection,
so two source trees can be checked for identical behaviour:

    python tools/agent_fingerprint.py --out fp_head.json
    PYTHONPATH=~/projects/equi-resrl-square python tools/agent_fingerprint.py --out fp_pinned.json
    python tools/agent_fingerprint.py --compare fp_head.json fp_pinned.json

The deep check is stronger, and covers any config in CANDIDATES. It runs 8
updates and one act(), then saves every tensor: the state dict, the optimizer
state, the action and the last update's metrics. Two saves are compared
exactly, tensor by tensor:

    python tools/agent_fingerprint.py --deep head.pt --candidate critic-fn
    PYTHONPATH=~/projects/equi-resrl-1e33b8f python tools/agent_fingerprint.py \
        --deep old.pt --candidate critic-fn
    python tools/agent_fingerprint.py --compare-deep head.pt old.pt

With --device cuda it turns on PyTorch's deterministic mode, so that two runs
of the same tree on the same GPU can match at all.

PYTHONPATH selects the tree that is measured; ``meta.resfit_path`` records it.
Only APIs that already existed at b8e1de7, the commit the Square reference
seeds run from, are used here, so that tree can be measured too. Keep it that
way: do not pass n_arms, compute_metrics or any other argument added since.
A candidate that sets a field an old tree lacks fails with an AttributeError.

tests/test_regression_agent.py pins this output.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
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
OPTIMIZERS = ("encoder_opt", "critic_opt", "actor_opt")
DEEP_UPDATES = 8                # the deep check; every 4th updates the actor, as in training


def _candidates() -> dict[str, dict[str, object]]:
    """Every config the schedule of 2026-09-30 can launch, as dotted overrides.

    critic-fn and clipq each test one change. squash and depth are then built
    on whichever of the two passed: neither, either or both. So each of those
    four bases appears alone, with squash and with depth; "all-four" has every
    change at once. All are at the actor_norm default from that date, "none".
    "target-fix" is the reference as it ran, with the actor's batch norm.

    The normalization arms of 2026-10-01 are each one change off target-fix:
    the actor's IID batch norm replaced by FieldNorm or InnerBatchNorm, or
    InnerBatchNorm added after the encoder's convs.
    """
    fn = {"equivariance.critic_norm": "field_norm"}
    clipq = {"agent.clip_q_target_to_reward_range": True}
    squash = {"equivariance.residual_squash": True}
    depth = {"equivariance.num_critic_layers": 2}
    bases = {"default": {}, "critic-fn": fn, "clipq": clipq, "critic-fn+clipq": {**fn, **clipq}}

    new_default = {"equivariance.actor_norm": "none"}
    out = {"target-fix": {}}
    for base_name, base in bases.items():
        out[base_name] = {**new_default, **base}
        for top_name, top in (("squash", squash), ("depth", depth)):
            name = top_name if base_name == "default" else f"{base_name}+{top_name}"
            out[name] = {**new_default, **base, **top}
    out["all-four"] = {**new_default, **fn, **clipq, **squash, **depth}
    out["actor-fn"] = {"equivariance.actor_norm": "field_norm"}
    out["actor-inner"] = {"equivariance.actor_norm": "inner_bn"}
    out["enc-inner"] = {"equivariance.encoder_norm": "inner_bn"}
    return out


CANDIDATES = _candidates()


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


def build_reference_agent(device: str = "cpu", overrides: dict | None = None,
                          **equivariance) -> QAgent:
    """The equivariant QAgent at the Square reference config, seeded.

    `overrides` maps dotted config paths to values, as in CANDIDATES.
    Keyword arguments override EquivarianceConfig fields, for tests of one
    option. The fingerprint itself always uses the reference."""
    cfg = ResidualEquiTD3SquareConfig()
    cfg.agent.device = device
    cfg.agent.actor.actor_last_layer_init_scale = INIT_SCALE
    # The reference ran with batch norm in the actor. actor_norm defaults to
    # "none" from 2026-09-30; trees older than 2026-09-29 lack the field and
    # always built the batch norm.
    if hasattr(cfg.equivariance, "actor_norm"):
        cfg.equivariance.actor_norm = "iid_bn"
    for path, value in {**(overrides or {}),
                        **{f"equivariance.{k}": v for k, v in equivariance.items()}}.items():
        *parents, name = path.split(".")
        node = cfg
        for parent in parents:
            node = getattr(node, parent)
        if not hasattr(node, name):
            raise AttributeError(f"the config has no field {path!r}")
        setattr(node, name, value)

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
    agent.to(device)  # the normalizer is built on CPU; the trainer moves it the same way
    return agent


def fixed_batch(weights: bool = False, *, cameras=CAMERAS, state_dim: int = STATE_DIM,
                action_dim: int = ACTION_DIM) -> TensorDict:
    """A replay batch shaped like the trainer's, rebuilt identically per call.

    Rebuild it for every use: QAgent.update writes features into the obs dicts.
    weights=True adds the importance weights the replay buffer attaches to
    every batch, all 1 under uniform sampling. They send the critic loss down
    its weighted branch, as in training. The fingerprint predates them.
    The layout defaults to Square's; tests/test_sync_free.py also builds a
    two-arm batch.
    """
    g = torch.Generator().manual_seed(1)

    def obs() -> TensorDict:
        images = {
            cam: torch.randint(0, 256, (BATCH, *OBS_SHAPE), generator=g, dtype=torch.uint8)
            for cam in cameras
        }
        return TensorDict(
            {
                **images,
                "observation.state": torch.randn(BATCH, state_dim, generator=g),
                "observation.base_action": torch.rand(BATCH, action_dim, generator=g) * 2 - 1,
            },
            batch_size=[BATCH],
        )

    batch = TensorDict(
        {
            "obs": obs(),
            "action": torch.rand(BATCH, action_dim, generator=g) * 2 - 1,
            "next": TensorDict(
                {"obs": obs(), "reward": torch.tensor([0.0, 1.0, 0.0, 1.0])},
                batch_size=[BATCH],
            ),
            "gamma": torch.full((BATCH,), 0.99**3),  # algo.n_step=3
            "nonterminal": torch.tensor([True, False, True, True]),
        },
        batch_size=[BATCH],
    )
    if weights:
        batch["_weight"] = torch.ones(BATCH)
    return batch


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


def _leaves(prefix: str, obj) -> dict[str, object]:
    """A nested dict/list, such as a state dict, flattened to path -> leaf."""
    if isinstance(obj, dict):
        items = obj.items()
    elif isinstance(obj, (list, tuple)):
        items = enumerate(obj)
    else:
        return {prefix: obj.detach().cpu() if torch.is_tensor(obj) else obj}
    out = {}
    for key, value in items:
        out.update(_leaves(f"{prefix}/{key}", value))
    return out


def deep(agent: QAgent) -> dict[str, object]:
    """Everything the agent holds after DEEP_UPDATES updates and one act().

    The updates run in train mode on the weighted batch, so the random crops,
    the target noise, the importance weights and the actor step all take part.
    The action is sampled, with exploration noise, as in training. Returns
    path -> tensor (on CPU) or number."""
    device = next(agent.parameters()).device
    agent.train(True)
    _seed(UPDATE_SEED)
    for k in range(DEEP_UPDATES):
        metrics = agent.update(fixed_batch(weights=True).to(device), stddev=STDDEV,
                               update_actor=(k + 1) % 4 == 0, bc_batch=None, ref_agent=agent)
    agent.train(False)
    with torch.no_grad():
        action = agent.act(fixed_batch()["obs"].to(device), eval_mode=False, stddev=STDDEV)

    out = {"action": action}
    out.update(_leaves("state_dict", agent.state_dict()))
    for name in OPTIMIZERS:
        out.update(_leaves(name, getattr(agent, name).state_dict()))
    out.update(_leaves("metrics", metrics))
    return out


def compare_deep(a: dict, b: dict) -> list[str]:
    """Paths whose values are not exactly equal, or present in one only."""
    def equal(x, y) -> bool:
        if torch.is_tensor(x) and torch.is_tensor(y):
            return x.dtype == y.dtype and x.shape == y.shape and torch.equal(x, y)
        return not torch.is_tensor(x) and not torch.is_tensor(y) and x == y

    return sorted(k for k in a.keys() | b.keys() if k not in a or k not in b or not equal(a[k], b[k]))


def _deterministic_cuda() -> None:
    """Make two runs of one tree on one GPU agree, where PyTorch can.

    TF32 stays on, as in the trainer. cuDNN's autotuner (benchmark) picks
    kernels by timing them, so it is turned off."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--out", help="measure the tree on PYTHONPATH and write JSON here")
    mode.add_argument("--compare", nargs=2, metavar=("A", "B"),
                      help="exit non-zero unless the two fingerprints are identical")
    mode.add_argument("--deep", metavar="OUT",
                      help="run the deep check on the tree on PYTHONPATH and save it here")
    mode.add_argument("--compare-deep", nargs=2, metavar=("A", "B"),
                      help="exit non-zero unless the two deep checks are identical")
    parser.add_argument("--candidate", default="target-fix", choices=list(CANDIDATES),
                        help="the config the deep check builds")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"),
                        help="where the deep check runs")
    args = parser.parse_args()

    if args.deep:
        if args.device == "cuda":
            _deterministic_cuda()
        agent = build_reference_agent(args.device, overrides=CANDIDATES[args.candidate])
        meta = {"resfit_path": sys.modules[QAgent.__module__].__file__,
                "candidate": args.candidate, "device": args.device}
        torch.save({"meta": meta, "values": deep(agent)}, args.deep)
        print(f"wrote {args.deep} ({args.candidate} on {args.device}, "
              f"resfit from {meta['resfit_path']})")
        return 0

    if args.compare_deep:
        saves = [torch.load(path) for path in args.compare_deep]
        for path, save in zip(args.compare_deep, saves):
            print(f"{path}: {save['meta']}")
        differing = compare_deep(saves[0]["values"], saves[1]["values"])
        for k in differing:
            print(f"DIFFERS {k}")
        if differing:
            return 1
        print(f"identical: {len(saves[0]['values'])} values")
        return 0

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
