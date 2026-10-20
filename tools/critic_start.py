"""How a critic starts: how far apart its heads begin, and how pessimistic its TD target is.

    PYTHONPATH=$PWD python tools/critic_start.py --submit <arm> [--submit <arm> ...] \\
        [--spec <name>=--config-name=<config>,<override>,...] [--match-to <name>] \\
        [--batch real|reference] [--dataset local/robomimic-mh-square-rerendered] [--seeds 5]

Two numbers per config, each the mean over `--seeds` initializations (torch, numpy and random seeded
0, 1, ...), in train mode, the target critic's mode:
  - spread: the SD across the critic's heads of each head's mean Q over the batch;
  - min of two: the batch mean of the smaller of two heads' Q, averaged over every pair of heads.
    The TD target bootstraps from the min over a random pair (`critic.q_value`), so this is the
    target's pessimism before gamma.

`--submit <arm>` takes the arm's overrides from `./submit.sh`'s dry run, so the config measured is the
one launched; `--spec` names any other. `--match-to <name>` prints, for every other config, the
`equivariance.critic_last_layer_init_scale` that gives it that config's spread. The output Linear is
the last layer of each head, so scaling it by s scales every head's Q, and both numbers, by s
exactly; the printed scale is that ratio times the config's own scale.

Batches:
  - real (the default): `--n` transitions of `--dataset`, at frames spread evenly over the dataset, in
    the replay buffer's format (uint8 images, raw state and action), with the equivariant
    normalizer built from the dataset's statistics and the robot base from `env_probes/`, as the
    trainer builds them;
  - reference: `tools/agent_fingerprint.fixed_batch` (random images and states, Square's layout),
    the batch of the 2026-10-08 measurement (EXPERIMENTS.md, the fifth arm's PRE-REGISTERED).

Vanilla RL only (no base policy): the batch carries no base action. ResFiT's stack is measured on the
reference batch only, as on 2026-10-08: its state and action scalers sit outside the agent.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def submit_overrides(arm: str) -> list[str]:
    """What ./submit.sh would launch `arm` with (seed 1), read from its dry run."""
    with tempfile.TemporaryDirectory() as cache:
        (Path(cache) / "offline_buffer_cache").mkdir()
        env = {k: v for k, v in os.environ.items() if k != "SMOKE"}
        env.update({"DRY_RUN": "1", "SKIP_PREFLIGHT": "1", "PYTHON": "true", "CACHE_DIR": cache})
        out = subprocess.run([str(REPO / "submit.sh"), arm, "1"], env=env, capture_output=True,
                             text=True, check=True, cwd=REPO).stdout
    return [line.strip() for line in out.splitlines() if line.startswith("    ")]


def compose(overrides: list[str]):
    from hydra import compose as hydra_compose
    from hydra import initialize

    import resfit.rl_finetuning.config.residual_td3  # noqa: F401  registers the configs

    flag, config_name = overrides[0].split("=", 1)
    assert flag == "--config-name", overrides[0]
    with initialize(version_base=None, config_path=None):
        return hydra_compose(config_name=config_name, overrides=overrides[1:])


def _seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def build_agent(cfg, device: str, stats: dict, rotation_centre):
    """The agent the trainer builds from `cfg`, with `stats` for the equivariant normalizer."""
    from omegaconf import open_dict

    from tools.agent_fingerprint import ACTION_DIM, CAMERAS, OBS_SHAPE, STATE_DIM

    assert cfg.base_policy is None, "vanilla RL only: the batches carry no base action"
    with open_dict(cfg):
        cfg.agent.device = device
    shapes = dict(obs_shape=OBS_SHAPE, prop_shape=(STATE_DIM,), action_dim=ACTION_DIM,
                  rl_cameras=list(CAMERAS), residual_actor=False)
    if cfg.equivariance is None:
        from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent

        return QAgent(cfg=cfg.agent, **shapes).to(device)
    from resfit.rl_finetuning.equi_off_policy.networks.equi_normalizer import build_equivariant_normalizer
    from resfit.rl_finetuning.equi_off_policy.rl.q_agent import QAgent

    agent = QAgent(agent_cfg=cfg.agent, equi_cfg=cfg.equivariance, **shapes)
    normalizer = build_equivariant_normalizer(stats=stats, robot_base_xy=torch.tensor(rotation_centre))
    agent.enc.set_normalizer(normalizer, robot_base_xy=list(rotation_centre))
    return agent.to(device)


def reference_batch():
    from tools.agent_fingerprint import ACTION_DIM, ROTATION_CENTRE, STATE_DIM, _stats, fixed_batch

    batch = fixed_batch(base_action=False)
    stats = {"observation.state": _stats(STATE_DIM), "action": _stats(ACTION_DIM)}
    return dict(batch["obs"]), batch["action"], stats, ROTATION_CENTRE


def real_batch(dataset_name: str, n: int):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    from tools.agent_fingerprint import CAMERAS

    dataset = LeRobotDataset(dataset_name)
    frames = [dataset[int(i)] for i in np.linspace(0, dataset.num_frames - 1, n).round()]
    obs = {cam: torch.stack([(f[cam] * 255.0).clamp(0, 255).to(torch.uint8) for f in frames])
           for cam in CAMERAS}
    obs["observation.state"] = torch.stack([f["observation.state"] for f in frames]).float()
    action = torch.stack([f["action"] for f in frames]).float()
    task = dataset_name.split("robomimic-mh-")[-1].split("-")[0].capitalize()
    probe = json.loads((REPO / "env_probes" / f"{task}.json").read_text())
    return obs, action, dataset.meta.stats, tuple(probe["base_xy"])


def measure(cfg, batch, device: str, seeds: int) -> dict[str, float]:
    obs, action, stats, rotation_centre = batch
    spreads, min_twos = [], []
    for seed in range(seeds):
        _seed(seed)
        agent = build_agent(cfg, device, stats, rotation_centre)
        agent.train(True)
        on_device = {k: v.to(device) for k, v in obs.items()}
        with torch.no_grad():
            if cfg.equivariance is None:
                q = agent.critic(agent._encode(on_device, augment=False), on_device["observation.state"],
                                 action.to(device))
            else:
                q = agent.critic(agent.enc(on_device)[1], action.to(device))
        q = q.squeeze(-1)                                    # [heads, batch]
        spreads.append(q.mean(dim=1).std().item())
        pairs = itertools.combinations(range(q.shape[0]), 2)
        min_twos.append(float(np.mean([torch.minimum(q[i], q[j]).mean().item() for i, j in pairs])))
    return {"spread": float(np.mean(spreads)), "spread_sd": float(np.std(spreads)),
            "min_two": float(np.mean(min_twos)),
            "scale": cfg.equivariance.critic_last_layer_init_scale if cfg.equivariance is not None else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--submit", action="append", default=[], help="an arm in ./submit.sh")
    parser.add_argument("--spec", action="append", default=[], help="<name>=--config-name=<c>,<override>,...")
    parser.add_argument("--match-to", help="a config name: print the scale that matches its spread")
    parser.add_argument("--batch", choices=("real", "reference"), default="real")
    parser.add_argument("--dataset", default="local/robomimic-mh-square-rerendered")
    parser.add_argument("--n", type=int, default=256, help="transitions in the real batch")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    configs = {arm: submit_overrides(arm) for arm in args.submit}
    for spec in args.spec:
        name, overrides = spec.split("=", 1)
        configs[name] = overrides.split(",")
    batch = real_batch(args.dataset, args.n) if args.batch == "real" else reference_batch()

    results = {}
    for name, overrides in configs.items():
        cfg = compose(overrides)
        if cfg.equivariance is None and args.batch == "real":
            raise SystemExit(f"{name}: ResFiT's stack is measured on --batch reference only")
        results[name] = measure(cfg, batch, args.device, args.seeds)
        print(f"{name:40s} spread {results[name]['spread']:.4f} ± {results[name]['spread_sd']:.4f}   "
              f"min of two {results[name]['min_two']:+.4f}   last-layer scale {results[name]['scale']}",
              flush=True)
    if args.match_to:
        target = results[args.match_to]["spread"]
        for name, r in results.items():
            if name != args.match_to:
                own = 1.0 if r["scale"] is None else r["scale"]
                print(f"{name:40s} matches {args.match_to}'s spread at scale {own * target / r['spread']:.4f}")
    print(json.dumps({"batch": args.batch, "dataset": args.dataset if args.batch == "real" else None,
                      "seeds": args.seeds, "results": results}))


if __name__ == "__main__":
    main()
