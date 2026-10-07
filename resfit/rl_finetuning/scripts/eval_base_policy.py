"""Measure a base policy's success rate the way a training run's evals measure it.

    python -m resfit.rl_finetuning.scripts.eval_base_policy \\
        --config-name residual_equi_td3_square_config --num-evals 10 [--seed 1000] \\
        [--out PATH] [hydra overrides ...]

Runs the trainer's own evaluator, run_dexmg_evaluation, with a residual of
exactly zero, --num-evals times, on one eval env built as the trainer builds
its own. The result is the bar a residual run's evals are read against.

Why several 50-episode evals rather than one long one: each eval stops when
its cfg.eval_num_episodes-th episode finishes and drops the episodes still
running. Those skew long and failing, so the bias depends on the episode
count, and a base rate is only comparable when measured with the same count.

Env initial states are not seeded, here or in the trainer: gymnasium 1.1's
vector envs have no seed(), so the trainer's `eval_env.seed(cfg.seed + 1)` is
skipped, and each spawned worker seeds robosuite's placement sampler from OS
entropy. --seed fixes only the policy side (e.g. diffusion sampling noise).

Writes JSON to outputs/base_rates/ and never starts wandb.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import socket
import sys
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

import resfit.rl_finetuning.config.residual_td3  # noqa: F401  (registers the configs)
from resfit.dexmg.environments.dexmg import create_vectorized_env
from resfit.lerobot.utils.load_policy import (
    RUNS_ROOT,
    download_policy_from_wandb,
    git_state,
    load_policy,
    software_versions,
)
from resfit.rl_finetuning.equi_off_policy.networks.equi_normalizer import (
    IdentityScaler,
    IdentityStandardizer,
)
from resfit.rl_finetuning.utils.evaluate_dexmg import run_dexmg_evaluation
from resfit.rl_finetuning.wrappers.residual_env_wrapper import BasePolicyVecEnvWrapper

BASE_RATES_DIR = RUNS_ROOT.parent / "base_rates"


class _ZeroCritic:
    @staticmethod
    def q_value(feat, action: torch.Tensor) -> torch.Tensor:
        return torch.zeros(action.shape[0], 1, device=action.device)


class ZeroResidualAgent:
    """Stands in for QAgent inside run_dexmg_evaluation, with a zero residual.

    Implements exactly what run_dexmg_evaluation(equivariant=True) calls:
    act, enc, critic.q_value, eval, train and residual_actor. Its Q-values are
    zeros and mean nothing. Reusing the trainer's evaluator, rather than a
    separate rollout loop, keeps the episode accounting identical.
    """

    residual_actor = True
    critic = _ZeroCritic()

    def __init__(self, action_dim: int):
        self.action_dim = action_dim

    def act(self, obs: dict[str, torch.Tensor], *, eval_mode=False, stddev=0.0, cpu=True) -> torch.Tensor:
        state = obs["observation.state"]
        action = torch.zeros(state.shape[0], self.action_dim, device=state.device)
        return action.cpu() if cpu else action

    def enc(self, obs):
        return None, None

    def eval(self):
        return self

    def train(self, mode: bool = True):
        return self


def compose_config(name: str, overrides: list[str]) -> DictConfig:
    with hydra.initialize(version_base=None, config_path=None):
        return hydra.compose(config_name=name, overrides=overrides)


def seed_everything(seed: int) -> None:
    """The trainer's seeding (train_residual_td3.main)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_eval_env(cfg: DictConfig, base_policy, device: str) -> BasePolicyVecEnvWrapper:
    """The trainer's eval env: same env count, same wrapper.

    Identity normalizers are exact for a zero residual on either path: the
    env receives the base action itself.
    """
    num_cpus_available = os.cpu_count() - 1 if os.cpu_count() is not None else 1
    num_envs = min(cfg.eval_num_envs, cfg.eval_num_episodes, num_cpus_available)
    vec_env = create_vectorized_env(
        env_name=cfg.task,
        num_envs=num_envs,
        device=device,
        video_key=cfg.video_key,
        debug=cfg.debug,
    )
    return BasePolicyVecEnvWrapper(
        vec_env=vec_env,
        base_policy=base_policy,
        action_scaler=IdentityScaler(),
        state_standardizer=IdentityStandardizer(),
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--num-evals", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=None,
                        help="default: outputs/base_rates/<task>__<bc>__seed<seed>.json")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, as for the trainer")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    cfg = compose_config(args.config_name, args.overrides)
    bp = cfg.base_policy
    bc_run = bp.wandb_id.split("/")[-1]
    out = args.out or BASE_RATES_DIR / f"{cfg.task}__{bc_run}_{bp.wt_type}_{bp.wt_version}__seed{args.seed}.json"
    if out.exists():
        sys.exit(f"{out} exists; a measurement is a record, pass --out to write elsewhere")

    # Performance flags as in the trainer; TF32 changes the policy's numerics.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Load, then seed, then build the env: the trainer's order.
    policy_dir, _ = download_policy_from_wandb(
        bp.wandb_id, step=bp.wt_type, artifact_version=bp.wt_version)
    base_policy = load_policy(policy_dir)
    base_policy.to(device)
    base_policy.eval()
    seed_everything(args.seed)
    env = build_eval_env(cfg, base_policy, device)
    agent = ZeroResidualAgent(action_dim=env.action_space.shape[-1])

    started = datetime.now()
    rates = []
    for i in range(args.num_evals):
        metrics, _ = run_dexmg_evaluation(
            env=env,
            agent=agent,
            num_episodes=cfg.eval_num_episodes,
            device=device,
            global_step=i,
            save_video=False,
            save_q_plots=False,
            equivariant=True,
        )
        rates.append(metrics["eval/success_rate"])
        print(f"eval {i + 1}/{args.num_evals}: {rates[-1]:.2f}", flush=True)
    env.close()

    total = args.num_evals * cfg.eval_num_episodes
    mean = float(np.mean(rates))
    record = {
        "task": cfg.task,
        "config_name": args.config_name,
        "overrides": args.overrides,
        "base_policy": {**OmegaConf.to_container(bp), "local_dir": str(policy_dir)},
        "seed": args.seed,
        "env_initial_states_seeded": False,
        "eval_num_envs": env.num_envs,
        "episodes_per_eval": cfg.eval_num_episodes,
        "rates": rates,
        "mean": mean,
        "se_binomial": float(np.sqrt(mean * (1 - mean) / total)),
        "git": git_state(),
        "versions": software_versions(),
        "host": socket.gethostname(),
        "started": started.isoformat(timespec="seconds"),
        "finished": datetime.now().isoformat(timespec="seconds"),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2))
    print(f"base rate {mean:.3f} ± {record['se_binomial']:.3f} over {total} episodes -> {out}")


if __name__ == "__main__":
    main()
