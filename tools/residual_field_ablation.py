"""Re-evaluate a saved residual policy with whole action fields of its residual set to zero.

    python -m tools.residual_field_ablation <run_id> --step 10000 [--episodes 100] \\
        [--variants full,none,-xy,-z,-tilt,-yaw,-gripper,tilt-only] [--videos full,-tilt,tilt-only]

Which field of the residual breaks the policy (EXPERIMENTS.md, 2026-10-12 PRE-REGISTERED, "the
field test")? It loads `models/step_<N>.pt` from the run's package (checkpoint_interval), builds
the agent and the eval env as the trainer builds them (train_residual_td3.build_agent,
eval_base_policy.build_eval_env), and runs the trainer's own evaluator once per variant:
  full        the residual as trained;
  none        no residual: the base policy alone;
  -<field>    the residual with that field set to zero;
  tilt-only   the residual with every field but tilt set to zero.
Fields: xy (action_xy), z, tilt (action_rx_ry), yaw (action_rz), gripper (action_hand), on every
arm. Setting a whole field to zero keeps the policy equivariant: a pair is zeroed as a whole.

Writes outputs/diagnostics/residual-field-ablation/<run_id>_step<N>.json (per variant: successes,
episodes, rate, binomial SE, and the evaluator's tilt readout) and, for --videos, each variant's
eval video beside it. Env initial states are not seeded, here or in the trainer (eval_base_policy.py
explains); --seed fixes the policy side. The config is composed from the run's launch argv
(run_config). Never starts wandb. Equivariant residual runs only.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import sys
from datetime import datetime
from pathlib import Path

import torch

FIELDS = {"xy": "action_xy", "z": "action_z", "tilt": "action_rx_ry", "yaw": "action_rz",
          "gripper": "action_hand"}
VARIANTS = ("full", "none", *(f"-{f}" for f in FIELDS), "tilt-only")


def zeroed_fields(variant: str) -> set[str]:
    """The ACTION_FIELDS names a variant sets to zero."""
    if variant == "full":
        return set()
    if variant == "none":
        return set(FIELDS.values())
    if variant == "tilt-only":
        return set(FIELDS.values()) - {FIELDS["tilt"]}
    if variant.startswith("-") and variant[1:] in FIELDS:
        return {FIELDS[variant[1:]]}
    raise ValueError(f"unknown variant {variant!r}; the variants are {', '.join(VARIANTS)}")


def field_mask(action_layout, action_fields, zeroed: set[str], device) -> torch.Tensor:
    """1 on every component the residual keeps, 0 on every component of a zeroed field, every arm."""
    mask = torch.ones(sum(sl.stop - sl.start for sl, _ in action_layout), device=device)
    for (sl, _kind), name in zip(action_layout, action_fields):
        if name.rsplit("_", 1)[0] in zeroed:  # action_rx_ry_0 -> action_rx_ry
            mask[sl] = 0.0
    return mask


class FieldMaskedAgent:
    """Stands in for QAgent inside run_dexmg_evaluation, its residual's masked fields set to zero.

    act() is the agent's, times the mask; everything else the evaluator reads (enc, critic,
    combine_actions, residual_actor, eval, train) is the agent's own.
    """

    def __init__(self, agent, mask: torch.Tensor):
        self._agent, self._mask = agent, mask

    def act(self, obs, *, eval_mode=False, stddev=0.0, cpu=True):
        residual = self._agent.act(obs, eval_mode=eval_mode, stddev=stddev, cpu=False) * self._mask
        return residual.cpu() if cpu else residual

    def eval(self):
        self._agent.train(False)
        return self

    def train(self, mode: bool = True):
        self._agent.train(mode)
        return self

    def __getattr__(self, name):
        return getattr(self._agent, name)


def run_config(package: Path):
    """The run's config, composed from its launch's argv at this code. A field added since the run
    takes its default, which leaves it as it ran (test_no_ops.py; test_sweep_arms.py checks an
    anchor's composition against its recorded config); a checkpoint's own saved config would lack
    the field."""
    from hydra import compose, initialize

    import resfit.rl_finetuning.config.residual_td3  # noqa: F401  registers the configs

    argv = json.loads((package / "manifest.json").read_text())["argv"][1:]
    name = next(a.split("=", 1)[1] for a in argv if a.startswith("--config-name="))
    overrides = [a for a in argv if not a.startswith(("--config-name=", "hydra."))]
    with initialize(version_base=None, config_path=None):
        return compose(config_name=name, overrides=overrides)


def _parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_id")
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--videos", default="full,-tilt,tilt-only")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

    from resfit.lerobot.utils.load_policy import RUNS_ROOT, download_policy_from_wandb, git_state, load_policy
    from resfit.rl_finetuning.off_policy.common_utils import utils
    from resfit.rl_finetuning.scripts.eval_base_policy import build_eval_env, seed_everything
    from resfit.rl_finetuning.scripts.train_residual_td3 import build_agent
    from resfit.rl_finetuning.utils.checkpoint import load_residual_model
    from resfit.rl_finetuning.utils.evaluate_dexmg import run_dexmg_evaluation

    args = _parse_args(argv)
    variants = args.variants.split(",")
    videos = set(args.videos.split(",")) if args.videos else set()
    for v in variants:
        zeroed_fields(v)  # unknown variants fail before anything loads
    packages = sorted(RUNS_ROOT.glob(f"*/{args.run_id}"))
    if len(packages) != 1:
        sys.exit(f"{args.run_id}: {len(packages)} run packages under {RUNS_ROOT}")
    path = packages[0] / "models" / f"step_{args.step}.pt"
    if not path.exists():
        sys.exit(f"{path} does not exist: was the run launched with checkpoint_interval?")
    out_dir = args.out or RUNS_ROOT.parent / "diagnostics" / "residual-field-ablation"
    out = out_dir / f"{args.run_id}_step{args.step}.json"
    if out.exists():
        sys.exit(f"{out} exists; a measurement is a record, pass --out to write elsewhere")

    cfg = run_config(packages[0])
    if cfg.equivariance is None or cfg.base_policy is None:
        sys.exit("the field test is for equivariant residual runs")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":  # the trainer's numerics
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    bp = cfg.base_policy
    policy_dir, _ = download_policy_from_wandb(bp.wandb_id, step=bp.wt_type, artifact_version=bp.wt_version)
    base_policy = load_policy(policy_dir)
    base_policy.to(device)
    base_policy.eval()
    seed_everything(args.seed)
    env = build_eval_env(cfg, base_policy, device)
    image_keys = [cfg.rl_camera] if isinstance(cfg.rl_camera, str) else list(cfg.rl_camera)
    agent = build_agent(
        cfg,
        obs_shape=tuple(env.observation_space[image_keys[0]].shape[1:]),
        prop_dim=env.observation_space["observation.state"].shape[1],
        action_dim=env.action_space.shape[1],
        image_keys=image_keys,
        dataset_stats=LeRobotDatasetMetadata(cfg.offline_data.name).stats,
        device=torch.device(device),
    )
    saved = load_residual_model(agent, path)
    agent.train(False)

    started = datetime.now()
    results = {}
    for variant in variants:
        mask = field_mask(agent.enc.action_layout, agent.enc.action_fields, zeroed_fields(variant), device)
        with utils.eval_mode(agent):
            metrics, video = run_dexmg_evaluation(
                env=env, agent=FieldMaskedAgent(agent, mask), num_episodes=args.episodes, device=device,
                global_step=args.step, save_video=variant in videos, save_q_plots=False,
                run_name=f"{args.run_id}_step{args.step}_{variant}", output_dir=out_dir, equivariant=True)
        k = round(metrics["eval/success_rate"] * args.episodes)
        p = k / args.episodes
        results[variant] = {
            "successes": k, "episodes": args.episodes, "rate": p,
            "se_binomial": math.sqrt(p * (1 - p) / args.episodes),
            "zeroed": sorted(zeroed_fields(variant)),
            **{key.split("/", 1)[1]: value for key, value in metrics.items() if key.startswith("eval/tilt")},
            "video": str(video) if video else None,
        }
        print(f"{variant:>10}: {k}/{args.episodes} ({p:.2f})", flush=True)
    env.close()

    out_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "run_id": args.run_id, "step": args.step, "checkpoint": str(path),
        "checkpoint_global_step": saved["global_step"], "seed": args.seed,
        "env_initial_states_seeded": False, "eval_num_envs": env.num_envs, "results": results,
        "git": git_state(), "host": socket.gethostname(), "started": started.isoformat(timespec="seconds"),
        "finished": datetime.now().isoformat(timespec="seconds"),
    }, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
