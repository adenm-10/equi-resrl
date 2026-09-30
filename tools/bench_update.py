"""Time one environment step's worth of learning, for any config in CANDIDATES.

What the trainer does between two env steps, minus the env step itself: one act(), whose
action is read on the CPU as env.step reads it, then algo.num_updates_per_iteration updates
on a batch of algo.batch_size, the last with the actor, then one torch.cuda.synchronize().
The agent is built by tools/agent_fingerprint.py, with the trainer's cuDNN and TF32 settings.

    python tools/bench_update.py --candidate default
    PYTHONPATH=~/projects/equi-resrl-1e33b8f python tools/bench_update.py --candidate default

PYTHONPATH selects the tree that is measured, as for agent_fingerprint.py, which is imported
from next to this file. A tree whose QAgent.update takes compute_metrics is timed with it off,
as between two log steps. Older trees computed metrics on every update, and are timed that way.

Prints the wall time per env step, and the GPU's busy time: the kernel times torch.profiler
records, summed. When the two are about equal the loop is GPU-bound. On a GPU that another run
is using, absolute numbers read high: compare two trees measured back to back, not across days.
"""

from __future__ import annotations

import argparse
import inspect
import sys
import time

import torch
from torch.profiler import ProfilerActivity, profile

from agent_fingerprint import (
    BATCH,
    CANDIDATES,
    STDDEV,
    build_reference_agent,
    fixed_batch,
)
from resfit.rl_finetuning.config.residual_td3 import ResidualEquiTD3SquareConfig
from resfit.rl_finetuning.equi_off_policy.rl.q_agent import QAgent
from resfit.rl_finetuning.off_policy.common_utils import utils

BATCH_SIZE = ResidualEquiTD3SquareConfig().algo.batch_size
UPDATES = ResidualEquiTD3SquareConfig().algo.num_updates_per_iteration


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--candidate", default="default", choices=list(CANDIDATES))
    parser.add_argument("--steps", type=int, default=20, help="env steps timed")
    parser.add_argument("--warmup", type=int, default=5, help="env steps run first, untimed")
    args = parser.parse_args()

    # As the trainer sets them.
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda")
    agent = build_reference_agent("cuda", overrides=CANDIDATES[args.candidate])
    agent.train(True)
    skips_metrics = "compute_metrics" in inspect.signature(agent.update).parameters
    update_kwargs = {"compute_metrics": False} if skips_metrics else {}

    # One distinct batch per update. They are copied to the GPU here, before timing; the
    # trainer's copy is timed under batch_sampling, not gradient_update.
    batch = torch.cat([fixed_batch(weights=True)] * (BATCH_SIZE // BATCH)).to(device)
    batches = [batch.clone() for _ in range(UPDATES)]
    act_obs = fixed_batch()["obs"][:1].to(device)  # num_envs=1

    def env_step() -> None:
        # Clones, because act() and update() write the encoder's features into their input.
        with torch.no_grad(), utils.eval_mode(agent):
            agent.act(act_obs.clone(), eval_mode=False, stddev=STDDEV, cpu=False).cpu()
        for i in range(UPDATES):
            agent.update(batches[i].clone(), stddev=STDDEV, update_actor=i == UPDATES - 1,
                         bc_batch=None, ref_agent=agent, **update_kwargs)
        torch.cuda.synchronize()

    for _ in range(args.warmup):
        env_step()
    start = time.perf_counter()
    for _ in range(args.steps):
        env_step()
    wall_ms = (time.perf_counter() - start) / args.steps * 1e3

    profiled = 3
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(profiled):
            env_step()
    kernels = [e for e in prof.key_averages() if e.device_type.name == "CUDA"]
    gpu_ms = sum(e.self_device_time_total for e in kernels) / 1e3 / profiled

    print(f"resfit from {sys.modules[QAgent.__module__].__file__}")
    print(f"{args.candidate}: batch {BATCH_SIZE}, {UPDATES} updates per env step, "
          f"metrics {'off' if skips_metrics else 'on every update'}")
    print(f"  wall     {wall_ms:7.1f} ms / env step  (mean of {args.steps})")
    print(f"  GPU busy {gpu_ms:7.1f} ms / env step  (mean of {profiled}, profiled)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
