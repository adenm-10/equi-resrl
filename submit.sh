#!/bin/bash
# Rewritable submission script. Overwrite this in place for each new experiment
# rather than adding another file under shell/paper_runs/ (there are ~150 there
# already). The durable record of what was run is docs/EXPERIMENTS.md plus the
# resolved config in wandb -- NOT this file.
#
#   ./submit.sh 1          # seed 1 on GPU 0
#   ./submit.sh 2 1        # seed 2 on GPU 1
#   SKIP_PREFLIGHT=1 ./submit.sh 1    # escape hatch, discouraged
#
# Currently configured for: equivariant residual TD3 on Can, reproducing the
# lost z8yoqylh run. See docs/EXPERIMENTS.md, "Reproduction of z8yoqylh".
#
# WHY THERE ARE ALMOST NO OVERRIDES HERE
# --------------------------------------
# The README and the coffee/square scripts pass n_step=5, gamma=0.995,
# stddev=0.025, action_scale=0.2. The reference run did NOT: its resolved config
# shows n_step=3, gamma=0.99, stddev=0.05, action_scale=0.1 -- i.e. the config
# defaults, with `algo.prefetch_batches=4` as the only algorithmic override.
# Every Can run in this project was launched that way.
#
# Two reasons not to "improve" on that:
#   1. n_step and gamma are part of the replay-buffer cache key. Overriding
#      either invalidates both Can caches (5 GB + 16 GB) and forces a dataset
#      rebuild plus a 10k-step env warmup.
#   2. It would no longer be a reproduction.

set -euo pipefail

SEED="${1:?usage: ./submit.sh <seed> [gpu_index]}"
GPU="${2:-0}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU"

# --- gate -------------------------------------------------------------------
# Blocks the launch if imports break, configs drift, the equivariance tests
# fail, or the machine cannot hold the run. See preflight.py.
if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
    echo "Running pre-submission gate..."
    python preflight.py --task Can --concurrent-seeds 1
else
    echo "WARNING: preflight gate skipped via SKIP_PREFLIGHT=1"
fi

# --- run --------------------------------------------------------------------
# Naming follows the repo's existing ablation-script convention
# (shell/paper_runs/ablations/boxcleanup/nstep/2b_boxcleanup_nstep5_1.sh):
#   group = the variant, shared across seeds
#   name  = variant-seedindex
#   seed  = small explicit integer, not the 10-digit random default
GROUP="equi-residual-rl"

echo
echo "Launching ${GROUP}-${SEED} on GPU ${GPU} (commit $(git rev-parse --short HEAD))"
echo

python -m resfit.rl_finetuning.scripts.train_residual_td3 \
    --config-name=residual_equi_td3_can_config \
    algo.prefetch_batches=4 \
    wandb.project=robomimic-can-final \
    wandb.name="${GROUP}-${SEED}" \
    wandb.group="${GROUP}" \
    wandb.notes="equivariant residual TD3 on Can, reproduction of z8yoqylh, seed ${SEED}" \
    seed="${SEED}" \
    debug=false
