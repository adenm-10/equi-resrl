#!/bin/bash
# Rewritable submission script. Overwrite this in place for each new experiment.
# The durable record is docs/EXPERIMENTS.md plus the resolved config in wandb.
#
#   ./submit.sh 1          # seed 1 on GPU 0
#   ./submit.sh 2 1        # seed 2 on GPU 1
#   SKIP_PREFLIGHT=1 ./submit.sh 1    # escape hatch, discouraged
#
# Currently configured for: equivariant residual TD3 on SQUARE at HEAD.
# Pre-registered in docs/EXPERIMENTS.md as seed-group `equi-square-v1`.
#
# WHY actor_last_layer_init_scale IS PASSED EXPLICITLY
# ----------------------------------------------------
# HEAD defaults this to 0.0. Every run in the verified n=2 Can replication
# (38z4wr9z, sw2qwfs9) and both recovered successes (z8yoqylh, 36pfxsww)
# resolved to 1e-4, because that was the default at caf83f3. Relying on the
# default here would silently change the experiment. Passed explicitly so the
# resolved config says what was intended.
#
# WHY THERE ARE NO OTHER OVERRIDES
# --------------------------------
# n_step, gamma, buffer_size and learning_starts are all part of the replay
# buffer cache key. Overriding any of them changes the key. Defaults kept so
# the hashes match preflight's mirror in cache_hashes().

set -euo pipefail

SEED="${1:?usage: ./submit.sh <seed> [gpu_index]}"
GPU="${2:-0}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU"

PY="${PYTHON:-python}"
if ! "$PY" -c "import torchrl, escnn" >/dev/null 2>&1; then
    FALLBACK="$HOME/miniforge3/envs/residual/bin/python"
    if [[ -x "$FALLBACK" ]]; then
        echo "note: '$PY' lacks the deps; falling back to $FALLBACK"
        PY="$FALLBACK"
    else
        echo "error: no interpreter with torchrl+escnn found." >&2
        echo "       run 'conda activate residual', or set PYTHON=/path/to/python" >&2
        exit 1
    fi
fi

GROUP="equi-square-v1"

OVERRIDES=(
    --config-name=residual_equi_td3_square_config
    algo.prefetch_batches=4
    agent.actor.actor_last_layer_init_scale=1e-4
    wandb.project=robomimic-square-final
    "wandb.name=${GROUP}-${SEED}"
    "wandb.group=${GROUP}"
    "wandb.notes=equi-square/head/seed${SEED}"
    "seed=${SEED}"
    debug=false
)

if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
    echo "Running pre-submission gate..."
    "$PY" preflight.py --task Square --concurrent-seeds 1 --compose "${OVERRIDES[@]}"
else
    echo "WARNING: preflight gate skipped via SKIP_PREFLIGHT=1"
fi

echo
echo "Launching ${GROUP}-${SEED} on GPU ${GPU} (commit $(git rev-parse --short HEAD))"
echo

"$PY" -m resfit.rl_finetuning.scripts.train_residual_td3 "${OVERRIDES[@]}"