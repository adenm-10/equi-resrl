#!/bin/bash
# Rewritable submission script. Overwrite this in place for each new experiment.
# The durable record is docs/EXPERIMENTS.md plus the resolved config in wandb.
#
#   ./submit.sh <arm> <seed> [gpu]      # e.g. ./submit.sh image-fix 1 0
#   DRY_RUN=1 ./submit.sh image-fix 1   # run the gate for that arm, then stop
#   SKIP_PREFLIGHT=1 ./submit.sh ...    # escape hatch, discouraged
#
# Currently configured for: the Square reference with act()'s image scaling fixed,
# pre-registered in docs/EXPERIMENTS.md, "2026-10-02 — PRE-REGISTERED — Square: the
# reference with the image fix, seeds 1 and 2". Launch it inside tmux from a worktree
# pinned to a commit, with CACHE_DIR pointing at the checkout that holds the buffer
# caches; tools/gpu_queue.sh does all three.
#
# The Square ablations of 2026-09-30/10-01 (seed-group square-ablations-1001) are
# parked. Their arms launch from ccff539, the commit they ran at.
#
# THE ARM
# -------
# image-fix is target-fix, the ablations' reference, at a commit where act() reads
# the env's float images as they are (EXPERIMENTS.md, 2026-10-02 CORRECTION). The
# config and its overrides are target-fix's, unchanged: the Square reference config,
# the actor's IID batch norm at its three sites, last-layer init scale 1e-4, stopped
# at 100k steps and judged by the training trend (tools/square_verdict.py). Seeds 1
# and 2 are target-fix's 0rehtuoq and 2cchirbo.
#
# The wandb project and group come from the config: every Square equivariant run
# sits in one group, equi-residual-rl. The run name says which arm and seed;
# "actor-bn" marks the actor's IID batch norm, as for the 2026-10-01 arms.
#
# n_step, gamma, buffer_size and learning_starts are part of the replay-buffer
# cache key, so nothing here overrides them: the arm reads the Square caches, and
# the gate fails if it would have to rebuild them.

set -euo pipefail

ARM="${1:?usage: ./submit.sh <arm> <seed> [gpu_index]}"
SEED="${2:?usage: ./submit.sh <arm> <seed> [gpu_index]}"
GPU="${3:-0}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU"

if [[ -z "${CACHE_DIR:-}" || ! -d "$CACHE_DIR/offline_buffer_cache" ]]; then
    echo "error: CACHE_DIR must name the checkout holding offline_buffer_cache/" >&2
    exit 1
fi
if [[ "${DRY_RUN:-0}" != "1" && -z "${TMUX:-}" && "${ALLOW_NO_TMUX:-0}" != "1" ]]; then
    echo "error: launch inside tmux, so a closed terminal cannot kill the run" >&2
    exit 1
fi

PY="${PYTHON:-python}"
if ! "$PY" -c "import torchrl, escnn" >/dev/null 2>&1; then
    echo "error: '$PY' lacks torchrl+escnn. Run 'conda activate residual'," >&2
    echo "       or set PYTHON=/path/to/python" >&2
    exit 1
fi

case "$ARM" in
    # target-fix's overrides, at the image-scaling fix. Nothing else.
    image-fix) ARM_OVERRIDES=() ;;
    *)
        echo "error: unknown arm '$ARM'" >&2
        exit 2
        ;;
esac

NAME="equi-residual-rl-actor-bn-${ARM}-${SEED}"

OVERRIDES=(
    --config-name=residual_equi_td3_square_config
    algo.prefetch_batches=4
    algo.total_timesteps=100000
    agent.actor.actor_last_layer_init_scale=1e-4
    equivariance.actor_norm=iid_bn
    "${ARM_OVERRIDES[@]}"
    "wandb.name=${NAME}"
    "wandb.notes=square-image-fix-1002/${ARM}/seed${SEED}"
    "seed=${SEED}"
    debug=false
)

if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
    echo "Running pre-submission gate..."
    "$PY" preflight.py --task Square --concurrent-seeds 1 --require-caches \
        --compose "${OVERRIDES[@]}"
else
    echo "WARNING: preflight gate skipped via SKIP_PREFLIGHT=1"
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN=1: would launch ${NAME} on GPU ${GPU} with:"
    printf '    %s\n' "${OVERRIDES[@]}"
    exit 0
fi

echo
echo "Launching ${NAME} on GPU ${GPU} (commit $(git rev-parse --short HEAD))"
echo

"$PY" -m resfit.rl_finetuning.scripts.train_residual_td3 "${OVERRIDES[@]}"
