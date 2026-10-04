#!/bin/bash
# Rewritable submission script. Overwrite this in place for each new experiment.
# The durable record is docs/EXPERIMENTS.md plus the resolved config in wandb.
#
#   ./submit.sh <arm> <seed> [gpu]              # e.g. ./submit.sh tuned-rlpd-square 1 0
#   DRY_RUN=1 ./submit.sh tuned-rlpd-square 1   # run the gate for that arm, then stop
#   SKIP_PREFLIGHT=1 ./submit.sh ...            # escape hatch, discouraged
#
# Currently configured for: tuned RLPD, vanilla RL with no base policy, on Can and Square,
# pre-registered in docs/EXPERIMENTS.md, "2026-10-04 — PRE-REGISTERED — tuned RLPD on Can and
# Square" (seed-group tuned-rlpd-1002). Launch it inside tmux from a worktree pinned to a commit,
# with CACHE_DIR pointing at the main checkout; tools/gpu_queue.sh does all three.
#
# The image-fix pair (square-image-fix-1002) launched from b667647, and the parked Square
# ablations (square-ablations-1001) launch from ccff539, the commits they ran at.
#
# THE ARMS
# --------
# Each is its stack's residual design made vanilla, the way the ResFiT paper made ResFiT vanilla
# (TunedRLPDConfig in config/residual_td3.py). The two arms of a task differ in their stack
# only: critic heads, networks, encoder, normalization and action bound.
#   tuned-rlpd-{square,can}       ResFiT's stack: upstream's RLPD config with 10 critic heads.
#   equi-tuned-rlpd-{square,can}  the equivariant stack. PENDING: Aden decides its design once
#                                 the ResFiT arms report (2026-10-04), and until then these arms
#                                 refuse to launch. Their overrides are the plan's provisional
#                                 design: the image-fix reference's IID batch norm in the actor,
#                                 and the actor's output bounded by equi_tanh.
# All four run 150k steps; the verdict (tools/tuned_rlpd_verdict.py) reads the evals at
# 110k-150k. They build their buffers in memory (write_buffer_caches=false), so the gate
# requires no cache.

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

PENDING=0
case "$ARM" in
    tuned-rlpd-square)
        CONFIG=tuned_rlpd_square_config; TASK=Square; STACK=resfit; ARM_OVERRIDES=() ;;
    tuned-rlpd-can)
        CONFIG=tuned_rlpd_can_config; TASK=Can; STACK=resfit; ARM_OVERRIDES=() ;;
    equi-tuned-rlpd-square)
        CONFIG=equi_tuned_rlpd_square_config; TASK=Square; STACK=equi; PENDING=1
        ARM_OVERRIDES=(equivariance.actor_norm=iid_bn equivariance.residual_squash=true) ;;
    equi-tuned-rlpd-can)
        CONFIG=equi_tuned_rlpd_can_config; TASK=Can; STACK=equi; PENDING=1
        ARM_OVERRIDES=(equivariance.actor_norm=iid_bn equivariance.residual_squash=true) ;;
    *)
        echo "error: unknown arm '$ARM'" >&2
        exit 2
        ;;
esac

if [[ "$PENDING" == "1" && "${DRY_RUN:-0}" != "1" ]]; then
    echo "error: $ARM is pending Aden's design decision (TODO B12); only DRY_RUN=1 runs it" >&2
    exit 3
fi

NAME="${ARM}-${SEED}"

OVERRIDES=(
    "--config-name=${CONFIG}"
    algo.total_timesteps=150000
    "${ARM_OVERRIDES[@]}"
    "wandb.name=${NAME}"
    "wandb.notes=tuned-rlpd-1002/${ARM}/seed${SEED}"
    "seed=${SEED}"
    debug=false
)

if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
    echo "Running pre-submission gate..."
    "$PY" preflight.py --task "$TASK" --method tuned-rlpd --stack "$STACK" --concurrent-seeds 1 \
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
