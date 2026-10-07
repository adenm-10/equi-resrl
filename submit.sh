#!/bin/bash
# Rewritable submission script. Overwrite this in place for each new experiment.
# The durable record is docs/EXPERIMENTS.md plus the resolved config in wandb.
#
#   ./submit.sh <arm> <seed> [gpu]                     # e.g. ./submit.sh tuned-rlpd-can-bc-encoder 1 1
#   DRY_RUN=1 ./submit.sh tuned-rlpd-can-bc-encoder 1  # run the gate for that arm, then stop
#   SKIP_PREFLIGHT=1 ./submit.sh ...                   # escape hatch, discouraged
#
# Currently configured for: the BC term training the encoder, tuned RLPD on Can, pre-registered in
# docs/EXPERIMENTS.md, "2026-10-07 — PRE-REGISTERED — tuned RLPD on Can: the BC term trains the
# encoder" (seed-group tuned-rlpd-can-bc-encoder-1007). Launch it inside tmux from a worktree pinned
# to a commit, with CACHE_DIR pointing at the main checkout; tools/gpu_queue.sh does all three.
#
# Earlier experiments launch from the commits they ran at: τ and the BC term
# (tuned-rlpd-can-tau-bc-1006) from ed47957, the Can screen (tuned-rlpd-can-screen-1006) from
# 187d895, tuned-rlpd-1002 from a0353a5 (its two equivariant arms, still pending Aden's design, are
# in that commit's submit.sh), the image-fix pair from b667647, and the parked Square ablations
# from ccff539. Upstream's code out of the box (upstream-clean-1007) does not launch from here:
# ~/projects/resfit-upstream-runs/launch.sh runs it, in its own clone and env.
#
# THE ARM
# -------
# train_residual_td3.py at tuned_rlpd_can_config for 30k steps on Can, vanilla RL with no base
# policy. Its reference is tuned-rlpd-can-bc (kc8z7vsf, at ed47957).
#   tuned-rlpd-can-bc-encoder   the BC term at 0.025, as tuned-rlpd-can-bc, and its loss also trains
#                               the image encoder: agent.bc_backprop_encoder=true.

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

SEED_GROUP=tuned-rlpd-can-bc-encoder-1007
NAME="${ARM}-${SEED}"
NAMING=("wandb.name=${NAME}" "wandb.notes=${SEED_GROUP}/${ARM}/seed${SEED}")

case "$ARM" in
    tuned-rlpd-can-bc-encoder)
        ARM_OVERRIDES=(--config-name=tuned_rlpd_can_config agent.bc_loss_coef=0.025
                       agent.bc_backprop_encoder=true) ;;
    *)
        echo "error: unknown arm '$ARM'" >&2
        exit 2
        ;;
esac

OVERRIDES=(
    "${ARM_OVERRIDES[@]}"
    algo.total_timesteps=30000
    "${NAMING[@]}"
    "seed=${SEED}"
    debug=false
)

if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
    echo "Running pre-submission gate..."
    "$PY" preflight.py --task Can --method tuned-rlpd --stack resfit --concurrent-seeds 1 \
        --compose "${OVERRIDES[@]}"
else
    echo "WARNING: preflight gate skipped via SKIP_PREFLIGHT=1"
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN=1: would launch ${NAME} on GPU ${GPU}:"
    printf '    %s\n' "${OVERRIDES[@]}"
    exit 0
fi

echo
echo "Launching ${NAME} on GPU ${GPU} (commit $(git rev-parse --short HEAD))"
echo

"$PY" -m resfit.rl_finetuning.scripts.train_residual_td3 "${OVERRIDES[@]}"
