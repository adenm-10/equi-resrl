#!/bin/bash
# Rewritable submission script. Overwrite this in place for each new experiment.
# The durable record is docs/EXPERIMENTS.md plus the resolved config in wandb.
#
#   ./submit.sh <arm> <seed> [gpu]                     # e.g. ./submit.sh tuned-rlpd-can-tau 1 0
#   DRY_RUN=1 ./submit.sh tuned-rlpd-can-tau 1         # run the gate for that arm, then stop
#   SKIP_PREFLIGHT=1 ./submit.sh ...                   # escape hatch, discouraged
#
# Currently configured for: τ and a BC term for tuned RLPD on Can, pre-registered in
# docs/EXPERIMENTS.md, "2026-10-06 — PRE-REGISTERED — tuned RLPD on Can: τ 0.005 and a BC term"
# (seed-group tuned-rlpd-can-tau-bc-1006). Launch it inside tmux from a worktree pinned to a commit,
# with CACHE_DIR pointing at the main checkout; tools/gpu_queue.sh does all three.
#
# Earlier experiments launch from the commits they ran at: the Can screen
# (tuned-rlpd-can-screen-1006) from 187d895, tuned-rlpd-1002 from a0353a5 (its two equivariant
# arms, still pending Aden's design, are in that commit's submit.sh), the image-fix pair from
# b667647, and the parked Square ablations from ccff539.
#
# THE ARMS
# --------
# All run train_residual_td3.py at tuned_rlpd_can_config for 30k steps on Can, vanilla RL with no
# base policy. Their control is the screen's, tuned-rlpd-can-control (r1hvijqh, at 187d895).
#   tuned-rlpd-can-tau      critic_target_tau 0.005, the paper's stated value, not rlpd.py's 0.01.
#   tuned-rlpd-can-bc       a BC term on the demo half: agent.bc_loss_coef=0.025.
#   tuned-rlpd-can-tau-bc   both.

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

SEED_GROUP=tuned-rlpd-can-tau-bc-1006
NAME="${ARM}-${SEED}"
NAMING=("wandb.name=${NAME}" "wandb.notes=${SEED_GROUP}/${ARM}/seed${SEED}")

case "$ARM" in
    tuned-rlpd-can-tau)
        ARM_OVERRIDES=(--config-name=tuned_rlpd_can_config agent.critic_target_tau=0.005) ;;
    tuned-rlpd-can-bc)
        ARM_OVERRIDES=(--config-name=tuned_rlpd_can_config agent.bc_loss_coef=0.025) ;;
    tuned-rlpd-can-tau-bc)
        ARM_OVERRIDES=(--config-name=tuned_rlpd_can_config agent.critic_target_tau=0.005
                       agent.bc_loss_coef=0.025) ;;
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
