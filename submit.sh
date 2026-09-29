#!/bin/bash
# Rewritable submission script. Overwrite this in place for each new experiment.
# The durable record is docs/EXPERIMENTS.md plus the resolved config in wandb.
#
#   ./submit.sh <arm> <seed> [gpu]      # e.g. ./submit.sh target-fix 1 0
#   DRY_RUN=1 ./submit.sh q10 1         # run the gate for that arm, then stop
#   SKIP_PREFLIGHT=1 ./submit.sh ...    # escape hatch, discouraged
#
# Currently configured for: the Square sweep, seed-groups equi-square-sweep-<arm>.
# Pre-registered in docs/EXPERIMENTS.md, "2026-09-29 — PRE-REGISTERED — Square
# sweep". Launch it inside tmux from a worktree pinned to a commit, with
# CACHE_DIR pointing at the checkout that holds the buffer caches;
# tools/gpu_queue.sh does all three.
#
# WHAT EVERY ARM SHARES
# ---------------------
# The equi-square-v1 reference config, stopped at 150k steps: the screen reads
# its verdict from the 110k-150k evals. Nothing in training depends on
# total_timesteps before the stop (the stddev and clipping schedules are flat),
# so these are the first 150k steps of a 300k run.
#
# The last-layer init scale is set once per arm. The reference ran at 1e-4
# (submit.sh at b8e1de7); the config default is 0.0.
#
# n_step, gamma, buffer_size and learning_starts are part of the replay-buffer
# cache key, so no arm overrides them: every arm reads the same Square caches,
# and the gate fails if it would have to rebuild them.

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

INIT=1e-4
case "$ARM" in
    # The reference config at this commit, whose only change is the target-critic
    # fix (EXPERIMENTS.md, 2026-09-29 CORRECTION). Every other arm has it too.
    target-fix) ARM_OVERRIDES=() ;;
    # Upstream ResFiT's 10 critic heads. The default went from 10 to 2 in aabde0f.
    q10)        ARM_OVERRIDES=(agent.critic.num_q=10) ;;
    # TD targets clamped to [0, 1], the range of a sparse 0/1 return.
    clipq)      ARM_OVERRIDES=(agent.clip_q_target_to_reward_range=true) ;;
    # An L2 penalty pulling the residual towards zero.
    l2)         ARM_OVERRIDES=(agent.actor.action_l2_reg_weight=0.1) ;;
    # He init in the actor as well, so every network keeps escnn's default.
    he-all)     ARM_OVERRIDES=(equivariance.use_orth_init=false) ;;
    *)
        echo "error: unknown arm '$ARM'" >&2
        exit 2
        ;;
esac

GROUP="equi-square-sweep-${ARM}"

OVERRIDES=(
    --config-name=residual_equi_td3_square_config
    algo.prefetch_batches=4
    algo.total_timesteps=150000
    "agent.actor.actor_last_layer_init_scale=${INIT}"
    "${ARM_OVERRIDES[@]}"
    wandb.project=robomimic-square-final
    "wandb.name=${GROUP}-${SEED}"
    "wandb.group=${GROUP}"
    "wandb.notes=square-sweep/${ARM}/seed${SEED}"
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
    echo "DRY_RUN=1: would launch ${GROUP}-${SEED} on GPU ${GPU} with:"
    printf '    %s\n' "${OVERRIDES[@]}"
    exit 0
fi

echo
echo "Launching ${GROUP}-${SEED} on GPU ${GPU} (commit $(git rev-parse --short HEAD))"
echo

"$PY" -m resfit.rl_finetuning.scripts.train_residual_td3 "${OVERRIDES[@]}"
