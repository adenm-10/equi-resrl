#!/bin/bash
# Rewritable submission script. Overwrite this in place for each new experiment.
# The durable record is docs/EXPERIMENTS.md plus the resolved config in wandb.
#
#   ./submit.sh <arm> <seed> [gpu]      # e.g. ./submit.sh critic-fn 1 0
#   DRY_RUN=1 ./submit.sh clipq 1       # run the gate for that arm, then stop
#   SKIP_PREFLIGHT=1 ./submit.sh ...    # escape hatch, discouraged
#
# Currently configured for: the Square ablations with the actor's batch norm, pre-registered
# in docs/EXPERIMENTS.md, "2026-09-30 — PRE-REGISTERED — Square: the actor-none gate, then
# four ablations", as amended on 2026-10-01 after the gate failed: "2026-10-01 — AMENDMENT
# — the four ablations run with the actor's batch norm". Launch it inside tmux from a
# worktree pinned to a commit, with
# CACHE_DIR pointing at the checkout that holds the buffer caches;
# tools/gpu_queue.sh does all three.
#
# WHAT EVERY ARM SHARES
# ---------------------
# The Square reference config at the new default, stopped at 150k steps: the
# verdict comes from the 110k-150k evals. Nothing in training depends on
# total_timesteps before the stop (the stddev and clipping schedules are flat),
# so these are the first 150k steps of a 300k run.
#
# Every arm has batch norm (iid_bn) at the actor's three sites, as every run before
# 2026-09-30 had. Without it both runs of 2026-09-30 collapsed by 10k steps. The
# default is "none", so it is set explicitly here. The last-layer init scale is
# 1e-4, as the reference ran.
#
# The wandb project and group come from the config: every Square equivariant
# run sits in one group, equi-residual-rl. The run name says which arm and seed,
# and "actor-bn" tells these runs from the 2026-09-30 ones without batch norm.
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

case "$ARM" in
    # The reference: the target-critic fix and the actor's batch norm, nothing else.
    # It ran on 2026-09-29 (0rehtuoq) and was stopped at 124k without a verdict.
    target-fix) ARM_OVERRIDES=() ;;
    # FieldNorm, each field normalized by its own statistics, at the critic's
    # visual projection, trunk and head.
    critic-fn)  ARM_OVERRIDES=(equivariance.critic_norm=field_norm) ;;
    # TD targets clamped to [0, 1], the range of a sparse 0/1 return.
    clipq)      ARM_OVERRIDES=(agent.clip_q_target_to_reward_range=true) ;;
    # squash (the residual's mean bounded by action_scale, as ResFiT's Tanh bounds
    # it) and depth (two hidden layers per critic head, as ResFiT has), each on
    # whichever of critic-fn and clipq passed: neither, either, or both.
    squash)     ARM_OVERRIDES=(equivariance.residual_squash=true) ;;
    depth)      ARM_OVERRIDES=(equivariance.num_critic_layers=2) ;;
    critic-fn-squash)
                ARM_OVERRIDES=(equivariance.critic_norm=field_norm
                               equivariance.residual_squash=true) ;;
    critic-fn-depth)
                ARM_OVERRIDES=(equivariance.critic_norm=field_norm
                               equivariance.num_critic_layers=2) ;;
    clipq-squash)
                ARM_OVERRIDES=(agent.clip_q_target_to_reward_range=true
                               equivariance.residual_squash=true) ;;
    clipq-depth)
                ARM_OVERRIDES=(agent.clip_q_target_to_reward_range=true
                               equivariance.num_critic_layers=2) ;;
    critic-fn-clipq-squash)
                ARM_OVERRIDES=(equivariance.critic_norm=field_norm
                               agent.clip_q_target_to_reward_range=true
                               equivariance.residual_squash=true) ;;
    critic-fn-clipq-depth)
                ARM_OVERRIDES=(equivariance.critic_norm=field_norm
                               agent.clip_q_target_to_reward_range=true
                               equivariance.num_critic_layers=2) ;;
    *)
        echo "error: unknown arm '$ARM'" >&2
        exit 2
        ;;
esac

NAME="equi-residual-rl-actor-bn-${ARM}-${SEED}"

OVERRIDES=(
    --config-name=residual_equi_td3_square_config
    algo.prefetch_batches=4
    algo.total_timesteps=150000
    agent.actor.actor_last_layer_init_scale=1e-4
    equivariance.actor_norm=iid_bn
    "${ARM_OVERRIDES[@]}"
    "wandb.name=${NAME}"
    "wandb.notes=square-ablations-1001/${ARM}/seed${SEED}"
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
