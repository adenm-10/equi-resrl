#!/bin/bash
# Rewritable submission script. Overwrite this in place for each new experiment.
# The durable record is docs/EXPERIMENTS.md plus the resolved config in wandb.
#
#   ./submit.sh <arm> <seed> [gpu]                     # e.g. ./submit.sh tuned-rlpd-can-control 1 0
#   DRY_RUN=1 ./submit.sh tuned-rlpd-can-control 1     # run the gate for that arm, then stop
#   SKIP_PREFLIGHT=1 ./submit.sh ...                   # escape hatch, discouraged
#
# Currently configured for: the tuned-RLPD screen on Can, pre-registered in docs/EXPERIMENTS.md,
# "2026-10-06 — PRE-REGISTERED — tuned RLPD on Can: upstream's trainer, and a screen of three
# changes" (seed-group tuned-rlpd-can-screen-1006). Launch it inside tmux from a worktree pinned to
# a commit, with CACHE_DIR pointing at the main checkout; tools/gpu_queue.sh does all three.
#
# Earlier experiments launch from the commits they ran at: tuned-rlpd-1002 from a0353a5 (its two
# equivariant arms, still pending Aden's design, are in that commit's submit.sh), the image-fix
# pair from b667647, and the parked Square ablations from ccff539.
#
# THE ARMS
# --------
# All run 30k steps on Can, vanilla RL with no base policy.
#   upstream-rlpd-can              upstream's own trainer, train_rlpd_dexmg.py, as the paper's
#                                  launcher shell/paper_runs/can/2_can_rlpd.sh runs it, with its
#                                  critic heads set back to upstream's 10. It always writes its
#                                  buffer caches (about 16 GB for the online one), so it runs with
#                                  CACHE_DIR=$CACHE_DIR/local_runs/upstream-rlpd and wandb offline:
#                                  its trainer uploads every best checkpoint (STANDARDS 6.4).
#   tuned-rlpd-can-control         train_residual_td3.py at tuned_rlpd_can_config, as lg8qad3l
#                                  ran, now with the critic diagnostics logged.
#   tuned-rlpd-can-actor-lr        actor LR 1e-6, the residual config's, not 1e-4.
#   tuned-rlpd-can-critic-warmup   10k critic-only updates before the actor's first, as the
#                                  residual config does.
#   tuned-rlpd-can-zero-init       the actor's last layer starts at zero, as the residual
#                                  config's does: the first actions are the dataset midpoint.
#   tuned-rlpd-can-combined        all three of the above.

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

SEED_GROUP=tuned-rlpd-can-screen-1006
NAME="${ARM}-${SEED}"
NAMING=("wandb.name=${NAME}" "wandb.notes=${SEED_GROUP}/${ARM}/seed${SEED}")

TRAINER=train_residual_td3
case "$ARM" in
    upstream-rlpd-can)
        TRAINER=train_rlpd_dexmg
        ARM_OVERRIDES=(--config-name=rlpd_can_config algo.prefetch_batches=4 agent.critic.num_q=10
                       wandb.mode=offline wandb.project=robomimic-can-final wandb.group=upstream-rlpd) ;;
    tuned-rlpd-can-control)
        ARM_OVERRIDES=(--config-name=tuned_rlpd_can_config) ;;
    tuned-rlpd-can-actor-lr)
        ARM_OVERRIDES=(--config-name=tuned_rlpd_can_config agent.actor_lr=1e-6) ;;
    tuned-rlpd-can-critic-warmup)
        ARM_OVERRIDES=(--config-name=tuned_rlpd_can_config algo.critic_warmup_steps=10000) ;;
    tuned-rlpd-can-zero-init)
        ARM_OVERRIDES=(--config-name=tuned_rlpd_can_config agent.actor.actor_last_layer_init_scale=0.0) ;;
    tuned-rlpd-can-combined)
        ARM_OVERRIDES=(--config-name=tuned_rlpd_can_config agent.actor_lr=1e-6
                       algo.critic_warmup_steps=10000 agent.actor.actor_last_layer_init_scale=0.0) ;;
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

if [[ "$TRAINER" == "train_rlpd_dexmg" ]]; then
    # Upstream's trainer keeps its caches and its run folder under CACHE_DIR, apart from ours.
    export CACHE_DIR="$CACHE_DIR/local_runs/upstream-rlpd"
    mkdir -p "$CACHE_DIR/offline_buffer_cache"
fi

if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
    echo "Running pre-submission gate..."
    if [[ "$TRAINER" == "train_rlpd_dexmg" ]]; then
        # The gate composes train_residual_td3's configs only; tests/test_sweep_arms.py composes
        # this arm through upstream's.
        "$PY" preflight.py --task Can --method tuned-rlpd --stack resfit --concurrent-seeds 1
        # The gate does not know this trainer writes its online buffer at full capacity, about
        # 16 GB, whatever the run's length. Filling the disk mid-write corrupts it.
        FREE_GB=$(( $(df --output=avail -k "$CACHE_DIR" | tail -1) / 1024 / 1024 ))
        if (( FREE_GB < 26 )); then
            echo "error: $FREE_GB GB free; upstream's trainer writes about 16 GB of cache" >&2
            exit 4
        fi
    else
        "$PY" preflight.py --task Can --method tuned-rlpd --stack resfit --concurrent-seeds 1 \
            --compose "${OVERRIDES[@]}"
    fi
else
    echo "WARNING: preflight gate skipped via SKIP_PREFLIGHT=1"
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN=1: would launch ${NAME} on GPU ${GPU} with ${TRAINER}:"
    printf '    %s\n' "${OVERRIDES[@]}"
    exit 0
fi

echo
echo "Launching ${NAME} on GPU ${GPU} (commit $(git rev-parse --short HEAD)) with ${TRAINER}"
echo

"$PY" -m "resfit.rl_finetuning.scripts.${TRAINER}" "${OVERRIDES[@]}"
