#!/bin/bash
# Rewritable submission script. Overwrite this in place for each new experiment.
# The durable record is docs/EXPERIMENTS.md plus the resolved config in wandb.
#
#   ./submit.sh <arm> <seed> [gpu]                     # e.g. ./submit.sh resfit-tuned-rlpd-can 1 0
#   DRY_RUN=1 ./submit.sh resfit-tuned-rlpd-can 1      # run the gate for that arm, then stop
#   SKIP_PREFLIGHT=1 ./submit.sh ...                   # escape hatch, discouraged
#
# Currently configured for: the fixed-env baselines, pre-registered in docs/EXPERIMENTS.md,
# "2026-10-07 — PRE-REGISTERED — the fixed-env baselines" (seed-group fixed-env-baselines-1007).
# Launch them inside tmux from a worktree pinned to a commit, with CACHE_DIR pointing at the main
# checkout; tools/gpu_queue.sh does all three.
#
#   SMOKE=1 ./submit.sh <arm> <seed> [gpu]   # the arm's own overrides for 1,000 steps after 300
#                                            # learning starts, no eval, recorded offline in
#                                            # wandb project equi-resrl-smoke: speed and RAM
#
# Earlier experiments launch from the commits they ran at: tuned RLPD on Can with no warm-up
# (tuned-rlpd-can-no-warmup-1007) from f581d9c, the BC term training the encoder
# (tuned-rlpd-can-bc-encoder-1007) from 05fd8d0, τ and the BC term (tuned-rlpd-can-tau-bc-1006)
# from ed47957, the Can screen (tuned-rlpd-can-screen-1006) from 187d895, tuned-rlpd-1002 from
# a0353a5, the image-fix pair from b667647, and the parked Square ablations from ccff539. Upstream's
# code out of the box (upstream-clean-1007) does not launch from here:
# ~/projects/resfit-upstream-runs/launch.sh runs it, in its own clone and env.
#
# THE ARMS
# --------
# train_residual_td3.py for 150k steps, mujoco 3.3.2, both buffers filled in the run
# (read_buffer_caches=false write_buffer_caches=false). Tuned RLPD on Can is vanilla RL with no base
# policy; residual RL on Square corrects the BC policy run_3hzs5bz1_best:v9, pinned.
#   resfit-tuned-rlpd-can           tuned_rlpd_can_config as it is. Its seed 1 is the validation.
#   equi-tuned-rlpd-can             equi_tuned_rlpd_can_config at Aden's design.
#   resfit-residual-square          residual_td3_square_config with 10 critic heads.
#   equi-residual-square            residual_equi_td3_square_config at Aden's design, the residual's
#                                   last layer scaled by 1e-4 at init.
#   equi-residual-square-zero-init  the same, scaled by 0: the residual starts at exactly zero.
# Aden's design: equi_tanh on the actor, 10 critic heads, two hidden layers per head with field norm,
# no norm in the actor. wandb: both tuned-RLPD arms in robomimic-can-final / tuned-rlpd, ResFiT's
# Square residual in robomimic-square-final / residual-rl, the equivariant ones in equi-residual-rl:
# where each kind of run has gone before.

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

SEED_GROUP=fixed-env-baselines-1007
DESIGN=(
    equivariance.residual_squash=true
    agent.critic.num_q=10
    equivariance.num_critic_layers=2
    equivariance.critic_norm=field_norm
    equivariance.actor_norm=none
)

case "$ARM" in
    resfit-tuned-rlpd-can)
        ARM_OVERRIDES=(--config-name=tuned_rlpd_can_config)
        PROJECT=robomimic-can-final GROUP=tuned-rlpd
        GATE=(--task Can --method tuned-rlpd --stack resfit) ;;
    equi-tuned-rlpd-can)
        ARM_OVERRIDES=(--config-name=equi_tuned_rlpd_can_config "${DESIGN[@]}")
        PROJECT=robomimic-can-final GROUP=tuned-rlpd
        GATE=(--task Can --method tuned-rlpd --stack equi) ;;
    resfit-residual-square)
        ARM_OVERRIDES=(--config-name=residual_td3_square_config agent.critic.num_q=10
                       base_policy.wt_version=v9)
        PROJECT=robomimic-square-final GROUP=residual-rl
        GATE=(--task Square --method residual --stack resfit) ;;
    equi-residual-square)
        ARM_OVERRIDES=(--config-name=residual_equi_td3_square_config "${DESIGN[@]}"
                       agent.actor.actor_last_layer_init_scale=1e-4 base_policy.wt_version=v9)
        PROJECT=robomimic-square-final GROUP=equi-residual-rl
        GATE=(--task Square --method residual --stack equi) ;;
    equi-residual-square-zero-init)
        ARM_OVERRIDES=(--config-name=residual_equi_td3_square_config "${DESIGN[@]}"
                       agent.actor.actor_last_layer_init_scale=0.0 base_policy.wt_version=v9)
        PROJECT=robomimic-square-final GROUP=equi-residual-rl
        GATE=(--task Square --method residual --stack equi) ;;
    *)
        echo "error: unknown arm '$ARM'" >&2
        exit 2
        ;;
esac

if [[ "${SMOKE:-0}" == "1" ]]; then
    # The arm as it launches, briefly, and kept apart from the real runs' records.
    NAME="smoke-${ARM}-${SEED}" NOTES="smoke/${SEED_GROUP}/${ARM}/seed${SEED}"
    PROJECT=equi-resrl-smoke GROUP=smoke STEPS=1000
    RUN_LENGTH=(algo.learning_starts=300 eval_first=false eval_interval_every_steps=100000
                wandb.mode=offline)
else
    NAME="${ARM}-${SEED}" NOTES="${SEED_GROUP}/${ARM}/seed${SEED}" STEPS=150000
    RUN_LENGTH=()
fi
NAMING=("wandb.project=${PROJECT}" "wandb.group=${GROUP}" "wandb.name=${NAME}" "wandb.notes=${NOTES}")

OVERRIDES=(
    "${ARM_OVERRIDES[@]}"
    "algo.total_timesteps=${STEPS}"
    read_buffer_caches=false
    write_buffer_caches=false
    "${RUN_LENGTH[@]}"
    "${NAMING[@]}"
    "seed=${SEED}"
    debug=false
)
# Hydra's own folder for the run (.hydra/ and the job log) goes under CACHE_DIR with every other
# record, not into the launch worktree. It sets where Hydra writes, not the run's config.
RECORDS=("hydra.run.dir=$(cd "$CACHE_DIR" && pwd)/outputs/\${now:%Y-%m-%d}/\${now:%H-%M-%S}")

if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
    echo "Running pre-submission gate..."
    # Two runs at a time, one per GPU (the pre-registration's resources).
    "$PY" preflight.py "${GATE[@]}" --concurrent-seeds 2 --compose "${OVERRIDES[@]}"
else
    echo "WARNING: preflight gate skipped via SKIP_PREFLIGHT=1"
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN=1: would launch ${NAME} on GPU ${GPU}:"
    printf '    %s\n' "${OVERRIDES[@]}"
    printf '  with %s\n' "${RECORDS[@]}"
    exit 0
fi

echo
echo "Launching ${NAME} on GPU ${GPU} (commit $(git rev-parse --short HEAD))"
echo

# exec: bash reads nothing more of this file once the trainer starts, so a rewrite cannot reach it.
exec "$PY" -m resfit.rl_finetuning.scripts.train_residual_td3 "${OVERRIDES[@]}" "${RECORDS[@]}"
