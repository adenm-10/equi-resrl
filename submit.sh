#!/bin/bash
# Rewritable submission script. Overwrite this in place for each new experiment.
# The durable record is docs/EXPERIMENTS.md plus the resolved config in wandb.
#
#   ./submit.sh <arm> <seed> [gpu]                     # e.g. ./submit.sh equi-tuned-rlpd-can-nonorm 1 1
#   DRY_RUN=1 ./submit.sh equi-tuned-rlpd-can-nonorm 1 # run the gate for that arm, then stop
#   SKIP_PREFLIGHT=1 ./submit.sh ...                   # escape hatch, discouraged
#
# Currently configured for: equivariant tuned RLPD on Can without FieldNorm in the critic,
# pre-registered in docs/EXPERIMENTS.md, "2026-10-08 — PRE-REGISTERED — equivariant tuned RLPD on Can
# without FieldNorm", and ResFiT's tuned RLPD without LayerNorm, added by "2026-10-08 —
# PRE-REGISTERED — a third arm: ResFiT's tuned RLPD on Can without LayerNorm", and the no-norm
# arm with the env's frames through the demos' video codec, added by "2026-10-08 — PRE-REGISTERED —
# a fourth arm: the no-norm arm with env frames coded like the demos'", and RegularLayerNorm where
# ResFiT has LayerNorm, added by "2026-10-08 — PRE-REGISTERED — a fifth arm: RegularLayerNorm in the
# actor and the critic, its start matched to ResFiT's" (seed-group critic-norm-can-1008). Launch
# inside tmux from a worktree pinned to
# a commit, with CACHE_DIR pointing at the main checkout; tools/gpu_queue.sh does all three.
#
#   SMOKE=1 ./submit.sh <arm> <seed> [gpu]   # the arm's own overrides for 1,000 steps after 300
#                                            # learning starts, no eval, recorded offline in
#                                            # wandb project equi-resrl-smoke: speed and RAM
#
# Earlier experiments launch from the commits they ran at: the fixed-env baselines
# (fixed-env-baselines-1007) from 9cf207d, tuned RLPD on Can with no warm-up
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
# (read_buffer_caches=false write_buffer_caches=false). Tuned RLPD on Can: vanilla RL, no base
# policy. The equivariant arms are the fixed-env baselines' equi-tuned-rlpd-can (eyyi99q4) with
# its critic's FieldNorm replaced, which erased every trivial-type input (2026-10-08 FINDING):
#   equi-tuned-rlpd-can-nonorm      equivariance.critic_norm=none: no norm in the critic.
#   equi-tuned-rlpd-can-layernorm   equivariance.critic_norm=layer_norm: RegularLayerNorm, a
#                                   LayerNorm over each sample's whole layer, as ResFiT's critic has.
#   resfit-tuned-rlpd-can-nolayernorm  the fixed-env baselines' resfit-tuned-rlpd-can (d047it6c) with
#                                   no LayerNorm in its critic and actor MLPs; its ViTs keep theirs.
#   equi-tuned-rlpd-can-nonorm-democodec  the nonorm arm (xm9cuo92) with every env camera frame
#                                   through the demos' video codec (encode_env_frames_like_demos):
#                                   rollouts, evals and the online buffer.
#   equi-tuned-rlpd-can-ln-matched  RegularLayerNorm after every hidden Linear of the actor and the
#                                   critic, where ResFiT's use_layer_norm puts LayerNorm, and each
#                                   critic head's last Linear scaled by 0.3 at init, which matches
#                                   the heads' starting spread to ResFiT's.
# The rest of Aden's design as before: equi_tanh on the actor, 10 critic heads, two hidden layers
# per head, no norm in the actor. wandb: robomimic-can-final / tuned-rlpd, with the other
# tuned-RLPD arms.

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

SEED_GROUP=critic-norm-can-1008
DESIGN=(
    equivariance.residual_squash=true
    agent.critic.num_q=10
    equivariance.num_critic_layers=2
    equivariance.actor_norm=none
)

case "$ARM" in
    equi-tuned-rlpd-can-nonorm)
        ARM_OVERRIDES=(--config-name=equi_tuned_rlpd_can_config "${DESIGN[@]}"
                       equivariance.critic_norm=none)
        PROJECT=robomimic-can-final GROUP=tuned-rlpd
        GATE=(--task Can --method tuned-rlpd --stack equi) ;;
    equi-tuned-rlpd-can-layernorm)
        ARM_OVERRIDES=(--config-name=equi_tuned_rlpd_can_config "${DESIGN[@]}"
                       equivariance.critic_norm=layer_norm)
        PROJECT=robomimic-can-final GROUP=tuned-rlpd
        GATE=(--task Can --method tuned-rlpd --stack equi) ;;
    equi-tuned-rlpd-can-nonorm-democodec)
        ARM_OVERRIDES=(--config-name=equi_tuned_rlpd_can_config "${DESIGN[@]}"
                       equivariance.critic_norm=none encode_env_frames_like_demos=true)
        PROJECT=robomimic-can-final GROUP=tuned-rlpd
        GATE=(--task Can --method tuned-rlpd --stack equi) ;;
    equi-tuned-rlpd-can-ln-matched)
        # RegularLayerNorm in the actor too, so not "${DESIGN[@]}", which sets actor_norm=none.
        ARM_OVERRIDES=(--config-name=equi_tuned_rlpd_can_config equivariance.residual_squash=true
                       agent.critic.num_q=10 equivariance.num_critic_layers=2
                       equivariance.actor_norm=layer_norm equivariance.critic_norm=layer_norm
                       equivariance.critic_last_layer_init_scale=0.3)
        PROJECT=robomimic-can-final GROUP=tuned-rlpd
        GATE=(--task Can --method tuned-rlpd --stack equi) ;;
    resfit-tuned-rlpd-can-nolayernorm)
        ARM_OVERRIDES=(--config-name=tuned_rlpd_can_config agent.critic.use_layer_norm=false
                       agent.actor.use_layer_norm=false)
        PROJECT=robomimic-can-final GROUP=tuned-rlpd
        GATE=(--task Can --method tuned-rlpd --stack resfit) ;;
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
