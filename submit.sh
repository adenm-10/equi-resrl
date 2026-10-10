#!/bin/bash
# Rewritable submission script. Overwrite this in place for each new experiment.
# The durable record is docs/EXPERIMENTS.md plus the resolved config in wandb.
#
#   ./submit.sh <arm> <seed> [gpu]                     # e.g. ./submit.sh equi-tuned-rlpd-square-clipq 1 0
#   DRY_RUN=1 ./submit.sh equi-tuned-rlpd-square-clipq 1  # run the gate for that arm, then stop
#   SKIP_PREFLIGHT=1 ./submit.sh ...                   # escape hatch, discouraged
#
# Currently configured for: stabilizing equivariant tuned RLPD on Square (seed-group
# square-trlpd-stab-1010), pre-registered in docs/EXPERIMENTS.md, "2026-10-10 — PRE-REGISTERED —
# stabilizing equivariant tuned RLPD on Square". Launch inside tmux from a worktree pinned to a
# commit, with CACHE_DIR pointing at the main checkout.
#
#   SMOKE=1 ./submit.sh <arm> <seed> [gpu]   # the arm's own overrides for 1,000 steps after 300
#                                            # learning starts, no eval, recorded offline in
#                                            # wandb project equi-resrl-smoke: speed and RAM
#
# Earlier experiments launch from the commits they ran at: square-rerendered-1009's arms (its
# anchor for this group, equi-tuned-rlpd-square-rerendered, among them) from eb90a0c;
# critic-norm-can-1008's arms from 6b6df8c (nonorm, layernorm), 0a9626f
# (resfit-tuned-rlpd-can-nolayernorm), e3793ce (the demo-codec arm) and 6e61e65 (ln-matched); the
# fixed-env baselines (fixed-env-baselines-1007) from 9cf207d, tuned RLPD on Can with no warm-up
# (tuned-rlpd-can-no-warmup-1007) from f581d9c, the BC term training the encoder
# (tuned-rlpd-can-bc-encoder-1007) from 05fd8d0, τ and the BC term (tuned-rlpd-can-tau-bc-1006) from
# ed47957, the Can screen (tuned-rlpd-can-screen-1006) from 187d895, tuned-rlpd-1002 from a0353a5,
# the image-fix pair from b667647, and the parked Square ablations from ccff539. Upstream's code
# out of the box (upstream-clean-1007) does not launch from here: ~/projects/resfit-upstream-runs/
# launch.sh runs it, in its own clone and env.
#
# THE ARMS
# --------
# train_residual_td3.py, equi_tuned_rlpd_square_config, on Square for up to 300k steps (the
# config's default), mujoco 3.3.2, both buffers filled in the run (read_buffer_caches=false
# write_buffer_caches=false), the demos re-rendered by our env
# (offline_data.name=local/robomimic-mh-square-rerendered), the env's frames uncoded, the actor
# squashed by equi_tanh (equivariance.residual_squash=true), no norm unless an arm says otherwise.
# Each arm is vc5o977f's launch (equi-tuned-rlpd-square-rerendered 1, eb90a0c) plus one change;
# wandb robomimic-square-final / tuned-rlpd.
#   equi-tuned-rlpd-square-clipq     5 heads of one hidden layer; the TD target clamped to [0, 1]
#                                    (agent.clip_q_target_to_reward_range=true).
#   equi-tuned-rlpd-square-pooledln  5 heads of one hidden layer; LayerNorm on each head's pooled
#                                    features (equivariance.critic_pooled_norm=layer_norm), each
#                                    head's last Linear scaled at init to start as spread as the
#                                    anchor's critic.
#   equi-tuned-rlpd-square-ln        5 heads of one hidden layer; RegularLayerNorm after every
#                                    hidden Linear of the actor and the critic, where ResFiT's
#                                    use_layer_norm puts LayerNorm, the start matched likewise.
#   equi-tuned-rlpd-square-10x2      10 heads of two hidden layers each, gkck6eqp's critic.

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

SEED_GROUP=square-trlpd-stab-1010
SQUASH=equivariance.residual_squash=true
FIVE_HEADS=agent.critic.num_q=5
DEMOS=offline_data.name=local/robomimic-mh-square-rerendered
# Each head's last Linear at init, scaled to start as spread as the anchor's critic (tools/critic_start.py).
POOLED_LN_SCALE=0.8835
LN_SCALE=0.8168

case "$ARM" in
    equi-tuned-rlpd-square-clipq)
        ARM_OVERRIDES=(--config-name=equi_tuned_rlpd_square_config "$SQUASH" "$FIVE_HEADS" "$DEMOS"
                       agent.clip_q_target_to_reward_range=true) ;;
    equi-tuned-rlpd-square-pooledln)
        ARM_OVERRIDES=(--config-name=equi_tuned_rlpd_square_config "$SQUASH" "$FIVE_HEADS" "$DEMOS"
                       equivariance.critic_pooled_norm=layer_norm
                       "equivariance.critic_last_layer_init_scale=${POOLED_LN_SCALE}") ;;
    equi-tuned-rlpd-square-ln)
        ARM_OVERRIDES=(--config-name=equi_tuned_rlpd_square_config "$SQUASH" "$FIVE_HEADS" "$DEMOS"
                       equivariance.actor_norm=layer_norm equivariance.critic_norm=layer_norm
                       "equivariance.critic_last_layer_init_scale=${LN_SCALE}") ;;
    equi-tuned-rlpd-square-10x2)
        ARM_OVERRIDES=(--config-name=equi_tuned_rlpd_square_config "$SQUASH" "$DEMOS"
                       agent.critic.num_q=10 equivariance.num_critic_layers=2) ;;
    *)
        echo "error: unknown arm '$ARM'" >&2
        exit 2
        ;;
esac
PROJECT=robomimic-square-final GROUP=tuned-rlpd
GATE=(--task Square --method tuned-rlpd --stack equi)

if [[ "${SMOKE:-0}" == "1" ]]; then
    # The arm as it launches, briefly, and kept apart from the real runs' records.
    NAME="smoke-${ARM}-${SEED}" NOTES="smoke/${SEED_GROUP}/${ARM}/seed${SEED}"
    PROJECT=equi-resrl-smoke GROUP=smoke STEPS=1000
    RUN_LENGTH=(algo.learning_starts=300 eval_first=false eval_interval_every_steps=100000
                wandb.mode=offline)
else
    NAME="${ARM}-${SEED}" NOTES="${SEED_GROUP}/${ARM}/seed${SEED}" STEPS=300000
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
    # CONCURRENT: the runs not yet going that will be once this wave is up, this one included (runs
    # already going are out of MemAvailable already): 3, 2 and 1 for clipq, pooledln and ln, each
    # launched once the one before is training, then 1 (the pre-registration's resources).
    "$PY" preflight.py "${GATE[@]}" --concurrent-seeds "${CONCURRENT:-1}" --compose "${OVERRIDES[@]}"
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
