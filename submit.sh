#!/bin/bash
# Rewritable submission script. Overwrite this in place for each new experiment.
# The durable record is docs/EXPERIMENTS.md plus the resolved config in wandb.
#
#   ./submit.sh <arm> <seed> [gpu]                               # e.g. ./submit.sh resfit-residual-square-noln 1 0
#   DRY_RUN=1 ./submit.sh resfit-residual-square-noln 1           # run the gate for that arm, then stop
#   SKIP_PREFLIGHT=1 ./submit.sh ...                             # escape hatch, discouraged
#
# Currently configured for: ResFiT's residual on Square given the equivariant stack's differences
# (seed-group resfit-equi-diff-1012), pre-registered in docs/EXPERIMENTS.md, "2026-10-12 —
# PRE-REGISTERED — ResFiT's residual on Square, given the equivariant stack's differences one at a
# time; the equivariant residual given ResFiT's layout". Its second group, equi-resfit-arch-1013,
# joins this table with its code. Launch inside tmux from a worktree pinned to a commit, with
# CACHE_DIR pointing at the main checkout.
#
#   SMOKE=1 ./submit.sh <arm> <seed> [gpu]   # the arm's own overrides for 1,000 steps after 300
#                                            # learning starts, no eval, recorded offline in
#                                            # wandb project equi-resrl-smoke: speed and RAM
#
# Earlier experiments launch from the commits they ran at: residual-tilt-1012's arms from be0b7bb;
# resfit-rx-1012's arm from 60fce72; square-residual-start-1011's arms from 0af7655;
# square-trlpd-stab-1010's arms from 6b2acc6; square-rerendered-1009's arms from eb90a0c;
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
# resfit-equi-diff-1012: train_residual_td3.py, residual_td3_square_config, on Square for 150k
# steps, mujoco 3.3.2, both buffers filled in the run (read_buffer_caches=false
# write_buffer_caches=false), 10 critic heads, the BC policy v9. Each arm is oj5k0jqb's launch
# (resfit-residual-square 1, 9cf207d) plus one change, on its original demos unless the change is
# the demos; wandb robomimic-square-final / residual-rl.
#   resfit-residual-square-rerendered  the demos every equivariant run trains on, re-rendered by
#                                      our env (offline_data.name=local/robomimic-mh-square-rerendered).
#   resfit-residual-square-noln        no LayerNorm in the actor or the critic
#                                      (agent.actor.use_layer_norm=false
#                                      agent.critic.use_layer_norm=false); the ViT encoders keep
#                                      theirs.
# Later arms (limits, noln-all; then equi-resfit-arch-1013's) join with the code they need.

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

# oj5k0jqb's launch, less its run length and naming: the anchor every ResFiT arm is one change from.
RESFIT_ANCHOR=(--config-name=residual_td3_square_config agent.critic.num_q=10 base_policy.wt_version=v9)
RERENDERED=offline_data.name=local/robomimic-mh-square-rerendered

case "$ARM" in
    resfit-residual-square-rerendered)
        ARM_OVERRIDES=("${RESFIT_ANCHOR[@]}" "$RERENDERED") ;;
    resfit-residual-square-noln)
        ARM_OVERRIDES=("${RESFIT_ANCHOR[@]}" agent.actor.use_layer_norm=false
                       agent.critic.use_layer_norm=false) ;;
    *)
        echo "error: unknown arm '$ARM'" >&2
        exit 2
        ;;
esac
case "$ARM" in
    resfit-*)
        SEED_GROUP=resfit-equi-diff-1012 PROJECT=robomimic-square-final GROUP=residual-rl
        GATE=(--task Square --method residual --stack resfit) ;;
esac
ARM_STEPS=150000

if [[ "${SMOKE:-0}" == "1" ]]; then
    # The arm as it launches, briefly, and kept apart from the real runs' records.
    NAME="smoke-${ARM}-${SEED}" NOTES="smoke/${SEED_GROUP}/${ARM}/seed${SEED}"
    PROJECT=equi-resrl-smoke GROUP=smoke STEPS=1000
    RUN_LENGTH=(algo.learning_starts=300 eval_first=false eval_interval_every_steps=100000
                wandb.mode=offline)
else
    NAME="${ARM}-${SEED}" NOTES="${SEED_GROUP}/${ARM}/seed${SEED}" STEPS=${ARM_STEPS}
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
    # already going are out of MemAvailable already): 2 when two arms launch at once, else 1 (the
    # pre-registration's resources).
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
