#!/bin/bash
# Rewritable submission script. Overwrite this in place for each new experiment.
# The durable record is docs/EXPERIMENTS.md plus the resolved config in wandb.
#
#   ./submit.sh <arm> <seed> [gpu]                               # e.g. ./submit.sh resfit-residual-square-noln 1 0
#   DRY_RUN=1 ./submit.sh resfit-residual-square-noln 1           # run the gate for that arm, then stop
#   SKIP_PREFLIGHT=1 ./submit.sh ...                             # escape hatch, discouraged
#
# Currently configured for: the encoder's interface (seed-group enc-iface-1023), arch-swap-eee's agent
# with each of three wider interfaces from its encoder to its critic, pre-registered in
# docs/EXPERIMENTS.md, "2026-10-23 — PRE-REGISTERED — the encoder's interface (enc-iface-1023)". The
# groups before it stay in this table, each as it launched: arch-swap-1021 ("2026-10-20 —
# PRE-REGISTERED — the architecture swap"; from aec1bdc), equi-speed-matched-1017 ("2026-10-17 — PRE-REGISTERED — the
# equivariant residual on Square with the actor's parts at matched speed, and the speed's two
# edges"; from 8e8eceb), equi-actor-speed-1016 ("2026-10-16 — PRE-REGISTERED — the equivariant residual on
# Square with arch-wide's actor alone, and with a faster actor"; from dd6f753),
# equi-arch-parts-1015 ("2026-10-15 — PRE-REGISTERED — the equivariant residual on
# Square with each part of ResFiT's layout alone, and all four again"; its seed 3s launch from
# 974e281), resfit-equi-diff-1012 and equi-resfit-arch-1013 ("2026-10-12 — PRE-REGISTERED —
# ResFiT's residual on Square, given the equivariant stack's differences one at a time; the
# equivariant residual given ResFiT's layout", and that date's amendment), equi-residual-long-1012
# (the equivariant residual's anchors again, to a fixed 150k), and resfit-ln-place-1013 (which of
# ResFiT's LayerNorms its residual needs, "2026-10-13 — PRE-REGISTERED — ResFiT's residual on Square
# with LayerNorm off in one place"). equi-residual-square-arch-wide now launches into
# equi-arch-parts-1015; its seed 1 (grvwhcfr) launched into equi-resfit-arch-1013 from 95490f9.
# Launch inside tmux from a worktree pinned to a commit, with CACHE_DIR pointing at the main
# checkout.
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
#   resfit-residual-square-noln-all    noln, and the ViT encoders' LayerNorms too
#                                      (agent.vit.use_layer_norm=false): no LayerNorm anywhere.
#   resfit-residual-square-limits      the equivariant stack's action limits, in env units
#                                      (agent.actor.action_bound=equivariant): the residual's xy
#                                      and tilt pairs each in a circle of radius 0.1 and the rest
#                                      within ±0.1; the critic scores the sum clipped as equi_clip
#                                      clips it; the env clips the executed action to its box.
#   resfit-residual-square-combined    rerendered + noln + limits, launched iff each holds alone.
#   resfit-residual-square-smallcritic the equivariant residual's critic heads: 5 of 1 hidden layer
#                                      (agent.critic.num_q=5 agent.critic.num_layers=1), at
#                                      ResFiT's width.
#   resfit-residual-square-noise       the equivariant stack's noise: σ in env units on every axis
#                                      (agent.actor.noise_units=env).
# rerendered and noln launched from 7db124d; limits and noln-all from their CODE commits; combined,
# smallcritic and noise (AMENDMENT of 2026-10-12, "every run to a fixed 150k") from noise's.
#
# equi-resfit-arch-1013: the equivariant residual, residual_equi_td3_square_config, on Square for
# 150k steps, on the re-rendered demos. Each arm is eaqorqki's launch (square-residual-start-1011,
# 0af7655) with ResFiT's layout; wandb robomimic-square-final / equi-residual-rl. As amended on
# 2026-10-12 (Aden): the actor keeps its depth, which is ResFiT's already.
#   equi-residual-square-arch          RegularLayerNorm in the actor and the critic where ResFiT
#                                      has LayerNorm, and ResFiT's 10 critic heads of 2 hidden
#                                      layers (agent.critic.num_q=10 equivariance.num_critic_layers=2).
#   equi-residual-square-arch-wide     arch, with the actor at 448 fields
#                                      (equivariance.actor_degree_channel=448): ResFiT's actor's
#                                      parameter count, nearly.
#
# equi-residual-long-1012: the equivariant residual's two anchors again, each its own launch for a
# fixed 150k, at seed 2; wandb robomimic-square-final / equi-residual-rl.
#   equi-residual-square-long          6ng62tpm's launch (square-rerendered-1009): the actor's last
#                                      layer at 1e-4.
#   equi-residual-square-init0-long    eaqorqki's launch (square-residual-start-1011): at zero.
#
# resfit-ln-place-1013: oj5k0jqb's launch with LayerNorm off in one of the three places
# resfit-residual-square-noln and noln-all turn it off; wandb robomimic-square-final / residual-rl.
#   resfit-residual-square-noln-actor  none in the actor (agent.actor.use_layer_norm=false): after
#                                      its compression and each hidden layer.
#   resfit-residual-square-noln-critic none in the critic (agent.critic.use_layer_norm=false): after
#                                      its input projection and each head's hidden layers.
#   resfit-residual-square-noln-vit    none in the two ViT encoders (agent.vit.use_layer_norm=false):
#                                      two per transformer layer and a final one.
#
# equi-arch-parts-1015: eaqorqki's launch (the init0 anchor) with arch-wide's four parts, each
# alone, and all four again, to a fixed 150k at seed 2, which pairs each with oav5gk76 (init0 at
# seed 2); wandb robomimic-square-final / equi-residual-rl.
#   equi-residual-square-arch-wide     all four below (as in equi-resfit-arch-1013).
#   equi-residual-square-actorln       RegularLayerNorm after each of the actor's three hidden
#                                      Linears (equivariance.actor_norm=layer_norm).
#   equi-residual-square-criticln      RegularLayerNorm after the critic's two projections and each
#                                      head's hidden Linear (equivariance.critic_norm=layer_norm).
#   equi-residual-square-heads         ResFiT's 10 critic heads of 2 hidden layers
#                                      (agent.critic.num_q=10 equivariance.num_critic_layers=2).
#   equi-residual-square-actor448      the actor at 448 fields (equivariance.actor_degree_channel=448).
#
# equi-actor-speed-1016: eaqorqki's launch again, at seed 2 (paired with oav5gk76); wandb
# robomimic-square-final / equi-residual-rl.
#   equi-residual-square-actorln448    arch-wide's actor alone: actorln and actor448 at once
#                                      (equivariance.actor_norm=layer_norm
#                                      equivariance.actor_degree_channel=448), init0's critic.
#   equi-residual-square-actorlr1e-5   the actor's learning rate at 1e-5, 10x (agent.actor_lr=1e-5).
#   equi-residual-square-actorlr3e-5   at 3e-5, 30x (agent.actor_lr=3e-5).
#
# equi-speed-matched-1017: each arm is a parent arm above, at seed 2 (so its parent's seed-2 run's
# starting weights, bit for bit), plus the actor's learning rate (agent.actor_lr); wandb
# robomimic-square-final / equi-residual-rl.
#   equi-residual-square-actorln-lr3e-6    actorln (8sgfkskv) at 3e-6: the 2x2's norm cell.
#   equi-residual-square-actor448-lr3e-6   actor448 (vnpqtnva) at 3e-6: the 2x2's width cell.
#   equi-residual-square-actorln448-lr1e-5 actorln448 (x3xopn1y) at 1e-5: everything stacked.
#   equi-residual-square-actorlr1e-4       init0-long (oav5gk76) at 1e-4: the plain actor's rate
#                                          curve's fourth point.
#
# arch-swap-1021: the equivariant residual's agent, residual_equi_td3_square_config, on Square for
# 150k steps, on the re-rendered demos, at seeds 2 and 3. Each arm is eaqorqki's launch with each of its
# three networks, [encoder, actor, critic], of a family (equivariance.encoder_family, actor_family,
# critic_family): equivariant (E, the escnn network), ablated (A, the same network exported to plain
# torch layers, its weights unconstrained) or resfit (R, ResFiT's own, wrapped). An E or A critic has
# arch-wide's heads (10 of 2 hidden layers, RegularLayerNorm). Every actor steps at ResFiT's 1e-6, the
# config's, but aaa's, at a quarter of it: its residual grew 4 to 6.5 times as fast as eee's at one rate.
# The critic and encoder keep 1e-4. wandb robomimic-square-final / equi-residual-rl.
#   arch-swap-eee   [E, E, E] arch-wide's launch again, at the families' commit: the anchor.
#   arch-swap-aaa   [A, A, A] arch-wide's networks, unconstrained (agent.actor_lr=2.5e-7).
#   arch-swap-ere   [E, R, E] ResFiT's actor on the equivariant encoder and critic.
#   arch-swap-ara   [A, R, A] ResFiT's actor on the ablated encoder and critic.
#   arch-swap-rrr   [R, R, R] ResFiT's networks in this agent, at ResFiT's rates.
#   arch-swap-rra   [R, R, A] ResFiT's ViTs and actor, the ablated critic behind them.
#
# enc-iface-1023: arch-swap-eee's launch (arch-wide's networks), on Square for 150k steps, on the
# re-rendered demos, at seeds 2 and 3, with the interface from its encoders to its critic grown in one
# of three ways (docs/plans/2026-10-23-encoder-interface.md). The critic sees 288 numbers per frame in
# the anchor (256 from the agentview camera, 32 from the wrist); ResFiT's sees 20,736. Every rate is
# the anchor's: the actor's 1e-6, the critic's and the encoder's 1e-4. wandb robomimic-square-final /
# equi-residual-rl.
#   iface-eee        arch-swap-eee's launch again, at the interface's commit: the anchor. 288 / 288.
#   iface-wide-all   every encoder twice as wide: 64 agentview fields, the wrist CNN's stages doubled,
#                    64 wrist outputs. 576 to the critic and to the actor.
#   iface-wide-out   the last layers alone wider: the agentview encoder's last conv to 128 fields, the
#                    wrist's to 256. 1,280 to the critic and to the actor.
#   iface-window     the critic's window: a 9x9 conv over each encoder's 9x9 map, 96 regular fields
#                    from the agentview's and 224 channels from the wrist's, into the critic alone
#                    through a Linear that starts at zero. 1,280 to the critic, 288 to the actor.
#
# Every arm here runs to its full 150k. From 2026-10-17 (DEVIATION) no automatic stop of any kind
# applies, the divergence stop included: Aden stops a run by hand.

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
# smallcritic sets its own head count.
RESFIT_BASE=(--config-name=residual_td3_square_config base_policy.wt_version=v9)
RESFIT_ANCHOR=("${RESFIT_BASE[@]}" agent.critic.num_q=10)
NOLN=(agent.actor.use_layer_norm=false agent.critic.use_layer_norm=false)
RERENDERED=offline_data.name=local/robomimic-mh-square-rerendered
# eaqorqki's launch, less its run length, its naming and its head count (5), which every arm on it
# sets. 6ng62tpm's is the same but for its actor's last layer, at 1e-4.
EQUI_BASE=(--config-name=residual_equi_td3_square_config equivariance.residual_squash=true
           base_policy.wt_version=v9 "$RERENDERED")
EQUI_ANCHOR=("${EQUI_BASE[@]}" agent.actor.actor_last_layer_init_scale=0.0)
RESFIT_LAYOUT=(equivariance.actor_norm=layer_norm equivariance.critic_norm=layer_norm
               agent.critic.num_q=10 equivariance.num_critic_layers=2)
# arch-wide's critic heads, for an equivariant or ablated critic beside another actor (arch-swap-1021).
SWAP_HEADS=(equivariance.critic_norm=layer_norm agent.critic.num_q=10 equivariance.num_critic_layers=2)

# arch-swap-eee's, the anchor of enc-iface-1023.
IFACE_ANCHOR=("${EQUI_ANCHOR[@]}" "${RESFIT_LAYOUT[@]}" equivariance.actor_degree_channel=448)

case "$ARM" in
    iface-eee)
        ARM_OVERRIDES=("${IFACE_ANCHOR[@]}") ;;
    iface-wide-all)
        ARM_OVERRIDES=("${IFACE_ANCHOR[@]}" equivariance.enc_degree_channel=64
                       "equivariance.ih_channels=[32,64,128,256]" equivariance.ih_out_channel=64) ;;
    iface-wide-out)
        ARM_OVERRIDES=("${IFACE_ANCHOR[@]}" equivariance.enc_out_channel=128 equivariance.ih_out_channel=256) ;;
    iface-window)
        ARM_OVERRIDES=("${IFACE_ANCHOR[@]}" equivariance.critic_window=96 equivariance.critic_window_ih=224) ;;
    arch-swap-eee)
        ARM_OVERRIDES=("${EQUI_ANCHOR[@]}" "${RESFIT_LAYOUT[@]}" equivariance.actor_degree_channel=448) ;;
    arch-swap-aaa)
        ARM_OVERRIDES=("${EQUI_ANCHOR[@]}" "${RESFIT_LAYOUT[@]}" equivariance.actor_degree_channel=448
                       equivariance.encoder_family=ablated equivariance.actor_family=ablated
                       equivariance.critic_family=ablated agent.actor_lr=2.5e-7) ;;
    arch-swap-ere)
        ARM_OVERRIDES=("${EQUI_ANCHOR[@]}" "${SWAP_HEADS[@]}" equivariance.actor_family=resfit) ;;
    arch-swap-ara)
        ARM_OVERRIDES=("${EQUI_ANCHOR[@]}" "${SWAP_HEADS[@]}" equivariance.encoder_family=ablated
                       equivariance.actor_family=resfit equivariance.critic_family=ablated) ;;
    arch-swap-rrr)
        ARM_OVERRIDES=("${EQUI_ANCHOR[@]}" agent.critic.num_q=10 equivariance.encoder_family=resfit
                       equivariance.actor_family=resfit equivariance.critic_family=resfit) ;;
    arch-swap-rra)
        ARM_OVERRIDES=("${EQUI_ANCHOR[@]}" "${SWAP_HEADS[@]}" equivariance.encoder_family=resfit
                       equivariance.actor_family=resfit equivariance.critic_family=ablated) ;;
    resfit-residual-square-rerendered)
        ARM_OVERRIDES=("${RESFIT_ANCHOR[@]}" "$RERENDERED") ;;
    resfit-residual-square-noln)
        ARM_OVERRIDES=("${RESFIT_ANCHOR[@]}" "${NOLN[@]}") ;;
    resfit-residual-square-noln-all)
        ARM_OVERRIDES=("${RESFIT_ANCHOR[@]}" "${NOLN[@]}" agent.vit.use_layer_norm=false) ;;
    resfit-residual-square-limits)
        ARM_OVERRIDES=("${RESFIT_ANCHOR[@]}" agent.actor.action_bound=equivariant) ;;
    resfit-residual-square-combined)
        ARM_OVERRIDES=("${RESFIT_ANCHOR[@]}" "$RERENDERED" "${NOLN[@]}" agent.actor.action_bound=equivariant) ;;
    resfit-residual-square-smallcritic)
        ARM_OVERRIDES=("${RESFIT_BASE[@]}" agent.critic.num_q=5 agent.critic.num_layers=1) ;;
    resfit-residual-square-noise)
        ARM_OVERRIDES=("${RESFIT_ANCHOR[@]}" agent.actor.noise_units=env) ;;
    resfit-residual-square-noln-actor)
        ARM_OVERRIDES=("${RESFIT_ANCHOR[@]}" agent.actor.use_layer_norm=false) ;;
    resfit-residual-square-noln-critic)
        ARM_OVERRIDES=("${RESFIT_ANCHOR[@]}" agent.critic.use_layer_norm=false) ;;
    resfit-residual-square-noln-vit)
        ARM_OVERRIDES=("${RESFIT_ANCHOR[@]}" agent.vit.use_layer_norm=false) ;;
    equi-residual-square-arch)
        ARM_OVERRIDES=("${EQUI_ANCHOR[@]}" "${RESFIT_LAYOUT[@]}") ;;
    equi-residual-square-arch-wide)
        ARM_OVERRIDES=("${EQUI_ANCHOR[@]}" "${RESFIT_LAYOUT[@]}" equivariance.actor_degree_channel=448) ;;
    equi-residual-square-long)
        ARM_OVERRIDES=("${EQUI_BASE[@]}" agent.critic.num_q=5 agent.actor.actor_last_layer_init_scale=1e-4) ;;
    equi-residual-square-init0-long)
        ARM_OVERRIDES=("${EQUI_ANCHOR[@]}" agent.critic.num_q=5) ;;
    equi-residual-square-actorln)
        ARM_OVERRIDES=("${EQUI_ANCHOR[@]}" agent.critic.num_q=5 equivariance.actor_norm=layer_norm) ;;
    equi-residual-square-criticln)
        ARM_OVERRIDES=("${EQUI_ANCHOR[@]}" agent.critic.num_q=5 equivariance.critic_norm=layer_norm) ;;
    equi-residual-square-heads)
        ARM_OVERRIDES=("${EQUI_ANCHOR[@]}" agent.critic.num_q=10 equivariance.num_critic_layers=2) ;;
    equi-residual-square-actor448)
        ARM_OVERRIDES=("${EQUI_ANCHOR[@]}" agent.critic.num_q=5 equivariance.actor_degree_channel=448) ;;
    equi-residual-square-actorln448)
        ARM_OVERRIDES=("${EQUI_ANCHOR[@]}" agent.critic.num_q=5 equivariance.actor_norm=layer_norm
                       equivariance.actor_degree_channel=448) ;;
    equi-residual-square-actorlr1e-5)
        ARM_OVERRIDES=("${EQUI_ANCHOR[@]}" agent.critic.num_q=5 agent.actor_lr=1e-5) ;;
    equi-residual-square-actorlr3e-5)
        ARM_OVERRIDES=("${EQUI_ANCHOR[@]}" agent.critic.num_q=5 agent.actor_lr=3e-5) ;;
    equi-residual-square-actorln-lr3e-6)
        ARM_OVERRIDES=("${EQUI_ANCHOR[@]}" agent.critic.num_q=5 equivariance.actor_norm=layer_norm
                       agent.actor_lr=3e-6) ;;
    equi-residual-square-actor448-lr3e-6)
        ARM_OVERRIDES=("${EQUI_ANCHOR[@]}" agent.critic.num_q=5 equivariance.actor_degree_channel=448
                       agent.actor_lr=3e-6) ;;
    equi-residual-square-actorln448-lr1e-5)
        ARM_OVERRIDES=("${EQUI_ANCHOR[@]}" agent.critic.num_q=5 equivariance.actor_norm=layer_norm
                       equivariance.actor_degree_channel=448 agent.actor_lr=1e-5) ;;
    equi-residual-square-actorlr1e-4)
        ARM_OVERRIDES=("${EQUI_ANCHOR[@]}" agent.critic.num_q=5 agent.actor_lr=1e-4) ;;
    *)
        echo "error: unknown arm '$ARM'" >&2
        exit 2
        ;;
esac
case "$ARM" in
    iface-*)
        SEED_GROUP=enc-iface-1023 PROJECT=robomimic-square-final GROUP=equi-residual-rl
        GATE=(--task Square --method residual --stack equi) ;;
    arch-swap-*)
        SEED_GROUP=arch-swap-1021 PROJECT=robomimic-square-final GROUP=equi-residual-rl
        GATE=(--task Square --method residual --stack equi) ;;
    resfit-residual-square-noln-actor|resfit-residual-square-noln-critic|resfit-residual-square-noln-vit)
        SEED_GROUP=resfit-ln-place-1013 PROJECT=robomimic-square-final GROUP=residual-rl
        GATE=(--task Square --method residual --stack resfit) ;;
    resfit-*)
        SEED_GROUP=resfit-equi-diff-1012 PROJECT=robomimic-square-final GROUP=residual-rl
        GATE=(--task Square --method residual --stack resfit) ;;
    equi-residual-square-actorln-lr3e-6|equi-residual-square-actor448-lr3e-6|equi-residual-square-actorln448-lr1e-5|equi-residual-square-actorlr1e-4)
        SEED_GROUP=equi-speed-matched-1017 PROJECT=robomimic-square-final GROUP=equi-residual-rl
        GATE=(--task Square --method residual --stack equi) ;;
    equi-residual-square-actorln448|equi-residual-square-actorlr1e-5|equi-residual-square-actorlr3e-5)
        SEED_GROUP=equi-actor-speed-1016 PROJECT=robomimic-square-final GROUP=equi-residual-rl
        GATE=(--task Square --method residual --stack equi) ;;
    equi-residual-square-arch-wide|equi-residual-square-actorln|equi-residual-square-criticln|equi-residual-square-heads|equi-residual-square-actor448)
        SEED_GROUP=equi-arch-parts-1015 PROJECT=robomimic-square-final GROUP=equi-residual-rl
        GATE=(--task Square --method residual --stack equi) ;;
    equi-residual-square-arch*)
        SEED_GROUP=equi-resfit-arch-1013 PROJECT=robomimic-square-final GROUP=equi-residual-rl
        GATE=(--task Square --method residual --stack equi) ;;
    equi-residual-square-*long)
        SEED_GROUP=equi-residual-long-1012 PROJECT=robomimic-square-final GROUP=equi-residual-rl
        GATE=(--task Square --method residual --stack equi) ;;
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
