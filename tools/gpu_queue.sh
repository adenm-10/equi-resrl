#!/bin/bash
# Run queued launches on one GPU, one after another, so the GPU never idles.
#
#   tools/gpu_queue.sh <gpu> <queue-file>
#
# Each line of the queue file is "<commit> <arm> <seed>". The runner takes the
# first such line and deletes it before launching, so a crash cannot repeat
# it, then runs ./submit.sh <arm> <seed> <gpu> in the worktree pinned to that
# commit, $HOME/projects/equi-resrl-<commit>. Blank lines and # comments are
# skipped, a line STOP ends the runner, and an empty queue is re-read every
# minute. A failed launch is logged and the queue moves on.
#
# Keep the queue file outside every repo, so editing it never dirties a tree,
# and run this script from a pinned worktree too: bash reads a script while it
# runs, so editing the copy being executed corrupts the runner.
#
# Logs go to $CACHE_DIR/outputs/queue/: gpu<gpu>.log records every launch, and
# each launch's console output goes to its own file beside it.

set -uo pipefail

GPU="${1:?usage: tools/gpu_queue.sh <gpu> <queue-file>}"
QUEUE="${2:?usage: tools/gpu_queue.sh <gpu> <queue-file>}"

[[ -n "${TMUX:-}" ]] || { echo "error: run the queue inside tmux" >&2; exit 1; }
[[ -n "${CACHE_DIR:-}" ]] || { echo "error: set CACHE_DIR to the checkout holding the caches" >&2; exit 1; }
[[ -f "$QUEUE" ]] || { echo "error: no queue file $QUEUE" >&2; exit 1; }

LOG_DIR="$CACHE_DIR/outputs/queue"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/gpu${GPU}.log"
log() { echo "$(date '+%F %T') gpu${GPU} $*" | tee -a "$LOG"; }

log "runner started on $QUEUE (pid $$)"
while true; do
    N="$(grep -n -m1 -vE '^[[:space:]]*(#|$)' "$QUEUE" | cut -d: -f1)"
    if [[ -z "$N" ]]; then
        sleep 60
        continue
    fi
    LINE="$(sed -n "${N}p" "$QUEUE")"
    sed -i "${N}d" "$QUEUE"

    read -r COMMIT ARM SEED EXTRA <<< "$LINE"
    if [[ "$COMMIT" == "STOP" ]]; then
        log "STOP read; runner exits"
        exit 0
    fi
    if [[ -z "${SEED:-}" || -n "${EXTRA:-}" ]]; then
        log "SKIP malformed line: '$LINE' (want '<commit> <arm> <seed>')"
        continue
    fi

    WT="$HOME/projects/equi-resrl-$COMMIT"
    HEAD="$(git -C "$WT" rev-parse HEAD 2>/dev/null || true)"
    if [[ "$HEAD" != "$COMMIT"* ]]; then
        log "SKIP $ARM $SEED: $WT is not a worktree at $COMMIT (HEAD '${HEAD:-none}')"
        continue
    fi
    DIRTY="$(git -C "$WT" status --porcelain | grep -vx '?? artifacts' || true)"
    if [[ -n "$DIRTY" ]]; then
        log "SKIP $ARM $SEED: $WT is not clean: $(echo "$DIRTY" | tr '\n' ' ')"
        continue
    fi

    OUT="$LOG_DIR/$(date '+%Y%m%d-%H%M%S')_${ARM}-${SEED}_gpu${GPU}.log"
    log "LAUNCH $ARM seed $SEED at $(git -C "$WT" log -1 --format='%h %s') -> $OUT"
    START=$(date +%s)
    (cd "$WT" && ./submit.sh "$ARM" "$SEED" "$GPU") 2>&1 | tee "$OUT"
    CODE=${PIPESTATUS[0]}
    log "EXIT $CODE for $ARM seed $SEED after $(( ($(date +%s) - START) / 60 )) min"
done
