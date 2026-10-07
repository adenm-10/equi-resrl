#!/bin/bash
# Run queued launches on one GPU, one after another, so the GPU never idles.
#
#   tools/gpu_queue.sh <gpu> <queue-file>
#
# Each line of the queue file is "<commit> <arm> <seed>". The runner takes the
# first such line and deletes it before launching, so a crash cannot repeat it,
# then runs ./submit.sh <arm> <seed> <gpu> in the worktree pinned to that commit,
# $HOME/projects/equi-resrl-<commit>. Blank lines and # comments are skipped, a
# line STOP ends the runner, and an empty queue is re-read every minute
# ($QUEUE_POLL_SECONDS, 60). A failed launch is logged and the queue moves on.
#
# A line launches only when both of these hold; otherwise the runner logs why,
# leaves the line in place, and tries again a minute later:
#   - the disk under $CACHE_DIR has at least $QUEUE_MIN_FREE_GB GB free (5);
#   - the line really was deleted. On a full disk `sed -i` cannot write and the
#     line stays first: a runner that launched anyway started the same arm 44
#     times in 45 seconds on 2026-10-07.
#
# Run it from the main checkout. Its body is one function, called on the last
# line, so bash has read the whole script before it runs anything, and editing
# this file cannot corrupt a runner already running. Keep the queue file outside
# every repo, so editing it never dirties a tree.
#
# Logs go to $CACHE_DIR/outputs/queue/: gpu<gpu>.log records every launch, and
# each launch's console output goes to its own file beside it.

set -uo pipefail

log() {
    local line
    line="$(date '+%F %T') gpu${GPU} $*"
    echo "$line"
    # A line cut short by a full disk ends without a newline. Start a fresh line, so the
    # next entry is not appended to it (on 2026-10-07 an EXIT line was).
    if [[ -s "$LOG" && -n "$(tail -c1 "$LOG")" ]]; then
        printf '\n' >> "$LOG" 2>/dev/null
    fi
    printf '%s\n' "$line" >> "$LOG" 2>/dev/null || echo "gpu${GPU}: could not write $LOG" >&2
}

# Delete line $1 of the queue. Fails if sed fails or the file did not change.
delete_line() {
    local before
    before="$(cksum < "$QUEUE")"
    sed -i "${1}d" "$QUEUE" 2>/dev/null && [[ "$(cksum < "$QUEUE")" != "$before" ]]
}

main() {
    GPU="${1:?usage: tools/gpu_queue.sh <gpu> <queue-file>}"
    QUEUE="${2:?usage: tools/gpu_queue.sh <gpu> <queue-file>}"
    local min_free_gb="${QUEUE_MIN_FREE_GB:-5}"
    local poll_s="${QUEUE_POLL_SECONDS:-60}"

    [[ -n "${TMUX:-}" ]] || { echo "error: run the queue inside tmux" >&2; exit 1; }
    [[ -n "${CACHE_DIR:-}" ]] || { echo "error: set CACHE_DIR to the main checkout" >&2; exit 1; }
    [[ -f "$QUEUE" ]] || { echo "error: no queue file $QUEUE" >&2; exit 1; }

    local log_dir="$CACHE_DIR/outputs/queue"
    mkdir -p "$log_dir"
    LOG="$log_dir/gpu${GPU}.log"

    log "runner started on $QUEUE (pid $$)"
    local n line commit arm seed extra free_kb wt head dirty out start code
    while true; do
        n="$(grep -n -m1 -vE '^[[:space:]]*(#|$)' "$QUEUE" | cut -d: -f1)"
        if [[ -z "$n" ]]; then
            sleep "$poll_s"
            continue
        fi
        line="$(sed -n "${n}p" "$QUEUE")"
        read -r commit arm seed extra <<< "$line"

        if [[ "$commit" == "STOP" ]]; then
            delete_line "$n" || log "could not delete the STOP line; it stays in $QUEUE"
            log "STOP read; runner exits"
            exit 0
        fi

        free_kb="$(df --output=avail -k "$CACHE_DIR" | tail -1 | tr -d ' ')"
        if (( free_kb < min_free_gb * 1024 * 1024 )); then
            log "WAIT '$line': $(( free_kb / 1024 / 1024 )) GB free under $CACHE_DIR," \
                "below $min_free_gb GB; the line stays queued"
            sleep "$poll_s"
            continue
        fi
        if ! delete_line "$n"; then
            log "WAIT '$line': could not delete it from $QUEUE; not launching"
            sleep "$poll_s"
            continue
        fi

        if [[ -z "${seed:-}" || -n "${extra:-}" ]]; then
            log "SKIP malformed line: '$line' (want '<commit> <arm> <seed>')"
            continue
        fi

        wt="$HOME/projects/equi-resrl-$commit"
        head="$(git -C "$wt" rev-parse HEAD 2>/dev/null || true)"
        if [[ "$head" != "$commit"* ]]; then
            log "SKIP $arm $seed: $wt is not a worktree at $commit (HEAD '${head:-none}')"
            continue
        fi
        dirty="$(git -C "$wt" status --porcelain)"
        if [[ -n "$dirty" ]]; then
            log "SKIP $arm $seed: $wt is not clean: $(echo "$dirty" | tr '\n' ' ')"
            continue
        fi

        out="$log_dir/$(date '+%Y%m%d-%H%M%S')_${arm}-${seed}_gpu${GPU}.log"
        log "LAUNCH $arm seed $seed at $(git -C "$wt" log -1 --format='%h %s') -> $out"
        start=$(date +%s)
        (cd "$wt" && ./submit.sh "$arm" "$seed" "$GPU") 2>&1 | tee "$out"
        code=${PIPESTATUS[0]}
        log "EXIT $code for $arm seed $seed after $(( ($(date +%s) - start) / 60 )) min"
    done
}

main "$@"; exit $?
