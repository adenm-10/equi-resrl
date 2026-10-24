---
name: wrapup
description: Clean up the equi-resrl repo and the machine state around it, periodically or at the end of a session. Inventories uncommitted changes, untracked and gitignored files, tracked files nothing uses, launch worktrees, branches, tmux sessions and disk; gives every item one fate from a fixed table (keep, commit, park, delete, ask Aden); finishes, parks or reverts unfinished work by a fixed rule; applies nothing until Aden has answered the table; and ends with a paste-ready entrypoint paragraph that lets a fresh session find everything this session produced. Use when the user asks to clean up or tidy the repo, reclaim disk, deal with leftover or unfinished work, or wants a handoff paragraph for the next session.
---

# Wrapup

Goal: nothing left in the repo or on the machine that the docs cannot explain, and no record lost
getting there. Everything that stays has a reason; everything that goes is scratch, or Aden said yes.

This skill handles files, processes and unfinished work. Writing the session into the docs is
`/session-end`'s job: if this session's work is not in the docs yet, run it first, then come back
here. Do not write the docs a second way.

## The rules

- **The record is not cleanup.** Never delete, move or rewrite these unless Aden names them: run
  packages in `outputs/runs/`, `wandb/run-*` records, `artifacts/`, Hydra folders under
  `outputs/<date>/`, the buffer caches, `docs/papers/`, `~/equi-resrl-preserve/`, any file a doc
  cites (grep the docs for its path), and any past EXPERIMENTS.md entry. A run deleted from wandb is
  restored from its local record; a deleted local record is gone for good.
- **Live work is not cleanup.** A running process, a job in an armed queue, the worktree either one
  runs from, and every tmux session STATUS.md lists stay, however finished they look. Runs end on
  their own at 150k; stopping one is Aden's call.
- **Nothing is applied until Aden has answered the table.** Delete rows can be approved as a batch,
  Ask rows one by one. Reading and listing need no approval.
- **Not a refactor.** Removing code a recorded config can still select changes what a run can reach,
  which is priority 1's business, not cleanup's. Respect any freeze in STATUS.md.

## 1. Take inventory (read-only)

Read STATUS.md's running section and the top of TODO.md first, so you know which runs, queues,
watchers and worktrees are meant to exist right now. Then:

```bash
git status --short; git stash list; git branch -vv; git worktree list
ps -eo pid,etime,args | grep -E 'train_|queue|watch' | grep -v grep
for s in $(tmux ls -F '#{session_name}'); do     # 0 children: an idle shell, its job has ended
  echo "$s $(pgrep -P "$(tmux list-panes -t "$s" -F '#{pane_pid}' | head -1)" | wc -l)"; done
df -h .; du -sh wandb outputs artifacts *_buffer_cache 2>/dev/null
cat CLAUDE.md docs/*.md .claude/skills/*/SKILL.md .claude/settings.json | wc -c   # budget 150 KB
```

A launch worktree is in use if a live process runs from it (`readlink /proc/<pid>/cwd`) or its path
appears under `outputs/queue/launchers/`. If the docs are over budget, say by how much and which
docs could shrink. EXPERIMENTS.md cannot (append-only), and condensing the others is `/session-end`'s
step 2.

In the tracked tree, look for what nothing reaches: a module nothing imports (STANDARDS 4.3), a tool
no test, script, queue or doc runs, a test of a tool that is gone, a launch script for no recorded
run, and media or binaries under `docs/` (CLAUDE.md: docs are plain markdown). "Nothing reaches it"
means a grep of the repo, `outputs/queue/launchers/` and the docs for its name, not a guess from its
age.

## 2. Sort every item into one table

One row per item, and every item gets exactly one fate:

| Fate | When | What happens |
|---|---|---|
| **Keep** | Record, live, or local on purpose and already ignored | Nothing; one summary line per category, not a row each |
| **Commit** | This session's finished work: tests pass, docs written | `/git_proc`, docs in the same commit |
| **Park** | Unfinished work, by the rule in step 3 | A `wip/` branch and a TODO item |
| **Delete** | Regenerable or scratch and cited by no doc: bytecode, `.pytest_cache`, stray logs, empty dirs, this session's throwaway scripts and smoke outputs | Deleted |
| **Ask** | Everything else: idle tmux sessions, unused worktrees and branches, disk reclaims, local-on-purpose files git still shows, tracked files nothing reaches | Aden decides |

Each Ask row says what the item is, its size, why it looks unneeded, the check that makes removing
it safe (a hash match with the copy that stays, no process in the worktree), and a recommendation.

Show the table grouped by section (session work, unfinished work, files, processes, disk) so Aden
can answer a section at a time. Then stop and wait.

## 3. Unfinished work

Unfinished means not ready for main: code half-written, a config field with no branch behind it
(STANDARDS 1.2), a failing test, a plan not yet approved, a run that crashed or was stopped. Apply
the first case that fits:

1. **Finish it** if what is left is small, inside what Aden asked for, and the gate can check it.
   It becomes a Commit row.
2. **Park it** otherwise. Commit it to `wip/<YYYY-MM-DD>-<slug>` via `/git_proc`, return main to
   HEAD for those paths, and add a TODO.md item naming the branch, what works, what does not, and
   the next step. Never a stash: nothing a new session reads points at one.
3. **Revert it** only when Aden says the idea is dropped, and record the decision and its reasoning
   in TODO.md or PROGRESS.md. The reasoning is what a diff cannot give back.

Not code: a plan in `docs/plans/` stays, with a status line at its top (draft, approved, done,
superseded). A crashed or stopped run gets its EXPERIMENTS.md entry like any other, and its package
stays.

## 4. Apply, then verify

In order: Commit rows, Park rows, then the approved removals. Before deleting, write the list of
what goes under `outputs/cleanup-<YYYYMMDD>/`, and save an idle tmux session's scrollback there
before killing it. Unlink a worktree's `artifacts` symlink before removing it (TODO B15 has the
commands). The cleanup's own changes (removed tracked files, `.gitignore` additions, and the doc
lines they make stale) go in one commit separate from the session's work; its message names each
removed file and the last commit that has it. Give the cleanup one line in the session's PROGRESS
entry.

So the next cleanup does not ask the same question twice: an untracked file Aden keeps goes into
`.gitignore` under a labelled `# --- ... ---` section, and a tracked file Aden keeps gets the header
comment STANDARDS 4.3 asks for.

Then check:

- `git status --short` is empty, or shows only what Aden chose to leave.
- If tracked code was removed: `python -m pytest tests/test_imports.py -q`, and
  `python preflight.py --skip-tier2` if it was anything a launch reaches.
- No doc links to a removed file.
- Disk free, before and after.

## 5. Write the entrypoint paragraph

End with one paragraph in a fenced block, ready to paste as the first message of a new session. A
reader with no memory of this session must be able to find from it everything the session produced.
Every claim carries its pointer:

- What was done and decided, in two sentences, and the PROGRESS.md entry's heading.
- The commits by short SHA, and any `wip/` branch with what is left on it.
- The EXPERIMENTS.md entries by date and tag, the seed-group ids, run ids and package paths.
- New plans, tools and diagnostics, by path.
- What is still running: its tmux names and when it should end.
- What waits on Aden.
- Where to start: `/session-start`, then the docs and code to read first to build on this work.

Use only facts already in the docs, and add no new reading of them. If a pointer cannot be found,
the docs are missing something: fix that through `/session-end` rather than writing around it.

After the paragraph, a few lines: committed, parked, deleted, left for Aden, disk reclaimed.
