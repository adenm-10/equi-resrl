---
name: session-end
description: Close out a work session by writing everything into the docs so a fresh session can pick up from them. Updates STATUS, PROGRESS, TODO, and EXPERIMENTS with what was done, what was decided, and what the immediate and long-term next steps are; takes any easy, safe opportunities to condense code or docs without losing functionality or readability; then asks the user whether to commit everything to git. Use at the end of a session, or when the user says they are wrapping up, stopping for now, or wants the session written up for handoff.
---

# Session end

Goal: leave the repo in a state where a fresh session reading only the docs can continue without
asking what happened. Assume the reader has no memory of this session.

Priorities for this skill, in order: **correctness, then conciseness.** Never trade away a correct
detail to make a doc shorter.

## 1. Update the docs

**[docs/PROGRESS.md](../../../docs/PROGRESS.md)** — prepend a new dated entry. Follow the existing
entry format: Done / Findings / Decisions / Next. Keep it short; anything needing a paragraph goes
in another doc with a pointer from here. Record **decisions and their reasoning**, not just actions
— reasoning is the part that cannot be recovered from a diff.

**[docs/STATUS.md](../../../docs/STATUS.md)** — rewrite in place, do not append. Update what works,
what does not, the active question, the last verified commit, and any freeze. If nothing changed,
say so rather than churning the text.

**[docs/TODO.md](../../../docs/TODO.md)** — mark finished items done or remove them, add anything
discovered, re-order if priorities moved. If an item was resolved by a user decision, record the
decision and the reasoning rather than deleting the item silently.

**[docs/EXPERIMENTS.md](../../../docs/EXPERIMENTS.md)** — append an entry for any run launched or
finished. **Append-only: never edit or delete a past entry.** If an old entry turns out to be wrong,
add a new one correcting it and leave the original. Use the schema at the top of the file, including
the required `Config caveats` field — check the provenance table in ARCHITECTURE.md for the run's
commit, since some logged hyperparameters do not affect the network.

Also update **[docs/ARCHITECTURE.md](../../../docs/ARCHITECTURE.md)**,
**[docs/EQUIVARIANCE.md](../../../docs/EQUIVARIANCE.md)**, or
**[docs/STANDARDS.md](../../../docs/STANDARDS.md)** if code moved, a symmetry decision was made, or
a convention was established.

Write in plain English where plain English does the job. These are handoff documents first, so
completeness beats brevity when the two genuinely conflict.

## 2. Condense, carefully

Look for easy, safe reductions in code and docs — duplicated explanations across docs, stale
sections, dead code, redundant helpers. Take the ones that are clearly safe.

Hard limits:

- **No functionality change.** If a reduction could alter behavior, leave it and add a TODO instead.
- **No readability loss.** Shorter is not better if it is harder to follow.
- **Respect any active freeze** in STATUS.md. Frozen modules are not touched, however tempting the
  cleanup.
- **Never delete a past EXPERIMENTS.md entry**, even a redundant-looking one.

If you find nothing safe to condense, say so. A clean pass is a fine outcome.

## 3. Verify

- Internal doc links resolve.
- Docs agree with each other — especially STATUS.md against the actual `git log`.
- Size check: `du -ch $(git ls-files docs .claude CLAUDE.md) | tail -1` stays under 150 KB.
- If any code changed, confirm it still imports.

## 4. Ask about committing

Summarize in a few lines what changed, then **ask the user whether to commit everything to git.**
Do not commit without an answer.

If they say yes: stage the doc updates and code changes together in one commit, write a message
describing the session's substance (not just "update docs"), and report the result. If the branch is
`main`, ask before committing directly to it.
