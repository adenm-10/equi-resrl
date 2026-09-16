---
name: session-start
description: Start a work session on the equi-resrl project by catching up on state. Reads the four state docs (STATUS, PROGRESS, TODO, EXPERIMENTS), checks them against git log and the wandb run directories for staleness, traces into whatever code or run logs the next tasks touch, then reports in plain English where the equivariant residual RL work stands and what the next immediate steps are, asks any clarifying questions, and asks what the user wants to work on. Use at the beginning of a session, or when the user says they are picking the project back up, wants to catch up, or asks where things stand.
---

# Session start

Goal: come up to speed on the equivariant residual RL work, report it plainly, and find out what
Aden wants to do. Do **not** start making changes — this skill ends with a question, not a commit.

## 1. Read the state docs

Read all four. They are small and they are the point of this skill.

1. [docs/STATUS.md](../../../docs/STATUS.md) — what is true right now. **Check for an active freeze**; there is currently one on `resfit/rl_finetuning/equi_off_policy/` until the lost run is reproduced.
2. [docs/PROGRESS.md](../../../docs/PROGRESS.md) — what happened in recent sessions. Top 2–3 entries.
3. [docs/TODO.md](../../../docs/TODO.md) — the prioritized backlog, organized as H (housekeeping), R (reproduce the lost run), P1–P4 (standards, tests, cleanup, science).
4. [docs/EXPERIMENTS.md](../../../docs/EXPERIMENTS.md) — the run record. Read the summary table and any entry STATUS.md points at.

Then skim for orientation, not in depth:

5. [CLAUDE.md](../../../CLAUDE.md) — the three priorities, environment setup, protocol.
6. [docs/ARCHITECTURE.md](../../../docs/ARCHITECTURE.md) — module map and the **config-field provenance table**, which lists the config fields that are logged to wandb and ignored by the code.
7. [docs/EQUIVARIANCE.md](../../../docs/EQUIVARIANCE.md) — only if the session will touch the equivariant modules.
8. [docs/STANDARDS.md](../../../docs/STANDARDS.md) — only if the session will write code.

The two reference papers are in `docs/papers/` (gitignored): `resfit.pdf` is the residual
off-policy RL algorithm this builds on, `SO2ERL.pdf` is the SO(2) equivariant RL work the symmetry
design follows. Read the relevant one only if the session needs the theory — the markdown docs
cover the implementation.

## 2. Check the docs against reality

Docs go stale, and in this project the most common form is an unlogged run. Spend a minute:

```bash
git log --oneline -5
git status --short
ls -1t wandb/run-* | head -5
```

- Is HEAD the commit STATUS.md names as last verified?
- Are there uncommitted changes the docs do not mention?
- Are there wandb run directories newer than the newest EXPERIMENTS.md entry?

If any check disagrees with the docs, **say so before anything else.** Do not silently trust the
docs or silently trust the repo — report the conflict and let Aden resolve it.

## 3. Trace into code as needed

Use ARCHITECTURE.md's module map and "where to find things" index rather than searching blind.
Read what the next TODO items actually touch, preferring the specific functions the docs name over
whole files. The main training script is 1316 lines; its phase order is summarized in
ARCHITECTURE.md, so read that first and then jump to the phase that matters.

Two things that will mislead you if you skip them:

- **The step-0 evaluation in any run is the base BC policy alone**, because the residual initializes
  to exactly zero. It is the bar, not a result.
- **Check the provenance table before believing a run's config.** Several `EquivarianceConfig`
  fields are accepted, logged, and ignored, so a command-line override may have had no effect.

## 4. Report

Write a **brief, plain-English** summary Aden can read in under a minute:

- **Where we are** — 2–4 sentences on the state of the project.
- **What happened last session** — 1–2 sentences from PROGRESS.md.
- **Next planned steps** — the top TODO items in order, one line each on why.
- **Anything blocking** — the active freeze, decisions waiting on Aden, unlogged runs, anything
  the staleness check turned up.

Plain language; skip jargon where a normal word works. This is a status report, not a technical
document.

## 5. Ask

Two things, in order:

1. **Clarifying questions**, only if you genuinely need an answer to proceed well. When you ask,
   assume Aden has no context loaded: give the full background, list the apparent options, and give
   pros and cons for each, then recommend one. Do not ask what you can answer by reading.
2. **What does Aden want to work on?** Offer the top TODO items as concrete options, but ask
   openly — they may have something else in mind. Do not assume the next TODO item is the plan.

Then stop and wait.

## Project context, in case the docs are missing or wrong

The one-paragraph version, so this skill still works if a doc is broken. The repo is a fork of
ResFiT (residual off-policy RL for finetuning BC policies). Stage 1 trains a base BC policy; stage
2 freezes it and trains a small TD3 residual on top. The research question is whether making that
residual SO(2)-equivariant, following the SO(2) equivariant RL paper, improves sample efficiency —
even though the base policy it corrects is not equivariant. The baseline reproduces on the Can and
Square robomimic tasks. An equivariant version trained successfully once on Can, but that run's
data was deleted, so reproducing it across at least 3 seeds is the current priority. The three
project priorities, in order, are scientific consistency, code simplicity and correctness, and
human interpretability.
