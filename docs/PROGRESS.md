# Progress

A chronological record of work sessions: what was done, what was decided, what changed. Append a
new entry at the top for each session.

**How this differs from the other docs:**

| Doc | Answers |
|---|---|
| PROGRESS.md (this file) | What have we *done*, session by session? |
| [STATUS.md](STATUS.md) | What is true *right now*? (rewritten in place) |
| [TODO.md](TODO.md) | What should we do *next*? |
| [EXPERIMENTS.md](EXPERIMENTS.md) | What did each training *run* show? (append-only, scientific record) |

Keep entries short. If something needs a paragraph of explanation it probably belongs in one of the
other four docs, with a pointer from here.

---

## 2026-09-16 — Phase 2 + launch readiness

Built the pre-submission gate, the submit script, and the config reconstruction. No changes under
`equi_off_policy/`; the freeze held.

**Done**

- **`tests/` — 148 tests across 7 files.** Independent group-action implementation, layout checks
  against the physics, actor equivariance, critic invariance, `equi_clip` commutation, and no-op
  pins. See [tests/README.md](../tests/README.md).
- **`preflight.py`** — three-tier gate (static / GPU construction / resources) reporting GO or
  NO-GO, exiting non-zero so launches can be blocked on it. Currently **GO**.
- **`submit.sh`** — single rewritable launcher, runs the gate first, follows the repo's existing
  ablation naming convention (group = variant, name = variant-seed, explicit integer seeds).
- **Tagged `repro-z8yoqylh-base`** at `caf83f3`.
- **Pre-registered the reproduction** in EXPERIMENTS.md with six numbered assumptions.

**Findings**

1. **The equivariant implementation is correct.** Declared representations match the physics at all
   8 group elements — including the interleaved rotation-column regrouping, the most error-prone
   part of the design. Actor equivariance and critic invariance both hold to `< 1e-4`. So the
   `7dae4925` collapse was *not* a broken symmetry, which strengthens the `FieldNorm` explanation.
2. **The reference run used config defaults, not the README overrides.** `a3e3zylp`'s resolved
   config is `n_step=3, gamma=0.99, stddev=0.05, action_scale=0.1` — the dataclass defaults, with
   `algo.prefetch_batches=4` as the only algorithmic override. The README and the coffee/square
   scripts pass `n_step=5 gamma=0.995 stddev=0.025 action_scale=0.2`. Copying those would have been
   wrong twice over: not a reproduction, *and* `n_step`/`gamma` are cache-key fields, so it would
   have invalidated 21 GB of buffers and forced a multi-hour rebuild.
3. **`enc_degree_channel` changed 16 → 32 in `caf83f3`.** The one genuine fork in the
   reconstruction. Going with 32 (the target commit's default, changed in the same commit titled
   "got equivariant agent working"); 16 is the first thing to retry if it fails. Assumption 2 in
   the pre-registration.
4. **The 45° vision equivariance error is ≈0.32 relative** — much larger than expected. Exact at
   right angles, a third off at diagonals, because rotating a pixel grid by 45° needs resampling.
   A second approximation source on top of the tilted camera. Recorded in EQUIVARIANCE.md.
5. **`train_bc_dexmg.py` parses CLI args at import time** (line 217, outside the `__main__` guard at
   905), so the module cannot be imported. Skipped in the import test with a dedicated test
   documenting the defect. Not on the reproduction path.

**Next**

Commit, then the live smoke run (`debug=true`) watching `dQ_da`. If it is still ~2e-4, stop and
re-diagnose instead of starting a 3-day run. Otherwise `./submit.sh 1`, then seeds 2 and 3.

---

## 2026-09-16 — Pre-flight audit for the reproduction

Checked the machine and wandb rather than assuming. Two findings changed the picture.

**Ready, verified**

Environment matches the 300k run exactly (torchrl 0.9.2, tensordict 0.9.1, escnn 1.0.13, CUDA on
both 3090s). BC base policy needs no retraining — still live in wandb and cached locally. Both Can
buffer caches computed by hash and present on disk (`cd03f9df` offline 5 GB, `9ff5ddd4` online
16 GB), so startup is minutes and there is no parallel-seed cache-write race.

**Findings**

1. **Group pooling is not what fixed the collapse.** `nn.GroupPooling` was already in the critic
   head at `7dae4925`, and all three surviving 300k runs at that commit ended at exactly 0.00 with
   three different seeds and configs. The real change at `caf83f3` is the **removal of `FieldNorm`
   and the ReLUs** from the critic head. `FieldNorm` subtracts each field's projection onto the
   trivial subspace, which for `regular_repr` is its group-invariant component — the very thing the
   following `GroupPooling` extracts. Matches `dQ_da ≈ 2e-4` with a low stable `critic_loss`.
   Good news for the reproduction: the target differs from the failures in the right way. Raises the
   risk on P3.4 (restoring critic normalization) correspondingly.
2. **`msfkjwab`, the Can baseline, is also deleted from wandb** — not just `z8yoqylh`. It survives
   only as `wandb/run-20260226_193630-msfkjwab/` on this disk. R7 revised: it cannot be re-pulled.
3. **Two 300k runs existed only in wandb**: `czjqzg0b` (2026-05-07) and `qe2by47h` (2026-05-16).
   Both collapsed. `qe2by47h` is the most recent equivariant run of any kind. Now in EXPERIMENTS.md.
4. **RAM is the binding constraint, not VRAM.** ~21 GB per run (16 GB preallocated online buffer +
   5 GB offline) against 58 GB available. Two seeds in parallel, not three.
5. **There is no launch script for Can residual RL**, equivariant or otherwise. Past runs were
   launched from commands kept in `note.txt`.
6. `x2w6phzf` used **N=12**, not N=8 — the only run that did.

**Next**

R1–R5, then launch. R section of TODO.md rewritten with the audit results.

---

## 2026-09-16 — Phase 1: standards, skills, allowlist

Documentation and tooling only. No changes to the training pipeline; the reproduction freeze held.

**Done**

- **[STANDARDS.md](STANDARDS.md) written** — 28 rules across configuration, equivariant code, the
  two agent families, code hygiene, running experiments, and logging. Every rule derived by reading
  the existing code, each citing an exemplar file or a live anti-pattern.
- **Three skills finalized** in `.claude/skills/`: `session-start`, `session-end`,
  `analyze-experiment`. Made repo-specific rather than generic — each carries the project traps that
  separate a useful answer from a misleading one (step-0 eval is the base policy; check the
  provenance table before believing a run config; `wandb-summary.json` holds only the last value).
- **`.claude/settings.json`** — read-only Bash allowlist to cut permission prompts.

**Findings**

1. **The scalar ablation is not a controlled ablation.** `use_equivariant_model=False` is supposed
   to isolate equivariance, but the two stacks differ in six ways at once: actor depth (1 hardcoded
   vs `num_layers` honored), actor dropout (ignored vs applied), output squashing (none vs `Tanh`),
   return type (tensor vs `TruncatedNormal`), critic head nonlinearities (0 ReLUs vs 2), and
   ensemble implementation (loop vs `vmap`). Running it today would give an uninterpretable result.
   Recorded as STANDARDS.md rule 3.3; P4's ablation task is now blocked on matching the stacks
   rather than on compute. This is the most consequential thing Phase 1 turned up.
2. **Layer count and normalization are each configured twice** — `agent.actor.num_layers` and
   `equivariance.num_actor_layers`, `agent.*.use_layer_norm` and `equivariance.use_norms`. The
   baseline path reads the first of each pair, the equivariant path the second, so setting the wrong
   one looks reasonable and does nothing. STANDARDS.md rule 1.4.
3. **`train/` and `training/` are both in use** for nearly the same thing. Rule 6.1 picks
   `train/` for per-update metrics and `training/` for run-level state, and forbids a third
   spelling.
4. **`layer_norm` in the baseline actor is an int used as a 3-way enum** (1 = after every layer,
   2 = last only), undocumented and only discoverable by reading `build_fc`. Upstream code; rule 1.5
   says do not copy it.

**Next**

R1–R3: tag `caf83f3`, reconstruct the lost config with assumptions written down, fix the run-naming
convention. Then the smoke gate and 3 seeds.

---

## 2026-09-16 — Phase 0: documentation from scratch

First session of the documentation effort. No code changes to the training pipeline.

**Done**

- Created this doc set: CLAUDE.md, ARCHITECTURE.md, EQUIVARIANCE.md, EXPERIMENTS.md, STATUS.md,
  TODO.md, PROGRESS.md, and a STANDARDS.md stub.
- Backfilled EXPERIMENTS.md from all 220 local wandb run directories. 23 runs got past ~1000 steps;
  5 reached the full 300k. Success rates recomputed from the `✓`/`✗` markers in each run's
  `output.log` rather than read off wandb, so they are reproducible from disk.
- Built the config-field provenance table in ARCHITECTURE.md by reading HEAD line by line.
- Housekeeping: papers → `docs/papers/` (gitignored, 18 MB); 24 root `run_2026-*` dirs →
  `local_runs/`; `.gitignore` extended.

**Findings that changed the plan**

1. **Six `EquivarianceConfig` fields are silently no-ops** at `caf83f3` — `num_actor_layers`,
   `num_critic_layers`, both `dropout`s, and the critic's `use_norms` / `use_orth_init`. They are
   accepted, logged to wandb, and ignored. So every past run's recorded hyperparameters misdescribe
   the network that trained: `a3e3zylp` was launched with `num_actor_layers=3` and that did nothing.
   This is why the reproduction freeze exists and why the cleanup is gated behind bit-identity
   tests. Full table in ARCHITECTURE.md.
2. **Critic group pooling is present at HEAD** ([critic.py:35-37](../resfit/rl_finetuning/equi_off_policy/rl/critic.py#L35)).
   Confirms Aden's recollection — the reproduction can proceed from `caf83f3` without recovering
   code from the other machine.
3. **Every equivariant run that exists locally collapses.** 14 runs, 5 commits, one failure mode:
   the residual saturates at `action_scale` and success falls to zero. `x2w6phzf` is the
   informative one — Square, 0.44 → 0.72 → 0.00 — which rules out a representational-capacity
   explanation and points at stability.
4. **43 of 197 failed runs died with escnn types in the traceback.** Field-type mismatches were
   being found by launching training runs. Motivated moving construction-time layout assertions to
   the front of Phase 2.
5. **The agentview camera is tilted 44.9°**, so vision equivariance is approximate, not exact.
   Reviewed and accepted the same day — see decisions below.
6. **Can is a weak task for the core question.** Base policy at 0.92, ceiling at 1.00. Square
   starts at 0.52 and discriminates far better. Can is still right for the reproduction because
   that is what the lost run used.

**Decisions**

- **Reproduction freeze.** No edits under `resfit/rl_finetuning/equi_off_policy/` until the
  `z8yoqylh` result is reproduced. Priority 1 (scientific consistency) overriding priority 2 — known
  defects stay in place so that a reproduction failure cannot be confused with a change we made.
- **Tilted camera accepted.** Approximate vision equivariance is considered a useful prior; the
  proprioception and action branches are exactly equivariant. Not scheduled for work.
- **`rl_utils.py` deleted.** Untracked, unimported, and unimportable — it had
  `from __future__ import annotations` on line 15, a hard `SyntaxError`, so it had never once
  loaded. Its one original idea, `TrivialLayerNorm`, is preserved verbatim in TODO item P3.4.
- **`note.txt` and `log.txt` removed from tracking.** Content migrated into these docs; both
  recoverable via `git show HEAD~1:note.txt`.
- **BC retraining is not needed.** Base policies for Can, Square, and TwoArmCoffee are cached under
  `artifacts/`. Both Can and Square use **Diffusion** policies, not ACT — verified from the cached
  `policy/config.json`, and worth knowing since the README example and the upstream paper use ACT.

**Next**

R1–R3 in TODO.md: tag the reproduction baseline, reconstruct the lost config with assumptions
written down, and fix the run-naming convention before launching 3 seeds.
