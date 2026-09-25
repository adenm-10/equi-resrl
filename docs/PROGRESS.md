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

## 2026-09-25 — The encoder goes bimanual, and Square restarts

The `caf83f3` replication reported (both seeds reproduced, first n=2 in the project). Work moved to
a second single-arm task and the first two-arm task.

**Done**

- **Pinned the single-arm encoder before touching it** (`2611adb`). `tests/test_regression_single_arm.py`
  fingerprints the actor and critic features with a sum plus a seeded random projection; the
  projection is what catches a field reordering the sum would miss. Wired into the Tier 1 gate.
- **Generalised `ResObsEnc` to `n_arms`.** Per arm the state is `[eef_pos 3, eef_quat 4,
  gripper_qpos G]` and the action `[delta_pose 6, hand H]`. `enc_ih` became a `ModuleList`, one per
  wrist camera. The regression fingerprint stayed bit-identical throughout, so Can and Square runs
  before and after remain comparable. `actor.py` and `critic.py` needed no changes — they already
  derive every offset from `FieldType.size`.
- **Normalizer, probe and agent followed.** Per-arm normalizer keys sharing one rotation centre;
  `detect_robot_base_xy` → `detect_robot_bases`, which returns *every* base rather than silently
  taking the first; `QAgent` now asserts encoder dims against the env instead of ignoring them.
- **Tests parameterised over arm count.** The `obs_enc` fixture yields both layouts, so all of Tier 2
  runs twice. Suite 74 → 231 passing.
- **`preflight.py` generalised.** `image_keys` and `num_episodes` moved into `TASKS`; added
  `TwoArmBoxCleanup`; cache sizes are now estimated per task and the disk check blocks on the
  estimate rather than a fixed 40 GB.
- **Launched Square** (`equi-square-v1` seed 1), and **relaunched it** after the original died.
- **Started the BoxCleanup ACT BC run**, best 0.64 at 21k.

**Decided**

- **TwoArmBoxCleanup, C8 about the midpoint of the two bases** — as a geometric prior, not an exact
  symmetry. The task fixes object yaw, and the hands are chiral, so six of the eight group elements
  are not physical symmetries. Recorded in EQUIVARIANCE.md so no later reader mistakes it for the
  Can/Square case.
- **`num_episodes=1000` and `total_timesteps=500_000`** for BoxCleanup, to match the paper and stay
  consistent with the other runs.
- **ACT, not diffusion, for the BoxCleanup base policy**, matching the ResFiT authors' own
  simulation launchers.
- **Wait for Square before launching BoxCleanup.** It finishes around 2026-09-29 and frees 22.4 GB,
  which is what BoxCleanup's caches need, plus the second GPU. That lets the baseline and
  equivariant arms run concurrently under identical wall-clock conditions rather than weeks apart,
  and they share caches anyway. Cost is ~3.5 days of an idle GPU 1; the alternative was launching
  into a ~6 GB disk margin and re-running later for pairing. Resolves TODO B5.

**Found**

- **`action_scale` does not bound the residual.** The squash after the actor's final Linear is
  commented out, so `|mu|` is not ≤ 1 (1.27x measured on two arms). The old
  `test_actor_respects_action_scale` asserted a bound that held only by luck of initialisation
  magnitude. The real bound is `equi_clip(..., bound=1.0)`.
- **`pytest tests/` was a trap on this machine.** escnn caches basis tensors per representation, so
  once a GPU module exists a later CPU build dies on a device mismatch — and alphabetical collection
  put the CPU regression module last. Preflight never saw it because it runs Tier 1 in a separate
  subprocess. `conftest.py` now orders that module first.
- **The two BoxCleanup arms share their buffer caches.** The cache key has no equivariance term, so
  the equivariant run costs no extra disk.
- **`ablate_equi_obs_encoder.py` is imported by nothing** and still references a normalizer key that
  no longer exists. Left under the freeze; queued for P3.5.
- **The BoxCleanup BC policy's evaluation is unstable, and it matters.** 31 evals of 100 episodes
  each: a clear climb from 0.26 to roughly 0.65, but oscillating +/-0.15 the whole way. The last six
  are 0.68 0.53 0.67 0.66 0.69 0.53. The recorded peak of **0.74 at step 24k is an outlier draw**,
  not the policy's true rate. `base_policy.wt_type="best"` selects on exactly this noisy signal, so
  the saved checkpoint is biased high — which means the residual run's step-0 will look lower than
  the BC "best" for reasons that are not a pipeline bug. Read step-0 as the base rate, and put the
  whole sequence in the pre-registration.

**Next**

1. When BC finishes: verify the artifact, set `base_policy.wandb_id` in both the config and
   `preflight.TASKS`, and back the checkpoint up.
2. When Square finishes: free its caches, pre-register both BoxCleanup arms, and launch them
   together.
3. Independent of both: write up the five undocumented successful runs (TODO B2).

## 2026-09-16 — The lost run was not lost, and the target was wrong

Started as a session to move the previous session's setup onto a second workstation. Turned into
recovering `z8yoqylh` and discovering the reproduction had been aimed at the wrong commit.

**Done**

- **Moved the repo setup to `boce-WS-01`** (`10.188.60.163`, user `aden`, `~/projects/equi-resrl`).
  There was already a GitHub remote sitting at `caf83f3`; pushed the 14 unpushed commits plus tags,
  pulled there. Docs, `.claude/` skills, 148 tests, `preflight.py`, `submit.sh` all in place; gate
  passes.
- **Recovered `z8yoqylh`** at `boce-WS-01:~/projects/equi-resrl/wandb/run-20260522_083555-z8yoqylh`,
  173 MB, config and logs intact. Backed up to `~/equi-resrl-preserve/`.
- **Corrected the record** in EXPERIMENTS.md (append-only, new section) and repointed
  `preflight.py` at `z8yoqylh`'s environment. Tagged `repro-z8yoqylh-actual` at `7dae4925`.
- **Launched the 3-seed replication** on `boce-WS-01`, seed-group `equi-repro-7dae4925`: seeds
  2114495708 / 1 / 2 as `9qgsaxsr` / `4sseggkv` / `dxbept4u`. Pre-registered before launching.
- **Killed `m0ylcivk`**, the `caf83f3` run started earlier in the day, once it was clear it was not
  the reproduction.

**Findings**

1. **`z8yoqylh` ran at `7dae4925`, not `caf83f3`** — from its own `wandb-metadata.json`. That is
   the same commit as the three collapsed runs, so `7dae4925` is not an architectural failure, and
   **the `FieldNorm` hypothesis is dead**: `FieldNorm` was in the critic head of the run that
   reached 0.94.
2. **The launch carried an override no doc mentioned:**
   `agent.actor.actor_last_layer_init_scale=1e-4`, against a default of 0.0 that sits in the config
   with `1e-4` commented out beneath it. `z8yoqylh` and `a3e3zylp` share a commit and a config and
   differ in only this and the host/stack — one reached 0.94, the other 0.00.
3. **Host correlates perfectly with outcome across the whole project.** Every collapsed equivariant
   run ran on `ZXP-S-works`; the one success ran on `boce-WS-01`, whose stack is torch 2.6.0 /
   torchrl 0.7.0 / escnn 1.0.11. `preflight.py` had been validating the environment against
   `a3e3zylp` — a run that collapsed to 0.00.
4. **The successful run had robot-base centering disabled.** At `7dae4925` every call site is
   commented out, so it rotated about the world origin — the case EQUIVARIANCE.md calls a silent
   symmetry break. It reached 0.94 anyway. The feature is active at HEAD, so **HEAD is not the
   architecture that produced 0.94.**
5. Recovered specifics: seed **2114495708**, best **0.94** (not ~0.92 from the screenshot), runtime
   **47.3 h** measured. `enc_degree_channel=32` was the one assumption guessed right.
6. Two path traps in the worktree setup: `_CACHE_ROOT` defaults to the *current directory*, and the
   BC artifact downloads into `./artifacts/`. Handled with `CACHE_DIR` and a symlink; without them
   it is a 21 GB rebuild.
7. Each run holds ~25 GB resident, not ~21 GB. Three is the ceiling on `boce-WS-01`, not a
   comfortable three.

**Decisions**

- **Replicate before explaining.** Three seeds at the recovered config first; isolating init-scale
  versus stack is deferred to R11. Rationale: with two candidate causes and n=1, isolating first
  risks spending ~94 h explaining a result that may not replicate.
- **Three seeds in parallel over sequential.** Sharing GPU 1 costs 1.85x (measured, 79.7 versus
  145-150 ms/update) but gets all three in ~87 h against ~94 h sequential. The replication arm runs
  alone on GPU 0 so the number that matters most is the cleanest.
- **Left the wrong tag in place.** `repro-z8yoqylh-base` points at `caf83f3`; it is already pushed,
  so it was superseded rather than moved.
- **Freeze stays on** `equi_off_policy/`, with a new reason: three runs are in flight and the
  analysis compares them against `z8yoqylh`.

**Next**

R7 — the 10k and 20k evaluations, where every past collapse was already unambiguous. Then R8
(config caveats at `7dae4925`, which blocks the results entry) and R10 (`/analyze-experiment`, then
lift the freeze).

---

## 2026-09-16 — Phase 2 + launch readiness

Built the pre-submission gate, the submit script, and a reconstruction of `z8yoqylh`'s config. No
changes under `equi_off_policy/`; the freeze held.

**Condensed 2026-09-16.** Several findings in this entry were aimed at `caf83f3` and are superseded
by the recovery of `z8yoqylh` later the same day — see the top entry and the correction section in
[EXPERIMENTS.md](EXPERIMENTS.md). What each one got right and wrong is noted inline.

**Done**

- **`tests/` — 148 tests across 7 files.** Independent group-action implementation, layout checks
  against the physics, actor equivariance, critic invariance, `equi_clip` commutation, no-op pins.
- **`preflight.py`** — three-tier gate (static / GPU construction / resources), exiting non-zero so
  launches can be blocked on it. **`submit.sh`** — single rewritable launcher.
- Tagged `repro-z8yoqylh-base` at `caf83f3` and pre-registered the reproduction with six
  assumptions. **Both aimed at the wrong commit.** The tag is superseded by
  `repro-z8yoqylh-actual` at `7dae4925`; the pre-registration by the correction entry.

**Findings**

1. **The equivariant implementation is correct.** Representations match the physics at all 8 group
   elements, including the interleaved rotation-column regrouping. Actor equivariance and critic
   invariance hold to `< 1e-4`. **Still true and still important** — it is why the collapses were
   never a broken symmetry. The inference drawn from it at the time, that this strengthened the
   `FieldNorm` explanation, is dead.
2. **`a3e3zylp` ran on config defaults, not the README overrides** — `n_step=3, gamma=0.99,
   stddev=0.05, action_scale=0.1`, with `algo.prefetch_batches=4` the only algorithmic override.
   Still true, and `z8yoqylh`'s recovered config confirms it used the same defaults. Copying the
   README's `n_step=5 gamma=0.995` would also have invalidated 21 GB of cache, since both are
   cache-key fields.
3. **Chose `enc_degree_channel=32`** over 16. Correct by luck: `z8yoqylh` ran at 32, but at
   `7dae4925`, where the default was 16 — so it was set explicitly, not inherited.
4. **The 45° vision equivariance error is ≈0.32 relative**, much larger than expected. Exact at
   right angles, a third off at diagonals, because rotating a pixel grid by 45° needs resampling.
   Recorded in EQUIVARIANCE.md.
5. **`train_bc_dexmg.py` parses CLI args at import time** (line 217, outside the `__main__` guard at
   905), so it cannot be imported. Skipped in the import test with a dedicated test documenting the
   defect. Not on the reproduction path.

---

## 2026-09-16 — Pre-flight audit for the reproduction

Checked the machine and wandb rather than assuming.

**Condensed 2026-09-16**, for the same reason as the entry above. **Finding 1 of this audit was the
`FieldNorm` hypothesis, and it is now known false** — `z8yoqylh` ran at `7dae4925` with `FieldNorm`
present in the critic head. The audit's error was reasoning about which commit the run used from the
commit titles and the local evidence, rather than from the run's own metadata, which existed the
whole time on a machine nobody checked.

**Ready, verified**

Environment on `ZXP-S-works` matched `a3e3zylp` exactly. BC base policy needed no retraining. Both
Can buffer caches present by hash, so startup is minutes and there is no parallel-seed write race.

**Findings that still stand**

1. **`msfkjwab`, the Can baseline, is deleted from wandb.** It survives only as
   `ZXP-S-works:wandb/run-20260226_143630-msfkjwab/`. It cannot be re-pulled.
2. **Two 300k runs existed only in wandb:** `czjqzg0b` (2026-05-07) and `qe2by47h` (2026-05-16),
   both collapsed. Now in EXPERIMENTS.md.
3. **RAM is the binding constraint, not VRAM.** Correct in kind, wrong in size: the estimate was
   ~21 GB per run, and the measurement on `boce-WS-01` is ~25 GB.
4. **There was no launch script for Can residual RL.** Past runs were launched from commands kept
   in `note.txt` — which is why `z8yoqylh`'s override was invisible until its metadata was read.
5. `x2w6phzf` used **N=12**, the only run that did.

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

**Findings** — all four became STANDARDS.md rules, so they are recorded there rather than restated
here.

1. **The scalar ablation is not a controlled ablation** (rule 3.3). `use_equivariant_model=False`
   is supposed to isolate equivariance, but the two stacks differ in six ways at once — actor depth,
   actor dropout, output squashing, return type, critic head nonlinearity count, and ensemble
   implementation. Running it today gives an uninterpretable result, so P4's ablation is blocked on
   matching the stacks, not on compute. The most consequential thing Phase 1 turned up, and still
   open.
2. **Layer count and normalization are each configured twice** (rule 1.4) — the baseline path reads
   `agent.*`, the equivariant path reads `equivariance.*`, so setting the wrong one of a pair looks
   reasonable and does nothing.
3. **`train/` and `training/` were both in use** for nearly the same thing (rule 6.1).
4. **`layer_norm` in the baseline actor is an int used as a 3-way enum** (rule 1.5), undocumented
   and only discoverable by reading `build_fc`. Upstream code; do not copy it.

**Next** — recorded as written, and superseded the same day: tag `caf83f3`, reconstruct the lost
config, fix run naming, then the smoke gate and 3 seeds. The commit was wrong; see the top entry.

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

**Decisions** — each now lives in the doc that owns it; pointers only, to avoid drift.

- **Reproduction freeze** on `resfit/rl_finetuning/equi_off_policy/` — priority 1 over priority 2,
  so a reproduction failure cannot be confused with a change we made. Still in force, with an
  updated reason; see STATUS.md.
- **Tilted camera accepted** as a useful approximation — EQUIVARIANCE.md assumption 1.
- **`rl_utils.py` deleted**, and **`note.txt` / `log.txt` untracked** — TODO H1/H3. The notes files
  are recoverable via `git show 8442ba2^:note.txt`.
- **BC retraining is not needed.** Policies for Can, Square and TwoArmCoffee are cached under
  `artifacts/`. Can and Square use **Diffusion** policies, not ACT — verified from the cached
  `policy/config.json`, and worth knowing because the README example and the upstream paper use ACT.

**Next**

R1–R3 in TODO.md: tag the reproduction baseline, reconstruct the lost config with assumptions
written down, and fix the run-naming convention before launching 3 seeds.
