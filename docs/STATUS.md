# Status

**Last updated:** 2026-09-29 · **Last verified commit:** the commit that adds this line (see git log)

This file describes what is true right now. Rewrite it in place; do not append. For history, see
[EXPERIMENTS.md](EXPERIMENTS.md) and [PROGRESS.md](PROGRESS.md).

## Where the project is

**Every equivariant run so far trained against a frozen target critic** (EXPERIMENTS.md,
2026-09-29 CORRECTION). escnn layers in eval mode compute with weights expanded once, and the
target critic sat in eval mode from construction, so it never saw a soft update: the critic never
bootstrapped. Fixed from 2026-09-29, with `equivariance.frozen_target_critic` to reproduce the old
behaviour. The Can replication and the Square reference below measure the frozen-target agent,
not the method.

**The Square reference degraded its base policy**, n=2: L10 0.52 and 0.39 against a base rate of
about 0.60. The Square sweep, pre-registered 2026-09-29, starts with `target-fix` (the fix alone),
then the normalization arms.

**The replication succeeded, and the project now has its first multi-seed result.** Both
`equi-repro-caf83f3` seeds ran the full 300k steps without collapsing and both ended above their
step-0 baseline: `38z4wr9z` 0.82 → 0.96, `sw2qwfs9` 0.94 → 0.96. Details in
[EXPERIMENTS.md](EXPERIMENTS.md), 2026-09-25 RESULTS.

Read that result carefully. `38z4wr9z` is the load-bearing arm — same seed as `z8yoqylh`, same 0.82
step-0, **+0.14**. `sw2qwfs9` drew a 0.94 base policy and gained only +0.02, so its headline number
is mostly a favourable draw. **n=2 for "does it reproduce", much weaker for "how much does it
help".**

The work has moved on to two new fronts: a second single-arm task (Square) and the first bimanual
task (TwoArmBoxCleanup).

**Runs now keep everything locally and upload only their best.** From `503a9a3`, each run writes a
persisted package under `outputs/runs/<project>/<run_id>/` and uploads one wandb artifact, the best
model plus its eval video, replaced in place. Only saving and uploading changed: the training step
and every logged metric name are identical to the pre-change runs `j04yoeui` and `xg0w2r0z`. Layout
and rules in ARCHITECTURE.md, "Run packages". The running Square seed and anything launched from
the pinned worktrees keep the old behaviour — a video upload per eval, and the local folder deleted
when the run completes.

## Running now

The Square sweep's queue, from worktrees pinned per commit (`~/projects/equi-resrl-<commit>`),
one runner per GPU in tmux. See EXPERIMENTS.md, 2026-09-29 PRE-REGISTERED, and the LAUNCHED
entries after it. Queue files live in `~/queues/`, launch logs in `outputs/queue/`.

**Both `equi-square-v1` seeds finished:** `j04yoeui` (seed 1, L10 0.52) and `88i0lpc8` (seed 2, L10
0.39). EXPERIMENTS.md, 2026-09-29 RESULTS.

## What is established

- **The equivariant agent reproduces on Can**, n=2, no collapse, **with a frozen target critic**:
  a measurement of that agent, not of the method (2026-09-29 CORRECTION).
- **Baseline residual TD3 works.** Square 0.52 → 0.90 (`870ws2c2`), Can 0.92 → 1.00 (`msfkjwab`).
  Both single-seed.
- **The equivariant implementation is correct, now for one and two arms.** 244 tests: declared
  representations match independently-derived physics at all 8 group elements, actor equivariant and
  critic invariant to `< 1e-4`, for both arm counts.
- **Single-arm behaviour is pinned.** `tests/test_regression_single_arm.py` fingerprints the encoder
  output; it stayed bit-identical through the bimanual generalisation, so Can and Square runs before
  and after remain comparable.
- **The BoxCleanup base policy is at about the paper's bar, not above it.** ACT, final 0.53. The
  stretch the `best` checkpoint comes from averages **0.61** against the paper's ≈0.6. Read the
  caveat below before quoting its peak.

## What is not

- **No Square run has had a working target critic, and no BoxCleanup residual run has run.** Both
  Square reference seeds finished, frozen-target; every claim about either task is provisional.
- **Five successful runs are still undocumented.** `lgc70vgd` (0.86 → **0.98**, 243,800 steps, no
  overrides), `q1ocx89o` (0.98), `3vl625tv` (0.96), `36pfxsww` (0.96) and `zyed0lew` (the
  non-equivariant Can baseline, 0.76 → 0.94) beat their step-0 and appear in no EXPERIMENTS.md
  entry. `lgc70vgd` also reached a *higher* best with `actor_last_layer_init_scale=0.0`, which
  undercuts TODO R11's premise.
- **`action_scale` does not bound the residual.** The squash after the actor's final Linear is
  commented out, so `|mu|` is not ≤ 1 — measured 1.27x on the two-arm actor. The real bound is
  `equi_clip(..., bound=1.0)`. Read `residual_l1` / `residual_l2` against 1.0, not against
  `action_scale`. See ARCHITECTURE.md rough edges.
- **TwoArmBoxCleanup has no exact spatial symmetry.** Chiral hands, asymmetric object placement, and
  `rotation=(0.0, 0.0)` means object yaw never varies. The equivariant arm there tests a *geometric
  prior*, not an exploited symmetry. Results must not be pooled with Can/Square. Full argument in
  [EQUIVARIANCE.md](EQUIVARIANCE.md).
- **The BoxCleanup BC policy's peak is not its rate, and it drifted down late.** 50 evals of 100
  episodes, oscillating ±0.15 throughout. Averages over ten evals each: **0.61** for 21k–30k, 0.54
  for 31k–40k, 0.51 for 41k–50k; final 0.53. The earlier reading of "true rate ≈0.65" came from the
  31k mark and does not hold. The **best of 0.74 at 24k is an outlier draw**, and it is the
  checkpoint `wt_type="best"` loads, so the residual run's step-0 should come in near 0.6, not 0.74 —
  regression to the mean, not a pipeline fault. Read step-0 as the base rate. The checkpoint is
  `run_xg0w2r0z_best:v7`, pinned by version in the BoxCleanup configs and backed up locally.
- **The ACT reset crash (TODO B3) is fixed** as of 2026-09-27, so an ACT base policy no longer
  crashes the residual env wrapper. No BoxCleanup residual run has launched yet.
- **`seed` does not reach the environments.** Every episode's initial state, training and eval, is
  drawn from OS entropy in every run so far. A seed fixes network init and noise, not episodes.
  EXPERIMENTS.md, 2026-09-27 CORRECTION.
- **The Can buffer caches are gone.** `8edf618e` and `43637c10` are no longer on disk, and no doc
  records their deletion; the next Can run rebuilds them.
- **The scalar ablation is still not usable as a control.** STANDARDS.md rule 3.3, TODO P4.
- **The gate cannot validate the commit it launches.** TODO R14.

## The plan, and why

**The Square sweep first** (pre-registered 2026-09-29): one seed per arm, a 150k-step screen, an
arm advances if its 110k–150k mean reaches the base rate. `target-fix` asks whether the target-critic
fix alone is enough; the normalization arms follow, as Aden asked for them by 2026-10-02.

**BoxCleanup after that, one arm at a time, without writing caches** (`write_buffer_caches=false`).
A full run needs about 76 GiB of RAM (estimated), so it can't share the machine with a Square run,
and the two arms can't share caches: the cache key doesn't record the normalization path. A short
smoke run measures RAM and speed first. The earlier plan to run both arms concurrently on shared
caches (2026-09-25) is superseded.

## Do not delete

`z8yoqylh` exists only as `wandb/run-20260522_083555-z8yoqylh` and a backup in
`~/equi-resrl-preserve/`; it is not in wandb. `msfkjwab` survives only as
`ZXP-S-works:wandb/run-20260226_143630-msfkjwab/`. The BoxCleanup base policy is backed up as
`~/equi-resrl-preserve/run_xg0w2r0z_best:v7`; its sha256 is in EXPERIMENTS.md, 2026-09-27 CODE. Three runs have already been lost this way.
TODO H4 would restore `z8yoqylh` with `wandb sync`.
