# Status

**Last updated:** 2026-09-25 · **Last verified commit:** `503a9a3` (run packages), plus its doc pass

This file describes what is true right now. Rewrite it in place; do not append. For history, see
[EXPERIMENTS.md](EXPERIMENTS.md) and [PROGRESS.md](PROGRESS.md).

## Where the project is

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

| What | wandb | GPU | Started | Notes |
|---|---|---|---|---|
| `equi-square-v1` seed 1 | `j04yoeui` | 0 | 2026-09-25 11:45 | Restarted at `2611adb`; see below. ~38k steps; evals 0.62, 0.56, 0.54, 0.60 through 30k |

GPU 1 is idle: the BoxCleanup BC run `dexmg-boxcleanup-bc/xg0w2r0z` finished at 16:31, 50k steps.

**The Square run was restarted.** The original (`0prso90n`, launched 09:17 at `b8e1de7`) died at
step 6,400 when a paste landed on its controlling terminal. The replacement runs the same config at
`2611adb` inside tmux. No `.py` on the training path differs between those commits; recorded in
EXPERIMENTS.md as a correction.

Square's step-0 came in at **0.62** against `870ws2c2`'s recorded 0.52. At n=50 episodes that gap is
about 1.4 standard errors, so it is within noise and not the 0.00 anomaly that the gate watches for.

## What is established

- **The equivariant agent reproduces on Can**, n=2, no collapse. See above.
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

- **No residual run on Square or BoxCleanup has finished.** Every claim about them is provisional.
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
  regression to the mean, not a pipeline fault. Read step-0 as the base rate. The checkpoint exists
  only as the wandb artifact `run_xg0w2r0z_best`; the old code deleted the local copy.
- **BoxCleanup cannot run yet.** The residual env wrapper's ACT reset path raises `NameError` on the
  first successful episode. TODO B3.
- **The Can buffer caches are gone.** `8edf618e` and `43637c10` are no longer on disk, and no doc
  records their deletion; the next Can run rebuilds them.
- **The scalar ablation is still not usable as a control.** STANDARDS.md rule 3.3, TODO P4.
- **The gate cannot validate the commit it launches.** TODO R14.

## The plan, and why

**Both BoxCleanup arms wait for Square**, which finishes around 2026-09-29. Decided 2026-09-25.

Square's 22.4 GB of caches free then, and BoxCleanup needs **~51 GB** (27.4 offline + 24 online).
At the decision there was 56 GB free, a ~5 GB margin, and `preflight.py` returned NO-GO on disk. A
cleanup the same evening brought it to 67 GB, a ~16 GB margin, which clears the gate's 10 GB bar.
Waiting also frees GPU 0 (GPU 1 is already idle), so the two arms can run **concurrently under
identical wall-clock conditions** instead of weeks apart. They share their caches anyway: the cache
key has no equivariance term, so the second arm costs no extra disk once the first has built them.

At roughly 1.5–1.8 s/step, 500k steps is 9–10 days per arm. The ACT reset crash (TODO B3) has to be
fixed before either arm launches; the fix touches only the ACT branch, so Can and Square stay
comparable.

## Do not delete

`z8yoqylh` exists only as `wandb/run-20260522_083555-z8yoqylh` and a backup in
`~/equi-resrl-preserve/`; it is not in wandb. `msfkjwab` survives only as
`ZXP-S-works:wandb/run-20260226_143630-msfkjwab/`. Three runs have already been lost this way.
TODO H4 would restore `z8yoqylh` with `wandb sync`.
