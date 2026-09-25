# Status

**Last updated:** 2026-09-25 · **Last verified commit:** `2611adb`, plus this session's uncommitted
bimanual work

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

## Running now

| What | wandb | GPU | Started | Notes |
|---|---|---|---|---|
| `equi-square-v1` seed 1 | `j04yoeui` | 0 | 2026-09-25 11:45 | Restarted at `2611adb`; see below |
| BoxCleanup ACT BC | `dexmg-boxcleanup-bc/xg0w2r0z` | 1 | 2026-09-25 09:47 | 50k steps, best **0.64** @ 21k |

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
- **The equivariant implementation is correct, now for one and two arms.** 230 tests: declared
  representations match independently-derived physics at all 8 group elements, actor equivariant and
  critic invariant to `< 1e-4`, for both arm counts.
- **Single-arm behaviour is pinned.** `tests/test_regression_single_arm.py` fingerprints the encoder
  output; it stayed bit-identical through the bimanual generalisation, so Can and Square runs before
  and after remain comparable.
- **The BoxCleanup base policy clears the paper's bar.** ACT, best 0.64 at 21k against the paper's
  ≈0.6. Note the eval spread: 0.40 / 0.64 / 0.52 at 20k / 21k / 22k, so ±0.12 at n=100.

## What is not

- **Nothing on Square or BoxCleanup has finished.** Every claim about them is provisional.
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
- **The scalar ablation is still not usable as a control.** STANDARDS.md rule 3.3, TODO P4.
- **The gate cannot validate the commit it launches.** TODO R14.

## Disk

61 GB free. BoxCleanup needs **~51 GB** of buffer cache (27.4 offline + 24 online), leaving ~9 GB
margin, so `preflight.py` currently returns NO-GO on disk for that task. Square's 22.4 GB frees when
it finishes, around 2026-09-29.

The two BoxCleanup arms **share** their caches — the cache key has no equivariance term — so the
equivariant run costs no extra disk once the baseline has built them.

## Do not delete

`z8yoqylh` exists only as `wandb/run-20260522_083555-z8yoqylh` and a backup in
`~/equi-resrl-preserve/`; it is not in wandb. `msfkjwab` survives only as
`ZXP-S-works:wandb/run-20260226_143630-msfkjwab/`. Three runs have already been lost this way.
TODO H4 would restore `z8yoqylh` with `wandb sync`.
