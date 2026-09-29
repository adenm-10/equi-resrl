# Status

**Last updated:** 2026-09-29 · **Last verified commit:** `1e33b8f` (sweep options), plus its doc pass

This file describes what is true right now. Rewrite it in place; do not append. For history, see
[EXPERIMENTS.md](EXPERIMENTS.md) and [PROGRESS.md](PROGRESS.md).

## Where the project is

**Every equivariant run so far trained against a frozen target critic** (EXPERIMENTS.md,
2026-09-29 CORRECTION and NOTE). escnn layers in eval mode compute with weights expanded once, and
the target critic sat in eval mode from construction, so it never saw a soft update and the critic
never bootstrapped its own values. It is fixed from `93a8ad4`: the target critic runs in train mode,
as the actor target does. `equivariance.frozen_target_critic=True` reproduces the old behaviour.

So the earlier results measure the frozen-target agent, not the method:
- **Can replication** (2026-09-25 RESULTS): n=2, no collapse. `38z4wr9z` 0.82 → 0.96 carries it;
  `sw2qwfs9` drew a 0.94 base policy.
- **Square reference `equi-square-v1`** (2026-09-29 RESULTS): both seeds degraded the base policy,
  with L10 0.52 and 0.39.

**The Square sweep is running** (pre-registered and amended 2026-09-29). It has 18 arms, one seed
each, stopped at 150k steps. An arm advances if its 110k–150k mean reaches **B_sq = 0.508**, the
Square base rate measured over 500 episodes.
- First comes `target-fix`, the reference config with only the fix: does the fix alone make the
  residual help?
- Then the normalization arms, which Aden asked for by 2026-10-02, then the rest.
- Every arm has the fix.

**BoxCleanup** has had a smoke run only. It passed as a tooling check and measured what the pilot needs:
59.8 GiB at the start, projecting to about 82 GiB for a full run, and 0.69 s/step, so 500k steps
is about 4 days.

## Running now

| What | wandb | GPU | Started | Notes |
|---|---|---|---|---|
| Square sweep `target-fix` seed 1 | `0rehtuoq` | 0 | 2026-09-29 17:58 | From `~/projects/equi-resrl-ce96bb5`; ends ~09-30 18:00 |
| Square sweep `critic-ln` seed 1 | `qlaonpv4` | 1 | 2026-09-29 18:58 | From `~/projects/equi-resrl-1e33b8f`; ends ~09-30 19:00 |

**The queue.**
- A runner per GPU in tmux sessions `gpu0` and `gpu1` (`tools/gpu_queue.sh`) reads
  `~/queues/gpu{0,1}.txt`, one `<commit> <arm> <seed>` per line. Each launch runs from the worktree
  pinned to its commit.
- The BoxCleanup smoke run finished at 18:58 and passed (PROGRESS 2026-09-29); GPU 1's runner then started.
- Every other arm is queued at `1e33b8f` (`~/projects/equi-resrl-1e33b8f`).
- Logs: `outputs/queue/gpu{0,1}.log` records every launch, and each launch's console goes to its own
  file beside it.

| GPU | Order (about 23.5 h each; `q10` and `matched` about 28 h) |
|---|---|
| 0 | `target-fix` → `actor-ln` → `enc-inner` → `q10` → `matched` → `clipq` → `orth-critic` → `critic-last` → `orth-enc` |
| 1 | smoke → `critic-ln` → `critic-iid` → `actor-none` → `squash` → `l2` → `he-all` → `one-proj` → `narrow` → `critic-deep` |

- **By Friday 2026-10-02 noon:** `target-fix`, `critic-ln`, `actor-ln` and `critic-iid`.
- **Later on Friday:** `enc-inner` (~18:00) and `actor-none` (~23:00).
- **The whole queue:** about 2026-10-08 on GPU 1 and 2026-10-09 on GPU 0 (`critic-deep` moved to GPU 1 to balance them).

**RAM:** two Square runs at 40.3 GiB each fit; a third does not.

## What is established

- **The target-critic bug and its fix.** `tests/test_regression_agent.py` shows the old target
  never changing and the new one following its soft updates. The frozen reference's logged TD
  target barely moved during warm-up; `target-fix`'s moves.
- **Baseline residual TD3 works.** Square 0.52 → 0.90 (`870ws2c2`), Can 0.92 → 1.00 (`msfkjwab`),
  both single-seed and both with 10 critic heads. The equivariant default is 2 heads (2026-09-29
  CORRECTION).
- **The equivariant implementation is correct**, for one and two arms and for every sweep option:
  - the declared representations match independently derived physics;
  - the actor is equivariant and the critic invariant to < 1e-4, in train and eval mode;
  - 404 tests in Tiers 1–2.
- **The whole agent is pinned.** At the config defaults, `tests/test_regression_agent.py` gives the
  same parameters, outputs and update, bit for bit, as `f3e6315` and `b8e1de7`. So sweep arms at
  newer commits are comparable to the reference seeds, apart from the target-critic fix.
- **The BoxCleanup base policy is at about the paper's bar, not above it.** ACT, final 0.53. The
  stretch the `best` checkpoint comes from averages **0.61**, against the paper's ≈0.6. Its recorded
  0.74 at 24k is an outlier draw. The checkpoint is `run_xg0w2r0z_best:v7`, pinned by version and
  backed up.

## What is not

- **The base rate is uncertain.** The base-rate script gave 0.508 ± 0.022 (500 episodes). The
  trainer-path step-0 evals pool to 0.573 (172/300, including both sweep arms so far). No
  systematic cause was found.
  The sweep uses 0.508, as pre-registered. At that bar, the screen would have passed frozen-target
  seed 1. Each arm's step-0 adds a check.
- **Five successful runs are still undocumented** (TODO B2): `lgc70vgd`, `q1ocx89o`, `3vl625tv`,
  `36pfxsww` and `zyed0lew`. All are frozen-target, except `zyed0lew`, which is non-equivariant.
- **`action_scale` bounds the residual only with `residual_squash=True`,** the sweep's `squash` and
  `matched` arms. Otherwise the bound is `equi_clip(..., bound=1.0)`, so read `residual_l1` and
  `residual_l2` against 1.0.
- **TwoArmBoxCleanup has no exact spatial symmetry.** Chiral hands and asymmetric object placement,
  and `rotation=(0.0, 0.0)` means object yaw never varies. The equivariant arm there tests a
  *geometric prior*, not an exploited symmetry, and must not be pooled with Can or Square. See
  [EQUIVARIANCE.md](EQUIVARIANCE.md).
- **`seed` does not reach the environments.** Every episode's initial state is drawn from OS entropy,
  in every run so far, and the decision is to keep it that way (TODO B8).
- **The Can buffer caches are gone** (TODO H7), and Can needs a fixed-target re-run anyway (TODO B6).
- **The gate's RAM estimate is 21 GB per run against a measured 40.3 GiB** (TODO B7, S10). It does
  not count running trainers either.
- **The scalar ablation is still not usable as a control** (STANDARDS 3.3, TODO P4). **The gate cannot
  validate the commit it launches** (TODO R14).

## The plan, and why

**The Square sweep, in its queue order.**
- The target-critic fix is the most likely single cause of the Square failure, so `target-fix` runs
  first and every other arm is read against it.
- The normalization arms follow, since the critic has none and ResFiT's has LayerNorm throughout.
  Aden asked for them first.
- An arm that advances gets a pre-registered stage 2: seeds 1 and 2 at 300k.

**BoxCleanup after the sweep, one arm at a time, without writing caches.**
- A full run needs about 82 GiB of RAM (projected from the smoke run's measured 59.8), so it can't
  share the machine with a Square run. At 0.69 s/step, an arm takes about 4 days.
- The two arms can't share caches: the cache key doesn't record the normalization path.
- The launcher, pre-registration and base rate are TODO B7.

## Do not delete

- `z8yoqylh` exists only as `wandb/run-20260522_083555-z8yoqylh` and a backup in
  `~/equi-resrl-preserve/`; it is not in wandb. TODO H4 would restore it with `wandb sync`.
- `msfkjwab` survives only as `ZXP-S-works:wandb/run-20260226_143630-msfkjwab/`.
- The BoxCleanup base policy is backed up as `~/equi-resrl-preserve/run_xg0w2r0z_best:v7`; its
  sha256 is in EXPERIMENTS.md, 2026-09-27 CODE.
- The pinned worktrees `~/projects/equi-resrl-{ce96bb5,1e33b8f}` while any queued launch still names
  them. They also hold those runs' `wandb/` folders.

Three runs have already been lost to deletions.
