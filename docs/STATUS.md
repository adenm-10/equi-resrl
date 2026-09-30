# Status

**Last updated:** 2026-09-30 · **Last verified commit:** the sync-free update (EXPERIMENTS.md,
2026-09-30 CODE), on top of `7612801`

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

**The Square sweep is closed.** Aden stopped `target-fix` and `critic-ln` on 2026-09-30 before
their verdicts (EXPERIMENTS.md, STOPPED), and its other 16 arms never ran.

**Its replacement is running: the actor-none gate, then four ablations** (2026-09-30
PRE-REGISTERED). A run advances if its 110k–150k mean reaches **B_sq = 0.508**, the base policy's
success rate over 500 episodes.
- `actor-none` is the reference at the new defaults: the target-critic fix, no normalization in
  the actor. If it doesn't advance, nothing more is queued and the implementation gets reworked.
- `critic-fn` runs alongside it. `clipq` follows if the gate passes, then `squash` and `depth` on
  whichever of `critic-fn` and `clipq` advanced.

**From 2026-09-30 the update is about 2× faster, and `actor_norm` defaults to `none`**
(EXPERIMENTS.md, 2026-09-30 CODE).
- The update no longer waits for the GPU: 487 host syncs per env step became 0, and an env step's
  learning went from about 760 to 400 ms on a shared GPU. Bit-identical training, checked on CPU
  and GPU against `ce96bb5` and `1e33b8f`.
- Every earlier run had batch norm in the actor. `actor-none` is the reference at the new default.
- **wandb:** every Square equivariant run is now in group `equi-residual-rl`, and every Square BC
  run in `robomimic-square-bc`; past runs were regrouped (2026-09-30 CODE).

**BoxCleanup** has had a smoke run only. It passed as a tooling check and measured what the pilot needs:
59.8 GiB at the start, projecting to about 82 GiB for a full run, and 0.69 s/step, so 500k steps
is about 4 days.

## Running now

| What | GPU | Started | Ends (est.) |
|---|---|---|---|
| `actor-none` seed 1, the gate | 0 | 2026-09-30, see LAUNCHED | ~14 h later |
| `critic-fn` seed 1 | 1 | 2026-09-30, see LAUNCHED | ~14 h later |

**The queue.**
- A runner per GPU in tmux sessions `gpu0` and `gpu1` (`tools/gpu_queue.sh`) reads
  `~/queues/gpu{0,1}.txt`, one `<commit> <arm> <seed>` per line, every minute. Each launch runs from
  the worktree pinned to its commit.
- Nothing is queued behind the two runs: the next lines depend on the gate's verdict.
- The sweep's 16 unlaunched arms are kept in `~/queues/gpu{0,1}.txt.bak-20260930`.
- Logs: `outputs/queue/gpu{0,1}.log` records every launch, and each launch's console goes to its own
  file beside it.

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
  - 430 tests in Tiers 1–2.
- **The whole agent is pinned.** At the reference config, with the actor's batch norm,
  `tests/test_regression_agent.py` gives the same parameters, outputs and update, bit for bit, as
  `f3e6315` and `b8e1de7`. A second golden pins the new default, `actor_norm=none`. So runs at newer
  commits are comparable to the reference seeds, apart from the target-critic fix and the actor
  default.
- **The update makes no host sync** for any config the revised schedule can launch
  (`tests/test_sync_free.py`), and trains bit for bit as `1e33b8f` did for all of them.
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

**The Square ablations of 2026-09-30, gated on `actor-none`.**
- `actor-none` checks that the fixed agent at the new defaults can at least match vanilla BC. If
  it can't, more ablations would measure a broken baseline, so the implementation comes first.
- `critic-fn` (FieldNorm in the critic) runs alongside it; `clipq` (the TD target clipped to the
  reward range) follows the gate. Then `squash` and `depth`, built on whichever of the two passed.
- If FieldNorm and clipping both pass, nothing tests them together alone, so `squash` and `depth`
  on top of both would lack a one-change reference.
- An arm that advances gets a pre-registered stage 2: seeds 1 and 2 at 300k, against `actor-none`.

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
