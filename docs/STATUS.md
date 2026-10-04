# Status

**Last updated:** 2026-10-04 · **Last verified commit:** the tuned-RLPD pre-registration, on the
no-base-policy mode (EXPERIMENTS.md, 2026-10-04 CODE and PRE-REGISTERED)

This file describes what is true right now. Rewrite it in place; do not append. For history, see
[EXPERIMENTS.md](EXPERIMENTS.md) and [PROGRESS.md](PROGRESS.md).

## Where the project is

**The image-fix pair collapsed** (EXPERIMENTS.md, 2026-10-04 RESULTS). Both seeds of the residual
reference, with act() fixed, went from 0.46 and 0.52 at step 0 to 0.00 at every eval from 10k.
- The residual, now executed, grew to a mean |r| of about 0.45 by 30–50k, and its largest
  component reached the clip at 1.0 by 15–20k.
- The critic kept scoring it above the base action.
- So, as pre-registered, the IID-batch-norm actor's earlier stability came from executing almost
  no residual. **No equivariant residual design has yet held or beaten its base policy.**

**So the RL machinery is being tested without residual RL:** tuned RLPD, vanilla RL with no base
policy (TODO B12; EXPERIMENTS.md, 2026-10-04 PRE-REGISTERED).
- **Built:** a `base_policy=None` mode in `train_residual_td3.py`, for both agents, with four
  configs, a wrapper, a gate method, a verdict tool and four arms in `submit.sh`
  (2026-10-04 CODE). The residual path is unchanged, tensor for tensor.
- **Launching now: the two ResFiT arms,** Square and Can, seed 1, 150k steps each.
- **The two equivariant arms wait for Aden** (Aden's decision, 2026-10-04), who decides their
  design when the ResFiT arms report: the actor's norm and its output bound at least. `submit.sh`
  holds them with the plan's provisional design and refuses to launch them.

**Every equivariant run before `e15d36c` acted on near-blank images,** and every one before
`93a8ad4` trained against a frozen target critic (2026-10-02 and 2026-09-29 CORRECTIONs). The
Square ablations stay parked (2026-10-02 RESULTS and STOPPED).

**From 2026-09-30 the update is about 2× faster, and `actor_norm` defaults to `none`**
(EXPERIMENTS.md, 2026-09-30 CODE). Runs set `iid_bn` explicitly; TODO B10 settles the default.
- **wandb:** every Square equivariant residual run is in group `equi-residual-rl`, and every
  Square BC run in `robomimic-square-bc`. Tuned RLPD is in groups `tuned-rlpd` and
  `equi-tuned-rlpd`, in `robomimic-{square,can}-final`.

**BoxCleanup** has had a smoke run only: 59.8 GiB at the start, projecting to about 82 GiB for a
full run, and 0.69 s/step, so 500k steps is about 4 days.

## Running now

**Tuned RLPD, the ResFiT arms** (seed-group `tuned-rlpd-1002`), launched from the pre-registration
commit through `tools/gpu_queue.sh`. Run ids, startup checks and speed are in the LAUNCHED entry.

| Arm | Task | GPU | Bar (110k–150k eval mean) |
|---|---|---|---|
| `tuned-rlpd-square` | Square | 0 | B_sq = 0.508 |
| `tuned-rlpd-can` | Can | 1 | B_can = 0.894 |

- **How it runs:** one runner per GPU in tmux `gpu0` and `gpu1` (queues `~/queues/gpu{0,1}.txt`),
  started from `~/projects/equi-resrl-b667647`; each launch runs in the worktree pinned to its
  commit. No watcher: each run ends at 150k.
- **Expect** a step-0 eval near zero, from a randomly initialized actor. The paper's Tuned RLPD
  passes 90% on Square within 150k, and reaches high success on Can.
- **The verdict:** `tools/tuned_rlpd_verdict.py --project robomimic-{square,can}-final <id>`.

**RAM:** two Square residual runs at 40.3 GiB each fit; a third does not.

## What is established

- **The image-fix pair collapsed by its first eval,** both seeds (2026-10-04 RESULTS). With the
  residual executed, the IID-batch-norm reference does not hold its base policy.
- **The image-scaling bug and its fix.**
  - `tests/test_obs_formats.py` feeds the env's and the buffer's formats. They give the same encoder
    features and the same act(), now also for the vanilla agent.
  - On two trained checkpoints, the executed residual equals the trained one after the fix: cosine
    1.0000.
- **The target-critic bug and its fix.** `tests/test_regression_agent.py` shows the old target
  never changing and the new one following its soft updates.
- **Baseline residual TD3 works.** Square 0.52 → 0.90 (`870ws2c2`), Can 0.92 → 1.00 (`msfkjwab`),
  both single-seed and both with 10 critic heads. The equivariant default is 2 heads. These are
  the best policies the project has.
- **The base rates.** B_sq = 0.508 ± 0.022 and B_can = 0.894 ± 0.014, 500 episodes each
  (2026-09-29 and 2026-10-04 MEASUREMENTs).
- **The equivariant implementation is equivariant.** For one and two arms, for every sweep option,
  and now for the vanilla actor with every norm:
  - the declared representations match independently derived physics;
  - the actor is equivariant and the critic invariant to < 1e-4, in train and eval mode.
- **The whole agent is pinned.**
  - At the reference config `tests/test_regression_agent.py` gives the same parameters, outputs and
    update, bit for bit, as `f3e6315` and `b8e1de7`. A second golden pins `actor_norm=none`.
  - The deep check of all 17 `CANDIDATES` is identical to `b05736d`, tensor for tensor, after the
    no-base-policy mode (2026-10-04 CODE).
- **The update makes no host sync** for any config in `CANDIDATES` (`tests/test_sync_free.py`), nor
  for the vanilla agent at `actor_norm=none`. `iid_bn` still syncs (TODO H8).
- **The BoxCleanup base policy is at about the paper's bar, not above it:** ACT, final 0.53; its
  `best` checkpoint `run_xg0w2r0z_best:v7` comes from a stretch averaging 0.61. Backed up.

## What is not

- **Why the executed residual grows.** It grew without bound in the image-fix pair while the
  critic rated it above the base action (TODO B10). Not established: whether the cause is the
  unbounded actor (`residual_squash=false`), the two critic heads, or something else.
- **The base rate of Square is uncertain.** The base-rate script gave 0.508 ± 0.022. The
  trainer-path step-0 evals pool to 0.549 (439/800). No systematic cause was found. Verdicts use
  0.508, as pre-registered.
- **Five high-scoring runs are still undocumented** (TODO B2): `lgc70vgd`, `q1ocx89o`, `3vl625tv`,
  `36pfxsww` and `zyed0lew`. The four equivariant ones acted on near-blank images.
- **`action_scale` bounds the residual only with `residual_squash=True`.** Otherwise the bound is
  `equi_clip(..., bound=1.0)`, so read `residual_l1` and `residual_l2` against 1.0. Both are the
  update batch's residual, not the executed one.
- **TwoArmBoxCleanup has no exact spatial symmetry** (EQUIVARIANCE.md): the equivariant arm there
  tests a geometric prior, and must not be pooled with Can or Square.
- **`seed` does not reach the environments.** Every episode's initial state is drawn from OS
  entropy, in every run so far, and the decision is to keep it that way (TODO B8).
- **The Can buffer caches are gone** (TODO H7), and Can needs a re-run with both fixes (TODO B6).
  Tuned RLPD needs no cache: it builds its buffers in memory.
- **The gate's RAM estimate is 21 GB per run against a measured 40.3 GiB** (TODO B7, S10). It does
  not count running trainers either.
- **The scalar ablation is still not usable as a control** (STANDARDS 3.3, TODO P4), and it is
  residual-only. **The gate cannot validate the commit it launches** (TODO R14).

## The plan, and why

**1. The ResFiT tuned-RLPD arms** (TODO B12), now. They test whether this setup reproduces the
paper's Tuned RLPD, which makes them the bar any equivariant vanilla result is read against.

**2. Then Aden decides the equivariant arms' design,** and they launch after an AMENDMENT.
- At their provisional design (`iid_bn`), about 19 h per task; at `actor_norm=none` about 15 h.
- Their result separates "the equivariant stack does not learn" from "residual RL is the problem".

**3. BoxCleanup, one arm at a time, without writing caches** (TODO B3, B4, B7). A full run needs
about 82 GiB of RAM, so it can't share the machine with a Square run, and takes about 4 days.

## Do not delete

- `z8yoqylh` exists only as `wandb/run-20260522_083555-z8yoqylh` and a backup in
  `~/equi-resrl-preserve/`; it is not in wandb. TODO H4 would restore it with `wandb sync`.
- `msfkjwab` survives only as `ZXP-S-works:wandb/run-20260226_143630-msfkjwab/`.
- The BoxCleanup base policy is backed up as `~/equi-resrl-preserve/run_xg0w2r0z_best:v7`; its
  sha256 is in EXPERIMENTS.md, 2026-09-27 CODE.
- The pinned worktrees `~/projects/equi-resrl-<commit>` hold their runs' `wandb/` folders, and the
  parked arms launch from `~/projects/equi-resrl-ccff539`.

Three runs have already been lost to deletions.
