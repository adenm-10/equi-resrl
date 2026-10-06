# Status

**Last updated:** 2026-10-06 · **Last verified commit:** the Can screen's pre-registration, on the
critic diagnostics (EXPERIMENTS.md, 2026-10-06 CODE)

This file describes what is true right now. Rewrite it in place; do not append. For history, see
[EXPERIMENTS.md](EXPERIMENTS.md) and [PROGRESS.md](PROGRESS.md).

## Where the project is

**Tuned RLPD on ResFiT's stack failed on both tasks** (EXPERIMENTS.md, 2026-10-06 RESULTS).
`8ma5alnm` (Square) and `lg8qad3l` (Can) were at 0.00 at every eval to 150k, with 0 training
successes. The paper's Tuned RLPD is off zero by 20–30k on both. So this setup does not yet
reproduce the paper's baseline, and an equivariant tuned-RLPD result could not be read.
- **The failure:** the actor reached its Tanh bound within about 100 updates, before the critic
  had learned anything, and stayed near it (`action_l1` 0.68–0.90; the demos' is 0.29–0.32).
- **The critic** values the policy's own rollouts at about 0. By inference, it values the
  policy's action on demo states at about twice the demos' own value, so nothing pulls the
  actor back toward the demos. The control arm of the screen below measures this directly.
- **Not mainly exploration:** with demos in every batch, RLPD does not need an online success.
- **Our config equals upstream's** `rlpd_can_config`, at its original 10 heads, on every field
  the two share. The candidate gaps are between that config and the paper's text: no critic
  warm-up (upstream's RLPD trainer has none), and τ 0.01 against the text's 0.005.

**Aden's order (2026-10-06):** ResFiT tuned RLPD on Can, then on Square, then the equivariant
stack on Can, then on Square. Residual RL after.

**No equivariant residual design has yet held its base policy** with its residual executed
(2026-10-04 RESULTS, the image-fix pair). The tuned-RLPD failure looks like the same pattern, an
actor that runs to its bound while the critic scores the bound above the data's actions, but that
is a reading, not a test (TODO B10).

**Every equivariant run before `e15d36c` acted on near-blank images,** and every one before
`93a8ad4` trained against a frozen target critic (2026-10-02 and 2026-09-29 CORRECTIONs).
**Every ResFiT run before 2026-10-06 logged `debug/dQ_da_*` as 0** (2026-10-06 CODE).

- **wandb:** every Square equivariant residual run is in group `equi-residual-rl`, and every
  Square BC run in `robomimic-square-bc`. Tuned RLPD is in groups `tuned-rlpd` and
  `equi-tuned-rlpd`, in `robomimic-{square,can}-final`; upstream's trainer's runs are in
  `upstream-rlpd`, recorded offline.

**BoxCleanup** has had a smoke run only: 59.8 GiB at the start, projecting to about 82 GiB for a
full run, and 0.69 s/step, so 500k steps is about 4 days.

## Running now

**The tuned-RLPD Can screen** (seed-group `tuned-rlpd-can-screen-1006`; EXPERIMENTS.md,
2026-10-06 PRE-REGISTERED). Six 30k-step arms on Can, seed 1, through `tools/gpu_queue.sh`, four
at a time: runners `gpu0` and `gpu1`, plus `gpu0b` and `gpu1b`.

| Arm | What changes | Runner |
|---|---|---|
| `upstream-rlpd-can` | upstream's own trainer | `gpu0` |
| `tuned-rlpd-can-control` | nothing; logs the critic diagnostics | `gpu0b` |
| `tuned-rlpd-can-critic-warmup` | 10k critic-only updates | `gpu1` |
| `tuned-rlpd-can-actor-lr` | actor LR 1e-6 | `gpu1b` |
| `tuned-rlpd-can-zero-init` | actor last layer starts at zero | `gpu0`, second |
| `tuned-rlpd-can-combined` | all three | `gpu1`, second |

- **The verdict per arm:** off zero iff its 20k and 30k evals score at least 5 of 100. The
  off-zero arm of ours with the most successes runs 150k next, as an AMENDMENT to
  `tuned-rlpd-1002`.
- **Upstream's arm** writes its caches under `local_runs/upstream-rlpd/` and runs wandb offline:
  its history is in its local wandb record, and it deletes its own run folder at the end.
- **The disk is at 100%:** 29 GB free before upstream's 16 GB online cache. Delete
  `local_runs/upstream-rlpd/` when the screen is done (TODO B12).

**RAM:** about 15 GB per Can tuned-RLPD run. Two Square residual runs at 40.3 GiB each fit; a
third does not.

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
  residual-only. So no arm yet isolates equivariance in vanilla RL: the ResFiT and equivariant
  tuned-RLPD arms differ in seven ways. **The gate cannot validate the commit it launches** (TODO R14).

## The plan, and why

**1. Get ResFiT tuned RLPD learning on Can** (TODO B12), the screen above. It tells whether our
mode matches upstream's trainer in outcome, and whether any of three changes from the residual
recipe gets it off zero by 30k. The promoted arm then runs 150k against B_can = 0.894.

**2. The same on Square,** with what worked on Can.

**3. The equivariant stack on Can, then Square.** It first needs `critic_diagnostics` and its
design: the actor's norm and output bound, plus whatever the screen settles.

**4. Residual RL after that** (TODO B10): why the executed residual grows to the clip.

**BoxCleanup** (TODO B3, B4, B7) waits: a full run needs about 82 GiB of RAM and about 4 days.

## Do not delete

- `z8yoqylh` exists only as `wandb/run-20260522_083555-z8yoqylh` and a backup in
  `~/equi-resrl-preserve/`; it is not in wandb. TODO H4 would restore it with `wandb sync`.
- `msfkjwab` survives only as `ZXP-S-works:wandb/run-20260226_143630-msfkjwab/`.
- The BoxCleanup base policy is backed up as `~/equi-resrl-preserve/run_xg0w2r0z_best:v7`; its
  sha256 is in EXPERIMENTS.md, 2026-09-27 CODE.
- The pinned worktrees `~/projects/equi-resrl-<commit>` hold their runs' `wandb/` folders, and the
  parked arms launch from `~/projects/equi-resrl-ccff539`.

Three runs have already been lost to deletions.
