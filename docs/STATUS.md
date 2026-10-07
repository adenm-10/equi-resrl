# Status

**Last updated:** 2026-10-07 · **Last verified commit:** the no-warm-up pre-registration
(EXPERIMENTS.md, 2026-10-07 CODE and PRE-REGISTERED, "with no warm-up")

This file describes what is true right now. Rewrite it in place; do not append. For history, see
[EXPERIMENTS.md](EXPERIMENTS.md) and [PROGRESS.md](PROGRESS.md).

## Where the project is

**Tuned RLPD on ResFiT's stack does not learn on Can, under our trainer or upstream's**
(EXPERIMENTS.md, 2026-10-06 RESULTS, the screen). Six 30k-step arms on Can scored 0 in all 1,200
eval episodes, and our five scored 0 in 750 training episodes. The paper's Tuned RLPD is at about
0.7 there by 25k. Before that, `8ma5alnm` (Square) and `lg8qad3l` (Can) scored 0 at every eval
to 150k.
- **Upstream's own trainer fails the same way.** `train_rlpd_dexmg.py` at its released
  `rlpd_can_config`, as the paper's launcher runs it, scored 0 at 0, 10k, 20k and 30k, with the
  same signature. So the failure is not in our no-base-policy mode.
- **None of the three changes from the residual recipe got it off zero,** alone or together:
  actor LR 1e-6, a 10k-update critic warm-up, a zero-initialized last layer. They change how
  fast the actor reaches its Tanh bound, not where it ends up.
- **The mechanism, now measured.** It is offline RL's extrapolation error, not a Q blowup:
  - On demo states the critic values the policy's action slightly above the demo's, by about
    0.024, at every log step of every arm.
  - The TD target bootstraps through the policy's action, so that margin compounds over each
    episode. Demo values settle at 0.79–0.85 on average, against the demos' true 0.398, and a simple
    fixed-point formula predicts them to within 0.05.
  - The critic's slope at the policy's action is tiny (`dQ_da` about 6e-6), but its sign holds,
    and Adam moves the actor at its full learning rate along it, to the bound.
  - The online half is valued at about 0 for any action, correctly: the saturated policy never
    succeeds. So nothing pulls the actor back toward the demos' actions.
  - An actor started at the dataset midpoint, with a critic trained first, held the margin at
    about 0. It drifted out anyway, and the margin turned positive as it did.
- **Not exploration, and not the demos.** The runs use the same 300 MH demos the BC policies
  were trained on, and with demos in every batch no online success is needed to see reward.
- **What could fix it,** from the diagnostics: a BC term on the demo half, as in TD3+BC. The paper's
  Tuned RLPD does not use one. Our trainer now runs upstream's (`agent.bc_loss_coef`; TODO B14,
  done). τ 0.005, the paper's text, should not change the fixed point.
- **Neither got it off zero** (EXPERIMENTS.md, 2026-10-07 RESULTS, "τ 0.005 and a BC term"). τ
  0.005 changed nothing, as predicted. The BC term held the actor near the demos' actions (0.15
  per axis against 0.8) and lowered the demo values to about 0.65, but no rollout succeeded, in
  eval or training. Its imitation is coarse; the BC loss does not train the encoder, and nothing
  else that does has a signal.

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

**The BC term training the encoder, tuned RLPD on Can** (seed-group
`tuned-rlpd-can-bc-encoder-1007`; EXPERIMENTS.md, 2026-10-07 PRE-REGISTERED). One 30k-step arm,
`tuned-rlpd-can-bc-encoder` (`agent.bc_backprop_encoder=true`), on runner `gpu1`, against
`tuned-rlpd-can-bc` (`kc8z7vsf`). Off zero iff its 20k and 30k evals score at least 5 of 100.

**Tuned RLPD on Can with no warm-up** (seed-group `tuned-rlpd-can-no-warmup-1007`; EXPERIMENTS.md,
2026-10-07 PRE-REGISTERED). One 30k-step arm, `tuned-rlpd-can-no-warmup`
(`algo.learning_starts=0`: no random steps, no critic warm-up), on runner `gpu0`, against the
screen's control `r1hvijqh`. Same verdict rule.

**Ready, waiting on disk: upstream's code out of the box** (seed-group `upstream-clean-1007`;
EXPERIMENTS.md, 2026-10-07 PRE-REGISTERED). A fresh clone of upstream at `66262b06` in
`~/projects/resfit-upstream`, in a new conda env built from its README, `resfit-upstream`
(torchrl 0.9.2 and mujoco 3.3.2, against our 0.7.0 and 3.8.1). Its RLPD script on Can for 30k
steps, then its residual on Can with our BC policy as the env's control. Launched by hand in tmux,
not through `gpu_queue.sh`.
- **Blocked:** each run writes about 21 GB of caches, and 17 GB are free. Deleting
  `local_runs/upstream-rlpd/` (21 GB, read by nothing) frees the room; Aden approved it, but
  the delete needs his hands (the session's permissions block `rm -rf`).

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

**1. Get ResFiT tuned RLPD learning on Can** (TODO B12). The screen ruled out the three
recipe changes and showed upstream's trainer fails too; τ 0.005 and a BC term did not fix it
either. Next: upstream's code in its own environment, which rules the package versions in or
out. Then the BC loss training the encoder, a longer n-step, or a larger coefficient. A BC term
makes this TD3+BC-style RLPD, not the paper's Tuned RLPD; whether that is an acceptable comparison
baseline is Aden's call. An off-zero arm would run 150k against B_can = 0.894.

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
- **`rdoj0ex8`, upstream's Can arm,** exists only as its offline record,
  `~/projects/equi-resrl-187d895/wandb/offline-run-20261006_105334-rdoj0ex8/`. Its trainer
  deleted its run folder, and it was never synced.

Three runs have already been lost to deletions.
