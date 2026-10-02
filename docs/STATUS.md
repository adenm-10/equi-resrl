# Status

**Last updated:** 2026-10-02 · **Last verified commit:** the image-scaling fix (EXPERIMENTS.md,
2026-10-02 CODE)

This file describes what is true right now. Rewrite it in place; do not append. For history, see
[EXPERIMENTS.md](EXPERIMENTS.md) and [PROGRESS.md](PROGRESS.md).

## Where the project is

**Every equivariant run so far acted on near-blank images** (EXPERIMENTS.md, 2026-10-02
CORRECTION).
- The env hands act() float images in [0, 1], and the encoder divided them by 255, as it does the
  buffer's uint8 copies. So exploration and evals saw images in [−1, −0.992], while the update saw
  them at full contrast.
- On two trained Square checkpoints the actor executed a mean residual of 0.001 against the 0.3 it
  trained. Those runs ran the base policy plus exploration noise.
- Fixed from the commit of the 2026-10-02 CODE entry: images are converted by dtype, as ResFiT's
  agent does. `equivariance.float_images_as_uint8=True` reproduces the old scaling.
- Every equivariant run before `93a8ad4` also trained against a frozen target critic (2026-09-29
  CORRECTION).

**So no equivariant result yet measures the method.**
- That covers the Can replication, `equi-square-v1`, the sweep and the ablations.
- The IID-batch-norm actors executed almost nothing, so they held BC.
- The others collapsed, after executing whatever the actor computes from a blank image.

**The Square ablations are parked** (Aden's decision, 2026-10-02; EXPERIMENTS.md, 2026-10-02
RESULTS and STOPPED):
- **Did not advance:** `critic-fn` and `clipq`.
- **Collapsed by 30k:** `actor-fn` and `actor-inner`.
- **Died with the machine on 2026-10-02 at about 06:50:** `enc-inner` and `target-fix` seed 2.
- **Never ran:** `squash`, whose gate failed on an unseeded test (since fixed), and `depth`.

**From 2026-09-30 the update is about 2× faster, and `actor_norm` defaults to `none`**
(EXPERIMENTS.md, 2026-09-30 CODE). Runs set `iid_bn` explicitly until TODO B10 settles the default.
- **wandb:** every Square equivariant run is in group `equi-residual-rl`, and every Square BC run in
  `robomimic-square-bc`.

**BoxCleanup** has had a smoke run only. It measured what the pilot needs:
- 59.8 GiB at the start, projecting to about 82 GiB for a full run;
- 0.69 s/step, so 500k steps is about 4 days.

## Running now

**The image-fix pair** (EXPERIMENTS.md, 2026-10-02 PRE-REGISTERED, seed-group
`square-image-fix-1002`):
- What it is: `target-fix`'s launch at the fix, seeds 1 and 2, one per GPU, 100k steps each, judged
  by the training trend.
- Launched from the worktree pinned to the pre-registration's commit, by one runner per GPU in tmux
  `gpu0` and `gpu1` (`tools/gpu_queue.sh`, queues `~/queues/gpu{0,1}.txt`).
- No watcher: each run ends at 100k by itself.
- About 12 h each.
- The LAUNCHED entry records the run ids.

**RAM:** two Square runs at 40.3 GiB each fit; a third does not.

## What is established

- **The image-scaling bug and its fix.**
  - `tests/test_obs_formats.py` feeds the env's and the buffer's formats. They give the same encoder
    features and the same act().
  - `float_images_as_uint8=True` still reproduces the bug.
  - On the two trained checkpoints, the executed residual equals the trained one after the fix: cosine
    1.0000. Before it, the relative difference was 0.999.
- **The target-critic bug and its fix.** `tests/test_regression_agent.py` shows the old target
  never changing and the new one following its soft updates.
- **Baseline residual TD3 works.** Square 0.52 → 0.90 (`870ws2c2`), Can 0.92 → 1.00 (`msfkjwab`),
  both single-seed and both with 10 critic heads. The equivariant default is 2 heads (2026-09-29
  CORRECTION). These are the best policies the project has. No equivariant policy has been shown to
  beat its base policy.
- **The equivariant implementation is equivariant.** For one and two arms and for every sweep option:
  - the declared representations match independently derived physics;
  - the actor is equivariant and the critic invariant to < 1e-4, in train and eval mode.
- **The whole agent is pinned, and the fix did not move the update.**
  - At the reference config, with the actor's batch norm, `tests/test_regression_agent.py` gives the
    same parameters, outputs and update, bit for bit, as `f3e6315` and `b8e1de7`. A second golden
    pins the `actor_norm=none` default.
  - The deep check of all 17 `CANDIDATES` is identical to `ccff539`, tensor for tensor. Only act()
    on the env's float images changed.
- **The update makes no host sync** for any config in `CANDIDATES` (`tests/test_sync_free.py`).
- **The BoxCleanup base policy is at about the paper's bar, not above it.**
  - ACT, final 0.53. The stretch the `best` checkpoint comes from averages **0.61**, against the
    paper's ≈0.6.
  - Its recorded 0.74 at 24k is an outlier draw.
  - The checkpoint is `run_xg0w2r0z_best:v7`, pinned by version and backed up.

## What is not

- **The base rate is uncertain.** The base-rate script gave 0.508 ± 0.022 (500 episodes). The
  trainer-path step-0 evals pool to 0.557 (390/700). No systematic cause was found. The image bug
  does not touch either: the residual starts near zero. Verdicts use 0.508, as pre-registered.
- **Five successful runs are still undocumented** (TODO B2): `lgc70vgd`, `q1ocx89o`, `3vl625tv`,
  `36pfxsww` and `zyed0lew`. The four equivariant ones acted on near-blank images.
- **`action_scale` bounds the residual only with `residual_squash=True`.** Otherwise the bound is
  `equi_clip(..., bound=1.0)`, so read `residual_l1` and `residual_l2` against 1.0. Both are the
  update batch's residual, not the executed one.
- **TwoArmBoxCleanup has no exact spatial symmetry.**
  - The hands are chiral and object placement is asymmetric, and `rotation=(0.0, 0.0)` means object
    yaw never varies.
  - So the equivariant arm there tests a *geometric prior*, not an exploited symmetry, and must not
    be pooled with Can or Square. See [EQUIVARIANCE.md](EQUIVARIANCE.md).
- **`seed` does not reach the environments.** Every episode's initial state is drawn from OS entropy,
  in every run so far, and the decision is to keep it that way (TODO B8).
- **The Can buffer caches are gone** (TODO H7), and Can needs a re-run with both fixes (TODO B6).
- **The gate's RAM estimate is 21 GB per run against a measured 40.3 GiB** (TODO B7, S10). It does
  not count running trainers either.
- **The scalar ablation is still not usable as a control** (STANDARDS 3.3, TODO P4). **The gate cannot
  validate the commit it launches** (TODO R14).

## The plan, and why

**1. The image-fix pair** (TODO B11).
- The first test of the residual algorithm on Square with its actions executed.
- Its outcome decides whether the IID-batch-norm actor's earlier stability survives the fix. If
  either run collapses, it did not.

**2. Then tuned RLPD,** equivariant and ResFiT, on Can and Square (TODO B12). Vanilla RL with no base
policy, as a test of the RL machinery.
- By Aden's decision, the equivariant arms start from the image-fix pair's design, and the ResFiT
  arms from ResFiT's.
- If the pair collapses, the equivariant arms wait for Aden.
- The plan is `docs/plans/2026-10-02-tuned-rlpd.md` (untracked).

**3. BoxCleanup, one arm at a time, without writing caches** (TODO B3, B4, B7).
- A full run needs about 82 GiB of RAM, so it can't share the machine with a Square run.
- At 0.69 s/step, an arm takes about 4 days.

## Do not delete

- `z8yoqylh` exists only as `wandb/run-20260522_083555-z8yoqylh` and a backup in
  `~/equi-resrl-preserve/`; it is not in wandb. TODO H4 would restore it with `wandb sync`.
- `msfkjwab` survives only as `ZXP-S-works:wandb/run-20260226_143630-msfkjwab/`.
- The BoxCleanup base policy is backed up as `~/equi-resrl-preserve/run_xg0w2r0z_best:v7`; its
  sha256 is in EXPERIMENTS.md, 2026-09-27 CODE.
- The pinned worktrees `~/projects/equi-resrl-<commit>` hold their runs' `wandb/` folders, and the
  parked arms launch from `~/projects/equi-resrl-ccff539`.

Three runs have already been lost to deletions.
