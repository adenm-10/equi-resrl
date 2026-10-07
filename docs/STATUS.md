# Status

**Last updated:** 2026-10-07 · **Last verified commit:** the commit recording the render finding
(EXPERIMENTS.md, 2026-10-07 FINDING, "our env renders Can's front camera differently from the
demos", and the RESULTS after it)

This file describes what is true right now. Rewrite it in place; do not append. For history, see
[EXPERIMENTS.md](EXPERIMENTS.md) and [PROGRESS.md](PROGRESS.md).

## Where the project is

**Our conda env renders the scene differently from the demos, and tuned RLPD learns once it does
not** (EXPERIMENTS.md, 2026-10-07 FINDING and RESULTS, `upstream-clean-1007`).
- **The mismatch.** mujoco 3.8.1, in the `residual` env since 2026-05-21, renders robosuite's floor
  as darker brownish planks. The demos show light grey tiles, and so does mujoco 3.3.2. Both load
  robosuite at the same commit (`77a47512`) with identical scene and texture files. Rendering the
  same seeded scene in both, every pixel off the floor is identical.
- **Its size,** as the mean absolute pixel difference between demo first frames and env reset
  frames, front camera:

  | Task | Within demos | Our env (mujoco 3.8.1) | mujoco 3.3.2 |
  |---|---|---|---|
  | Can | 0.007 | 0.0995 | 0.010 |
  | Square | 0.007 | 0.019 (the floor is a thin strip at the top) | 0.009 |

  The wrist camera matches on both tasks. Figures and scripts: `outputs/diagnostics/` (outside git).
- **Upstream's tuned RLPD, unmodified, in upstream's env learns Can:** 0 at 10k, 0.88 at 20k, 0.80
  at 30k (`zwmk393l`). The same code, config and seed in our env scored 0 at every eval
  (`rdoj0ex8`). Its critic loss sat about 30× ours and its actor's gradients about 10× ours; its
  actor left the Tanh bound; by 10k its rollouts followed the demos' reach-then-bin sequence.
- **The reading:** in our env the critic can tell every demo frame from every online frame by the
  floor. "Demo floor, high value; env floor, zero" then fits all of its data without the action.
  That accounts for the signature every tuned-RLPD run here showed:
  - an action-blind critic (`dQ_da` about 6e-6) and an actor at its bound;
  - Q on online transitions at exactly 0, and on demos at 0.65–0.85;
  - Q about 0 from the first frame of every eval episode.
- **Not yet separated** from upstream's env's other differences (torch 2.7.1, torchrl 0.9.2,
  tensordict 0.9.1, against our 2.6.0, 0.7.0, 0.7.0). Only mujoco changes what the agent sees;
  the mujoco-only run is next (TODO B15).
- **Every run of the project so far saw the mismatch,** residual and equivariant included:
  strongly on Can, mildly on Square. So every tuned-RLPD result from 2026-10-04 to 2026-10-07 (the
  screen, τ, the BC term, BC-encoder `44enq26h`, no warm-up `hi8t2hps`) says nothing about its
  change in a matching env. What the mismatch did to the residual and equivariant runs is open.

**Aden's plan (2026-10-07), TODO B15:** consolidate the worktrees into main; pin mujoco 3.3.2
and put the render on each run's record; validate on Can; re-measure the base rates. Then new
baselines: ResFiT and equivariant tuned RLPD on Can, ResFiT and equivariant residual on Square,
with the equivariant design Aden set (B15).

**Earlier bugs, still true of the old runs:** every equivariant run before `e15d36c` acted on
near-blank images, and every one before `93a8ad4` trained against a frozen target critic
(2026-10-02 and 2026-09-29 CORRECTIONs). Every ResFiT run before 2026-10-06 logged
`debug/dQ_da_*` as 0 (2026-10-06 CODE).

- **wandb:** every Square equivariant residual run is in group `equi-residual-rl`, and every
  Square BC run in `robomimic-square-bc`. Tuned RLPD is in groups `tuned-rlpd` and
  `equi-tuned-rlpd`, in `robomimic-{square,can}-final`. Upstream's trainer's runs are in
  `upstream-rlpd` and `upstream-clean`, recorded offline.

**BoxCleanup** has had a smoke run only: 59.8 GiB at the start, projecting to about 82 GiB for a
full run, and 0.69 s/step, so 500k steps is about 4 days.

## Running now

Nothing. Both GPUs are idle. The four queue runners were stopped on 2026-10-07 and their tmux
sessions closed; B15's launches start new ones from the main checkout (`tools/gpu_queue.sh` can
now run from there).

## The machine

- **`residual`** (every run of ours): mujoco 3.8.1 (to be pinned to 3.3.2, B15), torch 2.6.0,
  torchrl 0.7.0, tensordict 0.7.0, escnn 1.0.11.
- **`resfit-upstream`** (upstream's clone, `~/projects/resfit-upstream` at `66262b06`): its
  README's env, with torch 2.7.1+cu128 and torchvision 0.22.1+cu128 in place of the 2.14.1 its
  unpinned install gives, which cannot load torchcodec 0.4.0 or torchrl 0.9.2's extension. Freeze
  files beside the clone. Launcher: `~/projects/resfit-upstream-runs/launch.sh`.
- **RAM:** about 15 GB per Can tuned-RLPD run at 30k. Two Square residual runs at 40.3 GiB each
  fit; a third does not.
- **Disk:** 35 GB free, after upstream's buffer caches (21 GB) were deleted on 2026-10-07. Once the
  online cache key records the render (B15), `online_buffer_cache/66c46c51` (16 GB, Square, old
  render) is read by nothing too.
- **Records:** every run's local record is in the main checkout: `wandb/`, `outputs/runs/`,
  `outputs/<date>/<time>/` (Hydra), `outputs/queue/`. A run launched from a pinned worktree writes
  them there too (ARCHITECTURE.md, "Run packages").

## What is established

- **The render mismatch and upstream's result,** above.
- **The squash is equivariant.** `equi_tanh` commutes with the group and bounds each rotating
  pair by its length and each scalar on its own. `action_xy` is normalized with one scale shared
  by x and y and no shift, and `action_rx_ry` by identity, so the bound is a disk in env units.
  54 tests in `tests/test_equivariance.py` pass (2026-10-07).
- **The image-fix pair collapsed by its first eval,** both seeds (2026-10-04 RESULTS). With the
  residual executed, unbounded (`residual_squash=false`), the IID-batch-norm reference does not
  hold its base policy. In the mismatched env.
- **The image-scaling bug and its fix.**
  - `tests/test_obs_formats.py` feeds the env's and the buffer's formats. They give the same encoder
    features and the same act(), also for the vanilla agent.
  - On two trained checkpoints, the executed residual equals the trained one after the fix: cosine
    1.0000.
- **The target-critic bug and its fix.** `tests/test_regression_agent.py` shows the old target
  never changing and the new one following its soft updates.
- **Baseline residual TD3 works, in the mismatched env.** Square 0.52 → 0.90 (`870ws2c2`), Can
  0.92 → 1.00 (`msfkjwab`), both single-seed and both with 10 critic heads. The equivariant default
  is 2 heads; our `CriticConfig.num_q` default has been 2 since 2026-04-23.
- **The base rates, in the mismatched env.** B_sq = 0.508 ± 0.022 and B_can = 0.894 ± 0.014, 500
  episodes each (2026-09-29 and 2026-10-04 MEASUREMENTs). B15 re-measures both.
- **The equivariant implementation is equivariant.** For one and two arms, for every sweep option,
  and for the vanilla actor with every norm:
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

- **Whether mujoco alone accounts for upstream's result,** rather than torch, torchrl or
  tensordict (B15's validation run).
- **What the render mismatch did to the residual and equivariant runs,** and the base rates under
  the correct render.
- **Why the executed residual grows.** It grew to the clip in the image-fix pair, unbounded, while
  the critic rated it above the base action (TODO B10). B15's equivariant residual arm bounds it.
- **The base rate of Square is uncertain.** The base-rate script gave 0.508 ± 0.022. The
  trainer-path step-0 evals pool to 0.549 (439/800). No systematic cause was found.
- **Five high-scoring runs are still undocumented** (TODO B2): `lgc70vgd`, `q1ocx89o`, `3vl625tv`,
  `36pfxsww` and `zyed0lew`. The four equivariant ones acted on near-blank images.
- **`action_scale` bounds the residual only with `residual_squash=True`.** Otherwise the bound is
  `equi_clip(..., bound=1.0)`, so read `residual_l1` and `residual_l2` against 1.0. Both are the
  update batch's residual, not the executed one.
- **TwoArmBoxCleanup has no exact spatial symmetry** (EQUIVARIANCE.md): the equivariant arm there
  tests a geometric prior, and must not be pooled with Can or Square.
- **`seed` does not reach the environments.** Every episode's initial state is drawn from OS
  entropy, in every run so far, and the decision is to keep it that way (TODO B8).
- **The buffer cache keys** record neither the normalization path (TODO B7) nor the render (B15).
  Tuned RLPD builds its buffers in memory; B15's baselines all will.
- **The gate's RAM estimate is 21 GB per run against a measured 40.3 GiB** (TODO B7, S10). It does
  not count running trainers either.
- **The scalar ablation is still not usable as a control** (STANDARDS 3.3, TODO P4), and it is
  residual-only. B15's equivariant arms match ResFiT's choices where equivariance allows, but the
  stacks still differ in encoder, augmentation and normalization. **The gate cannot validate the
  commit it launches** (TODO R14).
- **The equivariant agent logs no critic diagnostics:** the trainer calls `critic_diagnostics` for
  ResFiT's agent only (B15).

## The plan, and why

**1. B15, in order:** worktrees into main; mujoco 3.3.2 with the render on the record and in the
gate; tuned RLPD on Can in our trainer, which must get off zero as `zwmk393l` did; base rates
again; then the four baselines.

**2. Read the residual and equivariant history again** once the baselines report: which
conclusions survive the render fix.

**3. BoxCleanup** (TODO B3, B4, B7) waits: a full run needs about 82 GiB of RAM and about 4 days.

## Do not delete

- `z8yoqylh` exists only as `wandb/run-20260522_083555-z8yoqylh` and a backup in
  `~/equi-resrl-preserve/`; it is not in wandb. TODO H4 would restore it with `wandb sync`.
- `msfkjwab` survives only as `ZXP-S-works:wandb/run-20260226_143630-msfkjwab/`.
- The BoxCleanup base policy is backed up as `~/equi-resrl-preserve/run_xg0w2r0z_best:v7`; its
  sha256 is in EXPERIMENTS.md, 2026-09-27 CODE.
- **Three offline records were never synced,** so their only copies are in `wandb/` and, copied
  and verified, in `~/equi-resrl-preserve/wandb-offline-only/`: **`rdoj0ex8`** (upstream's
  trainer in our env), **`zwmk393l`** (upstream's trainer in its env) and `9atavlcs` (that run's
  first launch, which died at startup).
- **The 15 pinned worktrees hold no records now.** Their 32 records, and upstream's clone's 2,
  were moved into `wandb/` and `outputs/` on 2026-10-07 (EXPERIMENTS.md, CODE). Removing the
  worktrees is Aden's (TODO B15 has the commands).
- `outputs/diagnostics/`: the render figures and the scripts that made them, cited by the FINDING.

Three runs have already been lost to deletions.
