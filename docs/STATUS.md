# Status

**Last updated:** 2026-10-13, about 10:00 · **Last verified commit:** this one, with
`resfit-ln-place-1013`'s pre-registration, arms and watcher group (EXPERIMENTS.md, 2026-10-13
PRE-REGISTERED). Its sweep tests (CPU 104, GPU 27 for the ResFiT arms) and watcher tests passed. It
changes no agent code from `0b14c17`, whose gate passed both baseline seeds this morning.

This file describes what is true right now. Rewrite it in place; do not append. For history, see
[EXPERIMENTS.md](EXPERIMENTS.md) and [PROGRESS.md](PROGRESS.md).

## Where the project is

**New on 2026-10-13: without LayerNorm, ResFiT's residual fails as the equivariant one does**
(EXPERIMENTS.md, 2026-10-13 NOTE).
- `noln` (`c1xca5uk`) lost the base by 10k: 0 of 150 at 10k–30k. It never won it back: 0.032 at
  110k–150k, where ResFiT's anchors reach 0.714. `noln-all` (`ra6024xn`, the ViTs' LayerNorms off
  too) is at 0 of 150 so far.
- `rerendered` (`bue9dupf`) holds, and beats the anchors late (0.828 at 110k–150k). `fk9ayyco`
  (rx's range) lands on them (0.716).
- One seed each. Seed 2s of `noln` and `noln-all` are due by the rule.

**The equivariant residual on Square loses the base policy within 10k steps in every run, and the
loss looks like a wrist that rolls after the grasp** (EXPERIMENTS.md, 2026-10-12 FINDING and NOTE).
Three runs on the re-rendered demos, seed 1 each: `6ng62tpm` (the actor's last layer at 1e-4),
`eaqorqki` (at zero) and `doevaarp` (a 50k critic warm-up).
- **The shape:** each holds the base at first (0.50–0.62 at step 0), falls to 0–0.02 by 10k, then
  partly recovers as its residual pulls back (`doevaarp` 0.22 at 30k; `eaqorqki` 0.30 at 80k). The
  training curve's early peak is the base policy, before the residual has grown.
- **The behavior:** the approach and the grasp look like the base's; from about step 70 the wrist
  rolls a little more each step, until the gripper points sideways holding the nut away from the
  peg. The base never does this. Later (`doevaarp` at 100k) a second mode appears: grasps that
  never lift the nut.
- **Tilt is not the whole cause** (EXPERIMENTS.md, 2026-10-12 NOTE, the field test).
  - At `ls0thn4u`'s 10k checkpoint, zeroing any one field of the residual leaves 0–1 success of
    100: −tilt 1, against 57 with the whole residual zeroed. Tilt alone scores 22.
  - `notilt` (`dczf4k7v`), which cannot tilt, scored 0.02, 0.00 and 0.14 at 10k, 20k and 30k.
  - So the damage is spread over the residual's fields, though tilt alone does real harm.
- **The two stacks bound tilt differently.** ResFiT's rx cap is 0.036 env units, set by its demos'
  narrow rx range; the equivariant stack's tilt pair is a disk of radius 0.1, one bound per pair
  (EQUIVARIANCE.md, "Clipping has to respect the representation"). ResFiT's residual is larger and
  saturated, and works, so size alone does not explain the collapse.
- **Ruled out or held:** the 1e-4 start is not the cause (`eaqorqki`), and a better-informed critic
  did not stop the drift (`doevaarp`). A penalty on the residual's size is held: the weight that
  would stop the drift (about 1.8, from the per-sample slope of about 0.018) would hold ResFiT's
  useful residual down too.
- **Now running** (Running now, a self-launching queue): ResFiT's residual given the equivariant
  stack's differences one at a time (`resfit-equi-diff-1012`); the equivariant residual given
  ResFiT's layout (`equi-resfit-arch-1013`); and the equivariant residual's anchors again, to a
  fixed 150k (`equi-residual-long-1012`). Every run goes to a fixed 150k, stopped only on the
  divergence stop. `notilt` has recovered partly: 0.16–0.36 over 90k–140k, below the base.

**Every other experiment was cancelled on 2026-10-12** (DEVIATION), Aden judging their lessons in:
`eaqorqki` was stopped at 80.2k, and `square-trlpd-stab-1010` keeps one seed per arm and no verdict.
- **Equivariant tuned RLPD learned Square, the first tuned RLPD to do so in our trainer** (`vc5o977f`:
  0.50 at 80k), **then its critic ran away** from about 92k (demo value 212 at peak; a target here
  cannot exceed 1, 2026-10-10 MEASUREMENT) and every eval from 100k was 0. Gradient clipping at 1.0
  was on throughout.
- **In `square-trlpd-stab-1010` no critic diverged,** nor did the anchor's seed 2 (`6yh4l5mq`), so
  "no divergence" in an arm is weak evidence. All four passed their learning check. `clipq`
  converged at 230k (0.628) and `ln` at 160k (0.17–0.24). `10x2` and the seed 2s were dropped.
- **The fixed-env baselines' ResFiT residual arm passes** (`oj5k0jqb` 0.800 and `s8opblsd` 0.628
  over 110k–150k, pooled 0.714 against B_sq = 0.582), but the seeds differ by about 4 SE.
  Earlier, every other run was cancelled on 2026-10-10 (DEVIATION), leaving
  `critic-norm-can-1008`'s democodec arm and the fixed-env baselines' ResFiT Can arm without
  verdicts.

**The equivariant critic at the fixed-env baselines' design (`critic_norm=field_norm`) cannot see
the wrist camera, the arm's height or the gripper opening, nor score the z, yaw or gripper actions**
(EXPERIMENTS.md, 2026-10-08 FINDING): `FieldNorm` erases every trivial-type input. That accounts for
both equivariant arms' failure in the matching render. `eyyi99q4` was stopped and the four remaining
field-norm runs held (DEVIATION of that date); equivariant tuned RLPD on Can now runs with
`critic_norm=none` and `layer_norm` (PRE-REGISTERED, `critic-norm-can-1008`). Neither has
escaped so far: both settle at zero online value, `layer_norm` by 5k (cut at 16.4k, DEVIATION) and
`none` by 10k. **With the env's frames through the demos' video codec, the `none` arm learns Can:
0.94 at 10k** (`gkck6eqp`, one seed; NOTE of 2026-10-08), where the same arm uncoded (`xm9cuo92`)
scored 0 at every eval to 30k. **The working account,** which that result supports: with no reward online, a critic's online
value is held up only by what it carries over from similar demo states. ResFiT's `d047it6c` carried
over about 0.3 for 13.6k steps before its first success; the failing critics carry over nothing.
Of the two tests that followed, the codec learned (`gkck6eqp`, above) and RegularLayerNorm with the
critic's start matched to ResFiT's did not (`496bq48s`, 0 at every eval to 100k). ResFiT's stack
without LayerNorm (`9phsy6sf`) was stopped: its actor saturated in its first 200 updates and never
had a gradient again. `xm9cuo92`'s critic, with no norm and stuck at zero, diverged from 54k
(2026-10-09 NOTE); `gkck6eqp`'s, the same critic learning, stayed bounded.

**The demos are now rendered by our env** (TODO B16; CODE, 2026-10-08, "re-rendered demos"):
`local/robomimic-mh-{can,square}-rerendered`, the robomimic demos' states rendered by our env and
stored losslessly, built and verified (NOTE, 2026-10-08). Building them found that **every online
observation after a reset shows the arm one 2 ms physics substep behind the simulator's state**:
robosuite observes without recomputing positions after its last substep. Up to 2 mm at the gripper
and a few levels on the wrist camera, in every run so far. The re-rendered demos reproduce it
(Aden's choice); the env is unchanged. They match the online observations exactly in state, and in
frames up to the GPU's own residue (1 level on a few edge pixels of about 1–2% of Can's wrist
frames; 2026-10-09 CODE). Every Square run since 2026-10-09 trains on them. Aden skipped the
planned validation run on Can (2026-10-09).

**Our conda env rendered the scene differently from the demos, and tuned RLPD learns once it does
not** (EXPERIMENTS.md, 2026-10-07 FINDING and RESULTS, `upstream-clean-1007`). **From 2026-10-07 the
env is pinned to mujoco 3.3.2, whose render matches the demos, and the gate checks it** (CODE,
"mujoco 3.3.2").
- **The mismatch.** mujoco 3.8.1, in the `residual` env from 2026-05-21 to 2026-10-07, renders
  robosuite's floor as darker brownish planks. The demos show light grey tiles, and so does mujoco
  3.3.2. Both load robosuite at the same commit (`77a47512`) with identical scene and texture files.
  Rendering the same seeded scene in both, every pixel off the floor is identical.
- **Its size,** as the gate's render check measures it (`tools/render_check.py`): the mean absolute
  pixel difference between the average of 40 demo first frames and of 40 env reset frames, front
  camera. The check fails above 2.5x the gap between the even and the odd demos:

  | Task | Within demos | Bar | mujoco 3.8.1 | mujoco 3.3.2 (pinned) |
  |---|---|---|---|---|
  | Can | 0.0050 | 0.0125 | 0.0977 | 0.0103 ± 0.0003 |
  | Square | 0.0052 | 0.0130 | 0.0166 ± 0.0007 (the floor is a thin strip) | 0.0086 ± 0.0005 |

  Even the matching render sits at about 2x the within-demo gap (an even colour offset; the demos
  are stored as video), which is why the bar is 2.5x. The wrist camera matches on both tasks.
  Figures, frames and scripts: `outputs/diagnostics/` (outside git).
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
  tensordict 0.9.1, against our 2.6.0, 0.7.0, 0.7.0). Only mujoco changes what the agent sees.
  B15's validation run, our trainer in our env with mujoco alone changed, separates them.
- **Every run of the project so far saw the mismatch,** residual and equivariant included:
  strongly on Can, mildly on Square. So every tuned-RLPD result from 2026-10-04 to 2026-10-07 (the
  screen, τ, the BC term, BC-encoder `44enq26h`, no warm-up `hi8t2hps`) says nothing about its
  change in a matching env. What the mismatch did to the residual and equivariant runs is open.

**Aden's plan (2026-10-07), TODO B15:** consolidate the worktrees into main; pin mujoco 3.3.2
and put the render on each run's record; validate on Can; re-measure the base rates. Then new
baselines: ResFiT and equivariant tuned RLPD on Can, ResFiT and equivariant residual on Square,
with the equivariant design Aden set (B15).
- **Done 2026-10-07:** every record in main, the queue runner fixed, mujoco pinned and on the
  gate and the record, `read_buffer_caches` for runs that must not read a cache, and the
  equivariant agent's critic diagnostics (EXPERIMENTS.md, the three CODE entries of that day).
- **Pre-registered 2026-10-07, after Aden's review:** five arms, ten runs, 150k steps
  (EXPERIMENTS.md, PRE-REGISTERED, "the fixed-env baselines"; seed-group
  `fixed-env-baselines-1007`). Aden added the fifth, the equivariant residual with its last layer
  initialized to zero. The validation run goes first; the other nine follow only if it scores 5
  of 100 at 20k and 30k. The two stacks' differences, the disk bound among them, are in
  ARCHITECTURE.md, "The two stacks side by side", for Aden's review.

**Earlier bugs, still true of the old runs:** every equivariant run before `e15d36c` acted on
near-blank images, and every one before `93a8ad4` trained against a frozen target critic
(2026-10-02 and 2026-09-29 CORRECTIONs). Every ResFiT run before 2026-10-06 logged
`debug/dQ_da_*` as 0 (2026-10-06 CODE). Every equivariant residual run launched before `534cc71`,
`residual-tilt-1012`'s arms among them, scored its eval Q (the videos' "Q =" overlay and
`value/q_trajectories`) at the residual alone, not at the action executed (2026-10-12 CORRECTION).
Training was never affected.

- **wandb:** every Square equivariant residual run is in group `equi-residual-rl`, and every
  Square BC run in `robomimic-square-bc`. Tuned RLPD is in groups `tuned-rlpd` and
  `equi-tuned-rlpd`, in `robomimic-{square,can}-final`. Upstream's trainer's runs are in
  `upstream-rlpd` and `upstream-clean`, recorded offline.

**BoxCleanup** has had a smoke run only: 59.8 GiB at the start, projecting to about 82 GiB for a
full run, and 0.69 s/step, so 500k steps is about 4 days.

## Running now

**Every run below goes to a fixed 150k, stopped only on the divergence stop** (EXPERIMENTS.md,
2026-10-12 AMENDMENT, "every run to a fixed 150k", Aden). The reason: a residual that loses the
base can win it back late (`eaqorqki` 0.04 at 50k, 0.30 at 80k), and every earlier equivariant
residual was stopped before that could show. `fk9ayyco` keeps its own rule: collapse only, to 150k.

**The schedule from 2026-10-13 09:10** (EXPERIMENTS.md, 2026-10-13 DEVIATION and AMENDMENT, and
PRE-REGISTERED `resfit-ln-place-1013`, Aden): the equivariant residual's reference to 150k, then
LayerNorm in ResFiT's residual (is it needed; then which placement), then the actor's bounds and
noise. The actor's architecture, the critic's size and the weight-init scheme are set aside.
`arch-wide` (`grvwhcfr`) was stopped at 10k (0.64, **0.12**: the only equivariant residual not at
0.00–0.02 at 10k), and `arch`, `smallcritic` and `combined` dropped. It launches itself through
`outputs/queue/launchers/`. Times are estimates: a ResFiT run takes about 11 h from launch to 150k,
an equivariant one about 29 h, each including about 1 h 15 min for the gate and labelling.

| Launch | Run | Group | Question | 30k readout | 150k |
|---|---|---|---|---|---|
| done | `fk9ayyco` ResFiT rx: 57 of 150, 0.716 | `resfit-rx-1012` | rx range | done | done |
| done | `bue9dupf` ResFiT `rerendered`: 62 of 150, 0.828 | `resfit-equi-diff-1012` | demos | done | done |
| done | `c1xca5uk` ResFiT `noln`: 0 of 150, 0.032. **Breaks** | `resfit-equi-diff-1012` | (1) | done | done |
| running | `dczf4k7v` equi `notilt`: 0.16 at 140k | `residual-tilt-1012` | tilt | done | Tue ~09:20 |
| running | `oav5gk76` equi `init0` long, seed 2: 0.62, 0.00, 0.00, 0.20 | `equi-residual-long-1012` | init0 | done (10 of 150) | Wed ~07:00 |
| running | `ra6024xn` ResFiT `noln-all`: 0 of 150, **breaks** | `resfit-equi-diff-1012` | — | done | Tue ~14:40 |
| Tue ~09:20 | equi baseline (`6ng62tpm`'s launch), seed 2 | `equi-residual-long-1012` | reference | Tue ~16:15 | Wed ~14:30 |
| Tue ~09:30 | equi baseline, seed 3 | `equi-residual-long-1012` | reference spread | Tue ~16:30 | Wed ~14:45 |
| Tue ~14:45 | ResFiT `noln`, seed 2 | `resfit-equi-diff-1012` | (1) | Tue ~18:15 | Wed ~02:00 |
| Wed ~02:00 | ResFiT `noln-actor`, seed 1 | `resfit-ln-place-1013` | (2a) | Wed ~06:00 | Wed ~13:00 |
| Wed ~07:00 | ResFiT `noln-critic`, seed 1 | `resfit-ln-place-1013` | (2a) | Wed ~11:00 | Wed ~18:00 |
| Wed ~13:00 | ResFiT `noln-vit`, seed 1 | `resfit-ln-place-1013` | (2a) | Wed ~17:00 | Wed ~24:00 |
| Wed ~14:30 | ResFiT `limits`, seed 1 | `resfit-equi-diff-1012` | actor bounds | Wed ~18:30 | Thu ~01:30 |
| Wed ~14:45 | ResFiT `noise`, seed 1 | `resfit-equi-diff-1012` | actor noise | Wed ~18:45 | Thu ~01:45 |
| Wed ~18:00 | ResFiT `noln-actor`, seed 2 | `resfit-ln-place-1013` | (2a) | Wed ~22:00 | Thu ~05:00 |
| Wed ~24:00 | ResFiT `noln-critic`, seed 2 | `resfit-ln-place-1013` | (2a) | Thu ~04:00 | Thu ~11:00 |
| Thu ~01:30 | ResFiT `noln-vit`, seed 2 | `resfit-ln-place-1013` | (2a) | Thu ~05:30 | Thu ~12:30 |
| Thu ~01:45 | open: seed 2 of `limits` or `noise` if either breaks, else a second `init0` or `notilt` seed | — | — | | |

- **The queue:** tmux `rfe-queue` runs `slot_queue.sh` over `jobs-resfit-equi-diff-1012.txt`. Each
  job goes into the first slot to have been free for 300 s, two runs per GPU, and gets its own tmux
  session, `q-<arm>-<seed>`. Its log is `outputs/queue/launchers/slot_queue.log`. The scripts are in
  `outputs/queue/launchers/`, which is gitignored (TODO: move them into `tools/` once the queue is
  empty).
- **Watchers,** each enforcing its group's rule (`tools/square_rerendered_watch.py`):
  - from the 2026-10-12 amendment's worktree: `rfe-watch` (`resfit-equi-diff-1012`), `long-watch`
    (`equi-residual-long-1012`) and `tilt-watch` (`residual-tilt-1012`);
  - from `resfit-ln-place-1013`'s worktree: `ln-watch` (`resfit-ln-place-1013`);
  - `rx-watch` and `arch-watch` were stopped on 2026-10-13, their groups' runs ended.
- **tmux, 2026-10-13:** the idle sessions of finished runs, smokes, launchers and watchers were
  removed. Left: the four watchers, `rfe-queue`, the runs (`sq-tilt-notilt`, `q-*`), and the old
  queue runners `gpu0` and `gpu1` (idle) and `gpu1-bars`.
- **Seed 2s:** `noln`'s and `resfit-ln-place-1013`'s are in the queue; any other a rule calls for is launched by hand.
- **For the next session:** see TODO 5e.

- **What each run records beyond its evals and videos:**
  - `notilt` and `diag` log the residual and the critic's slope per action field
    (`debug/residual_field/*`, `debug/dQ_da_field/*`) and save `models/step_<N>.pt`. They launched
    before `534cc71`, so their eval Q overlays keep the old error and they log no `eval/tilt_*`; the
    field test, run from `534cc71`, has both.
  - `fk9ayyco` (from `60fce72`) has neither, as Aden chose: its evals and videos suffice.
- **Resources:** about 20 GB of RAM per run at start and 35 GB by 150k; about 9.9 GB of GPU memory
  per run, so two per GPU. **The disk had 29.6 GB free at 10:00 on 2026-10-13,** after the pip
  and conda caches were cleared (10 GB, Aden's approval). It falls about 0.45 GB an hour on
  average, in bursts at evals and run ends, so it lasts to about Friday. The old worktrees are
  about 4 MB each; the next to clear is the wandb artifact cache (14 GB, +4 GB a day). Equivariant
  checkpoints are 87 MB each and ResFiT's 223 MB.
- **How to follow them:** wandb `robomimic-square-final`, groups `residual-rl` (`fk9ayyco`) and
  `equi-residual-rl`, notes starting with the seed group. `outputs/queue/watch.log` has every
  watcher action, and consoles are in `outputs/queue/`.
  - Watch `eval/success_rate`, `debug/q_data_online` against `debug/q_data_demo`,
    `train/residual_l1_magnitude` and `debug/dQ_da_mean_abs`, and for `notilt` and `diag` the
    per-field keys.

## The machine

- **`residual`** (every run of ours): mujoco 3.3.2 from 2026-10-07 (3.8.1 before, in every
  earlier run), torch 2.6.0, torchrl 0.7.0, tensordict 0.7.0, escnn 1.0.11. Each run logs these
  as `versions` in its config.
- **`resfit-upstream`** (upstream's clone, `~/projects/resfit-upstream` at `66262b06`): its
  README's env, with torch 2.7.1+cu128 and torchvision 0.22.1+cu128 in place of the 2.14.1 its
  unpinned install gives, which cannot load torchcodec 0.4.0 or torchrl 0.9.2's extension. Freeze
  files beside the clone. Launcher: `~/projects/resfit-upstream-runs/launch.sh`.
- **RAM:** about 15 GB per Can tuned-RLPD run at 30k. Two Square residual runs at 40.3 GiB each
  fit; a third does not.
- **Disk:** 34 GB free on 2026-10-12, before `residual-tilt-1012`'s checkpoints (up to 5.4 GB).
  Two cache deletions on 2026-10-07, with Aden's approval: upstream's buffer caches (21 GB) and
  `online_buffer_cache/66c46c51` (16 GB, Square, frames from mujoco 3.8.1, which the new key cannot
  reach). `offline_buffer_cache/1d31ed9b` (6.4 GB, Square) stays: its key did not move.
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
- **The base rates, measured again in the matching env on 2026-10-07:** B_sq = 0.582 ± 0.022 and
  B_can = 0.852 ± 0.016, 500 episodes each, base policies `v9` and `v7` (EXPERIMENTS.md, LAUNCHED
  and MEASUREMENT of that date). In the mismatched env they were 0.508 and 0.894. B_can's ten evals
  are more spread than binomial noise (SD 0.080 against 0.050).
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
- **The buffer cache keys** do not record a residual buffer's normalization path (TODO B7). The
  online key records mujoco from 2026-10-07. `read_buffer_caches=false` keeps a run off every
  cache; B15's baselines all set it.
- **The gate's RAM estimate is 21 GB per run against a measured 40.3 GiB** (TODO B7, S10). It does
  not count running trainers either.
- **The scalar ablation is still not usable as a control** (STANDARDS 3.3, TODO P4), and it is
  residual-only. B15's equivariant arms match ResFiT's choices where equivariance allows, but the
  stacks still differ in encoder, augmentation and normalization. **The gate cannot validate the
  commit it launches** (TODO R14).
- **An equivariant run's random stream depends on its eval schedule:** the actor's `std=0` path,
  taken by every eval and actor loss, draws a random number it scales by 0 (ARCHITECTURE.md,
  "Known rough edges"). Left as it is, for comparability.
- **Which of the equivariant residual's fields breaks the policy on Square.** Tilt is an inference
  from the videos (2026-10-12 FINDING); `notilt` and the field test test it.
- **Whether training and execution should combine the residual alike** (TODO H11). The actor loss
  and the TD target clip each pair of base plus residual to the unit disk; the env adds them and
  robosuite clips each component. They differ only for pairs longer than 1.

## The plan, and why

**0. Now (2026-10-12): the tilt diagnosis** (Running now; TODO B16 item 5d). Why: the
equivariant residual's collapse on Square looks like a wrist roll, and its tilt range is about 3×
ResFiT's on rx. What follows each outcome is set in TODO 5d: a tilt bound of 0.03 iff `notilt`
holds the base through 30k but fails its 100k check; the penalty on the residual's size only if the
per-field slopes separate drifting fields from useful ones; the critic as the target if `notilt`
falls in a new mode.

**1. B16's remaining items** (TODO): the ablations on Can that set the residual's design, and
ResFiT's residual on the same demos. Removing the old worktrees is Aden's (TODO B15 item 1).

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
- `outputs/diagnostics/`: the render figures and the scripts that made them, cited by the FINDING,
  and the frames and repeats behind the gate's bar (`render_frames_mujoco{332,381}.npz`,
  `render_check_mujoco332_repeats10.jsonl`), cited by the 2026-10-07 CODE entry.
- `outputs/env/`: `pip freeze` of `residual` before and after the mujoco pin.

Three runs have already been lost to deletions.
