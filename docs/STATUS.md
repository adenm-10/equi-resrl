# Status

**Last updated:** 2026-10-17, about 14:00 · **Last verified commit:** this file's own commit, the
pre-registration of `equi-speed-matched-1017` and its code (config only); its runs launch from it, in
its own pinned worktree. The running runs launched from `dd6f753` (`equi-actor-speed-1016`) and
`974e281` (`vnpqtnva`). No agent code changed since `0b14c17`. Main is ahead of origin and not pushed:
pushing waits for Aden.

**Now: `equi-speed-matched-1017` is pre-registered** (EXPERIMENTS.md, 2026-10-17 PRE-REGISTERED;
[the plan](plans/2026-10-17-speed-matched.md), `docs/plans/`, local; TODO B19). Four config-only arms
at seed 2, each an existing seed-2 run plus `agent.actor_lr`. Next: pin a worktree at this commit,
dry-run each arm against its parent's recorded argv, then start `rfe-queue` with
`jobs-equi-speed-matched-1017.txt` and the watch-only `matched-watch`. Slots free as the four running
runs reach 150k: about 21:50 on 10-17, then about 00:00, 00:05 and 00:30 on 10-18. Also due:
- the group's LAUNCHED entry, once each run is in wandb (about 1 h 20 min after its launch);
- **`vnpqtnva`'s 150k NOTE** (about 21:50), reported only.
- **`equi-actor-speed-1016`'s 150k readout** (about 00:00–00:30 on 10-18) against its pre-registered
  bars and reading (TODO B18).

This file describes what is true right now. Rewrite it in place; do not append. For history, see
[EXPERIMENTS.md](EXPERIMENTS.md) and [PROGRESS.md](PROGRESS.md).

## Where the project is

**New on 2026-10-17** (EXPERIMENTS.md, that date's NOTEs, LAUNCHED and DEVIATION):
- **`equi-arch-parts-1015` at 150k** (110k–150k pooled, of 250; init0 `oav5gk76` 83):
  - `arch-wide` (`h216n857`) **160**: better than init0 and **back to the base**, one episode short of
    ResFiT's level (161). Its 0.82 at 140k is the best equivariant eval after step 0.
  - `actorln` **104**, one episode under "better than init0" (105).
  - `criticln` 71 and `heads` 56: the critic parts add nothing.
  - One seed each.
- **`equi-actor-speed-1016` at 90k** (reported; main readout tonight). All three arms are recovering
  far faster than init0. At 40k–90k (of 300): `actorln448` 164, `actorlr1e-5` 132, `actorlr3e-5` 150;
  `arch-wide` 175, init0 59.
  - So `arch-wide`'s actor with init0's critic tracks `arch-wide`.
  - **A faster actor learning rate alone, with no network change, gets most of the gain.**
  - The "matches `arch-wide`" bar is **139 of 250**.
- **The enabling factor is how fast the actor can move its output early** (Claude's reading; one seed
  each).
  - The critic's action slope is the same in every arm (6.3–8.1e-5).
  - The early residual (0–3k) orders the arms as their recovery does: 0.002–0.006 fail; 0.012–0.014
    partly; 0.033–0.065 recover.
  - The calibration (2026-10-16 MEASUREMENT): one AdamW step moves the output by the rate times the
    summed size of the last layer's input. The norm raises that input's scale about 4 times, and the
    width its channels 3.5 times. Together they give about 12 times init0's step, as a 10-times rate
    does.
- **Speed is not all of it.** Both rate arms sit at the limit (0.100) on yaw and z from 3k to 10k.
  They lose the base at 10k and 20k as init0 does, and score 12 and 4 of 150 at 10k–30k, against
  `actorln448`'s 22 and `arch-wide`'s 39.
  - Every equivariant actor first pushes yaw one way. init0's stays there; the layout arms' correct
    it within about 3k steps; the rate arms' hit the limit, then reverse by 10k–20k.
  - `equi-speed-matched-1017` asks which part keeps the actor off the limit at matched speed.
- **Against ResFiT** (2026-10-17 NOTE): `arch-wide` sits inside ResFiT's seed range on every window,
  at its lower end late (0.640 against the anchors' 0.800 and 0.628). It does this with 6.8M
  parameters against 28.2M (its critic 2.8M against 23.6M).
  - Unlike ResFiT's, its actor's gradient falls over the run (0.038 to 0.012, against ResFiT's 0.06
    to 0.33–0.37).
  - Its value of the demos is lower (0.38 against 0.47).
- **Aden's decisions (2026-10-17):**
  - No seed 3 for any run for now: init0's and `actor448`'s set aside, `actorln`'s not run.
  - **Every run to its full 150k with no automatic stop of any kind,** the divergence stop included.
    Aden stops runs by hand. So `rfe-queue`, `parts-watch`, `speed-watch` and `long-watch` were
    stopped at 12:57.
  - Next is `equi-speed-matched-1017`: the 2×2 of norm and width at matched speed, plus everything
    stacked and init0 at lr 1e-4.

**The week before** (2026-10-13 to 10-16; EXPERIMENTS.md, those dates' NOTEs):
- **ResFiT's residual needs its actor's LayerNorm, not its critic's or its ViTs'.**
  - At two seeds each, 110k–150k: actor's off 0.144 and 0.264; critic's off 0.776 and 0.720; the
    ViTs' off 0.792 and 0.812; actor's and critic's off (`noln`) 0.032 at both; the anchors 0.714
    pooled.
  - Without it the residual is near its limit within 3k steps (0.07 of 0.1, against 0.03), its actor's
    gradient then falls (0.015 late), and its online value stays near zero.
  - The registered 10k–30k bar could not separate the critic's or the ViTs' arms. It sits inside seed
    spread; the late window separates them cleanly. That is why later groups read 110k–150k.
- **What the equivariant stack carries does not hurt ResFiT's residual:**
  - its action limits (0.800), noise (0.824) and rotation range (`fk9ayyco` 0.716);
  - the re-rendered demos (`bue9dupf` 0.828).
- **The equivariant baseline:** 0.272 and 0.216 at 110k–150k (seeds 2 and 3, `od8u86i1`,
  `dja2kbb6`), below the base (0.582).
  - init0 (`oav5gk76`) scored 0.332, ahead of the baseline by about its own seed spread. Not
    established.
  - `notilt` scored 0.200, so removing tilt did not help.

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
- **Run since** (2026-10-13 to 10-15): ResFiT's residual given the equivariant stack's differences
  one at a time, and its LayerNorms off one place at a time ("The week before", above); the
  equivariant residual's anchors to 150k. `notilt` ended at 0.200 at 110k–150k.

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

**Every run goes to its full 150k with no automatic stop of any kind** (Aden, 2026-10-17 DEVIATION,
which extends the 2026-10-12 AMENDMENT to the divergence stop). Aden stops a run early by hand if one
needs it. New groups go in the watcher's watch-only set.

**Four runs, two per GPU, none watched, nothing queued:**

| Run | Arm | Group | From | GPU | At 13:00 on 10-17 | 150k due |
|---|---|---|---|---|---|---|
| `vnpqtnva` | `actor448` 2 | `equi-arch-parts-1015` (wave 2) | `974e281` | 0 | 100k | 10-17 about 21:50 |
| `762bhig5` | `actorlr3e-5` 2 | `equi-actor-speed-1016` | `dd6f753` | 0 | 90k | 10-18 about 00:00 |
| `0jqo4jrx` | `actorlr1e-5` 2 | `equi-actor-speed-1016` | `dd6f753` | 1 | 90k | 10-18 about 00:05 |
| `x3xopn1y` | `actorln448` 2 | `equi-actor-speed-1016` | `dd6f753` | 1 | 90k | 10-18 about 00:30 |

- Each runs at 5,100–5,300 steps per hour and ends itself at 150k, completing its own run package.
- With no watcher, nothing FLAGs critic values above 1.5 or finalizes a package left incomplete by a
  crash (`finalize_run_package` in `resfit/lerobot/utils/load_policy.py`, by hand). The largest critic
  value so far is 0.47.
- Every run finished up to 2026-10-17, with its readouts, is in EXPERIMENTS.md. A ResFiT run took
  11–13 h from launch to 150k. An equivariant one takes about 30.5 h (128 fields) to 32.5 h (448)
  beside another equivariant run, about 1 h 20 min of that in the gate and demo labelling.

- **The queue:** `outputs/queue/launchers/retry/slot_queue.sh`, normally in tmux `rfe-queue`
  (stopped at 12:57 on 10-17; the plan's Phase 4 starts it again).
  - Start it with `bash slot_queue.sh <jobs file> 2>&1 | tee -a slot_queue.log` (the plan has the
    full command). Job lines are `<worktree> <arm> <seed> <smoke_first> [if=<script>]`.
  - Each job goes into the first slot to have been free for 300 s, two runs per GPU, and gets its own
    tmux session, `q-<arm>-<seed>`.
  - A launch its gate stops is tried again, up to 3 tries 600 s apart (2026-10-13 LAUNCHED).
  - Its log is `outputs/queue/launchers/slot_queue.log`. The scripts are in `outputs/queue/launchers/`,
    which is gitignored (TODO: move them into `tools/`).
  - **The queue reads its jobs file once, at start.** To change the jobs, stop `rfe-queue` while it is
    only waiting for a slot (its last log line is a launch, more than a minute old), rewrite the file
    and start it again.
  - A job with `if=<script>` runs that script (in `retry/`) once a slot is free, with `JOB_WORKTREE`,
    `JOB_ARM` and `JOB_SEED` set, and launches only on exit 0. `parts_seed3_due.sh`, which ran
    `tools/parts_seed3.py`, is unused now.
  - The gate prints nothing until each tier ends, so its console is blank for its first 40 minutes.
- **Watchers** (`tools/square_rerendered_watch.py --group <seed group>`, one tmux session per group,
  from the group's pinned worktree): none is running.
  - `equi-speed-matched-1017`'s will be `matched-watch`, watch-only. It logs evals, FLAGs and
    finalizes, and never signals.
  - The older groups' watchers enforced stops that no longer apply.
- **tmux:** the finished runs' `q-*` sessions and `sq-tilt-notilt` are idle shells; the old queue
  runners `gpu0`, `gpu1` and `gpu1-bars` are idle too.

- **What each run records beyond its evals and videos:**
  - `notilt` and `diag` log the residual and the critic's slope per action field
    (`debug/residual_field/*`, `debug/dQ_da_field/*`) and save `models/step_<N>.pt`. They launched
    before `534cc71`, so their eval Q overlays keep the old error and they log no `eval/tilt_*`; the
    field test, run from `534cc71`, has both.
  - `fk9ayyco` (from `60fce72`) has neither, as Aden chose: its evals and videos suffice.
- **Resources: four runs fill the machine's 125 GB of RAM.** A run is its trainer and 11 worker
  processes: about 21–23 GB at start (the workers 10.6 GB of it), and about 34 GB at 150k as the
  trainer's buffer grows (EXPERIMENTS.md, 2026-10-15 MEASUREMENT). In the last four-run overlap (10-14
  03:00–06:24) available memory sat at 0.3–0.7 GB and about 1 GB went to swap (225 GB), with no run
  lost and no slowdown. Four at 150k come to about 136 GB. At 10:40 on 2026-10-17, with four runs at
  80k–90k: 121 GB in use and 3 GB in swap. About 9.9 GB of GPU memory per run: two runs peak at
  21–23 GB of a 24.5 GB GPU.
  - **The disk had 23 GB free at 13:00 on 2026-10-17,** the wandb cache at 5.8 GB. It was cleared on
    2026-10-15 (21.7 GB; EXPERIMENTS.md, that date's NOTE on `vqzx1ir9`).
  - The cache keeps a copy of every best upload, including those wandb has since deleted, and it
    grew 18 GB in the two days before. Clear it again with `wandb artifact cache cleanup 2GB` when
    the disk falls below about 15 GB. Each run's current best is also in its run package's
    `wandb_best/`.
  - The old worktrees are about 4 MB each. Equivariant checkpoints are 87 MB each and ResFiT's
    223 MB.
- **How to follow them:** wandb `robomimic-square-final`, groups `residual-rl` (ResFiT) and
  `equi-residual-rl`, notes starting with the seed group. `outputs/queue/watch.log` has every
  watcher action, and consoles are in `outputs/queue/`.
  - Watch `eval/success_rate`, `debug/q_data_online` against `debug/q_data_demo`,
    `train/residual_l1_magnitude`, `train/actor_grad_norm`, `debug/q_advantage_ratio` and
    `debug/dQ_da_mean_abs` (ARCHITECTURE.md, "Reading the critic and actor diagnostics"), and the
    per-component residual (`debug/residual_field/*`): the pooled mean hides a component at its limit.
  - Read a live run's history with `_history_from_record` (`resfit/lerobot/utils/load_policy.py`) on
    its `run-<id>.wandb` record; the run package's `wandb_logs/` is written only when it ends.

## The machine

- **`residual`** (every run of ours): mujoco 3.3.2 from 2026-10-07 (3.8.1 before, in every
  earlier run), torch 2.6.0, torchrl 0.7.0, tensordict 0.7.0, escnn 1.0.11. Each run logs these
  as `versions` in its config.
- **`resfit-upstream`** (upstream's clone, `~/projects/resfit-upstream` at `66262b06`): its
  README's env, with torch 2.7.1+cu128 and torchvision 0.22.1+cu128 in place of the 2.14.1 its
  unpinned install gives, which cannot load torchcodec 0.4.0 or torchrl 0.9.2's extension. Freeze
  files beside the clone. Launcher: `~/projects/resfit-upstream-runs/launch.sh`.
- **RAM:** 125 GB, and 225 GB of swap. About 15 GB per Can tuned-RLPD run at 30k. Square runs:
  four at once fill it near 150k (Running now, "Resources"). The older count of 40.3 GiB per Square
  residual run, with two fitting and not a third, predates 2026-10-13, when four first ran at once.
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
- **What the equivariant residual needs, settled.** Not its action space, demos or noise, which
  ResFiT carries. Not its critic: the critic parts add nothing, and the action slope is the same in
  every arm. **Its actor has to move fast enough early:** `arch-wide`'s layout, `arch-wide`'s actor
  alone, or a 10–30 times actor rate (2026-10-17 NOTEs; one seed each).
  - **Open:** whether speed alone is enough at 110k–150k (tonight's readout); what the norm adds
    beyond speed (`equi-speed-matched-1017`); why every equivariant run still loses the base at 10k
    (0.00–0.04, against ResFiT's 0.16–0.40); and the late gap to ResFiT's best seed.
- **Whether training and execution should combine the residual alike** (TODO H11). The actor loss
  and the TD target clip each pair of base plus residual to the unit disk; the env adds them and
  robosuite clips each component. They differ only for pairs longer than 1.

## The plan, and why

**0. Now: the actor parts at matched speed** (`equi-speed-matched-1017`, pre-registered on
2026-10-17; [the plan](plans/2026-10-17-speed-matched.md), TODO B19). Every arm is an existing seed-2
run plus a learning rate:
- `actorln` and `actor448` at 3e-6 fill the 2×2 of norm and width at about `arch-wide`'s early speed.
  The other two cells are `actorlr1e-5` and `actorln448`.
- `actorln448` at 1e-5 is everything stacked.
- init0 at 1e-4 extends the plain actor's rate curve.

Why: the speed explains most of the gain, but the rate arms pin at the limit early and lose the base,
and the layout arms do not. The 2×2 asks whether the norm is what keeps the actor in range, and
whether the small normed actor plus a faster rate (0.3M parameters, against 3.4M) matches
`arch-wide`. That would be the cheap recipe. Running before it: `equi-actor-speed-1016`, whose
110k–150k says whether speed alone is enough (TODO B18).

**1. Read the residual and equivariant history again** once the plan reports: which conclusions
survive the render fix.

**2. BoxCleanup** (TODO B3, B4, B7) waits: a full run needs about 82 GiB of RAM and about 4 days.

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
