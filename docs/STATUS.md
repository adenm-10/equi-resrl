# Status

**Last updated:** 2026-10-30, 09:40 · **Last verified commit:** `adebb3c`, Stack's RL side (EXPERIMENTS.md,
2026-10-30 CODE). Its 33 Stack tests pass, the arms' build-and-update tests on the CPU, and so do the 312 Tier 1
tests of the arms and Stack; the full gate's GPU tier runs at Stack's first launch. `e7710f4` (O1) passed the
full gate at O1's first launch (1,078 GPU-tier tests). Where each group launches from:
- `obj-1027` from `e7710f4`;
- `long-1027` from `e2fdb27`;
- Stack's base policy from `3c23fe9`, and Stack's RL runs from `adebb3c` (its pre-registration's commit, to come).

Main is pushed to origin through `4a6ab43`; the commits since are local.

**Claude acts on its own recommendations** (Aden, 2026-10-28 morning; EXPERIMENTS.md, the 2026-10-28
DECISION): launches, gross-failure checks, the readouts and the doc budget, each call recorded with its
reason. Waiting for Aden: any change to a pre-registered design, and any deletion the DECISION does not
cover.
- **Stack (TODO B25) is being built** (approved 10-29; the 2026-10-29 DEVIATION). Its base policy trains
  (`8g47hqyc`); the headroom rule picked its step 2000 (0.67), measured next; its RL code is committed
  (`adebb3c`). Its runs are pre-registered once the base is measured.
- **O1's `rrr` arms run at 1e-6**, as pre-registered.

**The work is back in the VSCode session from about 08:45 on 10-29** (Aden: so they can watch it there).
tmux `claude-o1` is closed; its handoff is `outputs/claude-o1/handoff.md`. **`long-1027`'s seeds 5 are
cancelled** (the DEVIATION): each long arm has one run, and the queue holds O1's eight jobs.

**Cleaned up on 10-29 (Aden's go):**
- **Aden ran two deletions by hand:** `outputs/cleanup-20261028/delete_by_hand.sh` (2.5 GB, the seven
  directories the 2026-10-28 DECISION approved), and the three old Claude scratchpads in `/tmp/claude-1003/`
  (7.2 GB). `delete_by_hand_2.sh` (September's 14 wandb folders, 3.2 GB) is not run, and stays Aden's call.
- **The finished groups' worktrees are removed:** `equi-resrl-58d55d7` (enc-iface) and `equi-resrl-aec1bdc`
  (arch-swap). With them went the enc-iface watcher and the finished runs' idle tmux sessions; their
  scrollbacks are in `outputs/cleanup-20261029/scrollback/`.
- **The other worktrees go as their groups end:** `a5b5cc8` went at 09:35 on 10-30 with actor-design's watcher
  (its scrollback in `outputs/cleanup-20261030/scrollback/`); `e2fdb27` goes after `long-eee 4` (about 11:20
  on 10-30), `e7710f4` after O1, `3c23fe9` after Stack's base policy is measured and its package read.
- **`~/equi-resrl-preserve` stays.** It holds `z8yoqylh`'s only copy.
- **The 10-28 reclaim** (2.67 GB), and what it kept, is listed in `outputs/cleanup-20261028/`. 196 of the
  199 April–May wandb folders are the only copies, and stay.

**The disk:** 10 GB free at 08:55 on 10-30, after the rolling reclaim of actor-design's last two duplicate
best models (0.63 GB).
- tmux `wandb-cache` trims wandb's artifact cache to about 2 GB at 07:35 and 19:35, and each launch adds the
  1 GB BC policy.
- **What the queue still writes:** about 8 GB, net of the rolling reclaim. That is O1's eight runs and
  smokes (about 4.5 GB), `long-eee 4`'s last four checkpoints (1.8 GB), and packages and wandb folders.
- **Stack adds:** its demos (1.1 GB, downloaded), its dataset and its base policy.
- **The rules:** the queue launches only while 8 GB is free (`DISK_FLOOR_GB`). Each finished run's duplicate
  `models/best_model.pt` is deleted once byte-identical to `wandb_best/` (`deleted_best_models.txt`).
  Claude is woken below 9 GB.

**Now: O1 (`obj-1027`) runs, Stack's base policy trains and is measured, `long-eee 4` ends at 300k;
actor-design is read** (EXPERIMENTS.md, the 2026-10-27 PRE-REGISTERED entries, the 2026-10-29 DEVIATION, the
2026-10-30 entries). O1 is the nut's pose alone or with the peg's, on `eee` and `rrr`, eight runs: two run,
six queued. `long-1027` has one run per arm: `long-resfit 4` is done, `long-eee 4` ends about 11:20 on 10-30.
Stack, `eee` against `rrr` (TODO B25), is being built; latency is sidelined. One queue places
every job: tmux `order-queue`,
`outputs/queue/launchers/order-1027/order_queue.sh` (log `order_queue.log` there; "Running now", below).
- **`long-resfit 4` (`3ctaaap1`) is complete at 300k** (EXPERIMENTS.md, the 2026-10-28 and 2026-10-29
  NOTEs). Its pools by window: 0.848, 0.820, 0.876, 0.880. At 150k it matched `bue9dupf`'s 0.828, so
  ResFiT's re-rendered arm is 0.838 over two runs. It was not still climbing at 150k by the registered
  rule, levelled off by 210k–300k, never collapsed, and ended inside its prediction (0.85–0.90). Its
  critic's online value kept rising (0.21 to 0.36).
- **`long-eee 4` (`rkks1a9p`)**, the `eee` arm's one run: pooled 0.648 at 110k–150k, inside eee's four runs'
  range (mean 0.599), every signal comparable, so **eee has five runs at 150k, mean 0.609** (the 2026-10-29
  NOTE). From 160k: 0.80 0.80 0.76 0.70 0.78 0.82 0.68 0.72 0.76 0.72 0.74 0.84 0.76, level, not climbing. The
  `long-1027` NOTE after its 300k, about 11:20 on 10-30.
- **`arch-swap-1021` is complete and read** (EXPERIMENTS.md, the 2026-10-26 NOTE). Pooled at 110k–150k,
  two seeds: `ere` 0.692, `rrr` 0.648, `eee` 0.540, `rra` 0.402, `aaa` 0.388, `ara` 0.314 (the twelve
  step-0 evals, the base alone, 0.552). Counted at 0.13: C1 (`aaa` − `eee` −0.152) and C2 (`ara` − `ere`
  −0.378), the constraint; C4 (`ere` − `eee` +0.152), ResFiT's actor; C6 (`rrr` − `rra` +0.246), ResFiT's
  critic. Not counted: H (−0.066), C3 (−0.074), C5 (+0.088). No run with an unconstrained critic beat its
  step 0. **C4 against all four runs of eee's launch is +0.093** (their mean 0.599, range 0.516–0.676; the
  2026-10-27 NOTE), under the bar: reported, not registered. Aden's calls (TODO B20; the last 2026-10-26
  DECISION): the drift readout dropped; a slower dense critic planned after `actor-design-1026`, its place
  in the new order open (TODO B24, call 7).
- **`actor-design-1026` is complete and read** (EXPERIMENTS.md, the 2026-10-30 NOTE). eee's agent with its actor
  wired as ResFiT's (`actor-resfit-shape`, 0.584; `-wide`, 0.640) or unconstrained (`actor-eae`, 0.570),
  two seeds each, pooled at 110k–150k, against `arch-swap`'s `eee` (0.540) and `ere` (0.692). **None of D1–D5
  counts at 0.13** (+0.044, +0.056, +0.030, +0.108, +0.122), every pair at matched speed. By the registered
  reading, the actor's wiring, width and constraint are not the gap at this bar. Against eee's five runs
  (0.609), the three arms sit within 0.04 of eee and C4 (`ere` − `eee`) is +0.083. The anchor pair were eee's
  two lowest. Next by the registered reading: more seeds of `ere` (it has two), Aden's call (TODO B26).
- **`enc-iface-1023`** (EXPERIMENTS.md, 2026-10-23 PRE-REGISTERED; the 2026-10-24 and 10-26 DECISIONs;
  the 2026-10-26 LAUNCHED and NOTE; the 2026-10-27 NOTE).
  - Seed 2, provisional, 110k–150k: `iface-window` 0.236 (`od14pmu3`; never above its step 0; its critic
    leaned on the window, 1.55 times the summary late, and valued online data as the unconstrained
    critics did) and `iface-wide-out` 0.452 (`gcrewpq2`). The anchor `iface-eee` scored 0.676
    (`mgcifv1y`), against `eee 2`'s 0.516.
  - `iface-wide-all 2` (`5shh877a`) was stopped at about 113k (Aden), a description only.
  - **Complete as cut:** `iface-window 3` (`n94ucnlv`) reached 150k at 0.360, under its step 0 throughout. The
    arm's two seeds average 0.298, against eee's 0.599 (the 2026-10-28 NOTE). `iface-wide-out 3`,
    `iface-eee 3` and `iface-wide-all 3` were dropped, so only descriptions remain.
- **What goes with beating the base** (a 49-run analysis, 2026-10-27; EXPERIMENTS.md, the 2026-10-28 NOTE;
  `outputs/diagnostics/success-correlates/`).
  - Two necessary conditions: the early speed band (0 of 21 runs outside it beat the base) and a
    constrained or ResFiT critic (0 of 6 unconstrained).
  - The strongest early signals are the evals and the training success at 10k–30k (ρ +0.86 and +0.80).
  - eee's gap to ResFiT is the critic's value of online data: 0.005 early against ResFiT's 0.082.
  - All 17 runs above the base were still climbing at 150k, and so were 15 of the 32 below. eee's launch
    climbed at about a third of ResFiT's rate over 50k–150k (+0.08 against +0.27 per 100k).

The actor-speed line is closed on its readouts; what it leaves is in "What the actor-speed groups
leave", below.

This file describes what is true right now. Rewrite it in place; do not append. For history, see
[EXPERIMENTS.md](EXPERIMENTS.md) and [PROGRESS.md](PROGRESS.md).

## Where the project is

**The actor-speed groups** (2026-10-15 to 10-20; EXPERIMENTS.md, the 2026-10-20 LAUNCHED, NOTEs and
DECISION) are finished and read, and what they leave is in "What the actor-speed groups leave", below.
Of their 13 runs at seed 2, `arch-wide` (`h216n857`) scored best: 160 of 250 at 110k–150k (0.640),
ResFiT's seed-2 level, and the only one above the base (0.553) by more than noise. Every fast arm settled
by 50k–70k at or below the base. Aden's decision (2026-10-20): the next experiments are architectural, with
no follow-up of the speed line. Condensed on 2026-10-27; the 13-run table is in the 2026-10-20 NOTEs.

**The week before** (2026-10-13 to 10-16; EXPERIMENTS.md, those dates' NOTEs; condensed on 2026-10-27):
- **ResFiT's residual needs its actor's LayerNorm, not its critic's or its ViTs'.** With the actor's off it
  scored 0.144 and 0.264 at 110k–150k, against the anchors' 0.714, and its residual was near its limit
  within 3k steps. The registered 10k–30k bar sat inside seed spread, which is why later groups read
  110k–150k.
- **What the equivariant stack carries does not hurt ResFiT's residual:** its action limits, noise and
  rotation range, and the re-rendered demos (0.716–0.828).
- **The equivariant baseline** scored 0.272 and 0.216 (`od8u86i1`, `dja2kbb6`), below the base (0.582).
  init0 (`oav5gk76`) scored 0.332, which is not established.

**The equivariant residual on Square lost the base policy within 10k steps in every run of 2026-10-12, and
the loss looked like a wrist that rolls after the grasp** (EXPERIMENTS.md, 2026-10-12 FINDING and NOTEs;
condensed on 2026-10-27).
- Each run held the base at step 0 (0.50–0.62), fell to 0–0.02 by 10k, then partly recovered.
- No single field of the residual carries it. In the field test, zeroing any one field leaves 0–1
  successes of 100, against 57 with the whole residual zeroed. `notilt` ended at 0.200 at 110k–150k, so
  tilt is not the cause.
- Ruled out: the 1e-4 start (`eaqorqki`) and a better-informed critic (`doevaarp`).
- A penalty on the residual's size is held, since it would hold ResFiT's useful residual down too.
- The 10k dip is in every arm since (0.00–0.24 in arch-swap), but arch-swap's `eee` and `ere` now end
  above the base.

**Every other experiment was cancelled on 2026-10-12** (DEVIATION; condensed on 2026-10-28, and the
2026-10-10 to 10-12 entries hold the detail):
- **Equivariant tuned RLPD learned Square** (`vc5o977f`: 0.50 at 80k), the first tuned RLPD to do so in
  our trainer. **Then its critic ran away** from about 92k, and every eval from 100k was 0.
- `square-trlpd-stab-1010` keeps one seed per arm and no verdict. No critic in it diverged.
- **The fixed-env baselines' ResFiT residual arm passes** (`oj5k0jqb` 0.800, `s8opblsd` 0.628 at
  110k–150k, against B_sq's 0.582). Its two seeds are about 4 SE apart.

**The equivariant critic with `critic_norm=field_norm` cannot see the wrist camera, the arm's height or the
gripper opening, nor score the z, yaw or gripper actions** (EXPERIMENTS.md, 2026-10-08 FINDING). `FieldNorm`
erases every trivial-type input, which accounts for both equivariant arms' failure in the matching render.

On Can, tuned RLPD with `critic_norm=none` learned only with the env's frames passed through the demos'
video codec (`gkck6eqp`: 0.94 at 10k, one seed). Without the codec (`xm9cuo92`) it scored 0 at every eval
to 30k. The 2026-10-08 and 10-09 NOTEs hold the rest; this was condensed on 2026-10-28.

**The working account:** with no reward online, a critic's online value is held up only by what it carries
over from similar demo states. ResFiT's `d047it6c` carried over about 0.3 for 13.6k steps before its first
success; the failing critics carried over nothing.

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

**Our conda env rendered the scene differently from the demos until 2026-10-07; from then the env is pinned
to mujoco 3.3.2, whose render matches them, and the gate checks it** (EXPERIMENTS.md, 2026-10-07 FINDING,
RESULTS and CODE, "mujoco 3.3.2"; condensed on 2026-10-27, the detail is there).
- **The mismatch:** mujoco 3.8.1 (in `residual` from 2026-05-21 to 10-07) renders robosuite's floor as dark
  planks; the demos and 3.3.2 show light grey tiles. Every pixel off the floor is identical.
- **Its size** (`tools/render_check.py`: mean absolute pixel difference between 40 demo first frames and 40
  env reset frames, front camera; the bar is 2.5 times the even-odd demo gap): Can 0.0977 under 3.8.1 and
  0.0103 under 3.3.2 (bar 0.0125); Square 0.0166 and 0.0086 (bar 0.0130). The wrist camera matches on both.
- **What it did:** the critic could tell every demo frame from every online frame by the floor, which fits
  tuned RLPD's failure signature (an action-blind critic, online Q at exactly 0). Upstream's tuned RLPD,
  unmodified, learns Can in upstream's env (`zwmk393l`: 0.88 at 20k) and not in ours (`rdoj0ex8`). Not yet
  separated from upstream's newer torch, torchrl and tensordict; only mujoco changes what the agent sees.
- **Every run before 10-07 saw it,** strongly on Can, mildly on Square; tuned-RLPD results of 10-04 to 10-07
  say nothing about their change in a matching env.
- **Aden's plan of 10-07 (TODO B15)** followed: worktrees into main, the pin on the gate and the record, the
  base rates again, and the fixed-env baselines (2026-10-07 PRE-REGISTERED; ARCHITECTURE.md, "The two
  stacks side by side").

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

## What the actor-speed groups leave

For planning the architectural experiments. Every number is from EXPERIMENTS.md's 2026-10-15 to
2026-10-20 entries, at one seed per arm.
- **The best equivariant recipe so far: `arch-wide`** (`h216n857`, `submit.sh`'s
  `equi-residual-square-arch-wide`). It has RegularLayerNorm after every hidden Linear of the actor
  and the critic, 10 critic heads of 2 hidden layers, and the actor at 448 fields.
  - It scored 160 of 250 at 110k–150k, ResFiT's seed-2 level.
  - It has 6.8M parameters (actor 3.364M, critic 2.8M, encoder 0.627M), against ResFiT's 28.2M
    (critic 23.6M).
  - Its actor alone, with init0's critic (`actorln448`, 4.8M), scored 128. The 128-field normed actor
    at a 3e-6 rate (`actorln-lr3e-6`, 1.8M) scored 124.
- **The actor's norm is the part that matters.** At matched early speed it adds +0.12 at 10k–30k and
  +0.09 at 110k–150k, and keeps the residual off its limit early. The width adds speed only. The
  critic's parts add nothing alone (`criticln` 71, `heads` 56). With `arch-wide`'s actor they may add
  something: 160 against 128, about 2 standard errors at the measured noise.
- **Early speed sets the level.** One actor step moves the residual by the rate times the summed size
  of the actor's last layer's input (2026-10-16 MEASUREMENT; extended in the 2026-10-17
  PRE-REGISTERED entry). So any new actor or norm changes it.
  - The calibration script is `outputs/diagnostics/actor-step-size-1017/calibrate.py` (CPU, about
    5 min). It took the real runs' early speed to within 1.2–2.8 times.
  - The best results came at a 0–3k residual of 0.03–0.04 (`train/residual_l1_magnitude`), which is
    also ResFiT's own (0.029–0.030).
  - Faster than about 0.045 settles lower. Slower than about 0.015 is still rising at 150k.
- **Every equivariant arm still loses the base by 10k** (0.00–0.22, against ResFiT's anchors' 0.30 and
  0.22). Every fast one then settles at or below the base by 50k–70k. Both are open.
- **The residual's limit:** late in training, every arm's xy and tilt corrections are near their 0.1
  limit. Read the per-component `debug/residual_field/*`: the pooled `train/residual_l1_magnitude`
  hides a component at its limit.
- **Reading an eval:**
  - A 50-episode eval of a fixed policy swings by about ±0.15–0.20 (SD about 0.10, 1.4 times the
    binomial one).
  - The base alone pools to 0.553 over 1,250 episodes (B_sq's 500 and 15 step-0 evals).
  - The binomial two-standard-error bars used so far are about 1.4 times too narrow. Read 250-episode
    windows, never one eval.
  - Training-episode success per 10k (`training/episode_return` > 0) is a cross-check.
- **Saved policies:** each run package's `wandb_best/` holds its best eval's checkpoint (`best_model.pt`,
  and `best.json` with its step). These groups saved no `models/step_*.pt`, and
  `tools/residual_field_ablation.py` loads only those.
- **The freeze note.** From `0b14c17` to the families' code commit no file under `resfit/` changed,
  so every equivariant run from `oav5gk76` (2026-10-13) to `xsordrhh` ran the same agent and trainer
  code. **The freeze ends at the families' commit** (2026-10-20 CODE).
  - The equivariant agent is still bit for bit as before: `tests/test_regression_agent.py`'s goldens
    and the deep check of all 19 `CANDIDATES` against `6517d0c` are unchanged.
  - Even so, runs before it compare with runs after it only through an anchor run again at the new
    commit. In `arch-swap-1021` that is `arch-swap-eee`, `arch-wide`'s launch.
  - The live switches the groups used: `agent.actor_lr`, `equivariance.actor_norm`,
    `equivariance.critic_norm`, `equivariance.actor_degree_channel`, `agent.critic.num_q` and
    `equivariance.num_critic_layers`.
  - Two config fields are not what they look like: `equivariance.num_actor_layers` is dead (the actor
    has one policy layer whatever its value), and `equivariance.initialize` reaches the encoder only
    (ARCHITECTURE.md's provenance table).
- **Cost:** an equivariant run takes 29–32 h to 150k beside another, two per GPU.
  - About 1 h 20 min of that is the gate and demo labelling. The gate's Tier 2 alone is about 60 min,
    678 tests.
  - Four runs at once fill the RAM: about 136 GB at 150k, against 125 GB plus swap.

## Running now

**At 09:40 on 2026-10-30** (the 2026-10-29 and 10-30 LAUNCHED entries):

| GPU | Running | Ends |
|---|---|---|
| 0 | `obj-nut-eee 2` (`ifsnjb9f`, 0.74 at 100k) and `obj-peg-eee 2` (`3vimywbf`, from 06:54; early speed 0.0383, predicted 0.0359) | 150k about 20:00 on 10-30; about 13:00 on 10-31 |
| 1 | `long-eee 4` (`rkks1a9p`, 0.76 at 280k) and `stack-base 1` (`8g47hqyc`, Stack's base policy, from 00:12; 0.91 at 19k) | 300k about 11:20 on 10-30; 50k about 22:00 on 10-30 |

`actor-eae 3` (`xymahaba`) and `actor-resfit-shape-wide 3` (`jfdajea9`) reached 150k at 00:07 and 00:46 on 10-30,
packages complete; actor-design is read.

**Next on GPU 1, when `long-eee 4` exits (about 11:20):** B_stack, Stack's base measured at the headroom rule's
pick (`run_8g47hqyc_model_step_2000:v0`, 500 episodes, about 30 min), from tmux `stack-bstack`
(`outputs/queue/launchers/stack/measure_base.sh`, waiting on `long-eee 4`'s trainer; log
`outputs/diagnostics/stack/bstack.log`). The queue counts it as a slot, so `obj-nut-rrr`'s smoke follows it.

**Queued in `order-1027`**, restarted at 08:55 on 10-30: `obj-nut-rrr 2` (back after its smoke's chance FAIL;
the 2026-10-30 DEVIATION: smokes now evaluate 8 episodes), `obj-peg-rrr 2`, then the four at seed 3. Stack's
runs join once pre-registered.

How the queue works now:
- `JOBS` holds only jobs not yet launched. A restart skips a job only while its `q-*` session exists, and
  finished runs' idle sessions are closed by cleanups. At 09:54 the old list would have relaunched
  `long-resfit 4` and `actor-resfit-shape 3`; `--check` caught it.
- `held()` counts a `train_bc_dexmg` or `eval_base_policy` process as a slot.
- Earlier copies of the script are kept beside it: `order_queue.sh.before-20261029`, `.before-stack`,
  `.before-20261030` and `.before-evalbase`.
- The queue's start line still reads `AD_WORKTREE` (the removed `a5b5cc8`); drop it at the next restart.
- `e2fdb27` passed the full gate's Tier 2 at both long launches (970 tests). O1's smokes are in
  `outputs/diagnostics/obj/smokes/`.

**Every run goes to its full 150k with no automatic stop of any kind** (Aden, 2026-10-17 DEVIATION,
which extends the 2026-10-12 AMENDMENT to the divergence stop). Aden stops a run early by hand if one
needs it. New groups go in the watcher's watch-only set.

How the runs are launched and followed:
- **The queue:** `outputs/queue/launchers/order-1027/order_queue.sh`, in tmux `order-queue`, from 10-27 with
  `AD_WORKTREE=/home/aden/projects/equi-resrl-a5b5cc8` and `LONG_WORKTREE` the long runs' pinned worktree;
  restarted 13:19 with the long runs first, 16:28 with O1's `eee` jobs, 09:42 on 10-28 with its `rrr` jobs (`OBJ_WORKTREE` `equi-resrl-e7710f4`), and 15:28 with an 8 GB free-disk floor (`DISK_FLOOR_GB`). It is `order-1026`'s queue with the long runs added, at most two at once
  (`LONG_MAX`), and a memory floor: a job launches only while MemAvailable, less 23 GB for each launch in the
  last 3 h, is 28 GB or more. A `tools.latency_eval` process holds a slot as a trainer does. Its log is
  `order_queue.log` beside it, also appended to `../slot_queue.log`. `--check` prints each GPU's occupants,
  where each job would fit, the memory and the long count.
  - At each slot free for 300 s (two per GPU), the first job in its list that fits that GPU's memory
    launches there through `retry/run_job.sh`, in tmux `q-<arm>-<seed>`. The run's gate goes first; a
    launch its gate stops is tried again, up to 3 tries 600 s apart.
  - The fit: a CNN run holds about 11 GB, a ViT run about 9.1 GB, against 23.5 GB for a pair; nothing
    but a ViT run goes beside `iface-wide-all` (the 2026-10-26 LAUNCHED entry).
  - A new arm's first job runs the arm's integration smoke in its slot first (its group's
    `launch_smoke.sh` and `check_smoke.py`, under `outputs/diagnostics/<group>/smokes/`). A failed smoke
    drops the arm, and the log says so.
  - **Its jobs are in the script (`JOBS`), read once at start.** To change them, stop it while it is only
    waiting, edit, and start it again, as `order-1024` became `order-1026` on 10-26.
  - It is `order-1024/order_queue.sh` but for its jobs, which was `retry/slot_queue.sh` (tmux `rfe-queue`,
    stopped on 10-24) but for the fit; all three count a slot with the same `held()`.
  - **The gate skips Tier 2 on a commit that already passed it** (Aden, 2026-10-26 DECISION, afternoon).
    `run_job.sh` sets `PYTHON` to `retry/gate_python.sh`, which adds `--skip-tier2` to `preflight.py` for
    a commit listed in `retry/tier2_passed.txt` (`58d55d7`, `a5b5cc8`), and is plain python otherwise.
    Tiers 1 and 3 still run. A new commit's first launch runs the full gate, and from 10-27 a full gate that
    passes, on a clean tree with CUDA visible, adds its commit to the list itself (TODO B24, call 6; the
    old wrapper is `gate_python.sh.before-20261027`).
  - The gate prints nothing until each tier ends, so its console is blank for its first minutes, or
    about two hours with Tier 2 (953 tests, 124 minutes at `a5b5cc8`).
  - The render-guard test (`test_every_guard_passes_on_cans_first_demos`) still fails now and then in
    a gate that starts within minutes of a run exiting; the retry has passed it every time. Its failing
    assertion is still not recorded.
  - The scripts are in `outputs/queue/launchers/`, which is gitignored (TODO: move them into `tools/`).
- **The digest** (tmux `queue-digest`; `order-1027/digest.py` every 30 minutes, by `run_digest.sh`). It
  rewrites `outputs/queue/launchers/order-1027/digest.md`: each long run's four windows and checkpoints,
  MemAvailable, swap and each trainer's resident memory, and each run's state, step and evals, its early
  speed against the band and its arm's prediction, the gross-failure check of `actor-resfit-shape 2`
  (a FLAG line in `watch.log` once, if every eval at 10k–50k is at or below 0.10), the queue's smokes,
  launches and drops, every FLAG, and the GPUs, RAM and disk. It changes nothing about any run, and stops
  itself when the queue is done. **Read it first when a session opens.**
- **Watchers** (`tools/square_rerendered_watch.py --group <seed group>`, one tmux session per group, from
  the group's pinned worktree), each watch-only: logged, FLAGged and finalized, never signalled.
  - `long-watch` (`long-1027`, from its pinned worktree);
  - `obj-watch` (`obj-1027`, from `equi-resrl-e7710f4`);
  - `swap-watch`, `iface-watch` and `ad-watch` are stopped (on 10-26, 10-29 and 10-30), with all their groups'
    runs exited, and their worktrees removed. Stack's base policy has no watcher; its console is
    `outputs/queue/*_stack-base-1_gpu1.log`.
  - The older groups' entries in the watcher still enforce stops that no longer apply.
- **tmux, at 09:40 on 10-30:**
  - `order-queue` and `queue-digest`;
  - `q-<arm>-<seed>`, one per running job (finished runs' idle sessions are closed);
  - `stack-bstack`, waiting to measure Stack's base;
  - the watchers `long-watch` and `obj-watch`;
  - `wandb-cache` (the 12-hourly cache cleanup) and `disk-watch` (`/dev/shm/disk_watch.sh`: the free space
    every 20 s; it logs, and wakes a Claude session below 8 GB when one is open).

  Earlier cleanups' scrollback: `outputs/cleanup-20261024/`, `-20261026/tmux_scrollback/`, `-20261027/` and
  `-20261029/scrollback/` and `-20261030/scrollback/`.
  The queue checks a job's `q-*` session only on a restart.

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
  - **The disk had 15.8 GB free on 2026-10-28** (at the top). The wandb cache is cleared to 2 GB every
    12 h by tmux `wandb-cache` (`outputs/queue/launchers/wandb_cache_cleanup.sh`).
  - The cache keeps a copy of every best upload, including those wandb has since deleted, and it
    grew 18 GB in the two days before. Clear it again with `wandb artifact cache cleanup 2GB` when
    the disk falls below about 15 GB. Each run's current best is also in its run package's
    `wandb_best/`.
  - A worktree is about 4 MB. `eee`'s checkpoints are 444 MB each (escnn's expanded weight matrices,
    saved in eval mode; the 2026-10-28 LAUNCHED entry) and ResFiT's 223 MB. Each
    run's eval videos come to about 110 MB (TODO H13).
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
  for the vanilla agent at `actor_norm=none`, nor for the five mixes of network families
  `arch-swap-1021` runs. `iid_bn` still syncs (TODO H8).
- **The BoxCleanup base policy is at about the paper's bar, not above it:** ACT, final 0.53; its
  `best` checkpoint `run_xg0w2r0z_best:v7` comes from a stretch averaging 0.61. Backed up.

## What is not

- **Whether mujoco alone accounts for upstream's result,** rather than torch, torchrl or
  tensordict (B15's validation run).
- **What the render mismatch did to the residual and equivariant runs,** and the base rates under
  the correct render.
- **Why the executed residual grows.** It grew to the clip in the image-fix pair, unbounded, while
  the critic rated it above the base action (TODO B10). B15's equivariant residual arm bounds it.
- **The base rate of Square is uncertain.** The base-rate script gave 0.508 ± 0.022 in the mismatched
  env, and B_sq 0.582 ± 0.022 in the matching one. The 15 step-0 evals of the last runs pool to 0.533
  (400 of 750), and with B_sq's to 0.553. A single 50-episode eval of the base spreads 1.4 times
  wider than binomial (0.34–0.68; 2026-10-20 NOTE). No systematic cause was found.
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
- **Whether the weight constraint itself costs anything.** The control is now built: the ablated
  family (from 2026-10-20) is the equivariant network unconstrained, starting as the same function
  (STANDARDS 3.3). `arch-swap-1021` runs it; nothing has run yet. The scalar ablation it replaces
  never ran and is gone. **The gate cannot validate the
  commit it launches** (TODO R14).
- **An equivariant run's random stream depends on its eval schedule:** the actor's `std=0` path,
  taken by every eval and actor loss, draws a random number it scales by 0 (ARCHITECTURE.md,
  "Known rough edges"). Left as it is, for comparability.
- **What the equivariant residual needs, settled.** Not its action space, demos or noise, which
  ResFiT carries. Not its critic's parts alone. **Its actor has to move fast enough early, and stay
  off its limit:** an early residual near 0.03–0.04 with the actor's norm (2026-10-20 NOTEs; one seed
  each).
  - **Open:**
    - why every equivariant run still loses the base by 10k (0.00–0.22, against ResFiT's anchors'
      0.22–0.30);
    - what holds every fast arm at or below the base from 50k–70k: the residual's range (xy and tilt
      at their limit late) or seed variation (ResFiT's seed 2 settles where `arch-wide` does);
    - whether `arch-wide`'s critic parts help in combination (160 against `actorln448`'s 128, about 2
      standard errors at the measured noise).

## The plan, and why

**0. Now: `actor-design-1026`** (TODO B22), read about the evening of 10-30: whether ResFiT's actor beats
ours (C4) through its wiring, its free parameters, or its lack of a constraint.

**1. Next, in Aden's order** (the 2026-10-27 DECISIONs; TODO B24 has the to-do list and the calls):
- **The long runs** (2026-10-27 PRE-REGISTERED): ResFiT's agent and `eee` to 300k, to see where each
  levels off. Seeds 5 were cancelled on 10-29 (Aden), so one run per arm; `long-eee 4` ends about 10-30.
- **O1** (TODO B23, B24 call 5; the 2026-10-27 CODE and PRE-REGISTERED entries): the nut alone or
  with the peg, on `eee` and `rrr`, eight runs, all queued, from 10-29. It targets the critic's low value
  of online data.
- **The new task, Stack, `eee` against `rrr`** (TODO B25; approved and being built from 10-29). Square's
  symmetry under our group is extrinsic (2026-10-27 FINDING). In Stack every cube-to-cube direction occurs.
  Turned Square is dropped. Built: the env, the 1,000 demos in our env, the RL config and arms (`adebb3c`;
  both arms' early speed calibrated in the band). Training: the base policy. Next: B_stack, the runs'
  pre-registration, their smokes.
- **Then latency (sidelined) and the tuning.**

**2. BoxCleanup** (TODO B3, B4, B7) waits: a full run needs about 82 GiB of RAM and about 4 days.

**3. Read the residual and equivariant history again:** which conclusions survive the render fix.

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
- **Worktrees hold no records.** The 32 records of the 15 early pinned worktrees, and upstream's
  clone's 2, were moved into `wandb/` and `outputs/` on 2026-10-07 (EXPERIMENTS.md, CODE); every run
  since records into main. The 19 unused worktrees were removed on 2026-10-24 (TODO B15); `58d55d7`
  and `aec1bdc` stay while their groups run.
- `outputs/cleanup-20261024/`: the lists of what the 2026-10-24 cleanup deleted, and the killed tmux
  sessions' scrollback.
- `outputs/diagnostics/`: the render figures and the scripts that made them, cited by the FINDING,
  and the frames and repeats behind the gate's bar (`render_frames_mujoco{332,381}.npz`,
  `render_check_mujoco332_repeats10.jsonl`), cited by the 2026-10-07 CODE entry.
- `outputs/env/`: `pip freeze` of `residual` before and after the mujoco pin.

Three runs have already been lost to deletions.
