# TODO

Priority order. Each item names the doc section it should update when finished.

Tags: `[SC]` scientific consistency · `[CS]` code simplicity and correctness ·
`[HI]` human interpretability. See the priority ranking in [CLAUDE.md](../CLAUDE.md).

---

## B — Square and the bimanual task

**Now (10-31, 13:55): the wrist sweep (B29) is planned and waits on Aden, who asked for the queue to be upended
for it (the 10-31 DECISION), not yet applied.** O1 (B23, B24) has three of eight done (`obj-peg-eee 2` 0.648, `obj-peg-rrr 2` 0.872);
Stack (B25) three of four (`stack-1030`; `stack-rrr 2` relaunched at 13:18). The queue's disk floor is 6 GB from
13:13 (the 10-31 DECISION after it; STATUS, the disk). The long runs
(B24) and the actor's design (B22) are done and read. B26, B27, B28 and B30 wait on Aden.

### B29. The wrist sweep (`wrist-1031`): `eee` without the front camera, and a residual in the gripper's frame — PLANNED 2026-10-31, WAITING ON ADEN `[SC]`

**The plan:** `docs/plans/2026-10-31-wrist-sweep.md` (local, DRAFT): the seven arms, the build in eight steps
with file and line targets, the queue, the timeline and the readouts. **Why:** the 10-31 FINDING (Stack is
extrinsic about the base; the front camera is centred on the table middle; seen from the gripper, Stack's layout
is nearly closed) and the two papers read on 10-31 (the 10-31 DECISION).
- **Aden's calls so far:** NA-eee in; NA-rrr and NA-resfit cut; the G family only (the constrained,
  de-rotated-wrist design dropped); "upend all currently queued experiments".
- **Waiting on Aden (the plan's section 8):** the disk deletions; `stack-rrr 2` at the head and the backfill;
  the G arms with or without gripper xy; the heading frame or the full gripper frame; the pretrained encoders
  frozen or fine-tuned; the encoder widths; the wave order.
- **Then:** the build (one commit, one full gate), the pre-registration, and a new queue `order-1031`.

### B30. What costs ResFiT's networks 0.19 in our agent — NOT PLANNED, WAITING ON ADEN `[SC]`

The 10-31 NOTE: ResFiT's own agent 0.838 against `rrr` 0.648 on the re-rendered demos (post hoc), against `eee`
− `rrr` of −0.039. The action space was ruled out on 10-15; the input normalization is untested. The test: `rrr`
with ResFiT's input handling (a z-scored raw state, a per-axis rescaled base action), one switch on the ResFiT
family, about 10 h a run. If it closes the gap, part of the cost of equivariance is the normalization it forces.

### B24. The next phase, in Aden's order: the long runs, then O1, then a new task; latency and the tuning after — the long runs first in the queue 2026-10-27; O1 planned (Aden, 2026-10-27) `[SC]`

The order is Aden's (EXPERIMENTS.md, the 2026-10-27 DECISIONs, morning and afternoon; Claude's
recommendations against the morning one on record). [The plan](plans/2026-10-27-long-and-tuning.md)
(local) has the long and tuning groups' designs, [the steps plan](plans/2026-10-27-long-O-L-steps.md)
(local) O1's.

**What it rests on:** a 49-run analysis (`outputs/diagnostics/success-correlates/analyze.py`, its table
`runs.json`; EXPERIMENTS.md, the 2026-10-28 NOTE). Every one of the 17 runs above the base was still
climbing at 150k, which is the case for the long runs; so were 15 of the 32 below.

**Aden's calls** (2026-10-27):
1–3. **Decided and done:** the core pair, `long-resfit` and `long-eee` (ResFiT's own agent and
   `arch-swap-eee` run on), to 300k at seeds 4 and 5 with checkpoints every 50k, two at once (the CODE and
   PRE-REGISTERED entries). Since the afternoon they are first in the queue `order-1027`, actor-design's
   seeds 3 behind them; the iface pair is dropped (the afternoon DECISION). Seeds 4 run from 10-28 (`rkks1a9p`, `3ctaaap1`).
4. **DONE:** the duplicate best-model reclaim (6.9 GB).
5. **O1: APPROVED as a 2×2** (the 2026-10-27 evening DECISION): the nut alone or the nut and the peg, on
   `eee` and `rrr`, seeds 2 and 3, 150k, after actor-design's seeds 3. Each object takes the object-centric
   paper's shape (7 numbers: position and quaternion). The steps plan's calls 2a–2c are as recommended.
   Built on the evening of 10-27 (`6e74a1e`, pre-registered at `e7710f4`).
6. **DONE:** the gate adds a commit to `retry/tier2_passed.txt` itself once a full gate passes.
7. **The slower dense critic (B20):** with the tuning screens (Claude's recommendation, not objected to).
8. **DONE 2026-10-28:** the 49-run analysis as an EXPERIMENTS.md NOTE. It corrects 16 to 17 runs above
   the base. Each readout from actor-design's on checks the two conditions against the runs since.
9. **Latency (L) is sidelined** until the long runs, O1 and a new task have results; B23 keeps its design.
10. **The new task: Stack, `eee` against `rrr` (B25), APPROVED 10-29 and being built.** Turned Square is
    dropped (Aden).

**The to-do** (coarse; the days are targets):
1. **The long runs: DONE 10-30** (seeds 5 CANCELLED 10-29, Aden; the DEVIATION), one run per arm.
   `long-resfit 4` levelled at 0.88; `long-eee 4` was still climbing at 150k, then held at 0.74–0.80 and
   ended at 0.800; the gap to ResFiT narrowed from 0.200 to 0.080 (the 2026-10-30 `long-1027` NOTE). Not
   done: the 200-episode re-evaluations of the checkpoints (the secondary readout); B27.
2. **O1: BUILT 10-27** (the CODE entry). The rest:
   - all eight pre-registered and queued; the `rrr` arms at 1e-6 (the 2026-10-28 DECISION).

   The steps, as planned (the steps plan's section 2):
   - the pose through the wrapper;
   - the demos' column from robomimic's states;
   - its field types and tests: with the switch off, everything is today's bit for bit, and a turned raw
     observation gives the declared representation;
   - the four arms' early speeds;
   - the smokes, the pre-registration, and the eight jobs in the queue.

   Committed and queued about 10-28; the first launch about 10-30.
3. **While they run.**
   - `tools/residual_field_ablation.py` for ResFiT's agent, for the long runs' 200-episode
     re-evaluations; it reads the equivariant agent only.
   - The readout scripts: each long run's curve by window, and O1 against `eee` and `rrr`.
   - B22's readout: DONE 10-30, nothing counted; the tuning anchor stays `eee`.

**The results, roughly:** ResFiT's long runs about 10-29 midday and 10-31 night; `eee`'s about 10-30
afternoon and 11-01; O1 about 11-02.

**Then, after these results:**
- **A new task** (call 10): Stack, `eee` against `rrr` (B25), once Aden approves it.
- **The tuning group** (the plan's section 3).
  - Stage 1: 50k screens at seed 2, each one change from the anchor. They are scored on the online value
    at 20k–50k, with the speed band as a gate. The arms: critic rate 3e-4 and 3e-5, updates per step 8 and
    2, demo fraction 0.25, n-step 5, τ 0.01, critic warm-up 20k.
  - Stage 2: the best 2–3 at 150k, seeds 2–4.

### B25. Stack, `eee` against `rrr`: the new task, planned 2026-10-27, APPROVED 2026-10-29 (Aden: "asap"), PRE-REGISTERED and queued 2026-10-30 `[SC]`

**Progress (10-30, 22:05):**
- **Steps 1–3 are done:** the demos replay (954 of 1,000 action replays succeed); `Stack` is in the env
  (`d2d4f4c`); the dataset `local/mimicgen-stack-d0-rerendered` (`ab15ad9`). See the 2026-10-29 CODE entries.
- **Step 4 is done:** the base (`8g47hqyc`) trained from 00:12 to 22:01 on 10-30 (data loading made it about
  22 h, not 9–12; 0.92 at 50k, 0.99 at best; the NOTE). It reached 0.91–0.96 by 9k–16k, so the headroom rule picked its step 2000 (0.67), uploaded
  again as `run_8g47hqyc_model_step_2000:v0` (the 2026-10-30 NOTE). **B_stack = 0.602** (500 episodes, standard
  error 0.022, 11:18–11:33 on 10-30), inside 0.30–0.85, so step 2000 stays.
- **Steps 6–7 are done** (`adebb3c`; the 2026-10-30 CODE entry): the config, the gate's entry, the robot-base
  probe (Square's), the arms and their tests, the early-speed calibration (`stack-eee` 0.0381, `stack-rrr`
  0.0310, both in the band).
- **Step 8 is done:** the four runs pre-registered (the 2026-10-30 PRE-REGISTERED entry, `stack-1030`: S1 =
  `stack-eee` − `stack-rrr` at 0.13; S2 against Square's −0.108, described). The queue, its smoke launcher
  and checker, the watcher (`stack-watch`) and the digest were set up from `9944eeb`, restarted at 09:21.
- **Step 9, the runs:** `stack-eee 2` (`tjl95db5`, 112k at 10:15 on 10-31) ends about 15:30; `stack-eee 3`
  (`3l3j5z3v`, from 23:05) about 08:00–10:00 on 11-01. `stack-rrr 2` (`cs4ezjj2`) died at 30k when the disk
  filled (the 10-31 FAILURE) and runs again from about 12:40, then `stack-rrr 3`. The readout about 10:00 on 11-01.
- **Removed** at 11:40: `equi-resrl-adebb3c` (B_stack's worktree) and tmux `stack-bstack` (its scrollback in
  `outputs/cleanup-20261030/scrollback/`). At 22:03: `equi-resrl-3c23fe9` (the base policy's) and tmux
  `q-stack-base-1`, its run folder moved to the main checkout first (its deletion: STATUS, Aden's call).
- **Step 5 (the symmetry record)** is not done; it needs no slot.
- **Found:** `train_bc_dexmg.py` crashes without `--wandb_enable`. Its `if wandb is not None` tests the
  module, which is never None. Every real run enables wandb. And its `upload_best` keeps only the latest
  best in wandb, so a rule that picks an earlier best reads the run's local `best_step_<step>/`.

The plan of record for the new task (the 2026-10-27 evening DECISION, item 4). Stack is robosuite's own
task, MimicGen's Stack D0: two cubes within ±8 cm of the table's middle, at any yaw. Every cube-to-cube
direction occurs, where Square's nut-to-peg direction spans 18.5°. ResFiT reported no Stack result, so
every arm is trained here. The arms are `eee` and `rrr`, as in O1; ResFiT's own agent is an optional
reference.

1. **Demos** (half a day, CPU):
   - download MimicGen's `core/stack_d0.hdf5` (Hugging Face `amandlek/mimicgen_datasets`; 1,000 demos,
     recorded under robosuite 1.4.1);
   - check that they replay under our 1.5.1: set each state by joint name, then check the cubes' placement
     and that each demo's last state counts as a success;
   - if they don't replay: map qpos by name (as the turned-demo replays did), or script grasp-and-place
     demos.
2. **Env** (a few hours): add `Stack` to `dexmg.py` (`ENV_ROBOTS`, the horizon, the single-arm camera and
   low-dim key sets), and confirm the success signal (sparse reward 1 at a stack) ends the episode as
   Square's does.
3. **Dataset** (a few hours, a little GPU for EGL): a LeRobot dataset re-rendered through our env.
   `tools/rerender_demos.py` copies its non-image columns from a LeRobot source, which Stack lacks, so it
   needs a path from the hdf5 alone. Add `tools/render_check.py`'s expectations for Stack.
4. **Base policy** (about a day of GPU): the diffusion base, trained as v9 was, then 200 or more episodes
   of the base alone. **A headroom rule, set before training:** if the base succeeds above about 0.85,
   train it on fewer demos or move to Stack D1 (±20 cm, wider camera).
5. **The symmetry record** (CPU): the layout test is DONE (the 10-31 FINDING: about the base no turned Stack
   layout is real; this corrects the pre-registration's premise). Not done: the turned-demo replays
   (`outputs/diagnostics/symmetry/turned_replay.py`).
6. **Agent plumbing:**
   - robot-base detection and the normalizer's statistics from the new dataset;
   - every Square-specific spot in the trainer, preflight (its task checks and render check), the
     watcher and the digest.
7. **Arms:** `stack-eee` and `stack-rrr`. For each: the early-speed check, `test_sweep_arms.py`, the
   queue's `kind()` and `smokes_of()`, and the smokes.
8. **Pre-register:** seeds 2 and 3, 150k, pools at 110k–150k, the 0.13 bar, and the predictions (`eee`'s
   gain over `rrr` larger than on Square).
9. **Runs:** four, two at a time, about 3–4 days; then the readout.

**In all, about 1–1.5 weeks,** mostly waiting on the GPU. Steps 1–3 and 5 need no training slot, so they
can run while O1 trains. The detailed copy is `docs/plans/2026-10-27-long-O-L-steps.md`, section 4 (local).

### B23. The observation, latency and chunk experiments — O1 BUILT 2026-10-27 (B24); L SIDELINED and C parked (Aden, 2026-10-26 and 10-27) `[SC]`

Ideas Aden asked for on 2026-10-26. O1 is built and pre-registered (B24, call 5). L is sidelined until the
long runs, O1 and a new task have results; its design calls stand as recommended: eee as the anchor, a
delay on observations and actions with the base seeing the delayed frames, 0–2 steps drawn per episode and
held, and the policy not told its delay. The steps plan's section 3 (local) has L's build. Every new field
is off by default, and identical when off.

**Where things stand** (read from the code on 2026-10-26):
- **No latency.** The env runs at 20 Hz, and the action from observation t is applied at t.
- **The base plans in chunks.** The Square base (`3hzs5bz1` v9) is a diffusion policy (DDPM, 100 steps,
  fresh noise each plan). It plans 16 steps from 2 observations and runs 15 of them open-loop, so it
  re-plans every 0.75 s.
- **The residual acts every step** on this step's base action alone. It does not see where it is in the
  chunk, or the rest of the plan.

**O. Object-centric observation** (after `docs/papers/obj_cent_resrl.pdf`: residual RL on 6-DoF object
poses plus proprio plus the base action, with pose noise and dropout for sim-to-real).
- O1: **built 2026-10-27**, as a 2×2: the nut alone, or the nut with the peg (the old O4), on `eee` and
  `rrr` (the CODE and PRE-REGISTERED entries of that date).
- O2 `obj-critic`: the pose to the critic alone. Is the critic's view the bottleneck (arch-swap: the
  critic is where the constraint mattered)? The actor stays deployable without a pose estimator.
- O3 `obj-only`: pose, proprio and base action, no images (the paper's recipe). Do images matter to the
  residual? Likely several times faster per run (unmeasured).
- Later: the paper's noise and dropout (5 mm, 0.1 rad), for sim-to-real.

**L. Sampled latency and control delay** (after `docs/papers/steadyTray.pdf`: observation delay and
control latency, sampled and held over many steps, were critical to its transfer).
- L0, eval only, no training: existing best models (`ere`, `rrr`, `eee`, and the base alone) at 0–3
  steps (0–150 ms) of observation delay, and of action delay. How fragile is what we have?
- L1 `act-delay-2`: every action applied 2 steps (100 ms) late. Can the residual compensate?
- L2 `delay-rand`: observation and action delay each sampled from 0–2 steps per episode and held;
  evaluated at 0, 1, 2 and 3. Robust at each delay, and at what cost at zero?
- L3 `delay-rand-aware`: L2, with the actions still in flight in the actor's and critic's input. A
  delayed problem is not fully observed without them.
- Build: an env wrapper with an observation FIFO and an action FIFO, delay 0 identical to now. The base
  sees the delayed frames too, as on a robot; it was trained without delay, so its own level falls.
  Delay is a trivial quantity, and the in-flight actions are action fields.

**C. How the base's chunks reach the residual.**
- C0, eval only: trained best models with the base re-planning every 15 steps (as trained), 8, 4 and
  1. Does the residual's correction depend on the chunk schedule?
- C1 `chunk-phase`: the step's index within the chunk to actor and critic (a trivial field). The
  cheapest fix for the hidden part.
- C2 `chunk-lookahead`: the base's next 4 planned actions to actor and critic (fields of the base action's
  types). Can it correct a plan it sees coming?
- C3 `replan-4`: the base re-plans every 4 steps. Does a shorter open-loop stretch leave the residual
  less to do? The base's step-0 eval changes too.
- C4 `base-deterministic`: the base's noise fixed per episode, or DDIM with no noise. How much of what
  the residual corrects is the base's own sampling randomness?
- Later: a residual over the whole chunk at each re-plan (Policy Decorator's, which ResFiT contrasts
  itself with); temporal ensembling (a re-plan every step); steering the base's diffusion noise instead
  of adding to its action.

### B28. The actor's rate set in action units, or the actor held until the critic informs it — WAITING ON ADEN `[SC]`

Each new arm's actor rate is picked by hand so its residual's 0–3k speed lands in 0.025–0.045 (no run outside
it beat the base: the 2026-10-28 NOTE on 49 runs). The rate becomes a speed through a gain the architecture
sets. AdamW moves each weight about the rate per step, so the output moves about rate × the last layer's input
count × their mean size (the 2026-10-16 MEASUREMENT). LayerNorm (4×) and 448 fields (3.5×) make that 12×
between actors; new inputs or a new task shift it again (Stack's data about 8%). AdamW also gives a weak critic
full-size steps, which is why too fast pins the residual at its 0.1 limit before the critic informs it. Two
ways to remove the hand-tuning, neither built (Claude to Aden, 10-30 evening):
1. **The rate in action units:** the actor's rate divided by its last layer's input size, so one number is one
   speed for every actor. It changes how new runs compare with earlier ones, so it is Aden's call.
2. **The actor held** until the critic values online data: the tuning plan's "critic warm-up 20k" arm (B24).

### B27. Whether `eee`'s later runs go past 150k — WAITING ON ADEN `[SC]`

`long-eee 4` gained 0.15 from 110k–150k to 260k–300k (0.648 to 0.800) while ResFiT's run gained 0.03, so the
gap at 150k is partly how long `eee` runs. One run, so a description (the `long-1027` NOTE). Options: Stack's or
O1's `eee` runs to 300k (each about 30 h more), a second `long-eee` seed, or the 200-episode re-evaluations of
`long-eee 4`'s checkpoints at 150k and 300k (`tools/residual_field_ablation.py` reads the equivariant agent
already; a few GPU hours, no training slot). Claude's recommendation: the re-evaluations first. They are
cheapest, and at 200 episodes (standard error about 0.035) they test whether the 0.15 is real.

### B26. More seeds of `ere` (the actor-design readout's registered next step) — WAITING ON ADEN `[SC]`

The 2026-10-30 NOTE: with nothing counted in B22, the registered next step is more seeds of `eee` and `ere`.
eee has five runs at 150k (mean 0.609); `ere` has two (0.688, 0.696), and C4 against eee's five is +0.083,
under the bar. Seeds 4–6 of `arch-swap-ere` would settle whether ResFiT's actor helps at all. About 3 slots of
30 h; after O1 and Stack, or in place of their seeds 3. Aden's call.

### B22. The actor's design: the equivariant actor wired as ResFiT's, at two widths, and arch-wide's actor unconstrained (`actor-design-1026`) — DONE 2026-10-30: read, nothing counted (Aden, 2026-10-26) `[SC]`

**Read** (EXPERIMENTS.md, the 2026-10-30 NOTE): D1–D5 +0.044, +0.056, +0.030, +0.108, +0.122, none at 0.13,
every pair at matched speed. The arms pooled 0.584 (shaped), 0.640 (shaped, wide), 0.570 (`eae`) against
`eee` 0.540 and `ere` 0.692; against eee's five runs (0.609) all three are within 0.04. The tuning anchor
stays `eee` (B24). Next: B26. The record of the build and the runs follows.

C4 counted in B20: ResFiT's actor on our encoder and critic beat ours by 0.152. Three arms on eee's launch,
seeds 2 and 3 (EXPERIMENTS.md, 2026-10-26 CODE, DECISION and PRE-REGISTERED; the runs in the 10-27 to 10-30
LAUNCHED entries): `ResFiTShapedActor` (`equivariance.actor_design=resfit`) at 128 and 362 fields, and `eae`;
calibrated in `outputs/diagnostics/actor-design/`. Its worktree `a5b5cc8` and watcher `ad-watch` were removed on
10-30.

### B21. The encoder's interface: the encoders widened, their last layers widened, or a window for the critic (`enc-iface-1023`) — DONE 2026-10-28 as cut by the 2026-10-26 and 10-27 DECISIONs (Aden, 2026-10-23, 10-24 and 10-26) `[SC]`

[The plan](plans/2026-10-23-encoder-interface.md) (local). Its record is EXPERIMENTS.md: the 2026-10-23
CODE, MEASUREMENT and PRE-REGISTERED entries; the 2026-10-26 LAUNCHED and NOTE; the 2026-10-27 NOTE; and
the 2026-10-28 NOTE.
- **Results:**
  - `iface-window` two seeds 0.298, never above step 0; Prediction 1 failed.
  - `iface-wide-out 2` 0.452.
  - `iface-wide-all 2` stopped at 113k.
  - `iface-eee 2` 0.676, making eee's four runs average 0.599.
- **Not read:** I1, I2 and I4, which have one seed on a side, and I3 as a contrast, since its anchor has
  one run.
- **Left, optional:** probe `eee 3`'s encoder (`python -m tools.encoder_probe probe eee3`).

### B20. The architecture swap: each network of the equivariant agent ablated or ResFiT's (`arch-swap-1021`) — DONE 2026-10-26: read, its NOTE written; the drift readout dropped (Aden, 2026-10-20 and 10-26) `[SC]`

[The plan](plans/2026-10-20-arch-swap.md) (local), approved by Aden with Claude's recommendations on
its four questions (2026-10-20). Six arms, [encoder · actor · critic] each equivariant (E), ablated
(A) or ResFiT's (R): `eee` (`arch-wide` again), `aaa`, `ere`, `ara`, `rrr`, `rra`, at seeds 2 and 3.
- **Done:** the code (2026-10-20 CODE); the calibration, the smokes, the DECISION and PRE-REGISTERED
  entries (2026-10-20; the rates Aden's, 10-21); the launches from `aec1bdc` (2026-10-23 and 10-26
  LAUNCHED); all twelve at 150k by 09:36 on 10-25; `aaa 3`, `eee 3` and `ere 3`'s `metrics.jsonl` rebuilt
  from wandb's server (2026-10-24 NOTE); the readout (2026-10-26 NOTE): `ere` 0.692, `rrr` 0.648, `eee`
  0.540, `rra` 0.402, `aaa` 0.388, `ara` 0.314; C1, C2, C4 and C6 count, C3, C5 and H do not; no run with
  an unconstrained critic beat its step 0.
- **Phase 2** (Aden, 2026-10-26 DECISION): the actor's design, B22.
- **Decided** (Aden, on Claude's recommendations; EXPERIMENTS.md 2026-10-26 DECISION, the last of the day):
  1. **The drift readout is dropped.** Every A run's saved best is its untrained step-0 model, so it would
     measure nothing of training.
  2. **The slower dense critic is the first candidate after B22 reads,** to be planned and pre-registered
     for Aden's review. It is `aaa` at `agent.critic_lr` about 3.3e-5. Open for the plan:
     - `agent.critic_lr` sets the encoder's rate too, so: slow both, or the critic alone?
     - `ara` too?
     - GPU 0 is free from about 10-29 01:30.

     It separates "the constraint helps the critic" from "the dense critic is too fast at 1e-4".
  3. **C4's margin:** no new runs. At B22's readout, C4 is also reported with `iface-eee 2` and 3 as more
     `eee` replicates. More `eee` and `ere` seeds only if B22 reads nothing.

### B19. The actor parts at matched speed, and the speed's two edges (`equi-speed-matched-1017`) — DONE 2026-10-19 (Aden, 2026-10-17) `[SC]`

[The plan](plans/2026-10-17-speed-matched.md) (local). The four arms are `rqgb47jo`, `7y6cp532`,
`xsordrhh` and `p7rodove`, from `8e8eceb`, at seed 2. EXPERIMENTS.md has the 2026-10-17
PRE-REGISTERED entry, and the 2026-10-20 LAUNCHED entry and 150k NOTE.
- **At matched speed, the actor's norm keeps the residual off its limit early and scores more.** It
  adds +0.12 at 10k–30k and +0.09 at 110k–150k. The width adds nothing late and costs early.
- No arm matches `arch-wide`. The cheap recipe scored 124, and everything stacked 78, below its
  parent. init0's rate curve is flat from 1e-5 to 1e-4.
- No follow-up is planned (2026-10-20 DECISION).

### B18. arch-wide's actor alone, and a faster actor (`equi-actor-speed-1016`) — DONE 2026-10-18 (Aden, 2026-10-16) `[SC]`

[The plan](plans/2026-10-16-actor-speed.md) (local). The runs are `x3xopn1y`, `0jqo4jrx` and
`762bhig5`. Their 150k NOTE is in EXPERIMENTS.md (2026-10-20).
- Every arm is better than init0, and none matches `arch-wide` (139): `actorln448` 128, `actorlr1e-5`
  110, `actorlr3e-5` 115.
- For the rate arms the reading is "speed is part of it". For `actorln448` it is "the critic parts add
  something in combination"; the registered next step for it is not run (2026-10-20 DECISION).

### B17. Which part of ResFiT's layout helps the equivariant residual — DONE 2026-10-17 (Aden, 2026-10-15) `[SC]`

[The plan](plans/2026-10-15-equi-arch-parts.md) (local). Its NOTEs are in EXPERIMENTS.md (2026-10-17,
and 2026-10-20 for `actor448`). At 110k–150k, of 250, at one seed each: `arch-wide` 160, `actorln` 104,
`actor448` 93, init0 83, `criticln` 71, `heads` 56. No seed 3 (2026-10-17 DEVIATION).

### B16. Demos rendered by our env, then the equivariant residual on them — DONE through item 5e (Aden, 2026-10-08) `[SC]`

**Done; condensed 2026-10-27.** The demos' video codec gave the critic a perfect cue for telling online
frames from demo frames, so Aden chose to fix the demos (EXPERIMENTS.md, 2026-10-08 CODE, "re-rendered
demos"). What was done:
- `tools/rerender_demos.py`, with its guards and tests; frames one physics substep late, as `step()`
  observes;
- both datasets built and verified, their figures in each `meta/rerender.json`;
- the Square runs (2026-10-10 NOTE);
- the stabilization group, cancelled (2026-10-12 DEVIATION);
- the residual's start (2026-10-12 NOTE);
- the tilt diagnosis, not the cause (2026-10-13 and 10-15 NOTEs);
- the 2026-10-13 schedule: ResFiT's residual needs the actor's LayerNorm (2026-10-15 NOTEs).

**Still open from it:**
- If the stabilization group (5b) is ever resumed, its second-wave candidates: the clamp and the pooled
  norm together; the LayerNorm arm plus per-pixel LayerNorm in the encoders.
- `vqzx1ir9`'s 150k NOTE (B17 step 0.2).
- Move `outputs/queue/launchers/` into `tools/`.
- Make `preflight.py` record a failing assertion, not just the test's name.
- From the regroup (Aden, 2026-10-09), not triggered as written: the residual's own design ablations on
  Can (norm, squash, head count and depth, N = 8, 4 or none). Caveat: Can's base sits at 0.852, which
  leaves little headroom.

### B15. Fix the render, then new baselines — RAN 2026-10-07 to 10-10 (Aden's plan, 2026-10-07) `[SC]` `[CS]`

**Done; condensed 2026-10-27.** The record runs from EXPERIMENTS.md's 2026-10-07 FINDING (mujoco 3.8.1
renders the floor unlike the demos) and its CODE entries ("records in main", "mujoco 3.3.2") to the
2026-10-10 NOTE on `s8opblsd`:
- the worktrees' records moved into main;
- `tools/gpu_queue.sh` fixed;
- mujoco pinned to 3.3.2, with the render check in the gate;
- B_sq and B_can re-measured at 500 episodes;
- `critic_diagnostics` for both agents;
- the five baseline arms.

**Kept here because no entry holds them:**
- **Removing launch worktrees.** Do it only for worktrees that no live process or armed queue uses
  (`/wrapup` checks), and unlink the symlink first, so nothing can follow it:
  ```bash
  cd ~/projects/equi-resrl
  for w in ~/projects/equi-resrl-*/; do
      [ -L "$w/artifacts" ] && unlink "$w/artifacts"
      git worktree remove --force "$w"   # --force: 7dae4925 has an untracked empty tea_debug.log
  done
  git worktree prune && git worktree list
  ```
- **Why the zero-init arm was added** (Aden's review, 2026-10-07): `a3e3zylp` (0.88 → 0.00) against
  `z8yoqylh` (1e-4, 0.82 → 0.94). That evidence is confounded with host and software, and both runs saw
  near-blank images. A zero-initialized residual leaves zero within 4 actor updates, with no NaN. The
  tuned-RLPD actor keeps its default init, since it outputs the whole action (Aden).

### B11 — DONE 2026-10-04: the reference with the image fix collapsed, both seeds `[SC]`

`fal5sfot` and `fqo83rvd` (`square-image-fix-1002`) went from 0.46 and 0.52 at step 0 to 0.00 at
every eval from 10k (EXPERIMENTS.md, 2026-10-04 RESULTS). Executed, the residual grew to the clip
while the critic rated it above the base action. So, as pre-registered, the IID-batch-norm actor's
earlier stability came from executing almost no residual, and the equivariant tuned-RLPD arms
waited for Aden (B12; now B15).

### B12 — RESOLVED 2026-10-07: tuned RLPD failed because of our env's render `[SC]` `[CS]`

Every tuned-RLPD arm in our env scored 0 at every eval: `8ma5alnm` and `lg8qad3l` to 150k, the
Can screen, τ 0.005 and the BC term, the BC term training the encoder (`44enq26h`), and no warm-up
(`hi8t2hps`) (EXPERIMENTS.md, 2026-10-04 to 2026-10-07 RESULTS). Upstream's code in upstream's env
reached 0.88 at 20k on Can (`zwmk393l`), against 0 for the same code in ours (`rdoj0ex8`). The
measured cause is the render (2026-10-07 FINDING). So none of those arms says how its change
behaves in an env that matches the demos. Next is B15.
- **Not run:** `clean-residual-can`, the pre-registered env control (it needs about 21 GB of disk).
  An env where tuned RLPD reaches 0.88 is plainly working; whether to run it anyway is Aden's call.
- **Moot unless the fixed env still fails:** the warm-up alternatives, a larger BC coefficient,
  BC pretraining, and the questions for the paper's author (PROGRESS.md, 2026-10-07).
- Plan doc: `docs/plans/2026-10-02-tuned-rlpd.md` (untracked).

**Updates:** EXPERIMENTS.md, STATUS.md.

### B14. DONE 2026-10-06 — our trainer ignored `agent.bc_loss_coef` `[CS]`

Wired as upstream's `train_rlpd_dexmg.py` does: the demo half goes to each actor update when the
coefficient is positive (EXPERIMENTS.md, 2026-10-06 CODE, "the actor's BC term"). The 0 path is
unchanged, bit for bit. The trainer refuses a positive coefficient in a residual or equivariant
run, and `bc_loss_dynamic` at any value but 0. ARCHITECTURE.md has the details.
- **Left open:** the equivariant agent's BC path. It needs a test before an equivariant BC arm.
  Delete this entry once that is done or decided against.

### B0 — PARKED 2026-10-02: the Square ablations, with the actor's batch norm `[SC]`

Aden's decision. Recorded in EXPERIMENTS.md, 2026-10-02 LAUNCHED, RESULTS and STOPPED:
- **Did not advance:** `critic-fn` and `clipq`.
- **Collapsed at 30k:** `actor-fn` and `actor-inner`.
- **No verdict:** `enc-inner` and `target-fix` seed 2, killed when the machine went down.
- **Never ran:** `squash` (failed gate) and `depth`.

Every one of these acted on near-blank images, so none of them tested its design. The arms stay
launchable at `ccff539`. GPU 0's leftover line `ccff539 depth 1` is commented out in
`~/queues/gpu0.txt`, with the file as it was in `gpu0.txt.bak-20261002`.

### B10. Why the executed residual collapses the policy — REOPENED 2026-10-02 `[SC]` `[CS]`

Both runs without the actor's batch norm went from 0.52 to 0.00–0.02 by 10k (2026-10-01 RESULTS).
FieldNorm and InnerBatchNorm in its place collapsed too, by 30k (2026-10-02 RESULTS). That looked
like it falsified the vision-share prediction. But act() saw no usable vision under the bug, so the
prediction was never tested.
- **All of it ran under the image bug** (2026-10-02 CORRECTION). act() saw near-blank images. With
  them, the IID-batch-norm actor executed about 0.001 of a 0.3 residual. The other actors executed
  whatever they compute from a blank image, which cannot be measured: their best checkpoints are
  from step 0.
- So the collapses and batch norm's "stability" both fit the bug, and neither is established. The
  2026-10-01 RESULTS' "ruled out: batch norm muting the residual at eval" was measured on buffer
  images only.
- **B11 answered the first half: no** (2026-10-04 RESULTS). Executed, the IID-batch-norm actor's
  residual grew to the clip and both seeds collapsed by 10k, while the critic rated the residual
  above the base action. So no equivariant residual design has yet been stable with its residual
  executed. Why the residual grows is now the question; tuned RLPD (B12) checks the RL machinery
  without the residual first.
- `actor_norm=none` is still the config default (STANDARDS 1.6). Decided for the new baselines:
  `none`, with the residual bounded by `residual_squash=true` (B15).
- **All of it also ran under the render mismatch** (2026-10-07 FINDING), mildly on Square. B15's
  equivariant residual arm, bounded and in the fixed env, is the next test.
**Updates:** EXPERIMENTS.md, ARCHITECTURE.md.

### B13. Small findings from 2026-10-02 `[CS]` `[SC]`

- **FIXED 2026-10-06: the ResFiT agent never logged `debug/dQ_da_*`.** `combined_action` got
  no `retain_grad()`, so the metric read 0.0 in every ResFiT run before then. It now mirrors the
  equivariant agent (2026-10-06 CODE).

- **`agent.critic.orth` does nothing** (2026-10-06 CORRECTION). ResFiT's critic gets PyTorch's
  default init whatever it says, in every ResFiT run so far, upstream's included. It breaks
  STANDARDS 1.2, but fixing the init would change every ResFiT run's critic. Decide between
  deleting the field and fixing the init as an explicit, recorded change, after the tuned-RLPD
  screen.

- **The residual path clips twice, differently.**
  - `QAgent._combine_actions` gives the critic `equi_clip(base + residual)`: each rotating pair
    goes onto the unit disk.
  - The buffer stores the unclipped sum, and the env clips it per axis, to a box.
  - So the critic is trained on actions that were not the ones executed. 3.2% (Square) and 14.9%
    (Can) of demo xy actions lie outside the disk.
- **The eval video's Q annotation is wrong for the equivariant agent.** `evaluate_dexmg.py` scores
  the residual alone, because of its `if agent.residual_actor and not equivariant`. Cosmetic.
- **`encoder_orth_init` gives a near-dead encoder** for about 1 delta-orthonormal init in 20: output
  about 1e-8, 85% zeros. No run has used it (2026-10-02 CODE).
- **`ablate_equi_obs_encoder.py`, which nothing imports, has the same image bug.** One more reason
  to delete it (P3.5).
**Updates:** ARCHITECTURE.md ("Known rough edges").

### B1 — DONE 2026-09-25 `[CS]` `[SC]`

`ResObsEnc` generalised to `n_arms`; normalizer, probe, `QAgent` asserts, config and tests followed.
Single-arm output verified bit-identical by `tests/test_regression_single_arm.py`, so prior Can and
Square runs stay comparable. `preflight.py` now carries `image_keys` / `num_episodes` per task and
estimates cache size per task.

### B2. Record the five undocumented high-scoring runs `[SC]`

`lgc70vgd` (0.86 → **0.98**, 243,800 steps, no overrides), `q1ocx89o` (0.98), `3vl625tv` (0.96),
`36pfxsww` (0.96), and `zyed0lew` (non-equivariant Can baseline, 0.76 → 0.94). None has an
EXPERIMENTS.md entry. `lgc70vgd` reached a higher best with `actor_last_layer_init_scale=0.0`, which
contradicts the premise of R11 — resolve that before running R11.
The four equivariant ones acted on near-blank images (2026-10-02 CORRECTION), so they record the
base policy's draws, not the method. That makes the init-scale question, and R11, moot.
**Updates:** EXPERIMENTS.md, then R11.

### B3. Launch the BoxCleanup baseline `[SC]`

**Progress 2026-09-27/29.**
- Done: the ACT reset crash is fixed (`f428c19`, with a test), and the BC checkpoint
  `run_xg0w2r0z_best:v7` is backed up and pinned in both configs and in `preflight.TASKS` (`93a8ad4`).
- `write_buffer_caches=false` exists, and the gate can require caches.
- The smoke run (PROGRESS 2026-09-29) measures RAM and speed.
- Still open: the launch itself, which now follows B11 and B12 (STATUS, "The plan").
- The rest of this item is the original text.

Blocked on Square finishing (B5) and on **a crash in the residual env wrapper**:
`BasePolicyVecEnvWrapper.step` resets an ACT base policy with `env_ids=terminated_envs`, an undefined
name, so the run dies with `NameError` on its first successful episode. Can and Square never reach
that branch. Fix it to `env_ids=reset_ids`, which touches only ACT runs, and cover it with a test.

The ACT BC run finished: `dexmg-boxcleanup-bc/xg0w2r0z`, 50k steps. Back up `run_xg0w2r0z_best`
first — the old code deleted the local copy, so the wandb artifact is the only one. Then set
`base_policy.wandb_id` in **both** `config/residual_td3.py` and `preflight.TASKS` — they must agree —
pin the id in `test_config.py`, pre-register, rewrite `submit.sh`, and have ~51 GB free.

The pre-registration must carry the BC policy's **full evaluation sequence**, not its peak. It
oscillates ±0.15; ten-eval averages are 0.61 for 21k–30k, 0.54 for 31k–40k and 0.51 for 41k–50k, and
the recorded best of 0.74 at 24k, the checkpoint `wt_type="best"` loads, is an outlier. Without that
on record, the residual run's step-0 will look like a discrepancy when it is not.
**Updates:** EXPERIMENTS.md, STATUS.md.

### B4. Launch the equivariant BoxCleanup `[SC]`

Blocked on B3 building the shared caches and on a free GPU. Config
`residual_equi_td3_box_clean_config` already exists. The pre-registration must restate that this
measures a geometric prior, not an exploited symmetry — see EQUIVARIANCE.md.
**Updates:** EXPERIMENTS.md, STATUS.md.

### B5 — SUPERSEDED 2026-09-29 `[SC]`

The two BoxCleanup arms can't share caches: the cache key doesn't record the normalization path,
and the baseline's data is normalized differently. A full run also needs about 82 GiB of RAM
(projected from the 2026-09-29 smoke run's measured 59.8 GiB), and takes about 4 days at 0.69
s/step. So they run **one at a time, without writing caches**, after the Square sweep or
scheduled against it. The original decision follows.

### B5 (original) — DECIDED 2026-09-25 `[SC]`

**Wait for Square, then run both BoxCleanup arms concurrently.** Square finishes around 2026-09-29
and frees 22.4 GB plus GPU 1. Sequential would have been ~19 days and would have launched into a
~5 GB disk margin; the pairing also gives both arms identical wall-clock conditions, and they share
caches regardless. Cost is ~3.5 days of an idle GPU 1.

---

### B6. Re-run the Can replication with the target-critic and image fixes `[SC]`

Every Can result, the replication included, trained against a frozen target critic (EXPERIMENTS.md,
2026-09-29 CORRECTION). After B11 and B12, re-run the Can reference config at a commit with both
fixes (the target critic and act()'s images, 2026-10-02 CORRECTION), over 3 seeds, before any Can
claim is repeated. The Can caches are gone (H7), so the first
run rebuilds them.
**Updates:** EXPERIMENTS.md, STATUS.md.

### B7. The rest of the 2026-09-29 non-sweep plan `[SC]` `[CS]`

From `docs/plans/2026-09-29-non-ablation-work.md` (untracked). S1, S4, S5, S6-Square and S8 are done.
- **S9, the cache key.** Record the normalization path (raw vs dataset-normalized), the base
  policy's version, and the baseline's `action_scale` in both keys. Migrate the two Square caches
  by renaming, leaving symlinks, and check that older commits still load through them. The gate
  then derives its disk need from `write_buffer_caches`.
- **S10, the RAM gate.** Per-task RAM estimates from measured PSS, counting running trainers.
  `RAM_PER_RUN_GB = 21` is half the measured 40.3 GiB.
- **The gate's BC-policy check looks in the launch worktree** (`preflight.py`, Tier 3:
  `REPO.glob("artifacts/...")`), but since 2026-10-07 the trainer keeps BC policies under
  `CACHE_DIR/artifacts/`. So a launch from a worktree warns "BC policy not in artifacts/ — will
  download ~1 GB" while the policy is cached in main (`equi-residual-square` seed 1, 2026-10-07:
  the warning, and main's `run_3hzs5bz1_best:v9` files untouched). Look under `CACHE_DIR`, as the
  cache checks do. Not changed in the fixed-env baselines' launch commit `9cf207d`.
- **S11, `tools/eval_report.py`.** Parse evals from `output.log`, and implement the stop rule as a
  parameter, with golden and calibration tests. That makes the sweep's verdicts mechanical.
- **S12, the records.** Enter the BoxCleanup base rate (E1-box, still to measure: about 1–2 h, 25 GiB).
  Correct the Can replication's headline, now also for the frozen target (B6).
- **S13, the bimanual pilot prep.** A launcher, its pre-registration and a launch checklist, with
  `num_q=10` and the paper's settings, using the smoke run's measurements.
**Updates:** TODO.md, EXPERIMENTS.md, ARCHITECTURE.md.

### B8. Env seeding, and the Square BC `[SC]`

- Env initial states are unseeded in every run (2026-09-27 CORRECTION). **Decision: leave them
  unseeded** for comparability with every earlier run. The trainer's dead
  `if hasattr(env, "seed")` branches should then go, with a comment.
- Whether to train the Square BC policy longer: `3hzs5bz1` was still improving at its 50k budget.
  This is for later, and blocks nothing: `870ws2c2` lifted this base to 0.90.

### B9 — RESOLVED 2026-09-30: a reference at the new actor default `[SC]`

`actor-none` is that reference, and runs first as the gate (EXPERIMENTS.md, 2026-09-30
PRE-REGISTERED). That gate ran under the image bug, and B10 reopens it.

## H — Housekeeping

### H1, H3 — RESOLVED 2026-09-16 `[CS]` `[HI]`

`rl_utils.py` deleted: untracked, unimported, and a hard `SyntaxError` on line 15, so it had never
once loaded. Its only original content, `TrivialLayerNorm`, is preserved verbatim in P3.4 below.
`note.txt` and `log.txt` removed from tracking, content migrated into these docs and recoverable via
`git show 8442ba2^:note.txt`. Four unreachable modules remained, deferred to P3.5 (all deleted by
2026-10-24).

### H2 — CLOSED 2026-10-24: the two git stashes are gone `[HI]`

`stash@{0}` on `aabde0f` (*"transferring to new compute"*) and `stash@{1}` on `d9909ce` were never
reviewed, and no longer exist: `git stash list` is empty and neither survives as a dangling commit
(`git fsck --no-reflog`), so nothing is left to review. The one dangling stash commit found,
`3f86708` (2026-09-27, on `f3e6315`), changes only `residual_env_wrapper.py`, and that file at
`7f9c42a` is the stash's version exactly: its work landed.

### H4. Restore `z8yoqylh` to wandb `[SC]`

It exists only as two local directories on `boce-WS-01` — see the last section of STATUS.md.
`wandb sync ~/projects/equi-resrl/wandb/run-20260522_083555-z8yoqylh` would put it back in the cloud
from local data. Aden's call, because it publishes; recommended, because this project has already
lost two runs to wandb deletion and this is the one that matters.
**Updates:** STATUS.md, EXPERIMENTS.md.

### H5 — RESOLVED 2026-10-24: EXPERIMENTS.md left out, the budget 350 KB `[HI]`

**Aden, 2026-10-24:** the first option below, with the ceiling set to what the rewritable docs hold now plus a
little room: 350 KB for everything but EXPERIMENTS.md (329 KB that day; the whole set was 999 KB).
Over budget means condensing the rewritable docs, never trimming EXPERIMENTS.md. CLAUDE.md, the
session-end skill and the wrapup skill state it with one command.

**The problem it resolved** (condensed on 2026-10-27): EXPERIMENTS.md is append-only, and rule 5.4 requires
a pre-registration for every run, so it can only grow. Under the old ceiling (150 KB for `CLAUDE.md`,
`docs/*.md`, the skills and `settings.json`; 201 KB by 2026-09-25) every session that launched anything
breached it further. The options were to leave EXPERIMENTS.md out (recommended, and chosen), raise the
ceiling, or keep cutting the rewritable docs. The last was not recommended, since the next cuts would
trade correctness for size.

**Updates:** CLAUDE.md.

### H9. `debug/enc_normalizer_grad_norm` is a running sum, not a gradient norm `[CS]` `[HI]` (found 2026-10-10)

The equivariant normalizer is attached (`train_residual_td3.py:528`) after the agent has built its
optimizers and its clip list (`q_agent.py:242, 265`). So its parameters, which require grad, take
gradients that no optimizer steps, no clip touches, and nothing ever zeroes. The logged norm is
their running sum: it climbs steadily in every run (0.2 to 1.07 in `gkck6eqp`) and reached 3e3 in
`vc5o977f`. Harmless to training: the normalization never moves. Fix: `requires_grad_(False)` on
them in `set_normalizer`, checked bit for bit by the regression goldens. Kept out of
`square-trlpd-stab-1010`'s commit to keep its diff to the arms. **Updates:** ARCHITECTURE.md
("Known rough edges").

### H10. The gate's subprocesses have no timeout `[CS]` (found 2026-10-10)

`preflight.py` runs its pytest tiers and the render check through `subprocess.run` with no
timeout, so a test that hangs holds up the launch with nothing printed. On 2026-10-10 the identity
guard of `tools/rerender_demos.py` would have done that on GPU 0 (EXPERIMENTS.md, 2026-10-10 CODE;
fixed there). Fix: a timeout on each tier, generous against its usual time (Tier 1 about 3 min,
Tier 2 about 25 on a shared GPU), that records FAIL with the test that was running.

### H11. Training and execution combine the residual differently `[SC]` `[CS]` (found 2026-10-12)

The equivariant agent's actor loss, TD target and `critic_diagnostics` combine base and residual
with `equi_clip`, a disk of radius 1 on each pair (`q_agent.combine_actions`). The env wrapper adds
them without it, and robosuite clips each component to ±1. They differ only when |base xy +
residual xy| > 1 (or the tilt pair's), which the demos' xy length passes at about its 99th
percentile. Measure how often the executed and the trained action differ in a run's online data,
and by how much, before deciding whether execution should clip as training does. That change would
alter what every later run executes, so it would need its own entry.

### H12 — DONE 2026-10-24: the gate on the cleanup `[CS]`

The cleanup deleted four tools, two test files, two test functions and those files' two entries in
preflight's Tier 1 list (PROGRESS.md, 2026-10-24). `python preflight.py --skip-tier2` on `dd7e543`,
run at low priority beside the four runs (Aden's yes): every Tier 1 check passes (the render on Can
and Square, git state, 213 tests passed and 2 skipped). Tier 3's one FAIL is RAM headroom for a fifth
run (6 GB available, about 21 GB needed), the four runs, not the code. Tier 2 was not run (the GPUs
are full); the queued `enc-iface-1023` jobs run it from `58d55d7`, which the cleanup did not touch.

### H13 — DECIDED 2026-10-24: the trainer keeps saving every eval's video; `/wrapup` deletes them `[HI]`

About 110 MB per 150k Square run (16 evals). Nothing reads them, and each run's best video is kept
in `wandb_best/`; the 2026-10-24 cleanup deleted 7.15 GB of them (EXPERIMENTS.md, that date's NOTE).
**Aden, 2026-10-24: no code change.** A switch would change what a run package holds in the middle of
two groups; instead every `/wrapup` deletes the eval videos of finished runs, the best kept
(EXPERIMENTS.md, 2026-10-24 DECISION).

### H8. Left over from the sync-free update (2026-09-30) `[CS]` `[SC]`

From `docs/plans/2026-09-30-sync-free-update.md` (untracked).
- **The normalizer's gradients accumulate.** `set_normalizer` attaches it after `QAgent` builds its
  optimizers, so its scale and offset collect gradients that no optimizer steps and no `zero_grad`
  clears. Training is unaffected, but `debug/enc_normalizer_grad_norm` is a running sum, not a
  per-update norm. `requires_grad_(False)` fixes it, is exact for training, and changes that metric,
  so it needs its own CODE entry.
- **Still syncing:** `actor_norm=iid_bn` (escnn's `IIDBatchNorm1d` asserts on its running stats,
  30 syncs per env step, which make a run about 27% slower). `encoder_norm=inner_bn` no longer
  syncs: from 2026-10-01 it is `RegularInnerBatchNorm`, not escnn's `InnerBatchNorm`.
- **The ResFiT baseline's `VitEncoder`** calls `torch.cuda.synchronize()` twice per forward. Check it
  before B3.
- **bf16 autocast and `channels_last` on the encoders:** about 2× more, measured in scratch, but they
  change the numbers. A separate, pre-registered change.

### H6 — DONE 2026-09-25: stop sending large files to wandb `[SC]` `[HI]`

Aden's item. Each run now keeps a persisted package under `outputs/runs/<project>/<run_id>/` and
uploads one artifact, the best model plus its eval video, replaced in place. Layout in ARCHITECTURE.md
"Run packages", the rule in STANDARDS.md 6.4, the decisions and their reasons in PROGRESS.md
2026-09-25. Resuming a residual run is still not supported: `best_model.pt` has no optimizer state.

### H7. The Can buffer caches are gone `[SC]`

`offline_buffer_cache/8edf618e` and `online_buffer_cache/43637c10` were on disk on 2026-09-18
(EXPERIMENTS.md) and are not now; nothing records who deleted them or why. The next Can run rebuilds
them, so budget its disk and time. `test_config.py` still pins those hashes, which is correct — they
are what a rebuild must reproduce.
**Updates:** STATUS.md.

---

## R — Replicate the recovered run

**Superseded 2026-10-02.** Every run this section concerns acted on near-blank images
(EXPERIMENTS.md, 2026-10-02 CORRECTION). That covers `z8yoqylh`, the `7dae4925` seeds and the
`caf83f3` replication. So their outcomes say nothing about the method, and R9–R11 and R16 are moot
until B6 re-runs Can with both fixes. The original text follows.

**The `7dae4925` replication ran and failed on all three seeds, because it was
aimed at the wrong commit.** `z8yoqylh` ran `caf83f3`'s code with `HEAD` reading `7dae4925`; the
proof is a 117-field config match plus five schema fields that only exist at `caf83f3`. Both the
results and the correction are in [EXPERIMENTS.md](EXPERIMENTS.md), 2026-09-18 entries.

### R1–R6 — DONE 2026-09-16, then INVALIDATED 2026-09-18 `[SC]`

The recovery and the tooling work stand. The retargeting does not: R2 corrected the target from
`caf83f3` to `7dae4925`, which was backwards, and the tag `repro-z8yoqylh-actual` at `7dae4925` is
wrong for the same reason. Two tags now point at non-targets; neither is being moved, on the same
grounds as last time.

### R7 — DONE 2026-09-18 `[SC]`

All three `7dae4925` seeds sat at 0.00 for every evaluation after step 0. Seed 2114495708 completed
300k at 0.00 in 34.6 h; seeds 1 and 2 were killed at 209,500 steps after 20 consecutive 0.00
evaluations. Numbers in EXPERIMENTS.md.

### R8 — RESOLVED 2026-09-18, against the runs `[SC]`

The config caveats are settled: the three runs' config differs from `z8yoqylh`'s in five fields
(`critic.drop` vs `dropout`, `enc_degree_channel` 16 vs 32, and the three `use_*` flags absent).
`use_norms`, `use_orth_init` and `use_equivariant_model` genuinely **do not exist** at `7dae4925` —
the prior claim that they did was circular, assuming the commit in order to conclude the fields
existed.

### R12 — DONE 2026-09-18 `[SC]`

Launched `equi-repro-caf83f3`, two seeds, one per GPU: 2114495708 as `38z4wr9z` and seed 1 as
`sw2qwfs9`. Full preflight GO including Tier 2. Startup verified on both. ETA ~47 h each, so both
report around 2026-09-20. Details in EXPERIMENTS.md.

The reboot moved the machine to kernel 6.8.0-138 and driver 580.178.04, away from `z8yoqylh`'s
6.8.0-111 / 580.173.02 — recorded in the pre-registration as the one variable this replication
cannot hold fixed.

### R16. Watch the 10k and 20k evaluations `[SC]`

**The decision point, and it arrives well before the runs finish.** `z8yoqylh` was at 0.80 by its
second evaluation and never dropped below 0.80 again. Every collapse in the record — now including
all three `7dae4925` seeds — was unambiguous by 10k-20k, and nothing has ever recovered from 0.00 at
20k. Both seeds clearing their step-0 rate at 20k is the first real evidence this worked.

Read from `~/projects/equi-resrl-caf83f3/wandb/run-*/files/output.log`, **not** from
`~/repro_caf_s<seed>.log`, which is block-buffered and lags by thousands of steps.

If both collapse, suspect the kernel/driver change before concluding anything about `caf83f3` — it
is the one difference from `z8yoqylh`'s environment that this attempt could not eliminate.
**Updates:** EXPERIMENTS.md, STATUS.md.

### R13. Re-examine the four conclusions that rested on `7dae4925` `[SC]` `[HI]`

The correction entry lists them. Two need real work rather than a doc edit:

- **The `FieldNorm` hypothesis is reopened.** It was ruled out on the premise that `FieldNorm` sat in
  the successful run's critic head. That premise is gone.
- **Robot-base centering was probably enabled**, not disabled. At `caf83f3` the centering block is
  live (`equi_normalizer.py:425`); the "disabled" finding came from reading `7dae4925`, where every
  call site is commented out. So `z8yoqylh` likely rotated about the robot base and the symmetry was
  not silently broken — which removes a standing worry rather than adding one.

**Updates:** EQUIVARIANCE.md, STATUS.md, ARCHITECTURE.md.

### R14. The gate cannot validate the commit it launches `[CS]`

`preflight.py` and `tests/` do not exist at `caf83f3` or `7dae4925`, so the whole suite and the
equivariance checks run against HEAD while the launch runs older code. True of both replication
attempts. Worth fixing before the next one, or at minimum stating in every entry.
**Updates:** tests/README.md, STANDARDS.md.

### R15. `m0ylcivk`'s local directory is gone `[SC]`

It was the `caf83f3` run killed on 2026-09-16 for "not being the reproduction" — aimed at the right
commit after all. EXPERIMENTS.md:349 says do not delete that directory; it is not on disk. Check
whether it survives in wandb, and if so whether its early evaluations agree with `z8yoqylh`.
**Updates:** EXPERIMENTS.md.

### R9. Baseline arm — the local copy is the only copy `[SC]`

`msfkjwab` is deleted from wandb and survives only as
`ZXP-S-works:wandb/run-20260226_143630-msfkjwab/`. EXPERIMENTS.md already holds its numbers
recomputed from that log, so it cannot be re-pulled. **Recommend: compare against the recorded
single-seed baseline, state the 3-vs-1 asymmetry plainly, and spend compute on Square instead**,
where the base policy sits at 0.52 and the measurement actually discriminates.

### R10. Analyze against the pre-registration, then lift the freeze `[SC]` `[HI]`

Use `/analyze-experiment`. Success rate and episode length, mean and spread across the three seeds.
The question the lost comparison raised: is the flat equivariant episode length real, or was it one
seed?
**Updates:** STATUS.md, EXPERIMENTS.md, PROGRESS.md.

### R11. Isolate the two candidate causes `[SC]`

Deferred until R10 confirms the result. Then two runs at ~35-47 h each, **at `caf83f3`** — the
commit references here were `7dae4925` until the 2026-09-18 correction:

- `caf83f3` + `actor_last_layer_init_scale=0.0` on `boce-WS-01`, same seed — isolates the override.
- `caf83f3` + `actor_last_layer_init_scale=1e-4` on `ZXP-S-works` — isolates the software stack.

The second is the more interesting one. Host correlates perfectly with outcome across the entire
project history — every collapsed equivariant run on `ZXP-S-works`, the one success on
`boce-WS-01` — and no experiment has ever tested it.

---

## P1 — Standards — DONE 2026-09-16

28 rules in [STANDARDS.md](STANDARDS.md), three skills in `.claude/skills/`. The significant finding
is rule 3.3: the scalar ablation is not a controlled ablation. See P4.

Remaining follow-up: `analyze-experiment` reads run data from local `wandb/run-*/files/`, and since
2026-09-25 from run packages first. It still has to cope with **two machines** — the runs it needs to
compare live on `boce-WS-01`, not here.

---

## P2 — Equivariance tests — MOSTLY DONE 2026-09-16

`preflight.py` (3 tiers) plus `tests/` (244 tests as of 2026-09-25), passing on both machines. Full detail in
[tests/README.md](../tests/README.md). **Result: the symmetry is implemented correctly** (act() had a
separate image-scaling bug until `e15d36c`: EXPERIMENTS.md, 2026-10-02 CORRECTION) —
declared representations match the physics at all 8 group elements, the actor is equivariant and the
critic invariant to `< 1e-4`. So the collapses were not a broken symmetry.

**Still to do (P2.4, does not block the replication)**

- Full `QAgent.act` plus `_combine_actions`: assert the weaker property that holds when a
  non-equivariant base policy is summed with an equivariant residual.
- ~~Negative control: `use_equivariant_model=False` must **break** equivariance — a test proving the
  tests can fail.~~ **Done 2026-10-20**, with the ablated family: `tests/test_ablated.py`.
- Normalizer: assert isotropic scale and zero offset on every `irrep(1)` field in
  `build_equivariant_normalizer`. Currently guaranteed only by reading the code.
- **New:** the suite runs against HEAD only. The runs that matter are at `7dae4925`, and that tree
  differs in at least robot-base centering, so decide whether the suite should be runnable against a
  worktree. Relates to R8.

---

## P3 — Cleanup, gated on R10

This was gated on the reproduction (R10), which the image bug made moot (2026-10-02 CORRECTION).
STATUS.md has no freeze now: `tests/test_regression_agent.py` pins the update in its place. The
gate needs re-deciding with Aden.

### P3.1 Wire or remove every dead config field `[CS]` `[SC]`

**Progress 2026-09-29:** `num_critic_layers` is wired (default 1); `use_norms` is replaced on the
equivariant path by `actor_norm`/`critic_norm`/`encoder_norm`; the critic and encoder have their own
orth-init fields. Still to do: `num_actor_layers`, `dropout`, `initialize`, `loss_cfg`, `return_logits`.

Make `use_norms`, `use_orth_init`, `num_actor_layers`, `num_critic_layers`, `dropout`, and
`initialize` either actually branch or not exist. Use P2.3's bit-identity tests to prove the
cleanup did not move behavior. Only after this do these become real, ablatable knobs.

### P3.2 Resolve the reachable-code rough edges `[CS]`

From ARCHITECTURE.md "Known rough edges": the `std=0` versus `std=None` divergence in
`Actor.forward`, the ignored `return_logits`, the stored-and-unread `loss_cfg`, the commented-out
`assert False` debug blocks. The base-policy reset timing in `BasePolicyVecEnvWrapper` is listed
there too, but it is not a cleanup: changing it changes Can and Square behaviour, so it needs its own
comparison runs.

### P3.3 Decide about the distributional critic `[CS]`

`agent.critic.loss` (C51 / HL-Gauss) is settable and unreachable on the equivariant path. Either
implement it or make the config reject it.

---

### P3.4 Restore normalization to the critic head, using `TrivialLayerNorm` `[CS]`

**2026-10-10: built and running.** `equivariance.critic_pooled_norm=layer_norm` is this
`TrivialLayerNorm`, with the field check made strict (trivial, not one-dimensional; C8 has a
one-dimensional sign irrep). Its first run is `square-trlpd-stab-1010`'s `pooledln` arm
(EXPERIMENTS.md, 2026-10-10 CODE and PRE-REGISTERED).

**2026-09-29:** the sweep's `critic-ln` and `critic-iid` arms add normalization before the
pooling (`RegularLayerNorm`, `IIDBatchNorm1d`). `TrivialLayerNorm`, after the pooling, stays
untested.

**Scope note added 2026-09-16:** this describes the critic head **at HEAD / `caf83f3`**, where every
normalization layer is commented out and `use_norms` does nothing. At `7dae4925`, where the only
successful run lives, the head still had `FieldNorm` and the ReLUs active. So "the critic trains
with no normalization" is a property of HEAD, not of the run that reached 0.94.

At HEAD, then: the critic trains with no normalization at all while the actor has active batch norm.
That asymmetry is a plausible contributor to the instability seen as `dQ_da ≈ 2e-4` and residual
saturation — though note that the run which worked had `FieldNorm` present, so the previous
reasoning that `FieldNorm` was itself the problem no longer holds. These are update-batch
metrics, from runs that acted on near-blank images, and no equivariant run has "worked"
(2026-10-02 CORRECTION). See
[EQUIVARIANCE.md](EQUIVARIANCE.md), "What actually changed".

The obvious fix has a trap. escnn's `FieldNorm` cannot be used after `GroupPooling`, because its own
docstring warns: *"If a field is only containing trivial irreps, this layer will just set its values
to zero."* Post-pooling features are exactly all-trivial, so `FieldNorm` there would zero the Q
values.

Aden had already written the correct workaround, in the now-deleted `rl_utils.py`. It never ran (the
file had a `SyntaxError`), and it was never confirmed whether the critic head was its intended
destination — but the docstring says so explicitly. Preserved here verbatim:

```python
class TrivialLayerNorm(torch.nn.Module):
    """LayerNorm for GeometricTensors with all-trivial field type.

    Trivial reps act as identity under the group, so a plain LayerNorm
    is equivariant on these features. Use this AFTER GroupPooling, where
    FieldNorm would zero the signal (mean == feature for size-1 fields).

    Operates on flat [B, C] GeometricTensors (post-pooling, pre-final-Linear).
    """

    def __init__(self, in_type: nn.FieldType):
        super().__init__()
        for rep in in_type.representations:
            assert rep.size == 1, (
                f"TrivialLayerNorm requires all-trivial reps; "
                f"got a rep of size {rep.size}"
            )
        self.in_type = in_type
        self.out_type = in_type
        self.norm = torch.nn.LayerNorm(in_type.size)

    def forward(self, x: nn.GeometricTensor) -> nn.GeometricTensor:
        assert x.type == self.in_type, (
            f"TrivialLayerNorm input type mismatch: "
            f"expected {self.in_type}, got {x.type}"
        )
        return nn.GeometricTensor(self.norm(x.tensor), self.out_type)
```

Belongs in `equi_rl_utils.py` alongside `FieldNorm`. Gated on R10 — it changes critic behavior, so it
cannot land before the reproduction. R10 is moot since 2026-10-02 (P3's note). Any version of this must be covered by the P2.3 critic
invariance test, since normalization is exactly where equivariance tends to break.

### P3.5 Sweep the remaining unreachable modules — FILES DONE 2026-10-24 `[CS]`

**Done 2026-10-24 (Aden approved, in the repo cleanup):** `field_norm.py`, `rotation_transformer.py`,
`rotation_utils.py`, and `equi_off_policy/common_utils/utils.py`, which this list had missed, are
deleted; all four are in git at `7f9c42a`. The standalone `FieldNorm` was the worse copy: it detached
its statistics and had no spatial axes, and no run ever imported it (ARCHITECTURE.md). The fourth,
`ablate_equi_obs_encoder.py`, was deleted on 2026-10-20 with the scalar ablation it duplicated.
**Left:** `equi_rl_utils.py`'s `TruncatedNormal` and `clip_action_norm` are unreachable too: only the deleted
scalar actor's distribution path was near them, and it used `off_policy`'s.

---

## P4 — Science, after reproduction

Not yet planned in detail. Recorded so it is not forgotten.

- ~~Test the top-down camera hypothesis.~~ **Closed 2026-09-16** — reviewed and accepted as an
  approximation. The cheap diagnostic (render before/after a 45° rotation, compare against a 45°
  image rotation) stays documented in EQUIVARIANCE.md assumption 1 should it ever be needed, but is
  not scheduled.
- ~~**Match the two stacks, then run the scalar ablation.**~~ **Replaced 2026-10-20** by the ablated
  family, which differs from the equivariant stack in its weight constraint alone (STANDARDS 3.3).
  It runs in B20, `arch-swap-1021`.
- **Diagnose the residual saturation.** `residual_l2 ≈ action_scale` with `dQ_da ≈ 2e-4` across
  multiple runs. Is the critic providing any usable action gradient? Those are update-batch values,
  from runs that acted on near-blank images, and the bound is 1.0 without `residual_squash`. Re-pose
  this after B11.
- **Expand to more tasks.** Square next — its base policy sits at 0.52, so it discriminates far
  better than Can's 0.92. Then the two-arm dexmg tasks, which need BC base policies trained
  (`wandb_id` is still `TODO` for TwoArmBoxCleanup and TwoArmCanSortRandom).
- **Equivariant SAC.** No SAC exists anywhere in the repo, so this is a genuine build: equivariant
  squashed-Gaussian actor (equivariant mean, invariant log-std), entropy temperature, twin critics.
  The SO(2) equivariant RL paper's headline result is equivariant SAC, so the architecture maps
  over — but it needs P2 in place to be trustworthy.
