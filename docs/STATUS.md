# Status

**Last updated:** 2026-10-24 · **Last verified commit:** `dd7e543`, the 2026-10-24 repo cleanup,
which removed only code no run reaches: Tier 1 of the gate passes on it (TODO H12; Tier 2 not run,
the GPUs full). Later commits change docs and skills only. The queued `enc-iface-1023` jobs launch from
`58d55d7`, its pre-registration (its code is the commit before it, EXPERIMENTS.md, 2026-10-23 CODE),
and the running arch-swap runs are from `aec1bdc`. Main is pushed to origin (2026-10-24, Aden).

**Now: two groups in one queue.** `arch-swap-1021` is finishing, and **`enc-iface-1023` starts in the
next free slot, its critic-window arm first** (Aden, 2026-10-24 DECISION: `iface-window 2` and
`iface-wide-out 2` first, then `rra` 2 and 3, then `iface-wide-all` and the rest). One queue places
both groups' remaining jobs: tmux `order-queue`, `outputs/queue/launchers/order-1024/order_queue.sh`
(log `order_queue.log` there), which fits each job beside what is on a GPU by memory.
- **`arch-swap-1021`** (EXPERIMENTS.md, 2026-10-20 PRE-REGISTERED; the 2026-10-23 LAUNCHED and seed-2
  NOTE): at 09:00 on 10-24, `rrr` 2 and 3, `ere 2`, `eee 2`, `aaa 2` and `ara 2` finished (`rrr` pools
  0.648, so H does not count); `aaa 3`, `eee 3`, `ere 3` and `ara 3` running, at 150k about 12:23,
  16:52, 19:03 and 00:21 on 10-25; `rra 2` and `rra 3` queued, about 19:10 on 10-24 and 00:30 on 10-25;
  the last at 150k about 13:00 on 10-25. Seed 2: `ere 2` 172 and `rrr 2` 171 of 250, `eee 2` 129,
  `aaa 2` 85, `ara 2` 73 (110k–150k). **The local wandb records of `aaa 3`, `eee 3` and `ere 3` stopped
  at the 10-23 disk-full minutes** (the 2026-10-24 NOTE): wandb's server has every step, but their
  packages' `metrics.jsonl` is rebuilt from the server as each ends (tmux `metrics-rebuild`, Aden's
  yes), and `swap-watch` logs nothing of them. Tmux: `swap-watch`, `wandb-cache`, `metrics-rebuild`, one
  `q-arch-swap-<arm>-<seed>` per running job.
- **`enc-iface-1023`** (EXPERIMENTS.md, 2026-10-23 CODE, MEASUREMENT, DECISION and PRE-REGISTERED;
  [the plan](plans/2026-10-23-encoder-interface.md), local). The equivariant critic sees 288 numbers per
  frame where ResFiT's sees 20,736; the probe found that the agentview encoder's 9x9 map holds what its
  1x1 output loses (the nut's xy relative to the gripper, 0.11–0.14 R² at equal size), and more than
  ResFiT's tokens. Four arms on `arch-swap-eee`'s launch, at seeds 2 and 3: `iface-eee` (the anchor),
  `iface-wide-all` (the encoders twice as wide), `iface-wide-out` (their last layers wider; 1,280 numbers)
  and `iface-window` (a window over the 9x9 maps for the critic alone; 1,280). Each new arm's smoke runs
  in its first job's slot. Expected in wandb: `iface-window 2` about 14:45 on 10-24 (window share at 30k
  about 20:45), `iface-wide-out 2` about 19:15, `iface-wide-all 2` about 10:00 on 10-25, `iface-eee 2`
  13:00–20:00 on 10-25; the last at 150k about 10-28. `iface-watch` is the watcher.
- **Disk:** 31 GB free at 11:16 on 10-24, after the cleanup deleted the finished runs' eval videos
  (7.15 GB; EXPERIMENTS.md, 2026-10-24 NOTE). The 10-23 reclaim of duplicate best models (12.1 GB, the
  DECISION) had left 23 GB and about 9 GB to spare when both groups are done; about 16 GB now.

The actor-speed line is closed on its readouts; what it leaves is in "What the actor-speed groups
leave", below.

This file describes what is true right now. Rewrite it in place; do not append. For history, see
[EXPERIMENTS.md](EXPERIMENTS.md) and [PROGRESS.md](PROGRESS.md).

## Where the project is

**New on 2026-10-19 and 10-20** (EXPERIMENTS.md, the 2026-10-20 LAUNCHED, NOTEs and DECISION):
- **Every run of the three groups is finished.** Each reached 150k with no stop, its package
  complete. At 110k–150k, of 250, at seed 2 (the base alone pools to 0.553; ResFiT's anchors scored
  200 and 157):

  | Run | Arm | 110k–150k |
  |---|---|---|
  | `h216n857` | `arch-wide` | **160 (0.640)** |
  | `x3xopn1y` | `actorln448` (`arch-wide`'s actor, init0's critic) | 128 |
  | `rqgb47jo` | `actorln` at lr 3e-6 | 124 |
  | `762bhig5` | init0 at lr 3e-5 | 115 |
  | `0jqo4jrx` | init0 at lr 1e-5 | 110 |
  | `p7rodove` | init0 at lr 1e-4 | 106 |
  | `8sgfkskv` | `actorln` | 104 |
  | `7y6cp532` | `actor448` at lr 3e-6 | 97 |
  | `vnpqtnva` | `actor448` | 93 |
  | `oav5gk76` | init0, the anchor | 83 |
  | `xsordrhh` | `actorln448` at lr 1e-5 | 78 |
  | `5cj1rbpu` | `criticln` | 71 |
  | `dhhq151o` | `heads` | 56 |

  `arch-wide` is still the best equivariant residual. It is the only one above the base by more than
  noise, and by about 2 standard errors at the eval's measured noise. No other arm matches it (139).
- **`equi-actor-speed-1016`:** every arm is better than init0, and none matches `arch-wide`. For both
  rate arms the registered reading is "speed is part of it". For `actorln448` it is "the critic parts
  add something in combination"; its gap to `arch-wide` opens only after 100k.
- **`equi-speed-matched-1017`:** all four cells of the 2×2 of norm and width reached matched early
  speed.
  - **The actor's norm keeps the residual off its limit early.** Both norm cells stay under it at
    3k–10k. Both no-norm cells pin z and yaw at 0.100.
  - It also scores more: +0.12 at 10k–30k, and +0.09 at 110k–150k.
  - The width adds nothing late (−0.02) and costs early (−0.05).
  - The cheap recipe, `actorln` at a 3× rate, does not match `arch-wide` (124).
  - Past the matched speed, the norm does not protect: everything stacked (78) is below its parent
    (128).
  - init0's learning-rate curve is flat from 1e-5 to 1e-4 (110, 115, 106).
- **Across the 13 runs** (the 2026-10-20 NOTE, Claude's reading):
  - **A 50-episode eval is about 1.4 times as noisy as the binomial rules assume.** The base alone
    scored 0.34–0.68 at step 0 across 15 runs. So `arch-wide`'s 0.82 at 140k is most likely a lucky
    eval: its training-episode success did not move.
  - Every fast arm settles by 50k–70k, at or below the base, and stays flat. ResFiT's seed 2 settles
    the same way (0.628); its seed 1 keeps climbing (0.800).
  - Late in training, the residual is near its limit in every arm: xy at 0.086–0.099 of 0.1, and tilt
    at 0.063–0.096.
  - What holds the plateau is open: the residual's range and seed variation both fit.
- **Aden's decision (2026-10-20):** the next experiments are architectural, and no follow-up of the
  speed line is planned. Every run still goes to its full 150k with no automatic stop (2026-10-17),
  and no seed 3 is run for now.

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

**Nothing is running or queued** (2026-10-20), and both GPUs are idle. The last runs, the four of
`equi-speed-matched-1017`, finished on 10-19 between 03:11 and 07:57, each complete.

**Every run goes to its full 150k with no automatic stop of any kind** (Aden, 2026-10-17 DEVIATION,
which extends the 2026-10-12 AMENDMENT to the divergence stop). Aden stops a run early by hand if one
needs it. New groups go in the watcher's watch-only set.

How the last groups were launched and followed, for the next ones:
- **The queue:** `outputs/queue/launchers/retry/slot_queue.sh`, in tmux `rfe-queue`. Not running: it
  placed arch-swap's first ten jobs and was stopped on 10-24, its last two moved to
  `order-1024/order_queue.sh` (which fits each job to a GPU by memory; the 2026-10-24 DECISION).
  - Start it with `bash slot_queue.sh <jobs file> 2>&1 | tee -a slot_queue.log`. Job lines are
    `<worktree> <arm> <seed> <smoke_first> [if=<script>]`.
  - Each job goes into the first slot to have been free for 300 s, two runs per GPU, and gets its own
    tmux session, `q-<arm>-<seed>`.
  - A launch its gate stops is tried again, up to 3 tries 600 s apart (2026-10-13 LAUNCHED).
  - The render-guard test (`test_every_guard_passes_on_cans_first_demos`) still fails now and then in
    a gate that starts within minutes of a run exiting. The retry passed it on 10-18, as on 10-12; its
    failing assertion is still not recorded.
  - Its log is `outputs/queue/launchers/slot_queue.log`. The scripts are in `outputs/queue/launchers/`,
    which is gitignored (TODO: move them into `tools/`).
  - **The queue reads its jobs file once, at start.** To change the jobs, stop `rfe-queue` while it is
    only waiting for a slot (its last log line is a launch, more than a minute old), rewrite the file
    and start it again.
  - A job with `if=<script>` runs that script (in `retry/`) once a slot is free, with `JOB_WORKTREE`,
    `JOB_ARM` and `JOB_SEED` set, and launches only on exit 0.
  - The gate prints nothing until each tier ends, so its console is blank for about its first hour.
  - To check its slot count, source its functions:
    `source <(sed -n '1,/^mapfile/p' slot_queue.sh | grep -v '^mapfile')`, then `held 0; held 1`.
- **Watchers** (`tools/square_rerendered_watch.py --group <seed group>`, one tmux session per group,
  from the group's pinned worktree): none is running. `matched-watch` was stopped at 08:59 on 10-20,
  after its four runs had exited.
  - A watch-only group (`WATCH_ONLY_ARMS`) is logged, FLAGged and finalized, and never signalled.
  - The older groups' entries in the watcher still enforce stops that no longer apply.
- **tmux:** only live sessions are left: one `q-<arm>-<seed>` per running job, the queue, the
  watchers, `metrics-rebuild`, `wandb-cache` and `disk-watch`. Every idle one was killed on 10-24 (36:
  the finished runs' `q-*`, `sq-tilt-notilt`, and the old queue runners `gpu0`, `gpu1` and `gpu1-bars`,
  which polled queue files empty since 10-10), its scrollback saved in
  `outputs/cleanup-20261024/tmux_scrollback/`. The queue checks a job's `q-*` session only on a
  restart, and none of the killed sessions' jobs is in its list.

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
  - **The disk had 31 GB free on 2026-10-24** ("Disk", above). The wandb cache is cleared to 2 GB every
    12 h by tmux `wandb-cache` (`outputs/queue/launchers/wandb_cache_cleanup.sh`).
  - The cache keeps a copy of every best upload, including those wandb has since deleted, and it
    grew 18 GB in the two days before. Clear it again with `wandb artifact cache cleanup 2GB` when
    the disk falls below about 15 GB. Each run's current best is also in its run package's
    `wandb_best/`.
  - A worktree is about 4 MB. Equivariant checkpoints are 87 MB each and ResFiT's 223 MB. Each
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

**0. Now: `arch-swap-1021`** (TODO B20): which of the equivariant agent's three networks, and
which property of each (its weight constraint, or ResFiT's design), holds the equivariant residual
below ResFiT's. Running, to about 03:00 on 10-25; every early speed was inside the band and `rrr 2`
passed the gross-failure check against `bue9dupf`. Once all twelve are at 150k: `/analyze-experiment`
against the PRE-REGISTERED rules, the NOTE, the drift readout (TODO B20), then phase 2. The
actor-speed line closed on its readouts; "What the actor-speed groups leave" has what it found.

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
