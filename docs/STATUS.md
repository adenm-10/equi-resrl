# Status

**Last updated:** 2026-10-27 · **Last verified commit:** `0d9fb30`, the actor's design (EXPERIMENTS.md,
2026-10-26 CODE): its new tests, the sweep-arm tests for its three arms, the regression, config, no-op
and ablated-family suites pass on the CPU, and its host-sync tests on GPU 0; the rest of Tier 2 runs as
each launch's gate. `actor-design-1026` launches from `a5b5cc8`, its pre-registration; enc-iface's last jobs from
`58d55d7`. Main was pushed to origin through `ef6829d` on 2026-10-26 (Aden); no code has changed since.

**Now: `actor-design-1026` runs its seed 2, and `enc-iface-1023` its last runs. The next phase is planned
in Aden's order** (EXPERIMENTS.md, 2026-10-27 DECISION; TODO B24): the long runs, then the object-centric
and sampled-latency experiments (TODO B23's O and L), then more tasks and the tuning. **None of it is coded
or queued.** Aden's calls come first (TODO B24's list). One queue places both running groups' jobs: tmux
`order-queue`, `outputs/queue/launchers/order-1026/order_queue.sh` (log `order_queue.log` there), which
fits each job beside what is on a GPU by memory ("Running now", below).
- **`arch-swap-1021` is complete and read** (EXPERIMENTS.md, the 2026-10-26 NOTE). Pooled at 110k–150k,
  two seeds: `ere` 0.692, `rrr` 0.648, `eee` 0.540, `rra` 0.402, `aaa` 0.388, `ara` 0.314 (the twelve
  step-0 evals, the base alone, 0.552). Counted at 0.13: C1 (`aaa` − `eee` −0.152) and C2 (`ara` − `ere`
  −0.378), the constraint; C4 (`ere` − `eee` +0.152), ResFiT's actor; C6 (`rrr` − `rra` +0.246), ResFiT's
  critic. Not counted: H (−0.066), C3 (−0.074), C5 (+0.088). No run with an unconstrained critic beat its
  step 0. **C4 against all four runs of eee's launch is +0.093** (their mean 0.599, range 0.516–0.676; the
  2026-10-27 NOTE), under the bar: reported, not registered. Aden's calls (TODO B20; the last 2026-10-26
  DECISION): the drift readout dropped; a slower dense critic planned after `actor-design-1026`, its place
  in the new order open (TODO B24, call 7).
- **`actor-design-1026`** (EXPERIMENTS.md, 2026-10-26 CODE, DECISION and PRE-REGISTERED; the 2026-10-27
  LAUNCHED). It is eee's agent with its actor wired as ResFiT's (`actor-resfit-shape`, 128 fields, at 3e-6;
  `actor-resfit-shape-wide`, 362 fields, ResFiT's free parameters per hidden layer, at 1e-6), or arch-wide's
  actor unconstrained (`actor-eae`, at 2.5e-7). Seeds 2 and 3, against `arch-swap`'s `eee` and `ere`.
  - `actor-resfit-shape 2` (`j457zalw`) and `actor-eae 2` (`e4220e3z`) are running. Both smokes passed, both
    early speeds are at their predictions (0.0355, 0.0321), and the gross-failure check passed.
  - `actor-resfit-shape-wide 2` is in its smoke from 08:03 on 10-27.
  - The readout is about 10:30 on 10-29, since runs take about 33 h with four on the machine. Watcher:
    tmux `ad-watch`.
- **`enc-iface-1023`** (EXPERIMENTS.md, 2026-10-23 PRE-REGISTERED; the 2026-10-24 and 10-26 DECISIONs;
  the 2026-10-26 LAUNCHED and NOTE; the 2026-10-27 NOTE).
  - Seed 2, provisional, 110k–150k: `iface-window` 0.236 (`od14pmu3`; never above its step 0; its critic
    leaned on the window, 1.55 times the summary late, and valued online data as the unconstrained
    critics did) and `iface-wide-out` 0.452 (`gcrewpq2`). The anchor `iface-eee` scored 0.676
    (`mgcifv1y`), against `eee 2`'s 0.516.
  - `iface-wide-all 2` (`5shh877a`) was stopped at about 113k (Aden), a description only.
  - Still to run: `iface-window 3` (`n94ucnlv`, running, to 150k about 01:30 on 10-28), then
    `iface-wide-out 3` and `iface-eee 3`, unless Aden has them wait behind the long runs (TODO B24, call 3).
  - `iface-wide-all 3` is dropped, so I1 is not read. Watcher: tmux `iface-watch`.
- **What goes with beating the base** (a 49-run analysis, 2026-10-27; `outputs/diagnostics/success-correlates/`;
  TODO B24). Not yet an EXPERIMENTS.md entry, which is Aden's call.
  - Two necessary conditions: the early speed band (0 of 21 runs outside it beat the base) and a
    constrained or ResFiT critic (0 of 6 unconstrained).
  - The best early signal is training success at 10k–30k (ρ +0.80).
  - eee's gap to ResFiT is the critic's value of online data: 0.005 early against ResFiT's 0.082.
  - Every run above the base was still climbing at 150k.
- **Disk:** 18 GB free at 08:31 on 10-27. About 10 GB when both groups are done; the next `/wrapup`'s
  eval-video cleanup comes before then, and the long runs need the duplicate best models reclaimed (TODO B24, call 4).

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

**At 09:00 on 2026-10-27:**

| GPU | Running | Ends |
|---|---|---|
| 0 | `actor-eae 2` (`e4220e3z`, about 83k) and `iface-window 3` (`n94ucnlv`, about 77k) | about 22:30–23:30 on 10-27; about 01:30 on 10-28 |
| 1 | `actor-resfit-shape 2` (`j457zalw`, about 87k); `actor-resfit-shape-wide`'s smoke from 08:03, then its seed 2 | about 22:00–23:00 on 10-27; about 18:30 on 10-28 |

Queued in `order-1026`, in order: `actor-resfit-shape 3`, `actor-eae 3`, `actor-resfit-shape-wide 3`,
`iface-wide-out 3`, `iface-eee 3`. `iface-eee 2` (`mgcifv1y`) finished at 07:58 on 10-27.

**Every run goes to its full 150k with no automatic stop of any kind** (Aden, 2026-10-17 DEVIATION,
which extends the 2026-10-12 AMENDMENT to the divergence stop). Aden stops a run early by hand if one
needs it. New groups go in the watcher's watch-only set.

How the runs are launched and followed:
- **The queue:** `outputs/queue/launchers/order-1026/order_queue.sh`, in tmux `order-queue`, started at
  09:44 on 10-26 with `AD_WORKTREE=/home/aden/projects/equi-resrl-a5b5cc8`. Its log is `order_queue.log`
  beside it, also appended to `../slot_queue.log`. `--check` prints each GPU's occupants and where each job
  would fit.
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
    Tiers 1 and 3 still run. A new commit's first launch runs the full gate: add the commit to the list
    once its Tier 2 passes.
  - The gate prints nothing until each tier ends, so its console is blank for its first minutes, or
    about two hours with Tier 2 (953 tests, 124 minutes at `a5b5cc8`).
  - The render-guard test (`test_every_guard_passes_on_cans_first_demos`) still fails now and then in
    a gate that starts within minutes of a run exiting; the retry has passed it every time. Its failing
    assertion is still not recorded.
  - The scripts are in `outputs/queue/launchers/`, which is gitignored (TODO: move them into `tools/`).
- **The digest** (tmux `queue-digest`; `order-1026/digest.py` every 30 minutes, by `run_digest.sh`). It
  rewrites `outputs/queue/launchers/order-1026/digest.md`: each run's state, step and evals, its early
  speed against the band and its arm's prediction, the gross-failure check of `actor-resfit-shape 2`
  (a FLAG line in `watch.log` once, if every eval at 10k–50k is at or below 0.10), the queue's smokes,
  launches and drops, every FLAG, and the GPUs, RAM and disk. It changes nothing about any run, and stops
  itself when the queue is done. **Read it first when a session opens.**
- **Watchers** (`tools/square_rerendered_watch.py --group <seed group>`, one tmux session per group, from
  the group's pinned worktree), each watch-only: logged, FLAGged and finalized, never signalled.
  - `ad-watch` (`actor-design-1026`, from `equi-resrl-a5b5cc8`);
  - `iface-watch` (`enc-iface-1023`, from `equi-resrl-58d55d7`);
  - `swap-watch` (`arch-swap-1021`) was stopped on 10-26, all its runs exited. `equi-resrl-aec1bdc`
    stays: the launcher scripts name it.
  - The older groups' entries in the watcher still enforce stops that no longer apply.
- **tmux, at 09:00 on 10-27:**
  - `order-queue` and `queue-digest`;
  - `q-actor-resfit-shape-2`, `q-actor-eae-2` and `q-iface-window-3`, one per running job;
  - the watchers `ad-watch` and `iface-watch`;
  - `wandb-cache` (the 12-hourly cache cleanup) and `disk-watch` (`/dev/shm/disk_watch.sh`: the free space
    every 20 s; it logs, and woke a Claude session below 8 GB when one was open).

  The 10-26 cleanup closed the 10 idle sessions and `swap-watch`; their scrollback is in
  `outputs/cleanup-20261026/tmux_scrollback/`, as the 10-24 cleanup's is in `outputs/cleanup-20261024/`.
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
  - **The disk had 21 GB free on 2026-10-26** ("Disk", above). The wandb cache is cleared to 2 GB every
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

**0. Now: `actor-design-1026`** (TODO B22), read about 10:30 on 10-29: whether ResFiT's actor beats ours
(C4) through its wiring, its free parameters, or its lack of a constraint. `enc-iface-1023`'s last runs
beside it (TODO B21).

**1. Next, in Aden's order** (2026-10-27 DECISION; TODO B24 has the to-do list and the calls):
- **The long runs:** ResFiT's agent and `eee` to 300k, to see whether 150k is too short and whether the gap
  closes. Every run above the base was still climbing at 150k.
- **Then O and L** (TODO B23): the nut's pose to the agent, aimed at the critic's missing value of online
  data, and sampled latency, for robustness.
- **Then more tasks, and the tuning** aimed at the critic's signals.

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
