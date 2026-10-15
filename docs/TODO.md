# TODO

Priority order. Each item names the doc section it should update when finished.

Tags: `[SC]` scientific consistency · `[CS]` code simplicity and correctness ·
`[HI]` human interpretability. See the priority ranking in [CLAUDE.md](../CLAUDE.md).

---

## B — Square and the bimanual task

### B17. Which part of ResFiT's layout helps the equivariant residual — NOW (Aden, 2026-10-15) `[SC]`

**The plan, step by step: [docs/plans/2026-10-15-equi-arch-parts.md](plans/2026-10-15-equi-arch-parts.md)**
(local, not in git). Why: ResFiT's residual needs its actor's LayerNorm and carries the equivariant
action space (B16 item 5e), and `arch-wide`, the equivariant residual with ResFiT's layout, was the
only equivariant run not at zero at 10k. Seed group `equi-arch-parts-1015`, all at seed 2 on the
init0 anchor (pairing each with `oav5gk76`); main readout 110k–150k against its 0.332 (better iff
at least 105 of 250).

0. **DONE 2026-10-15:** the wandb artifact cache cleared (21.7 GB; 31 GB free), `vqzx1ir9`'s 150k
   NOTE (0.812 late), and `ln-watch`, `rfe-watch`, `tilt-watch` stopped.
1. **DONE 2026-10-15:** pre-registered (EXPERIMENTS.md, 2026-10-15 PRE-REGISTERED), with the
   AMENDMENT for init0's seed 3 (`equi-residual-long-1012`).
2. **DONE 2026-10-15:** `submit.sh`, `tests/test_sweep_arms.py`, the watcher's group, and their test
   and README rows.
3. **DONE 2026-10-15:** committed, with this item, and pinned in a worktree.
4. **NOW:** wave 1 runs (EXPERIMENTS.md, 2026-10-15 LAUNCHED): `arch-wide` `h216n857`, `actorln`
   `8sgfkskv`, `criticln` `5cj1rbpu`, `heads` `dhhq151o`, watched by `parts-watch`. Wave 2
   (`actor448`, init0 seed 3) waits in `rfe-queue` for the first two slots wave 1 frees, about
   2026-10-16 16:00–16:25 (measured; STATUS, "Running now"); its LAUNCHED entry then.
5. Readouts: NOTEs at 150k; seed 3 of the best one or two better than init0; then by the
   pre-registration's "what follows". Check the disk each session: the wandb cache refills with
   each best upload (STATUS, "Resources").

### B16. Demos rendered by our env, then the equivariant residual on them — DONE through item 5e (Aden, 2026-10-08) `[SC]`

Why: the demos' video codec gave the critic a perfect online/demo cue, and coding the env's frames
to match let equivariant tuned RLPD learn Can (`gkck6eqp`, 0.94 at 10k). Aden chose to fix the
demos rather than the env for the residual runs (EXPERIMENTS.md, 2026-10-08 CODE, "re-rendered
demos").

1. **DONE 2026-10-08: `tools/rerender_demos.py`,** its guards and tests. Every frame after an
   episode's first is rendered one physics substep late, as robosuite's `step()` observes (Aden's
   choice over making the env exact), and observation.state with it.
2. **DONE 2026-10-08: both datasets built and verified** (EXPERIMENTS.md, NOTE, "the re-rendered
   datasets"); the figures are in each `meta/rerender.json`.
3. **DONE 2026-10-08: checked against the env.** No frame statistic tells them from env frames;
   the identity guard allows the GPU's 1-level residue (2026-10-09 CODE, Aden's option A).
4. **Skipped (Aden, 2026-10-09): the validation run on Can.** The Square runs went ahead directly.
5. **DONE 2026-10-10: the equivariant residual and equivariant tuned RLPD on Square**
   (`square-rerendered-1009` seed 1; EXPERIMENTS.md, 2026-10-10 NOTE). The residual collapsed by
   its first eval (`6ng62tpm`). Tuned RLPD learned to 0.50 at 80k, then its critic ran away from
   about 92k and every eval from 100k was 0 (`vc5o977f`). Its seed 2 (`6yh4l5mq`) runs as the
   anchor of item 5b.
5b. **CANCELLED 2026-10-12: stabilizing equivariant tuned RLPD on Square** (`square-trlpd-stab-1010`;
   EXPERIMENTS.md, 2026-10-12 DEVIATION). Seed 1 of `clipq`, `pooledln` and `ln` passed the
   learning check and no critic diverged, nor did the anchor's seed 2; `clipq` converged at 230k
   (0.628). Aden dropped `10x2` and the three seed 2s, so the group has one seed per arm and no
   verdict. Its analysis NOTE is not written. Second-wave candidates, if it is ever resumed: the
   clamp and the pooled norm together; the LayerNorm arm plus per-pixel LayerNorm in the encoders.
5c. **DONE 2026-10-12: the equivariant residual's start and the critic's readiness**
   (`square-residual-start-1011`; EXPERIMENTS.md, 2026-10-12 NOTE). Both arms lost the base policy
   by 10k. The 1e-4 start is not the cause, and a better-informed critic did not stop the drift.
   The penalty named next is held (2026-10-12 FINDING: it would hold ResFiT's useful residual down
   too).
5d. **DONE 2026-10-13: the tilt diagnosis** (plan `docs/plans/2026-10-12-residual-tilt-diagnosis.md`,
   local). No single field of the residual carries the collapse (the field test, 2026-10-12 NOTE),
   and `notilt` ended at 0.200 at 110k–150k, within the equivariant baseline's 0.216–0.272
   (2026-10-13 and 2026-10-15 NOTEs). Its conditional follow-ups (a tilt bound of 0.03, the L2
   penalty) are not triggered: tilt is not the cause.
5e. **DONE 2026-10-15: the 2026-10-13 schedule** (EXPERIMENTS.md, 2026-10-15 LAUNCHED and NOTEs).
   - (1) LayerNorm is needed by ResFiT's residual (`noln` 0 of 150 at both seeds); (2a) the actor's
     is the one (`noln-actor` 0.144, 0.264 late; `noln-critic` 0.776, 0.720; `noln-vit` 0.792 and
     0.800 to 130k). By the registered 10k–30k rule `noln-critic` also "breaks" (37, 34 of 150), but
     the anchor's own seed 2 scored 35 there: the bar sits inside seed spread (recorded as a
     conflict).
   - `limits` (0.800) and `noise` (0.824) hold: the action space does not break ResFiT.
   - The equivariant baseline's 150k reference: 0.272, 0.216; init0 seed 2 0.332; neither back to
     the base.
   - Left: `vqzx1ir9`'s (`noln-vit` 2) 150k NOTE (B17 step 0.2). `noln-vit`'s seed 3 was set aside
     (2026-10-15 DEVIATION).
   - Still open from this item: move `outputs/queue/launchers/` into `tools/` (the queue is empty
     between experiments now); make `preflight.py` record a failing assertion, not just the test's
     name (the render-guard test that failed three gates on 2026-10-12; it has not failed since the
     300 s settle wait).
6. **The regroup** (Aden, 2026-10-09) was for neither arm training by 100k. Tuned RLPD trained, so
   it is not triggered as written, but its residual half stands: **the residual's collapse is a
   different failure** (no value carried to online states; online value negative by 20k), which
   item 5b does not address. Ablations on Can, to set the residual's design: norm (none or
   RegularLayerNorm), the squash, head count, head depth, and N = 8, 4 or no equivariance. Norm
   also on the equivariant residual on Can (`residual_equi_td3_can_config`), since Aden expects it
   may come out differently for a residual. Caveat: Can's base policy is at 0.852, so a residual
   has little room to show a gain. Claude's suggestion (2026-10-10): the equivariant residual on
   Can with the critic that learned Can on both seeds. If it still collapses, the fault is in the
   residual's path, not the critic design.
7. **Later, for a fair stack comparison:** ResFiT's residual on the same demos.

### B15. Fix the render, then new baselines — NEXT (Aden's plan, 2026-10-07) `[SC]` `[CS]`

Why: our `residual` env's mujoco 3.8.1 renders the floor differently from the demos, and
upstream's tuned RLPD learns Can in upstream's env (mujoco 3.3.2) but not in ours (EXPERIMENTS.md,
2026-10-07 FINDING and RESULTS, `upstream-clean-1007`). STATUS has the numbers.

1. **DONE 2026-10-07: the worktrees' records are in main** (EXPERIMENTS.md, CODE, "records in
   main"). 34 records and 34 Hydra folders moved, hashes verified; the three never-synced records
   also copied to `~/equi-resrl-preserve/wandb-offline-only/`. New runs record into main wherever
   they launch from. **Left for Aden: remove the 15 worktrees,** which now hold only code,
   bytecode and an `artifacts` symlink into main's `artifacts/`. Unlink the symlink first, so
   nothing can follow it:
   ```bash
   cd ~/projects/equi-resrl
   for w in ~/projects/equi-resrl-*/; do
       [ -L "$w/artifacts" ] && unlink "$w/artifacts"
       git worktree remove --force "$w"   # --force: 7dae4925 has an untracked empty tea_debug.log
   done
   git worktree prune && git worktree list
   ```
   Fewer worktrees from here (Aden, 2026-10-07): B15 launches from one, at its launch commit, removed
   when its runs end.
2. **DONE 2026-10-07: `tools/gpu_queue.sh` fixed** (same CODE entry). It launches only once the
   line is deleted and 5 GB are free, a log line cut off by a full disk no longer swallows the
   next, and it runs from main, safe to edit while running. The `gpu0.log` append was never
   broken: the EXIT line was glued to the cut-off LAUNCH line (the CODE entry's CORRECTION).
3. **DONE 2026-10-07: mujoco 3.3.2 pinned, and the render on the record and in the gate**
   (EXPERIMENTS.md, CODE, "mujoco 3.3.2"). `pip freeze` moved by that one line. Every run logs
   its `versions`; the online cache key records mujoco; the gate fails on another mujoco and on a
   render that does not match the demos (`tools/render_check.py`), on Can and Square. Its bar is
   2.5x the within-demo gap, not about 2x: the matching render sits at 2.06x on Can. New:
   `read_buffer_caches=false`, for the baselines' "no cache reads".
4. **Re-measure B_sq and B_can,** 500 episodes each with `eval_base_policy`, as before.
5. **DONE 2026-10-07: the equivariant agent logs `critic_diagnostics`** (same CODE entry), and
   the trainer calls it for both agents.
6. **The baselines,** each pre-registered before launch. 150k steps, 2 seeds, 50-episode evals
   every 10k, buffers built in memory with no cache reads:

   | Arm | Task | Config | Overrides |
   |---|---|---|---|
   | ResFiT tuned RLPD | Can | `tuned_rlpd_can_config` | none |
   | Equivariant tuned RLPD | Can | `equi_tuned_rlpd_can_config` | the design below; actor init at its default |
   | ResFiT residual | Square | `residual_td3_square_config` | `agent.critic.num_q=10` (the paper's and `870ws2c2`'s) |
   | Equivariant residual | Square | `residual_equi_td3_square_config` | the design below; `agent.actor.actor_last_layer_init_scale=1e-4` |
   | Equivariant residual, zero init | Square | `residual_equi_td3_square_config` | the design below; `agent.actor.actor_last_layer_init_scale=0.0` |

   - **Aden's review of the arms (2026-10-07):**
     1. Base policies pinned, `v9` (Square) and `v7` (Can, for B_can), not `latest`.
     2. wandb where each kind of run has gone: both tuned-RLPD arms in `robomimic-can-final` /
        `tuned-rlpd` (all 11 earlier tuned-RLPD Can runs; `equi-tuned-rlpd` never held one); ResFiT's
        Square residual in `robomimic-square-final` / `residual-rl` (all 5 earlier, `870ws2c2`
        among them; its config's default project was never used); the equivariant ones in
        `equi-residual-rl`, the config default.
     3. The disk bound is a stack difference, written up for review: ARCHITECTURE.md, "The two
        stacks side by side".
     4. The actor as built; `num_actor_layers` is dead and not set.
     5. Two seeds.
     6. The zero-init arm added. The evidence against zero init is `a3e3zylp` (0.88 → 0.00)
        against `z8yoqylh` (1e-4, 0.82 → 0.94), confounded with host and software, both on
        near-blank images (EQUIVARIANCE.md, "What actually changed"). At the design, a
        zero-initialized residual leaves zero within 4 actor updates with no NaN (checked
        2026-10-07); `equi_tanh` has gradient 1 at the origin.
   - **The ResFiT tuned-RLPD arm's seed 1 is the validation:** its 20k and 30k evals must score at
     least 5 of 100, as `zwmk393l` did. Launch the rest only after it has.
   - **The equivariant design** (Aden, 2026-10-07), on both equivariant arms:
     `equivariance.residual_squash=true` (equi_tanh, times `action_scale`: 0.1 for the residual,
     1.0 vanilla; checked equivariant, STATUS); `agent.critic.num_q=10`;
     `equivariance.num_critic_layers=2`; `equivariance.critic_norm=field_norm`;
     `equivariance.actor_norm=none`; the actor at 2 layers × 128 regular fields, like the critic
     (`num_actor_layers=2`, `actor_degree_channel=128`, `critic_degree_channel=128`, the defaults).
     The tuned-RLPD actor keeps its default init: it outputs the whole action, not a correction,
     so it need not start small (Aden).
   - **A 10-minute smoke run of each equivariant arm first,** for speed and RAM: 10 heads and a
     second critic layer will be slower than the 0.44 s/step of earlier equivariant runs. Measured
     for the tuned-RLPD arm on 2026-10-07: 0.41 s/step over 800 steps (`rhqmlon0`, CODE, "mujoco
     3.3.2"), RAM not measured. The residual arm's is still to do.
   - **Verdicts:** tuned RLPD passes iff the mean of its 110k–150k evals is at least the new B_can;
     a residual arm passes iff that mean beats the new B_sq by at least 2 standard errors.
   - **Resources:** about 40 GiB per Square residual run and about 25 GB per tuned-RLPD run at
     150k, so two at a time, one per GPU. About 8 h per ResFiT arm; 18 h or more per equivariant
     arm. With no cache read, a Square residual run fills its offline buffer in the run: 26.6 min
     (`u0itew7i`, the same entry).

**Updates:** EXPERIMENTS.md, STATUS.md, CLAUDE.md (setup), tests/README.md (the gate check).

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
`git show 8442ba2^:note.txt`. Four unreachable modules remain, deferred to P3.5.

### H2. Review two git stashes `[HI]`

`stash@{0}` on `aabde0f`, titled *"transferring to new compute"*. `stash@{1}` on `d9909ce`. Contents
still unreviewed. That first title is more interesting than it looked: work has moved between the
two workstations before, and that is exactly how `z8yoqylh` came to be invisible from
`ZXP-S-works`. Worth five minutes of `git stash show -p`.
**Updates:** STATUS.md.

### H4. Restore `z8yoqylh` to wandb `[SC]`

It exists only as two local directories on `boce-WS-01` — see the last section of STATUS.md.
`wandb sync ~/projects/equi-resrl/wandb/run-20260522_083555-z8yoqylh` would put it back in the cloud
from local data. Aden's call, because it publishes; recommended, because this project has already
lost two runs to wandb deletion and this is the one that matters.
**Updates:** STATUS.md, EXPERIMENTS.md.

### H5. The 150 KB doc budget no longer fits, and cannot `[HI]`

Needs a decision from Aden. The budget in CLAUDE.md is 150 KB for `CLAUDE.md` + `docs/*.md` +
skills + `settings.json`. After the run-package session on 2026-09-25 it is **201 KB by the
`wc -c` check in CLAUDE.md, roughly 34% over**. That session added 11.7 KB, mostly the Run packages
section in ARCHITECTURE.md and its PROGRESS.md entry; the bimanual session before it added the
EQUIVARIANCE.md, ARCHITECTURE.md and TODO.md material that work required.

The structural problem: **EXPERIMENTS.md is 40.4 KB, 27% of the budget, and is append-only by
project rule** — and rule 5.4 requires a pre-registration for every run, plus a correcting entry
whenever a past claim turns out wrong. It can only grow, so the ceiling will be breached further
every session that launches anything. This session alone added ~10.6 KB there, all of it mandated.

Three options:

1. **Exclude EXPERIMENTS.md from the budget** and apply the ceiling to the docs that are rewritable.
   Most consistent with the append-only rule. Recommended.
2. **Raise the ceiling** to 200 KB and revisit later. Simple, delays the question.
3. **Keep cutting the rewritable docs.** Not recommended — the remaining content is load-bearing,
   and the next cuts would trade correctness for size, which the session-end skill forbids.

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
- Negative control: `use_equivariant_model=False` must **break** equivariance — a test proving the
  tests can fail.
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

### P3.5 Sweep the four remaining unreachable modules `[CS]`

`ablate_equi_obs_encoder.py`, `field_norm.py`, `rotation_transformer.py`, `rotation_utils.py`. All
tracked, so deletion is reversible. Check `field_norm.py` against `equi_rl_utils.py`'s `FieldNorm`
before removing — if the standalone copy is the better one, keep that instead. See H1.

`ablate_equi_obs_encoder.py` is now the most urgent of the four: since the bimanual change it looks
up the normalizer key `action_gripper`, which no longer exists, so wiring it up would raise. The
scalar ablation goes through `ResObsEnc(equivariant=False)` instead.

---

## P4 — Science, after reproduction

Not yet planned in detail. Recorded so it is not forgotten.

- ~~Test the top-down camera hypothesis.~~ **Closed 2026-09-16** — reviewed and accepted as an
  approximation. The cheap diagnostic (render before/after a 45° rotation, compare against a 45°
  image rotation) stays documented in EQUIVARIANCE.md assumption 1 should it ever be needed, but is
  not scheduled.
- **Match the two stacks, then run the scalar ablation.** `use_equivariant_model=False` is meant to
  be the control separating "equivariance is wrong" from "residual RL is misconfigured on this
  path". It cannot serve that role yet: the scalar and equivariant modules differ in six ways at
  once — actor depth, actor dropout, output squashing (`Tanh` vs none), actor return type
  (distribution vs tensor), critic head nonlinearity count (2 ReLUs vs 0), and ensemble
  implementation (`vmap` vs loop). Full table in [STANDARDS.md](STANDARDS.md) rule 3.3.
  **Blocked on matching depth, nonlinearity count, squashing, and clipping semantics**, not on
  compute. Until then the ablation yields an uninterpretable result.
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
