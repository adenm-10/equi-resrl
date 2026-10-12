# Progress

A chronological record of work sessions: what was done, what was decided, what changed. Append a
new entry at the top for each session.

**How this differs from the other docs:**

| Doc | Answers |
|---|---|
| PROGRESS.md (this file) | What have we *done*, session by session? |
| [STATUS.md](STATUS.md) | What is true *right now*? (rewritten in place) |
| [TODO.md](TODO.md) | What should we do *next*? |
| [EXPERIMENTS.md](EXPERIMENTS.md) | What did each training *run* show? (append-only, scientific record) |

Keep entries short. If something needs a paragraph of explanation it probably belongs in one of the
other four docs, with a pointer from here.

---

## 2026-10-12 (afternoon) — ResFiT's residual given the equivariant stack's differences, pre-registered; the field test recorded

**Found**

- The field test at `ls0thn4u`'s 10k checkpoint: no single field of the residual is the cause
  (EXPERIMENTS.md, NOTE). `notilt` fell to 0.02 and 0.00 at 10k and 20k anyway.
- The two stacks' action limits, scaling and noise, measured from the code and the demos.
  - ResFiT's per-axis limits come from the demos' min and max, which outlier frames set.
  - At the 99th percentile, rx and ry are 0.09 and 0.17 (Square), so one tilt radius fits the
    demos well.
  - ResFiT's executed actions are clamped to the demos' range ×1.1, which is [−0.48, 0.24] on rx.
- Parameter counts per component: ResFiT's actor has 13× the equivariant one's parameters and its
  critic 28×, at the same channel widths (EXPERIMENTS.md, the pre-registration's table).

**Done**

- Pre-registered `resfit-equi-diff-1012` and `equi-resfit-arch-1013`.
- `submit.sh` and `tests/test_sweep_arms.py` for their config-only arms.
- The watcher's two new groups.

**Decided** (Aden)

- Priorities `rerendered`, `noln`, `limits`; dropped `noise`, `smallcritic`, `crop`, `noln-actor`.
- `limits` covers all three limits.
- `equi-arch` keeps LayerNorm's own start.
- `limits` may launch overnight without a smoke.

---

## 2026-10-12 — the residual's collapse is a wrist roll; every other run cancelled; the tilt diagnosis launched

**Found** (EXPERIMENTS.md, 2026-10-12 FINDING, NOTE and CORRECTION)

- The equivariant residual's collapse on Square looks like a wrist that rolls after the grasp, in
  all three runs. The training curve's early peak is the base policy, before the residual grows;
  the recovery tracks the residual pulling back; a second mode, grasps that never lift, appears by
  100k (`doevaarp`). Tilt is the suspect, by inference.
- `square-residual-start-1011`: both arms lost the base by 10k. Neither the zero start nor a 50k
  critic warm-up stopped the drift.
- The penalty on the residual's size, named next overnight, was sized 256× too small:
  `debug/dQ_da_mean_abs` is already divided by the batch. The weight that would stop the drift
  (about 1.8) would hold ResFiT's useful residual down too, so the penalty is held.
- ResFiT's rx cap is 0.036 env units, against the equivariant tilt disk's 0.1.
- The eval scored the equivariant residual's Q at the residual alone, not the executed action;
  fixed in `534cc71`. Training was never affected.
- Training and execution combine the residual differently for pairs longer than 1 (TODO H11).

**Done**

- Every other experiment cancelled (Aden): `eaqorqki` stopped at 80.2k, and the waiting tuned-RLPD
  arms dropped (DEVIATION).
- `60fce72`: `resfit-rx-1012` pre-registered, ResFiT's residual with rx at the equivariant stack's
  range, and launched (`fk9ayyco`).
- `be0b7bb`: `residual-tilt-1012` pre-registered and launched (`notilt` `dczf4k7v`, `diag` `ls0thn4u`).
  New code:
  - `equivariance.residual_field_bound`, one bound per action field, so the actor stays
    equivariant;
  - per-field residual and slope logging on both stacks;
  - `checkpoint_interval`, live.

  The smoke run passed: tilt exactly zero, the per-field keys logged, the step-0 checkpoint saved.
- `534cc71`: the eval's Q fix, a tilt readout in every eval, and `tools/residual_field_ablation.py`.
  For the tool, the trainer's agent construction moved verbatim into `build_agent`.
- `outputs/diagnostics/residual-collapse-1012/` (outside git): contact sheets, histograms and the
  blind-scoring script for the rolled share.
- STATUS.md rewritten. TODO: 5b cancelled, 5c done, 5d (the tilt diagnosis) and H11 added.

**Decided** (Aden, on the plan `docs/plans/2026-10-12-residual-tilt-diagnosis.md`, local)

- `resfit-rx-1012` runs from before the new logging, with its evals and videos.
- An absolute bound per field. `notilt` bounds the actor's mean only; its noise is unchanged.
- A field-test verdict naming another field, or none, goes to Aden; the runs stay as they are.
- `diag` to 60k, with a checkpoint every 2k.

**Open**

- Was dropping tuned RLPD's seed 2s and `10x2` intended? Aden's instruction named running
  experiments; those were waiting.
- The doc budget is about 650 KB against 150 KB, mostly EXPERIMENTS.md (TODO H5).

---

## 2026-10-11 — the stabilizer runs past 150k; two runs stopped for RAM; the equivariant residual's start and warm-up, pre-registered

**Found**

- No critic diverged in `square-trlpd-stab-1010` or the anchor's seed 2 through 150k (2026-10-11
  DEVIATION and NOTE). All four seed 1s passed the learning check, and `ln` converged at 160k.
- `6ng62tpm`'s critic ignored its own online successes: its online value was 0.005 over (5k, 10k],
  while its online episodes over 0–5k earned 0.127. ResFiT's residual's online value tracks what
  its online episodes earn. The equivariant residual's actor is not more aggressive than ResFiT's:
  its residual was a third of ResFiT's at 5k, and its actor's gradient 10–25× smaller.
- The equivariant residual runs start the actor's last layer at 1e-4 by an override. The config's
  default, ResFiT's runs and ARCHITECTURE.md's step-0 claim are all 0.0.

**Done**

- `6yh4l5mq` and `lclh933n` stopped by hand past 150k (Aden), to free RAM for the residual runs.
- `square-residual-start-1011` pre-registered: `init0` and `warmup50k`.
  - Rewrote `submit.sh` and `tests/test_sweep_arms.py` for it. The new tests check that each arm
    starts from `6ng62tpm`'s weights, and that `init0`'s residual is exactly zero and learns from
    there.
  - Added the watcher's group.
  - The critic warm-up's console line now prints the critic's value on each buffer.

**Decided**

- Aden: stop pooledln and the anchor's seed 2, and keep `clipq`.
- Aden: both residual arms. Claude chose 50k updates for "longer".

---

## 2026-10-10 — the Square results; every other run cancelled; four ways to keep tuned RLPD's critic bounded

(No entries were written here for 2026-10-08 and 2026-10-09; EXPERIMENTS.md and STATUS.md carry
those sessions.)

**Found** (EXPERIMENTS.md, 2026-10-10 NOTEs and MEASUREMENT)

- On the re-rendered Square demos, the equivariant residual collapsed by its first eval
  (`6ng62tpm`). Equivariant tuned RLPD learned to 0.50 at 80k, the first tuned RLPD to learn Square
  in our trainer, and then its critic ran away (demo value 212; `vc5o977f`), with gradient clipping
  on. The two are different failures: the residual's critic carries no value online, tuned RLPD's
  carries value and then diverges.
- The fixed-env baselines' ResFiT residual arm passes (0.714 pooled), but its seeds differ by about
  4 SE.
- No TD target can exceed 1, on the demos (`tools/target_bound.py`) or online. So the clamp is
  exact, and any critic value above 1 is the critic's error.
- `debug/enc_normalizer_grad_norm` is a running sum of gradients nothing uses (TODO H9).

**Done**

- Every other run cancelled (Aden): `juhq4p36` stopped at 122.6k, `resfit-tuned-rlpd-can` 2 taken
  off the queue (DEVIATION).
- `square-rerendered-1009` amended with a divergence stop before tuned RLPD's seed 2 (`6yh4l5mq`,
  the new group's anchor) launched.
- `equivariance.critic_pooled_norm`: `TrivialLayerNorm`, Aden's P3.4 design, after each head's
  pooling. Tests for invariance and for the bound on Q it gives.
- The watcher runs per seed group and has the divergence stop. `tools/critic_start.py` measures a
  critic's start; tests pin both.
- `square-trlpd-stab-1010` pre-registered (CODE, PRE-REGISTERED): clamp, pooled LayerNorm,
  RegularLayerNorm where ResFiT has LayerNorm, and `gkck6eqp`'s 10 × 2 critic, each one change from
  `vc5o977f`.
- A gate on GPU 0 would have hung in Tier 1, in `tools/rerender_demos.py`'s identity guard. Its
  online env asked for render device 1 outside `CUDA_VISIBLE_DEVICES=0` and died at import, and the
  guard waited on it with no timeout. Every gate since `01f7c8c` had run on GPU 1. Fixed: the device
  is now the trainer's, and a dead env raises (CODE).
- `submit.sh`'s RAM check counts the runs not yet going (`CONCURRENT`). At a fixed 4 it counted the
  anchor twice and would have refused every launch.
- Launched from `6b2acc6`, one after another: `clipq` (`jdu1nkxw`), `pooledln` (`lclh933n`) and `ln`
  (`0avs073d`), beside the anchor (`6yh4l5mq`). Both watchers run from the worktree, and the
  anchor's now has the divergence stop. `10x2` waits for a free slot.

**Decided** (Aden)

- The plan in `docs/plans/2026-10-10-equi-trlpd-square-stabilizers.md` (local):
  - the LayerNorm arm at the actor and critic sites only, one hidden layer per head;
  - the normed arms' starts matched to the anchor's;
  - the divergence stop;
  - only these runs on the machine;
  - the 10 × 2 arm included.
- Matched over 20 seeds rather than the plan's 5 (Claude). The anchor's start varies too much by
  seed for 5 to pin it.

---

## 2026-10-07 (evening) — B15 phases 0–3: records in main, mujoco 3.3.2 on the gate

**Done** (EXPERIMENTS.md, the three CODE entries of 2026-10-07 after the RESULTS)

- **Every run record is in main** (Aden: all experiments in one place, fewer worktrees). 34
  records and their Hydra folders moved out of 14 worktrees and upstream's clone, hashes
  verified; the three never-synced ones are also in `~/equi-resrl-preserve/`. New runs record
  into `CACHE_DIR` wherever they launch from. The 15 worktrees hold only code now; removing them
  is Aden's (TODO B15 has the commands).
- **The queue runner** launches only once its line is deleted and 5 GB are free, and runs from
  main. The "broken" `gpu0.log` append was a misreading: the EXIT line was glued to a cut-off line.
- **mujoco 3.3.2 pinned,** nothing else changed. The gate fails on another version and on a
  render that does not match the demos. The planned "about 2x" bar would have failed a correct
  env: the matching render sits at 2.06x on Can (an even colour offset; the demos are video).
  The bar is 2.5x, set by measuring both versions.
- **`versions` on every run's record, mujoco in the online cache key, and `read_buffer_caches`**
  for runs that must not read a cache (both Square residual arms would have read
  `1d31ed9b`, whose key does not record its normalization path).
- **The equivariant agent logs `critic_diagnostics`.** A new bit-identity test caught a first
  version moving torch's RNG through the actor's `std=0` path, which draws a zeroed random
  number; evals and actor losses take that path in every equivariant run.
- Deleted, approved: upstream's buffer caches (21 GB) and Square's old-render online cache
  (16 GB). 51 GB free.
- **Checked end to end** by two smoke runs from a scratch directory with cache reads off
  (`u0itew7i`, ResFiT residual on Square; `rhqmlon0`, equivariant tuned RLPD on Can): every record
  in main, `versions` on both, 0.115 and 0.41 s a step. Square's offline buffer takes 26.6 min to
  fill without its cache. The gate said GO.

- **The baselines pre-registered** (EXPERIMENTS.md, PRE-REGISTERED, "the fixed-env baselines"):
  five arms, ten runs. `submit.sh` and `tests/test_sweep_arms.py` rewritten for them; the
  equivariance checks now run on each arm's own networks. `tools/baseline_verdict.py` judges them
  and, with no arguments, shows every run's evals so far.

**Decided**

- Aden: fewer worktrees. B15 launches from one, at its launch commit, removed when its runs end.
- The render bar is 2.5x the within-demo gap (above).
- Aden's review of the arms: base policies pinned; wandb where each kind of run has gone; the disk
  bound written up as a stack difference (ARCHITECTURE.md, "The two stacks side by side"); the
  actor as built; two seeds; a fifth arm, the equivariant residual at zero init, since the one
  run that blamed zero init (`a3e3zylp`) was confounded with host and broken images.

**Open**

- Aden's review of "The two stacks side by side".
- The doc budget is about 450 KB against 150 KB, mostly EXPERIMENTS.md (TODO H5).

---

## 2026-10-07 (afternoon) — the render mismatch, and upstream's tuned RLPD learning in its env

**Done**

- **Found why tuned RLPD never learned here** (EXPERIMENTS.md, 2026-10-07 FINDING). Aden asked
  why Q collapses to 0. On eval rollouts Q was about 0 from the first frame, though that frame is
  the same scene as a demo's. Comparing demo first frames with env reset frames: our env's
  mujoco 3.8.1 renders the floor differently from the demos and from mujoco 3.3.2, on the same
  robosuite commit and assets. Can's front camera differs by 0.0995 per pixel (0.007 within
  demos); Square's by 0.019, since its floor is a thin strip. Figures in `outputs/diagnostics/`.
- **Ran upstream's code in its own env** (`zwmk393l`). The README's unpinned install gave torch
  2.14.1, which cannot load torchcodec 0.4.0 or torchrl 0.9.2's extension; the first launch died.
  Repaired with torch 2.7.1+cu128 (the README's own index), recorded as the one deviation
  (EXPERIMENTS.md, LAUNCHED). It reached 0.88 at 20k and 0.80 at 30k on Can (RESULTS,
  `upstream-clean-1007`), against 0 for the same code in our env (`rdoj0ex8`).
- **Recorded the BC-encoder and no-warm-up arms** (RESULTS): both 0 at every eval, both in the
  mismatched env, so their pre-registered next steps are on hold.
- **Checked the equivariant squash** for Aden: `equi_tanh` bounds each rotating pair (dx, dy and
  rx, ry) by its length and each scalar on its own, and the pairs are normalized isotropically.
  54 tests pass.
- **Aden set the plan** (TODO B15): worktrees into main, the render fixed and on the record, a
  validation run on Can, base rates again, then ResFiT and equivariant tuned RLPD on Can and
  residual on Square, with his equivariant design (10 heads, two critic layers with field norm,
  no actor norm, the squash, a 1e-4 last layer for the residual only).
- Comparison tooling, outside git: `outputs/diagnostics/plot_env_compare.py` reads offline wandb
  records directly; `to_webm.sh` converts eval videos for VS Code's preview.

**Open**

- The doc budget is about 430 KB against 150 KB, mostly EXPERIMENTS.md (TODO H5).
- Disk: 14 GB free; upstream's caches (21 GB) can go.

---

## 2026-10-07 — τ and BC did not fix tuned RLPD; upstream's code set up out of the box

**Done**

- **Recorded the τ/BC arms** (EXPERIMENTS.md, RESULTS, "τ 0.005 and a BC term"): all three at 0,
  τ as predicted. The BC term held the actor near the demos, coarsely, with no successes.
- **Aden's reading:** the training return is 0 throughout, so there is no learning signal; are
  the demos bad, or the exploration seeding wrong? The demos are not bad: all 300 succeed, BC
  learns 0.89 from them in this env, and the RL action path matches the residual one. The critic
  does see their reward. What is missing is contrast, two actions with different outcomes at one
  state, which only a rollout that sometimes succeeds supplies. A critic warm-up was tested
  (screen) and cannot supply it.
- **Diffed our fork against upstream's `66262b06`:** the RLPD path does upstream's math. The
  untested difference is the package environment (torchrl 0.7.0 vs 0.9.2, mujoco 3.8.1 vs
  3.3.2).
- **Cloned upstream fresh and built its env from its README** (`~/projects/resfit-upstream`, env
  `resfit-upstream`), and pre-registered two runs in it (EXPERIMENTS.md, PRE-REGISTERED,
  "upstream's code, out of the box").
- **Checked upstream's trainer for uploads:** it creates public Hugging Face datasets and uploads
  its buffer caches only when `HF_OFFLINE_BUFFER_REPO` / `HF_ONLINE_BUFFER_REPO` are set. They are
  not, here.

- **The two warm-ups, and what is faithful,** asked by Aden:
  - **The buffer warm-up** (`learning_starts`, 10k env steps, no learning) and **the critic
    warm-up** (`critic_warmup_steps`, critic-only updates) are separate. Upstream's RLPD trainer
    has the first, with U(±1) actions, and never reads the second's field.
  - **Faithful to the released code** is what we run, field for field. **Faithful to the paper's
    text**, Algorithm 1 without its base policy, would make the warm-up actions U(±0.2) around the
    midpoint, plus the code's 10k critic warm-up. Algorithm 1 itself shows no critic-only phase.
  - **What they do in ResFiT:** both run on the BC policy. Its warm-up rollouts stay near the
    demos and mostly succeed, so the buffer holds near-demo states with different outcomes. The
    zero-initialized residual makes the frozen actor output exactly the BC action, so the critic
    warm-up evaluates the BC policy, with no extrapolation. Without a base policy, neither phase
    has a succeeding policy to work on: the frozen actor is a random network, and its warm-up
    targets query actions no data covers (the 0.74–0.80 demo values before the actor moved).
  - **Their working analog without a base policy** is a policy trained from the demos: BC
    pretraining, or a BC term. That departs from the paper's Tuned RLPD as written.
- **Why the encoder learns from the critic only,** asked by Aden: upstream, like SAC-AE and DrQ,
  detaches the actor's features, because the actor's loss −Q(s, π(s)) could raise Q by changing
  the features the critic sees rather than the action. A BC loss has fixed targets and cannot
  exploit the critic; its costs are interference in the shared encoder and AdamW, and features
  that move under the critic's targets. `bc_backprop_encoder` adds the BC gradient only; the Q
  term stays detached.
- **Made training possible with no warm-up** (EXPERIMENTS.md, CODE, "training can start with no
  warm-up"): updates wait for one online batch. Pre-registered and launched a no-warm-up arm
  (PRE-REGISTERED, "with no warm-up"), Aden's request: demos in the buffer, no random steps.
- **Warm-up alternatives, for later:**
  - a shorter random warm-up;
  - demos only in each batch until the online buffer fills;
  - warm-up rollouts from a BC-pretrained actor;
  - episodes started from demo states.

  Only the last two put successes in the online half; the third is what ResFiT's warm-ups do,
  with the BC policy as the actor's start.
- **Pre-registered and launched the BC-encoder arm** (EXPERIMENTS.md, PRE-REGISTERED, "the BC
  term trains the encoder"), with a test that the flag does what it says.

**Open**

- **The clean-env runs wait on disk:** 17 GB free, 21 GB needed. `local_runs/upstream-rlpd/`
  (21 GB) is approved for deletion; the session's permissions block `rm -rf`, so Aden deletes it.
- The doc budget is far over its 150 KB (about 370 KB, mostly EXPERIMENTS.md), as it was before.

---

## 2026-10-06 (evening) — the BC term wired; τ 0.005 and BC arms launched on Can

**Done**

- **Wired `agent.bc_loss_coef` into our trainer** (TODO B14, done; EXPERIMENTS.md, CODE, "the
  actor's BC term"), as upstream's trainer passes it. The 0 path is unchanged, bit for bit.
- **Pre-registered and launched three Can arms** at Aden's request: τ 0.005, a BC term at 0.025
  (TD3+BC's α = 2.5 balance at the control's Q scale), and both (EXPERIMENTS.md,
  PRE-REGISTERED, "τ 0.005 and a BC term").
- **Which of ResFiT's settings are for the residual alone,** asked by Aden: of the nine fields in
  which upstream's residual and RLPD configs differ, all but τ serve the residual. The zero init,
  the 0.1 cap, the low actor LR, the critic warm-up and the small noise keep the residual near
  the BC policy (the paper's Sec. III-B). Without a base policy "small" means near the middle of
  the demos' action range, not near a good action, so none of them carries over. τ is in the
  general recipe (Sec. III-C) that the paper says Tuned RLPD shares.
- **Aden's notes from a conversation with the paper's author**, the parts that bear on the
  replication:
  - sim used the released benchmark demos, successes only. Checked: all 300 Can MH demos end in
    success. Each ends with 5–6 frames flagged done, all rewarded, in both trainers alike;
  - n-step returns were their most sensitive setting; they suggest sweeping it on a sim task
    with a similar horizon;
  - UTD above 2–4 did not help them;
  - nothing about Tuned RLPD's config. Questions for the author are in Aden's hands: was Fig. 5
    made with the released `rlpd_can_config`, and with what τ, BC term, warm-up and seeds.

**Open**

- **The three arms finish around 16:30–17:00 EDT.** Then a RESULTS entry by the pre-registered rule.
- **`local_runs/upstream-rlpd/`** (about 21 GB) is still on disk, for Aden to delete.

---

## 2026-10-06 (later) — the Can screen: no arm off zero; the failure mechanism measured

**Done**

- **Recorded the screen** (EXPERIMENTS.md, RESULTS, the screen). All six 30k-step arms scored 0 at
  every eval, upstream's own trainer included. None promoted.
- **Measured the mechanism** from the critic diagnostics, all five of our arms:
  - on demo states the critic values the policy's action about 0.024 above the demo's, at every
    log step;
  - the target bootstraps through the policy's action, so that compounds to demo values of
    0.79–0.85, against the demos' true 0.398. A fixed-point formula predicts them to within 0.05;
  - Adam moves the actor at its full rate along the critic's tiny but steady slope, to the bound;
  - offline RL's extrapolation error, settling, not a Q blowup.
- **Answered Aden's recipe checklist** against our tuned RLPD: n-step, UTD, heads, delayed actor,
  target smoothing, shift augmentation, sampling and demos all match the paper and upstream. The
  differences: τ is 0.01, not 0.005; LayerNorm is in the actor as well as the critic (the
  paper's text names the critic, its ablation both); the actor is not zero-initialized.
- **Found `agent.bc_loss_coef` dead in our trainer** (TODO B14; ARCHITECTURE.md).

**Open**

- **The next pre-registration needs Aden.** Most likely a BC-term arm (after B14), with τ 0.005 and
  a critic-only LayerNorm as record-keeping arms. A BC term departs from the paper's Tuned RLPD.
- **`local_runs/upstream-rlpd/`** (about 21 GB) is still on disk, for Aden to delete.
- `main` is ahead of origin and unpushed.

---

## 2026-10-06 — tuned RLPD's ResFiT arms failed; why; a Can screen launched

**Done**

- **Recorded `8ma5alnm` and `lg8qad3l`** (EXPERIMENTS.md, RESULTS): 0.00 at every eval, 0
  training successes, on both tasks.
- **Read the failure from the logs:**
  - the actor hit its Tanh bound within about 100 updates and stayed near it;
  - the critic valued the policy's own rollouts at about 0, but its batch-mean target stayed at
    0.40, which puts the demo half near 0.8 against the demos' true 0.335–0.398;
  - not mainly exploration.
- **Compared with upstream and the paper.** Our config equals upstream's `rlpd_can_config`, at 10
  heads, on every shared field (now a test). Against the paper's text, upstream's RLPD has no
  critic warm-up and uses τ 0.01, not 0.005. Read the paper's Fig. 5: its Tuned RLPD is at about
  0.7 on Can by 25k.
- **Smoke-ran upstream's `train_rlpd_dexmg.py`,** its first run here. Its Can offline buffer
  overwrites 2,456 of 62,456 demo transitions, and it dumps its online buffer at full capacity,
  16 GB.
- **Added `critic_diagnostics`** (Q split by buffer) and **fixed `dQ_da` on ResFiT's agent**
  (CODE). The full suite gave 533 passed.
- **Pre-registered and launched the Can screen,** `tuned-rlpd-can-screen-1006`: upstream's
  trainer, a control, and actor LR 1e-6, critic warm-up 10k and zero-init, alone and together.

**Decided (Aden)**

- **The order:** ResFiT tuned RLPD on Can, then Square, then the equivariant stack on Can, then
  Square. Residual RL after.
- **Screen the residual recipe's three safeguards.** In place of the 0.1 cap, which with no base
  policy cannot cancel Can's midpoint yaw, use zero-init. Add a combined arm.

**Open**

- The disk is at 100%. Upstream's caches under `local_runs/upstream-rlpd/` go when the screen is
  done; deleting the smoke run's 16 GB online cache was declined this session, so it is still
  there.
- `main` is ahead of origin and unpushed.

---

## 2026-10-04 — the image-fix pair collapsed; tuned RLPD built, its ResFiT arms launched

**Done**

- **Recorded the image-fix pair** (EXPERIMENTS.md, RESULTS). Both seeds were at 0.00 at every eval
  from 10k. The executed residual grew to the clip, while the critic rated it above the base
  action.
- **Measured B_can = 0.894 ± 0.014** (MEASUREMENT), the Can bar for tuned RLPD.
- **Built the plan's Phase 2** (`docs/plans/2026-10-02-tuned-rlpd.md`; EXPERIMENTS.md, CODE):
  - a `base_policy=None` mode in `train_residual_td3.py`, for both agents;
  - four configs, `NormalizedVecEnvWrapper`, and the non-residual equivariant encoder and actor;
  - `preflight.py --method tuned-rlpd`, `tools/tuned_rlpd_verdict.py`, and four arms in
    `submit.sh`.
- **Verified the residual path is unchanged:**
  - the deep check of all 17 `CANDIDATES` is identical to `b05736d`;
  - every residual config resolves byte for byte as before;
  - the reference's debug run hit both Square caches.
- **Verified the vanilla path:** new tests in seven files, and four 600-step debug runs: the
  residual reference, and vanilla ResFiT on Square and Can and equivariant on Square. The full
  suite gave 514 passed.
- **Pre-registered `tuned-rlpd-1002` and launched its two ResFiT arms** from `a0353a5`:
  `8ma5alnm` (Square) and `lg8qad3l` (Can), ending around 18:30–19:00 (PRE-REGISTERED,
  LAUNCHED). Gate GO with no warnings; step-0 0/50 on both; 0.11–0.12 s/step.

**Decisions** (Aden's)

- **Option (c):** build all four arms, launch the two ResFiT arms, and decide the equivariant
  arms' design once those report. The image-fix pair's pre-registration held them back.
- The output-bound question (plan rule 2) is folded into that decision. `submit.sh` holds the
  plan's provisional design: `iid_bn` and `equi_tanh`.

**Findings**

- **The ResFiT agent never logs `debug/dQ_da_*`:** its combined action is not a leaf (TODO B13).
- **A trainer's prints are block-buffered when its output goes to a file.** Run debug runs with
  `python -u` to read them live.

**Next**

- **Watch the ResFiT arms;** each takes about 5.5 h. Then RESULTS from
  `tools/tuned_rlpd_verdict.py` and `/analyze-experiment`.
- **Aden decides the equivariant arms' design** (TODO B12). Then an AMENDMENT, and they launch.

---

## 2026-10-02 — act() saw near-blank images in every equivariant run; fixed; the image-fix pair launched

**Done**

- **The machine went down at about 06:50 and rebooted at 14:22,** killing `enc-inner`, `target-fix`
  seed 2, the runners and the watcher.
  - Every queue run is written up: LAUNCHED, RESULTS (none advances; both actor-norm arms collapsed)
    and STOPPED.
  - Both crashed packages were finalized by hand.
- **Fixed the image-scaling bug** (`e15d36c`; EXPERIMENTS.md, CORRECTION and CODE):
  - act() got the env's float images and divided them by 255 again.
  - Two trained checkpoints executed a 0.001 residual against the 0.3 they trained.
  - After the fix the two agree to cosine 1.0000, and the update is bit-identical for all 17
    `CANDIDATES`.
  - New tests in `tests/test_obs_formats.py`, and STANDARDS 4.5: test on the inputs the env
    produces.
- **The gate's intermittent failure was most likely an unseeded test,** not broken equivariance.
  One delta-orthonormal encoder init in 20 gives an output near 1e-8, where the relative error is
  noise. The test is now seeded.
- **Parked the Square ablations, then pre-registered and launched the image-fix pair** (`b667647`):
  `fal5sfot` and `fqo83rvd`, ending around 2026-10-03 06:00.
- **Planned tuned RLPD,** equivariant and ResFiT, on Can and Square
  (`docs/plans/2026-10-02-tuned-rlpd.md`, untracked; TODO B12 holds its decisions).

**Findings**

- **No equivariant result yet measures the method.** No Can checkpoint survives, so the Can
  numbers cannot be checked directly. The best policies remain ResFiT's: `870ws2c2` on Square and
  `msfkjwab` on Can.
- **The pair's first 2.9k steps look like every earlier run's,** as observations only:
  - training success 3/10 and 1/9 (the same seeds before the fix: 4/10, 2/10; `870ws2c2`: 3/11);
  - residual 0.05.
- **What will tell:** before the fix the residual grew to about 0.3 by 20–30k, 4× ResFiT's capped
  0.1. It is now executed.
- **Read earlier entries through the CORRECTION.** Their claims about learning, stability, collapse
  causes and residual sizes come from runs that acted on near-blank images, or from update-batch
  metrics. Their queue states are history: the queue died at 06:50.

**Decisions** (Aden's)

- The Square ablations are parked.
- The image-fix pair runs before tuned RLPD.
- The equivariant tuned-RLPD arms start from the design of the best equivariant residual (the B11
  reference), because its performance at least indicates stability. The ResFiT arms take ResFiT's
  design.
- I read "made vanilla" as the fields ResFiT's own residual and RLPD configs differ in, applied the
  same way to both stacks. The paper describes Tuned RLPD as ResFiT "without a base policy or
  residuals".

**Next**

- **Aden answers the open question:** the equivariant actor's output bound (TODO B12; `equi_tanh`
  recommended).
- **Build the plan's Phase 2 while B11 runs.**
- **When B11 ends:** RESULTS from `tools/square_verdict.py fal5sfot fqo83rvd`. If either run
  collapsed, the equivariant RLPD arms wait for Aden.

---

## 2026-10-01 (later) — Stop at 100k and judge the trend; three normalization arms; the queue runs itself

**Done**

- **What the actor's batch norm changes, measured** on 512 demo observations, with the same weights
  for each norm:
  - **Vision's share of the actor trunk's pre-activation variance:** 99.3% with IID batch norm at
    init (99.5% at `target-fix`'s 10k), 58.7% with none, 97.3% with FieldNorm, and 98.7% with
    InnerBatchNorm. Hypothesis 1 of the entry below stands.
  - **About half the ReLU units are active without batch norm,** so the no-BN actor is not nearly
    affine; hypothesis 2's mechanism is out.
  - **InnerBatchNorm is not IID batch norm under another name.** On Square a field's 8 channels have
    different batch means: 35–85% of the pooled variance at 10k, by site. So InnerBatchNorm's outputs
    are 0.39–0.81× IID's (medians by site), and its gradients pass through its statistics, which
    escnn's `IIDBatchNorm1d` detaches (iid.py, lines 191 and 202).
- **The verdict moved from the eval level to the training trend, at 100k** (EXPERIMENTS.md,
  AMENDMENT). This was Aden's observation, checked on eight Square runs: the two that learned
  raised their training-episode success by 0.4–0.5 per 100k (z 4–5); every run that held BC was
  flat. An eval-level cut at 100k would have failed `870ws2c2` and passed three runs that never
  learned. Pre-registered before the running pair's 10k eval.
- **Code** (EXPERIMENTS.md, CODE): `RegularInnerBatchNorm` for the actor and the encoder; the arms
  `actor-fn`, `actor-inner` and `enc-inner`; `tools/square_verdict.py`; `tools/run_watch.py`.
- **The queue runs unattended until about 2026-10-03:** see STATUS.

**Decided**

- **Norms are tested one location at a time** against `target-fix`: the actor with FieldNorm and
  InnerBatchNorm, the encoder with InnerBatchNorm, and the critic with FieldNorm (`critic-fn`,
  running) and LayerNorm (`critic-ln`, which has data to 110k).
- **Not tested:**
  - batch norm in the critic, which would normalize the stored, target and actor-loss batches each
    with their own statistics;
  - FieldNorm in the encoder, which would scale every pixel's field to unit size.
- **GPU 1's free last slot runs `target-fix` seed 2,** which needs no new code; the no-BN diagnostic
  can replace it.

---

## 2026-10-01 — The actor collapses without batch norm; the ablations restart with it

**Done**

- **The gate failed.** `actor-none` and `critic-fn` went from 0.52 to 0.00–0.02 by 10k and stayed
  there (EXPERIMENTS.md, RESULTS). With the actor's batch norm, `target-fix` held 0.46–0.70.
- **Ruled out,** from the logs and from `target-fix`'s 10k checkpoint:
  - a larger residual: the no-BN residual was smaller at every step;
  - batch norm muting the residual at eval: eval and train residuals agree to 1%.
- **The ablations restarted with `actor_norm=iid_bn`** (AMENDMENT): `critic-fn` and `clipq` first.
  The 11 arms at `iid_bn` are bit-identical to `1e33b8f` on CPU.

**Open: why the actor collapses without batch norm** (TODO B10). The hypotheses, each with a test:
1. **Scale.** Batch norm scaled the visual features up about 7× (input variance 0.021). Without it,
   the trunk weighs proprioception and the base action far more than vision.
   Test: each input block's share of the trunk's pre-activations, in a no-BN actor.
2. **ReLU operating point.** Batch norm re-centred the trunk and policy layers (means about one
   standard deviation). Without it most units stay active, so the actor is nearly affine in its
   inputs, and the critic is too flat in the action to correct it (dQ/da ≈ 8e-5).
   Test: the fraction of active units, and how much of the residual is a linear function of the
   base action.
3. **One damaging direction.** The collapse comes while the mean residual is only 0.1, so it is
   probably one dimension or one systematic push, the gripper being the likeliest.
   Test: per-dimension residuals at eval; evals with single residual dimensions zeroed.

**Candidate fixes**, besides keeping `iid_bn`:
- `layer_norm` in the actor: ResFiT's actor has LayerNorm after every hidden layer, and it is
  per-sample, so there are no running statistics;
- `field_norm` in the actor;
- a norm only at the visual projection, which tests hypothesis 1 directly;
- bounding the residual (`residual_squash`), which limits the harm but not the cause.

---

## 2026-09-30 — The update no longer waits for the GPU; the actor's default is no normalization

**Done**

- **Why the equivariant run was about 4× slower than ResFiT per env step.** Two causes:
  - host syncs: 487 per env step, where the CPU waited for the GPU instead of queueing ahead;
  - compute: the equivariant encoder does about 10× ResFiT's FLOPs per sample.
  - Running more runs in parallel doesn't help: RAM fits two Square runs, and one run now fills a
    GPU.
- **The syncs are gone, with bit-identical training.** In `escnn_fixes.py`, escnn's basis
  expansions hash by identity and `GroupPooling` slices with ints. Metrics are computed only on the
  logged update (`compute_metrics`). The crop offsets are drawn and checked on the CPU, and the
  quaternion is reordered by slicing.
  - Checks: the fingerprint against `ce96bb5` and `1e33b8f`, and a new deep check (every tensor
    after 8 updates and an `act()`) for 14 configs on CPU and 4 on GPU. Everything is identical.
  - Speed: 1.9–2.0× per env step's learning. EXPERIMENTS.md, 2026-09-30 CODE.
- **`actor_norm` defaults to `none`** (Aden's decision). No run had tested `iid_bn` against it, and
  the SO(2) code has no normalization. TODO B9 records what that does to comparisons with
  `target-fix`.
- **New:** `tests/test_sync_free.py`; `tools/agent_fingerprint.py --deep`, with `CANDIDATES`, the
  configs the revised schedule can launch; `tools/bench_update.py`; STANDARDS rule 6.5.
- **Queues paused** at 11:23 for the revised schedule (TODO B0).
- **The sweep was stopped and replaced** (afternoon, Aden's decisions):
  - `target-fix` and `critic-ln` stopped before their verdicts, and recorded as STOPPED;
  - a new pre-registration: `actor-none` first as a gate (it must at least match vanilla BC,
    0.508), `critic-fn` alongside it, then `clipq`, then `squash` and `depth` on whichever passed.
    150k steps and the sweep's rule;
  - `submit.sh` rewritten for these 11 arms, and `tests/test_sweep_arms.py` pinned to them.
- **wandb naming** (Aden's decision): one group per task and method. `equi-residual-rl` holds every
  Square equivariant run, and `robomimic-square-bc` every Square BC run. Runs are named for arm
  and seed, and six past runs were regrouped. STANDARDS 5.2 says so.

**Decided**
- One `torch.cuda.synchronize()` per env step, so `timing/gradient_update` still measures the GPU's
  time in the updates.
- Not in this change: the normalizer's accumulating gradients, bf16, and the `iid_bn` and `inner_bn`
  syncs (TODO H8).

---

## 2026-09-29 — The target critic never updated; the Square sweep launched

**Done**

- **Every equivariant run's target critic was frozen at initialization.**
  - escnn layers in eval mode compute with weights expanded at `.train(False)`, and the target critic
    sat in eval mode from construction.
  - Fixed in `93a8ad4`: it now runs in train mode, as the actor target does.
    `equivariance.frozen_target_critic=True` reproduces the bug.
  - A test copies the critic into the target with tau=1 and checks both behaviours.
  - Warm-up confirms it: the frozen reference's mean TD target stayed at −0.150 to −0.164 over
    10k critic updates, while `target-fix`'s moved from −0.138 to −0.005.
  - EXPERIMENTS.md, 2026-09-29 CORRECTION.
- **Records:**
  - both `equi-square-v1` seeds' RESULTS (L10 0.52 and 0.39);
  - the `num_q` 10 → 2 default change (CORRECTION);
  - a note on `rj70yeid`;
  - the Square base rate, **B_sq = 0.508 ± 0.022** over 500 episodes, about 2.5 SE below the pooled
    step-0 of 0.60, with no systematic cause found.
- **The sweep's code** (`718de4d`):
  - per-network norm fields, with a new `RegularLayerNorm` (ResFiT's LayerNorm, made equivariant);
  - critic and encoder orth init, and the critic's last-layer scale;
  - `use_visual_proj`, and `residual_squash` with `equi_tanh`;
  - `ih_out_channel`, and `num_critic_layers` wired (default 2 → 1, what ran).
  - The agent fingerprint is unchanged at the defaults. Tier 2 grew from 154 to 285 tests: every
    option's symmetry in both modes, and every arm built and updated.
- **Launch tooling** (`93a8ad4`):
  - `submit.sh <arm> <seed> [gpu]`;
  - `tools/gpu_queue.sh`, with a per-GPU queue of `<commit> <arm> <seed>`, each launched from a
    worktree pinned to its commit (`~/projects/equi-resrl-<commit>`);
  - `write_buffer_caches`;
  - the gate looks under `CACHE_DIR`, and `--require-caches` makes a miss block.
- **Launched:**
  - `target-fix` (`0rehtuoq`, GPU 0, `ce96bb5`, 17:58), queued behind the base-rate eval;
  - the BoxCleanup smoke run (`f6n0jtv9`, GPU 1, `ce96bb5`), after which GPU 1's queue starts.
  - Its robot-base probe matched the env, bases (−0.56, ∓0.25), and is committed (`1e33b8f`).
- **The BoxCleanup smoke run passed as a tooling check.** It ran 3k steps at the paper's settings:
  `num_q=10`, `action_scale` 0.2, actor LR 5e-6, init 1e-4, `write_buffer_caches=false`.
  - The ACT reset path survived real successes, with no `NameError`.
  - 229,712 offline transitions were labelled in about 20 min, and no cache was written.
  - **Memory: 59.8 GiB** (PSS) with the online buffer at 10k of 200k: main 35.1, and 9 workers at
    about 2.75 each. A full run projects to about 82 GiB, so it can't share the machine with a
    Square run.
  - **Speed: about 0.69 s/step,** plus about 3.5 min per 50-episode eval. So 500k steps is about
    4 days, not the 9–10 estimated earlier.
  - Evals: 0.72 at step 0, then 0.08, 0.72 and 0.62 at 1k, 2k and 3k. `residual_l1` rose 0.30 → 0.56,
    well past `action_scale` 0.2, since the residual is unbounded. A tooling check, not a result,
    but a reason to settle `residual_squash` before the pilot.
  - wandb `f6n0jtv9` in `equi-resrl-smoke`, left in place; its package is
    `outputs/runs/equi-resrl-smoke/f6n0jtv9/`.

**Decided**

- **Aden:** the normalization arms first, finished by 2026-10-02; the rest on Claude's judgement. No
  `c4`, no dropout or weight-decay arm, no actor-LR arm.
- **Every sweep arm carries the target-critic fix.** `target-fix`, the fix alone, is both the first
  arm and the reference for the rest.
- **The LayerNorm arms use `RegularLayerNorm`, not `FieldNorm`.** Recorded as an amendment before
  either launched.
- **B_sq stays 0.508, as pre-registered,** which makes the screen looser than its calibration.
- **Worktrees are pinned per commit, never moved.** A running trainer can spawn processes that
  import from its tree, so moving a tree under it could mix commits in one run.
- **From the 2026-09-29 non-sweep plan:**
  - A2: the BoxCleanup pair runs one arm at a time, after the sweep or scheduled against it, once
    the smoke run has measured RAM;
  - A3: env seeding stays off;
  - A4: `docs/plans/` stays untracked;
  - A1, the doc budget, is still Aden's (TODO H5).

## 2026-09-27 — Pinning the whole agent, and BoxCleanup prep

**Done**

- **`tools/agent_fingerprint.py` and `tests/test_regression_agent.py`.** They pin the whole
  equivariant agent (parameters, outputs, one TD3 update), identical at `f3e6315` and `b8e1de7`, which
  keeps new-commit Square runs comparable to the reference.
- **The ACT reset crash (B3) is fixed, with a test.** The BoxCleanup BC checkpoint
  `run_xg0w2r0z_best:v7` is backed up and pinned.
- **`scripts/eval_base_policy.py`** measures base rates through the trainer's eval path.
- **Launched `equi-square-v1` seed 2 (`88i0lpc8`)** from the worktree pinned at `b8e1de7`.
- **CORRECTION:** `seed` never reaches the environments.

## 2026-09-25 — Run packages: keep everything locally, upload only the best

Closed Aden's "stop sending giant files to wandb" item. No training-path change.

**Done**

- **Run packages** (`503a9a3`). Both trainers write into `outputs/runs/<project>/<run_id>/` and keep
  it: the existing local saves, `wandb_best/` (an exact copy of the one upload), `wandb_logs/`
  (wandb's files, the `.wandb` record, `metrics.jsonl`) and a `manifest.json` with the commit and
  uncommitted files. Each run uploads only `run_<id>_best`, the best model plus its eval video,
  replaced in place; the best video goes to the existing panel once at the end. Helpers in
  `load_policy.py`; ARCHITECTURE.md "Run packages", STANDARDS.md rule 6.4.
- **The residual trainer saves its best weights** for the first time (`save_residual_model` /
  `load_residual_model` in `utils/checkpoint.py`). Weights and config only, for re-evaluation.
- **Removed the BC trainer's dead `--output_dir` flag.**
- **`tests/test_run_package.py`**, 13 tests against a fake wandb plus a residual save/load round
  trip, wired into Tier 1. Suite 231 → 244.
- **Two smoke runs in the throwaway project `equi-resrl-smoke`**, launched from the branch, both
  passing every check: `8ocaetgn` (BC, Can ACT, 3k steps, bests 0.65 → 0.85) and `rr5ytsrc`
  (residual, Square equi, 1.5k steps, bests 0.40 → 0.60). One surviving upload each, byte-identical
  to `wandb_best/`; one captioned panel video; `metrics.jsonl` matching the record row for row;
  metric names identical to `xg0w2r0z` and `j04yoeui`; the BC upload loads through the unchanged
  `download_policy_from_wandb`, and the residual upload reloads strictly. Tooling checks, so not in
  EXPERIMENTS.md. Afterwards both runs and their uploads were deleted from wandb (the project
  `equi-resrl-smoke` is left empty), and their weights were removed locally. Their packages in
  `outputs/runs/equi-resrl-smoke/` keep the manifest, `wandb_logs/` with the `.wandb` record, and the
  videos, 15 MB.
- **Cleanup, 8.9 GB freed, 67 GB now free.** Trimmed the wandb artifact cache to 1 GB (7.1 GB, mostly
  copies of the BoxCleanup BC run's uploads, which stay on wandb). Moved the dead `0prso90n` run's folder from the repo root to
  `outputs/runs/robomimic-square-final/0prso90n/`, and removed the empty BoxCleanup diffusion
  `bc_run_*` folder. The live Square run's `run_2026-09-25_11-45-19_*` folder is untouched; its old
  code deletes it when the run completes.

**Decided**

- **Overwrite, then delete.** wandb artifacts are immutable, so a new best version is uploaded and
  committed before the older ones are deleted; a failed upload never leaves a run without a best.
- **Persist the package instead of deleting the run folder.** The old end-of-run delete removed the
  only local copy of models and videos from every run that completed, and the project has lost runs
  to wandb deletion before. The kept `.wandb` record restores a run with `wandb sync`.
- **Leave every other panel alone**, so dashboards and metric names stay comparable.
- **One new test file, no new source files.** The helpers live in `load_policy.py`, which both
  trainers already import.

**Found**

- **wandb's `Run.scan_history()` drops the final history row**, which holds the last eval and the
  panel video. `metrics.jsonl` is parsed from the local `.wandb` record instead.
- **escnn saves different state dicts in train and eval mode**; its cached basis buffers exist only
  in eval mode. Residual weights are saved and loaded in eval mode.
- **`load_residual_model` has to map tensors to the agent's device**, because the normalizer
  rebuilds its parameters from the loaded tensors instead of copying into existing ones. Caught by
  the round-trip test.
- **The ACT reset in the residual env wrapper references an undefined name** and would crash
  BoxCleanup on its first success. Also, the base policy is reset only on success, not timeout, and
  diffusion's reset clears every env's queue. TODO B3, ARCHITECTURE.md rough edges.
- **The BoxCleanup BC run finished lower than it looked at 31k.** Ten-eval averages 0.61 → 0.54 →
  0.51, final 0.53; the "≈0.65" in the previous entry does not hold. STATUS.md.
- **The Can buffer caches are gone** (TODO H7), and `docs/papers/SO2ERL.pdf` is missing, although
  the session-start skill points to it.

**Next**

1. Fix the ACT reset crash (TODO B3), back up `run_xg0w2r0z_best`, then wire its id in.
2. When Square finishes: free its caches, pre-register both BoxCleanup arms, launch them together.
3. Independent of both: write up the five undocumented successful runs (TODO B2).

## 2026-09-25 — The encoder goes bimanual, and Square restarts

The `caf83f3` replication reported (both seeds reproduced, first n=2 in the project). Work moved to
a second single-arm task and the first two-arm task.

**Done**

- **Pinned the single-arm encoder before touching it** (`2611adb`). `tests/test_regression_single_arm.py`
  fingerprints the actor and critic features with a sum plus a seeded random projection; the
  projection is what catches a field reordering the sum would miss. Wired into the Tier 1 gate.
- **Generalised `ResObsEnc` to `n_arms`.** Per arm the state is `[eef_pos 3, eef_quat 4,
  gripper_qpos G]` and the action `[delta_pose 6, hand H]`. `enc_ih` became a `ModuleList`, one per
  wrist camera. The regression fingerprint stayed bit-identical throughout, so Can and Square runs
  before and after remain comparable. `actor.py` and `critic.py` needed no changes — they already
  derive every offset from `FieldType.size`.
- **Normalizer, probe and agent followed.** Per-arm normalizer keys sharing one rotation centre;
  `detect_robot_base_xy` → `detect_robot_bases`, which returns *every* base rather than silently
  taking the first; `QAgent` now asserts encoder dims against the env instead of ignoring them.
- **Tests parameterised over arm count.** The `obs_enc` fixture yields both layouts, so all of Tier 2
  runs twice. Suite 74 → 231 passing.
- **`preflight.py` generalised.** `image_keys` and `num_episodes` moved into `TASKS`; added
  `TwoArmBoxCleanup`; cache sizes are now estimated per task and the disk check blocks on the
  estimate rather than a fixed 40 GB.
- **Launched Square** (`equi-square-v1` seed 1), and **relaunched it** after the original died.
- **Started the BoxCleanup ACT BC run**, best 0.64 at 21k.

**Decided**

- **TwoArmBoxCleanup, C8 about the midpoint of the two bases** — as a geometric prior, not an exact
  symmetry. The task fixes object yaw, and the hands are chiral, so six of the eight group elements
  are not physical symmetries. Recorded in EQUIVARIANCE.md so no later reader mistakes it for the
  Can/Square case.
- **`num_episodes=1000` and `total_timesteps=500_000`** for BoxCleanup, to match the paper and stay
  consistent with the other runs.
- **ACT, not diffusion, for the BoxCleanup base policy**, matching the ResFiT authors' own
  simulation launchers.
- **Wait for Square before launching BoxCleanup.** It finishes around 2026-09-29 and frees 22.4 GB,
  which is what BoxCleanup's caches need, plus the second GPU. That lets the baseline and
  equivariant arms run concurrently under identical wall-clock conditions rather than weeks apart,
  and they share caches anyway. Cost is ~3.5 days of an idle GPU 1; the alternative was launching
  into a ~6 GB disk margin and re-running later for pairing. Resolves TODO B5.

**Found**

- **`action_scale` does not bound the residual.** The squash after the actor's final Linear is
  commented out, so `|mu|` is not ≤ 1 (1.27x measured on two arms). The old
  `test_actor_respects_action_scale` asserted a bound that held only by luck of initialisation
  magnitude. The real bound is `equi_clip(..., bound=1.0)`.
- **`pytest tests/` was a trap on this machine.** escnn caches basis tensors per representation, so
  once a GPU module exists a later CPU build dies on a device mismatch — and alphabetical collection
  put the CPU regression module last. Preflight never saw it because it runs Tier 1 in a separate
  subprocess. `conftest.py` now orders that module first.
- **The two BoxCleanup arms share their buffer caches.** The cache key has no equivariance term, so
  the equivariant run costs no extra disk.
- **`ablate_equi_obs_encoder.py` is imported by nothing** and still references a normalizer key that
  no longer exists. Left under the freeze; queued for P3.5.
- **The BoxCleanup BC policy's evaluation is unstable, and it matters.** 31 evals of 100 episodes
  each: a clear climb from 0.26 to roughly 0.65, but oscillating +/-0.15 the whole way. The last six
  are 0.68 0.53 0.67 0.66 0.69 0.53. The recorded peak of **0.74 at step 24k is an outlier draw**,
  not the policy's true rate. `base_policy.wt_type="best"` selects on exactly this noisy signal, so
  the saved checkpoint is biased high — which means the residual run's step-0 will look lower than
  the BC "best" for reasons that are not a pipeline bug. Read step-0 as the base rate, and put the
  whole sequence in the pre-registration.

**Next**

1. When BC finishes: verify the artifact, set `base_policy.wandb_id` in both the config and
   `preflight.TASKS`, and back the checkpoint up.
2. When Square finishes: free its caches, pre-register both BoxCleanup arms, and launch them
   together.
3. Independent of both: write up the five undocumented successful runs (TODO B2).

## 2026-09-16 — The lost run was not lost, and the target was wrong

Started as a session to move the previous session's setup onto a second workstation. Turned into
recovering `z8yoqylh` and discovering the reproduction had been aimed at the wrong commit.

**Done**

- **Moved the repo setup to `boce-WS-01`** (`10.188.60.163`, user `aden`, `~/projects/equi-resrl`).
  There was already a GitHub remote sitting at `caf83f3`; pushed the 14 unpushed commits plus tags,
  pulled there. Docs, `.claude/` skills, 148 tests, `preflight.py`, `submit.sh` all in place; gate
  passes.
- **Recovered `z8yoqylh`** at `boce-WS-01:~/projects/equi-resrl/wandb/run-20260522_083555-z8yoqylh`,
  173 MB, config and logs intact. Backed up to `~/equi-resrl-preserve/`.
- **Corrected the record** in EXPERIMENTS.md (append-only, new section) and repointed
  `preflight.py` at `z8yoqylh`'s environment. Tagged `repro-z8yoqylh-actual` at `7dae4925`.
- **Launched the 3-seed replication** on `boce-WS-01`, seed-group `equi-repro-7dae4925`: seeds
  2114495708 / 1 / 2 as `9qgsaxsr` / `4sseggkv` / `dxbept4u`. Pre-registered before launching.
- **Killed `m0ylcivk`**, the `caf83f3` run started earlier in the day, once it was clear it was not
  the reproduction.

**Findings**

1. **`z8yoqylh` ran at `7dae4925`, not `caf83f3`** — from its own `wandb-metadata.json`. That is
   the same commit as the three collapsed runs, so `7dae4925` is not an architectural failure, and
   **the `FieldNorm` hypothesis is dead**: `FieldNorm` was in the critic head of the run that
   reached 0.94.
2. **The launch carried an override no doc mentioned:**
   `agent.actor.actor_last_layer_init_scale=1e-4`, against a default of 0.0 that sits in the config
   with `1e-4` commented out beneath it. `z8yoqylh` and `a3e3zylp` share a commit and a config and
   differ in only this and the host/stack — one reached 0.94, the other 0.00.
3. **Host correlates perfectly with outcome across the whole project.** Every collapsed equivariant
   run ran on `ZXP-S-works`; the one success ran on `boce-WS-01`, whose stack is torch 2.6.0 /
   torchrl 0.7.0 / escnn 1.0.11. `preflight.py` had been validating the environment against
   `a3e3zylp` — a run that collapsed to 0.00.
4. **The successful run had robot-base centering disabled.** At `7dae4925` every call site is
   commented out, so it rotated about the world origin — the case EQUIVARIANCE.md calls a silent
   symmetry break. It reached 0.94 anyway. The feature is active at HEAD, so **HEAD is not the
   architecture that produced 0.94.**
5. Recovered specifics: seed **2114495708**, best **0.94** (not ~0.92 from the screenshot), runtime
   **47.3 h** measured. `enc_degree_channel=32` was the one assumption guessed right.
6. Two path traps in the worktree setup: `_CACHE_ROOT` defaults to the *current directory*, and the
   BC artifact downloads into `./artifacts/`. Handled with `CACHE_DIR` and a symlink; without them
   it is a 21 GB rebuild.
7. Each run holds ~25 GB resident, not ~21 GB. Three is the ceiling on `boce-WS-01`, not a
   comfortable three.

**Decisions**

- **Replicate before explaining.** Three seeds at the recovered config first; isolating init-scale
  versus stack is deferred to R11. Rationale: with two candidate causes and n=1, isolating first
  risks spending ~94 h explaining a result that may not replicate.
- **Three seeds in parallel over sequential.** Sharing GPU 1 costs 1.85x (measured, 79.7 versus
  145-150 ms/update) but gets all three in ~87 h against ~94 h sequential. The replication arm runs
  alone on GPU 0 so the number that matters most is the cleanest.
- **Left the wrong tag in place.** `repro-z8yoqylh-base` points at `caf83f3`; it is already pushed,
  so it was superseded rather than moved.
- **Freeze stays on** `equi_off_policy/`, with a new reason: three runs are in flight and the
  analysis compares them against `z8yoqylh`.

**Next**

R7 — the 10k and 20k evaluations, where every past collapse was already unambiguous. Then R8
(config caveats at `7dae4925`, which blocks the results entry) and R10 (`/analyze-experiment`, then
lift the freeze).

---

## 2026-09-16 — Phase 2 + launch readiness

Built the pre-submission gate, the submit script, and a reconstruction of `z8yoqylh`'s config. No
changes under `equi_off_policy/`; the freeze held.

**Condensed 2026-09-16.** Several findings in this entry were aimed at `caf83f3` and are superseded
by the recovery of `z8yoqylh` later the same day — see the top entry and the correction section in
[EXPERIMENTS.md](EXPERIMENTS.md). What each one got right and wrong is noted inline.

**Done**

- **`tests/` — 148 tests across 7 files.** Independent group-action implementation, layout checks
  against the physics, actor equivariance, critic invariance, `equi_clip` commutation, no-op pins.
- **`preflight.py`** — three-tier gate (static / GPU construction / resources), exiting non-zero so
  launches can be blocked on it. **`submit.sh`** — single rewritable launcher.
- Tagged `repro-z8yoqylh-base` at `caf83f3` and pre-registered the reproduction with six
  assumptions. **Both aimed at the wrong commit.** The tag is superseded by
  `repro-z8yoqylh-actual` at `7dae4925`; the pre-registration by the correction entry.

**Findings**

1. **The equivariant implementation is correct.** Representations match the physics at all 8 group
   elements, including the interleaved rotation-column regrouping. Actor equivariance and critic
   invariance hold to `< 1e-4`. **Still true and still important** — it is why the collapses were
   never a broken symmetry. The inference drawn from it at the time, that this strengthened the
   `FieldNorm` explanation, is dead.
2. **`a3e3zylp` ran on config defaults, not the README overrides** — `n_step=3, gamma=0.99,
   stddev=0.05, action_scale=0.1`, with `algo.prefetch_batches=4` the only algorithmic override.
   Still true, and `z8yoqylh`'s recovered config confirms it used the same defaults. Copying the
   README's `n_step=5 gamma=0.995` would also have invalidated 21 GB of cache, since both are
   cache-key fields.
3. **Chose `enc_degree_channel=32`** over 16. Correct by luck: `z8yoqylh` ran at 32, but at
   `7dae4925`, where the default was 16 — so it was set explicitly, not inherited.
4. **The 45° vision equivariance error is ≈0.32 relative**, much larger than expected. Exact at
   right angles, a third off at diagonals, because rotating a pixel grid by 45° needs resampling.
   Recorded in EQUIVARIANCE.md.
5. **`train_bc_dexmg.py` parses CLI args at import time** (line 217, outside the `__main__` guard at
   905), so it cannot be imported. Skipped in the import test with a dedicated test documenting the
   defect. Not on the reproduction path.

---

## 2026-09-16 — Pre-flight audit for the reproduction

Checked the machine and wandb rather than assuming.

**Condensed 2026-09-16**, for the same reason as the entry above. **Finding 1 of this audit was the
`FieldNorm` hypothesis, and it is now known false** — `z8yoqylh` ran at `7dae4925` with `FieldNorm`
present in the critic head. The audit's error was reasoning about which commit the run used from the
commit titles and the local evidence, rather than from the run's own metadata, which existed the
whole time on a machine nobody checked.

**Ready, verified**

Environment on `ZXP-S-works` matched `a3e3zylp` exactly. BC base policy needed no retraining. Both
Can buffer caches present by hash, so startup is minutes and there is no parallel-seed write race.

**Findings that still stand**

1. **`msfkjwab`, the Can baseline, is deleted from wandb.** It survives only as
   `ZXP-S-works:wandb/run-20260226_143630-msfkjwab/`. It cannot be re-pulled.
2. **Two 300k runs existed only in wandb:** `czjqzg0b` (2026-05-07) and `qe2by47h` (2026-05-16),
   both collapsed. Now in EXPERIMENTS.md.
3. **RAM is the binding constraint, not VRAM.** Correct in kind, wrong in size: the estimate was
   ~21 GB per run, and the measurement on `boce-WS-01` is ~25 GB.
4. **There was no launch script for Can residual RL.** Past runs were launched from commands kept
   in `note.txt` — which is why `z8yoqylh`'s override was invisible until its metadata was read.
5. `x2w6phzf` used **N=12**, the only run that did.

---

## 2026-09-16 — Phase 1: standards, skills, allowlist

Documentation and tooling only. No changes to the training pipeline; the reproduction freeze held.

**Done**

- **[STANDARDS.md](STANDARDS.md) written** — 28 rules across configuration, equivariant code, the
  two agent families, code hygiene, running experiments, and logging. Every rule derived by reading
  the existing code, each citing an exemplar file or a live anti-pattern.
- **Three skills finalized** in `.claude/skills/`: `session-start`, `session-end`,
  `analyze-experiment`. Made repo-specific rather than generic — each carries the project traps that
  separate a useful answer from a misleading one (step-0 eval is the base policy; check the
  provenance table before believing a run config; `wandb-summary.json` holds only the last value).
- **`.claude/settings.json`** — read-only Bash allowlist to cut permission prompts.

**Findings** — all four became STANDARDS.md rules, so they are recorded there rather than restated
here.

1. **The scalar ablation is not a controlled ablation** (rule 3.3). `use_equivariant_model=False`
   is supposed to isolate equivariance, but the two stacks differ in six ways at once — actor depth,
   actor dropout, output squashing, return type, critic head nonlinearity count, and ensemble
   implementation. Running it today gives an uninterpretable result, so P4's ablation is blocked on
   matching the stacks, not on compute. The most consequential thing Phase 1 turned up, and still
   open.
2. **Layer count and normalization are each configured twice** (rule 1.4) — the baseline path reads
   `agent.*`, the equivariant path reads `equivariance.*`, so setting the wrong one of a pair looks
   reasonable and does nothing.
3. **`train/` and `training/` were both in use** for nearly the same thing (rule 6.1).
4. **`layer_norm` in the baseline actor is an int used as a 3-way enum** (rule 1.5), undocumented
   and only discoverable by reading `build_fc`. Upstream code; do not copy it.

**Next** — recorded as written, and superseded the same day: tag `caf83f3`, reconstruct the lost
config, fix run naming, then the smoke gate and 3 seeds. The commit was wrong; see the top entry.

---

## 2026-09-16 — Phase 0: documentation from scratch

First session of the documentation effort. No code changes to the training pipeline.

**Done**

- Created this doc set: CLAUDE.md, ARCHITECTURE.md, EQUIVARIANCE.md, EXPERIMENTS.md, STATUS.md,
  TODO.md, PROGRESS.md, and a STANDARDS.md stub.
- Backfilled EXPERIMENTS.md from all 220 local wandb run directories. 23 runs got past ~1000 steps;
  5 reached the full 300k. Success rates recomputed from the `✓`/`✗` markers in each run's
  `output.log` rather than read off wandb, so they are reproducible from disk.
- Built the config-field provenance table in ARCHITECTURE.md by reading HEAD line by line.
- Housekeeping: papers → `docs/papers/` (gitignored, 18 MB); 24 root `run_2026-*` dirs →
  `local_runs/`; `.gitignore` extended.

**Findings that changed the plan**

1. **Six `EquivarianceConfig` fields are silently no-ops** at `caf83f3` — `num_actor_layers`,
   `num_critic_layers`, both `dropout`s, and the critic's `use_norms` / `use_orth_init`. They are
   accepted, logged to wandb, and ignored. So every past run's recorded hyperparameters misdescribe
   the network that trained: `a3e3zylp` was launched with `num_actor_layers=3` and that did nothing.
   This is why the reproduction freeze exists and why the cleanup is gated behind bit-identity
   tests. Full table in ARCHITECTURE.md.
2. **Critic group pooling is present at HEAD** ([critic.py:35-37](../resfit/rl_finetuning/equi_off_policy/rl/critic.py#L35)).
   Confirms Aden's recollection — the reproduction can proceed from `caf83f3` without recovering
   code from the other machine.
3. **Every equivariant run that exists locally collapses.** 14 runs, 5 commits, one failure mode:
   the residual saturates at `action_scale` and success falls to zero. `x2w6phzf` is the
   informative one — Square, 0.44 → 0.72 → 0.00 — which rules out a representational-capacity
   explanation and points at stability.
4. **43 of 197 failed runs died with escnn types in the traceback.** Field-type mismatches were
   being found by launching training runs. Motivated moving construction-time layout assertions to
   the front of Phase 2.
5. **The agentview camera is tilted 44.9°**, so vision equivariance is approximate, not exact.
   Reviewed and accepted the same day — see decisions below.
6. **Can is a weak task for the core question.** Base policy at 0.92, ceiling at 1.00. Square
   starts at 0.52 and discriminates far better. Can is still right for the reproduction because
   that is what the lost run used.

**Decisions** — each now lives in the doc that owns it; pointers only, to avoid drift.

- **Reproduction freeze** on `resfit/rl_finetuning/equi_off_policy/` — priority 1 over priority 2,
  so a reproduction failure cannot be confused with a change we made. Still in force, with an
  updated reason; see STATUS.md.
- **Tilted camera accepted** as a useful approximation — EQUIVARIANCE.md assumption 1.
- **`rl_utils.py` deleted**, and **`note.txt` / `log.txt` untracked** — TODO H1/H3. The notes files
  are recoverable via `git show 8442ba2^:note.txt`.
- **BC retraining is not needed.** Policies for Can, Square and TwoArmCoffee are cached under
  `artifacts/`. Can and Square use **Diffusion** policies, not ACT — verified from the cached
  `policy/config.json`, and worth knowing because the README example and the upstream paper use ACT.

**Next**

R1–R3 in TODO.md: tag the reproduction baseline, reconstruct the lost config with assumptions
written down, and fix the run-naming convention before launching 3 seeds.
