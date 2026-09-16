# Experiment log

**Append-only. Never edit or delete a past entry.** If an entry turns out to be wrong, add a new
entry that corrects it and leave the original in place. The point of this file is to be an honest
record of what was tried, including what failed, so the same dead ends are not walked twice.

## How to read a number in this file

- **step-0 eval** — the evaluation at global step 0. Because
  `agent.actor.actor_last_layer_init_scale = 0.0`, the residual starts at exactly zero, so this
  number is **the base BC policy alone**. It is the bar each run must clear.
- **best** — the highest evaluation success rate reached at any point.
- **final** — the success rate at the last evaluation.

A run whose `best` equals its `step-0` learned nothing useful: its best moment was before training
started. Several runs below look successful if you only read `best`.

- Evaluations are 50 episodes each, every `eval_interval_every_steps` (10,000).
- Success rates below were recomputed from the `✓`/`✗` markers in each run's local `output.log`,
  not read off wandb, so they are reproducible from disk.

## Entry schema

Every new entry uses these fields:

```
### <date> — <run_id> — <one-line title>

- Commit / Config / Overrides / Task / Seed / Seed-group / wandb
- Steps completed / step-0 eval / best / final
- Config caveats:   which logged hyperparameters were no-ops at this commit
- Outcome:          what happened
- Interpretation:   what it means
- Changed vs. previous: what was different
```

**The "config caveats" field is required.** Several `EquivarianceConfig` fields are logged to wandb
but ignored by the code, and which ones varies by commit. Without this field the log repeats the
same false claims the wandb configs make. See the provenance table in
[ARCHITECTURE.md](ARCHITECTURE.md).

---

## Summary of everything so far

220 local wandb run directories, of which 23 got past a few hundred steps. The rest are launch
failures and immediate crashes, summarized in bulk at the bottom.

**Seven runs reached the full 300k steps.** Two of them exist only in wandb, found 2026-09-16.

| Date | Run | Task | Equi | Commit | step-0 | best | final | Verdict |
|---|---|---|---|---|---|---|---|---|
| 2026-02-24 | `870ws2c2` | Square | no | `debc9b6c` | 0.52 | **0.90** | 0.90 | Baseline works, +0.38 over base policy. |
| 2026-02-26 | `msfkjwab` | Can | no | `807efd65` | 0.92 | **1.00** | 1.00 | Baseline works, +0.08, saturates. |
| 2026-04-02 | `x2w6phzf` | Square | yes | `a948786d` | 0.44 | 0.72 | **0.00** | Improved, then collapsed. Used N=12. |
| 2026-05-07 | `czjqzg0b` | Can | yes | `7dae4925` | 0.88 | 0.88 | **0.00** | Collapsed. wandb-only. |
| 2026-05-07 | `a3e3zylp` | Can | yes | `7dae4925` | 0.88 | 0.88 | **0.00** | Collapsed immediately, never recovered. |
| 2026-05-16 | `qe2by47h` | Can | yes | `7dae4925` | 0.82 | 0.82 | **0.00** | Collapsed. Latest equi run. wandb-only. |
| *(unknown)* | `z8yoqylh` | Can | yes | ? | n/a | ~0.92 | ~0.92 | **Succeeded. Data lost, see below.** |

**Three runs at `7dae4925` - three seeds, three configs - all ended at exactly 0.00.** That is an
architectural failure, not seed variance, and it is why the difference between `7dae4925` and
`caf83f3` matters so much.

**The short version of the science so far:** the ResFiT baseline reproduces on both Can and Square.
Every equivariant run that exists on this machine either collapsed to zero or never got off the
ground. One equivariant run *did* work — it is the one whose data was deleted, and reproducing it
is the current top priority.

---

## The lost run

### unknown date — `z8yoqylh` — first equivariant run to train successfully

- **Commit:** unknown. Believed to be `caf83f3` or later.
- **Config:** `residual_equi_td3_can_config`
- **Task:** Can · **Seed:** unknown · **Seeds:** 1
- **wandb:** `aden-mckinney10-university-of-central-florida/robomimic-can-final/runs/z8yoqylh` — **deleted**
- **Steps:** 300,000
- **step-0 eval:** not recoverable · **best:** ≈0.92 · **final:** ≈0.92

**Data status: lost.** The wandb run was deleted. Its local directory is on a different computer
and is not retrievable. No local `wandb/run-*-z8yoqylh/` exists here. Everything below is
**transcribed by eye from a screenshot of the wandb plots** and should be treated as approximate
and unverifiable.

- `eval/success_rate` oscillates roughly 0.80–0.925 across training, ending near 0.92.
- `eval/mean_successful_episode_length` stays roughly flat in the 147–162 band for all 300k steps.

Paired against `msfkjwab` as the baseline in the same screenshot:

| | equi (`z8yoqylh`) | baseline (`msfkjwab`) |
|---|---|---|
| success rate, range | 0.80 – 0.925 | 0.875 – 1.00 |
| success rate, final | ≈0.92 | 1.00 |
| episode length, start → end | ≈155 → ≈152 (flat) | ≈150 → ≈105 (falling) |

- **Config caveats:** unknown, but if the commit was at or after `caf83f3` then
  `num_actor_layers`, `num_critic_layers`, `dropout`, and the critic's `use_norms` / `use_orth_init`
  were all inert.
- **Outcome:** trained stably to 300k steps without collapsing. The first equivariant run to do so.
- **Enabled by:** *(originally recorded as "adding group pooling to the critic head". **This was
  wrong** — corrected 2026-09-16 in "What the wandb audit changed" below. Group pooling was already
  present at `7dae4925`, where three runs collapsed. The real change at `caf83f3` was removing
  `FieldNorm` and the ReLUs from the critic head. Original claim left here per the append-only
  rule.)*
- **Interpretation:** two observations, in increasing order of interest.
  1. Equivariant success rate is below baseline (≈0.92 vs 1.00). Given that the equivariant trace
     oscillates by ±0.06 within a single run, a 0.08 gap from one seed each is **not** separable
     from seed noise. No conclusion should be drawn from it yet. This is the reason the rerun uses
     at least 3 seeds.
  2. The episode-length divergence is the more interesting signal. The baseline learns to finish
     the task faster over training (≈150 → ≈105 steps); the equivariant version does not move at
     all (flat ≈152). The two agents are improving along different axes. If that holds up across
     seeds, it is a sharper research question than the success-rate gap — it suggests the
     equivariant residual is finding a different kind of solution, not just a slightly worse one.
- **Next action:** recreate from HEAD with ≥3 seeds. See [TODO.md](TODO.md).

---

## Full-length runs that exist locally

### 2026-02-24 — `870ws2c2` — ResFiT baseline reproduced on Square

- **Commit:** `debc9b6c` · **Config:** `residual_td3_square_config` · **Overrides:** `algo.prefetch_batches=4`
- **Task:** Square · **Seed:** 3433789739 · **Seeds:** 1
- **Steps:** 300,000 · **31 evals**
- **step-0:** 0.52 · **best:** 0.90 · **final:** 0.90
- **Config caveats:** none relevant — non-equivariant path, so the `EquivarianceConfig` no-ops do not apply.
- **Outcome:** worked. Residual RL lifted the base policy from 0.52 to 0.90 and held it.
- **Interpretation:** the baseline algorithm reproduces on Square. This is the codebase-is-working
  check, and it passed. Square is the more informative of the two tasks because the base policy
  leaves a lot of headroom.

### 2026-02-26 — `msfkjwab` — ResFiT baseline reproduced on Can

- **Commit:** `807efd65` · **Config:** `residual_td3_can_config` · **Overrides:** `algo.prefetch_batches=4`
- **Task:** Can · **Seed:** 2968076544 · **Seeds:** 1
- **wandb:** `.../robomimic-can-final/runs/msfkjwab` — **deleted** (confirmed 2026-09-16).
  Survives only as the local directory `wandb/run-20260226_193630-msfkjwab/`. **Do not delete it —
  it is the only copy of the Can baseline.**
- **Steps:** 300,000 · **31 evals**
- **step-0:** 0.92 · **best:** 1.00 · **final:** 1.00
- **Config caveats:** none relevant.
- **Outcome:** worked. 0.92 → 1.00.
- **Interpretation:** the baseline works on Can too, but the base policy is already at 0.92, so
  there is only 0.08 of headroom and the metric saturates. **Can is a weak task for measuring
  whether equivariance helps** — a ceiling at 1.00 with a floor at 0.92 leaves almost nothing to
  detect. Square, with its 0.52 starting point, discriminates far better. Worth weighing when
  choosing where to spend compute after the reproduction.
- This is the baseline curve in the `z8yoqylh` comparison screenshot.

### 2026-04-02 — `x2w6phzf` — equivariant Square: learned, then collapsed

- **Commit:** `a948786d` · **Config:** `residual_equi_td3_square_config` · **Overrides:** `algo.prefetch_batches=4`
- **Task:** Square · **Seed:** 3923189280 · **Seeds:** 1
- **Steps:** 300,000 · **31 evals**
- **step-0:** 0.44 · **best:** 0.72 · **final:** 0.00
- **Config caveats:** at `a948786d`, critic group pooling was not yet present.
- **Outcome:** rose from 0.44 to 0.72, then fell to 0.00 and stayed there.
- **Interpretation:** the most informative failure in the log. The equivariant residual *can* learn
  — it gained 0.28 over the base policy — and then destroyed itself. That rules out "equivariant
  network cannot represent a useful residual" and points at a stability problem. The collapse
  pattern matches the other pre-group-pooling runs.
- **Changed vs. previous:** first full-length equivariant run.

### 2026-05-07 — `a3e3zylp` — equivariant Can: immediate collapse, 41 hours

- **Commit:** `7dae4925` · **Config:** `residual_equi_td3_can_config` · **Overrides:** `algo.prefetch_batches=4 equivariance.num_actor_layers=3`
- **Task:** Can · **Seed:** 834536592 · **Seeds:** 1
- **Steps:** 300,000 · **31 evals** · **wall clock:** ~41 h (148,669 s)
- **step-0:** 0.88 · **best:** 0.88 · **final:** 0.00
- **Config caveats:** **the `num_actor_layers=3` override did nothing.** At `7dae4925` the actor's
  layer loop was already commented out, so the network had one hidden layer regardless. The wandb
  config for this run records `num_actor_layers: 3` and is wrong. `use_norms` and `use_orth_init`
  on the critic were also inert.
- **Outcome:** 0.88 at step 0, then 0.00 at all 30 subsequent evaluations. Never recovered.
- **Diagnostic numbers at step 300k:** `residual_l2 ≈ 0.211` against `action_scale = 0.2` — the
  residual was pinned at its maximum magnitude for essentially the whole run. `dQ_da_mean_abs ≈
  2.2e-4`, i.e. the critic's gradient with respect to the action was almost flat, so the actor was
  ascending a nearly featureless surface. `critic_loss ≈ 0.0035` and stable, so the critic was
  fitting *something* well — just not something with useful action-dependence.
- **Interpretation:** the residual saturated at its clip boundary and overwhelmed a base policy
  that was already at 0.88. Two candidate causes, not yet separated: the critic provides no usable
  action gradient (consistent with `dQ_da`), or the residual is equivariant in a frame that does
  not match the observation frame. 41 hours for a null result is the direct argument for the smoke
  gate and the equivariance tests.
- **Superseded by:** `z8yoqylh`, which added critic group pooling and did not collapse.

---

## PRE-REGISTERED, NOT YET LAUNCHED

### Reproduction of `z8yoqylh` — equivariant residual TD3 on Can, 3 seeds

Written before launching, per STANDARDS.md rule 5.4. Fill in results after.
**Seed-group id: `equi-residual-rl`.**

| | |
|---|---|
| **Commit** | launched at `d446171`. Everything under `resfit/` is **byte-identical** to `caf83f3` (tagged `repro-z8yoqylh-base`) — `git diff caf83f3 HEAD -- resfit/` is empty. The eight commits in between are docs, tests and tooling only, so the training code is the tagged baseline. |
| **Config** | `residual_equi_td3_can_config` |
| **Overrides** | `algo.prefetch_batches=4` only, plus naming and `seed` |
| **Launcher** | `./submit.sh <seed> [gpu]` |
| **Task** | Can · **Seeds** 1, 2, 3, run sequentially |
| **wandb** | project `robomimic-can-final`, group `equi-residual-rl`, names `equi-residual-rl-{1,2,3}` |
| **Expected runtime** | **~58 h per seed** (revised 2026-09-16, see below). 3 seeds sequential is ~7.3 days. |

**Hypothesis.** At `caf83f3` the equivariant residual agent trains to 300k steps on Can without
collapsing, reaching an evaluation success rate in the 0.80–0.92 band — matching the screenshot of
the lost `z8yoqylh` run — where all three runs at `7dae4925` collapsed to exactly 0.00.

**What decides it.** Primary: `eval/success_rate` at 300k, and whether it ever collapses to 0.00.
Secondary: `eval/mean_successful_episode_length`, to test whether the flat-episode-length
observation from the lost comparison survives multiple seeds. Early indicator:
`debug/dQ_da_mean_abs` during critic warmup — see below.

**Predicted early signal — ORIGINAL VERSION WAS WRONG, corrected below.**

*Original text, left per the append-only rule:* "The `FieldNorm` hypothesis says the collapsed runs
had a critic with almost no action dependence (`dQ_da_mean_abs ≈ 2e-4`). If removing `FieldNorm` is
what fixed the run, `caf83f3` should show a materially larger `dQ_da` within the first few thousand
updates. If `dQ_da` is still ~2e-4, stop and re-diagnose rather than burning 3 days."

**Why that was wrong.** The `2e-4` reference was `a3e3zylp`'s value at step **300,000** — the end of
a collapsed run. Comparing our step-100 value against it compares two different quantities. At
matched steps every run starts in the same place:

| Run | step 100 | mean, first 2k steps | final |
|---|---|---|---|
| `a3e3zylp` (collapsed, enc16) | 1.03e-04 | 9.81e-05 | 2.23e-04 |
| `czjqzg0b` (collapsed, enc16) | 7.66e-05 | 1.03e-04 | 2.67e-04 |
| `qe2by47h` (collapsed, enc32) | 1.31e-04 | 1.25e-04 | **1.35e+02** |
| `m0ylcivk` (this run, enc32) | 8.60e-05 | 8.60e-05 | *running* |

Our 8.60e-05 is inside the collapsed runs' early range (7.7e-05 to 1.3e-04), so **a single early
`dQ_da` reading does not discriminate**. The gate as written was worthless.

**Also note `qe2by47h` ended at 1.35e+02** — its Q-gradient *exploded* by six orders of magnitude,
where the other two ended near 2-3e-04. So "vanishing action gradient" is not one clean story
across the three collapsed runs, and any diagnosis resting on it needs to account for that split.

**Corrected early signal.** Compare the *trajectory* of `dQ_da` over the first 10-20k steps against
the three collapsed runs over the same window, not a single reading against a single late value.
Divergence from all three early trajectories is the signal; matching them is not evidence of
anything.

**The honest decision point is the 2nd and 3rd evaluations** (10k and 20k steps, ~1.5 and ~3 h in).
Every collapse was unambiguous by then: `a3e3zylp` 0.88 -> 0.00, `avjcx236` 0.86 -> 0.02,
`1zrjfc56` 0.88 -> 0.18.

### Runtime: ~58 h per seed, not ~31 h

Corrected before launch. The first estimate used `qe2by47h` (31.1 h, the only prior run that was
alone on the machine at `enc_degree_channel=32`). That was wrong, because `caf83f3` **also doubled
the encoder depth**.

At `7dae4925` each of the four encoder stages had one residual block and the second was commented
out. At `caf83f3` all eight are active:

```
7dae4925:  4 active residual blocks     caf83f3:  8 active residual blocks
```

Measured on this machine during the smoke run: **163 ms per critic update** against `qe2by47h`'s
**82 ms** — a 1.99x ratio, matching the block doubling almost exactly. Gradient updates are 93% of
wall clock and there are 1.2M of them (300k steps x UTD 4), giving ~54 h of gradient time and
**~58 h per seed**.

### What actually changed in `caf83f3`: three candidates, not one

This matters for interpreting the result. `caf83f3` changed three things at once on the
critic/encoder path:

1. **`FieldNorm` and the ReLUs removed from the critic head** — the original hypothesis.
2. **Encoder depth doubled**, 4 -> 8 residual blocks. A deeper encoder producing better features is
   an equally good explanation for a critic that previously had no usable action gradient.
3. **`enc_degree_channel` default raised 16 -> 32.**

So **if the reproduction succeeds, it will not tell us which of the three fixed it.** Isolating them
needs follow-up runs at ~58 h each. Worth deciding whether that is the right spend before doing it —
a cheaper route may be to check `dQ_da` under each variant for a few thousand steps rather than
running each to 300k.

One candidate was eliminated. `qe2by47h` logged `use_norms: false`, which raised the possibility
that it ran without `FieldNorm` and collapsed anyway. It did not: at `7dae4925` the critic head read
`CriticConfig.use_layer_norm` (default `True`) and `q_agent` hardcoded `use_layer_norm = True`, so
`equivariance.use_norms` never reached it. All three collapsed runs had `FieldNorm`.

### Assumptions in this reconstruction

`z8yoqylh`'s config is unrecoverable. These are the judgement calls, recorded so the reproduction
can be re-examined if it fails.

1. **Config defaults, not the README overrides.** The README and the coffee/square launch scripts
   pass `n_step=5 gamma=0.995 stddev=0.025 action_scale=0.2`. The reference run `a3e3zylp` did
   **not** — its resolved config shows `n_step=3, gamma=0.99, stddev=0.05, action_scale=0.1`, i.e.
   the dataclass defaults, with `algo.prefetch_batches=4` as the only algorithmic override. Every
   Can run in this project was launched that way, so the reproduction is too.
2. **`enc_degree_channel = 32`, the `caf83f3` default.** This is the one real fork. The default was
   **16** at `7dae4925` and was raised to **32** in `caf83f3` — the same commit titled "got
   equivariant agent working for the first time". `a3e3zylp` therefore ran at 16 and collapsed.
   Assuming the author committed the width they had working, 32 is the better estimate, so
   `submit.sh` does not override it. **If the reproduction fails, retrying at 16 is the first thing
   to try.**
3. **`use_norms=True`, `use_orth_init=True`, `use_equivariant_model=True`.** All three fields were
   *added* in `caf83f3` and did not exist at `7dae4925`. Their defaults are taken as intended. Per
   the no-op tests, `use_norms` and `use_orth_init` affect only the actor and encoder anyway.
4. **Buffer caches are reused.** The Can offline (`cd03f9df`) and online (`9ff5ddd4`) caches are on
   disk and will hit, because no cache-key field changed between the two commits. Verified by
   timestamp that both were built by run `1zrjfc56`, an **equivariant** Can run, so the data matches
   the equivariant normalization path. No cross-path contamination.
5. **All three seeds share identical warmup and offline data**, since the caches hit. They differ
   only in network initialization and in online exploration after warmup. This reduces variance but
   means the seeds are **not fully independent**, and the reported spread will understate true seed
   variance. It is also what the reference runs did, so it is the right choice for a reproduction —
   but any claim about seed variance must carry this caveat.
6. **`stddev_step = 300_000` with `stddev_max == stddev_min`**, so the exploration-noise schedule is
   effectively constant. Unchanged from the reference runs.

### Known defects present at `caf83f3`, deliberately not fixed

Kept in place so a reproduction failure cannot be confused with a change we introduced. All are
scheduled in TODO P3 behind the reproduction.

- Six `EquivarianceConfig` fields are inert; `use_norms` half-survives via an encoder bias coupling.
  Now pinned by executable tests (`tests/test_no_ops.py`), so the provenance table is fact.
- The critic head has **no normalization at all** while the actor has active batch norm.
- The online buffer cache key omits `base_policy_wandb_id`, so swapping the BC policy would silently
  reuse a warmup buffer built with the old one.
- Neither cache key records which normalization path produced the data.
- `# BREAK HERE` and a commented-out `# if False:` sit directly above the critic-warmup call.
- The scalar ablation (`use_equivariant_model=False`) differs from the equivariant stack in six ways
  beyond equivariance, so it is not usable as a control. STANDARDS.md rule 3.3.

### Results

*To be filled in after each seed completes. Do not edit the sections above.*

| Seed | wandb | Steps | step-0 | best | final | ep-len start → end | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | **`m0ylcivk`** | *running* | | | | | launched 2026-09-16 12:05 at `f6716c9`, GPU 0 |
| 2 | | *not launched* | | | | | held pending seed 1 |
| 3 | | *not launched* | | | | | held pending seed 1 |

Run URL: `https://wandb.ai/aden-mckinney10-university-of-central-florida/robomimic-can-final/runs/m0ylcivk`

Seed 1 launch log kept at `wandb/run-*-m0ylcivk/files/output.log`. **Do not delete that directory** —
it is the only copy if the wandb run is ever removed, which has already happened twice in this
project (`z8yoqylh`, `msfkjwab`).

---

## Runs that exist only in wandb, not locally

Found 2026-09-16 by listing `robomimic-can-final` through the wandb API. These ran on the other
computer, so there is no local `wandb/run-*` directory and no `output.log`. Metrics below come from
the wandb history API.

### 2026-05-07 — `czjqzg0b` — equivariant Can, 300k, collapsed

- **Commit:** `7dae4925` · **Config:** `residual_equi_td3_can_config` · **Task:** Can · **Seed:** 4210233833
- 300k steps, 31 evals, **33.2 h** · **step-0:** 0.88 · **best:** 0.88 · **final:** 0.00
- Logged equivariance: `N=8, enc_degree_channel=16, num_actor_layers=3, num_critic_layers=2`
- **Config caveats:** `num_actor_layers=3` and `num_critic_layers=2` were inert at this commit.
- Sibling of `a3e3zylp` (same day, same commit, different seed). Same collapse.

### 2026-05-16 — `qe2by47h` — equivariant Can, 300k, collapsed

- **Commit:** `7dae4925` · **Config:** `residual_equi_td3_can_config` · **Task:** Can · **Seed:** 695000613
- 300k steps, 31 evals, **31.1 h** · **step-0:** 0.82 · **best:** 0.82 · **final:** 0.00
- Logged equivariance: `N=8, enc_degree_channel=32, use_norms=False, use_orth_init=False`
- **Config caveats:** `use_orth_init=False` was inert on the critic. `use_norms=False` was *partly*
  live — it removed the actor's first two batch norms and flipped the encoder's `bias=not use_norms`,
  but the critic's norms were unaffected.
- **The latest equivariant run of any kind, and it collapsed.** Nine days after `a3e3zylp`.
- **Interpretation:** three runs at `7dae4925` — `a3e3zylp`, `czjqzg0b`, `qe2by47h` — with three
  different seeds and three different configs all reached 300k steps and all ended at exactly 0.00.
  A consistent architectural failure, not seed variance.

### 2026-04-02 — `x2w6phzf` — renamed, and used N=12

Already logged above from local data, with two corrections from the wandb config:

- It is named `first_equivariant_learning_regularized` in wandb, not `residual-rl`.
- It used **`N=12`**, not `N=8` — the only run in the project that did. `enc_degree_channel=32`,
  `num_actor_layers=3`.
- Runtime **87.6 h**, the longest run in the project.

---

## What the wandb audit changed

Two things, both important.

**1. The baseline arm `msfkjwab` is also deleted.** Not just `z8yoqylh`. `robomimic-can-final` now
contains only 4 runs: `x2w6phzf`, `czjqzg0b`, `a3e3zylp`, `qe2by47h` — all collapsed. The baseline
Can result (0.92 → 1.00) survives **only** as the local directory
`wandb/run-20260226_193630-msfkjwab/`. Do not delete it. TODO R7's plan to pull the baseline from
the API does not work; the local copy is the only copy.

**2. Group pooling is not what fixed the collapse.** `nn.GroupPooling` was already in the critic
head at `7dae4925`, where all three of the above runs collapsed. Diffing `7dae4925` against
`caf83f3` shows the real change: the critic head went from
`Linear → FieldNorm → ReLU → … → GroupPooling → Linear` to plain
`Linear → GroupPooling → Linear`, with every `FieldNorm` and `ReLU` commented out.

`FieldNorm` subtracts each field's projection onto the trivial-representation subspace — which, for
a `regular_repr` field, is exactly its group-invariant component. So it was removing the invariant
content that the following `GroupPooling` exists to extract. That matches the observed
`dQ_da ≈ 2e-4` with a low stable `critic_loss`. See EQUIVARIANCE.md for the full reasoning.

**Why this is good news for the reproduction:** `caf83f3` differs from the collapsed runs in
precisely the way that would explain the difference in outcome. The reproduction target is not the
same code that failed three times.

---

## Shorter runs

Crashed, killed, or abandoned early. Useful mainly as a record of the failure pattern.

| Date | Run | Commit | Task | Equi | Steps | Evals | step-0 | best | final |
|---|---|---|---|---|---|---|---|---|---|
| 2026-01-14 | `wkhkug3l` | `66262b06` | Square | no | 30,600 | 4 | 0.52 | 0.52 | 0.02 |
| 2026-01-15 | `ytxnl676` | `66262b06` | Square | no | 141,300 | 15 | 0.62 | 0.84 | 0.76 |
| 2026-01-31 | `kul7izf7` | `5c4f2645` | Square | no | 18,700 | 2 | 0.58 | 0.58 | 0.24 |
| 2026-01-31 | `psftynpk` | `debc9b6c` | Square | no | 23,200 | 3 | 0.54 | 0.54 | 0.48 |
| 2026-02-02 | `gkxw355f` | `debc9b6c` | Square | no | 109,200 | 11 | 0.58 | 0.76 | 0.64 |
| 2026-02-27 | `blcmficm` | `807efd65` | Can | yes | 57,600 | 6 | 0.82 | 0.82 | 0.72 |
| 2026-02-28 | `0fvr87c3` | `4983fcda` | Can | yes | 108,400 | 11 | 0.90 | 0.90 | 0.00 |
| 2026-03-02 | `ivpp0nz5` | `a948786d` | Square | yes | 5,900 | 7 | 0.00 | 0.00 | 0.00 |
| 2026-03-03 | `clm4mu7k` | `a948786d` | Square | yes | 3,500 | 1 | 0.00 | 0.00 | 0.00 |
| 2026-03-10 | `9rw1c891` | `a948786d` | Square | yes | 22,300 | 3 | 0.00 | 0.00 | 0.00 |
| 2026-03-10 | `4xpkxr18` | `a948786d` | Square | yes | 57,700 | 6 | 0.00 | 0.00 | 0.00 |
| 2026-04-01 | `8six0a8z` | `a948786d` | Square | yes | 2,600 | 1 | 0.14 | 0.14 | 0.14 |
| 2026-04-02 | `4q12wnej` | `a948786d` | Square | yes | 3,000 | 1 | 0.00 | 0.00 | 0.00 |
| 2026-04-02 | `8jkye61t` | `a948786d` | Square | yes | 2,900 | 1 | 0.24 | 0.24 | 0.24 |
| 2026-04-21 | `m4l85yty` | `a948786d` | Square | yes | 5,600 | 1 | 0.66 | 0.66 | 0.66 |
| 2026-04-21 | `n9ui0dx8` | `a948786d` | Square | yes | 1,700 | 1 | 0.58 | 0.58 | 0.58 |
| 2026-04-22 | `z3chjuoz` | `a948786d` | Square | yes | 12,700 | 2 | 0.48 | 0.48 | 0.00 |
| 2026-04-22 | `1zrjfc56` | `a948786d` | Can | yes | 63,900 | 7 | 0.88 | 0.88 | 0.18 |
| 2026-04-23 | `avjcx236` | `aabde0fe` | Can | yes | 20,000 | 3 | 0.86 | 0.86 | 0.02 |

All overrides were `algo.prefetch_batches=4` except `wkhkug3l`, which used the full override set
from the README example.

Patterns worth noting:

- **Every equivariant run before group pooling collapses.** Where the base policy starts high
  (Can, 0.82–0.90), the run falls to near zero. Where it starts low, it never leaves zero. This is
  one consistent failure mode across 14 runs and 5 commits, not a collection of unrelated bugs.
- **The March Square runs sit at 0.00 from step 0.** A step-0 eval of 0.00 means the *base policy*
  scored zero, which should be impossible given `870ws2c2` measured 0.52 on the same task. Most
  likely something in the observation or normalization pipeline was broken at `a948786d` for
  Square, so the base policy was being fed garbage. Then the step-0 numbers recover to 0.48–0.66 in
  late April with no config change recorded. Unexplained. Worth knowing about before trusting any
  Square number from `a948786d`.
- **Non-equivariant runs also dip at their final eval** (`wkhkug3l` 0.52→0.02, `kul7izf7`
  0.58→0.24). These are killed runs, so the last eval may have landed mid-instability. Not the same
  phenomenon as a sustained collapse.

### Bulk: launch failures

197 of the 220 local run directories never got past ~1000 steps or a second evaluation. Not
individually catalogued; if one is ever needed it is on disk under `wandb/run-*/files/output.log`.

Causes, counted by grepping every short run's log (a run can match more than one):

| Pattern | Runs |
|---|---|
| `KeyboardInterrupt` — manual interrupt during development | 96 |
| escnn / `FieldType` / `GeometricTensor` mentioned in a traceback | 43 |
| `AssertionError` | 33 |
| torchcodec video decode (`Could not push packet to decoder`) | 15 |
| No traceback and no matched pattern | 7 |

Two things worth reading off this table:

- **The dominant cause is manual interruption, not crashes.** About half these directories are
  development iterations that were killed on purpose. They are noise, not failures.
- **43 runs died with escnn types in the traceback.** That is the cost of having no equivariance
  tests: field-type and shape mismatches were being found by launching training runs. A
  construction-time layout assertion (TODO P2.4) would have caught most of them in seconds instead
  of minutes.

No CUDA OOM and no wandb artifact-fetch failures appeared anywhere, so the environment and the
checkpoint-download path have been reliable.

---

## What was never run

Recorded so the gaps are visible:

- **No multi-seed comparison of anything.** Every run above is a single seed. No result in this
  project currently has error bars.
- **No equivariant run on any two-arm dexmg task.** Only Can and Square have equivariant configs.
- **No run with `use_equivariant_model=False`** — the scalar ablation path exists in code
  (`ablate_equi_actor.py`, `ablate_equi_critic.py`) and has never been used in a logged experiment.
  That is the control that would separate "equivariance is wrong" from "residual RL is
  misconfigured on this path."
- **No SAC anything.** There is no SAC implementation in the repo.
- **No equivariance unit test has ever been run.** Every equivariance claim to date rests on code
  reading.

---

## 2026-09-16 — CORRECTION — `z8yoqylh` recovered from disk, and it ran at `7dae4925`

**This entry corrects several claims in the entries above. Per the append-only rule nothing
earlier was edited; read this entry as superseding them.**

The lost run was found on the second workstation (`boce-WS-01`, `10.188.60.163`) at
`~/projects/equi-resrl/wandb/run-20260522_083555-z8yoqylh`, 173 MB, with `config.yaml`,
`wandb-metadata.json`, `output.log` and `requirements.txt` all intact. It was never on
`ZXP-S-works`, which is why the earlier audit concluded it was gone.

### What the recovered metadata says

| Field | Value |
|---|---|
| **Commit** | **`7dae4925beaf2e7b2a68099413b98ea21615bef1`** |
| Host | `boce-WS-01`, 2 × RTX 4090, 24 logical cores, 134 GB RAM |
| Started | 2026-05-22T12:35:55Z |
| **Seed** | **2114495708** (recovered from the run directory name) |
| Runtime | 170,382.85 s = **47.3 h** |
| Stack | torch 2.6.0+cu124, torchrl 0.7.0, tensordict 0.7.0, **escnn 1.0.11**, numpy 1.26.4, CPython 3.10.20 |
| Overrides | `algo.prefetch_batches=4`, **`agent.actor.actor_last_layer_init_scale=1e-4`** |
| step-0 eval | 0.82 |
| **best** | **0.94** (progression 0.82 → 0.90 → 0.94) |

Resolved config, confirming the architecture: `enc_degree_channel: 32`,
`actor_degree_channel: 128`, `critic_degree_channel: 128`, `action_scale: 0.1`, `n_step: 3`,
`gamma: 0.99`, `stddev_max/min: 0.05` (flat schedule), `buffer_size: 200000`,
`critic_warmup_steps: 10000`, `num_updates_per_iteration: 4`, `use_norms: true`,
`use_orth_init: true`, `use_equivariant_model: true`.

### Claims above that are now false

1. **"Commit: unknown. Believed to be `caf83f3` or later."** It was `7dae4925`.
2. **"Three runs at `7dae4925` all ended at exactly 0.00 — that is an architectural failure, not
   seed variance."** `7dae4925` also produced the only success. It is not an architectural failure
   at that commit.
3. **The `FieldNorm` hypothesis is no longer motivated.** `FieldNorm` and the ReLUs were present in
   the critic head of the run that reached 0.94. Removing them in `caf83f3` cannot be what fixed a
   run that predates `caf83f3`.
4. **Assumption 3 of the `caf83f3` pre-registration is wrong.** `use_norms`, `use_orth_init` and
   `use_equivariant_model` were claimed to have been added in `caf83f3`; all three appear in
   `z8yoqylh`'s resolved config at `7dae4925`.
5. **"best ~0.92", from the screenshot.** Recomputed from `output.log`: **0.94**.
6. **Tag `repro-z8yoqylh-base` points at `caf83f3`, the wrong commit.** Left in place because it is
   already pushed; superseded by a tag at `7dae4925`.

Assumption 2 of that pre-registration — `enc_degree_channel = 32` — was **correct**.

### The natural experiment this exposes

`z8yoqylh` and `a3e3zylp` share a commit and a config file and differ in exactly two things.
`a3e3zylp`'s extra `equivariance.num_actor_layers=3` is a verified no-op, so the architectures are
identical.

| | `z8yoqylh` | `a3e3zylp` |
|---|---|---|
| Commit | `7dae4925` | `7dae4925` |
| Host / stack | boce-WS-01, torchrl 0.7.0, escnn 1.0.11 | ZXP-S-works, torchrl 0.9.2, escnn 1.0.13 |
| `actor_last_layer_init_scale` | **1e-4** | 0.0 (default) |
| Result | 0.82 → **0.94** | 0.88 → **0.00** |

At `7dae4925` the config file reads:

```python
actor_last_layer_init_scale=0.0,     # imp for residual
# actor_last_layer_init_scale=1e-4,  # imp for residual
```

Two candidate causes, confounded in the existing record: the init-scale override, and the software
stack. **Every collapsed equivariant run ran on `ZXP-S-works`; the single success ran on
`boce-WS-01`.** Host correlates perfectly with outcome across the whole history. Isolating the two
is deferred until the replication below confirms the result.

### Runtime reference, measured, 300k steps on Can

| Run | Model | GPU | Runtime |
|---|---|---|---|
| `msfkjwab` | baseline, non-equivariant | 3090 | 59,099 s = 16.4 h |
| `a3e3zylp` | equivariant, `7dae4925` | 3090 | 148,767 s = 41.3 h |
| `z8yoqylh` | equivariant, `7dae4925` | 4090 | 170,383 s = 47.3 h |

The equivariant model costs ~2.5× the baseline on identical hardware: 1.2M gradient updates
(300k × UTD 4) through escnn, with gradients at ~92% of wall clock. Note the 4090 run was *slower*
than the 3090 run, so the newer box is not a speedup — concurrent runs on that machine in May are
the likely explanation.

---

## PRE-REGISTERED, NOT YET LAUNCHED (2)

### Replication of `z8yoqylh` at the recovered config — 3 seeds on `boce-WS-01`

Written before launching, per STANDARDS.md rule 5.4. Supersedes the `caf83f3` pre-registration
above, which was aimed at the wrong commit. **Seed-group id: `equi-repro-7dae4925`.**

| | |
|---|---|
| **Commit** | `7dae4925`, checked out in a git worktree so HEAD keeps the docs, tests and gate |
| **Config** | `residual_equi_td3_can_config` |
| **Overrides** | `algo.prefetch_batches=4`, `agent.actor.actor_last_layer_init_scale=1e-4`, plus naming and `seed` |
| **Task** | Can · **Seeds** 2114495708 (the original), 1, 2 — run in parallel |
| **Host** | `boce-WS-01`, stack matching the reference run exactly |
| **Expected runtime** | ~47 h solo; the two sharing a GPU will take longer |

**Hypothesis.** At `7dae4925` with `actor_last_layer_init_scale=1e-4`, the equivariant residual
agent trains to 300k steps on Can without collapsing, reaching an evaluation success rate near
**0.94** on the original seed, and above its step-0 base rate on all three seeds.

**What decides it.** Primary: `eval/success_rate` at 300k, and whether any seed collapses to 0.00.
Seed 2114495708 is the exact-replication arm and is expected to land near 0.94; seeds 1 and 2
measure whether the result is robust to initialization and exploration. Secondary:
`eval/mean_successful_episode_length`, to test the flat-episode-length observation from the lost
comparison.

**Early read.** Every collapse in the record was unambiguous by the 10k and 20k evaluations
(0.00, 0.02, 0.18). Three seeds all clearing their step-0 rate at 20k is the first real signal.

**Assumptions.** Far fewer than the `caf83f3` attempt, because the config is recovered rather than
reconstructed:

1. **Passing `seed=2114495708` reproduces the original seeding.** Verified by reading
   `train_residual_td3.py:321-331` at `7dae4925`: a `None` seed is randomized and then immediately
   used to seed `random`, `numpy` and `torch`, so an explicit seed follows the identical path.
2. **The current `boce-WS-01` environment still matches the run's `requirements.txt`.** Confirmed
   field by field against the recovered file.
3. **Buffer caches hit.** The cache-key metadata dicts are byte-identical between `7dae4925` and
   HEAD, so the hashes the gate computed (`8edf618e` offline, `43637c10` online) are valid at the
   older commit. `CACHE_DIR` must point at the main checkout, because `_CACHE_ROOT` defaults to the
   *current directory* (`train_residual_td3.py:138`) and a worktree would otherwise miss both.
4. **Naming differences are scientifically inert.** The original ran as `wandb.name=residual-rl`
   with the stale `paper_runs/square/yet_to_come` note; these runs use the rule 5.2 convention.
5. **GPU contention does not affect results, only wall clock.** Two of the three seeds share a GPU.

**Known confound, deliberately not addressed here.** This replication holds both the init-scale
override and the host/stack at the successful run's values, so a success will not say which of the
two caused the original collapses. Isolating them comes after the result is confirmed.
