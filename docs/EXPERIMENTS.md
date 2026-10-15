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

### LAUNCHED 2026-09-16 16:00 EDT — `equi-repro-7dae4925`, all three seeds

| Seed | wandb run | GPU | Role |
|---|---|---|---|
| 2114495708 | **`9qgsaxsr`** | 0, alone | Exact replication of `z8yoqylh` |
| 1 | **`4sseggkv`** | 1, shared | Seed robustness |
| 2 | **`dxbept4u`** | 1, shared | Seed robustness |

Host `boce-WS-01`. Launched from a git worktree at `7dae4925`
(`~/projects/equi-resrl-7dae4925`) via `~/launch_repro.sh <seed> <gpu>`, with
`CACHE_DIR=~/projects/equi-resrl` and `artifacts/` symlinked to the main checkout.

**Startup verified on all three:** `Set random seed to <seed>` matches, both buffer caches
reported `found on disk` at the expected hashes (`8edf618e` offline / 62,454 transitions,
`43637c10` online / 10,000), no rebuild, no errors. The gate passed with the corrected version
expectations, so the environment matches `z8yoqylh`'s `requirements.txt` exactly.

**Two things observed at launch, recorded because they contradict the docs.**

1. **`env_probes/` does not exist at `7dae4925` and is not needed.** It is untracked at that
   commit (created 2026-05-21, one day before `z8yoqylh`), and every use of the robot-base probe
   in that tree is commented out — `train_residual_td3.py:423` and `:426`, plus the
   `equi_normalizer.py` centering block. **So the successful run had robot-base centering
   disabled.** The feature is active at HEAD and `preflight.py` checks for the file, which is why
   the worktree looked broken at first. It is not.
2. **Each run holds ~25 GB resident, not the ~21 GB in TODO R0.** Measured RSS: 24.8 / 23.6 /
   24.9 GB. Three runs leave 22 GB of 125 GB available — stable, since the buffers are
   preallocated, but tighter than planned. Three is the maximum on this machine, not a comfortable
   three.

**Measured warmup rates, showing the cost of GPU sharing:** 79.7 ms/update solo on GPU 0 against
145-150 ms/update for the two sharing GPU 1, a 1.85x penalty. Against `z8yoqylh`'s measured 47.3 h,
that projects **~47 h for seed 2114495708 and ~85-90 h for seeds 1 and 2.** Running the shared pair
sequentially instead would have cost ~94 h for both, so sharing is the better choice for getting
all three, at the price of the two robustness arms landing about two days after the replication arm.

**Config caveats: UNVERIFIED at this commit.** The schema requires this field and it cannot yet be
filled honestly. The provenance table in [ARCHITECTURE.md](ARCHITECTURE.md) lists which
`EquivarianceConfig` fields are accepted, logged and ignored, and the no-op tests pin that list at
HEAD / `caf83f3`. **Nobody has checked which fields are inert at `7dae4925`.** One prior claim about
that commit — that `use_norms`, `use_orth_init` and `use_equivariant_model` did not exist there —
proved false when `z8yoqylh`'s resolved config showed all three. Closing this is TODO R8, and it
blocks the results entry for these three runs.

---

## 2026-09-18 — RESULTS — `equi-repro-7dae4925`: all three seeds collapsed

The replication pre-registered above ran and **failed on every seed.** Recorded here as a result in
its own right; the reason it failed is the correction entry that follows.

| Seed | wandb | Step-0 | Every eval after step 0 | Final `eval/success_rate` | Steps reached |
|---|---|---|---|---|---|
| 2114495708 | `9qgsaxsr` | 0.84 | **0.00 × 30** | **0.00** | 300,000 — completed |
| 1 | `4sseggkv` | 0.82 | **0.00 × 20** | 0.00 | 209,500 — killed |
| 2 | `dxbept4u` | 0.84 | **0.00 × 20** | 0.00 | 209,500 — killed |

Against the pre-registered decision criterion — `eval/success_rate` at 300k, and whether any seed
collapses to 0.00 — this is an unambiguous negative on all three arms. Not one evaluation after
step 0, across 71 evaluations and three seeds, produced a single successful episode. Per rule 5.5
the step-0 figures are the base BC policy alone and are not results.

Seed 2114495708 is the informative arm: it ran the full 300k in **34.6 h** (124,642 s, faster than
the ~47 h projected) and ended at 0.00, with `debug/residual_max_abs` saturated at 1.0 and
`eval/mean_successful_episode_length` at 0. Seeds 1 and 2 were killed on 2026-09-18 at 07:48 EDT
after 20 consecutive 0.00 evaluations each, once the exact-replication arm had already finished at
0.00 and the cause below was identified. Their remaining ~90k steps could not have changed any
pre-registered conclusion. Both wandb directories are intact in
`~/projects/equi-resrl-7dae4925/wandb/`.

**The `dQ_da` early-signal gate, once more.** Seed 2114495708's final `debug/dQ_da_norm` was
2.60e-02 — neither the vanishing 2-3e-04 of two collapsed runs nor the exploding 1.35e+02 of
`qe2by47h`. A third distinct value on a third collapse. The 2026-09-16 note that this metric does
not discriminate collapse stands, and is now stronger.

**Config caveats.** The unverified field flagged in the pre-registration is now resolved, and
resolved against the runs: the config these three ran differs from `z8yoqylh`'s in five fields.
See the correction below. **These three runs are not a replication of `z8yoqylh` and should not be
cited as evidence about its architecture.** They are evidence about `7dae4925`.

---

## 2026-09-18 — CORRECTION — `z8yoqylh` ran `caf83f3`'s code, not `7dae4925`'s

**The 2026-09-16 correction was backwards.** It retargeted the reproduction from `caf83f3` to
`7dae4925` on the strength of the `git.commit` field in `z8yoqylh`'s recovered wandb metadata. That
field records the value of `HEAD` at launch. It does not record the working tree, and `z8yoqylh`'s
working tree was dirty. The original target, `caf83f3`, was right.

### The evidence

**1. Five config fields disagree with `7dae4925` and agree with `caf83f3`.** `z8yoqylh`'s resolved
config cannot have been produced by the code at `7dae4925`, because that code has no such fields:

| Config field | `z8yoqylh` logged | code at `7dae4925` | code at `caf83f3` |
|---|---|---|---|
| `agent.critic.<name>` | `dropout` | `drop` | `dropout` |
| `equivariance.enc_degree_channel` | 32 | 16 (default) | 32 (default) |
| `equivariance.use_equivariant_model` | True | **absent** | present |
| `equivariance.use_norms` | True | **absent** | present |
| `equivariance.use_orth_init` | True | **absent** | present |

`git log -S` dates the three `use_*` fields to `caf83f3` itself and the `enc_degree_channel`
default of 32 to `aabde0f`; `7dae4925` sits between them with 16.

**2. The full config composes exactly.** Composing `caf83f3`'s `residual_equi_td3_can_config` with
`z8yoqylh`'s recorded launch arguments plus `seed=2114495708` reproduces its logged config on **all
117 fields.** The only apparent differences are ten int-versus-float YAML renderings of identical
values (`0` / `0.0`, `1` / `1.0`, `-1` / `-1.0`) and `actor_name`, which is not a config input at
all — it is inferred at runtime from the base policy at `train_residual_td3.py:273`.

**3. The timing fits, to the hour.** `z8yoqylh` started 2026-05-22 at 08:35 EDT. `caf83f3` was
committed the **same morning at 10:28 EDT**, with the message *"got equivariant agent working for
the first time."* The run was launched from a tree already carrying the `caf83f3` changes, watched
for about two hours — its evaluation at 30k hit 0.94 — and then committed.

### Claims that are now false

| Claim, from the 2026-09-16 entry and STATUS.md | Actually |
|---|---|
| `z8yoqylh` ran at `7dae4925` | It ran `caf83f3`'s code with `HEAD` reading `7dae4925` |
| `7dae4925` produced both the collapses and the success | It produced **only collapses** — see the results entry above |
| The `FieldNorm` hypothesis is dead | **Reopened.** It was ruled out on the premise that `FieldNorm` was present in the successful run's critic head at `7dae4925`. That premise is gone. |
| The successful run had robot-base centering disabled | **Probably not.** At `7dae4925` every centering call site is commented out, which is where that claim came from. At `caf83f3` the centering block is live (`equi_normalizer.py:425`, with an explicit warning when no base is supplied). So `z8yoqylh` likely rotated about the robot base and the symmetry was *not* silently broken. |
| `enc_degree_channel = 32` was guessed correctly | The guess was right, but the three replication runs ran **16**, because that is `7dae4925`'s default and nobody overrode it |
| `use_norms` / `use_orth_init` / `use_equivariant_model` exist at `7dae4925` | They do not. That inference was circular — it assumed the commit in order to conclude the fields existed. |

`m0ylcivk`, the `caf83f3` run killed on 2026-09-16 for "not being the reproduction," was aimed at
the right commit. Its local `wandb/run-*-m0ylcivk` directory is **gone from disk**, despite the
do-not-delete note earlier in this file. It may survive in wandb.

### The lesson, for the standards

A recorded commit hash is necessary but **not sufficient** to determine the code that ran, because
wandb records `HEAD` and not the working tree. The resolved config is the stronger evidence: it is
produced by the code that actually executed. Where the two disagree, the config wins. Priority 1 in
CLAUDE.md — "a run's recorded config must fully determine the code that ran" — was satisfied here
only because the config schema happened to carry five discriminating fields.

---

## PRE-REGISTERED, NOT YET LAUNCHED (3)

### Replication of `z8yoqylh` at `caf83f3` — 3 seeds on `boce-WS-01`

Written before launching, per STANDARDS.md rule 5.4. Supersedes both earlier pre-registrations:
the first aimed at `caf83f3` from a reconstructed config, the second at `7dae4925` from a
misread commit field. This one aims at `caf83f3` with a **verified** config.
**Seed-group id: `equi-repro-caf83f3`.**

| | |
|---|---|
| **Commit** | `caf83f3`, in a git worktree at `~/projects/equi-resrl-caf83f3` |
| **Config** | `residual_equi_td3_can_config` — verified to compose to `z8yoqylh`'s 117 logged fields |
| **Overrides** | `algo.prefetch_batches=4`, `agent.actor.actor_last_layer_init_scale=1e-4`, plus naming and `seed` |
| **Task** | Can · **Seeds** 2114495708 (the original), 1 — amended from three, see below |
| **Host** | `boce-WS-01` |
| **Expected runtime** | ~35-47 h; both seeds run solo on their own GPU |

**Hypothesis.** At `caf83f3` with `actor_last_layer_init_scale=1e-4`, the equivariant residual agent
trains to 300k steps on Can without collapsing, reaching an evaluation success rate near **0.94** on
seed 2114495708 and above its step-0 base rate on both seeds.

**What decides it.** Primary: `eval/success_rate` at 300k, and whether either seed collapses to
0.00. Seed 2114495708 is the exact-replication arm. Secondary:
`eval/mean_successful_episode_length`, for the flat-episode-length question.

**Amended before launch: two seeds, not three.** Decided 2026-09-18 for wall clock — two seeds one
per GPU both finish in ~35-47 h, where three would have put two of them on a shared GPU and ~85-90 h
out, so the full result lands in about two days instead of four. The cost is stated plainly: **n=2
has no tiebreaker.** If the two seeds split — one near 0.94, one at 0.00 — this design cannot say
which is typical, and a third seed becomes necessary. That risk was accepted on the grounds that all
three `7dae4925` seeds collapsed identically, so a split looks unlikely. Recorded here rather than
edited in silently, per rule 5.4's purpose.

**Early read.** `z8yoqylh` was at 0.80 by its second evaluation and never below 0.80 again. Every
collapse in the record, now including all three `7dae4925` runs, was unambiguous by 10k-20k. A seed
sitting at 0.00 at 20k has failed; there is no case in the record of recovery from it.

**Assumptions.**

1. **`caf83f3` is the code that produced `z8yoqylh`.** The 117-field config match, the five
   discriminating schema fields, and the same-morning commit timing. This is the strongest
   provenance any run in this project has had.
2. **Passing `seed=2114495708` reproduces the original seeding.** `z8yoqylh` passed no `seed`
   override, so its seed was randomized and then logged. Verified at `caf83f3`
   (`train_residual_td3.py:345-352`): a `None` seed is randomized and then immediately used to seed
   `random`, `numpy` and `torch`, so an explicit seed follows the identical path.
3. **Results are not bit-reproducible.** `torch_deterministic = False` in the recovered config, so
   even the exact-replication arm will differ run to run. The `7dae4925` arms' step-0 evaluations
   came in at 0.84 / 0.82 / 0.84 against `z8yoqylh`'s 0.82 — that spread is the noise floor, and a
   near-0.94 outcome, not an identical one, is what counts as success.
4. **Buffer caches hit.** The gate computed `8edf618e` offline and `43637c10` online at HEAD; these
   must be re-confirmed against what the script prints at `caf83f3`. `CACHE_DIR` must point at the
   main checkout, because `_CACHE_ROOT` defaults to the current directory.
5. **The gate cannot validate this commit's code.** `preflight.py` and `tests/` do not exist at
   `caf83f3`, so the 148 tests and the equivariance checks run against HEAD, not against the code
   being launched. This was equally true of the `7dae4925` launch and is worth fixing.

**Known confound, deliberately not addressed.** As before, this holds the init-scale override and
the host/stack together at the successful run's values, so a success will not say which caused the
original collapses.

**Planned layout**, matching the `7dae4925` launch so wall-clock and GPU contention stay
comparable across the two attempts:

| Seed | GPU | Role |
|---|---|---|
| 2114495708 | 0, alone | Exact replication of `z8yoqylh` |
| 1 | 1, alone | Seed robustness |

**The CUDA block is cleared.** `boce-WS-01` was rebooted 2026-09-18 around 07:30 EDT; the
580.178.04 kernel module is now loaded and matches userspace, and torch sees both 4090s.

**New environment deviation from `z8yoqylh`, recorded because host and stack are live candidate
causes in this project.** The reboot moved the machine off the environment the successful run used:

| | `z8yoqylh`, 2026-05-22 | now |
|---|---|---|
| Kernel | 6.8.0-111 | **6.8.0-138** |
| NVIDIA driver | 580.173.02 | **580.178.04** |

The python stack — torch 2.6.0, torchrl 0.7.0, tensordict 0.7.0, escnn 1.0.11 — is unchanged and
still matches the recovered `requirements.txt` field for field, which is what `preflight.py`
validates. A driver and kernel bump is a weaker difference than a library difference, but it is not
nothing, and it is the one thing this replication cannot hold fixed. If both seeds collapse, this
is the first thing to suspect before concluding anything about `caf83f3`.

**To launch, after the reboot:**

```bash
cd ~/projects/equi-resrl && export PYTHONPATH="$PWD:$PYTHONPATH"
python preflight.py                      # must reach GO, with Tier 2 included this time
~/launch_repro.sh 2114495708 0 > ~/repro_caf_s2114495708.log 2>&1 &
~/launch_repro.sh 1 1 > ~/repro_caf_s1.log 2>&1 &
```

Then verify on each: `Set random seed to <seed>` matches, and both buffer caches report `found on
disk` at `8edf618e` (offline) and `43637c10` (online) with no rebuild. Those `~/*.log` files are
block-buffered and lag by thousands of steps — read
`~/projects/equi-resrl-caf83f3/wandb/run-*/files/output.log` instead. Record the three run ids here
once they are known.

### LAUNCHED 2026-09-18 08:30 EDT — `equi-repro-caf83f3`, both seeds

| Seed | wandb run | GPU | Role |
|---|---|---|---|
| 2114495708 | **`38z4wr9z`** | 0, alone | Exact replication of `z8yoqylh` |
| 1 | **`sw2qwfs9`** | 1, alone | Seed robustness |

Host `boce-WS-01`. Launched from the git worktree at `caf83f3`
(`~/projects/equi-resrl-caf83f3`) via `~/launch_repro.sh <seed> <gpu>`, with
`CACHE_DIR=~/projects/equi-resrl` and `artifacts/` symlinked to the main checkout. Both detached
with `setsid`, so they survive the launching session.

**Gate:** full GO at `ef547ed`, Tier 2 included — 69 static tests plus the 79-test equivariance
suite, all passing on the post-reboot 580.178.04 driver. This is the first launch in the project
where the GPU equivariance suite was verified against the same driver the run uses.

**Startup verified on both:** `Set random seed to <seed>` matches the requested seed; GPU assignment
confirmed by UUID rather than by `CUDA_VISIBLE_DEVICES` alone (`38z4wr9z` → `GPU-ca9f06b8`,
`sw2qwfs9` → `GPU-8d23cc42`); both buffer caches hit with **no rebuild** — `offline_buffer_cache/`
and `online_buffer_cache/` still contain only `8edf618e` and `43637c10`, both dated 2026-05-21, and
no new hash directory was created. RSS 23.7 and 23.8 GB, 47.5 GB of 125 GB, 55 GB still available.

**Robot-base centering is active**, which settles the inverted claim from the 2026-09-16 entry.
Both runs print `[robot_base_xy] Loaded from cache .../env_probes/Can.json: [-0.5, -0.1000]` at
startup. At `7dae4925` every such call site is commented out, which is where the "centering was
disabled" finding came from. At `caf83f3` it runs. **So `z8yoqylh` rotated about the robot base,
not the world origin, and the symmetry was not silently broken** — the concern raised in
[EQUIVARIANCE.md](EQUIVARIANCE.md) under "The rotation center" does not apply to the successful run.

**Early rate, and why it is slower than the failed attempt.** Critic warmup is running at ~116
ms/step against the 79.7 ms/update measured for the solo `7dae4925` run. That is expected and
mildly reassuring: `caf83f3` has `enc_degree_channel=32` where `7dae4925` had 16, so it should cost
more per step. The right comparator is `z8yoqylh`'s own measured **47.3 h**, not the failed
attempt's 34.6 h. Treat this as preliminary — the `~/repro_caf_*.log` files are block-buffered and
these numbers come from warmup, not the main loop.

**Config caveats.** Verifiable this time, and verified: the provenance table in
[ARCHITECTURE.md](ARCHITECTURE.md) and the no-op tests pin the inert `EquivarianceConfig` fields at
HEAD / `caf83f3`, and `caf83f3` is the commit being launched. The 117-field config match to
`z8yoqylh` is recorded in the pre-registration above.

---

## 2026-09-25 — RESULTS — `equi-repro-caf83f3`: both seeds reproduced

**The replication succeeded.** Both seeds ran the full 300,000 steps, neither collapsed, and both
ended above their step-0 baseline. This is the first multi-seed result in the project.

| Seed | wandb | Step-0 | Best | Final | Successful ep. length | Runtime |
|---|---|---|---|---|---|---|
| 2114495708 | `38z4wr9z` | 0.82 | **0.96** | 0.90 | 154.5 | 46.7 h |
| 1 | `sw2qwfs9` | 0.94 | **0.96** | 0.94 | 161.6 | 47.6 h |
| — | `z8yoqylh` (target) | 0.82 | 0.94 | 0.92 | 151.8 | 47.3 h |

Read the step-0 column before the best column (rule 5.5). `38z4wr9z` is the load-bearing arm: same
seed as `z8yoqylh`, it drew the same 0.82 step-0 and climbed **+0.14** to 0.96, slightly exceeding
the run it was replicating. `sw2qwfs9` drew a 0.94 base policy and gained only +0.02, so its
headline 0.94 is mostly a favourable draw rather than learning. The pair is n=2 for "does it
reproduce" and weaker than it looks for "how much does it help".

**No collapse in either run.** Zero evaluations with all 50 episodes failing, at any point in
either seed. Every previous collapse in this project was unambiguous by 10k–20k steps.

**Runtime matched the prediction.** 46.7 h and 47.6 h against the pre-registered ~47 h, which came
from `z8yoqylh`'s measured 47.3 h rather than the failed attempt's 34.6 h. The reasoning that
`caf83f3`'s wider encoder (`enc_degree_channel=32` vs 16) would cost more per step held.

**The kernel and driver deviation did not matter.** The runs executed on kernel 6.8.0-138 and
driver 580.178.04 where `z8yoqylh` ran on 6.8.0-111 and 580.173.02 — recorded in the
pre-registration as the one variable this replication could not hold fixed. Both seeds reproduced
anyway, so that difference is not load-bearing on Can.

**Episode length is now n=3 and still flat.** 151.8, 154.5, 161.6 across the three successful
equivariant runs, against the ≈105 the baseline reached on Can. The equivariant agent reaches a
comparable success rate without learning to finish faster. This was flagged as the more interesting
signal when it rested on a single run; it now rests on three, and it remains unexplained.

**What this does not establish.** Both seeds ran at `caf83f3` from the pinned worktree, so this
confirms the recovered configuration reproduces — it says nothing about HEAD, about Square, or
about whether equivariance beats the non-equivariant baseline. The baseline comparator `msfkjwab`
is still n=1 and still only on `ZXP-S-works`.

Use `/analyze-experiment` for the full metric-by-metric read; this entry records the outcome.

---

## PRE-REGISTERED, NOT YET LAUNCHED (4)

### Equivariant residual TD3 on Square at HEAD — `equi-square-v1`

- **Commit:** `3eae998` (HEAD) · **Config:** `residual_equi_td3_square_config`
- **Overrides:** `algo.prefetch_batches=4`, `agent.actor.actor_last_layer_init_scale=1e-4`
- **Task:** Square · **Host:** `boce-WS-01`, GPU 0 · **Seeds:** 1 first, then 2 and 3 sequentially
- **Launcher:** `submit.sh`, rewritten in place for this experiment

**Hypothesis.** The equivariant residual agent improves over the Square base policy. Square is the
discriminating task: its base policy sits at 0.52 (`870ws2c2`), where Can's starts at 0.82–0.94 and
leaves almost no headroom to measure.

**Decided by.** `eval/success_rate` at step 0 versus best versus final, and
`eval/mean_successful_episode_length`. A run whose best equals its step-0 learned nothing.

**Held fixed.** `n_step=3`, `gamma=0.99`, `buffer_size=200k`, `learning_starts=10k`,
`action_scale=0.1`, stddev 0.05, `num_episodes=300`. Base policy is the Diffusion checkpoint
`robomimic-square-bc/3hzs5bz1`, artifact `run_3hzs5bz1_best:v9`, verified as `type=diffusion`,
`n_obs_steps=2`, `chunk_size=16` — the same policy type as Can's `xhjdl8a7`, so base-policy type is
not confounded with task.

**Why `actor_last_layer_init_scale` is passed explicitly.** HEAD defaults it to `0.0`. Every run in
the `equi-repro-caf83f3` pair and both recovered successes (`z8yoqylh`, `36pfxsww`) resolved to
`1e-4`, because that was the default at `caf83f3`. Launching at HEAD without the override would
silently change the setting away from the one behind the only verified n=2 result. Passed on the
CLI so the resolved config states the intent rather than inheriting it.

**Not held fixed, stated because it matters.** This runs at **HEAD**, not at `caf83f3`. It is
therefore *not* a controlled comparison against `38z4wr9z` / `sw2qwfs9`. It tests whether the
algorithm works on a harder task; it does not isolate Square-versus-Can.

**Comparator.** `870ws2c2`, the ResFiT baseline on Square: 0.52 → 0.90, single seed, commit
`debc9b6c`. The 1-versus-n asymmetry is acknowledged in advance rather than discovered later.

**The abort condition.** The March 2026 Square runs logged step-0 evaluations of **0.00** on a task
whose base policy measures 0.52, then recovered to 0.48–0.66 in late April with no recorded config
change. That was never diagnosed. **If step 0 is not near 0.52, this run is void** and the
observation pipeline is the suspect, not the algorithm.

**Gate, run 2026-09-25 at `3eae998`:** GO. Tier 1 clean (torch 2.6.0, torchrl 0.7.0, escnn 1.0.11,
two RTX 4090s, working tree clean); Tier 2 clean (69 static + 79 equivariance tests); Tier 3 —
BC policy cached and resolving in wandb, robot base probe `Square.json` = `[-0.56, 0.0]` from a
single `robot0_base` with identity orientation, 127 GB RAM and 71 GB disk free. Two expected
warnings: both Square buffer caches miss and will be built on the first run —
**offline `1d31ed9b`** (~5 GB) and **online `66c46c51`** (~16 GB). Cross-check those two hashes
against what the training script prints at startup; a mismatch means `preflight.cache_hashes()` has
drifted from the meta dicts it mirrors.

---

## 2026-09-25 — CORRECTION — `equi-square-v1` launched at `b8e1de7`, not `3eae998`

The pre-registration above records `3eae998`. The run launched at 09:17:27; `b8e1de7`
was committed at 09:14:31, three minutes earlier, so `b8e1de7` is what ran.

**No Python changed between the two commits** — the diff is `docs/EXPERIMENTS.md` and
`submit.sh` only. So the code is identical and the result stands. The hash is corrected
here rather than edited above, per rule 5.6, and because this project's standing lesson
is that a recorded commit hash is not self-verifying.

Seeds 2 and 3 will be launched from a worktree pinned at `b8e1de7`
(`~/projects/equi-resrl-square`), so the seed group runs one commit throughout while
HEAD continues to change for the BoxCleanup work.

---

## 2026-09-25 — CORRECTION — `equi-square-v1` seed 1 was restarted at `2611adb`

The run pre-registered and launched as `0prso90n` **is dead**, and the seed-1 arm of
`equi-square-v1` is now wandb run **`j04yoeui`**, launched 2026-09-25 11:45 EDT.

| | Original | Replacement |
|---|---|---|
| wandb | `0prso90n` | **`j04yoeui`** |
| Launched | 09:17:22 at `b8e1de7` | 11:45 at `2611adb` |
| Died at | step 6,400 of 300,000 | — |
| Step-0 | 0.62 | (see the run) |

**Why it died.** It was launched in the foreground of an SSH session, not under tmux. While
preparing to detach, a multi-line paste landed on its controlling terminal and interrupted it; the
log ends in `KeyboardInterrupt`. Nothing about the configuration or the machine was implicated.

**Why the commit moved, and why it does not matter scientifically.** `2611adb` touches
`docs/`, `tests/test_regression_single_arm.py`, `preflight.py`, a BC launcher and
`env_probes/Square.json`. **No file on the training import path differs between `b8e1de7` and
`2611adb`**, so the replacement runs the same code as the original. Recorded here rather than by
editing the pre-registration, per rule 5.6.

**The original's step-0 of 0.62 stands as a measurement.** It is +0.10 against `870ws2c2`'s recorded
0.52 for the same base policy, which at n=50 episodes is about 1.4 standard errors — within noise.
It is not the 0.00 anomaly the gate watches for.

**Process note.** Both currently-running jobs are now under tmux. Launching a multi-day run in a
bare SSH foreground is how this one was lost.

---

## 2026-09-27 — LAUNCHED — `equi-square-v1` seed 2 (`88i0lpc8`)

| Seed | wandb run | GPU | Commit | Launched |
|---|---|---|---|---|
| 2 | **`88i0lpc8`** | 1 (`GPU-8d23cc42`) | `b8e1de7` | 2026-09-27 10:19 EDT |

The second seed of the group pre-registered above ("PRE-REGISTERED, NOT YET LAUNCHED (4)"),
launched as the 2026-09-25 correction said: from the worktree pinned at `b8e1de7`
(`~/projects/equi-resrl-square`), with `CACHE_DIR=~/projects/equi-resrl`, via that tree's
`./submit.sh 2 1`, inside tmux session `sq-ref-s2`. Project `robomimic-square-final`, name
`equi-square-v1-2`, the same overrides as seed 1.

**Gate:** GO from `b8e1de7`'s own `preflight.py`. Tier 1 69 passed, Tier 2 passed on GPU 1, 82 GB
RAM and 67 GB disk available. Its two warnings, both Square caches missing, were expected: that
gate looks for caches in the worktree, not under `CACHE_DIR`.

**Startup verified:**
- `Set random seed to 2`.
- Both caches hit in the main checkout, with no rebuild: offline `1d31ed9b` (80,429 transitions) and
  online `66c46c51` (10,000, the warm-up). No "Populating offline buffer" line.
- `[robot_base_xy] Loaded from cache .../equi-resrl/env_probes/Square.json: [-0.56, 0.0]`.
- The BC artifact `run_3hzs5bz1_best:latest` resolved to `v9`, the only version on disk.
- The process sits on `GPU-8d23cc42`.
- Step-0 eval **0.64** (32/50), against seed 1's 0.62.

**It trains the same agent as seed 1.** `tools/agent_fingerprint.py` builds the whole equivariant
agent at this config and records its parameters, its actor and critic outputs, and one TD3 update.
All 22 values are exactly identical at `b8e1de7` and at `f3e6315`. Seed 1 ran `2611adb`, which
changed no training-path file relative to `b8e1de7`. `tests/test_regression_agent.py` now pins those
values.

**Config caveats.** As for seed 1: `num_actor_layers` and `num_critic_layers` are logged and dead;
`initialize`, `use_norms` and `use_orth_init` are partial (ARCHITECTURE.md provenance table).
`action_scale=0.1` does not bound the residual; `equi_clip(bound=1.0)` does.

---

## 2026-09-27 — CORRECTION — `seed` never reaches the environments

**No run in this project has seeded its env initial states.** `cfg.seed` seeds `random`, numpy and
torch in the trainer process: network init, exploration noise and replay sampling. Every
simulated episode, training and eval alike, starts from an initial state drawn from OS entropy.

Why, at every commit that has run (`7dae4925`, `caf83f3`, `b8e1de7`, and HEAD):
- `create_vectorized_env` builds a gymnasium `AsyncVectorEnv` with `context="spawn"`. Each worker is
  a fresh interpreter, and numpy seeds its global RNG from OS entropy.
- robosuite's placement samplers draw object poses from that global RNG
  (`robosuite/utils/placement_samplers.py`).
- The trainer's `if hasattr(env, "seed"): env.seed(cfg.seed)` and `eval_env.seed(cfg.seed + 1)`
  never run, because gymnasium 1.1.1's vector envs have no `seed()`. The wrapped env's own `seed()`
  and `reset(seed=...)` ignore the seed anyway.

**Measured 2026-09-27.** Two processes, each seeded `random`/numpy/torch = 0, built 2-env Square
vector envs and reset them. They got different initial states: first env end-effector xy
`(-0.083, 0.010)` vs `(-0.113, -0.008)`.

**What this changes:**
- The 2026-09-18 pre-registration's point 2 holds only for the trainer process.
  `38z4wr9z` reproduced `z8yoqylh`'s network init and noise streams, not its episodes. Its point 3
  gave `torch_deterministic=False` as the reason runs differ; the unseeded environments are the
  larger reason, and they would remain with deterministic torch.
- Step-0 evals of different runs are independent draws, even at the same seed. That makes them
  poolable as base-rate samples.
- A run cannot be re-run to the same trajectory, whatever the seed.

Nothing is fixed here. Seeding the workers would change every future run's episodes relative to
past ones, so it needs its own decision.

---

## 2026-09-27 — CODE — BoxCleanup prep: ACT reset fixed, base policy pinned

Two changes on the BoxCleanup path. **Neither changes a Can or Square run.** No residual run has used
either path.

**1. The ACT reset crash (TODO B3) is fixed.** `BasePolicyVecEnvWrapper.step` passed an undefined
`terminated_envs`, so an ACT base policy raised `NameError` on the first successful episode. It now
passes `reset_ids`, the terminated envs, as upstream ResFiT does. The diffusion branch still resets
every env's queues on any termination, which every Can and Square run did.
`tests/test_env_wrapper.py` fails on the old code and passes on the fix. Unchanged, and now pinned by
a test: a timeout (`truncated`) does not reset the base policy.

**2. `ResidualTD3BoxCleanConfig.base_policy` is pinned** (STANDARDS rule 1.6). It was
`wandb_id="TODO"`. It is now `dexmg-boxcleanup-bc/xg0w2r0z`, `wt_type="best"`,
`wt_version="v7"`. The equivariant BoxCleanup config inherits it.
- **v7 is the step-24,000 checkpoint** (`trainer_state.pt`): ACT, chunk 20, 20 action steps,
  `n_obs_steps` 1, no temporal ensembling, three cameras. It was created 2026-09-25 17:02 UTC and
  is the last of eight `best` versions, aliased `best` and `latest`.
- The version is pinned, not `latest`, so the config determines the weights.
- sha256 of `policy/model.safetensors`: `e79fc307961e81135440c5eec84b2c592c4886cdac02ad6554809829602c67c9`.
  The backup `~/equi-resrl-preserve/run_xg0w2r0z_best:v7` matches the download.

---

## 2026-09-29 — RESULTS — `equi-square-v1`, both seeds: the reference config degrades Square

Pre-registered as "PRE-REGISTERED, NOT YET LAUNCHED (4)". Read the target-critic CORRECTION below
first: both seeds trained against a target critic frozen at initialization.

| | Seed 1 `j04yoeui` | Seed 2 `88i0lpc8` |
|---|---|---|
| Commit | `2611adb` | `b8e1de7` (pinned worktree) |
| Runtime | 47.2 h | 46.6 h |
| Step-0 | 0.62 | 0.64 |
| Best | 0.72 at 70k | 0.64 at 0 |
| Final (300k) | 0.54 | 0.28 |
| L10 (210k–300k) | **0.52** | **0.39** |
| Mean of 110k–150k | 0.548 | 0.412 |

- **Eval sequences**, 50 episodes each at 0, 10k, …, 300k, recomputed from `output.log`:
  - `j04yoeui`: 0.62, 0.56, 0.54, 0.60, 0.54, 0.52, 0.54, 0.72, 0.58, 0.64, 0.56, 0.52, 0.60, 0.54,
    0.54, 0.54, 0.50, 0.46, 0.44, 0.46, 0.56, 0.54, 0.44, 0.40, 0.50, 0.64, 0.56, 0.50, 0.42, 0.66,
    0.54.
  - `88i0lpc8`: 0.64, 0.52, 0.54, 0.56, 0.42, 0.54, 0.50, 0.38, 0.44, 0.44, 0.36, 0.42, 0.36, 0.44,
    0.44, 0.40, 0.44, 0.34, 0.58, 0.46, 0.38, 0.44, 0.46, 0.46, 0.42, 0.40, 0.32, 0.42, 0.34, 0.40,
    0.28.
- **Base rate.** The pooled step-0 of four runs on this base policy (`0prso90n` 0.62, `j04yoeui`
  0.62, `88i0lpc8` 0.64, `870ws2c2` 0.52) is **0.60 ± 0.035** (200 episodes). Step-0 evals are
  independent draws (2026-09-27 CORRECTION), so they pool.
- **Against base:** seed 1 −0.08 (about 2 combined SE), seed 2 −0.21 (about 5). **M_ref = 0.46.**
- **The residual saturates.** `residual_l1` ended at 0.75–0.79 in both seeds, against `870ws2c2`'s
  0.085: ResFiT's actor ends in Tanh × `action_scale`, and this one is bounded only by
  `equi_clip(bound=1.0)`.
- **Q estimates are far outside the reward range.** `actor_loss_base` (−mean Q) spans −7.9 to +11.5
  in seed 1 (98% within −2.0 to +5.1), on 0/1 rewards.
- **The working baseline dips before it learns.** `870ws2c2` went 0.52, then 0.24–0.38 until 70k,
  rose from 80k, averaged 0.68 over 110k–150k and reached L10 0.83. The equivariant seeds never
  rose.
- **Context, not a cause:** the base policy `3hzs5bz1` was still improving when its 50k-step BC budget
  ran out (10k-window means 0.16, 0.40, 0.49, 0.50, 0.54; best `v9`, 0.60, at 49k). `870ws2c2`
  lifted this same base to 0.90.
- **Config caveats:** as in the seed-2 launch entry. Also `num_q=2` and a frozen target critic (both
  corrections below).
- **Verdict:** no gain in either seed, a clear loss in one, n=2. Because of the frozen target critic
  this measures that bug, not the equivariant residual.

---

## 2026-09-29 — CORRECTION — the equivariant target critic never changed from its initialization

**Every equivariant residual run in this project computed its TD targets with the critic as it was
at initialization.** The target critic's parameters were soft-updated every step, but it never
computed with them.

**Why:**
- escnn 1.0.11's `nn.Linear` expands its basis coefficients into a weight matrix on every forward in
  train mode. In eval mode it computes with a `matrix` buffer, expanded once, when `.train(False)` is
  called.
- `QAgent.__init__` put `critic_target` in eval mode once, at construction. Nothing called it again:
  `QAgent.train()` toggles only the encoder, actor and critic, and `agent.eval()` and
  `utils.eval_mode(agent)` both go through it.
- `utils.soft_update_params` copies `parameters()`, which are the basis coefficients, not the expanded
  buffers.
- Every layer of the equivariant critic is an escnn `Linear` or `GroupPooling`, so the whole target
  critic was frozen.

**Scope:**
- The same code is at every commit that ran. `critic_target.train(False)` dates from `798e237`
  (2026-01-15), and it and the `train()` override are unchanged at `7dae4925`, at `caf83f3` (`z8yoqylh`,
  `38z4wr9z`, `sw2qwfs9`) and at `b8e1de7` and `2611adb` (`j04yoeui`, `88i0lpc8`).
- Unaffected: the actor target stays in train mode and re-expands on every forward, and the ResFiT
  baseline uses plain PyTorch layers.

**Measured 2026-09-29** on the reference Square agent:
- The online critic was perturbed and then copied into the target with `tau=1`.
- The target's outputs on the fixture batch were bit-identical before and after the copy: 0.6126,
  0.4362, 0.4747, 0.6222.
- Forcing a re-expansion changed them to about 1,200.
- `tests/test_regression_agent.py::test_target_critic_computes_with_its_soft_updated_weights`
  reproduces both behaviours.

**What this changes:**
- **The critic never bootstrapped.** The TD target in every equivariant run was
  `r + γ³ · min Q_init(s′, a′)`, where `Q_init` is a fixed, randomly initialized network, not a
  learned value. So the critic learned an n-step reward plus a fixed random function of the next
  state and action, and the actor climbed that function's action gradient.
- **That fits the Square failure,** but only as a reading, not a test: a residual 9× ResFiT's, and Q
  estimates from −7.9 to +11.5 on 0/1 rewards. The `target-fix` arm of the sweep pre-registered
  below tests it.
- **Every equivariant-versus-baseline comparison so far set an agent that could not bootstrap
  against one that could.** The Can replication (2026-09-25 RESULTS) stands as a measurement of the
  frozen-target agent, not of the method.
- **The 2026-09-27 fingerprint could not see this:** an update reads the target before it
  soft-updates it.

**The fix, from the commit that adds this entry:**
- The equivariant target critic runs in train mode, as the actor target does, so it re-expands its
  weights from its current parameters on every forward.
- The reference critic has no dropout or batch norm, so train mode changes nothing else. A
  batch-norm critic, which the sweep adds, then normalizes each target batch with its own
  statistics. Eval mode would read running statistics, which the soft update never copies.
- **New field `equivariance.frozen_target_critic`, default `False`.** `True` reproduces the old
  behaviour, for comparison runs.
- This is a behaviour change at an unchanged config (STANDARDS 1.6): a run config without this field
  ran frozen.
- A checkpoint saved before this commit no longer loads with `strict=True`: its target critic holds
  `matrix` buffers that a train-mode target doesn't have. No run resumes from one.

**Runs affected:** every equivariant run. The agent fingerprint is unchanged, because nothing else
about the agent changed.

---

## 2026-09-29 — CORRECTION — the critic-head default changed from 10 to 2 on 2026-04-23, unrecorded

- `aabde0f` ("transferring to new compute", 2026-04-23) changed `CriticConfig.num_q` from 10 to 2
  with no note here, which breaks STANDARDS 1.6.
- Upstream ResFiT (`6bd9f7c`) defaults to 10, and the paper launchers never override it.
- **10 heads**, per their wandb configs: the two working baselines, `870ws2c2` (Square) and
  `msfkjwab` (Can).
- **2 heads:**
  - every equivariant run checked: `z8yoqylh`, `38z4wr9z`, `sw2qwfs9`, `j04yoeui` and `88i0lpc8`;
  - the non-equivariant Can baseline `zyed0lew` (2026-05-15, project `robomimic-can-final`, local
    `config.yaml`), which improved 0.76 → 0.94 with 2 heads, n=1.
- **What it changes:** the comparisons against `870ws2c2` and `msfkjwab` differed in the critic
  ensemble as well as in the network and the target critic. The 2026-09-25 RESULTS line "baseline
  residual TD3 works" stands, but its Square evidence is at 10 heads.
- **Not changed:** the default stays 2, because every equivariant run used 2. A run meant to match
  the paper passes `agent.critic.num_q=10`. The sweep's `q10` arm tests it on Square.

---

## 2026-09-29 — NOTE — `rj70yeid`, a stopped BoxCleanup BC attempt

- A diffusion BC policy on TwoArmBoxCleanup (`train_bc_dexmg --policy diffusion`, 50,000 steps
  planned), launched 2026-09-25 09:36 at `b8e1de7`. Local `wandb/run-20260925_093615-rj70yeid`.
- It was stopped at step ~300: the last log line is 09:44:21, loss 0.131. That was when the base
  policy switched to ACT. Superseded by `xg0w2r0z`, the ACT policy pinned in the 2026-09-27 CODE entry.
- Nothing was measured. Recorded so that every run directory is accounted for.

---

## 2026-09-29 — PRE-REGISTERED — the Square sweep (`equi-square-sweep-<arm>`)

**Question.** Does fixing the target critic let the equivariant residual improve on Square's base
policy? And which single further change helps most?

**Stage 1, the screen (this entry):**
- One seed (seed 1) per arm, stopped at 150k steps (`algo.total_timesteps=150000`).
- Nothing in training depends on `total_timesteps` before the stop: the stddev schedule is flat at
  0.05, clipping and LR warm-up are off. These are the first 150k steps of a 300k run.
- **Verdict:**
  - An arm **advances** iff the mean of its five evals at 110k–150k (250 episodes) is at least B_sq.
  - Collapse (three consecutive evals ≤ 0.05) stops a run at any step, and it does not advance.
  - No other early stop.
- **B_sq** is the 500-episode base rate that `eval_base_policy` measures at the launch commit
  before the first arm (10 × 50 episodes, seed 1000; recorded as its own entry). The pooled step-0
  (0.60 ± 0.035) is only a sanity check on it.
- **Calibration against B = 0.60:**
  - the working ResFiT baseline `870ws2c2` would advance (0.68);
  - both frozen-target reference seeds would not (0.548 and 0.412).
- **What it can detect:** only large effects. The two reference seeds differ by 0.14 in this window.
  "Does not advance" rules out a large gain at n=1, and nothing more. **No stage-1 number is a
  comparison** (STANDARDS 5.3).

**Stage 2, pre-registered separately when an arm advances:**
- Seeds 1 and 2 at 300k. `target-fix` is judged against B_sq and against the frozen reference
  M_ref = 0.46. Every other arm is judged against `target-fix` at the same seeds.
- Promotion needs all three of:
  - M_arm − B_sq > 0.055;
  - M_arm at least its comparison's M plus 0.10;
  - both seeds' L10 above its comparison's M.
- Promoted arms get a 3-seed confirmation.

**Arms.**
- Every arm is `residual_equi_td3_square_config` plus `algo.prefetch_batches=4`,
  `algo.total_timesteps=150000`, init scale 1e-4 unless stated, and the overrides below.
- **Every arm includes the target-critic fix** (the default from this entry's commit). The
  reference seeds did not.
- Each arm's case in `submit.sh` holds exactly these overrides, and lands in the commit that adds
  its field.

| Arm | Overrides | Question |
|---|---|---|
| `target-fix` | none | Does the fix alone make the residual improve on base? The reference for every other arm |
| `critic-ln` | `equivariance.critic_norm=field_norm` | ResFiT's critic has LayerNorm after every hidden layer; ours has no normalization. `FieldNorm` (per-field layer norm) at the three hidden sites: visual projection, trunk, head |
| `critic-iid` | `equivariance.critic_norm=iid_bn` | Batch norm (`IIDBatchNorm1d`) in the critic at the same sites |
| `actor-ln` | `equivariance.actor_norm=field_norm` | `FieldNorm` instead of `IIDBatchNorm1d` at the actor's three sites |
| `actor-none` | `equivariance.actor_norm=none` | Does the actor's batch norm hurt? |
| `enc-inner` | `equivariance.encoder_norm=inner_bn` | `InnerBatchNorm` after each agentview conv (the sites commented out today); biases stay off |
| `q10` | `agent.critic.num_q=10` | Upstream's 10 critic heads |
| `squash` | `equivariance.residual_squash=true` | Bound the residual by `action_scale`, as ResFiT's Tanh does: tanh of each rotating pair's norm and of each scalar, then × `action_scale` |
| `matched` | `agent.critic.num_q=10 equivariance.residual_squash=true equivariance.actor_norm=field_norm equivariance.critic_norm=field_norm equivariance.critic_orth_init=true`, init scale 0.0 | ResFiT's settings, wherever they have an equivariant counterpart |
| `clipq` | `agent.clip_q_target_to_reward_range=true` | TD targets clamped to [0, 1] |
| `l2` | `agent.actor.action_l2_reg_weight=0.1` | An L2 penalty pulling the residual towards zero |
| `orth-critic` | `equivariance.critic_orth_init=true` | Delta-orthonormal init in the critic too (it is He today) |
| `critic-last` | `equivariance.critic_last_layer_init_scale=1e-3` | Each critic head's final layer scaled down at init, so Q starts near 0 |
| `he-all` | `equivariance.use_orth_init=false` | He init in the actor too (He is escnn's default) |
| `orth-enc` | `equivariance.encoder_orth_init=true` | Delta-orthonormal init in the agentview encoder |
| `one-proj` | `equivariance.use_visual_proj=false` | Do the actor and critic need both a visual projection and a trunk projection? |
| `critic-deep` | `equivariance.num_critic_layers=2` | Two hidden layers per critic head, as ResFiT has. The field is dead today, and its wired default is 1, what ran |
| `narrow` | `equivariance.enc_degree_channel=16 equivariance.actor_degree_channel=64 equivariance.critic_degree_channel=64` | Is capacity the limit? The wrist features stay at 32 through a new `equivariance.ih_out_channel` (default 32) |

**Order.** Aden asked for the normalization arms by Friday 2026-10-02, so those go first:
- **GPU 0:** `target-fix`, `actor-ln`, `enc-inner`, `q10`, `matched`, `clipq`, `orth-critic`,
  `critic-last`, `orth-enc`, `critic-deep`.
- **GPU 1, after the BoxCleanup smoke run:** `critic-ln`, `critic-iid`, `actor-none`, `squash`, `l2`,
  `he-all`, `one-proj`, `narrow`.
- A queue item whose code isn't committed when its GPU frees is skipped until it is. The queue
  order may change; each LAUNCHED entry records when the arm ran.

**Held fixed:**
- Everything not overridden: n_step 3, gamma 0.99, buffer 200k, learning_starts 10k, critic warm-up
  10k; `action_scale` 0.1, stddev 0.05, actor LR 1e-6, critic LR 1e-4, 300 demo episodes.
- 50-episode evals every 10k.
- BC `run_3hzs5bz1_best:v9`, and the Square caches `1d31ed9b` and `66c46c51`.
- `boce-WS-01`, with torch 2.6.0, torchrl 0.7.0 and escnn 1.0.11. Env initial states unseeded.

**Not held fixed:**
- **The commit.** The arms launch from up to three commits as their code lands, and each LAUNCHED
  entry names its commit. `tests/test_regression_agent.py` must pass unchanged at each of them: the
  defaults then build the same agent, bit for bit, as at `f3e6315` and `b8e1de7`, apart from the
  target critic.
- **Against the reference seeds:** the target-critic fix, and the commit.
- **Load:** the BoxCleanup smoke run shares the machine with the first arm for a few hours.

**Resources.**
- About 23.5 h per arm at 0.56 s/step; `q10` and `matched` about 28 h (+22% per update, measured).
- Two arms at a time: 40.3 GiB each, measured.
- 18 arms is about 9 days on both GPUs.

**Records.** A LAUNCHED entry per commit's batch of arms, and a RESULTS entry per finished arm: its
eval sequence, the 110k–150k mean and its verdict.

---

## 2026-09-29 — MEASUREMENT — Square base rate B_sq = 0.508 ± 0.022

- `eval_base_policy --config-name residual_equi_td3_square_config --num-evals 10 --seed 1000`, at
  `ce96bb5` from its pinned worktree, on `boce-WS-01`, GPU 0, 17:46–17:58.
  Record: `outputs/base_rates/Square__3hzs5bz1_best_latest__seed1000.json`.
- BC `run_3hzs5bz1_best:latest`, which resolves to `v9`, the only version on disk and the one every
  Square run loaded. 8 eval envs, as in the trainer. Env initial states are unseeded.
- **Per-eval rates:** 0.52, 0.52, 0.38, 0.46, 0.48, 0.56, 0.58, 0.48, 0.60, 0.50.
  **Mean 0.508 ± 0.022** (binomial SE, 500 episodes). The spread between evals matches 50-episode
  binomial noise.
- **It disagrees with the pooled step-0.** The four runs' step-0 evals gave 0.60 ± 0.035 (200
  episodes), and a 50-episode check with this script on 2026-09-27 gave 0.44. The script's 550
  episodes against the step-0 200 give z ≈ 2.5. Nothing systematic was found:
  - the script builds the trainer's eval env, with the same env count, wrapper, identity normalizers
    and `run_dexmg_evaluation`;
  - it loads the policy the same way;
  - its residual is exactly zero, where the reference runs' step-0 residual was ~1e-4.
  So B_sq stands as pre-registered. Every sweep arm's own step-0 eval adds 50 episodes from the
  trainer's path, and the RESULTS entries pool them as a running check.
- **What it does to the screen:** see the amendment below.

---

## 2026-09-29 — AMENDMENT — Square sweep pre-registration, before any affected arm launched

Two changes to "2026-09-29 — PRE-REGISTERED — the Square sweep". Only `target-fix` had launched
when this was written, and neither change touches it.

**1. `critic-ln`, `actor-ln` and `matched` use `layer_norm`, not `field_norm`.**
- `field_norm` (`FieldNorm`) normalizes each 8-channel field by its own statistics. ResFiT's
  LayerNorm normalizes by one mean and variance over all a layer's channels.
- The new `RegularLayerNorm` (`equivariance.*_norm=layer_norm`) does exactly that, with a scale
  and shift shared by each field's channels so that it stays equivariant. At its initial scale and
  shift it equals `torch.nn.functional.layer_norm`, per a test.
- It is the faithful counterpart of what the arms ask about, so the arms use it.
- `field_norm` stays available, and tested, but no arm uses it.
- Overrides now: `critic-ln` `equivariance.critic_norm=layer_norm`, `actor-ln`
  `equivariance.actor_norm=layer_norm`, and `matched` with `layer_norm` for both.

**2. The calibration, at the measured B_sq = 0.508 instead of the assumed 0.60.**
- `870ws2c2` (0.68) still advances, and frozen-target seed 2 (0.412) still does not.
- **Frozen-target seed 1 (0.548) would now advance.** At this bar the screen passes an arm that
  merely holds the base rate, so it is looser than the entry's calibration said.
- The verdict rule is unchanged, as pre-registered. Stage 2 does the discriminating: it requires
  +0.055 over B_sq and +0.10 over `target-fix`.
- Every RESULTS entry reports the arm's 110k–150k mean against both 0.508 and 0.60, so a reader can
  apply either.

---

## 2026-09-29 — CODE — the sweep's options, with defaults that build the reference agent

New `EquivarianceConfig` fields.
- **Normalization:** `actor_norm` (default `iid_bn`), `critic_norm` (`none`) and `encoder_norm`
  (`none`). They replace `use_norms` on the equivariant path, and `use_norms` is now read only by
  the scalar ablation. The actor's norm now reaches all three of its sites. Before, one site
  ignored `use_norms`, so a run with `use_norms=False` (`qe2by47h`) still had one batch norm.
- **Init:** `critic_orth_init` and `encoder_orth_init` (both `False`), and
  `critic_last_layer_init_scale` (`None`). `use_orth_init` stays the actor's.
- **Architecture:**
  - `use_visual_proj` (`True`);
  - `residual_squash` (`False`), with the new equivariant `equi_tanh`;
  - `ih_out_channel` (32), the wrist-feature width, until now tied to `enc_degree_channel`.
- **`num_critic_layers` is wired**, and its default changes from 2 to 1 (STANDARDS 1.6). While dead
  it logged 2, and every critic head had one hidden layer, which the default still builds.

**Held fixed:**
- At the defaults, `tests/test_regression_agent.py` passes unchanged: the whole agent (parameters,
  outputs and one update) is identical to `f3e6315` and `b8e1de7`.
- `tests/test_regression_single_arm.py` passes too.
- The encoder stays bias-free whatever `encoder_norm` is. Its old `bias=not use_norms` coupling
  (STANDARDS 1.3) is gone.

**Tests.**
- Every option keeps the symmetry, in train and eval mode.
- Every sweep arm in `submit.sh` composes, and builds and updates on the GPU (`tests/test_sweep_arms.py`).
- `test_no_ops.py` pins which sites each field reaches.
- On the GPU, cuDNN's TF32 convolutions put about 1e-3 of relative noise on the encoder. The new
  encoder tests turn TF32 off and assert 1e-4.

**Config caveats for runs from this commit:**
- `num_actor_layers` stays dead.
- `agent.actor.dropout`, `agent.critic.dropout` and `agent.critic.loss` stay dead on the equivariant
  path.
- `initialize` reaches only the encoder.

---

## 2026-09-29 — NOTE — what the frozen target critic still saw, with evidence from the runs

A precision on the target-critic CORRECTION above:
- **The frozen part was the critic's layers, not its inputs.** The target Q reads next-state features
  from the online encoder (`self.enc(next_obs)` under `no_grad`), and that encoder trains on the
  critic loss.
- So the TD target was `r + γ³ · min Q_init(φ(s′), a′)`: a fixed random critic applied to learnt
  features, and still no bootstrapping of the critic's own values.

**The runs show it.** Here is the logged batch mean of the TD target (`train/critic_qt`) during the
10k critic-only warm-up updates, when the actor is fixed:

| Run | 1k | 5k | 8.7k | 9.9k |
|---|---|---|---|---|
| `j04yoeui` (frozen) | −0.150 | −0.158 | −0.162 | −0.164 |
| `88i0lpc8` (frozen) | 0.083 | −0.034 | −0.055 | −0.053 |
| `0rehtuoq`, `target-fix` (fixed) | −0.138 | −0.052 | −0.005 | — |

- `j04yoeui`'s target barely moves. `88i0lpc8`'s settles by 5k, as its encoder does.
- `target-fix`'s keeps rising towards the reward's scale as the target learns.
- These are single runs, so this illustrates the fix and doesn't test it.

---

## 2026-09-29 — LAUNCHED — Square sweep `target-fix` seed 1 (`0rehtuoq`)

| Arm | Seed | wandb | GPU | Commit | Launched |
|---|---|---|---|---|---|
| `target-fix` | 1 | **`0rehtuoq`** | 0 (`GPU-ca9f06b8`) | `ce96bb5` | 2026-09-29 17:58 EDT |

- **How:** `tools/gpu_queue.sh 0 ~/queues/gpu0.txt` in tmux `gpu0`, from the worktree pinned at
  `ce96bb5` (`~/projects/equi-resrl-ce96bb5`), with `CACHE_DIR=~/projects/equi-resrl`, after the
  base-rate measurement on the same GPU. Name `equi-square-sweep-target-fix-1`, project
  `robomimic-square-final`.
- **Gate:** GO. Tier 1: 116 passed. Tier 2: 154 passed. Both caches found under `CACHE_DIR`, which
  `--require-caches` made blocking.
- **Startup verified:**
  - `Set random seed to 1`;
  - offline `1d31ed9b` (80,429) and online `66c46c51` found on disk, with no rebuild;
  - the probe centre is (−0.56, 0.0);
  - the resolved config has `num_q: 2` and `frozen_target_critic: False`;
  - the process sits on `GPU-ca9f06b8`.
- **Step-0 eval 0.60** (30/50). The trainer-path step-0 pool is now 150/250 = 0.60, against the
  base-rate script's 0.508 (MEASUREMENT above).
- **Config caveats:**
  - `num_actor_layers` and `num_critic_layers` are logged and dead at this commit. The critic has
    one hidden layer per head.
  - `use_norms` is partial, as for the reference seeds.
  - `action_scale` does not bound the residual.
- **Shares the machine** with the BoxCleanup smoke run for about its first 2.5 h.

---

## 2026-09-29 — LAUNCHED — Square sweep `critic-ln` seed 1 (`qlaonpv4`)

| Arm | Seed | wandb | GPU | Commit | Launched |
|---|---|---|---|---|---|
| `critic-ln` | 1 | **`qlaonpv4`** | 1 (`GPU-8d23cc42`) | `1e33b8f` | 2026-09-29 18:58 EDT |

- **How:** `tools/gpu_queue.sh 1 ~/queues/gpu1.txt` in tmux `gpu1`. It started as soon as the
  BoxCleanup smoke run ended, and ran `./submit.sh critic-ln 1 1` in the worktree pinned at
  `1e33b8f` (`~/projects/equi-resrl-1e33b8f`), with `CACHE_DIR=~/projects/equi-resrl`. Name
  `equi-square-sweep-critic-ln-1`.
- **Overrides:** the pre-registered ones, as amended: `equivariance.critic_norm=layer_norm`, init
  scale 1e-4, 150k steps.
- **Gate:** GO. Tier 1 and Tier 2 passed; Tier 2 included building this arm's agent and running one
  update.
- **Startup verified:**
  - `Set random seed to 1`;
  - offline `1d31ed9b` (80,429) and online `66c46c51` found on disk, with no rebuild;
  - the probe centre is (−0.56, 0.0);
  - the resolved config has `critic_norm: layer_norm`, `actor_norm: iid_bn` and
    `frozen_target_critic: False`;
  - the process sits on `GPU-8d23cc42`.
- **Step-0 eval 0.44** (22/50). The trainer-path step-0 pool is now 172/300 = 0.573; the base-rate
  script gave 273/550 = 0.496.
- **Config caveats:** `num_actor_layers` stays dead. The critic's `layer_norm` follows its visual
  projection, trunk and each head's hidden layer (`test_no_ops.py`).

---

## 2026-09-30 — CODE — `actor_norm` defaults to `none`, and the update no longer waits for the GPU

**The default change (STANDARDS 1.6): `equivariance.actor_norm`, `iid_bn` → `none`.**
- Every run before this commit had escnn's `IIDBatchNorm1d` at all three of the actor's sites. No
  run ever compared it against `none`, and the SO(2) equivariant RL code this project follows has
  no normalization anywhere.
- **A run from this commit that does not set `actor_norm` has no normalization in the actor.**
  That includes `submit.sh`'s arms, which are unchanged. At this commit `target-fix` builds what the
  pre-registered `actor-none` arm is, and every other arm differs from its pre-registered config in
  the actor too.
- So a run from this commit that changes one field differs from `target-fix` (`0rehtuoq`, at
  `ce96bb5`, `iid_bn`) in two. A reference at the new default, the `actor-none` arm, is the
  one-change comparison (TODO B9).
- `tests/test_regression_agent.py` pins both. The reference, now with `iid_bn` set explicitly, keeps
  its golden from `f3e6315`. A second golden pins `actor_norm=none`, and `1e33b8f` gives the same
  22 values.

**The speed-up: no trained number changes.** An env step's learning (one `act()` and four updates,
the last with the actor) made 487 host syncs, each one a point where the CPU waits for the GPU to
finish all the work queued so far. It now makes none. ARCHITECTURE.md, "Why the update used to
stall", has the breakdown.
- **escnn fixes**, in the new `equi_off_policy/common_utils/escnn_fixes.py`: escnn's basis-expansion
  modules hash by identity instead of comparing their sampled bases, and the critic's
  `GroupPooling` slices with Python ints instead of GPU scalars. Both are specific to escnn 1.0.11,
  which the environment pins.
- **Metrics only on the logged update.** `QAgent.update`, `update_critic` and `update_actor` take
  `compute_metrics`. The trainer passes `True` only to the last update before each logging step,
  whose metrics are the ones it logs, and `data/batch_terminal_R` and `data/terminal_share` follow
  it. Warm-up computes them every 100 updates, where it prints them. The `q_base_only` critic
  forward still runs on every actor update, since skipping it would change the running statistics
  of an `iid_bn` critic and the random draw of `min_random_pair`. `_td_errors` stays on the GPU.
- **Crops and rotation.** The random-crop offsets are drawn from the same CPU generator in the same
  order, scaled and range-checked on the CPU, and copied to the GPU from pinned memory.
  `get6DRotation` reorders the quaternion by slicing.
- **A startup assert:** `log_freq` must be a multiple of `algo.update_every_n_steps`, so every
  logging step has updates. It is 100 and 1 in every config.

**The evidence that it trains identically.** Every comparison is exact, with no tolerance:
- **Fingerprint** (`tools/agent_fingerprint.py --compare`): all 22 values identical to `ce96bb5`,
  where `target-fix` runs, and to `1e33b8f`.
- **Deep check, CPU** (`--deep`, new): 8 updates on a batch carrying the replay buffer's importance
  weights, every 4th with the actor, then one `act()` with exploration noise. It compares every
  tensor in the state dict, the three optimizers' state, the action and the last update's metrics,
  602 to 734 values per config. It is identical to `1e33b8f` for all 14 configs in
  `tools/agent_fingerprint.py`'s `CANDIDATES`: `target-fix` as it ran; and `critic-fn`, `clipq`,
  both, and neither, each alone, with `squash` and with `depth`, plus all four, at the new default.
  It is identical to `ce96bb5` for `target-fix`, the only one of them that tree can build.
- **Deep check, GPU,** in PyTorch's deterministic mode with TF32 on: `1e33b8f` was first exactly
  identical to itself. The branch is then identical to it for `target-fix`, `default`, `critic-fn`
  and `all-four`. Training runs don't use deterministic mode, so this shows the same computation,
  not the same run-to-run noise.
- **`compute_metrics=False`** on 7 of 8 updates leaves every tensor identical to computing metrics
  on all 8, for `target-fix` and `all-four` (`tests/test_regression_agent.py`).
- **No host syncs:** `tests/test_sync_free.py` runs an update without the actor, one with it, and
  an `act()` under `torch.cuda.set_sync_debug_mode("error")`. It passes for the 13 new-default
  `CANDIDATES` and for the two-arm layout. The same check, run by hand, raises on `target-fix` and
  on an update with metrics on, so it can fail.
- **The trainer, end to end:** a 600-step run with `debug=true`, 5 offline episodes, 200 warm-up
  updates and 2-episode evals. It logged every 100 steps with every metric present, its evals at 0,
  300 and 600 ran and saved a best model, and `Time%` read grad 72–81%, env 5–7%.
- **Tests:** 430 passed, 2 skipped, the whole suite with the GPU.

**Speed** (`tools/bench_update.py`, an env step's learning at batch 256, the two trees alternated
twice on GPU 0 while `target-fix` trained there):

| Config | `1e33b8f` | This commit | Ratio |
|---|---|---|---|
| `default` | 769, 754 ms | 396, 397 ms | 1.92× |
| `all-four` | 821, 794 ms | 405, 403 ms | 2.00× |

- The GPU's busy time is 350–400 ms per env step on both trees, and now equals the wall time: the
  loop is GPU-bound. On an unshared GPU both numbers would be lower.
- The env steps and evals don't speed up. The first run from this commit measures the end-to-end
  s/step, against 0.56 at `1e33b8f`.

**What the logs show differently:**
- `timing/*`, `training/SPS` and `timing/aggregate_steps_per_second` change value. The trainer now
  calls `torch.cuda.synchronize()` once per env step, at the end of the last update's
  `gradient_update` timer, so the GPU's time stays under `gradient_update` rather than moving to
  `env_step`.
- Every other metric keeps its value and meaning, and is logged at the same steps.

**Still syncing, and not used by any planned run:** `actor_norm=iid_bn` (30 syncs per env step,
escnn's own asserts) and `encoder_norm=inner_bn`. They train correctly, only slower (TODO H8).

---

## 2026-09-30 — STOPPED — `target-fix` (`0rehtuoq`) and `critic-ln` (`qlaonpv4`), without verdicts

Stopped at 14:07 EDT with SIGINT, by Aden's decision, before their 150k verdicts. They did not
look like they would succeed, and the GPUs were wanted for the runs pre-registered below. The
sweep's pre-registration allows no early stop but a collapse, so **neither run has a verdict**,
and neither counts as not advancing.

| Arm | wandb | Commit | Stopped at | Evals from step 0, every 10k |
|---|---|---|---|---|
| `target-fix` | `0rehtuoq` | `ce96bb5` | 124,400 | 0.60 0.70 0.48 0.48 0.68 0.52 0.48 0.62 0.62 0.46 0.48 0.46 0.62 (to 120k) |
| `critic-ln` | `qlaonpv4` | `1e33b8f` | 115,500 | 0.44 0.58 0.50 0.48 0.56 0.54 0.58 0.54 0.54 0.62 0.42 0.34 (to 110k) |

- In the verdict window, `target-fix` had two of five evals, 0.46 and 0.62 (mean 0.54, 100
  episodes); `critic-ln` had one, 0.34. Neither collapsed.
- wandb shows both as killed. Their packages were finalized by hand
  (`finalize_run_package`), so each holds wandb's record in
  `outputs/runs/robomimic-square-final/<id>/wandb_logs/`.
- **The sweep is closed.** Its other 16 arms never launched. The pre-registration below replaces
  them.

---

## 2026-09-30 — CODE — one wandb group per task and method, and run names that differ

Aden's decision: wandb groups no longer separate experiments.
- **Square equivariant RL.** The config's defaults are now project `robomimic-square-final` and
  group `equi-residual-rl`, the group name the Can reproduction established. `submit.sh` no longer
  sets either. It names runs `equi-residual-rl-<arm>-<seed>`, and the trainer still appends the
  start time, the hyperparameter string and the seed. The experiment goes in `wandb.notes`. This is
  a default change (STANDARDS 1.6), but only to logging: nothing trained changes.
- **BC.** `train_bc_dexmg.py` puts each run in a group named after its project, so
  `robomimic-square-bc` holds every Square BC run. It names runs
  `<policy>_<dataset>__<start time>__seed<seed>`. Before, every run of one policy on one dataset
  had the same name, and no group.
- **Past runs, regrouped in wandb** through the API. Only the group changed; ids, names and data
  are untouched.

| Project | Runs | Old group | New group |
|---|---|---|---|
| `robomimic-square-final` | `0prso90n`, `j04yoeui`, `88i0lpc8` | `equi-square-v1` | `equi-residual-rl` |
| `robomimic-square-final` | `0rehtuoq` | `equi-square-sweep-target-fix` | `equi-residual-rl` |
| `robomimic-square-final` | `qlaonpv4` | `equi-square-sweep-critic-ln` | `equi-residual-rl` |
| `robomimic-square-bc` | `3hzs5bz1` | none | `robomimic-square-bc` |

- The ResFiT baseline runs in `robomimic-square-final` (`870ws2c2` and four failed ones) were
  already together in group `residual-rl`, and keep it. Other tasks are unchanged.

---

## 2026-09-30 — PRE-REGISTERED — Square: the actor-none gate, then four ablations

**Question.** At the new defaults does the equivariant residual improve on Square's base policy?
The new defaults are the target-critic fix and no normalization in the actor, at the sync-free
commit. If it does, which of four changes help: FieldNorm in the critic, a clipped TD target, a
bounded residual, and a deeper critic head?

**Seed-group id: `square-ablations-0930`** (in each run's `wandb.notes`). Project
`robomimic-square-final`, group `equi-residual-rl`, names `equi-residual-rl-<arm>-<seed>`.

**Each run's verdict (stage 1)** is the 2026-09-29 sweep's rule, unchanged:
- One seed (seed 1), stopped at 150k steps.
- A run **advances** iff the mean of its five evals at 110k–150k (250 episodes) is at least
  **B_sq = 0.508**. That is vanilla BC performance: the Square base policy's success rate over 500
  episodes (2026-09-29 MEASUREMENT).
- A collapse (three consecutive evals ≤ 0.05) stops a run, and it does not advance. There is no
  other early stop.
- Each run is judged against B_sq, not against another run. So, as for the sweep, "does not
  advance" rules out a large gain at n=1, and nothing more. **No stage-1 number is a comparison**
  (STANDARDS 5.3).

**The gate: `actor-none` runs first.** It is the reference for every other arm, and Aden's rule
is that if it cannot at least match vanilla BC, the implementation is reworked before any further
ablation runs.
- `actor-none` and `critic-fn` launch together, one per GPU.
- **If `actor-none` does not advance:** nothing further is queued. `critic-fn` finishes and is
  recorded, but its verdict decides nothing. The next step is a new plan for the implementation.
- If `actor-none` collapses before 150k, `critic-fn` is stopped as well.
- **If `actor-none` advances:** `clipq` is queued. Once `critic-fn`'s and `clipq`'s verdicts are
  both in, `squash` and `depth` are queued on the base made of those that advanced:

| `critic-fn` | `clipq` | Then run |
|---|---|---|
| no | no | `squash`, `depth` |
| yes | no | `critic-fn-squash`, `critic-fn-depth` |
| no | yes | `clipq-squash`, `clipq-depth` |
| yes | yes | `critic-fn-clipq-squash`, `critic-fn-clipq-depth` |

- **Known gap:** if both advance, `critic-fn` and `clipq` together never run on their own, so the
  last pair has no one-change reference.

**Arms.** Every arm is `residual_equi_td3_square_config` plus `algo.prefetch_batches=4`,
`algo.total_timesteps=150000`, `agent.actor.actor_last_layer_init_scale=1e-4`,
`equivariance.actor_norm=none`, and these overrides. Each is one case in `submit.sh`.

| Arm | Overrides | Question |
|---|---|---|
| `actor-none` | none | Does the fixed agent, without the actor's batch norm, at least match vanilla BC? |
| `critic-fn` | `equivariance.critic_norm=field_norm` | FieldNorm at the critic's three hidden sites: visual projection, trunk, head |
| `clipq` | `agent.clip_q_target_to_reward_range=true` | TD targets clamped to [0, 1] |
| `squash` (and on each base) | `equivariance.residual_squash=true` | The residual bounded by `action_scale`, as ResFiT's Tanh bounds it |
| `depth` (and on each base) | `equivariance.num_critic_layers=2` | Two hidden layers per critic head, as ResFiT has |

**Stage 2, pre-registered separately when an arm advances.** Seeds 1 and 2 at 300k, judged against
`actor-none` at the same seeds, not against `target-fix`, which had the actor's batch norm. The
promotion criteria are the sweep's three.

**Held fixed:**
- Everything not overridden: n_step 3, gamma 0.99, buffer 200k, learning_starts 10k, critic warm-up
  10k; `action_scale` 0.1, stddev 0.05, actor LR 1e-6, critic LR 1e-4, 300 demo episodes.
- 50-episode evals every 10k.
- BC `run_3hzs5bz1_best:v9`, and the Square caches `1d31ed9b` and `66c46c51`.
- `boce-WS-01`, torch 2.6.0, torchrl 0.7.0, escnn 1.0.11. Env initial states unseeded.
- **The commit:** every arm launches from the commit that adds this entry. Apart from the wandb
  names, its code is `8407c27`'s (2026-09-30 CODE). Every arm here is one of
  `tools/agent_fingerprint.py`'s `CANDIDATES`, each shown to train bit for bit as at `1e33b8f`.

**Against earlier runs.** Against `target-fix` (`0rehtuoq`), `actor-none` differs in the actor's
batch norm only. The sweep's `clipq`, `squash` and `critic-deep` never ran; these arms are them
without the actor's batch norm.

**Resources.**
- About 14 h per run, estimated: `target-fix` spent 87.6% of its time in the updates, which are
  now 1.9× faster, so about 0.33 s/step. The first LAUNCHED entry measures it.
- Two runs at a time, 40.3 GiB each (measured for the sweep).

**Records.** A LAUNCHED entry per launch. A RESULTS entry per finished run: its eval sequence, its
110k–150k mean, and its verdict. The gate's outcome goes in `actor-none`'s RESULTS entry.

---

## 2026-09-30 — LAUNCHED — `actor-none` (`ekb6181b`) and `critic-fn` (`24y42etk`), seed 1

| Arm | Seed | wandb | GPU | Commit | Launched |
|---|---|---|---|---|---|
| `actor-none`, the gate | 1 | **`ekb6181b`** | 0 | `a95b3a3` | 2026-09-30 14:16 EDT (training 14:29) |
| `critic-fn` | 1 | **`24y42etk`** | 1 | `a95b3a3` | 2026-09-30 14:16 EDT (training 14:29) |

- **How:** queue lines `a95b3a3 <arm> 1` for `tools/gpu_queue.sh` in tmux `gpu0` and `gpu1`, run
  from the worktree `~/projects/equi-resrl-a95b3a3` with `CACHE_DIR=~/projects/equi-resrl`.
  Project `robomimic-square-final`, group `equi-residual-rl`, names
  `equi-residual-rl-<arm>-1`, notes `square-ablations-0930/<arm>/seed1`.
- **Gate:** GO for both. Tier 1 and Tier 2 passed, and both caches were found under `CACHE_DIR`.
- **Startup verified:**
  - `Set random seed to 1`;
  - offline `1d31ed9b` (80,429) and online `66c46c51` found on disk, with no rebuild;
  - the probe centre is (−0.56, 0.0);
  - the resolved configs have `actor_norm: none`, `num_q: 2`, `frozen_target_critic: False`, init
    scale 1e-4 and 150k steps. `critic-fn` has `critic_norm: field_norm`, and `actor-none` has
    `none`.
- **Step-0 eval: 0.52 for both** (26/50 each). The trainer-path step-0 pool is now 224/400 = 0.56.
- **Speed, as the CODE entry predicted:** 0.329 s/step (`actor-none`) and 0.337 (`critic-fn`) over
  their first 3k steps, against about 0.56 at `ce96bb5` and `1e33b8f`, so 1.7× end to end. The 10k
  critic warm-up updates took 12 min. Updates are 86% of the time. Each run should finish around
  2026-10-01 05:00.
- **Memory:** 41–42 GiB PSS each, with the online buffer near its 10k start.
- **Config caveats:** `num_actor_layers` stays dead. `action_scale` does not bound the residual
  without `residual_squash`.

---

## 2026-10-01 — RESULTS — the gate failed: `actor-none` and `critic-fn` both collapsed by 10k

| Arm | wandb | Evals from step 0, every 10k | 110k–150k mean | Verdict |
|---|---|---|---|---|
| `actor-none` | `ekb6181b` | 0.52 0.02 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 | 0.000 | collapse at 30k; does not advance |
| `critic-fn` | `24y42etk` | 0.52 0.00 0.00 0.00 0.02 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 | 0.000 | collapse at 30k; does not advance |

- **The gate failed.** `actor-none` cannot match vanilla BC (0.508); it scores 0.00 from 20k on.
- **A deviation:** a collapse should have stopped both runs at 30k. Nothing enforces that rule and no
  one was watching, so both ran to 150k (exit 0 at 04:58 and 05:21). No run was waiting, so only
  GPU time was lost.
- **Against `target-fix`,** n=1 each, so an observation, not a stage-1 claim. The same config with
  the actor's batch norm did not collapse: 0.70 at 10k, then 0.46–0.70 to 120k (STOPPED entry). Both
  runs without batch norm collapsed, each with a different critic.
- **What the logs rule out:**
  - **A larger residual.** The residual that training logs is smaller without batch norm at every
    step: `train/residual_l1_magnitude` is 0.00011 against 0.00098 at step 100, and 0.10 against
    0.24 at 10k. At 10k, when the no-BN runs already scored 0.02 and 0.00, the BN run scored 0.70.
  - **Batch norm muting the residual at eval.** On 512 demo observations, `target-fix`'s 10k
    checkpoint executes in eval mode (running statistics) what training sees (batch statistics):
    mean |residual| 0.213 against 0.216, with per-sample cosine similarity 0.99998.
- **What batch norm was doing**, from that checkpoint's running statistics:
  - at the visual projection, the input variance averages 0.021, so batch norm scaled the visual
    features up by about 7×;
  - at the trunk and the policy's hidden layer, it subtracted means (0.76 and 1.22 on average)
    about as large as their standard deviations (0.75 and 1.07).
- **The cause of the collapse is not established.** The analysis and the candidate fixes are in
  PROGRESS 2026-10-01 and TODO.

---

## 2026-10-01 — AMENDMENT — the four ablations run with the actor's batch norm

- **The gate failed** (RESULTS above). The pre-registration said nothing further would be queued
  until the implementation was reworked.
- **Aden's decision (2026-10-01):** run the four ablations now with `actor_norm=iid_bn`, the
  actor normalization every run before 2026-09-30 had, while the collapse without it is analysed.
  The decision was made after seeing both collapsed runs and `target-fix`'s partial sequence.
- **Arms:**
  - the same 11 cases in `submit.sh`, with `equivariance.actor_norm=iid_bn` in place of `none` in
    every arm;
  - the reference is `target-fix` again, overriding nothing else;
  - names `equi-residual-rl-actor-bn-<arm>-<seed>`, notes `square-ablations-1001/<arm>/seed<seed>`;
  - **seed-group id `square-ablations-1001`**.
- **Order:** `critic-fn` and `clipq` now, one per GPU. Then `squash` and `depth` on the base the
  2026-09-30 table gives. There is no gate.
- **Verdict:** unchanged. A run advances iff its 110k–150k mean is at least B_sq = 0.508. A
  collapse stops it.
- **These are mostly the 2026-09-29 sweep's configs:**
  - `clipq` and `squash` exactly, and `depth` is the sweep's `critic-deep`;
  - `critic-fn` is the sweep's `critic-ln` as first registered, before its amendment made it
    `layer_norm`;
  - the combinations are new.
- **The reference has no verdict.** `target-fix` (`0rehtuoq`) is these arms' base config, but it was
  stopped at 124k. Stage-1 verdicts are against B_sq, so they don't need it. A stage-2 comparison
  needs `target-fix` re-run at the same seeds.
- **The commit:** the commit that adds this entry. Its training code is `8407c27`'s. On CPU, the
  deep check of all 11 arms at `iid_bn` is identical to `1e33b8f`, tensor for tensor.
- **Speed:** `iid_bn` keeps escnn's 30 host syncs per env step (TODO H8), so these runs are slower
  than the 0.33 s/step of 2026-09-30. The LAUNCHED entry measures it.

---

## 2026-10-01 — LAUNCHED — `critic-fn` (`konwqk6l`) and `clipq` (`63k595oa`), actor BN, seed 1

| Arm | Seed | wandb | GPU | Commit | Launched |
|---|---|---|---|---|---|
| `critic-fn` | 1 | **`konwqk6l`** | 0 (`GPU-ca9f06b8`) | `b39a376` | 2026-10-01 07:50 EDT (wandb 08:03) |
| `clipq` | 1 | **`63k595oa`** | 1 (`GPU-8d23cc42`) | `b39a376` | 2026-10-01 07:51 EDT (wandb 08:03) |

- **How:** queue lines `b39a376 <arm> 1` for `tools/gpu_queue.sh` in tmux `gpu0` and `gpu1`, run
  from the worktree `~/projects/equi-resrl-b39a376` with `CACHE_DIR=~/projects/equi-resrl`. Names
  `equi-residual-rl-actor-bn-<arm>-1`, notes `square-ablations-1001/<arm>/seed1`, project
  `robomimic-square-final`, group `equi-residual-rl`. Consoles in
  `outputs/queue/20261001-075052_critic-fn-1_gpu0.log` and `20261001-075117_clipq-1_gpu1.log`.
- **Gate:** GO for both. Tier 1 and Tier 2 passed, and both caches were found under `CACHE_DIR`.
- **Startup verified:**
  - `Set random seed to 1`;
  - offline `1d31ed9b` (80,429) and online `66c46c51` loaded from disk, with no rebuild;
  - the probe centre is (−0.56, 0.0);
  - the resolved configs in wandb have `actor_norm: iid_bn`, init scale 1e-4 and 150k steps;
    `critic-fn` has `critic_norm: field_norm`, and `clipq` has `clip_q_target_to_reward_range: True`.
- **Step-0 eval:** `critic-fn` 0.64 (32/50), `clipq` 0.50 (25/50). The trainer-path step-0 pool is
  now 281/500 = 0.562.
- **Speed:** 0.422 s/step (`critic-fn`) and 0.418 (`clipq`) over their first 3k steps, against
  0.33 without the actor's batch norm (2026-09-30), so `iid_bn`'s host syncs cost about 27%.
  Updates are 88% of the time. The 10k critic warm-up took 15 min. At this rate 150k ends around
  2026-10-02 02:30, and 100k would end around 2026-10-01 20:30.
- **Memory:** 41–42 GiB PSS each, with the online buffer near its 10k start.
- **Config caveats:** `num_actor_layers` stays dead. `action_scale` does not bound the residual
  without `residual_squash`.

---

## 2026-10-01 — AMENDMENT — stop at 100k, judge by the training trend, and three normalization arms

Aden's decisions (2026-10-01, ~09:00). Written before either running arm's 10k eval: nothing from
`konwqk6l` or `63k595oa` after their step-0 evals has been looked at. Seed-group id unchanged,
`square-ablations-1001`.

**1. Every run in this group stops at 100k.**
- New launches set `algo.total_timesteps=100000`. `critic-fn` and `clipq` launched at 150k, so
  `tools/run_watch.py` stops them (SIGINT) after their 100k eval.
- Nothing in training depends on `total_timesteps` before the stop, so these are the first 100k
  steps of the earlier 150k and 300k runs.

**2. The verdict is the training trend, not the eval level.**
- **Why** (Aden's observation, checked on the eight Square runs with data to 100k). The training
  episodes of runs that learned improve from early on; those of runs that hold BC are flat. The eval
  level at 100k gets it backwards:

| Run | Training success, slope per 100k over 10k–100k (z) | Episode length, slope per 100k | Eval mean, 60k–100k |
|---|---|---|---|
| `870ws2c2`, ResFiT, learned (0.68 at 110k–150k) | +0.49 ± 0.10 (+5.1) | −73 ± 13 | 0.492 |
| `gkxw355f`, ResFiT, learned (0.76 by 90k) | +0.38 ± 0.09 (+4.0) | −73 ± 14 | 0.648 |
| `j04yoeui`, frozen target, seed 1 | +0.01 ± 0.10 | −3 ± 10 | 0.608 |
| `88i0lpc8`, frozen target, seed 2 | +0.04 ± 0.10 | −4 ± 10 | 0.424 |
| `0rehtuoq`, `target-fix` | −0.13 ± 0.10 | +7 ± 10 | 0.532 |
| `qlaonpv4`, `critic-ln` | −0.08 ± 0.10 | −1 ± 10 | 0.540 |
| `ekb6181b`, `actor-none` (collapsed) | −0.02 ± 0.01 | +1 ± 1 | 0.000 |
| `24y42etk`, `critic-fn` without actor BN (collapsed) | −0.01 ± 0.03 | +1 ± 2 | 0.000 |

  A "60k–100k eval mean ≥ B_sq" rule would fail `870ws2c2`, whose evals sat at 0.24–0.38 until
  80k, and pass three runs that never learned.
- **The data.** `training/episode_return`, one row per finished training episode (`num_envs` = 1),
  at the step the episode ended. The reward is 1 on success and 0 otherwise, so the row is γ^steps
  on success and 0 on failure; **success = return > 0**. These episodes carry the exploration noise
  (stddev 0.05).
- **A run advances iff all three hold:**
  - (a) the least-squares slope of success on step, over episodes ending in (10k, 100k], is at
    least 2 standard errors above zero (the usual OLS standard error, from the residual variance);
  - (b) the success fraction of episodes ending in (70k, 100k] is greater than that of episodes
    ending in [0, 10k]: the run's own start, under the same noise. So climbing out of a dip back to
    where it started does not count;
  - (c) it did not collapse (three consecutive evals ≤ 0.05; unchanged).
- **Reported for every run, deciding nothing:** the episode-length slope; the eval mean at
  60k–100k against B_sq = 0.508 and 0.60; the eval sequence.
- **Caveats.** The rule was chosen after seeing the eight runs above, so they calibrate it and do not
  test it. Only two runs learned, both the non-equivariant baseline. A learner half as fast
  (+0.2 per 100k) would sit near z = 2, so a slow learner can be missed.
- `tools/square_verdict.py` computes (a)–(c) and the reported numbers; every RESULTS entry from here
  on uses it.

**3. The reference.** `target-fix` (`0rehtuoq`) has data to 120k, so the rule applies to it, with
its data seen before the rule was chosen: slope −0.13 ± 0.10, **does not advance**. The reference
holds BC and does not learn. `critic-ln` (`qlaonpv4`, the sweep's critic LayerNorm): slope
−0.08 ± 0.10, does not advance, flagged the same way.

**4. Three normalization arms, each one change off `target-fix`, plus its second seed.**

| Arm | Overrides | Question |
|---|---|---|
| `actor-fn` | `equivariance.actor_norm=field_norm` | FieldNorm instead of IID batch norm at the actor's three sites: each sample normalized on its own, no batch statistics, the same in train and eval |
| `actor-inner` | `equivariance.actor_norm=inner_bn` | InnerBatchNorm at the actor's three sites: standard batch norm, statistics pooled over the batch and each field's 8 channels, with gradients through them (escnn's `IIDBatchNorm1d` detaches both) |
| `enc-inner` | `equivariance.encoder_norm=inner_bn` | InnerBatchNorm after each agentview conv (the sweep's `enc-inner`); the actor keeps IID batch norm |
| `target-fix`, seed 2 | none | The reference's second seed, for any stage-2 comparison |

- **Not tested, and why:**
  - batch norm in the critic: the critic normalizes three differently distributed batches, each with
    its own statistics (stored (s, a); the target's (s′, π(s′)), in train mode; the actor loss's
    (s, π(s))), so one input gets a different Q depending on its batch;
  - FieldNorm in the encoder: it would normalize every pixel's 8-channel field to unit size,
    background included.
- **Predictions, measured at init** (same weights, 512 demo observations): vision drives 97.3%
  (FieldNorm) and 98.7% (InnerBatchNorm) of the actor trunk's variance, against 99.3% with IID
  batch norm and 58.7% with none. If the collapse comes from the actor attending to proprioception
  and the base action (PROGRESS 2026-10-01), neither actor arm collapses.
- **`inner_bn` is one new module** for the actor and the encoder: torch batch norm per field, over
  the batch, the field's 8 channels and any spatial positions. That is escnn's InnerBatchNorm's
  computation, which cannot take the actor's flat features and makes host syncs. It replaces escnn's
  module in the encoder option, which no run has used.

**5. Order.** One queue per GPU; each run starts when the one before ends.

| GPU | Runs |
|---|---|
| 0 | `critic-fn` (running) → `actor-fn` → `enc-inner` → the depth arm |
| 1 | `clipq` (running) → `actor-inner` → the squash arm → `target-fix` seed 2 |

- The squash and depth arms are the 2026-09-30 table's, from `critic-fn`'s and `clipq`'s verdicts
  under this rule. They are written into the queues automatically once both runs have stopped.
- **Resources:** about 10–12 h per run, two at a time; everything ends around 2026-10-03 morning.

**6. Automation.** `tools/run_watch.py` runs in tmux beside the queue runners. It stops a run after
its 100k eval, stops one that collapses, and finalizes each stopped run's package. Every action it
takes is logged in `outputs/queue/watch.log`, and the LAUNCHED and RESULTS entries cite it.

**7. Names.** `equi-residual-rl-actor-bn-<arm>-<seed>` where the actor has IID batch norm
(`enc-inner`, `target-fix` and the squash and depth arms); `equi-residual-rl-<arm>-<seed>` for
`actor-fn` and `actor-inner`. Notes `square-ablations-1001/<arm>/seed<seed>`.

**8. Commit.** The new arms launch from the commit that adds their code, named in their LAUNCHED
entries. The arms that already existed must train there bit for bit as at `b39a376` (the deep
check, on CPU).

**Stage 2:** unchanged, pre-registered separately when an arm advances.

---

## 2026-10-01 — CODE — InnerBatchNorm for the actor and the encoder; the verdict script and the watcher

For the AMENDMENT above ("stop at 100k, judge by the training trend, and three normalization arms").

- **`RegularInnerBatchNorm`** (`equi_rl_utils.py`): torch's `BatchNorm1d` with each regular field as
  one channel, over the batch, the field's 8 channels and any spatial positions.
  - It gives escnn's `InnerBatchNorm`'s train outputs, input gradients, running statistics and eval
    outputs, flat and on 5×5 maps, to 1e-5 (`test_equivariance.py`). It is equivariant in train and
    eval mode as an actor, critic and encoder option, and makes no host sync (`test_sync_free.py`).
  - escnn's module cannot take the actor's flat features (its no-base-space path builds a
    `BatchNorm2d` over a 3D view) and slices with GPU index buffers, which syncs.
  - **New option** `actor_norm=inner_bn` (and `critic_norm=inner_bn`, which no arm uses).
  - **`encoder_norm=inner_bn` now builds it** in place of escnn's `InnerBatchNorm`. No run has used
    that option, so no comparison is affected; its state-dict keys differ from escnn's.
- **Unchanged for every existing arm:** the deep check of the 11 arms at `iid_bn` (8 updates and one
  act() on CPU) is identical to `b39a376`'s, tensor for tensor, 668–800 values per arm. The
  regression goldens pass.
- **`submit.sh`:** `algo.total_timesteps=100000`; the cases `actor-fn`, `actor-inner` and
  `enc-inner`; an arm can set the actor's norm. Names as the AMENDMENT says. `CANDIDATES` gains the
  three arms; the sync census covers those without the actor's IID batch norm, and the encoder's
  InnerBatchNorm on its own.
- **`tools/square_verdict.py`** computes the AMENDMENT's rule from a run's local wandb record. On
  the finished runs it gives the AMENDMENT's numbers (`0rehtuoq` −0.13 ± 0.10, `qlaonpv4`
  −0.08 ± 0.10; `ekb6181b` and `24y42etk` collapsed at 30k).
- **`tools/run_watch.py`** stops a run after its 100k eval if it was launched past 100k, stops any
  collapse, finalizes stopped packages, and queues the squash and depth arms once both verdicts are
  in. `tests/test_square_verdict.py` (Tier 1, in preflight) checks both tools' decisions on runs
  built to sit either side of each rule.

---

## 2026-10-02 — CORRECTION — every equivariant run acted on near-blank images

**The bug.**
- The update's images come from the replay buffer, which stores them as uint8 in [0, 255]
  (`utils/dtype.py`).
- act()'s images come from the env as float in [0, 1] (`dexmg.py`). That covers every exploration
  step and every eval.
- `ResObsEnc.forward` divided both by 255 before mapping to [−1, 1]. So act() saw [−1, −0.992].
- Measured on a live Square env:
  - act()'s encoder input lay in [−0.9998, −0.9922], std 0.0019;
  - the same frames in the buffer's format give [−0.95, 0.98], std 0.49.

**What it did, measured** on the two Square checkpoints with trained actors, both with the actor's
IID batch norm. The same frames were fed in each format: 256 demo frames, and 8 live-env frames
under "live".

| | `2cchirbo` (`target-fix` s2, 30k, eval 0.64) | `63k595oa` (`clipq`, 50k, eval 0.60) |
|---|---|---|
| Agentview features, spread across states, act() vs update | 0.0003 vs 0.1247 | 0.0001 vs 0.1026 |
| Mean \|residual\|, executed vs trained (demo frames) | 0.0014 vs 0.3057 | 0.0009 vs 0.2654 |
| Mean \|residual\|, executed vs trained (live frames) | 0.0014 vs 0.3457 | 0.0009 vs 0.3002 |
| Per-state cosine, executed vs trained residual | mean 0.79, min 0.74 | mean 0.38, min 0.19 |
| ‖executed − trained‖ / ‖trained‖ | 0.997 | 0.999 |

**These runs trained a residual of about 0.3 and executed about 0.001: they ran the base policy plus
exploration noise.**

**Which runs.** Every recorded equivariant run, and every run of the scalar ablation, which uses
the same encoder:
- **The forms of the bug:**
  - from `4983fcd` (2026-02-28), images were divided by 255 into [0, 1], so act() saw [0, 0.004];
  - from `aabde0f` (2026-04-23), they were mapped into [−1, 1], as above;
  - before `4983fcd`, the encoder did not scale images at all, a 255× mismatch the other way. No
    run in the summary table used it.
- **So it covers every recorded run:**
  - `x2w6phzf` (`a948786`);
  - the `7dae4925` runs;
  - `z8yoqylh`;
  - the Can replication (`38z4wr9z`, `sw2qwfs9`);
  - the four undocumented Can runs (`lgc70vgd`, `q1ocx89o`, `3vl625tv`, `36pfxsww`);
  - `equi-square-v1`;
  - the sweep;
  - every ablation.
- **Not affected:**
  - The ResFiT agent (`870ws2c2`, `msfkjwab`, `zyed0lew`): its `_encode` converts by dtype.
  - The base policies, which get the env's floats, as in BC training.
  - Step-0 evals: the residual starts near zero for any input. So B_sq and the step-0 pool stand.

**What it changes.**
1. **Every equivariant result measures the base policy plus what the actor computes from a blank
   image,** not the method.
   - For the IID-batch-norm actor that was about zero. The runs that "held BC" held it because they
     executed almost nothing.
   - No Can checkpoint survives, locally or in wandb, so the Can numbers cannot be measured.
     They come from the same code path. Read them as the base policy's draws until TODO B6 re-runs
     Can.
2. **Two earlier measurements fed only the buffer's format:**
   - 2026-10-01 RESULTS, "ruled out: batch norm muting the residual at eval";
   - STATUS, "the BN actor executes at eval what training sees".

   Both hold on uint8 images and fail on the env's.
3. **The actor-norm collapses (TODO B10) are reopened.**
   - The no-norm, FieldNorm and InnerBatchNorm actors executed whatever they compute from a blank
     image.
   - Their best checkpoints are all from step 0, so that cannot be measured now.
   - Both the collapses and the batch-norm runs' stability fit the bug. Neither is established.
4. **Every logged `train/residual_l1_magnitude` and `residual_l2` is the update batch's residual,**
   computed from buffer images, not the executed one (STANDARDS 6.3).
5. **The overestimated Q values fit the bug.** One example is Q ≈ 9 on 0/1 rewards in
   `u4p6uz3d`. The critic was asked to value residuals about 0.3 away from anything executed. That
   is a reading, not a test.

**How it went unseen.**
- Every test, golden and fingerprint fed uint8 images, the buffer's format; none fed the env's.
- The ResFiT agent converts by dtype. The equivariant encoder diverged from it without a comment,
  against STANDARDS 3.1.

**The fix:** CODE entry below. `equivariance.float_images_as_uint8=True` reproduces the old scaling.
Earlier runs' configs lack the field, and every earlier run had it in effect as `True`.

---

## 2026-10-02 — CODE — act() reads the env's float images as they are; the gate's unseeded test

**The fix.**
- `ResObsEnc._image_to_signed_unit` divides uint8 images by 255 and takes float images as already in
  [0, 1], then maps both to [−1, 1]. It reads the dtype, which is metadata, so there is no host sync.
  Both encoder branches use it.
- uint8 images go through the same operations as before, in the same order. **So every buffer
  batch, and so every update, is bit-identical to before.**
- **New field `equivariance.float_images_as_uint8`,** default `False`; `True` reproduces the old
  scaling. The scalar ablation reads it too, so it is not in `EQUIVARIANT_ONLY_FIELDS`.
- This is a behaviour change at an unchanged config (STANDARDS 1.6). From this commit, an
  equivariant run's act() sees what its update sees.

**Verified.**
- **The deep check is identical to `ccff539`, tensor for tensor,** for all 17 `CANDIDATES`
  (8 updates and one act() on uint8, on CUDA in deterministic mode). The regression goldens pass.
- **New `tests/test_obs_formats.py`** (Tier 2) feeds frames made as the env and the buffer make them:
  - The two formats give the same encoder features, in both branches, to 3.6e-7 relative. On CUDA,
    dividing by 255 multiplies by its reciprocal, so they can differ in the last bit; TF32 is off for
    these tests, since it rounds that up to about 1e-3.
  - They give the same act() at the Square reference config, to under 1e-4 relative.
  - With the flag, the image features differ by 1.0 (equivariant) and 0.37 (scalar) relative, and
    the buffer format stays bit-identical.
- **On the two trained checkpoints, with the fix:**
  - the executed residual equals the trained one, on demo and live-env frames: cosine 1.0000,
    relative difference 4e-5 to 1e-4 with TF32 on;
  - with the flag, they reproduce the CORRECTION's 0.0014 and 0.0009.
- `test_config.py`: the flag defaults to `False` in every equivariant config.

**The gate's unseeded test** (the `squash` arm's NO-GO).
- `test_vision_encoder_options_are_equivariant_at_right_angles[orth-init-*]` built its encoder
  without a seed, so each run tested another network.
- Over 20 seeded delta-orthonormal inits, 19 gave relative errors of 0.5–2.2e-6. One gave 0.87, but
  its output is about 1e-8 (85% exact zeros) and its absolute error 1e-8: float noise over a
  near-zero denominator, not broken equivariance. He init gave no such draw in 20.
- The test now seeds torch and numpy before the build. It fails on a degenerate output
  (max |out| ≤ 1e-3) instead of reporting it as non-equivariance.
- That about 1 in 20 delta-orthonormal encoder inits is near-dead is a property of
  `encoder_orth_init`, which no run has used (TODO).
- The `no_tf32` fixture moved to `conftest.py`, since two test files use it now.

**Also:**
- New rule STANDARDS 4.5: test the agent on the observations the env produces.
- `preflight.py`'s Tier 2 runs `test_obs_formats.py`.

---

## 2026-10-02 — LAUNCHED — the queue's launches of 2026-10-01 evening, written up afterwards

The runners and the watcher launched these unattended (2026-10-01 AMENDMENT, "Order" and
"Automation"). Recorded on 2026-10-02 from `outputs/queue/gpu{0,1}.log`, `watch.log`, each launch's
console log and the run packages.

| Arm | Seed | wandb | GPU | Commit | Launched (EDT) |
|---|---|---|---|---|---|
| `actor-inner` | 1 | **`qlxevddz`** | 1 | `ccff539` | 2026-10-01 20:20 (wandb 20:32) |
| `actor-fn` | 1 | **`q12tenbb`** | 0 | `ccff539` | 2026-10-01 20:28 (wandb 20:40) |
| `squash` | 1 | none | 1 | `ccff539` | 2026-10-01 23:42: **gate NO-GO**, exit 1 after 11 min |
| `enc-inner` | 1 | **`u4p6uz3d`** | 0 | `ccff539` | 2026-10-01 23:48 (wandb 00:01) |
| `target-fix` | 2 | **`2cchirbo`** | 1 | `ccff539` | 2026-10-01 23:54 (wandb 00:07) |

- **How:** queue lines `ccff539 <arm> <seed>`, run from `~/projects/equi-resrl-ccff539` with
  `CACHE_DIR=~/projects/equi-resrl`. The watcher wrote the squash and depth lines at 20:29 from
  `critic-fn`'s and `clipq`'s verdicts (both "does not advance", so the table's plain `squash` and
  `depth`). Names and notes as the AMENDMENT says.
- **Gate:** GO for the four runs (Tier 1: 133 passed, 2 skipped; Tier 2: 332 passed).
  **`squash` failed Tier 2:** `test_vision_encoder_options_are_equivariant_at_right_angles
  [orth-init-eval]`, a test of an encoder option no arm uses. The runner moved to the next line, as
  it is built to. The most likely cause, an unseeded test, is in the 2026-10-02 CODE entry above.
  `squash` never ran.
- **Startup verified** from the consoles: `Set random seed` to the arm's seed; offline `1d31ed9b`
  (80,429) and online `66c46c51` (10,000) loaded from disk; probe centre (−0.56, 0.0).
- **Step-0 evals:** `actor-inner` 0.56, `actor-fn` 0.54, `enc-inner` 0.62, `target-fix` seed 2
  0.46. The trainer-path step-0 pool is now 390/700 = 0.557.
- **Speed,** from each run's history after the warm-up: 0.346 s/step (`actor-fn`), 0.347
  (`actor-inner`), 0.426 (`target-fix` seed 2), and 0.895 (`enc-inner`): InnerBatchNorm after every
  agentview conv doubles the step.

---

## 2026-10-02 — RESULTS — the Square ablations: none advances, and both actor-norm arms collapse

From `tools/square_verdict.py` (the 2026-10-01 rule). Seed-group `square-ablations-1001`, seed 1.

| Arm | wandb | Evals from step 0, every 10k | Slope per 100k (z) | Success: start → late | Verdict |
|---|---|---|---|---|---|
| `critic-fn` | `konwqk6l` | 0.64 0.56 0.44 0.60 0.42 0.58 0.48 0.60 0.58 0.46 0.48 | −0.03 ± 0.10 (−0.3) | 0.60 → 0.54 | does not advance |
| `clipq` | `63k595oa` | 0.50 0.30 0.56 0.44 0.56 0.60 0.54 0.56 0.52 0.60 0.54 | +0.03 ± 0.10 (+0.3) | 0.38 → 0.52 | does not advance |
| `actor-fn` | `q12tenbb` | 0.54 0.00 0.00 0.00 | — | — | collapse at 30k; does not advance |
| `actor-inner` | `qlxevddz` | 0.56 0.00 0.00 0.00 | — | — | collapse at 30k; does not advance |

- **Reported, deciding nothing:** eval mean at 60k–100k 0.520 (`critic-fn`) and 0.552 (`clipq`),
  against B_sq = 0.508; episode-length slopes +11 ± 10 and +0 ± 10 per 100k. Neither collapsed.
- **The watcher acted as designed:** it stopped `critic-fn` and `clipq` after their 100k evals
  (20:28 and 20:20; both were launched to 150k), and `actor-fn` and `actor-inner` at their 30k
  collapse (23:48 and 23:42), and finalized all four packages.
- **The vision-share prediction failed.** FieldNorm and InnerBatchNorm in the actor both collapsed,
  though at init they give vision 97–99% of the actor trunk's variance, as IID batch norm does
  (2026-10-01 AMENDMENT, point 4).
- **Every run here acted on near-blank images** (CORRECTION above). `clipq`'s trained actor executed
  a mean residual of 0.0009 against the 0.27 it trained. So these verdicts say what the configs did
  under that bug, not whether the designs work.

---

## 2026-10-02 — STOPPED — `enc-inner` (`u4p6uz3d`) and `target-fix` seed 2 (`2cchirbo`): the machine went down

The machine went down at about 06:50 EDT on 2026-10-02 (the consoles' last writes are 06:43 and
06:50) and came back at 14:22. Both runs died with it, as did the two runners and the watcher.
Neither reached 100k, so **neither has a verdict.**

| Arm | wandb | Last logged step | Evals from step 0, every 10k | Training slope so far |
|---|---|---|---|---|
| `enc-inner` | `u4p6uz3d` | 25,400 | 0.62 0.50 0.52 | +0.82 ± 1.48 per 100k |
| `target-fix`, seed 2 | `2cchirbo` | 55,900 | 0.46 0.52 0.56 0.64 0.54 0.56 | −0.24 ± 0.28 per 100k |

- wandb shows both as crashed. Their packages were finalized by hand on 2026-10-02
  (`finalize_run_package`), so each holds wandb's record in `wandb_logs/`.
- **Never launched:** `squash` (its gate failed: LAUNCHED above) and `depth` (next on GPU 0's
  queue).
- **The group is parked** (Aden's decision, 2026-10-02): the ablations stop here. GPU 0's queue
  line `ccff539 depth 1` is commented out. The arms stay launchable at `ccff539`.

---

## 2026-10-02 — PRE-REGISTERED — Square: the reference with the image fix, seeds 1 and 2

**Question.** With act() fixed, the residual the actor trains is the residual it executes. Does the
reference equivariant residual then learn on Square, hold BC, or collapse? Aden's decision
(2026-10-02): run this pair before the tuned-RLPD experiment.

**Seed-group id `square-image-fix-1002`.**
- Project `robomimic-square-final`, group `equi-residual-rl`.
- Names `equi-residual-rl-actor-bn-image-fix-<seed>`, notes `square-image-fix-1002/image-fix/seed<seed>`.

**The arm, `image-fix`:** `target-fix`'s launch, unchanged, at the commit that adds this entry.
- **Overrides:** `residual_equi_td3_square_config` plus `algo.prefetch_batches=4`,
  `algo.total_timesteps=100000`, `agent.actor.actor_last_layer_init_scale=1e-4` and
  `equivariance.actor_norm=iid_bn`.
- **Seeds 1 and 2,** one per GPU. These are `target-fix`'s `0rehtuoq` and `2cchirbo`.

**Each run's verdict** is the 2026-10-01 rule, computed by `tools/square_verdict.py`. A run
advances iff all three hold:
- (a) the slope of training success over (10k, 100k] is at least 2 standard errors above zero;
- (b) success over (70k, 100k] beats success over [0, 10k];
- (c) it did not collapse (three consecutive evals ≤ 0.05).

Reported, deciding nothing: the eval mean over 60k–100k against B_sq = 0.508, the eval sequence, and
the episode-length slope.

**No early stop.** Each run ends at 100k. A run that collapses may be stopped by hand from then on.
That changes nothing in its verdict, since it fails (c).

**What the outcomes mean,** decided now:
- **Either run advances:** the reference learns once its actions are executed. The ablations'
  verdicts say nothing about their designs. Stage 2 is pre-registered separately.
- **Both hold without advancing:** the bug was not all that stopped learning. The reference's other
  choices remain suspects, two critic heads among them.
- **Either run collapses:** the IID-batch-norm actor's earlier stability came from executing almost
  no residual. The equivariant tuned-RLPD arm starts from this design (Aden's decision, 2026-10-02).
  If this happens, that arm does not launch on this design until Aden decides.

**Against earlier runs.**
- `target-fix` at the same seeds (`0rehtuoq` to 124k, `2cchirbo` to 56k) had the same config,
  seeds, caches and training code (the deep check). The two differ in act()'s image scaling only.
- Two seeds against two is an observation, not a comparison (STANDARDS 5.3).
- Env initial states are unseeded in both.

**Held fixed:**
- **Everything `target-fix` had:**
  - n_step 3, gamma 0.99, buffer 200k, learning_starts 10k, critic warm-up 10k;
  - `action_scale` 0.1, stddev 0.05, actor LR 1e-6, critic LR 1e-4, two critic heads;
  - 300 demo episodes;
  - 50-episode evals every 10k.
- **BC policy:** `run_3hzs5bz1_best:v9`.
- **The Square caches `1d31ed9b` and `66c46c51`.** The fix changes neither: the warm-up uses no
  actor, and the cache key holds no encoder field.
- **Environment:** `boce-WS-01`, torch 2.6.0, torchrl 0.7.0, escnn 1.0.11.
- **The commit:** the one that adds this entry. Its training code is `ccff539`'s apart from act()'s
  image scaling (the deep check).

**Resources.** About 12 h per run at 0.42 s/step, the rate of the earlier `iid_bn` runs. Two runs
at a time, at 41–42 GiB each.

**Launcher.** `submit.sh` and `tests/test_sweep_arms.py` are rewritten for this pair; the test checks
that `image-fix` launches `target-fix`'s overrides exactly. `tests/test_square_verdict.py` now checks the watcher's squash and
depth arms against `ccff539`'s `submit.sh`, where the parked group launches.

**Records.** A LAUNCHED entry, with the step-0 evals and the measured speed. A RESULTS entry per
run, from `tools/square_verdict.py`.

---

## 2026-10-02 — LAUNCHED — the image-fix pair, `fal5sfot` (seed 1) and `fqo83rvd` (seed 2)

| Arm | Seed | wandb | GPU | Commit | Launched (EDT) |
|---|---|---|---|---|---|
| `image-fix` | 1 | **`fal5sfot`** | 0 (`GPU-ca9f06b8`) | `b667647` | 2026-10-02 17:04 (wandb 17:19) |
| `image-fix` | 2 | **`fqo83rvd`** | 1 (`GPU-8d23cc42`) | `b667647` | 2026-10-02 17:04 (wandb 17:19) |

- **How:**
  - queue lines `b667647 image-fix <seed>`;
  - `tools/gpu_queue.sh`, in new tmux sessions `gpu0` and `gpu1`, started from
    `~/projects/equi-resrl-b667647` with `CACHE_DIR=~/projects/equi-resrl`;
  - consoles in `outputs/queue/20261002-170447_image-fix-{1_gpu0,2_gpu1}.log`;
  - no watcher.
- **Gate:** GO for both. Tier 1: 136 passed, 2 skipped. Tier 2: 298 passed, now including
  `test_obs_formats.py`.
- **Startup verified:**
  - `Set random seed` to 1 and 2;
  - offline `1d31ed9b` (80,429) and online `66c46c51` (10,000) loaded from disk;
  - probe centre (−0.56, 0.0);
  - the resolved configs in wandb have `actor_norm: iid_bn`, `float_images_as_uint8: False`,
    `frozen_target_critic: False`, `num_q: 2`, init scale 1e-4 and 100k steps.
- **Step-0 eval:** `fal5sfot` 0.46 (23/50), `fqo83rvd` 0.52 (26/50). The trainer-path step-0 pool is
  now 439/800 = 0.549.
- **Speed:** 0.417 s/step for both over steps 1k–3.7k, as for the earlier `iid_bn` runs. 100k ends
  around 2026-10-03 06:00.
- **Memory:** 27.6–27.7 GiB PSS per trainer process, not counting the eval workers.
- **The first 2.9k steps, as observations only; the verdict is at 100k.**
  - **Training success:** 3/10 and 1/9 episodes.
    - Before the fix, the same seeds had 4/10 and 2/10. `870ws2c2`, ResFiT's run that went on to
      learn, had 3/11 by 3k.
    - Training episodes carry the exploration noise (σ 0.05), so they sit below the evals.
    - At about 10 episodes per run, ±0.15, none of these differ.
  - **The residual,** on the update batch: mean |r| 0.049 and 0.056, max 0.18. That is the same as
    before the fix (0.051 for both) and as `870ws2c2` at 3k (0.048, in its normalized units). No metric
    logs the executed residual; after the fix, act() computes the same one on the same observation.
  - **What to watch: before the fix, this residual kept growing.** It reached 0.13 by 5k, 0.16–0.24
    by 10k, and about 0.3 by 20–30k. `870ws2c2`'s reached 0.055 by 10k and 0.072 by 30k, and its
    Tanh times `action_scale` caps it at 0.1. The reference has no cap (`residual_squash=false`), and
    that growth is now executed. This is the first test of the unbounded residual, the question the
    never-run `squash` arm was to ask.
- **Config caveats:** `num_actor_layers` stays dead. `action_scale` does not bound the residual
  without `residual_squash`. The equivariant actor ignores `stddev_clip`, so its target noise is not
  clipped. The same caveats hold for the four runs of the LAUNCHED entry for 2026-10-01's launches
  above, which omitted them.

---

## 2026-10-04 — RESULTS — the image-fix pair: both collapsed by the first eval

Seed-group `square-image-fix-1002` (2026-10-02 PRE-REGISTERED and LAUNCHED). Both runs finished
100k on 2026-10-03, at 05:25 and 05:35, after 12.1 and 12.3 h by the trainer's clock (0.44 s/step
with evals). Verdicts from `tools/square_verdict.py fal5sfot fqo83rvd`.

| Run | Seed | Step 0 | Evals 10k–100k | Training success, [0, 10k] → (70k, 100k] | Verdict |
|---|---|---|---|---|---|
| `fal5sfot` | 1 | 0.46 | 0.00 at all ten | 0.09 (34 episodes) → 0.00 (100) | does not advance: collapsed at 30k |
| `fqo83rvd` | 2 | 0.52 | 0.00 at all ten | 0.03 (33 episodes) → 0.00 (100) | does not advance: collapsed at 30k |

- **Read against the pre-registration: "either run collapses."** So the IID-batch-norm actor's
  earlier stability came from executing almost no residual (2026-10-02 CORRECTION), and the
  ablations' verdicts say nothing about their designs.
- **The residual,** mean |r| over the update batch, in env action units, against the clip at 1.0:

  | Step | 2k | 5k | 10k | 20k | 30k | 50k | 100k |
  |---|---|---|---|---|---|---|---|
  | `fal5sfot` | 0.03 | 0.13 | 0.18 | 0.28 | 0.43 | 0.44 | 0.47 |
  | `fqo83rvd` | 0.03 | 0.12 | 0.07 | 0.20 | 0.28 | 0.40 | 0.44 |

  - Its largest component reached the clip by 15–20k (`debug/residual_max_abs` 1.00).
  - ResFiT's `870ws2c2`, whose Tanh caps it at 0.1, had 0.055 at 10k.
  - Every training episode after 10k timed out at 300 steps.
- **The critic kept scoring the residual above the base action** while the policy failed:
  - `debug/q_advantage` was −0.0004 to +0.007, and positive at 19 of the 20 logged steps read
    (10 per run).
  - `train/critic_qt` stayed at 0.06–0.23 and the critic loss fell from about 7e-4 to 1e-4. Half
    of each batch is demos, which succeed, so those batch means do not show the online failure.
  - `dQ_da` stayed at 6–10e-5.
- **Aden's decision (2026-10-04),** as the pre-registration requires: the equivariant tuned-RLPD
  arms do not launch on this design. Build all four tuned-RLPD arms, launch the two ResFiT arms,
  and decide the equivariant arms' design when those report (TODO B12).
- Two seeds: an observation, not a comparison (STANDARDS 5.3). Packages in
  `outputs/runs/robomimic-square-final/{fal5sfot,fqo83rvd}/`.

---

## 2026-10-04 — MEASUREMENT — Can base rate B_can = 0.894 ± 0.014

- `eval_base_policy --config-name residual_equi_td3_can_config --num-evals 10 --seed 1000`, as for
  B_sq. Run at `b667647` from its pinned worktree, whose code is `b05736d`'s, on `boce-WS-01`, GPU 1,
  11:57–12:06. Record: `outputs/base_rates/Can__xhjdl8a7_best_latest__seed1000.json`.
- BC `run_xhjdl8a7_best:latest`, which resolves to `v7`, the only version on disk. 8 eval envs, as
  in the trainer. Env initial states are unseeded.
- **Per-eval rates:** 0.88, 0.94, 0.90, 0.90, 0.86, 0.84, 0.92, 0.86, 0.94, 0.90.
  **Mean 0.894 ± 0.014** (binomial SE, 500 episodes). The spread between evals matches 50-episode
  binomial noise.
- For comparison, `msfkjwab`'s step-0 eval (`residual_td3_can_config` at `807efd65`) was 0.92.
- It is the bar for the Can arms of tuned RLPD (PRE-REGISTERED below).

---

## 2026-10-04 — CODE — a no-base-policy mode for tuned RLPD; the residual path unchanged

**What it adds.** `base_policy=None` runs vanilla RL through `train_residual_td3.py`: there is no
base policy, and the actor outputs the whole action. It is the plan's Phase 2
(`docs/plans/2026-10-02-tuned-rlpd.md`, TODO B12). It is a mode of the existing trainer, not a
copied one, so there is one code path per behavior and recorded launch commands stay valid.
ARCHITECTURE.md, "Residual or vanilla", lists the nine branches on `residual_run`.
- **Config** (`residual_td3.py`):
  - `base_policy: BasePolicyConfig | None`.
  - `TunedRLPDConfig` holds the vanilla values: every field in which upstream's residual and RLPD
    configs differ, at its RLPD value.
  - Four registered configs: `tuned_rlpd_{square,can}_config` (ResFiT, 10 critic heads) and
    `equi_tuned_rlpd_{square,can}_config` (2 heads). A task's two configs differ only in heads,
    `equivariance` and wandb group.
  - `residual_only_fields_set` names any residual-only field a vanilla config turns on, and the
    trainer refuses to start if it finds one.
- **`NormalizedVecEnvWrapper`** (`wrappers/normalized_env_wrapper.py`): the residual wrapper
  less the base policy.
- **The equivariant agent:**
  - `ResObsEnc(base_action_input=False)`: no base action, and the actor's features are the
    critic's.
  - `Actor(residual_actor=False)`: no base action at the trunk or the policy MLP; it refuses
    features that carry one. Its bound is `residual_squash`, which then bounds the whole action.
  - The scalar ablation refuses non-residual mode at construction.
- **Cache keys:** a vanilla key adds `base_policy: None` and `normalization` (`identity` on the
  equivariant path, `dataset` on ResFiT's), so it can never name a residual buffer. Residual keys
  are untouched.
- **Logging:** a vanilla run logs `train/action_l{1,2}_magnitude` and `histograms/actions`, as
  upstream's RLPD does.
- **Gate:** `preflight.py --method tuned-rlpd --stack {equi,resfit}` skips the BC checks, mirrors
  the vanilla keys, reports that buffers are built in memory, and skips the probe for ResFiT.
- **Verdict:** `tools/tuned_rlpd_verdict.py --project <project>`. `square_verdict.load_history`
  now takes a project.
- **Launcher:** `submit.sh` holds the four arms. The two equivariant arms refuse to launch while
  their design is pending (TODO B12).

**The residual path is unchanged. Verified:**
- **The deep check is identical to `b05736d`, tensor for tensor,** for all 17 `CANDIDATES`: 8
  updates and one act(), on CUDA in deterministic mode, against the pinned worktree at `b667647`,
  whose code is `b05736d`'s. The regression goldens pass.
- **Every residual config resolves byte for byte as before** (`--cfg job --resolve`, all nine), and
  so does the image-fix pair's launch.
- **Its cache keys did not move:** the residual reference's debug run loaded `1d31ed9b` and
  `66c46c51` from disk, and `test_config.py` still pins all three tasks' residual keys.
- No default changed (STANDARDS 1.6).

**The vanilla path. Verified:**
- `test_env_wrapper.py`: the new wrapper steps exactly as the residual wrapper does with a base
  action that normalizes to zero: the same env action, buffer action, observations and rewards.
- The equivariant vanilla actor:
  - is equivariant for every actor norm, the squash and no visual projection, in train and eval
    mode, for one and two arms (`test_equivariance.py`);
  - with `equi_tanh`, keeps its output, the whole action, within the bound per block; without it,
    only the explored action stays within the bound;
  - has the right widths (`test_layouts.py`), makes no host sync at `actor_norm=none`
    (`test_sync_free.py`), and acts alike on the env's and the buffer's observations
    (`test_obs_formats.py`).
- The four arms compose. A task's two arms differ in their stack only, and each builds and
  updates on the GPU (`test_sweep_arms.py`).
- `test_config.py` pins the vanilla configs and the four vanilla cache keys.
- **The full suite: 514 passed, 2 skipped** (the two skips predate this change).

**Four debug runs, end to end** (`debug=true`, 600 steps, 8-episode evals at 0, 300 and 600):

| Run | What it showed |
|---|---|
| The residual reference: the image-fix pair's launch, critic warm-up cut to 100 | Loaded `66c46c51` (10,000) and `1d31ed9b` (80,429) from disk. Logs `residual_l1` as before. Evals 7, 4 and 3 of 8, its base policy. 37.7 GB peak RSS. |
| `tuned-rlpd-square` | No base policy; `ActionScaler` not widened. 80,429 demo transitions built in memory in 3.8 min, as many as the residual cache holds. 10k warm-up steps of U(±1). Keys `a2d779c4` and `6e94049f`. 0 of 8 at every eval. 14.2 GB. |
| `equi-tuned-rlpd-square`, at the provisional design | Keys `8b214742` and `8e67e7b9`. 0 of 8 at every eval. 14.0 GB. |
| `tuned-rlpd-can` | 62,454 demo transitions. The warm-up's 50 episodes in 10k steps show Can's 200-step horizon. Keys `83506f61` and `3718c816`. 0 of 8 at every eval. 12.2 GB. |

- Each vanilla run's keys are the ones `preflight.cache_hashes(task, "tuned-rlpd", stack)`
  mirrors.
- **Observed, deciding nothing:** `train/action_l1` over the first 600 steps was 0.88–0.95 for
  ResFiT on Square, 0.85–1.00 on Can, and 0.69–0.78 for the equivariant arm. ResFiT's Tanh
  saturates within its first 100 actor updates: 0.9985 on Can, with an actor gradient norm of
  0.0006. Upstream's RLPD config is the same.
- Peak RSS is at the start, with 10k online transitions. The online buffer then grows by about
  85 kB a transition.

---

## 2026-10-04 — PRE-REGISTERED — tuned RLPD on Can and Square

**Question.** Does the RL machinery learn in vanilla RL, with no base policy, on the two tasks the
residual work uses? The ResFiT paper's "Tuned RLPD" is this baseline: it "does not use a base
policy or residuals but does incorporate all of ResFiT's other off-policy RL design decisions".
On Square it is the only baseline over 90% within 150k steps; on Can, all the baselines reach
"high success rates in ~150k steps". Plan: `docs/plans/2026-10-02-tuned-rlpd.md` (untracked);
TODO B12 holds its decisions.

**Seed-group id `tuned-rlpd-1002`,** seed 1 per arm and task.
- Projects `robomimic-square-final` and `robomimic-can-final`.
- Groups `tuned-rlpd` (ResFiT) and `equi-tuned-rlpd`.
- Names `<arm>-<seed>`, notes `tuned-rlpd-1002/<arm>/seed<seed>`.

**The arms.** Each is its stack's residual design made vanilla, the way the paper made ResFiT
vanilla: every field in which upstream's residual and RLPD configs differ takes its RLPD value, on
both stacks (Aden's decisions, 2026-10-02; `TunedRLPDConfig`).

| Made vanilla, on both stacks | Residual | Vanilla |
|---|---|---|
| Base policy | yes | none (`base_policy=None`) |
| Actor LR | 1e-6 | 1e-4 |
| Critic target τ | 0.005 | 0.01 |
| `action_scale` | 0.1 | 1.0 |
| Actor last-layer init scale | 0.0 (the equivariant reference: 1e-4) | None |
| Critic warm-up | 10k updates | 0 |
| Warm-up actions, first 10k steps | base action + U(±0.2) | U(±1) |
| Exploration σ | 0.05 | 0.1 |
| Offline action labels | the base policy's | none |
| ResFiT path's `ActionScaler` range expansion | 0.1 | 0.0, as upstream's RLPD trainer |

| Kept from each stack's residual design | Equivariant | ResFiT |
|---|---|---|
| Critic heads | 2 (min of 2 for targets, mean for the actor) | 10, the same rules, as the paper ran it (2026-09-29 CORRECTION) |
| Critic head | 1 hidden layer, no norm, He init, GroupPooling → invariant Q | 2 hidden layers of 1024, LayerNorm, orthogonal init |
| Actor | 1 hidden layer of 128 regular fields, delta-orthonormal init; **norm pending** | 2 hidden layers of 1024, LayerNorm, Tanh |
| Encoder | C8 CNN on agentview (32 regular fields), CNN on the wrist (32), random crop 84→76 | ViT per camera, random shift (pad 4) |
| Normalization | identity outside; the equivariant normalizer inside the encoder | dataset statistics, outside the network |
| Action bound | **pending**; the clip is the unit disk per rotating pair, [−1, 1] per scalar | per-axis box (Tanh) |
| Target noise | not clipped (the equivariant actor ignores `stddev_clip`) | clipped at 0.3 |

Held fixed, by both stacks' configs: critic LR 1e-4, gradient clip 1.0, n_step 3, γ 0.99, 4 updates
per env step with the actor on the last, batch 256 half demos, buffer 200k, learning_starts 10k,
300 demo episodes, 50-episode evals every 10k, 150k steps.

So **the two stacks are not a controlled comparison of equivariance.** They differ in heads,
norms, depth, encoder and bound. Each arm is a stage-1 check of its own stack against its own
bar (STANDARDS 5.3).

**What launches now** (Aden's decision, 2026-10-04). The image-fix pair collapsed (RESULTS above),
so, as its pre-registration requires, the equivariant arms do not launch on its design. Aden
chose to build all four arms, launch the two ResFiT arms, and decide the equivariant design when
those report.

| Arm | Config | Launch |
|---|---|---|
| `tuned-rlpd-square` | `tuned_rlpd_square_config` | now, GPU 0 |
| `tuned-rlpd-can` | `tuned_rlpd_can_config` | now, GPU 1 |
| `equi-tuned-rlpd-square` | `equi_tuned_rlpd_square_config` + its design | pending Aden |
| `equi-tuned-rlpd-can` | `equi_tuned_rlpd_can_config` + its design | pending Aden |

- Every arm's overrides: `algo.total_timesteps=150000`, and the name, notes and seed.
- `submit.sh` holds the equivariant arms with the plan's provisional design,
  `equivariance.actor_norm=iid_bn` and `equivariance.residual_squash=true`, and refuses to launch
  them. An AMENDMENT records their design before they run.

**The verdict, per run** (`tools/tuned_rlpd_verdict.py`). A run **passes** iff the mean of its
five evals at 110k–150k (250 episodes) is at least its task's BC base rate:
- Square: B_sq = 0.508 (2026-09-29 MEASUREMENT);
- Can: B_can = 0.894 (2026-10-04 MEASUREMENT, above).

The precedent is the 2026-09-30 gate, "must at least match vanilla BC"; the paper's ~0.9 is
context, not the bar. **No early stop, and no collapse rule:** these runs start near zero.
Reported, deciding nothing: the step-0 eval (about 0 expected, from a randomly initialized actor),
best, final, every eval, and `train/action_l1_magnitude`.

**What the outcomes mean,** decided now:
- **A ResFiT arm passes:** the setup reproduces the paper's Tuned RLPD on that task, at least to
  the BC bar, and an equivariant arm there can be read against it.
- **A ResFiT arm fails:** the setup does not reproduce the paper on that task, and an equivariant
  result there can't be read.
- Once the equivariant arms run:
  - **both stacks pass:** the equivariant stack learns in vanilla RL; next is stage 2, 3 seeds;
  - **ResFiT passes and the equivariant arm fails:** the problem is in the equivariant stack, not
    in residual RL.

**Against earlier runs.** No earlier run is vanilla. The trainer runs the training steps of
upstream's `train_rlpd_dexmg.py`, which no run here has used (ARCHITECTURE.md, "Residual or
vanilla"). Env initial states are unseeded, as in every run.

**Held fixed:**
- **No caches.** `write_buffer_caches=false`, so each run builds its buffers in memory. The vanilla
  cache keys add `base_policy` and `normalization`, so they can never name a residual buffer
  (TODO B7, S9).
- **Environment:** `boce-WS-01`, torch 2.6.0, torchrl 0.7.0, escnn 1.0.11.
- **The commit:** the one that adds this entry. Its residual path is `b05736d`'s (2026-10-04 CODE).

**Resources.** About 8 h per ResFiT arm (`msfkjwab` took 16.4 h for 300k), one per GPU, at once.
- The debug runs (2026-10-04 CODE) peaked at 14.2 GB RSS on Square and 12.2 GB on Can, with 10k
  online transitions.
- The online buffer grows by about 85 kB a transition, so about 27 GB and 25 GB at 150k. The two
  fit beside each other.

**Records.** A LAUNCHED entry with the startup checks: no base policy loaded, buffers built in
memory, step-0 about 0, the speed and the RAM. A RESULTS entry per run, from
`tools/tuned_rlpd_verdict.py`.

---

## 2026-10-04 — LAUNCHED — tuned RLPD's ResFiT arms: `8ma5alnm` (Square) and `lg8qad3l` (Can)

| Arm | Seed | wandb | GPU | Commit | Launched (EDT) |
|---|---|---|---|---|---|
| `tuned-rlpd-square` | 1 | **`8ma5alnm`** | 0 | `a0353a5` | 2026-10-04 12:57 (wandb 13:19) |
| `tuned-rlpd-can` | 1 | **`lg8qad3l`** | 1 | `a0353a5` | 2026-10-04 12:58 (wandb 13:19) |

- **How:**
  - queue lines `a0353a5 tuned-rlpd-square 1` and `a0353a5 tuned-rlpd-can 1`;
  - the runners in tmux `gpu0` and `gpu1` (started 2026-10-02 from `~/projects/equi-resrl-b667647`,
    with `CACHE_DIR=~/projects/equi-resrl`) launched them in the new worktree
    `~/projects/equi-resrl-a0353a5`;
  - consoles in `outputs/queue/20261004-125755_tuned-rlpd-square-1_gpu0.log` and
    `20261004-125812_tuned-rlpd-can-1_gpu1.log`;
  - no watcher: each run ends at 150k.
- **Gate:** GO for both, with no warnings. Tier 1: 163 passed, 2 skipped. Tier 2: 351 passed.
  - The two pending equivariant arms: GO on Tiers 1 and 3 from the same worktree. Their Tier 2 was
    skipped, since the two launch gates ran it on that commit.
- **Startup verified, both runs:**
  - "No base policy"; the `ActionScaler` not widened (factor 0.0); `Set random seed to 1`;
  - buffers built in memory, nothing written: 80,429 (Square) and 62,454 (Can) demo transitions,
    under keys `a2d779c4`/`6e94049f` and `83506f61`/`3718c816`;
  - 10k warm-up steps of U(±1): 0 successes in 33 and 50 episodes;
  - the resolved configs in wandb have `base_policy: None`, `equivariance: None`, `num_q: 10`, actor
    LR 1e-4, τ 0.01, `action_scale` 1, init scale None, critic warm-up 0,
    `stddev_schedule: linear(0.1,0.1,300000)`, 150k steps, and no cache writes.
- **Step-0 eval:** 0/50 on both, from a randomly initialized actor.
- **Speed:** 0.112 s/step (Square) and 0.120 (Can) over steps 1k–3k, from each run's history.
  An eval takes about a minute. So 150k ends in about 5–5.5 h, around 18:30–19:00 on
  2026-10-04, sooner than the pre-registration's 8 h, which came from `msfkjwab` on the other
  machine.
- **Memory:** 13.8 GiB (Square) and 11.1 GiB (Can) PSS per trainer process, not counting the env
  workers, at about 13k online transitions. With both running, 69 GB of 125 are available.
- **The first 3k steps, as observations only; the verdict is at 110k–150k:**
  - **Training episodes:** 0 successes in 10 (Square) and 16 (Can); every one timed out.
  - **`train/action_l1`:** 0.75–0.80 (Square) and 0.84–0.91 (Can). The Tanh actor sits near its
    bound, as in the debug runs. Actor gradient norm 0.0015–0.05.
  - **Critic loss:** 1–2e-4 (Square), under 5e-5 (Can).
- **Config caveats:** none on this stack. The ResFiT agent does not log `debug/dQ_da_*`
  (TODO B13).

---

## 2026-10-06 — RESULTS — tuned RLPD's ResFiT arms: neither learned, on either task

Seed-group `tuned-rlpd-1002` (2026-10-04 PRE-REGISTERED and LAUNCHED). Both runs finished 150k on
2026-10-04, `8ma5alnm` at 18:21 and `lg8qad3l` at 18:35, about 5.0 and 5.3 h after wandb start.
Verdicts from `tools/tuned_rlpd_verdict.py`.

| Run | Task | Step 0 | Evals 10k–150k | Training episodes | 110k–150k mean | Bar | Verdict |
|---|---|---|---|---|---|---|---|
| `8ma5alnm` | Square | 0.00 | 0.00 at all 15 | 0 successes in 500, every one a 300-step timeout | 0.000 | B_sq = 0.508 | does not pass |
| `lg8qad3l` | Can | 0.00 | 0.00 at all 15 | 0 successes in 750, every one a 200-step timeout | 0.000 | B_can = 0.894 | does not pass |

- **Read against the pre-registration: "a ResFiT arm fails."** The setup does not reproduce the
  paper's Tuned RLPD on either task, so an equivariant tuned-RLPD result can't be read against it
  yet. The equivariant arms stay unlaunched.
- **Against the paper.** Read off its Fig. 5, Tuned RLPD is at about 0.7 on Can by 25k and about
  0.9 by 50k, and on Square at about 0.2 at 25k, 0.6 at 50k and over 0.9 by about 100k. So by 30k
  the paper's curve is clearly off zero on both tasks; ours never is.
- **What the logs show** (console every 100 steps, and the local history):

  | Step | 100 | 1k | 5k | 10k | 50k | 150k |
  |---|---|---|---|---|---|---|
  | Square `train/action_l1_magnitude` | 0.998 | 0.789 | 0.686 | 0.714 | 0.731 | 0.676 |
  | Square `train/critic_qt` | 0.04 | 0.43 | 0.47 | 0.42 | 0.39 | 0.40 |
  | Square `train/actor_grad_norm` | 0.0025 | 0.038 | 0.028 | 0.017 | 0.008 | 0.0036 |
  | Can `train/action_l1_magnitude` | 0.929 | 0.839 | 0.884 | 0.876 | 0.862 | 0.832 |
  | Can `train/critic_qt` | −0.09 | 0.37 | 0.40 | 0.42 | 0.42 | 0.41 |
  | Can `train/actor_grad_norm` | 0.013 | 0.0028 | 0.0029 | 0.0026 | 0.0052 | 0.0033 |

  - **The actor reached its Tanh bound within about 100 actor updates,** before the critic had
    learned anything (`critic_qt` 0.04 and −0.09 at step 100), and never came back.
  - **The demos are far from that.** The 300 demos' actions, normalized as the buffer stores
    them, average |a| = 0.29 (Square) and 0.32 (Can) over all 7 axes, and that includes the
    gripper axis, which is always ±1. The six arm axes average 0.03–0.37.
  - **The critic values the policy's own rollouts at about zero,** correctly: the eval Q-trajectory
    plots at 150k sit within 4e-4 of 0 on both tasks (Square was about 0.095 at 10k).
  - **Yet the batch-mean TD target stays at about 0.40.** Half of each batch is demos. If the
    online half's targets are about 0, the demo half's average about 0.8. The demos' true mean
    value is 0.335 (Square) and 0.398 (Can): γ^(steps to success) over the 300 demos' transitions.
    So, **inferred and not measured,** the critic values the policy's saturated actions on demo
    states at about twice what the demonstrator's own actions earn. That removes any gradient
    back toward the demos. The ResFiT agent logs no split of Q by half, and `debug/dQ_da_*` reads
    0 at every step (TODO B13).
  - **Then everything stops moving.** Critic loss under 5e-5 from about 7.5k; encoder gradient
    from 0.02 at step 100 to about 2e-5; actor gradient 0.003–0.008 for the last 100k steps.
- **Not mainly exploration.** With demos in half of every batch, RLPD does not need an online
  success to see reward. Here the actor never approached the demos' actions, so the missing
  online successes follow from the failure rather than cause it.
- **Against upstream.** No hyperparameter of these runs differs from upstream's released
  `rlpd_{square,can}_config` with its original 10 critic heads, and the two trainers build the
  same `QAgent` (ARCHITECTURE.md, "Residual or vanilla"). The differences found are between
  that config and the paper's text:
  - **critic warm-up:** 0 here and in upstream's RLPD config, and upstream's
    `train_rlpd_dexmg.py` has no warm-up code at all. The paper says demos are used "to warm up a
    critic", and the residual config does 10k critic-only updates;
  - **target τ:** 0.01 here and in upstream's RLPD config; 0.005 in the paper's text.
- The same pattern as the image-fix pair (2026-10-04 RESULTS): an actor that runs to its bound
  while the critic scores the bound above the data's actions. A reading, not a test.
- One seed per task: an observation, not a comparison (STANDARDS 5.3). Packages in
  `outputs/runs/robomimic-{square,can}-final/{8ma5alnm,lg8qad3l}/`, with each eval's Q-trajectory
  plot. No best checkpoint was saved: no eval beat 0.

---

## 2026-10-06 — CODE — the critic's Q split by buffer, `dQ_da` on ResFiT's agent, and the Can screen's launcher

**What it adds,** for the screen pre-registered below:
- **`QAgent.critic_diagnostics`** (ResFiT's agent). On the last update's batch, it reports the
  critic's Q for the data's action and for the policy's, on the online rows and on the demo rows,
  as `debug/q_data_{online,demo}` and `debug/q_policy_{online,demo}`. It also reports
  `debug/policy_data_gap_{online,demo}`, the mean |policy action − data action| per axis.
  - Q is the mean over heads. The policy's action is the actor's mean, combined with the base
    action in a residual run, as the actor loss combines it.
  - The trainer logs it at every log step, and prints the demo and online pairs on the console
    line. ResFiT's agent only, for now: the equivariant agent needs the same method before its
    tuned-RLPD arms run (TODO B12).
  - Logging only: the agent is in eval mode, the images are not augmented, and nothing draws a
    random number, so a run computes the same updates whether or not it logs.
- **`debug/dQ_da_*` now reads the critic's action gradient on ResFiT's agent** (TODO B13). It
  read 0.0 in every ResFiT run before this commit, `8ma5alnm` and `lg8qad3l` included:
  `combined_action` is not a leaf, and its gradient was never kept. One `retain_grad()`, as the
  equivariant agent has.
- **`submit.sh` is rewritten for the screen:** six arms. One, `upstream-rlpd-can`, runs upstream's
  `train_rlpd_dexmg.py`, with its caches under `$CACHE_DIR/local_runs/upstream-rlpd`, wandb
  offline, and a gate step that refuses it under 26 GB of free disk. The 2026-10-04 arms launch
  from `a0353a5`.

**Verified:**
- **The config claim:** `tests/test_sweep_arms.py` shows `tuned_rlpd_can_config` equal to
  upstream's `rlpd_can_config`, at 10 critic heads, on every field the two share, wandb naming
  aside. The fields only ours has are residual-only ones, off, and the two normalization floors,
  which upstream's trainer hard-codes at the same 0.1.
- **The diagnostics:** `tests/test_critic_diagnostics.py`. They match the same quantities
  computed from the agent's parts, in vanilla and residual mode. They skip an empty demo half,
  and they move no parameter and draw no random number on CPU or CUDA. `dQ_da` is now nonzero
  after an actor update.
- **Training numerics unchanged:** neither change alters an update. The equivariant agent is
  untouched, and its goldens and deep-check tests pass.
- **The full suite: 533 passed, 2 skipped** (the two skips predate this).
- **Upstream's trainer, end to end** (`debug=true`, 500 warm-up steps, 300 steps, Can): it runs
  here, its first run on this machine.
  - `action_l1` was 0.97 at step 100, as in `lg8qad3l`.
  - **Its Can offline buffer is sized horizon × episodes = 200 × 300 = 60,000** for 62,456 demo
    transitions, so the last 2,456 overwrite the first. Square's 90,000 slots hold its 80,431.
    Ours sizes the buffer exactly.
  - It wrote the offline cache `5b67f661` (4.8 GB) and an online cache `eef0c5e8`. The online
    buffer is dumped at its full 200k capacity: 16 GB for 500 transitions.
- **Our control, end to end** (`debug=true`, 500 warm-up steps, 300 steps, 200 critic warm-up
  updates): the diagnostics reach the console and the log, and critic warm-up runs with no base
  policy. Seen, deciding nothing: the demo rows already valued the policy's action above the
  demo's, 0.343 against 0.306 at step 300.

---

## 2026-10-06 — PRE-REGISTERED — tuned RLPD on Can: upstream's trainer, and a screen of three changes

**Questions.** Both on Can, the cheaper task and the faster one in the paper: its Tuned RLPD is
at about 0.7 there by 25k, so 30k steps can tell learning from not learning.
1. **Does upstream's own trainer fail the way ours did?** Our `tuned_rlpd_can_config` equals
   upstream's `rlpd_can_config`, at its original 10 critic heads, on every field the two share
   (`tests/test_sweep_arms.py` checks it), and both trainers build the same `QAgent`. So the
   expectation is yes. If upstream learns and our control does not, our mode differs from
   upstream somewhere.
2. **Which of three changes from ResFiT's residual recipe gets tuned RLPD off zero?** Each targets
   the failure the 2026-10-06 RESULTS describe: an actor that reaches its Tanh bound within about
   100 updates, against a critic that has learned nothing yet.

**Seed-group `tuned-rlpd-can-screen-1006`,** seed 1 per arm, 30k steps, Can.
- Project `robomimic-can-final`. Groups `tuned-rlpd` (ours) and `upstream-rlpd`.
- Names `<arm>-1`, notes `tuned-rlpd-can-screen-1006/<arm>/seed1`.

| Arm | Trainer | Change from `tuned_rlpd_can_config` | Why |
|---|---|---|---|
| `upstream-rlpd-can` | `train_rlpd_dexmg.py` | upstream's `rlpd_can_config` with `agent.critic.num_q=10`, as `shell/paper_runs/can/2_can_rlpd.sh` launches it | question 1 |
| `tuned-rlpd-can-control` | `train_residual_td3.py` | none: `lg8qad3l`'s config, with the new diagnostics (2026-10-06 CODE) | the screen's control, and the test of the mechanism |
| `tuned-rlpd-can-actor-lr` | ours | `agent.actor_lr=1e-6`, the residual config's, 100× lower | slows the run to the bound |
| `tuned-rlpd-can-critic-warmup` | ours | `algo.critic_warmup_steps=10000`, the residual config's | the actor's first update meets a trained critic; the paper says demos "warm up a critic", and upstream's RLPD trainer has no warm-up at all |
| `tuned-rlpd-can-zero-init` | ours | `agent.actor.actor_last_layer_init_scale=0.0`, the residual config's | the actor starts at the dataset midpoint, not at a random action |
| `tuned-rlpd-can-combined` | ours | all three | whether they only work together |

- **The residual's third safeguard, its 0.1 cap, is not tested** (Aden's decision, 2026-10-06).
  With no base policy, `action_scale=0.1` bounds the whole action, around each axis's dataset
  midpoint. Can's yaw axis spans −0.927 to 0.424, so the capped actor could only command −0.18 to
  −0.32: a yaw of 0.09–0.16 rad every step that it cannot cancel. It would also cap translation
  at about half the demos' average. `zero-init` keeps the recipe's "start small" without that
  bound.
- **Not changed, though the residual config differs:** τ (0.01, not 0.005), exploration σ (0.1,
  not 0.05) and the warm-up actions (U(±1)). They stay as in `lg8qad3l`.

**Held fixed:** everything else in `tuned_rlpd_can_config`, which is `lg8qad3l`'s: 10 critic
heads, critic LR 1e-4, 4 updates per step with the actor on the last, n-step 3, γ 0.99, batch 256
half demos, 300 demos, 10k U(±1) warm-up steps, buffers built in memory. Evals of 50 episodes at
0, 10k, 20k and 30k. The commit is the one that adds this entry; env initial states are unseeded,
as in every run.

**How upstream's arm differs in its running, not its config:**
- **Its Can offline buffer holds 60,000 transitions, not 62,456.** It sizes the buffer
  horizon × episodes = 200 × 300, so the last 2,456 demo transitions overwrite the first.
  Found in today's smoke run; an upstream bug, kept, since this arm tests upstream as it is.
- Its last eval is at step 29,999, not 30,000.
- It writes its buffer caches, under `local_runs/upstream-rlpd/`, away from ours. It loads the
  offline cache `5b67f661`, which today's smoke run built at this code, and writes a 16 GB
  online cache.
- **wandb runs offline,** since its trainer uploads every best checkpoint (STANDARDS 6.4). The
  history stays in its local wandb record, and at its end the trainer deletes its own run
  folder, models and videos included. Its console log in `outputs/queue/` stays.
- It logs no diagnostics beyond upstream's.

**The verdict, per arm,** from its evals at 20k and 30k (100 episodes):
- **Off zero** iff those two evals score at least 5 successes between them. `lg8qad3l` scored 0
  in 800 eval episodes and 0 in 750 training episodes; the paper's curve is about 0.5–0.7 there.
  So 5 of 100 says learning has started, without asking for the paper's number.
- **Promoted:** the off-zero arm of ours with the most successes at 20k and 30k runs 150k next,
  as an AMENDMENT to `tuned-rlpd-1002`, judged by that experiment's rule (the 110k–150k eval
  mean against B_can = 0.894). A tie goes to the arm with fewer changes; a remaining tie, to the
  higher 30k eval; a remaining tie, to Aden.
- **Reported, deciding nothing:** every eval, training successes, and `train/action_l1_magnitude`
  at steps 100, 1k and 10k against `lg8qad3l`'s 0.93, 0.84 and 0.88.

**What the outcomes mean,** decided now:
- **Upstream off zero and our control not:** our mode differs from upstream. The screen's
  results are set aside until the difference is found.
- **Neither off zero:** our mode matches upstream in outcome. Upstream's released config does not
  reproduce the paper's Fig. 5 on Can here, and the screen says whether any of the three changes
  does.
- **No arm off zero:** none of the three changes, alone or together, is enough by 30k. What to try
  next is decided from the diagnostics; the candidates are τ 0.005 (the paper's text), a BC term,
  and a lower exploration σ.
- **The mechanism,** from the control's new diagnostics at 10k, 20k and 30k. The 2026-10-06
  RESULTS reading predicts the demo half valued well above the demos' true mean of 0.398, about
  0.8, with `debug/q_policy_demo` at least `debug/q_data_demo`, and the online half near 0. If the
  demo values sit near 0.398 instead, that reading is wrong, and it gets a CORRECTION.
- One seed per arm: observations, not comparisons (STANDARDS 5.3). The screen chooses what runs
  next; it does not establish an effect.

**Resources.** Six runs of about 1–1.5 h, four at a time: a second queue runner on each GPU, in
tmux `gpu0b` and `gpu1b`. About 15 GB of RAM per run, of 121 GB available. Disk: 29 GB free
before upstream's 16 GB online cache.

---

## 2026-10-06 — CORRECTION — ResFiT's critic was never orthogonally initialized; `agent.critic.orth` does nothing

- **Found** by the session running the 2026-10-06 audit (`outputs/audit-2026-10-06/`), and
  reproduced here.
- **Why:**
  - `SpatialEmbQEnsemble._init_per_head_params` (`off_policy/rl/critic.py:560–570`) calls
    `utils.orth_weight_init(param[h])` on a raw tensor.
  - `orth_weight_init` (`off_policy/common_utils/utils.py:66–75`) acts only on `nn.Linear` and
    `nn.Conv` modules, so the call does nothing.
  - The ten heads keep PyTorch's default `nn.Linear` init, and nothing else in the critic reads
    `orth`.
- **Measured, twice:**
  - The audit's config tracer rebuilt the ResFiT agent with `agent.critic.orth` 0 and 2 instead
    of 1. State dict, metrics and actions were bit-identical each time.
    `outputs/audit-2026-10-06/config_physics/t2_report.txt` and `t2_sens*_*.json`.
  - Here: `tuned_rlpd_can_config`'s critic, seed 0, `orth=1` against `orth=0`: all 25 tensors
    equal. Head 0's first weight has max |WWᵀ − I| = 0.70, so it is not orthogonal. Its biases
    reach 0.031 = 1/√1040, PyTorch's default `nn.Linear` bound.
- **Scope:**
  - It is upstream code: the same line is at `6bd9f7c`, `critic.py:550`.
  - So it covers every ResFiT run in this project, the working residual baselines `870ws2c2` and
    `msfkjwab` included, and upstream's `train_rlpd_dexmg.py`.
  - All six arms of `tuned-rlpd-can-screen-1006` share it, so it confounds none of the screen's
    comparisons, nor the question of whether our mode matches upstream's.
  - The equivariant critic is a different module and is not affected.
- **What it corrects:** the 2026-10-04 PRE-REGISTERED table, "Critic head … 2 hidden layers of
  1024, LayerNorm, orthogonal init", for ResFiT's arm. ResFiT's critic has PyTorch's default init
  throughout. The untracked plan's table has the same error.
- **Not changed:** the code. Fixing it would change the critic's init in every ResFiT run, and so
  break comparability with `870ws2c2`, `msfkjwab` and the screen (CLAUDE.md, priority 1). So
  `agent.critic.orth` stays a field that does nothing, against STANDARDS 1.2 (TODO B13).

---

## 2026-10-06 — LAUNCHED — the tuned-RLPD Can screen (`tuned-rlpd-can-screen-1006`)

| Arm | Runner | wandb | Launched (EDT) |
|---|---|---|---|
| `upstream-rlpd-can` | `gpu0` | **`rdoj0ex8`**, offline: `~/projects/equi-resrl-187d895/wandb/offline-run-20261006_105334-rdoj0ex8` | 10:13; training from 10:53 |
| `tuned-rlpd-can-control` | `gpu1` | **`r1hvijqh`** | 10:14; training from 10:59 |
| `tuned-rlpd-can-zero-init` | `gpu0b` | at its start | 11:02 |
| `tuned-rlpd-can-critic-warmup` | `gpu1b` | at its start | 11:02 |
| `tuned-rlpd-can-combined` | `gpu0`, after upstream's | — | queued |
| `tuned-rlpd-can-actor-lr` | `gpu1`, after the control | — | queued |

- **All from `187d895`,** in the worktree `~/projects/equi-resrl-187d895`, with
  `CACHE_DIR=~/projects/equi-resrl`. Consoles in `outputs/queue/20261006-*_<arm>-1_gpu<g>.log`.
- **The runners differ from the pre-registration's plan, not the runs.** Another session's
  Square base-rate audit (`outputs/audit-2026-10-06/`) had both GPUs at 98–99% until about 11:00.
  So two arms started at 10:13, and the second runners, `gpu0b` and `gpu1b`, started at 11:02.
- **Gates:** GO for the first two. They took about 40 min, sharing the GPUs with the audit.
  - The gate's Tier 2 file list does not include the new `tests/test_critic_diagnostics.py`. It
    passed in the full suite at `187d895` (2026-10-06 CODE) (TODO B13).
- **Startup:**
  - **Upstream's arm** loaded the offline cache `5b67f661` at 60,000 transitions (the overwrite
    the pre-registration describes). It wrote its online cache `2d0562d5`, 16 GB, under
    `local_runs/upstream-rlpd/`.
  - **The control** built 62,456 demo transitions in memory, into a buffer sized 62,454. So our
    trainer also overwrites 2 of them: it sizes the buffer from the dataset's metadata, which
    is 2 transitions short. The same was true of `lg8qad3l`, whose LAUNCHED entry says 62,454.
  - Step-0 evals: 0/50 on both, from a randomly initialized actor.
- **The first steps, observations only; the verdict is at 20k–30k:**
  - **`action_l1` at step 100:** upstream 0.965, the control 0.926, against `lg8qad3l`'s
    0.929. By 1k, 0.85 on both, and 0.89–0.91 by 3–6k. Upstream's trainer runs to the Tanh
    bound as ours does.
  - **The control's critic diagnostics at 3.3k:** on the demo rows, Q of the data's action 0.786
    and of the policy's 0.808, against the demos' true mean of 0.398. On the online rows, −0.004
    and −0.002. That is what the 2026-10-06 RESULTS reading predicts. The pre-registered check of
    it is at 10k, 20k and 30k.
- **Memory:** 11.8 GiB RSS per trainer at the start; 70 GB available with two running.
- **Disk:** 29 GB free after upstream's online cache. The smoke run's online cache `eef0c5e8` had
  been deleted outside this session by then.

---

## 2026-10-06 — RESULTS — the tuned-RLPD Can screen: no arm got off zero, upstream's trainer included

Seed-group `tuned-rlpd-can-screen-1006` (2026-10-06 PRE-REGISTERED and LAUNCHED). All six arms ran
30k steps from `187d895` and exited 0, between 12:21 and 14:06 EDT. Read from each run's local
wandb record. Step means the env step after the 10k random warm-up steps, with 4 critic updates
and 1 actor update per step.

| Arm | wandb | Wall | Evals 0 / 10k / 20k / 30k | Successes, 20k + 30k | Training episodes | `action_l1` at 100 / 1k / 10k | Verdict |
|---|---|---|---|---|---|---|---|
| `upstream-rlpd-can` | `rdoj0ex8` (offline) | 128 min | 0 / 0 / 0 / 0 (last at 29,999) | 0 of 100 | not logged by its trainer | 0.965 / 0.845 / 0.878 | not off zero |
| `tuned-rlpd-can-control` | `r1hvijqh` | 139 min | 0 / 0 / 0 / 0 | 0 of 100 | 0 successes in 150 | 0.926 / 0.856 / 0.875 | not off zero |
| `tuned-rlpd-can-actor-lr` | `0ama3t4v` | 92 min | 0 / 0 / 0 / 0 | 0 of 100 | 0 in 150 | 0.509 / 0.664 / 0.788 | not off zero |
| `tuned-rlpd-can-critic-warmup` | `qiadwtn1` | 132 min | 0 / 0 / 0 / 0 | 0 of 100 | 0 in 150 | 0.876 / 0.903 / 0.799 | not off zero |
| `tuned-rlpd-can-zero-init` | `85fpzqmp` | 118 min | 0 / 0 / 0 / 0 | 0 of 100 | 0 in 150 | 0.793 / 0.875 / 0.821 | not off zero |
| `tuned-rlpd-can-combined` | `u4cwl6bt` | 93 min | 0 / 0 / 0 / 0 | 0 of 100 | 0 in 150 | 0.029 / 0.276 / 0.660 | not off zero |
| reference: `lg8qad3l` | | | 0 at all 16 evals to 150k | | 0 in 750 | 0.929 / 0.839 / 0.876 | |

Evals are 50 episodes each. Every training episode of our five arms was a 200-step timeout:
150 × 200 is all 30,000 steps. Wall times include the two GPUs being shared by four runs, and
the first two arms' gates sharing them with the audit (LAUNCHED entry).

**Read against the pre-registration:**
- **"Neither off zero": our mode matches upstream in outcome.** Upstream's released config, run by
  its own trainer as the paper's launcher runs it, does not reproduce the paper's Fig. 5 on Can
  here (about 0.7 by 25k there; 0 in 200 eval episodes here). Its logs show the same course as
  ours: `action_l1` 0.965 at step 100, `critic_qt` 0.45 and `debug/dQ_da_mean_abs` 6.5e-6 at 30k,
  against the control's 0.44 and 6.8e-6.
- **"No arm off zero": none of the three changes is enough by 30k, alone or together.** None
  is promoted, and `tuned-rlpd-1002` gets no AMENDMENT from this screen.
- **The mechanism holds.** The control's critic diagnostics at 10k, 20k and 30k:

  | Step | 10k | 20k | 30k |
  |---|---|---|---|
  | `debug/q_data_demo` | 0.840 | 0.829 | 0.876 |
  | `debug/q_policy_demo` | 0.866 | 0.852 | 0.902 |
  | `debug/q_data_online` | −0.001 | 0.000 | 0.010 |
  | `debug/q_policy_online` | −0.001 | 0.000 | 0.010 |
  | `debug/policy_data_gap_demo` | 0.90 | 0.78 | 0.78 |

  That is what the reading predicted: demo states valued at about 0.8, against the demos' true
  mean of 0.398; the policy's action valued at least as high as the demo's; the online half near
  0. No CORRECTION. The other four arms of ours reach the same state by 10k (below).
- One seed per arm: observations, not comparisons (STANDARDS 5.3).

**What the diagnostics add, from all five arms of ours.** Readings, single-seed:
- **A small, constant preference for the policy's action is enough to explain the inflated
  values.** Call δ = `q_policy_demo` − `q_data_demo`. From 5k to 30k it averages +0.022 to
  +0.026 in each arm, and is positive at all 251 log steps of every arm. The TD target for a demo
  transition bootstraps through the policy's action three steps on, not the demo's. So each
  3-step chunk adds γ³δ to the demo value, until the episode ends:
  - the expected demo value is 0.398 + (1 − 0.398) · γ³δ / (1 − γ³);
  - the factor 1 − 0.398 is the average over demo transitions of 1 − γ^(steps to success),
    which is how much of the bonus a state's remaining chunks can collect;
  - γ³ / (1 − γ³) is about 33, so a δ of 0.024 lifts the mean demo value by about 0.47.
  - Predicted against observed, mean from 5k: control 0.87 / 0.83, zero-init 0.90 / 0.85,
    critic-warmup 0.84 / 0.79, combined 0.85 / 0.80, actor-lr 0.87 / 0.82.
  - Every prediction is about 0.05 high. A guess at why: the target takes the min of 2 heads and
    a smoothed action, while δ is measured on the 10 heads' mean and the policy's mean action.
  - This is the extrapolation error of offline RL (BCQ, Fujimoto et al. 2019): the critic is
    trained only on the demos' actions at demo states and is queried at the policy's. It does
    not diverge here. δ stays small and γ³ < 1, so the values settle; `critic_qt` peaks at 0.47
    across all arms. **Not a Q blowup.**
- **Why the actor reaches the bound so fast, whatever the change:** Adam's step is about the
  learning rate whatever the gradient's size, as long as its sign holds.
  - The critic's slope at the policy's action is tiny: `debug/dQ_da_mean_abs` 5e-6 to 1.1e-5 from 5k
    in every arm, well under STANDARDS 6.3's "nearly flat" 1e-4. It is the gradient of the
    batch-mean loss, so the control's 6.5e-6 is about 1.7e-3 per sample per axis.
  - But the slope's sign holds, so the actor moves at the full rate along it: to the bound
    within about 100 updates at LR 1e-4 (control, zero-init, critic-warmup).
  - At LR 1e-6 it takes thousands: 0.51 at step 100, 0.79 by 5k (actor-lr); 0.66 at 10k and 0.85
    at 20k (combined).
- **Starting the actor at the dataset midpoint kept δ at about 0, until the actor left it.**
  The two arms with a 10k critic warm-up differ in their actor's start:
  - Critic-warmup started at a random actor. At step 100 its demo values were already 0.74
    (data) and 0.80 (policy), so δ = +0.06.
  - Combined started at zero-init, its actions 0.31 per axis from the demos' on average. After
    the same 10k critic-only updates, its demo values were 0.18 and 0.18, so δ = −0.005.
  - Combined's actor then drifted outward anyway, at its slow rate. As it did, δ turned positive
    and the demo value climbed: 0.57 at 1k, 0.77 at 5k.
  - So nothing in the critic pulls the actor toward the demos' actions. Its slope at the demo
    actions, on demo states, points away from them.
- **The online half carries no signal.** The saturated policy's rollouts leave the demos'
  states at once, and never succeed. So the critic values every online state at about 0,
  whatever the action, which is correct. The actor's only gradient comes from the demo half,
  and there it points to the bound.
- **What this says about the candidates the pre-registration named:**
  - **τ 0.005** changes how fast the target tracks the critic, not the fixed point above. So it is
    not expected to help. It is the paper's stated value, so it is still worth a run as a
    record.
  - **A lower exploration σ** has nothing to act on: the online half is at 0 for any action.
  - **A BC term on the demo half** gives the actor what the critic does not: a gradient back
    toward the demos' actions on demo states. Upstream's `QAgent.update_actor_rft` has one, and
    upstream's trainer passes it the offline half when `agent.bc_loss_coef > 0`. Ours ignores
    `bc_loss_coef` (a dead field, STANDARDS 1.2). The paper's Tuned RLPD does not use one:
    `rlpd.py`'s default is 0, and the paper's text describes none.
  - What to run next needs its own pre-registration, and Aden's approval.
- Packages in `outputs/runs/robomimic-can-final/{r1hvijqh,0ama3t4v,qiadwtn1,85fpzqmp,u4cwl6bt}/`.
  None saved a best checkpoint, since no eval beat 0. Each manifest records commit `187d895`;
  the only path it lists as uncommitted is `artifacts/`, the runs' own output.
- **Upstream's arm** has no package: its trainer deleted its run folder at the end, as the
  pre-registration says. Its history is in
  `~/projects/equi-resrl-187d895/wandb/offline-run-20261006_105334-rdoj0ex8/`, and its console log in
  `outputs/queue/20261006-101334_upstream-rlpd-can-1_gpu0.log`. Its caches,
  `local_runs/upstream-rlpd/` (about 21 GB), are still on disk, for Aden to delete.

---

## 2026-10-06 — CODE — the actor's BC term, live in our trainer (TODO B14)

`train_residual_td3.py` ignored `agent.bc_loss_coef`: its update always passed `bc_batch=None`.
Now it passes the term what upstream's `train_rlpd_dexmg.py` passes it.
- **The trainer:** on each actor update, when `agent.bc_loss_coef > 0`, it hands `agent.update`
  the batch's demo half as `bc_batch`. That is the separately sampled `offline_batch`, the same
  rows the critic's batch concatenates. `QAgent.update_actor_rft`, upstream's code and unchanged
  apart from logging, then adds `bc_loss_coef × Σ_axes (π(s) − a_demo)²`, averaged over the demo
  rows, to the actor's loss. The BC loss encodes those images again, with its own random shift.
  It detaches the features (`bc_backprop_encoder`, upstream's default False), so the term trains
  the actor and not the encoder.
- **Refused at startup:** a positive coefficient in a residual run (upstream's BC loss asserts a
  vanilla actor), on the equivariant agent (its BC path has no test), or with
  `algo.offline_fraction = 0`. Also `agent.bc_loss_dynamic` at any value but 0: its weight needs
  IBRL's `act_method`, which `QAgentConfig` does not have, so it would crash at the first update.
- **Logging:** `update_actor_rft` now logs `debug/dQ_da_*` as `update_actor` does. It is the
  critic's slope alone: the BC term reaches the actor through its own forward pass on `bc_batch`.
  It also logs `rft/bc_loss` and `rft/ratio` (always 1 here).
- **At `bc_loss_coef = 0`, every run before this one:** the trainer passes no demo half, and the
  actor takes `QAgent.update_actor`, as before. `tests/test_regression_agent.py` pins that update
  bit for bit, and passes.
- **Tests:** `tests/test_bc_term.py`, in the gate's Tier 2. On the fixed test batch, 100 updates
  at 0.025 moved the actor toward the demo rows' actions, 0.50 to 0.37 per axis. The same 100
  updates without the term moved it away, 0.50 to 0.86, as the tuned-RLPD actors do.
  `tests/test_sweep_arms.py` now passes the demo half to the BC arms' GPU update.
- **Not unit-tested:** that the trainer itself passes the demo half. A smoke run checked it: the
  trainer at `agent.bc_loss_coef=0.025`, 300 warm-up steps, 400 steps, debug mode, from the
  working tree of this commit. It ran without error, and `action_l1` read 0.58–0.62 at steps
  100–400. Without the term it reads 0.93 at step 100 (the screen's control) or 0.79 from a
  zero-initialized start. So the term reaches the actor.

---

## 2026-10-06 — PRE-REGISTERED — tuned RLPD on Can: τ 0.005 and a BC term

**Questions,** both on Can, after the screen (2026-10-06 RESULTS, the screen) found no fix among
the residual recipe's safeguards. Aden asked for these two on 2026-10-06. A critic-only LayerNorm
arm and a longer n-step arm were also proposed, and not chosen.
1. **Does τ 0.005, the paper's stated value, change the outcome?** Upstream's `rlpd.py` has 0.01.
   The paper lists τ in the recipe it says Tuned RLPD shares (Sec. III-C). Of the nine fields in
   which upstream's residual and RLPD configs differ, it is the only one the paper does not tie to
   the residual. The screen's reading says no: τ sets how fast the target tracks the critic, not
   the fixed point the demo values settle at.
2. **Does a BC term on the demo half get tuned RLPD off zero?** TD3+BC's remedy for the
   extrapolation error the screen measured. Nothing in the critic pulls the actor toward the
   demos' actions; the term does. **It departs from the paper's Tuned RLPD,** which uses none
   (upstream's `bc_loss_coef` is 0, and the paper describes none).

**Seed-group `tuned-rlpd-can-tau-bc-1006`,** seed 1 per arm, 30k steps, Can, from the commit that
adds this entry.
- Project `robomimic-can-final`, group `tuned-rlpd`.
- Names `<arm>-1`, notes `tuned-rlpd-can-tau-bc-1006/<arm>/seed1`.

| Arm | Change from `tuned_rlpd_can_config` | Why |
|---|---|---|
| `tuned-rlpd-can-tau` | `agent.critic_target_tau=0.005` | question 1 |
| `tuned-rlpd-can-bc` | `agent.bc_loss_coef=0.025` | question 2 |
| `tuned-rlpd-can-tau-bc` | both | whether they only work together |

**The control** is the screen's `tuned-rlpd-can-control` (`r1hvijqh`, at `187d895`): this config
with no change. It is not run again. At `bc_loss_coef = 0` the new commit's code path is the old
one, bit for bit (2026-10-06 CODE, above), and nothing else it changes reaches a run. Env initial
states are unseeded, in every run, so a second control would not repeat the first exactly
either.

**The coefficient, 0.025, is TD3+BC's balance, not a tuned value.**
- TD3+BC (Fujimoto & Gu, 2021) maximizes λ·Q − mean over axes of (π(s) − a)², with
  λ = α / mean |Q| and α = 2.5.
- Divided by λ, that is our Q loss plus (mean |Q| / α) × the mean over axes. Upstream's term sums
  over the 7 axes instead, so the matching coefficient is mean |Q| / (7α).
- Mean |Q| over the actor's batch, both halves, in the control: −`train/actor_loss_base`
  averaged 0.44 from 20k to 30k. So 0.44 / 17.5 = 0.025.
- TODO B14's rough 0.05 used the demo half's Q, about 0.85, alone. The actor's loss averages
  over both halves.
- At 0.025 the term dominates the actor's gradient for as long as the critic's slope stays as
  flat as the screen measured. The actor should then imitate the demos on demo states, until the
  critic learns to rank actions.

**Held fixed:** everything else in `tuned_rlpd_can_config`, as `r1hvijqh` ran it: 10 critic
heads, actor and critic LR 1e-4, 4 critic updates and 1 actor update per step, n-step 3, γ 0.99,
batch 256 half demos, the same 300 MH demos, 10k U(±1) warm-up steps, no critic warm-up, the
actor's default init, exploration σ 0.1, buffers built in memory. Evals of 50 episodes at 0,
10k, 20k and 30k.

**The verdict, per arm,** by the screen's rule:
- **Off zero** iff its 20k and 30k evals score at least 5 successes between them, of 100.
- **What runs next, if an arm is off zero, is Aden's decision,** since a BC arm is not the paper's
  Tuned RLPD. The candidate is the off-zero arm with the most successes at 20k and 30k (a tie: the
  fewer changes, then the higher 30k eval). It would run 150k as an AMENDMENT to
  `tuned-rlpd-1002`, judged by that experiment's rule: the 110k–150k eval mean against
  B_can = 0.894. With the term, it is named "Tuned RLPD + BC", not Tuned RLPD, and the equivariant
  side gets the same term.
- **Reported, deciding nothing:** every eval; training successes; `train/action_l1_magnitude` at
  100, 1k and 10k; and at 10k, 20k and 30k, `debug/q_{data,policy}_{demo,online}`,
  `debug/policy_data_gap_demo`, `debug/dQ_da_mean_abs` and `rft/bc_loss`.

**Predictions, decided now:**
- **`tau`:** not off zero. The demo values settle near the fixed point again, 0.75–0.9 at 30k,
  with δ = `q_policy_demo` − `q_data_demo` positive at most log steps. Slower to get there than
  the control, if anything.
- **`bc` and `tau-bc`:** `debug/policy_data_gap_demo` falls well below the control's 0.78–0.90,
  under 0.3 by 10k. With the policy's actions near the demos', δ falls to about 0, and the demo
  values fall toward their true mean of 0.398, not 0.8.
- **Whether success follows is the open question.** A single-step actor that imitates 300 MH
  demos may or may not succeed on Can by itself. If it does, the online half gets successes and
  failures near each other, the action contrast the critic has lacked.

**What the outcomes mean,** decided now:
- **`tau` off zero:** contradicts the screen's fixed-point reading, which gets a CORRECTION.
- **A BC arm off zero:** a BC term is enough, by 30k, to start learning here. The paper's Tuned
  RLPD as described still does not reproduce here.
- **A BC arm not off zero, with its gap under 0.3:** imitating the demos does not succeed on Can
  with this actor, and the critic still cannot rank actions. The next candidates: a longer
  n-step (the paper's author told Aden it was their most sensitive setting; PROGRESS.md,
  2026-10-06), letting the BC loss train the
  encoder (`agent.bc_backprop_encoder=true`), or a larger coefficient.
- **A BC arm with its gap not under 0.3:** the coefficient is too weak against the critic, or
  the term is not reaching the actor. `rft/bc_loss` says which.
- One seed per arm: observations, not comparisons (STANDARDS 5.3).

**Resources.** Three runs at once, about 1.5–2.5 h each: runners `gpu0` (`tau`), `gpu1` (`bc`)
and `gpu0b` (`tau-bc`). About 15 GB of RAM each, of 120 GB available. No buffer caches are
written; 29 GB of disk free.

---

## 2026-10-06 — LAUNCHED — τ 0.005 and a BC term on Can (`tuned-rlpd-can-tau-bc-1006`)

| Arm | Runner | wandb | Picked up / training from (EDT) |
|---|---|---|---|
| `tuned-rlpd-can-tau` | `gpu0` | at its start | 14:44 / 15:01 |
| `tuned-rlpd-can-bc` | `gpu1` | at its start | 14:44 / 15:01 |
| `tuned-rlpd-can-tau-bc` | `gpu0b` | at its start | 14:43 / 15:00 |

- **All from `ed47957`,** in the worktree `~/projects/equi-resrl-ed47957`, with
  `CACHE_DIR=~/projects/equi-resrl`, through `tools/gpu_queue.sh`. Consoles in
  `outputs/queue/20261006-144*_<arm>-1_gpu<g>.log`. Their wandb ids are in those consoles and in
  the project `robomimic-can-final`, by name and notes.
- **Gates:** GO for all three. A dry run of `tuned-rlpd-can-bc` at the same commit passed first
  (Tier 1: 163 passed; Tier 2: 357 passed, `tests/test_bc_term.py` among them). Its one warning
  was the disk: 30 GB free, and no cache to write.
- **As pre-registered:** the runners, the arms and the commit all match. The only difference
  from the plan is time: the three gates shared the GPUs, so each took about 17 min.

---

## 2026-10-07 — RESULTS — τ 0.005 and a BC term on Can: no arm off zero

Seed-group `tuned-rlpd-can-tau-bc-1006` (2026-10-06 PRE-REGISTERED and LAUNCHED). All three arms
ran 30k steps from `ed47957` and exited 0 on 2026-10-06, at 16:13 (`bc`), 16:50 (`tau-bc`) and
16:51 (`tau`) EDT. Read from each run's local wandb record, in `~/projects/equi-resrl-ed47957/wandb/`.
Nothing else ran overnight.

| Arm | wandb | Queue time | Evals 0 / 10k / 20k / 30k | Successes, 20k + 30k | Training episodes | `action_l1` at 100 / 1k / 10k | Verdict |
|---|---|---|---|---|---|---|---|
| `tuned-rlpd-can-tau` | `ns5mis29` | 126 min | 0 / 0 / 0 / 0 | 0 of 100 | 0 successes in 150 | 0.940 / 0.854 / 0.821 | not off zero |
| `tuned-rlpd-can-bc` | `kc8z7vsf` | 88 min | 0 / 0 / 0 / 0 | 0 of 100 | 0 in 150 | 0.550 / 0.587 / 0.614 | not off zero |
| `tuned-rlpd-can-tau-bc` | `uyputb0l` | 125 min | 0 / 0 / 0 / 0 | 0 of 100 | 0 in 150 | 0.577 / 0.597 / 0.593 | not off zero |
| control: `r1hvijqh` | | | 0 / 0 / 0 / 0 | 0 of 100 | 0 in 150 | 0.926 / 0.856 / 0.875 | |

Every training episode was a 200-step timeout. Queue time includes each arm's 17-min gate; `bc`
had GPU 1 to itself, the other two shared GPU 0.

**The diagnostics,** at 10k / 20k / 30k:

| | `tau` | `bc` | `tau-bc` | control |
|---|---|---|---|---|
| `debug/q_data_demo` | 0.809 / 0.806 / 0.889 | 0.670 / 0.654 / 0.666 | 0.666 / 0.649 / 0.660 | 0.840 / 0.829 / 0.876 |
| `debug/q_policy_demo` | 0.832 / 0.830 / 0.917 | 0.685 / 0.669 / 0.683 | 0.683 / 0.665 / 0.678 | 0.866 / 0.852 / 0.902 |
| `debug/q_data_online` | 0.010 / 0.000 / 0.001 | 0.004 / 0.002 / 0.001 | 0.140 / 0.013 / 0.000 | −0.001 / 0.000 / 0.010 |
| `debug/policy_data_gap_demo` | 0.84 / 0.82 / 0.84 | 0.15 / 0.14 / 0.15 | 0.16 / 0.15 / 0.15 | 0.90 / 0.78 / 0.78 |
| `debug/dQ_da_mean_abs` | 6.4e-6 / 6.2e-6 / 5.9e-6 | 2.6e-5 / 2.5e-5 / 2.5e-5 | 3.1e-5 / 2.6e-5 / 2.5e-5 | 6.1e-6 / 6.7e-6 / 6.8e-6 |
| `rft/bc_loss` | — | 0.34 / 0.29 / 0.31 | 0.36 / 0.30 / 0.32 | — |
| δ from 5k (positive at) | +0.023 (251/251) | +0.015 (251/251) | +0.015 (251/251) | +0.024 (251/251) |

**Read against the pre-registration:**
- **`tau`: as predicted.** Not off zero. The demo values settled at 0.81–0.89 (predicted
  0.75–0.9), with δ positive at every log step. No CORRECTION to the fixed-point reading.
- **`bc` and `tau-bc`: "a BC arm not off zero, with its gap under 0.3."** The gap was 0.14–0.16
  from 10k, as predicted. The pre-registered meaning: imitating the demos does not succeed on Can
  with this actor, and the critic still cannot rank actions.
- **Wrong, in part:** the prediction that δ would fall to about 0 and the demo values toward
  0.398. δ fell by a third, to +0.015, not to 0, and the demo values to about 0.65, not 0.4.
- **The fixed point still accounts for them.** The screen's formula,
  0.398 + 0.602 · γ³δ / (1 − γ³), predicts 0.86 (`tau`), 0.70 (`bc`) and 0.70 (`tau-bc`) from
  each arm's δ, against mean demo values from 5k of 0.81, 0.65 and 0.64. About 0.05 high, as on
  every screen arm.
- One seed per arm: observations, not comparisons (STANDARDS 5.3).

**What the BC arms add.** Readings, single-seed:
- **The term reached the actor and held it near the demos' actions,** at 0.15 per axis on demo
  states against the control's 0.78–0.90, and its actions at 0.6 in size, not the bound.
- **But the imitation is coarse.** `rft/bc_loss` of about 0.3 sums over 7 axes: a squared error
  of about 0.045 per axis, so about 0.21 RMS, against demo actions that average 0.32 in size.
- **A likely reason, untested:** the BC loss does not train the encoder
  (`bc_backprop_encoder` false, upstream's default). The encoder learns only from the critic,
  whose slope at the policy's action is nearly flat (2.5e-5, under STANDARDS 6.3's 1e-4). So the
  actor imitates from features that were never trained to tell the demos' states apart.
- **The online half still carries no signal.** It is valued at about 0, correctly: no rollout
  succeeded. `tau-bc`'s online values reached 0.14–0.18 at 5–10k and fell back to 0 by 30k, with
  no success behind them.
- **The next candidates the pre-registration named:** a longer n-step, the BC loss training the
  encoder (`agent.bc_backprop_encoder=true`), a larger coefficient.
- Packages in `outputs/runs/robomimic-can-final/{ns5mis29,kc8z7vsf,uyputb0l}/`; each manifest
  records `ed47957`, clean.

---

## 2026-10-07 — PRE-REGISTERED — upstream's code, out of the box: a fresh clone in its README's environment, on Can

**Question** (Aden, 2026-10-07): does upstream's code, cloned fresh and installed as its README
says, learn on Can with its released RLPD script?

**Why this is not `rdoj0ex8` again.** `rdoj0ex8` ran upstream's trainer and config (2026-10-06
RESULTS, the screen), but inside our fork and our conda env, and scored 0.
- **Our fork's code on that path does upstream's math.** A diff of `resfit/` against upstream at
  `66262b06` finds only:
  - logging (`dQ_da`, `critic_diagnostics`), and timing code;
  - `CriticConfig.drop` renamed `dropout`, read nowhere;
  - `num_q`'s default at 2, which `rdoj0ex8` set back to 10;
  - the eval video's upload moved;
  - a patch to torchrl's `PrioritizedSampler` serialization, unused under uniform sampling.
- **The environment was never tested.** Ours against upstream's README install, as built:

  | | torchrl | tensordict | mujoco | torch | numpy |
  |---|---|---|---|---|---|
  | ours (`residual`) | 0.7.0 | 0.7.0 | 3.8.1 | 2.6.0 | 1.26.4 |
  | upstream's (`resfit-upstream`) | 0.9.2 | 0.9.1 | 3.3.2 | 2.14.1 | 1.26.4 |

  torchrl builds the replay buffers and the n-step returns; mujoco is the physics. robosuite
  (1.5.1, commit `77a4751`), lerobot (`69901b9`), dexmimicgen and mimicgen are at upstream's
  pinned commits in both.

**The setup.**
- **Code:** `~/projects/resfit-upstream`, a clone of `github.com/amazon-far/residual-offpolicy-rl`
  at `66262b06` (2025-11-17, its latest), unmodified.
- **Environment:** a new conda env, `resfit-upstream`, built by its README's steps in order.
  `pip freeze` in `~/projects/resfit-upstream/pip_freeze_resfit-upstream.txt`.
  - **Deviations:** the env's name; and its setup script's `micromamba install -n residual ffmpeg`
    line, which fails harmlessly here (no micromamba) and was done with conda, for this env.
  - robosuite warns that `robosuite_models` is not installed; Can's Panda is built in.
- **Shared with every run so far:** the dataset (`ankile/robomimic-mh-can-image`, from the same
  local Hugging Face cache) and wandb's login.
- **No uploads:** `HF_OFFLINE_BUFFER_REPO` and `HF_ONLINE_BUFFER_REPO` unset, so its trainer writes
  its buffer caches locally only; wandb offline, since it uploads checkpoints (STANDARDS 6.4).
  Caches under `CACHE_DIR=~/projects/resfit-upstream/local_cache`, fresh.

**Seed-group `upstream-clean-1007`,** seed 1 per arm, 30k steps, Can. Project `robomimic-can-final`,
group `upstream-clean`, names `<arm>-1`, notes `upstream-clean-1007/<arm>/seed1`.

| Arm | Command, in the clone, env `resfit-upstream` | Why |
|---|---|---|
| `clean-rlpd-can` | `2_can_rlpd.sh`'s: `train_rlpd_dexmg --config-name=rlpd_can_config algo.prefetch_batches=4`, plus `algo.total_timesteps=30000 seed=1 wandb.mode=offline` and the naming | the question |
| `clean-residual-can` | `train_residual_td3 --config-name=residual_td3_can_config algo.prefetch_batches=4 base_policy.wandb_id=robomimic-can-bc/xhjdl8a7`, plus the same | the control: does the fresh env work at all |

- **Departures from the script, all deliberate:** 30k steps, not 300k (Fig. 5's Tuned RLPD is at
  about 0.7 on Can by 25k); seed 1 (the script sets none; `rdoj0ex8` ran seed 1); wandb offline and
  named. The residual arm's base policy is our Can BC policy, where the README says to put one.
- **The comparisons:** `clean-rlpd-can` against `rdoj0ex8`, the same command in our env;
  `clean-residual-can` against `msfkjwab`, our residual on Can (0.92 → 1.00 in our env).
- **Order:** `clean-rlpd-can` first. Each of upstream's trainers writes about 21 GB of caches (a
  16 GB online buffer, at full capacity), so the residual arm runs only when the disk has room
  for it.

**The verdict:**
- **`clean-rlpd-can` is off zero** iff its 20k and 30k evals score at least 5 successes between
  them, of 100: the screen's rule.
- **The fresh env is valid** iff `clean-residual-can`'s step-0 eval, the BC policy with a zero
  residual, scores at least 0.80 (B_can = 0.894, 50 episodes) and the run reaches 30k. Its later
  evals are reported, deciding nothing.

**What the outcomes mean,** decided now:
- **Off zero, env valid:** the environment is what differs. Next: find which package, torchrl
  first, in our env.
- **Not off zero, env valid:** upstream's released RLPD does not reproduce Fig. 5 on Can even as
  shipped. What is missing is in config or procedure the release does not include: a question for
  the paper's author. The BC-encoder arm and the others named on 2026-10-07 stay the next step.
- **Env not valid** (step-0 under 0.80, or a crash): `clean-rlpd-can` cannot be read until the
  fresh env is understood.
- One seed per arm: observations, not comparisons (STANDARDS 5.3).

**Resources.** One GPU at a time. About 15 GB of RAM. Disk: 17 GB free after the env, which is
too little for 21 GB of caches. It launches once `local_runs/upstream-rlpd/` (21 GB, no longer
read by anything) is deleted.

---

## 2026-10-07 — PRE-REGISTERED — tuned RLPD on Can: the BC term trains the encoder

**Question** (Aden, 2026-10-07): does the BC term get tuned RLPD off zero when its loss also trains
the image encoder? In the τ/BC arms (2026-10-07 RESULTS) the term held the actor near the demos'
actions, but coarsely: about 0.21 RMS per axis, against demo actions about 0.32 in size. A likely
reason is that the BC loss saw detached features, and the encoder learned only from a critic with
almost no action signal. Aden's reading is that no rollout ever succeeds, so nothing is learned;
this tests whether better imitation produces successful rollouts.

**Seed-group `tuned-rlpd-can-bc-encoder-1007`,** seed 1, 30k steps, Can, from the commit that adds
this entry. Project `robomimic-can-final`, group `tuned-rlpd`, name `tuned-rlpd-can-bc-encoder-1`,
notes `tuned-rlpd-can-bc-encoder-1007/tuned-rlpd-can-bc-encoder/seed1`.

| Arm | Change from `tuned-rlpd-can-bc` (`kc8z7vsf`) |
|---|---|
| `tuned-rlpd-can-bc-encoder` | `agent.bc_backprop_encoder=true` |

**What the flag does,** in upstream's `QAgent.update_actor_rft`, unchanged:
- On each actor update, the BC loss's gradient also reaches both cameras' ViT encoders, and the
  encoder's optimizer takes a second step. It is the critic's: AdamW, LR 1e-4. The BC gradient is
  clipped at norm 1.0.
- **The actor's Q term stays detached** (`QAgent.update` detaches the features first). So the
  encoder learns from the critic's TD loss on every update and from the BC loss on every fourth,
  never from the actor's Q loss.
- `tests/test_bc_term.py` checks that the BC step moves the encoder iff the flag is set.

**The reference,** `kc8z7vsf`, is not run again: the new commit changes tests, docs and
`submit.sh` only. **Held fixed:** everything else as `kc8z7vsf` ran: `bc_loss_coef` 0.025, τ 0.01,
10 critic heads, LRs 1e-4, 4 critic updates and 1 actor update per step, n-step 3, batch 256 half
demos, the 300 MH demos, 10k U(±1) warm-up steps, no critic warm-up. Evals of 50 episodes at 0,
10k, 20k and 30k.

**The verdict,** by the screen's rule:
- **Off zero** iff its 20k and 30k evals score at least 5 successes between them, of 100.
- **What runs next, if it is off zero, is Aden's decision:** a BC term is not the paper's Tuned
  RLPD.
- **Reported, deciding nothing:** every eval; training successes, and the step of the first; at
  10k, 20k and 30k, `rft/bc_loss`, `debug/policy_data_gap_demo`, `debug/q_{data,policy}_{demo,online}`,
  `debug/dQ_da_mean_abs`, `train/critic_loss` and `train/critic_qt`.

**Predictions, decided now:**
- **Better imitation:** `rft/bc_loss` under 0.2 by 10k (the reference's 0.29–0.36), and
  `debug/policy_data_gap_demo` under 0.12 (its 0.14–0.16).
- **Whether rollouts then succeed is the open question.** If some do, they show up in the
  training episodes first. The online half's values then rise above 0, and `dQ_da` above the
  reference's 2.5e-5: the action contrast the critic has lacked.
- **The risk, interference in the shared encoder.** The BC loss (about 0.3) is many times the
  critic's (under 1e-4), and both step one AdamW. So BC may set the features, and the critic fit
  on top of them. It is visible if `train/critic_loss` rises well above the reference's, or
  `train/critic_qt` leaves [0, 1], the range of the true returns.

**What the outcomes mean,** decided now:
- **Off zero:** the encoder's imitation quality was the bottleneck; with it, demos seed
  exploration. "Tuned RLPD + BC, encoder trained" becomes a candidate baseline, for Aden to judge.
- **Not off zero, imitation better** (`rft/bc_loss` under 0.2): imitating the demos well still
  does not succeed on Can with a single-step actor, at least by 30k. Next candidates: BC
  pretraining of actor and encoder before the online phase, a longer n-step.
- **Not off zero, imitation no better:** the encoder was not the limit. Next: a larger
  coefficient.
- One seed: an observation, not a comparison (STANDARDS 5.3).

**Resources.** One run, on runner `gpu1`, about 1.5 h. No buffer caches are written. GPU 0 stays
free for `upstream-clean-1007`.

---

## 2026-10-07 — CODE — training can start with no warm-up (`algo.learning_starts` below one batch)

`train_residual_td3.py` could not run without a buffer warm-up. At `algo.learning_starts=0` it
skipped the random steps, then sampled the empty online buffer at step 1 and crashed. The buffer
stays empty for the first n-step window: the n-step transform releases a transition once n = 3
steps follow it.
- **Updates now wait until the online buffer holds one online batch** (128 transitions). Until
  then the actor collects, untrained, with its exploration noise, and nothing learns. The
  logging step waits too: it reads the last update's metrics and batch.
- **A critic warm-up now refuses to start on a buffer holding less than one batch.**
- **Every earlier run is unaffected:** each started its main loop with 10k transitions in the
  online buffer, so the gate was open from its first update. `tests/test_regression_agent.py`
  pins the agent, which this does not touch.
- **Not unit-tested,** being the trainer's loop. A smoke run checked it: the trainer at
  `algo.learning_starts=0`, 400 steps, debug mode, from this commit's working tree. It took no
  random steps, logged nothing at step 100 (no update yet), logged from step 200, and finished
  without error. Its `action_l1` was 0.996 at step 200, about 70 actor updates in.

---

## 2026-10-07 — PRE-REGISTERED — tuned RLPD on Can with no warm-up: the demos in the buffer, no random steps

**Question** (Aden, 2026-10-07): with the demos already in every batch, does tuned RLPD need the
10k-step random warm-up, or does it get in the way?

**Why it might matter.** Sampling within the online buffer is uniform. So the warm-up's 10k random
transitions, all at reward 0 and far from the demos' states, make up much of the online half for a
long time: 10k / (10k + t) of it at step t, so two thirds at 5k, half at 10k, a quarter at 30k.
Without them, the online half is the actor's own rollouts from the start.

**Seed-group `tuned-rlpd-can-no-warmup-1007`,** seed 1, 30k steps, Can, from the commit that adds
this entry (and the CODE entry above). Project `robomimic-can-final`, group `tuned-rlpd`, name
`tuned-rlpd-can-no-warmup-1`, notes `tuned-rlpd-can-no-warmup-1007/tuned-rlpd-can-no-warmup/seed1`.

| Arm | Change from the screen's control (`r1hvijqh`) |
|---|---|
| `tuned-rlpd-can-no-warmup` | `algo.learning_starts=0` |

- **No critic warm-up,** as the control (`algo.critic_warmup_steps` 0).
- **The first steps:** the untrained actor collects with σ 0.1 noise, and learning starts once
  the online buffer holds 128 transitions, at about step 130. From there, 4 critic updates and 1
  actor update per step, as the control.
- **What this changes besides the data:** the step counter starts at the first env step, not
  after 10k random ones. So at 30k this arm has taken 30k env steps, the control 40k, and both
  about 120k critic updates. The comparison removes the random data and keeps the updates.
- **Held fixed:** everything else in `tuned_rlpd_can_config`, as `r1hvijqh` ran it. Evals of 50
  episodes at 0, 10k, 20k and 30k.

**The verdict,** by the screen's rule: **off zero** iff its 20k and 30k evals score at least 5
successes between them, of 100. **Reported, deciding nothing:** every eval; training successes;
`train/action_l1_magnitude` at 1k, 5k and 10k; and at 10k, 20k and 30k,
`debug/q_{data,policy}_{demo,online}`, `debug/policy_data_gap_{demo,online}` and
`debug/dQ_da_mean_abs`.

**Prediction: not off zero.** The screen's mechanism lives in the demo half, and nothing here
changes it.
- The actor reaches its bound within a few hundred updates, as the control's did.
- The demo values settle near 0.8 again, with δ positive at most log steps.
- **What does change is the online half's data.** Its actions are now the actor's own plus
  noise, not uniform ones. So `debug/policy_data_gap_online` should be near the noise's size,
  about 0.08 per axis, far below the control's. The online half is still valued at about 0: the
  saturated actor's rollouts still never succeed.

**What the outcomes mean,** decided now:
- **Off zero:** the random warm-up data were what held tuned RLPD at zero. That contradicts the
  screen's reading that the online half carries no signal either way, which gets a CORRECTION.
- **Not off zero:** the warm-up is not the obstacle; the online half has no successes with or
  without it. Removing it stays a candidate to pair with whatever makes the actor imitate the
  demos (the BC-encoder run, BC pretraining), since then the actor's own early rollouts are the
  useful data.
- One seed: an observation, not a comparison (STANDARDS 5.3).

**Resources.** One run on runner `gpu0`, about 1.5 h. No buffer caches. GPU 0 is otherwise free
until `upstream-clean-1007` has disk; the two can share it.

---

## 2026-10-07 — LAUNCHED — the BC-encoder and no-warm-up arms on Can, and a full disk between them

| Arm | Runner | Commit | wandb | Picked up / training from (EDT) |
|---|---|---|---|---|
| `tuned-rlpd-can-bc-encoder` | `gpu1` | `05fd8d0` | **`44enq26h`** | 07:51 / 08:07 |
| `tuned-rlpd-can-no-warmup` | `gpu0` | `f581d9c` | **`hi8t2hps`** | 08:24 / 08:41 |

- Both in worktrees pinned to their commits (`~/projects/equi-resrl-<commit>`), with
  `CACHE_DIR=~/projects/equi-resrl`, through `tools/gpu_queue.sh`. Gates: GO for both.
- **The disk filled at about 08:23 and emptied again by 08:25:45,** with nothing of ours writing in
  bulk. It had been near full for days, with another user's home holding 945 GB. Upstream's env
  build for `upstream-clean-1007` (07:25–07:45) took about 13 GB of the 29 GB that were free. A
  system snap update wrote 555 MB at 08:23. Free space was back at 16 GB by 08:25:45, with nothing
  of ours deleted.
- **What it did to `44enq26h`:** the run kept training. Its wandb history kept reaching wandb's
  servers (every upload answered 200 OK). But its local wandb record stopped at 08:23:57, about
  step 7.3k, and its writer has failed on every record since. **Read this run's history from
  wandb's servers, not from its local record.** Its console log may have lost lines during the
  window.
- **What it did to `hi8t2hps`'s launch:** the runner could not delete the arm's line from its
  queue file, a write, and it relaunches whatever line is first. So it retried the launch about
  once a second, 44 times, from 08:23:45 to 08:24:30. Each attempt died at once; the 44th, with
  space back, went through. The line was still in the queue afterwards, and would have launched
  a duplicate run when this one ended; it was removed by hand at 08:48. The arm's console log is
  empty, since `tee` dropped the file during the window: its console output is in the `gpu0`
  tmux pane and in wandb. Its local record is healthy.
- **A bug in `tools/gpu_queue.sh`,** found here: it launches even when deleting the line fails
  (TODO).

---

## 2026-10-07 — LAUNCHED — `upstream-clean-1007`'s `clean-rlpd-can`, after a repair to the env

| Arm | GPU | Code | wandb (offline) | Launched (EDT) |
|---|---|---|---|---|
| `clean-rlpd-can` | 1, shared with `44enq26h` | upstream `66262b06`, clean | **`zwmk393l`** | 09:07 |

- Through `~/projects/resfit-upstream-runs/launch.sh rlpd 1`, in tmux session `upstream-clean`.
  Console log `outputs/queue/20261007-090724_clean-rlpd-can-1_gpu1.log`. Its local wandb record is
  `~/projects/resfit-upstream/wandb/offline-run-20261007_090736-zwmk393l`.
- **The env as its README builds it could not run.** The README pins no torch, and lerobot asks
  only for `torch>=2.2.1`, so pip installed today's torch 2.14.1. Against it, the torchcodec 0.4.0
  that came with it fails to load, and so does torchrl 0.9.2's C++ extension (both undefined
  symbols). The first launch, at 09:04 (wandb offline `9atavlcs`), died in the first minute,
  decoding the demos' video while building the offline buffer; nothing trained.
- **The repair, the one change to the env:** torch 2.7.1+cu128 and torchvision 0.22.1+cu128, from
  the CUDA 12.8 index. The README's own fix-up step points at that index, and torchcodec 0.4 and
  torchrl 0.9 target torch 2.7. With them, torchcodec, torchrl's extension and CUDA all load.
  The freeze after it is `~/projects/resfit-upstream/pip_freeze_resfit-upstream-torch271.txt`;
  against the first freeze, only torch, torchvision, triton and the CUDA 12 libraries differ.
- Disk at launch: 38 GB free, after `local_runs/upstream-rlpd/` was deleted.

---

## 2026-10-07 — FINDING — our env renders Can's front camera differently from the demos

**What was measured.** The first frame of 20 demos, spread across the 300, and 20 env reset frames,
84×84, with a scratch script outside the repo. Each frame went through the same path as in training: demos
through `LeRobotDataset`, the env through `create_vectorized_env`. The number is the mean absolute
pixel difference, in [0, 1], between the average frame of each set.

| Camera | Demo half vs demo half | Demo vs our env (mujoco 3.8.1) | Demo vs upstream's env (mujoco 3.3.2) |
|---|---|---|---|
| `agentview` (front) | 0.0067 | **0.0995** | 0.0103 |
| `robot0_eye_in_hand` (wrist) | 0.0434 | 0.0474 | 0.0371 |

- **The difference is the floor.** In the demos and in upstream's env the front camera sees a
  light grey tiled floor; ours renders it as darker, brownish planks. Rendering the same three
  seeded reset scenes in both envs (identical `qpos`), every pixel off the floor is identical,
  and so is every pixel of the wrist camera. The floor alone lowers the front camera's channel
  means to 0.49/0.44/0.39, against the demos' 0.57/0.53/0.49. `observation.state` at reset is
  within one demo standard deviation on every axis.
- Figures, outside git: `outputs/diagnostics/can_render_gap_84.png` (what the agent sees: demo,
  upstream's env, ours, and the difference) and `can_render_gap_512.png` (the front camera at
  512×512). Scripts `render_dump.py` and `render_compose.py` beside them.
- **The cause is mujoco's version.** Both envs load the same robosuite, commit `77a47512`, with
  identical arena and texture files; the same scene renders differently. Our env has had mujoco
  3.8.1 since 2026-05-21, so **every run of this project, residual and equivariant included, saw
  front-camera frames online that differ from its demos'.**

**What it would explain** (a reading, not yet tested): tuned RLPD's critic can tell every demo
frame from every online frame by the floor alone. "Light floor, high value; dark floor, zero"
fits all its data without the action, since no online episode has succeeded. That matches what
the runs show:
- Q on online transitions settles at exactly 0 within about 2k updates, and Q on demo
  transitions at 0.65–0.85 (`r1hvijqh`, `kc8z7vsf`, `hi8t2hps`).
- The critic is blind to the action (dQ/da about 6e-6), so the actor drifts to its Tanh bound.
- **On every eval rollout Q is about 0 from the first frame** (`r1hvijqh` at 10k, −0.002;
  `kc8z7vsf` at 30k, 0.002), though a rollout's first frame is the same scene as a demo's.
- The BC term fits the actor to the demos' frames, and the env shows it another floor.

**What tests it:** `clean-rlpd-can` (above) runs in an env whose frames match the demos, but it
also differs from ours in torch, torchrl and tensordict. Changing only mujoco in a copy of our env
would separate the two.

---

## 2026-10-07 — RESULTS — the BC-encoder and no-warm-up arms on Can: neither off zero

| Arm | wandb | Evals at 0 / 10k / 20k / 30k (of 50) | Training episodes with a success | Verdict |
|---|---|---|---|---|
| `tuned-rlpd-can-bc-encoder` | `44enq26h` | 0 / 0 / 0 / 0 | 0 of 150 | **not off zero** |
| `tuned-rlpd-can-no-warmup` | `hi8t2hps` | 0 / 0 / 0 / 0 | 0 of 150 | **not off zero** |

Both finished 30k steps (exit 0): `44enq26h` at 09:13 after 82 min, `hi8t2hps` at 09:41 after
76 min. Histories read from wandb's servers (`44enq26h`'s local record is broken; see LAUNCHED).
`hi8t2hps`'s EXIT line is in the `gpu0` tmux pane but not in `outputs/queue/gpu0.log`, which has
had no line since 08:23:45, during the full disk.

**`44enq26h`, reported, at 1k / 5k / 10k / 20k / 30k:**
- `rft/bc_loss` 0.40 / 0.29 / 0.34 / 0.32 / 0.34; `debug/policy_data_gap_demo` 0.14 / 0.12 /
  0.14 / 0.16 / 0.15. **The predicted better imitation did not happen:** neither reached its
  threshold (0.2 and 0.12) after 5k, so the matching outcome is "not off zero, imitation no
  better".
- Q on demos (data / policy): 0.34 / 0.35 rising to 0.67 / 0.69. On online transitions: 0.08–0.18
  through 5k, then 0 by 20k.
- `debug/dQ_da_mean_abs` 2.2–2.6e-5, the reference's 2.5e-5. `train/critic_loss` 4e-4 falling to
  3e-5, and `train/critic_qt` 0.21–0.34, inside [0, 1]: no sign of the encoder interference the
  entry named as its risk.
- `train/action_l1_magnitude` 0.59–0.62: the BC term held the actor off its bound throughout.

**`hi8t2hps`, reported:**
- `train/action_l1_magnitude` 0.84 / 0.92 / 0.88 at 1k / 5k / 10k: near the bound, as predicted.
- Q on demos (data / policy) 0.74 / 0.77 at 1k, 0.85 / 0.88 at 30k. On online transitions 0.14
  at 1k, then about 0 from 5k on.
- `debug/dQ_da_mean_abs` 1.1e-5 at 1k, then 5.7–6.1e-6.
- **One prediction failed:** `debug/policy_data_gap_online` was 0.82–1.10, not about 0.08. The
  online half's actions came from earlier versions of an actor that moved between bounds, so the
  current actor's actions are far from them, noise or not.

**What the outcomes mean,** by the rules decided before launch:
- **BC-encoder:** "the encoder was not the limit", with a larger coefficient next.
- **No warm-up:** "the warm-up is not the obstacle", and removing it stays a candidate to pair
  with whatever makes the actor imitate the demos.
- **Both next steps are on hold.** Both arms ran in our env, which the FINDING above shows
  renders the front camera's floor differently from the demos. The BC term fit the actor to
  frames the env never shows, and the critic can tell demo from online transitions without the
  action. Until `clean-rlpd-can` and a mujoco-only change in our env say whether that gap is what
  holds tuned RLPD at zero, neither result says anything about the encoder or the warm-up in an
  env that matches the demos.
- One seed each: observations, not comparisons (STANDARDS 5.3).

---

## 2026-10-07 — RESULTS — `upstream-clean-1007`: upstream's tuned RLPD, in upstream's env, is off zero

| Arm | wandb (offline) | Evals at 0 / 10k / 20k / 29,999 (of 50) | Verdict |
|---|---|---|---|
| `clean-rlpd-can` | `zwmk393l` | 0 / 0 / **44 / 40** | **off zero:** 84 successes of 100 at 20k and 30k, against a bar of 5 |
| `clean-residual-can` | not run | — | env validity untested |

- **The run:** finished at 10:14, after 62 min of training (exit clean). Successful eval
  episodes took 75 steps on average at 20k and 68 at 30k. The paper's Fig. 5 has Tuned RLPD at
  about 0.7 on Can by 25k, so this is the paper's curve.
- **Against `rdoj0ex8`,** the same trainer, config and seed in our env: 0 at every eval. Only the
  env differs (mujoco 3.3.2, torch 2.7.1, torchrl 0.9.2, tensordict 0.9.1 here, against 3.8.1,
  2.6.0, 0.7.0 and 0.7.0), and only mujoco changes what the agent sees (the FINDING above).
- **How the two diverged** (`outputs/diagnostics/can_env_compare.png`, with `r1hvijqh` too):
  - from the first thousand updates, the critic's loss sat at 1–2e-4 here and 5e-6 in ours;
  - the actor's gradient norm was 0.03–0.1 here, 0.003 in ours;
  - the actor's mean |action| fell from 0.96 to 0.60 by 15k here, and stayed at 0.80–0.90 in ours.
- **What the rollouts show:**
  - At 10k, still 0/50, the arm reached the can, knocked it over and went on to the bins: the
    demos' sequence without the grasp. In our env the arm drives to one pose and holds it.
  - Its eval Q started each episode at 0.47–0.55. Our env's runs start it at about 0.
  - Videos: `outputs/diagnostics/videos/zwmk393l_upstream_env_step*.webm`.

**What the outcome means,** by the rule decided before launch:
- **"Off zero, env valid: the environment is what differs."** The env-valid half rests on
  `clean-residual-can`, which has not run: it needs about 21 GB of disk, and 14 GB are free while
  this run's caches stay. An env in which tuned RLPD reaches 0.88 is plainly working, so the
  control's purpose, to rule out a broken env, is met another way. Whether to run it anyway is
  Aden's decision.
- **The next step departs from the one decided.** The rule named torchrl first. The FINDING,
  measured since, shows mujoco changes what the agent sees, so mujoco goes first: our env and
  trainer, with mujoco alone at 3.3.2.
- **What it says about earlier work:** every tuned-RLPD result in our env (the screen, τ/BC,
  BC-encoder, no-warm-up) was run where demo and online frames differ. None of them says how
  those changes behave in an env that matches the demos. The residual and equivariant runs saw
  the same mismatch; what it did to them is open.
- One seed: an observation, not a comparison (STANDARDS 5.3). Here the gap is 0 against 0.88.

---

## 2026-10-07 — CODE — records in main: the worktrees' 34 records moved, and the queue runner fixed

TODO B15, steps 1 and 2. Aden's aim: every experiment in the main checkout, and fewer worktrees.

**The records.**
- **Moved:** 34 wandb records into `wandb/`, and their 34 Hydra folders into
  `outputs/<date>/<time>/`. 32 records came from 14 of the 15 pinned worktrees
  (`~/projects/equi-resrl-<commit>`; `8407c27` had none), 2 from upstream's clone
  (`~/projects/resfit-upstream`). 68 folders, 822 files, 1.29 GB, renamed within one filesystem,
  with no name collisions. A sha256 manifest of every file before the move matches every file
  after it. Manifest and server check in `outputs/worktree-consolidation-20261007/` (outside git).
- **31 of the 34 runs are on wandb's servers.** The other three are offline records that never
  reached them, so they are also copied, and checked, to `~/equi-resrl-preserve/wandb-offline-only/`:
  `rdoj0ex8` (upstream's trainer in our env), `9atavlcs` (the first `clean-rlpd-can` launch, which
  died at startup) and `zwmk393l` (upstream's trainer in its env).
- **Nothing else in the worktrees is a record:** bytecode, empty `tea_debug.log` files, an
  `artifacts` symlink to main's `artifacts/`, and one cached env probe.
- **New runs record into `CACHE_DIR`,** the main checkout, wherever their code runs from:
  - wandb's local record, by `wandb.init(dir=...)` in all three trainers;
  - BC checkpoints, which `download_policy_from_wandb` now downloads to
    `<CACHE_DIR>/artifacts/<name>:<version>`, wandb's own folder name;
  - Hydra's folder, by `hydra.run.dir` in `submit.sh`;
  - the run package and the queue's logs went there already.

  ARCHITECTURE.md, "Run packages". `tests/test_run_package.py` checks the first two. So a launch
  worktree now holds only code, and can be removed when its runs end.
- **Deleted, with Aden's approval:** upstream's buffer caches,
  `~/projects/resfit-upstream/local_cache/{offline,online}_buffer_cache/` (21 GB), which nothing
  reads. 35 GB free after.
- **The four queue runners were stopped** with STOP lines and their tmux sessions closed. All
  four were idle, two in `equi-resrl-b667647` and two in `equi-resrl-187d895`.

**The runner,** `tools/gpu_queue.sh`:
- **It launches a line only once the line is deleted** (sed's exit status, and a changed checksum),
  and only with at least 5 GB free (`QUEUE_MIN_FREE_GB`). Otherwise it logs why, leaves the line,
  and tries again a minute later. On the full disk it launched one arm 44 times.
- **It can run from main.** Its body is one function, called on the last line, so bash has read
  all of it before it runs, and an edit to the file cannot reach a running runner. `submit.sh` ends
  in `exec` for the same reason.
- **`log()` starts a fresh line** when the log ends without a newline.
- **The dirty check no longer excuses an untracked `artifacts`:** the symlink was needed only while
  BC downloads went to the working directory.
- **`tests/test_gpu_queue.py`,** in Tier 1, 5 tests, each running the real script on a scratch
  queue with a stub `submit.sh`. On the script before this commit 4 fail; the plain launch passes.

**CORRECTION** to 2026-10-07 RESULTS, "the BC-encoder and no-warm-up arms". It says
`outputs/queue/gpu0.log` "has had no line since 08:23:45". The append worked. The full disk cut
the 08:23:45 LAUNCH line off before its newline, and `hi8t2hps`'s EXIT line, at 09:41:54, was
appended to the end of it. So that EXIT line is in the log, on the cut-off line's row.

**No run's numbers change:** nothing here touches the agent, the trainer's computation or a
config. The gate's goldens pass.

---

## 2026-10-07 — CODE — mujoco 3.3.2, and the render on the record and in the gate

TODO B15 step 3. The `residual` env's mujoco 3.8.1 renders robosuite's floor differently from the
demos (2026-10-07 FINDING); 3.3.2 matches them.

**The env.** `pip install --no-deps mujoco==3.3.2`, and nothing else: `pip freeze` before and after
differs in exactly that line (both files in `outputs/env/`, outside git). `pip check` adds
dm-control 1.0.41's `mujoco>=3.8.1` to the requirements it already reported unmet (lerobot's
extras, dexmimicgen's numpy pin, lerobot's gymnasium pin); nothing imports dm-control. CLAUDE.md's
setup now has the line.

**The render, measured** with the new `tools/render_check.py`: the mean absolute difference, per
pixel and channel, between the average of 40 demo first frames and the average of 40 env reset
frames, front camera, through the paths training uses. On boce-WS-01, 10 repeats on 3.3.2 and 3 on
3.8.1 (env resets are unseeded, so the env side varies):

| Task | Between the even and the odd demos | Bar (2.5x) | mujoco 3.3.2 | mujoco 3.8.1 |
|---|---|---|---|---|
| Can | 0.0050 | 0.0125 | 0.0103 ± 0.0003 | 0.0977 (0.0973–0.0982) |
| Square | 0.0052 | 0.0130 | 0.0086 ± 0.0005 | 0.0166 ± 0.0007 |

- **The bar is 2.5x the within-demo gap, not "about 2x" as planned.** Even the matching render
  differs from the demos by 2.06x on Can, so a 2x bar fails a correct env at random. The leftover
  difference is an even colour offset, about 0.005 per channel, on static pixels too, and the same
  under both versions (frames in `outputs/diagnostics/render_frames_mujoco{332,381}.npz`); the
  demos are stored as video. At 2.5x the matching render passes by 7 standard deviations on Can and
  9 on Square, and 3.8.1 fails by 5 on Square, where the floor is a thin strip.
- **The check can fail:** it fails both tasks under 3.8.1. mujoco was put back at 3.8.1 for that
  measurement only, then at 3.3.2 again; `pip freeze` is back to the pinned state, line for line.
- **On the gate** (`preflight.py`, Tier 1): any mujoco but 3.3.2 fails it, and the render check
  runs on Can and Square whatever the launch, in a subprocess (about 75 s), and fails it on either.
  `tests/test_render_check.py` (Tier 1) pins the arithmetic on synthetic frames.

**On each run's record:** `versions` in its wandb config (all three trainers), its run manifest,
and `eval_base_policy`'s JSON: mujoco, robosuite, torch, torchrl, tensordict and escnn, from
package metadata. Not a config field; nothing reads it.

**The online cache key records mujoco,** in the trainer and in `preflight.cache_hashes`, read from
package metadata: importing mujoco there would come before `MUJOCO_GL` is set. Every online key
moved; no offline key did. Old to new: residual Can `43637c10` → `48638eb8`, Square `66c46c51` →
`1683475f`, TwoArmBoxCleanup `5709219c` → `dca5e179`; tuned RLPD Can `3718c816` → `9a511563`
(ResFiT) and `163d26c0` → `f3250e68` (equivariant), Square `6e94049f` → `e7739da8` and `8e67e7b9` →
`099fa3d6`. `tests/test_config.py` pins the new ones. The smoke run below printed the key
preflight's mirror computes at its settings. Square's old online cache `66c46c51`
(16 GB, frames from 3.8.1, unreachable now) was deleted, with Aden's approval: 51 GB free after.

**New field `read_buffer_caches`,** default true, which is every earlier run's behaviour. False
fills both buffers in the run and loads no cache, from disk or the Hub. It is for the baselines'
"no cache reads": a key records neither a residual buffer's normalization path (TODO B7, S9) nor,
before this commit, mujoco. The trainer refuses it with `write_buffer_caches=true`, which would
write over a cache it did not read. The gate's Tier 3 now reads both flags from the composed config.

**The smoke runs,** on this commit's code before it was committed (their manifests list it as
uncommitted changes on `4107466`), recorded offline in wandb project `equi-resrl-smoke`, launched
from an empty scratch directory with `CACHE_DIR` at main. `u0itew7i` started before the trainer's
diagnostics call stopped depending on the agent (the next entry), which changes nothing for the
ResFiT agent. Both set `read_buffer_caches=false
write_buffer_caches=false`, 300 learning starts and no eval:

| Run | What | Steps | Offline buffer, filled in the run | Speed |
|---|---|---|---|---|
| `u0itew7i` | ResFiT residual, Square (`residual_td3_square_config`) | 600, after the 10k-update critic warm-up (200 s) | 80,731 frames in 26.6 min: the base policy labels every frame | 0.115 s a step |
| `rhqmlon0` | equivariant tuned RLPD, Can, at the baselines' design | 800 | 62,756 frames in 3.1 min | 0.41 s a step (98 ms an update) |

- **Neither read a cache.** `u0itew7i` printed the online key `f1ec1c01`, the one
  `preflight.cache_hashes` computes at its settings.
- **Every record is in main:** the wandb records (`wandb/offline-run-20261007_121446-u0itew7i`,
  `wandb/offline-run-20261007_120927-rhqmlon0`), the Hydra folders (`outputs/2026-10-07/11-47-36`,
  `12-05-43`) and the run packages (`outputs/runs/equi-resrl-smoke/`). Each scratch directory got
  one empty `tea_debug.log`, which main's `.gitignore` already lists.
- **Both records hold `versions`,** manifest and wandb config: mujoco 3.3.2, robosuite 1.5.1, torch
  2.6.0+cu124, torchrl 0.7.0, tensordict 0.7.0, escnn 1.0.11.
- **At these speeds** 150k steps take about 5 h (ResFiT residual, Square) and 17 h (equivariant)
  before evals, with both GPUs and the CPU shared with the gate's tests and the other smoke run.

**The gate said GO on this tree,** at `--task Square --method residual --stack resfit` with the
ResFiT residual arm's overrides: Tier 1, 178 passed; Tier 2, 354 passed; the render check, Can
0.0099 against 0.0125 and Square 0.0090 against 0.0129; Tier 3, both caches "not read and not
written".

**No run's numbers change through the code:** the goldens pass. What changes is the env: from this
commit every run renders the scene the demos show.

---

## 2026-10-07 — CODE — the equivariant agent logs the critic's Q by buffer

TODO B15 step 5. The equivariant `QAgent` gets `critic_diagnostics`, and the trainer now calls it for
both agents at every log step: `debug/q_{data,policy}_{online,demo}` and
`debug/policy_data_gap_{online,demo}`, the diagnostics that measured tuned RLPD's failure
(2026-10-06 RESULTS, the screen).
- **It mirrors the ResFiT agent's,** with this agent's own choices where they differ: a residual
  policy's action is combined with the base action by `_combine_actions`, the disk clip its actor
  loss uses, not a per-axis clamp; and its actions are in this stack's units, which `IdentityScaler`
  leaves as the env's, so `policy_data_gap` is not on the ResFiT stack's normalized scale.
- **Found on the way: the equivariant actor's `std=0` path draws a random number.** It computes
  `torch.randn_like(mean) * 0`: the action is exact, but torch's RNG moves. A first version of the
  method called that path, and the new check in `tests/test_regression_agent.py` caught it: after
  one diagnosed update, every later update differed (268 tensors). The method now takes the
  actor's mean (`std=None`) and clips it as the `std=0` path does, which gives the same action bit
  for bit, with no draw.
- **The same draw is in the actor loss and in every eval's `act()`,** in every equivariant run so
  far. So an equivariant run's random stream depends on its eval schedule: two runs that differ
  only in when they evaluate train on different noise after the first eval. Left as it is, since
  changing it changes every equivariant run's updates; noted in ARCHITECTURE.md, "Known rough
  edges".
- **Tests.** `tests/test_regression_agent.py` (Tier 1, CPU): 8 updates with the trainer's log-step
  diagnostics after each are bit-identical to 8 without, at `target-fix` (the actor's batch norm)
  and `all-four` (field-norm critic, two critic layers, the squash). `tests/test_critic_diagnostics.py`
  (Tier 2): at the baselines' equivariant design, the method reports what it says in vanilla and
  residual mode, its policy action equals the one the actor loss scores, it skips an empty demo
  half, and it moves no parameter and draws no random number.
- **The scalar ablation's branch** (`equivariant=False`, residual-only) takes the actor's mean,
  as its eval does. Checked once by hand, on CPU: the six keys, and no random number drawn. No
  test covers it, and no planned run uses it.
- **In the trainer:** `rhqmlon0`, the equivariant smoke run in the entry above, printed
  `q_demo(data/policy)` and `q_online(data/policy)` at every log step, as the ResFiT agent's runs do.
- **No earlier run changes:** the ResFiT agent logged these already, and no equivariant run did.

---

## 2026-10-07 — PRE-REGISTERED — the fixed-env baselines: tuned RLPD on Can and residual RL on Square, each stack

**Question** (TODO B15, Aden's plan of 2026-10-07): with mujoco 3.3.2, whose render matches the
demos (CODE, "mujoco 3.3.2"), does tuned RLPD learn Can, and does residual RL lift the Square base
policy, in ResFiT's stack and in the equivariant one? Every earlier run of the project saw a floor
the demos do not show (2026-10-07 FINDING).

**Seed-group `fixed-env-baselines-1007`,** seeds 1 and 2 per arm, 150k steps, from the commit that
adds this entry, launched from one worktree pinned to it (`~/projects/equi-resrl-<commit>`). Every
record lands in main. Names `<arm>-<seed>`, notes `fixed-env-baselines-1007/<arm>/seed<seed>`.

| Arm | Task | Config | Overrides beyond the shared ones | wandb project / group |
|---|---|---|---|---|
| `resfit-tuned-rlpd-can` | Can | `tuned_rlpd_can_config` | none | `robomimic-can-final` / `tuned-rlpd` |
| `equi-tuned-rlpd-can` | Can | `equi_tuned_rlpd_can_config` | the design | `robomimic-can-final` / `tuned-rlpd` |
| `resfit-residual-square` | Square | `residual_td3_square_config` | `agent.critic.num_q=10`, base `v9` | `robomimic-square-final` / `residual-rl` |
| `equi-residual-square` | Square | `residual_equi_td3_square_config` | the design, `agent.actor.actor_last_layer_init_scale=1e-4`, base `v9` | `robomimic-square-final` / `equi-residual-rl` |
| `equi-residual-square-zero-init` | Square | `residual_equi_td3_square_config` | the design, `agent.actor.actor_last_layer_init_scale=0.0`, base `v9` | `robomimic-square-final` / `equi-residual-rl` |

- **Shared:** `algo.total_timesteps=150000 read_buffer_caches=false write_buffer_caches=false`
  (both buffers filled in the run, no cache read or written); evals of 50 episodes at 0 and every
  10k (16); 10 critic heads, n-step 3, γ 0.99, batch 256 with half demos, 4 updates per env step,
  a 200k buffer, one env.
- **The recipes,** by method, the same in both stacks (`tests/test_sweep_arms.py` pins them):
  tuned RLPD, 10k uniform-random warm-up steps, no critic warm-up, actor LR 1e-4, τ 0.01, σ 0.1,
  action scale 1.0, the actor's default init; residual, 10k warm-up steps of the base policy plus
  U(±0.2), 10k critic-warm-up updates, actor LR 1e-6, τ 0.005, σ 0.05, action scale 0.1.
- **The equivariant design** (Aden, 2026-10-07): `equivariance.residual_squash=true`,
  `agent.critic.num_q=10`, `equivariance.num_critic_layers=2`, `equivariance.critic_norm=field_norm`,
  `equivariance.actor_norm=none`. The actor as built: visual projection, trunk, one hidden layer,
  each 128 regular fields of C8.
- **The base policies, pinned:** `base_policy.wt_version=v9` for Square (`run_3hzs5bz1_best:v9`),
  and `v7` for Can's bar (`run_xhjdl8a7_best:v7`), what `latest` resolves to today.
- **wandb:** where each kind of run has gone before (Aden). Both tuned-RLPD arms share
  `tuned-rlpd`, so the equivariant one overrides its config's `equi-tuned-rlpd`, which has never
  held a run. ResFiT's Square residual overrides its config's project, which no run has used, for
  `robomimic-square-final` / `residual-rl`, where all five earlier ones are (`870ws2c2` among them).
- **Config caveats** (ARCHITECTURE.md, provenance table): `equivariance.num_actor_layers` is dead
  and not set; `agent.actor.dropout`, `agent.critic.dropout` and `agent.critic.orth` do nothing
  (TODO B13); the equivariant arms' configs carry ResFiT's network fields under `agent` (`enc_type`,
  `vit.*`, `actor.hidden_dim`, `critic.hidden_dim` and the like), which the equivariant agent does
  not read.

**What varies, and what does not.** Within a task the arms differ only in `equivariance`, and on
Square in the residual's last-layer init (`tests/test_sweep_arms.py`). That one switch changes the
whole stack: encoder, augmentation, normalization, critic and actor architecture, and the action
bound (ARCHITECTURE.md, "The two stacks side by side"). So each pair compares stacks, not
equivariance alone (STANDARDS 3.3). The two equivariant residual arms differ in the init only.
Across tasks the method differs too, so nothing is compared across them.

**The validation, first:** `resfit-tuned-rlpd-can` seed 1 launches alone on GPU 0. Its 20k and 30k
evals must score at least 5 successes of 100 between them (the screen's rule; upstream's code in
upstream's env, `zwmk393l`, scored 84). If they do, the other nine runs launch; if not, none does,
and the next step is Aden's. It separates the render from upstream's env's other differences
(torch 2.7.1, torchrl 0.9.2, tensordict 0.9.1): only mujoco changed here.

**The bars, measured while the validation runs,** on GPU 1 from the launch worktree, as before:
`eval_base_policy --num-evals 10 --seed 1000` at `residual_equi_td3_can_config` and
`residual_equi_td3_square_config`, with the pinned versions. 500 episodes each, recorded in
`outputs/base_rates/`. The old ones, B_can 0.894 and B_sq 0.508, were measured in the old render
and decide nothing here.

**Verdicts, per arm, both seeds pooled** (`tools/baseline_verdict.py`): p is the mean of the ten
evals at 110k–150k, five per seed, 500 episodes.
- **Tuned RLPD passes** iff p ≥ the new B_can.
- **A residual arm passes** iff p − B_sq ≥ 2·sqrt(p(1−p)/500 + B_sq(1−B_sq)/500).
- **Zero init against 1e-4, reported, deciding nothing:** each init's p, and whether a run
  collapsed by the rule of 2026-10-01 (three consecutive evals at or below 0.05). With two seeds a
  difference between them is an observation, not a comparison (STANDARDS 5.3); so is any claim
  that one stack beats the other.
- **Also reported:** every eval of every run; step 0 against the bar (the ResFiT and zero-init
  residual arms' step 0 is the base policy exactly, so their four step-0 evals are 200 more
  episodes on B_sq through the trainer's path); and at each 10k, `debug/q_{data,policy}_{demo,online}`,
  `debug/policy_data_gap_*`, `debug/dQ_da_mean_abs`, `train/critic_loss`, and
  `train/action_l1_magnitude` or `train/residual_l{1,2}_magnitude`.

**Predictions** (Claude's, approved by Aden on 2026-10-07):
0. Both bars come out at or above their old-render values (0.894, 0.508): the BC policies were
   trained on the demos' frames, which the env now matches.
1. The validation passes.
2. `resfit-tuned-rlpd-can`: off zero by 20k; p near B_can; even odds of passing.
3. `equi-tuned-rlpd-can`: off zero, below ResFiT's tuned RLPD; does not pass.
4. `resfit-residual-square`: passes.
5. `equi-residual-square`: does not collapse; does not pass. Low confidence.
6. Zero init against 1e-4: no difference beyond seed noise, and no run collapses because of its
   init. The 1e-4 residual starts at about 1.5e-5 in action units, far below anything that moves
   the base policy.

**Resources and order.** Two runs at a time, one per GPU, from runners `gpu0` and `gpu1`
(`tools/gpu_queue.sh`, queues in `~/queues/`); each launch runs the gate first (about 15 min).
About 6 h per ResFiT run and 19–20 h per equivariant run (0.115 and 0.41 s per step in the
2026-10-07 smoke runs, before evals); a Square residual run first fills its offline buffer, about
27 min. About 145 GPU-hours, so about three days from the validation's launch. RAM at most about
80 GiB (two Square residual runs), disk about 1.5 GB of records. On GPU 1, before its first
training run: the two bars, then `SMOKE=1` of `equi-residual-square`, for its speed and RAM
(the zero-init arm shares both). Then seed 1 of every arm before any seed 2, the equivariant runs
first on each GPU.

---

## 2026-10-07 — LAUNCHED and MEASUREMENT — the fixed-env baselines: the validation run, and the two bars

From the pre-registration above, at its commit `9cf207d`, in the one launch worktree
`~/projects/equi-resrl-9cf207d`. Every record is in main.

| Arm | Seed | Runner | wandb | Picked up / training from (EDT) |
|---|---|---|---|---|
| `resfit-tuned-rlpd-can` (the validation) | 1 | `gpu0` | **`d047it6c`** ([run](https://wandb.ai/aden-mckinney10-university-of-central-florida/robomimic-can-final/runs/d047it6c)) | 13:32 / about 13:56 |

- **Its gate said GO with no warning,** in the worktree: Tier 1, 188 passed; Tier 2, 385 passed;
  the render check passed on Can and Square. Before the commit, the gate had said GO at all four
  of the arms' task, method and stack combinations (Tier 2 at two of them, as it does not depend
  on the arm).
- **The other nine runs wait** for its 20k and 30k evals.

**The bars,** on GPU 1 from the worktree at `9cf207d` (clean), mujoco 3.3.2, `MUJOCO_GL=egl` as
in the trainer: `eval_base_policy --num-evals 10 --seed 1000`, base policies pinned.

| Bar | Config, base policy | Rates of the ten 50-episode evals | Mean ± binomial SE | Old render (3.8.1) | Change |
|---|---|---|---|---|---|
| **B_sq** | `residual_equi_td3_square_config`, `run_3hzs5bz1_best:v9` | 0.64 0.46 0.64 0.68 0.58 0.54 0.56 0.60 0.50 0.62 | **0.582 ± 0.022** | 0.508 ± 0.022 | +0.074, z = 2.35 |
| **B_can** | `residual_equi_td3_can_config`, `run_xhjdl8a7_best:v7` | 0.68 0.96 0.78 0.86 0.94 0.86 0.88 0.86 0.82 0.88 | **0.852 ± 0.016** | 0.894 ± 0.014 | −0.042, z = −2.00 |

- Records: `outputs/base_rates/Square__3hzs5bz1_best_v9__seed1000.json` and
  `Can__xhjdl8a7_best_v7__seed1000.json`, with their console logs beside them
  (`square_v9_seed1000_mujoco332.log`, `can_v7_seed1000_mujoco332.log`). 13:33–13:47 and
  13:47–13:57.
- **Prediction 0 holds for Square and fails for Can.** The Square base policy gains in the
  matching render; the Can one does not, and may lose.
- **B_can's evals are more spread than binomial noise:** their standard deviation is 0.080 against
  0.050 for 50 episodes at 0.85 (χ² 22.5 on 9 degrees of freedom), most of it the first eval's
  0.68, 3.4 binomial SDs below the mean. Without it the mean would be 0.871. Nothing in the record
  explains it. So B_can's binomial SE understates its uncertainty. The pre-registered rule takes
  the bar as measured, 0.852, and it stands. B_sq's spread is binomial (0.068 against 0.070).
- **The rule's numbers, with these bars:** tuned RLPD on Can passes at p ≥ 0.852. A residual arm
  on Square passes at p − 0.582 ≥ 2·sqrt(p(1−p)/500 + 0.582·0.418/500): at p ≥ 0.644, a margin
  of about 0.062 over the bar.

---

## 2026-10-07 — RESULTS — the fixed-env baselines' validation: tuned RLPD learns Can in our trainer once the render matches

`d047it6c`, `resfit-tuned-rlpd-can` seed 1, at `9cf207d` (the pre-registration above).

| Step | 0 | 10k | 20k | 30k |
|---|---|---|---|---|
| Eval success, 50 episodes | 0.00 | 0.00 | **0.70** | **0.80** |
| Upstream in its env, `zwmk393l` | 0.00 | 0.00 | 0.88 | 0.80 |
| Our control in the old render, `r1hvijqh` | 0.00 | 0.00 | 0.00 | 0.00 |

- **Passes, as predicted (prediction 1):** 75 successes of 100 at 20k and 30k, against the 5 the
  rule needs. The rule was already met at 20k (35), so the other nine runs were queued then, at
  14:38, by Aden's call to start them, not at 30k.
- **Training episodes,** with exploration noise: 0 of 50 succeed before 10k, 14 of 56 in
  (10k, 20k], 71 of 92 in (20k, 30k].
- **The critic, against the old render's signature** (means over the 1k steps before each eval):

  | | 10k | 20k | 30k | `r1hvijqh`, 8k–10k |
  |---|---|---|---|---|
  | `q_data_demo` | 0.63 | 0.62 | 0.62 | 0.81 |
  | `q_data_online` | 0.32 | 0.36 | 0.44 | −0.002 |
  | `policy_data_gap_demo` | 0.39 | 0.35 | 0.35 | 0.90 |
  | `policy_data_gap_online` | 0.78 | 0.70 | 0.61 | — |
  | `dQ_da_mean_abs` | 2.8e-5 | 3.2e-5 | 3.0e-5 | 6.7e-6 |
  | `critic_loss` | 1.9e-4 | 1.9e-4 | 1.4e-4 | 1.1e-5 |
  | `train/action_l1_magnitude` | 0.64 | 0.60 | 0.59 | 0.87 |

  Online transitions are valued at 0.3–0.4 from 10k, not at 0; the critic's loss is about 15×
  and its action gradient about 4× the old render's; the actor is off its bound and closer to the
  demos' actions.
- **What it settles:** of upstream's env's differences from ours, mujoco alone was enough. With
  torch 2.6.0, torchrl 0.7.0 and tensordict 0.7.0 kept, and only mujoco moved to 3.3.2, our trainer
  learns Can as upstream's did. So the 2026-10-07 FINDING's reading holds: in the old render the
  critic told demo frames from online ones by the floor.
- One seed, as pre-registered for the validation. The run goes on to 150k as seed 1 of its arm.

**The smoke run was stopped at Aden's call,** at 14:39, during its offline-buffer fill, so GPU 1
could start the baselines at once. Its speed and RAM come from `equi-residual-square` seed 1
instead, which runner `gpu1` launched at 14:39. Queues, from `9cf207d`:
- `gpu0`, after the validation run: `equi-tuned-rlpd-can` 1, `resfit-residual-square` 1,
  `resfit-tuned-rlpd-can` 2, `equi-tuned-rlpd-can` 2, `equi-residual-square-zero-init` 2.
- `gpu1`: `equi-residual-square` 1 (launched), `equi-residual-square-zero-init` 1,
  `equi-residual-square` 2, `resfit-residual-square` 2.

---

## 2026-10-07 — LAUNCHED — the fixed-env baselines: `equi-residual-square` seed 1

| Arm | Seed | Runner | Commit | wandb | Picked up / in wandb (EDT) |
|---|---|---|---|---|---|
| `equi-residual-square` | 1 | `gpu1` | `9cf207d` | **`kyv7clfc`** ([run](https://wandb.ai/aden-mckinney10-university-of-central-florida/robomimic-square-final/runs/kyv7clfc)) | 14:39 / 15:28 |

- Its gate said GO with one warning, "BC policy not in artifacts/": a false alarm. The gate looks
  in the launch worktree, and the trainer loaded main's `artifacts/run_3hzs5bz1_best:v9`, whose
  files it left untouched (TODO B7).
- Its offline buffer, 80,731 frames labeled by the base policy, filled in about 27 min after the
  15-min gate, as the smoke runs measured. 10.1 GiB resident at its first wandb step, beside
  `d047it6c` at 14.0 GiB.

---

## 2026-10-07 — NOTE — `kyv7clfc` (`equi-residual-square` seed 1) at 0.00 at 10k; ResFiT's Square residual moved ahead

An observation mid-run, deciding nothing; the run goes on to 150k as pre-registered.

- **Evals:** 0.60 at step 0 (the base policy, B_sq 0.582), **0.00 at 10k.**
- **Training episodes,** per 2k steps: 6 of 8 succeed in (0, 2k], then 1 of 7, 0 of 7, 0 of 6,
  0 of 7. The warm-up's base policy with U(±0.2) noise succeeded in 9 of 35.
- **The residual grows to its bound while the critic rates it above the base action:**
  `debug/residual_max_abs` 0.008, 0.049, 0.086, 0.097, 0.099 over the five 2k windows (the squash
  bounds it by `action_scale` 0.1); `train/residual_l1_magnitude` 0.002 to 0.035;
  `debug/q_advantage` positive throughout (0.0002 to 0.0024); online transitions valued at about
  −0.04, demo ones at 0.15–0.26; `debug/dQ_da_mean_abs` about 5e-5.
- This is the signature of `fal5sfot` and `fqo83rvd` (2026-10-04 RESULTS: 0.46 and 0.52 to 0.00
  from 10k, "the residual grew to the clip while the critic rated it above the base action"),
  now with the squash, two field-norm critic layers, 10 heads and the matching render.
- **Prediction 5 ("does not collapse") is at risk:** the collapse rule needs three consecutive
  evals at or below 0.05, so 20k and 30k decide it.

**Queue change, order only:** `resfit-residual-square` 1 moved to the front of `gpu0`'s queue,
ahead of `equi-tuned-rlpd-can` 1. It is the reference: the same recipe, base policy and env in
ResFiT's stack. If it also falls at 10k, the cause is shared, not the equivariant stack's. It
starts when `d047it6c` ends, about 19:30. No run's config changes.

---

## 2026-10-07 — NOTE — `kyv7clfc` collapsed by the 2026-10-01 rule; `d047it6c` finished

- **`kyv7clfc`** (`equi-residual-square` seed 1): 0.60, **0.00, 0.00, 0.02** at 0, 10k, 20k and
  30k, three consecutive evals at or below 0.05: collapsed, by the rule the pre-registration
  reports. Prediction 5 fails for this seed. None of its 33 training episodes in (10k, 20k]
  succeeded. It runs on to 150k as pre-registered.
- **`d047it6c`** (`resfit-tuned-rlpd-can` seed 1) finished at 18:52 after 320 min, exit 0, its
  package complete (507 MB). Evals 0.00 0.00 0.70 0.80 0.72 0.98 0.82 0.58 0.96 0.88 0.96 **1.00
  1.00 0.92 1.00 0.94**; the five at 110k–150k average 0.972 (B_can 0.852). The arm's verdict waits
  for seed 2. The 0.58 at 70k came while 62 of 67 training episodes in (65k, 70k] succeeded.
- **`resfit-residual-square` seed 1** launched on `gpu0` at 18:52, the reference for
  `kyv7clfc`. Its gate said GO with the false BC-policy warning (TODO B7).

---

## 2026-10-08 — NOTE — the fixed-env baselines overnight: ResFiT's residual lifts Square; both equivariant arms show the old critic signature

Observations mid-batch, deciding nothing; verdicts wait for both seeds.

| Run | Arm | Evals (0, 10k, …) | 110k–150k mean | Bar |
|---|---|---|---|---|
| `d047it6c` | `resfit-tuned-rlpd-can` 1 (done) | 0.00 0.00 0.70 0.80 0.72 0.98 0.82 0.58 0.96 0.88 0.96 1.00 1.00 0.92 1.00 0.94 | 0.972 | 0.852 |
| `oj5k0jqb` | `resfit-residual-square` 1 (done, 01:37, 404 min) | 0.50 0.30 0.58 0.54 0.40 0.54 0.56 0.64 0.70 0.64 0.76 0.72 0.84 0.90 0.66 0.88 | 0.800 | 0.582 |
| `kyv7clfc` | `equi-residual-square` 1 (to about 10:00) | 0.60 0.00 0.00 0.02 0.02 0.00 … 0.00 (130k) | — | 0.582 |
| `eyyi99q4` | `equi-tuned-rlpd-can` 1 (from 01:37) | 0.00 at all six, to 50k | — | 0.852 |

**The residual pair, by window** (means; training successes with exploration noise):

| | `oj5k0jqb` (ResFiT) | `kyv7clfc` (equivariant) |
|---|---|---|
| `debug/residual_max_abs`, from 10k | 0.099–0.100 | 0.099–0.100 |
| `train/residual_l1_magnitude`, (100k, 150k] | 0.083 | 0.034 |
| `debug/q_data_online`, (0, 5k] → (100k, 150k] | 0.022 → 0.204 | −0.026 → −0.004 (−0.065 at (10k, 20k]) |
| `debug/q_data_demo` | 0.25 → 0.47 | 0.19 → 0.37 |
| Training successes, (10k, 20k] / (50k, 100k] / (100k, 150k] | 12/38, 170/247, 223/279 | 0/33, 0/167, 0/108 |

Both residuals reach the 0.1 bound; ResFiT's is the larger. So the size of the residual does not by
itself explain the collapse. What differs is the critic: ResFiT's values online transitions above
zero and rising, and its actor's residual helps (0.30 at 10k, then up to 0.90). The equivariant
one values them at or below zero, which no return in {0, 1} supports.

**The tuned-RLPD pair:** `eyyi99q4` has, in the matching render, the signature every tuned-RLPD
run had in the old one, which `d047it6c` lost:

| (40k, 60k] | `eyyi99q4` (equivariant) | `d047it6c` (ResFiT) | `r1hvijqh` (ResFiT, old render, 8k–10k) |
|---|---|---|---|
| `q_data_online` | −0.003 | 0.481 | −0.002 |
| `q_data_demo` | 0.72 | 0.63 | 0.81 |
| `policy_data_gap_demo` | 0.73 | 0.37 | 0.90 |
| `dQ_da_mean_abs` | 6.8e-6 | 3.0e-5 | 6.7e-6 |
| `critic_loss` | 2.5e-5 | 1.5e-4 | 1.1e-5 |
| `train/action_l1_magnitude` | 0.74 | 0.58 | 0.87 |
| Training successes | 0/273 to 60k | 152/197 in (40k, 60k] | 0 |

**A reading, not yet tested:** the equivariant stack's critic still separates demo frames from
online ones, by something the render fix did not remove and ResFiT's stack does not see. The
render check's leftover (an even colour offset of about 0.005 per channel between the video-coded
demos and the env, 2026-10-07 CODE) and any difference in how the equivariant pipeline treats
offline and online images (crop, dtype, normalization) are candidates. Checking takes no training
run: compare the equivariant encoder's inputs and features on offline and online batches.

**Queue change, order only:** `resfit-residual-square` 2 moved to the front of `gpu1`'s queue, ahead
of `equi-residual-square-zero-init` 1 and `equi-residual-square` 2, so the equivariant runs wait on
Aden's decision. `kyv7clfc` and `eyyi99q4` run on as pre-registered.

---

## 2026-10-08 — FINDING — the equivariant critic's FieldNorm erases every trivial-type input

Found while debugging the fixed-env baselines' two equivariant arms (NOTE above). With
`equivariance.critic_norm=field_norm`, Aden's design for that batch, **the critic's Q cannot depend
on the wrist camera, the arm's height, the gripper opening, or the z, yaw and gripper actions.**
Analysis scripts, the frames they used and their outputs: `outputs/diagnostics/equi-critic-1008/`
(outside git; its README has the commands).

**The mechanism.** A trivial-type input reaches a hidden regular field of C8 only as a constant
over that field's 8 channels: the field's trivial part. `FieldNorm` subtracts each field's trivial
part, per sample, and adds a learnt constant back. Every hidden `Linear` of the critic (visual
projection, trunk, each head layer) is followed directly by `FieldNorm`, and the trivial inputs
enter only at those `Linear`s (the wrist features at the visual projection, proprioception and the
action at the trunk and again at the heads), so each is erased before any ReLU could mix it into
the other channels. EQUIVARIANCE.md records that `FieldNorm` removes a regular field's trivial
part; this is what that does to the critic's inputs.

**The measurement: forward perturbations, not gradients.** The largest change in Q (mean over the
10 heads) over every 4th frame of demo episode 37 of Square, at the batch's design:

| Input changed | `kyv7clfc`'s critic at step 0 (after 10k warm-up updates) | Fresh, `field_norm` | Fresh, `none` |
|---|---|---|---|
| Wrist image to noise / to black | 3.0e-7 / 1.8e-7 | 1.1e-6 / 8.3e-7 | 1.6e-3 / 8.7e-4 |
| State z +0.2 m | 1.6e-7 | 3.0e-7 | 5.7e-3 |
| Gripper qpos ±0.02 | 1.2e-7 | 3.7e-7 | 1.3e-2 |
| Action z +0.5 / yaw +0.5 / gripper sign flipped | 1.9e-7 / 1.5e-7 / 1.8e-7 | 2.7e-7 / 3.7e-7 / 2.7e-7 | 6.7e-3 / 3.4e-3 / 2.1e-2 |
| State x +0.1 m (an `irrep(1)` input) | 2.3e-3 | 9.6e-3 | 2.6e-3 |
| Action x +0.5 / rx +0.3 (`irrep(1)`) | 8.4e-2 / 4.6e-3 | 4.3e-2 / 2.2e-2 | 8.7e-3 / 5.8e-3 |
| Front image to noise | 0.80 | 0.38 | 0.08 |

- At a fresh init `none`, `iid_bn`, `inner_bn` and `layer_norm` (`RegularLayerNorm`, a LayerNorm
  over each sample's whole layer) all pass every input; only `field_norm` erases them.
- **The tests could not catch it:** a critic that ignores an input is invariant to it.
  `tests/test_sweep_arms.py` now checks that every input moves Q and the actor's action, for the
  arms it launches, and keeps this table's finding as a test of the `field_norm` critic.
- **The wrist encoder never trained.** Its gradient from one critic-loss backward pass, against
  the front encoder's, on 64 Can demo frames at a fresh init: 1.0e-9 with `field_norm`, 1.0e-2 with
  `none`, 1.4e-2 with `layer_norm`; ResFiT's wrist ViT gets 0.82 of its front ViT's. The runs logged
  the same (`debug/enc_enc_ih_grad_norm` about 1e-9). It is in `encoder_opt`, trained by the critic
  loss only, and the actor loss detaches the features, as in ResFiT: the wiring is right.

**What it accounts for, read from the logs** (not yet tested by a run without it):
- **Can (`eyyi99q4`):** the actor's z and gripper outputs get exactly zero gradient from Q, so it
  cannot learn to descend, grasp or lift, and the critic sees the grasp only through what survives
  of the front camera. No grasp, no reward, so online Q settles at 0. Eval Q-trajectories at 10k:
  `d047it6c`'s critic rates its own failing rollouts at 0.45–0.5; `eyyi99q4`'s at −0.005 ± 0.01,
  with spikes at the same timesteps in all 50 episodes: the arm follows one trajectory wherever the
  can is.
- **Square (`kyv7clfc`):** only the residual's xy and rx-ry parts get a gradient, from a critic
  blind to height, gripper and wrist; they reach the 0.1 disk bound by about 5k. The eval video at
  10k shows the hand rolled well off vertical, hovering at the nut; at step 0 it grasps, lifts and
  inserts. The critic's value is flat over the approach: at step 0 it is −0.13 for the first ~100
  steps of every eval episode, where `oj5k0jqb`'s ramps from about step 40; on demo episodes
  Q tracks the return (correlation 0.88–0.97) but sits at −0.12 to −0.14 over each episode's first
  third, where the return is up to 0.43.
- **So the "same signature as the old render" (NOTE above) is the same symptom from a different
  cause.**

**Also found, smaller:**
- **A codec cue separates demo frames from env frames in the matching render.** The demos are AV1
  `yuv420p` video; env frames are raw renders. Front camera, 40 demo first frames against 40 env
  resets: fine chroma detail (what 4:2:0 subsampling drops) is 1.95–2.08× higher in env frames on
  Can and 1.48–1.60× on Square, AUC 1.000 on both; mean colour is 0.004–0.006 higher per channel in
  env frames, AUC 1.000 on Can and 0.88–0.97 on Square. Round-tripping env frames through `yuv420p`
  takes the colour offset to chance (AUC 0.29–0.79): **the render check's leftover 0.005 offset is
  the codec, not the render.** Wrist camera: AUC 0.98 and 0.76 on Can, 0.54 and 0.56 on Square.
  Both stacks see it, and ResFiT learns Can (0.972). Whether a critic uses it is untested: on
  `kyv7clfc`'s critic, Q at reset frames was −0.123 on demo frames and −0.127 on env frames (head SD
  0.011), but reset frames sit in that critic's flat region, so the test is weak.
- **Proprioception conventions match.** At reset every state dimension overlaps between the demos'
  first frames and env resets (|difference| at most 0.53 SD, AUC 0.36–0.63); the quaternion is
  (x, y, z, w) in both, and the gripper qpos identical (±0.02083).
- **Neither augmentation changes pixel values.** `RandomShiftsAug` is an exact integer shift to
  5e-6; `CropRandomizer` is exact slicing; the chroma AUC stays 1.000 after either.
- **The wrist features are small:** RMS 0.17–0.18 against the front's 1.04–1.12, changing by
  0.01–0.02 over an episode against 0.26–0.34 (`kyv7clfc`, step 0). `ResEncoder76InHand`'s output
  LayerNorm is commented out.
- **Three diagnostics mislead.** `debug/enc_normalizer_grad_norm` is an accumulating sum: the
  normalizer's scale and offset are `nn.Parameter`s (`create_manual`), attached by `set_normalizer`
  after `_enc_params` and `encoder_opt` are built (`q_agent.py:242,265`; trainer `:527`), so they are
  never stepped or clipped and their gradients never zeroed. `debug/dQ_da_mean_abs` averages in the
  three action dimensions that are structurally zero under `field_norm`. `q_data_online` against
  `q_data_demo` compares states at different stages of their episodes.
- **A run saves a checkpoint only on a new best eval,** so `eyyi99q4` left none and `kyv7clfc` only
  its step-0 one.

Earlier runs with a `field_norm` critic (`critic-fn`, `24y42etk` and `konwqk6l`) and actor
(`actor-fn`, `q12tenbb`) all acted on near-blank images (2026-10-02 CORRECTION), so their results
say nothing about this.

---

## 2026-10-08 — DEVIATION — `eyyi99q4` stopped, and the four field-norm equivariant runs held

Aden's call, 2026-10-08, after the FINDING above. A departure from the fixed-env baselines'
pre-registration (2026-10-07): **that batch's two equivariant arms will have no verdict.**
- **`eyyi99q4`** (`equi-tuned-rlpd-can` 1) is stopped once its replacement's worktree and queue
  line are ready, at about 67k of 150k: 0 of 50 at all seven evals to 60k, no training success.
  Its GPU goes to the pre-registration below.
- **Held,** commented out of the queues with a note: `equi-tuned-rlpd-can` 2,
  `equi-residual-square` 2 and `equi-residual-square-zero-init` 1 and 2 (about 80 GPU-hours). They
  run the critic shown blind above. Re-enabling one is a line in a queue file.
- **`kyv7clfc`** (`equi-residual-square` 1) finishes as pre-registered, at about 10:10.
- **ResFiT's arms are unchanged:** `resfit-tuned-rlpd-can` 2 and `resfit-residual-square` 2 run
  after the two new arms, one per GPU, still from `9cf207d`.

---

## 2026-10-08 — PRE-REGISTERED — equivariant tuned RLPD on Can without FieldNorm: no critic norm, and RegularLayerNorm

**Question:** with the critic able to see every input (FINDING above), does equivariant tuned RLPD
learn Can? Two ways to drop `FieldNorm`, run side by side because ResFiT calls its critic's
LayerNorm crucial (taken from RLPD, "to mitigate catastrophic over-estimation"; its Fig. 5 ablation,
of the residual method on Can, falls to about 0 at 10k without it and reaches about 0.6 by 40k and
about 0.9 by 100k, against about 0.95 by 50k with it).

**Seed-group `critic-norm-can-1008`,** seed 1 per arm, 150k steps, from the commit that adds this
entry, launched from one worktree pinned to it. Names `<arm>-<seed>`, notes
`critic-norm-can-1008/<arm>/seed<seed>`. wandb `robomimic-can-final` / `tuned-rlpd`, with the
fixed-env baselines' tuned-RLPD runs.

| Arm | Overrides on `equi_tuned_rlpd_can_config` | GPU |
|---|---|---|
| `equi-tuned-rlpd-can-nonorm` | the design, `equivariance.critic_norm=none` | 1, after `kyv7clfc` |
| `equi-tuned-rlpd-can-layernorm` | the design, `equivariance.critic_norm=layer_norm` | 0, in place of `eyyi99q4` |

- **The design, otherwise as before:** `equivariance.residual_squash=true agent.critic.num_q=10
  equivariance.num_critic_layers=2 equivariance.actor_norm=none`, plus `algo.total_timesteps=150000
  read_buffer_caches=false write_buffer_caches=false`. `critic_norm=none` is the config's default,
  set explicitly.
- **What varies:** each arm differs from `eyyi99q4`'s launch in `equivariance.critic_norm` only, and
  the two arms from each other in it only (`tests/test_sweep_arms.py` pins all three). Against
  ResFiT's `d047it6c` they differ in the whole stack (ARCHITECTURE.md, "The two stacks side by
  side"), so nothing here compares stacks or equivariance (STANDARDS 3.3).
- **Config caveats:** as the fixed-env baselines' (that entry): `equivariance.num_actor_layers` is
  dead; `agent.actor.dropout`, `agent.critic.dropout` and `agent.critic.orth` do nothing; ResFiT's
  network fields under `agent` are not read by the equivariant agent.

**Rules:**
- **Seed 2 of an arm is queued iff its 20k and 30k evals total at least 5 successes of 100,** the
  validation's rule (2026-10-07). Aden's ask: a second seed if an arm looks successful.
- **An arm is stopped at 60k if every eval to 60k is 0 and no training episode has succeeded,**
  where `eyyi99q4` stood at its stop. Its GPU goes to the next queued line.
- **Verdict, once both seeds have run:** the fixed-env baselines' tuned-RLPD rule. p is the mean of
  the ten evals at 110k–150k, five per seed, 500 episodes; an arm passes iff p ≥ B_can = 0.852. With
  one seed, an observation.
- **Between the two arms:** an observation, deciding nothing (STANDARDS 5.3).
- **Reported at each 10k,** against `d047it6c` (ResFiT, same recipe) and `eyyi99q4` (field norm, to
  60k): the eval; training successes; means over the 1k steps before it of `debug/q_data_online`,
  `debug/q_data_demo`, `debug/policy_data_gap_{online,demo}`, `debug/dQ_da_mean_abs`,
  `train/critic_loss`, `train/action_l1_magnitude` and `debug/enc_enc_ih_grad_norm`; and the eval
  Q-trajectory plot.

**Predictions** (Claude's, before launch):
1. Both arms value online transitions above 0.1 by (5k, 10k] (`eyyi99q4` −0.013, `d047it6c` 0.30),
   with `dQ_da_mean_abs` above `eyyi99q4`'s 8.8e-6 there. Moderate confidence.
2. `layernorm` meets the seed-2 rule. Moderate confidence.
3. `nonorm` meets it too. Low confidence: ResFiT's ablation suggests a slower start without LayerNorm.
4. Neither arm's seed 1 averages B_can (0.852) over 110k–150k. Low confidence. Other stack
   differences stand: the disk bound (14.9% of Can's demo xy actions lie outside it), the wrist
   encoder's 1/100 share of the gradient, and the encoder.

**Resources and order.** About 0.41 s per step plus evals, so about 19–20 h per run, finishing
around 06:00 on 2026-10-09. RAM about 15 GB each, disk about 0.5 GB each. Each launch runs the gate
first (about 15 min). Queues after these: `resfit-tuned-rlpd-can` 2 on GPU 0, `resfit-residual-square`
2 on GPU 1; any seed 2 from the rule above goes ahead of them.

---

## 2026-10-08 — LAUNCHED — `critic-norm-can-1008` seed 1 of both arms; `eyyi99q4` stopped, `kyv7clfc` finished

From the pre-registration above, at its commit `6b6df8c`, in the worktree
`~/projects/equi-resrl-6b6df8c`. Every record is in main.

| Arm | Seed | Runner | wandb | Picked up / in wandb (EDT) |
|---|---|---|---|---|
| `equi-tuned-rlpd-can-layernorm` | 1 | `gpu0` | **`1ddrqdnd`** ([run](https://wandb.ai/aden-mckinney10-university-of-central-florida/robomimic-can-final/runs/1ddrqdnd)) | 09:25 / 09:47 |
| `equi-tuned-rlpd-can-nonorm` | 1 | `gpu1` | **`xm9cuo92`** ([run](https://wandb.ai/aden-mckinney10-university-of-central-florida/robomimic-can-final/runs/xm9cuo92)) | 10:10 / 10:32 |

- **Both gates said GO with no warning,** each in the worktree: Tier 1 188 passed, Tier 2 371
  passed (`1ddrqdnd`'s); the render check passed. Both run packages record `6b6df8c`, clean, and
  mujoco 3.3.2. Before the commit, `tests/test_sweep_arms.py` passed in full (23, its GPU tests
  included).
- **`eyyi99q4` stopped** (DEVIATION above): SIGINT at 09:25:26, exit 130 after 468 min, at step
  66,999. Evals 0.00 at all seven, 0 to 60k; 0 of 335 training episodes succeeded. Its package was
  finalized by hand (`finalize_run_package`), as the watcher does after a stop: 1,013 history rows.
- **`kyv7clfc` finished** (`equi-residual-square` 1): exit 0 at 10:10 after 1,171 min, package
  complete. Evals 0.60 0.00 0.00 0.02 0.02 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.02 0.00;
  0.004 over 110k–150k (B_sq 0.582). 9 of 502 training episodes succeeded, 2 of 484 after 5k. Its
  arm has no verdict: seed 2 is held.
- **The queues** (`~/queues/gpu{0,1}.txt`), after these: `9cf207d resfit-tuned-rlpd-can 2` on GPU 0
  and `9cf207d resfit-residual-square 2` on GPU 1; the four field-norm lines are commented out
  under a note citing the DEVIATION.

---

## 2026-10-08 — AMENDMENT — `critic-norm-can-1008`: `equi-tuned-rlpd-can-layernorm`'s stop check moves to 30k

Aden's call, at about 10:50, with `1ddrqdnd` at about 9k steps and before its 10k eval. **The
layernorm arm is stopped at its 30k eval if every eval to 30k is 0 and no training episode has
succeeded by then**: the pre-registered stop rule's condition, checked at 30k instead of 60k. The
nonorm arm keeps the 60k check, and both arms keep the seed-2 rule at 30k.

Why, from `1ddrqdnd`'s log to 9.1k: once its initial offset fades (TD targets −0.34 over its first
300 steps, from its heads' starting values; positive by 600–1,000), it follows `eyyi99q4`'s
trajectory. Over 9k–10k, `q_policy_online` is −0.014 (`eyyi99q4` −0.012 over 6k–8.5k, `d047it6c`
+0.31); `dQ_da_mean_abs` has fallen from 3.6e-5 to 8.6e-6 (`eyyi99q4` 8.8e-6); `actor_grad_norm`
has fallen to 0.0055 (`d047it6c` 0.057 and rising); 0 of 42 training episodes have succeeded. A
stop at 30k frees about 3.5 GPU-hours against the 60k check. Its GPU then takes the next line of
`gpu0`'s queue, `resfit-tuned-rlpd-can` 2; a nonorm seed 2 earned by the rule at 30k goes ahead of
`resfit-residual-square` 2 on whichever runner frees first.

---

## 2026-10-08 — PRE-REGISTERED — a third arm: ResFiT's tuned RLPD on Can without LayerNorm

**Question** (Aden, 2026-10-08): does ResFiT's stack need LayerNorm in its critic and actor MLPs to
learn Can by tuned RLPD? ResFiT calls the critic's LayerNorm crucial, but ablated it only for its
residual method (Fig. 5). In the equivariant stack, at the time of writing, the `layer_norm` arm
(`1ddrqdnd`, at about 10k) has settled at zero online value and the `none` arm (`xm9cuo92`, at about
3.4k) values online transitions at +0.07 to +0.11.

**Arm `resfit-tuned-rlpd-can-nolayernorm`,** seed 1, added to seed-group `critic-norm-can-1008`:
`--config-name=tuned_rlpd_can_config agent.critic.use_layer_norm=false agent.actor.use_layer_norm=false`
plus `algo.total_timesteps=150000 read_buffer_caches=false write_buffer_caches=false`. Name
`resfit-tuned-rlpd-can-nolayernorm-1`, notes `critic-norm-can-1008/resfit-tuned-rlpd-can-nolayernorm/seed1`,
wandb `robomimic-can-final` / `tuned-rlpd`. From the commit that adds this entry, in a worktree
pinned to it.
- **What varies:** it differs from `d047it6c`'s launch in those two switches only
  (`tests/test_sweep_arms.py`). Built, its critic has 0 LayerNorms (3 in `d047it6c`'s: the trunk
  projection and both head layers) and its actor 0 (3: the compression and both hidden layers); its
  two ViTs keep their 6. This is ResFiT's own ablation, "the layer norm for the actor and critic MLPs".
- **Config caveats:** `agent.critic.orth` does nothing (2026-10-06 CORRECTION); `agent.actor.dropout`
  and `agent.critic.dropout` do nothing (TODO B13). The 2026-10-08 audit also lists
  `critic.norm_weight`, `critic.feature_dim`, `actor.max_action_norm`, `vit.patch_size` and
  `vit.stride` as unread by this agent (not independently checked).

**Rules,** as for the other two arms: seed 2 is queued iff its 20k and 30k evals total at least 5
of 100; it is stopped at 60k if every eval to then is 0 and no training episode has succeeded; with
two seeds, it passes iff p (110k–150k, both seeds) ≥ B_can = 0.852. Against `d047it6c` it is one
seed each: an observation (STANDARDS 5.3). Reported at each 10k as for the others, against
`d047it6c`.

**Predictions** (Claude's, before launch):
1. It meets the seed-2 rule. Moderate confidence: ResFiT's residual ablation without LayerNorm
   still reached about 0.6 on Can by 40k.
2. Its online value is above 0.1 over (5k, 10k], as `d047it6c`'s was (0.30). Moderate confidence.
3. Its mean over 110k–150k is below `d047it6c`'s 0.972. Low confidence.

**Resources: a deviation from one run per GPU** (Aden's call). It shares GPU 0 with `1ddrqdnd`,
launched by hand in its own tmux session (`resfit-noln`) rather than by the `gpu0` runner, which
runs one launch at a time; its console goes to `outputs/queue/` like a runner launch's. Both runs
on GPU 0 slow down, so `1ddrqdnd`'s 30k stop check lands later in wall-clock time; neither run's
outcome depends on speed. When `1ddrqdnd` stops, the `gpu0` runner launches
`resfit-tuned-rlpd-can` 2 on GPU 0 beside this run. Three runs at once need about 45–60 GB of RAM;
the last gate saw 91 GB available.

---

## 2026-10-08 — DEVIATION — `1ddrqdnd` (`equi-tuned-rlpd-can-layernorm` 1) cut at 16.4k, before its 30k check

Aden's call, at about 11:25, after the analysis of why the failing critics settle at zero online
value (online value cut off from demo value from the first thousand steps; ResFiT's `d047it6c`
held 0.21 at 2k and about 0.3 to its first training success at 13.6k, with no reward online in
between). **The layernorm arm has no verdict, and its seed 2 is not queued.**
- **Stopped:** SIGINT at 11:26:29, exit 130 after 121 min. Last logged step 16,400. Evals 0.00 at
  0 and 10k; 0 of 82 training episodes succeeded. Over 14k–15k: `debug/q_data_online` −0.014,
  `debug/q_data_demo` 0.730, `debug/dQ_da_mean_abs` 7.9e-6 (`eyyi99q4` 8.1e-6 there). Its package
  was finalized by hand (`finalize_run_package`): 247 history rows.
- **Its GPU slot goes to a fourth arm** (PRE-REGISTERED below). `resfit-tuned-rlpd-can` 2 was
  commented out of `gpu0`'s queue just before the SIGINT, so the runner would not take it, and
  goes back in behind the fourth arm.
- **Prediction 1 of the pre-registration failed for this arm:** its online value over (5k, 10k]
  averaged −0.023, not above 0.1.

---

## 2026-10-08 — CODE — `encode_env_frames_like_demos`: the env's camera frames through the demos' video codec

**What it is.** A top-level config field, default `False`. `True` passes every camera frame the env
returns through the codec that stored the demos, inside the env wrapper
(`RobosuiteGymWrapper._process_obs`, `resfit/dexmg/environments/demo_codec.py`), so that rollouts,
evals and the online buffer all see coded frames. The video renders (`render()`) are untouched.
`False` is every run before 2026-10-08, bit for bit (`tests/test_demo_codec.py`). `True` adds a
field to the online buffer's cache key, and a key without it stays as it was; preflight's mirror of
the key (`cache_hashes`) follows.

**Why.** The demos reach the agent through lossy AV1 video and the env's frames do not. On Can's
front camera that leaves every env frame with about twice the demos' colour detail and an even
colour offset of about 0.005 per channel, each a perfect separator of env from demo frames (the
render check's leftover, 2026-10-07; measured per frame on 2026-10-08). A critic that can tell
online states from demo states can value online states at zero without touching its fit to the
demos, which is the signature of every failing tuned-RLPD run here.

**The codec, reproduced.** LeRobot's `encode_video_frames` defaults, which wrote the robomimic image
datasets: PyAV, `libsvtav1`, crf 30, `yuv420p`, `g=2`. ffprobe on the dataset's videos agrees: AV1
Main, `yuv420p`, TV range, keyframes at frames 0, 2, 4, ..., each followed by one P-frame (about 22
bytes at an episode's start). Each env frame is coded as a two-frame clip of itself, and the
keyframe is decoded. A one-frame clip codes it more coarsely: svt-av1 spends more bits on a
keyframe that a later frame refers to. PyAV's decode gives the dataset loader's demo frames bit for
bit, on both cameras. About 26 ms a frame, two cameras per env step, deterministic. libaom at crf
30, tried first, over-smooths Can's front camera and under-smooths Square's.

**Measured** (`tools/codec_check.py`): how well each per-frame statistic separates 40 demo first
frames (keyframes) from 40 env reset frames, as the area under the ROC curve; 0.5 cannot tell them
apart, 0 or 1 always can. "Detail" is the power left after a 2×2 box blur.

| Can, front camera | luma detail | cb detail | cr detail | mean R | mean G | mean B |
|---|---|---|---|---|---|---|
| raw | 0.65–0.80 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| through a one-frame clip | 0.09 | 0.16 | 0.19 | 0.23 | 0.80 | 0.65 |
| **through the two-frame clip (the switch)** | 0.45–0.59 | 0.48–0.50 | 0.40–0.47 | 0.48–0.50 | 0.59–0.60 | 0.62–0.64 |

Ranges are over two sets of resets. On the wrist camera, raw: cb 0.98–0.99, cr 0.76–0.80, the
colour means about 0.5; coded: 0.39 to 0.67 on every statistic. Luma detail on the wrist camera sits
near 0.65–0.69 either way: env resets are not the demos' start poses, which the wrist camera sees
close up. On Square's front camera, coded, cb detail still separates at 0.82 (raw 1.00).

**What it does not reproduce.** The demos' odd frames are P-frames, nearly copies of the keyframe
before them, so half the demo frames show the scene one step late; every coded env frame is a
keyframe. Nothing here touches the other ways online and demo states can differ: the
proprioception, the reset poses, and the behaviour itself.

**Tests.** `tests/test_demo_codec.py`, added to the gate's Tier 1: the settings are LeRobot's and the
dataset's, keyframe pattern included; the decode gives the loader's frames; the wrapper codes its
frames when asked and is bit-identical when not; the switch moves the online key and no other; the
trainer passes it to both envs; and, rendered in a subprocess, coded env frames sit within
[0.15, 0.85] on all twelve statistics while raw ones separate at 0.95 or more on the front camera's
colour.

---

## 2026-10-08 — PRE-REGISTERED — a fourth arm: the no-norm arm with env frames coded like the demos'

**Question** (Aden, 2026-10-08): with the one known perfect image cue between demo and env frames
removed (CODE above), does the equivariant critic's online value stay tied to its demo value, as
`d047it6c`'s did, instead of fading to zero as `xm9cuo92`'s has? `xm9cuo92` (`critic_norm=none`),
online value over demo value: 0.14 at 2k, 0.15 at 4k, 0.11 at 5k, 0.05 at 6k, 0.04 at 8k, 0.01 at
10k (0.005 / 0.675). `d047it6c`: 0.30 at 2k, 0.43 at 5k, 0.51 at 10k, before any training success.

**Arm `equi-tuned-rlpd-can-nonorm-democodec`,** seed 1, added to seed-group `critic-norm-can-1008`:
`--config-name=equi_tuned_rlpd_can_config`, the design (as the other equivariant arms),
`equivariance.critic_norm=none encode_env_frames_like_demos=true`, plus `algo.total_timesteps=150000
read_buffer_caches=false write_buffer_caches=false`. Name `equi-tuned-rlpd-can-nonorm-democodec-1`,
notes `critic-norm-can-1008/equi-tuned-rlpd-can-nonorm-democodec/seed1`, wandb
`robomimic-can-final` / `tuned-rlpd`. From the commit that adds this entry, in a worktree pinned to
it.
- **What varies:** it differs from `xm9cuo92`'s launch in `encode_env_frames_like_demos` only
  (`tests/test_sweep_arms.py`). Its evals see coded frames too, so its eval scores are the policy's
  on coded frames: the same task, other observations than every other run's evals.
- **Config caveats:** as for the other equivariant arms.

**Rules,** as for the other arms: seed 2 iff its 20k and 30k evals total at least 5 of 100; stopped
at 60k if every eval to then is 0 and no training episode has succeeded; with two seeds, it passes
iff p (110k–150k, both seeds) ≥ B_can = 0.852. Against `xm9cuo92`, one seed each: an observation
(STANDARDS 5.3). **The question's own readout:** its online value over its demo value, means over
(5k, 10k] and over (15k, 20k], each the ratio of the window's mean online and mean demo value.
"Tied" is at least 0.2 in both windows (`d047it6c` 0.45 and 0.58; `xm9cuo92` 0.03 over (5k, 10k]). Reported at each 10k as for the others, against
`xm9cuo92` and `d047it6c`.

**Predictions** (Claude's, before launch):
1. Its online value over demo value is below 0.2 over (5k, 10k]: removing the colour cue alone does
   not keep the online value up. Low confidence. The reason to expect it: `d047it6c` escaped on
   these same uncoded frames, and ResFiT's stack without LayerNorm (`9phsy6sf`, 3.4k steps at the
   time of writing) shows the failing start, on the same frames: online value −0.24 at 1k rising to
   −0.02 by 4k, demo value 0.21, `dQ_da_mean_abs` falling from 2.8e-5 to 1.4e-5. That points at the
   critic's norm and its starting values more than at the cue.
2. It does not meet the seed-2 rule. Low confidence.

**Resources.** GPU 0, launched by the `gpu0` runner from the line ahead of `resfit-tuned-rlpd-can` 2,
beside `9phsy6sf` (hand-launched in tmux `resfit-noln`). Two runs on GPU 0 and one on GPU 1. The
codec costs about 52 ms per env step, so about 15–20% slower than `xm9cuo92`. RAM about 15 GB.

---

## 2026-10-08 — LAUNCHED — `equi-tuned-rlpd-can-nonorm-democodec` 1 (`gkck6eqp`)

From the pre-registration above ("a fourth arm"), at its commit `e3793ce`, in the worktree
`~/projects/equi-resrl-e3793ce`, by the `gpu0` runner: picked up 11:54, in wandb 12:29
(**`gkck6eqp`**, [run](https://wandb.ai/aden-mckinney10-university-of-central-florida/robomimic-can-final/runs/gkck6eqp)).
- **The gate said GO:** Tier 1 198 passed and 2 skipped, Tier 2 389 passed; the render check passed
  on both tasks. The run package records `e3793ce`, clean, and mujoco 3.3.2. The online cache key
  the trainer built (`990f2e7c`) is the one preflight's mirror gives with the switch on.
- **Before the commit:** a 1,000-step smoke run of the arm exited 0, with no svt-av1 output in its
  console; env steps took 14.6% of its time (`xm9cuo92`: 6.6%).
- **Also launched by hand earlier, recorded here:** `resfit-tuned-rlpd-can-nolayernorm` 1 is
  **`9phsy6sf`** ([run](https://wandb.ai/aden-mckinney10-university-of-central-florida/robomimic-can-final/runs/9phsy6sf)),
  from `0a9626f`, in tmux `resfit-noln`, in wandb at 11:31.
- **The queues:** `gpu0` holds `9cf207d resfit-tuned-rlpd-can` 2, behind this run; `gpu1` holds
  `9cf207d resfit-residual-square` 2, behind `xm9cuo92`. The four field-norm lines stay commented out.

---

## 2026-10-08 — NOTE — `9phsy6sf`'s actor saturated in its first 200 updates and has had no gradient since

**`resfit-tuned-rlpd-can-nolayernorm` 1, at 29.6k:** evals 0.00 at 0, 10k and 20k; 0 of 148 training
episodes succeeded (`d047it6c` by 30k: 0.70 at 20k, 0.80 at 30k, 71 of 92 in (20k, 30k]).
- **Its actor sits at the corners of the action box.** `train/action_l1_magnitude` is exactly 1.000,
  and `train/actor_grad_norm` reads 0.0000 at every logged step from the first (200) to 29.6k
  (`d047it6c`: action 0.95 falling to 0.60, gradient 0.015 rising to 0.07). The actor's action has not
  moved since step 200; `debug/policy_data_gap_online` falls from 1.00 to 0.45 only as the online
  buffer fills with that action plus noise.
- **Not at initialization.** Built from the arm's config, its pre-tanh outputs on the reference batch
  have median |x| 0.06 (d047it6c's config: 0.27). Run for 200 updates on that one batch, all three
  of d047it6c's config, this arm's, and the critic's LayerNorm alone removed drive the action to
  the box. The actor's gradient then falls to about 1e-6 by update 25 in this arm and stays there;
  with the actor's LayerNorms it stays at 1e-3 to 3e-1. So it is the actor's LayerNorms, not the
  critic's, that keep its tanh out of saturation. The likely mechanism (not isolated): with nothing
  bounding the scale of each layer's output, Adam's steps on the 20,736-input compression layer
  grow the pre-tanh activations until tanh's slope is zero.
- **What it means for the arm's question.** The run does not test the critic's LayerNorm: its critic
  has been scoring one fixed corner action for 30k steps. Demo value 0.12–0.15 against the demos'
  mean discounted return of 0.40 is the critic's value of that action from demo states, as TD3's
  target asks. Its predictions 1 and 2 have failed (2 over (5k, 10k]: −0.023 online), and 3 is
  moot. A critic-only ablation (`agent.critic.use_layer_norm=false`) would test the critic's
  LayerNorm with a working actor.
- **A correction to the fourth arm's stated reason.** Its prediction 1 cites `9phsy6sf`'s start as
  pointing "at the critic's norm and its starting values more than at the cue". That reading does
  not hold: `9phsy6sf` failed through its actor. The prediction stands as registered; its reason
  does not.

---

## 2026-10-08 — DEVIATION — `9phsy6sf` (`resfit-tuned-rlpd-can-nolayernorm` 1) stopped at 33.9k

Aden's call, at about 13:00, after the NOTE above: its actor has had no gradient since step 200, so
the run cannot test its question, and it shared GPU 0 with `gkck6eqp`. A departure from its
pre-registered 60k stop rule. **The arm has no verdict, and its seed 2 is not queued** (its 20k and
30k evals total 0 of 100, so the rule would not have queued it either).
- **Stopped:** SIGINT at 13:04:51, exit 130. Last logged step 33,900. Evals 0.00 at 0, 10k, 20k and
  30k; 0 of 169 training episodes succeeded. Package finalized by hand: 509 history rows. Its tmux
  session `resfit-noln` is closed.
- **Its slot on GPU 0 goes to the fifth arm** (PRE-REGISTERED below).

---

## 2026-10-08 — PRE-REGISTERED — a fifth arm: RegularLayerNorm in the actor and the critic, its start matched to ResFiT's

**Question** (Aden, 2026-10-08): with the equivariant LayerNorm where ResFiT's `use_layer_norm` puts
LayerNorm, in the actor and the critic, and starting where ResFiT's critic starts, does equivariant
tuned RLPD learn Can, and does its critic keep a positive online value?

**Where the norms go.** ResFiT's switch puts a LayerNorm after every hidden Linear of its actor (the
compression and two hidden layers: 3) and of its critic (the input projection and each head's two
hidden layers); its ViTs keep their own either way. `equivariance.actor_norm` and `critic_norm` put
the norm after every hidden Linear of the equivariant actor (visual projection, trunk, hidden layer:
3) and critic (visual projection, trunk, each head's two: 2 + 2 per head). Two differences remain:
the equivariant critic has a trunk layer ResFiT's lacks (one more norm before the heads), and each
head's last hidden layer goes norm → max over the group → Linear, where ResFiT's goes LayerNorm →
ReLU → Linear. `RegularLayerNorm` is ResFiT's LayerNorm with one learnt scale and shift per regular
field instead of per channel, which equivariance requires. FieldNorm is not used: it erases every
trivial-type input (2026-10-08 FINDING), and Aden chose this arm alone.

**Why the start is matched.** Measured at init on the reference batch, 5 seeds each: the SD across
the 10 heads of each head's mean Q, and the mean of the min over pairs of heads (the TD target's
pessimism before γ).

| Critic at init | Spread across heads | Min of two heads |
|---|---|---|
| ResFiT, LayerNorm (`d047it6c`) | 0.298 ± 0.048 | −0.145 |
| ResFiT, critic LayerNorm off | 0.328 ± 0.050 | −0.154 |
| Equivariant, `critic_norm=none` (`xm9cuo92`) | 0.476 ± 0.481 | −0.087 |
| Equivariant, `critic_norm=layer_norm` (`1ddrqdnd`) | 1.024 ± 0.071 | −0.375 |
| **Equivariant, `layer_norm`, `critic_last_layer_init_scale=0.3`** | **0.307 ± 0.021** | **−0.113** |

The unscaled RegularLayerNorm critic starts 3.4× as spread as ResFiT's, from escnn's initialization
of each head's last Linear and the positive features the max over the group gives after a norm;
ResFiT's own LayerNorm barely moves its spread. A scale of 0.3 on each head's last Linear (an
existing field) matches ResFiT's. The actor's norm does not touch the critic's start.

**Arm `equi-tuned-rlpd-can-ln-matched`,** seed 1, added to seed-group `critic-norm-can-1008`:
`--config-name=equi_tuned_rlpd_can_config equivariance.residual_squash=true agent.critic.num_q=10
equivariance.num_critic_layers=2 equivariance.actor_norm=layer_norm equivariance.critic_norm=layer_norm
equivariance.critic_last_layer_init_scale=0.3`, plus `algo.total_timesteps=150000
read_buffer_caches=false write_buffer_caches=false`. Env frames uncoded, as `xm9cuo92` and
`1ddrqdnd` (Aden's choice). Name `equi-tuned-rlpd-can-ln-matched-1`, notes
`critic-norm-can-1008/equi-tuned-rlpd-can-ln-matched/seed1`, wandb `robomimic-can-final` /
`tuned-rlpd`. From the commit that adds this entry, in a worktree pinned to it.
- **What varies:** against `1ddrqdnd`'s launch, `equivariance.actor_norm` and
  `critic_last_layer_init_scale` only; against `xm9cuo92`'s, those and `critic_norm`
  (`tests/test_sweep_arms.py`, which also checks the start against ResFiT's on the GPU and the norm
  after every hidden Linear). Two changes against `1ddrqdnd`: if this arm differs from it, this
  entry cannot say which change did it.

**Rules,** as for the other arms: seed 2 iff its 20k and 30k evals total at least 5 of 100; stopped
at 60k if every eval to then is 0 and no training episode has succeeded; with two seeds, it passes
iff p (110k–150k, both seeds) ≥ B_can = 0.852. Against `xm9cuo92`, `1ddrqdnd` and `d047it6c`, one
seed each: an observation (STANDARDS 5.3). Readouts as the fourth arm's: online value over demo
value, the ratio of the window means over (5k, 10k] and (15k, 20k] (`d047it6c` 0.45 and 0.58;
`xm9cuo92` 0.03 over (5k, 10k]; `1ddrqdnd` −0.03 and −0.02); and its online value over its first 3k
steps (`1ddrqdnd` −0.33 at 1k, −0.08 at 2k; `d047it6c` +0.005 and +0.21).

**Predictions** (Claude's, before launch):
1. Its online value is at or above 0 by 1k and stays at or above 0 through 3k, unlike `1ddrqdnd`'s.
   Moderate confidence: the start that drove `1ddrqdnd` negative is gone.
2. Its online-over-demo ratio over (5k, 10k] is below 0.2. Low confidence: `xm9cuo92`, which also
   starts without the offset (min of two −0.087), fell to 0.03.
3. It does not meet the seed-2 rule. Low confidence.

**Resources.** GPU 0, beside `gkck6eqp`, launched by hand in tmux `ln-matched`: the `gpu0` runner runs
one launch at a time and is busy with `gkck6eqp`. Its console goes to `outputs/queue/` like a runner
launch's. About 0.41 s per step alone, slower shared. `resfit-tuned-rlpd-can` 2 stays queued on
`gpu0` behind `gkck6eqp`.

---

## 2026-10-08 — LAUNCHED — `equi-tuned-rlpd-can-ln-matched` 1 (`496bq48s`)

From the pre-registration above ("a fifth arm"), at its commit `6e61e65`, in the worktree
`~/projects/equi-resrl-6e61e65`, by hand in tmux `ln-matched` on GPU 0, beside `gkck6eqp`: launched
13:19, in wandb 13:48 (**`496bq48s`**,
[run](https://wandb.ai/aden-mckinney10-university-of-central-florida/robomimic-can-final/runs/496bq48s)).
Console: `outputs/queue/20261008-131952_equi-tuned-rlpd-can-ln-matched-1_gpu0.log`.
- **The gate said GO:** Tier 1 198 passed and 2 skipped, Tier 2 402 passed; the render check passed
  on both tasks. Before the commit, `tests/test_sweep_arms.py` and `tests/test_demo_codec.py` passed
  in full (63, GPU tests included). The run package records `6e61e65`, clean, and mujoco 3.3.2; the
  online cache key the trainer built (`f3250e68`) is preflight's.
- **Frames uncoded,** as pre-registered (Aden's choice, made before `gkck6eqp`'s first training
  successes at 4.9k and about 5.5k were seen).

---

## 2026-10-08 — NOTE — `gkck6eqp` (env frames through the demos' codec) scores 0.94 at 10k; `xm9cuo92` fails its seed-2 rule

**The first equivariant tuned-RLPD run to learn Can.** `equi-tuned-rlpd-can-nonorm-democodec` 1
(`gkck6eqp`), at 10.8k: evals 0.00 at 0 and **0.94 at 10k** (47 of 50, on coded frames as
pre-registered); 36 of 73 training episodes succeeded, the first at step 4,931. One seed.

| Windowed means | 0–2k | 2–4k | 4–6k | 6–8k | 8–10k |
|---|---|---|---|---|---|
| `gkck6eqp` online value | 0.083 | 0.173 | 0.171 | 0.227 | 0.266 |
| `xm9cuo92` online value (same arm, uncoded) | 0.018 | 0.077 | 0.055 | 0.030 | 0.007 |
| `d047it6c` online value (ResFiT) | 0.106 | 0.253 | 0.286 | 0.292 | 0.308 |
| `gkck6eqp` critic loss | 6.5e-4 | 3.2e-4 | 2.8e-4 | 2.7e-4 | 2.4e-4 |
| `xm9cuo92` critic loss | 5.3e-4 | 2.4e-4 | 1.3e-4 | 9.1e-5 | 6.6e-5 |
| `gkck6eqp` policy-to-demo action gap | 0.545 | 0.375 | 0.314 | 0.332 | 0.317 |
| `xm9cuo92` policy-to-demo action gap | 0.563 | 0.486 | 0.493 | 0.520 | 0.540 |
| `gkck6eqp` training successes | 0/10 | 0/10 | 4/11 | 10/16 | 13/17 |

- **The question's readout:** online over demo value over (5k, 10k] is 0.41 (0.231 / 0.567),
  `d047it6c` 0.45, `xm9cuo92` 0.03: tied. (15k, 20k] is still to come.
- **The account it bears on** (the PRE-REGISTERED entry's question): the arms differ only in the
  env's frames. Uncoded, the critic's online value fell to zero and its loss fell 8× below this
  run's, as a critic that tells online from demo frames apart and values every online frame at
  zero would; coded, the online value held at ResFiT's share and rose with the first successes.
  The eval Q-trajectories at 10k rise from 0.3–0.5 at an episode's first frame to about 1.0 at its
  success (steps 60–120). One seed; the arm's seed 2 is decided by its 30k rule.
- **Predictions:** 1 (ratio below 0.2 over (5k, 10k]) failed. 2 (no seed 2) is likely to fail.
- **`xm9cuo92`** (`equi-tuned-rlpd-can-nonorm` 1), at 35k: evals 0.00 at 0, 10k, 20k and 30k; 0 of
  175 training episodes succeeded. Its 20k and 30k evals total 0 of 100: **seed 2 not queued.** Its
  60k stop rule stands.
- **What this does not show yet:** that the policy acts well on uncoded frames (its evals are coded);
  anything about Square, whose front camera keeps some colour-detail separation after the codec
  (0.82, CODE above); or how the codec moves a residual run's base policy, which was trained on
  coded demo frames and has only been measured on uncoded ones (B_can, B_sq).

---

## 2026-10-08 — CODE — re-rendered demos: the robomimic demos' frames rendered by our env, losslessly, one physics substep late as the env observes

**What it is.** `tools/rerender_demos.py` builds two local LeRobot datasets in the LeRobot home,
`local/robomimic-mh-can-rerendered` and `local/robomimic-mh-square-rerendered`. A run selects one
with `offline_data.name`, an existing field: nothing in the trainer or the env changes, and
`encode_env_frames_like_demos` stays, default off. Nothing is uploaded. Each dataset records its
provenance in `meta/rerender.json`: the source file and its sha256, every guard's result, robosuite
and mujoco, and the commit it was built at (this entry's).

**Why.** The demos reach the agent through lossy AV1 video and the env's frames do not, which lets a
critic tell every demo frame from every online frame (CODE above, and the NOTE on `gkck6eqp`). The
codec switch removed that cue on Can by coding the online frames, evals included. Re-rendering
removes it at the source: the demo frames come from the renderer, the code path and the timing the
online frames come from, uncompressed on both sides (Aden's call, 2026-10-08, for the residual runs).

**How.** robomimic's v1.5 files (`robomimic/robomimic_datasets` at revision `74fa0184`,
`v1.5/{can,square}/mh/demo_v15.hdf5`, recorded as robosuite 1.5.0) hold every demo's simulator
state at every step. Each state is set into our env's own scene, and the observation is read
through `RobosuiteGymWrapper._process_obs`, as every online observation is. Not into the demos'
saved XML: reloading a scene from XML moves 1–4 front-camera pixels by 1–4 levels, from the
precision MuJoCo writes numbers at. The frames are stored as PNG inside the parquet files; actions,
done flags, timestamps and indices are copied from the source datasets.

**One physics substep late: a guard that failed, and the fix.** The first version's identity guard
(render a state the env reported; the frames must be the ones the env returned) passed at resets
and failed once the arm had moved. robosuite's `step()` runs 25 MuJoCo substeps of 2 ms. A MuJoCo
step first computes where every body is from the joint angles, then integrates, which updates the
joint angles. robosuite's observation timer fires right after the 25th substep (every step: 200 of
200 on Can, 300 of 300 on Square), and nothing recomputes the positions first. So every online
camera frame and end-effector pose shows the joint angles one substep (2 ms) before the state the
simulator holds; the gripper's joint readings are current. `reset()` recomputes the positions
before it observes, so an episode's first observation is exact. The source datasets' frames were
rendered from each state with positions recomputed, so mid-episode the demos show the state itself
and the online frames lag it. On the trainer's step path, full episodes of random actions, online
observations against renders of the same states:

| Online vs the state itself | Can (200 steps) | Square (300 steps) |
|---|---|---|
| Front camera | 186 of 7,056 pixels differ, mean 0.15 levels | 152, mean 0.20 levels |
| Wrist camera | 3,603 pixels, mean 1.9 levels | 1,327, mean 0.5 levels |
| End-effector position | 0.73 mm mean, 1.97 max | 0.62 mm mean, 1.95 max |

It has been in every run, ResFiT's included, and `gkck6eqp` learned with it, so it is not what
blocked learning; it is a mismatch between demo and online data all the same. Of three ways to
close it (make the env recompute positions before it observes; accept the gap; render the demos as
the env observes), **Aden chose the third.** The env, the trainer, the base policies and the bars
B_can and B_sq stay as they are, so every earlier run stays comparable; an exact env would have
changed what every new run's base policy sees. MuJoCo's Euler integrator moves the joint angles by
dt times the new velocity after computing the positions, so the previous substep's joint angles are
the state's moved back by dt × qvel. The tool sets each state, moves the joint angles back one
substep, recomputes the positions, puts the joint angles back and renders (`as_stepped`). Each
episode's first frame is rendered from its state, as `reset()` observes; the velocity is exactly
zero in all 600 demos' first states and in our resets, so the two agree there anyway.
**observation.state is re-derived the same way, not copied,** because its end-effector pose lags
online too. It moves from the source's by 0.33 mm on average on Can's first three demos (0.75 mm at
most; the full figures go in each dataset's `meta/rerender.json`), its gripper readings not at all,
and its statistics, which the equivariant normalizer reads, move with it. The replay guard still
holds the observation computed from each state to the source's column.

**Guards,** each of which stops the build:
- **identity:** an env touched only through the wrapper's `reset()` and `step()`, so robosuite's
  observation timers run as in training. At 3 resets and after each of 90 random actions, the
  frames and observation.state rendered from its state, as `reset()` or `step()` leaves it, equal
  what it returned, bit for bit. Before the fix this failed mid-episode.
- **layout:** the same joints (13 on Can, 11 on Square) at the same addresses; every demo body is
  ours and lands within 1e-8 m of where the demos' scene puts it; ours adds only the two mocap target
  bodies (`left_eef_target`, `right_eef_target`, parked at z = −1 m).
- **pairing:** all 300 demos of each task pair with the LeRobot episodes: same lengths, actions
  exactly equal (62,756 frames on Can, 80,731 on Square).
- **replay:** observation.state computed from each state equals the source's within 1.2e-7,
  float32's rounding, on every frame of both tasks; the stepped one differs from it only in the
  end-effector pose, within 5 mm and 0.05 (quaternion, up to sign).
- **columns,** after writing: the copied columns equal the source's, observation.state is the one
  rendered, and the action statistics equal the source's.
- **verify,** after the build: stored frames, decoded as the trainer decodes them, and stored
  states equal fresh renders.

Layout, pairing and replay passed on every demo of both tasks earlier on 2026-10-08, in a read-only
run with the identity guard at resets only; the build runs every guard again.

**Not changed: `lite_physics`.** The demo files record `lite_physics=false`; our env runs robosuite
1.5.1's default, `true`, and robosuite documents `false` as keeping compatibility with datasets
collected in robosuite 1.4.1 and earlier. It changes the physics (how a state follows from the last),
not what is rendered from a state, and matching it would change the env for every new run.
Unmeasured.

**Also in this commit.** `preflight.cache_hashes` takes the run's `offline_data.name`, which the
trainer puts in the offline cache key, and the gate reads the name from the composed config: the
re-rendered Can dataset moves equivariant tuned RLPD's offline key from `e83a0b31` to `a1942abb`,
and no online key moves. `tools/codec_check.py` and `tools/render_check.py` take `--rerendered`,
which compares env frames with the re-rendered demos instead of the source's.

**Tests.** `tests/test_rerender_demos.py`, in the gate's Tier 1. After a MuJoCo step the positions
are the previous substep's, and `as_stepped` recovers them (a small model, no renderer). The
stepped state may move only the end-effector pose. Rendered in a subprocess, every guard passes on
Can's first two demos. A built dataset is lossless, complete, and records passing guards and a
clean commit. The dataset name moves the offline key and no other, and the gate reads it.

---

## 2026-10-08 — NOTE — the re-rendered datasets: built, verified, indistinguishable from the env's frames; the identity guard is not bit-exact on Can's wrist camera

**Built** at `1c9cdbc`, clean, into `~/.cache/huggingface/lerobot/local/`, with every guard passing on
every demo (each dataset's `meta/rerender.json` holds the full report):

| | Can | Square |
|---|---|---|
| Episodes, frames | 300, 62,756 | 300, 80,731 |
| Actions, done flags, timestamps | the source's, exactly | the source's, exactly |
| observation.state from each state vs the source's | within 1.2e-7 | within 1.2e-7 |
| Stepped end-effector position vs the source's | 0.24 mm mean, 1.02 mm max | 0.16 mm mean, 1.65 mm max |
| Stepped end-effector quaternion, max (up to sign) | 0.0019 | 0.0035 |
| observation.state statistics, largest change | 5.1e-4 | 4.2e-4 |
| Identity guard (3 resets, 90 steps) | bit-identical | bit-identical |
| Size, build time | 1.2 GB, 12 min | 0.97 GB, 14 min |

**Verify:** at 5 episodes spread through each dataset, every stored frame, decoded as the trainer
decodes it, and every observation.state equals a fresh render.

**Against the env** (the plan's blocking check; all pass):
- `tools/codec_check.py --rerendered`, 40 re-rendered demo first frames against 40 env resets, raw on
  both sides: Can 0.49–0.61 on the front camera and 0.50–0.61 on the wrist; Square 0.49–0.66 and
  0.45–0.56. Every statistic within [0.15, 0.85]. Against the source's coded demos, raw env frames
  separate at 1.00 on Can's front-camera colour, and Square's front-camera colour detail still
  separated at 0.82 through the codec switch; re-rendered it is 0.66, so that was the codec, not the
  scene. Coded env frames separate from the re-rendered demos at 0.00 on colour: the codec switch
  and the re-rendered demos must not be combined.
- `tools/render_check.py --rerendered`, 3 repeats: Can's demo-vs-env gap 0.0029–0.0033 against
  0.0035 between halves of the demos, 0.8–0.9× (against the source: 2.06×); Square 0.0034–0.0042
  against 0.0045, 0.76–0.93×.
- What the source's codec changed (`--codec-change`, 10 episodes, the source's frames against
  renders of the same states): Can front camera 2.7 levels on keyframes and 3.1 on P-frames, 99% of
  pixels; wrist 2.7 and 3.3, 96%. Square front 1.8 and 2.4, 99%; wrist 1.9 and 2.3, 96%.

**The identity guard is not bit-exact on Can's wrist camera.** One run of the gate's rendered test
failed the identity guard (its message was not kept). Repeated, with every mismatch recorded:
- **Size:** in 620 Can observations, 10 wrist frames differed, each in 1–10 pixels along one edge,
  by 1 level, in runs of consecutive steps; in 750 more, 7. Square: 0 of 620. The front camera and
  observation.state: never.
- **Not the reconstruction:** the online env's joint angles one substep before each observation,
  saved from inside its step, equal `as_stepped`'s within 1.1e-15; every geom, camera and light pose
  within 2.3e-15 (600 Can steps).
- **Not the scene:** the list MuJoCo hands the GPU for the wrist camera (each geom's type, object,
  float32 pose, size, colour and camera distance, the GL cameras and the lights) is bit-identical
  between the online env and the re-render at all 750 steps, the 7 that differ included.
- **Not the guard's two envs sharing a GL context:** robosuite makes a context current only when it
  creates one, so two envs in one process render in whichever was created last. With the online env
  in its own process (a change to the guard, not yet committed), 2 of 3 runs of 620 observations
  still failed, by 3 and 4 wrist pixels.
- **So** the GPU renders the same scene to slightly different pixels on rare frames, most likely
  depending on what the context drew before. Training's envs render in sequence in their own
  processes; the datasets were rendered one episode per freshly reset scene. Bit-identical frames
  every time are out of reach from outside the renderer. The residue is at most 1 level on at most
  10 pixels of about 1–2% of Can's wrist frames, against the codec's 1.8–3.3 levels on 96–99% of
  pixels. Whether to relax the guard to it is Aden's call; the plan waits on it.

---

## 2026-10-08 — NOTE — `critic-norm-can-1008`'s rules: `xm9cuo92` stopped at 60k, `gkck6eqp`'s seed 2 queued; a queue line held for the re-rendered demos

- **`xm9cuo92` (`equi-tuned-rlpd-can-nonorm` 1) stopped by its 60k rule:** evals 0.00 at every 10k
  from 0 to 60k, 0 of 314 training episodes succeeded; online over demo value 0.03 over (5k, 10k] and
  over (15k, 20k]. SIGINT at 16:54:49, at about 62.8k. **Its exit and run package are unchecked:**
  the session's permission classifier blocked the command that waited on the process, so Aden
  confirms the exit (`outputs/queue/gpu1.log`) and the package. One seed, no verdict; its seed 2 was
  never queued (0 of 100 at 20k and 30k).
- **`gkck6eqp` (`equi-tuned-rlpd-can-nonorm-democodec` 1) meets its seed-2 rule:** 41 of 50 at 20k
  and 46 of 50 at 30k, 87 of 100; 216 of 292 training episodes succeeded by 33k. **Seed 2 queued**
  at 18:48 on `gpu0` as `e3793ce equi-tuned-rlpd-can-nonorm-democodec 2`, ahead of
  `resfit-tuned-rlpd-can` 2, so it starts when seed 1 ends; not on `gpu1`, whose two slots are kept
  for the re-rendered-demos plan (TODO B16). Its prediction 2 (no seed 2) failed. Its readout: online
  over demo 0.41 over (5k, 10k] and 0.65 over (15k, 20k], tied in both.
- **`496bq48s` (`equi-tuned-rlpd-can-ln-matched` 1) at 20k:** evals 0.00 at 0, 10k and 20k; 0 of 116
  training episodes succeeded; online over demo −0.01 over (5k, 10k] and −0.00 over (15k, 20k].
  Prediction 1 (online value at or above 0 by 1k and through 3k): partly. Its mean over the first 1k
  is −0.045 (`1ddrqdnd` −0.33 at 1k), then +0.07 and +0.03. Prediction 2 (ratio below 0.2 over (5k,
  10k]) holds. Its seed 2 needs at least 5 of 50 at 30k.
- **A queue line held:** `9cf207d resfit-residual-square 2`, at the head of `gpu1`, commented out at
  16:55 with a dated note, so GPU 1's two slots go to the re-rendered-demos plan's Can validation run
  and equivariant residual on Square; to be restored once the residual has launched.

---

## 2026-10-08 — NOTE — `496bq48s` (`equi-tuned-rlpd-can-ln-matched` 1) fails its seed-2 rule

At 30k: evals 0.00 at 0, 10k, 20k and 30k; 0 of 165 training episodes succeeded; online over demo
value −0.01 over (5k, 10k] and −0.00 over (15k, 20k]. Its 20k and 30k evals total 0 of 100: **seed 2
not queued.** Prediction 3 (no seed 2) holds. Its 60k stop rule stands, due at about 00:30–01:00 on
2026-10-09 at its present pace.

---

## 2026-10-09 — CODE — the identity guard allows the GPU's residue, and runs its online env in its own process

Aden chose option A (2026-10-09) for the identity guard of `tools/rerender_demos.py`, which failed
on rare Can wrist frames (NOTE of 2026-10-08 on the re-rendered datasets):
- **observation.state must still be exactly equal;** each frame must be bit-identical except for at
  most 32 pixels each off by at most 1 level (`FRAME_TOL_PIXELS`, `FRAME_TOL_LEVELS`). Measured: at
  most 10 pixels, 1 level, in about 1–2% of Can's wrist frames, on scenes the GPU was handed
  float32-identical. The report counts the frames that were not bit-identical and the most pixels
  any differed in.
- **The guard's online env runs in its own process,** as each training env does: robosuite makes a
  GL context current only when it creates one, so two envs in one process render in whichever was
  created last. That was not the residue's cause (it persisted), but the guard should not depend on
  it.
- **A standing test,** `test_env_frames_cannot_be_told_from_rerendered_demo_frames` (Tier 1): raw env
  frames against the re-rendered Can demos' first frames sit within [0.15, 0.85] on all twelve
  `tools/codec_check.py` statistics.

The datasets are not rebuilt: their rendering code is unchanged, they were built at `1c9cdbc` under
the stricter guard, which they passed, and `meta/rerender.json` records that guard's report.
`tests/test_rerender_demos.py` accepts either report.

---

## 2026-10-09 — DEVIATION and NOTE — `496bq48s` stopped 49k past its stop rule; `xm9cuo92`'s critic diverged before its stop; both packages finalized; `resfit-residual-square` 2 launched on GPU 0

- **`496bq48s` (`equi-tuned-rlpd-can-ln-matched` 1) stopped at 109.4k, not at 60k.** Its 60k rule
  was met at about 00:30 (evals 0.00 at every 10k from 0 to 60k, no training success), but nobody
  stopped it: on 2026-10-08 the session's permission classifier had blocked a follow-up to stopping
  `xm9cuo92`, so Claude left the next stop to Aden, who was away. It ran on with evals 0.00 to 100k
  and 0 of 536 training episodes succeeding; the extra 49k change nothing about the arm. SIGINT at
  08:35:30 on Aden's OK; console `EXIT 130`. One seed, no verdict, no seed 2.
- **`xm9cuo92` (`equi-tuned-rlpd-can-nonorm` 1) exited 130 at 16:54:52** on 2026-10-08, as the
  `gpu1` runner logged.
- **`xm9cuo92`'s critic diverged in its last 9k steps.** Its demo value (`q_demo`, the console's
  data value) sat at 0.68–0.72 from 40k to 52.5k, crossed 1 at 54.4k, 10 at 55.5k and 100 at 57.5k,
  and reached 310 by 63.2k, with critic losses of 5–430; its online value rose from 0.02 to 2–6. A
  0/1-reward return cannot exceed 1. No norm in its critic, and its online value stuck at zero for
  50k steps before. **`gkck6eqp`,** the same arm with the env's frames coded, which learned, stayed
  bounded through 96.7k (demo value at most 0.72, online 0.63, critic loss at most 0.0017), and so
  did `496bq48s` (RegularLayerNorm; at most 0.80). So a critic with no norm can stay bounded while
  it learns, and has diverged once while stuck; the runs pre-registered below watch for it.
- **Both packages finalized** with `finalize_run_package`, as the trainer does at a normal exit:
  `xm9cuo92` 949 history rows to 63.2k, `496bq48s` 1,642 rows to 109.4k.
- **`9cf207d resfit-residual-square 2`,** held on `gpu1` since 2026-10-08 16:55 so GPU 1's two slots
  could go to the re-rendered-demos runs, was launched by hand on GPU 0 instead, in `496bq48s`'s
  slot (tmux `resfit-sq2`, from `~/projects/equi-resrl-9cf207d`, 08:37), on Aden's OK. Its `gpu1`
  line stays commented out with a note saying so.

---

## 2026-10-09 — PRE-REGISTERED — the equivariant residual and tuned RLPD on Square, on the re-rendered demos

**Question** (Aden, 2026-10-09): on demos that match the env's observations (2026-10-08 CODE,
"re-rendered demos"), with no norm in the equivariant networks, does the equivariant residual lift
the Square base policy, and does equivariant tuned RLPD learn Square at all? If neither shows
training signal by 100k, the plan regroups: equivariant tuned RLPD made to learn Square first, and
ablations on Can, the residual's norm among them, to set the residual's design (Aden expects some
choices, norm among them, to come out differently for a residual).

**Seed-group `square-rerendered-1009`,** seed 1 of each arm, from the commit that adds this entry,
in a worktree pinned to it. Names `<arm>-<seed>`, notes `square-rerendered-1009/<arm>/seed<seed>`.

**The design, both arms** (Aden's, 2026-10-09): no norm in the actor, the critic or the encoder (the
configs' default); the actor's mean squashed by `equi_tanh`, as every equivariant run since the
fixed-env baselines (`equivariance.residual_squash=true`); 5 critic heads (`agent.critic.num_q=5`)
of one hidden layer each (the default); N = 8. The residual's last layer starts at 1e-4 of escnn's
scale, not at zero: Aden had trouble with the equivariant actor's last layer at exactly zero some
months ago. Shared: Square, mujoco 3.3.2, `offline_data.name=local/robomimic-mh-square-rerendered`,
env frames uncoded, both buffers filled in the run (`read_buffer_caches=false
write_buffer_caches=false`), and `algo.total_timesteps=300000`, the configs' default, as the cap for
"until convergence". Every schedule runs on fixed step counts, not the run length, so a 300k run is
a 150k run through 150k.

| Arm | Config | Overrides beyond the shared ones | wandb project / group |
|---|---|---|---|
| `equi-residual-square-rerendered` | `residual_equi_td3_square_config` | the design, `agent.actor.actor_last_layer_init_scale=1e-4`, `base_policy.wt_version=v9` | `robomimic-square-final` / `equi-residual-rl` |
| `equi-tuned-rlpd-square-rerendered` | `equi_tuned_rlpd_square_config` | the design | `robomimic-square-final` / `tuned-rlpd` |

Everything else is each config's default (the residual: actor lr 1e-6, action scale 0.1, 10k
critic warm-up on the base policy; tuned RLPD: actor lr 1e-4, action scale 1.0, no warm-up).
`tests/test_sweep_arms.py` pins each arm to its config's defaults but for these overrides.

- **What varies.** The residual arm against `kyv7clfc`'s launch (`equi-residual-square` 1, 0.004
  over 110k–150k): the critic's norm (FieldNorm, which erased every trivial input, to none), its
  heads (10 to 5) and depth (2 hidden layers to 1), the demos, and the run length. Five changes: if
  it differs from `kyv7clfc`, this entry cannot say which did it. Against `oj5k0jqb` (ResFiT's
  residual on Square, 0.800 over 110k–150k): the stack and the demos (it trained on the original
  coded demos); one seed each, an observation. Tuned RLPD has no Square reference: no tuned RLPD has
  learned Square in our trainer, and `8ma5alnm` (2026-10-04) ran in the mismatched render. Its
  context is `gkck6eqp`, the same stack learning Can with 10 heads of 2 layers on coded env frames.

**Rules.**
- **Residual, collapse:** stopped at the third eval in a row at or below 0.05. It starts at the base
  policy, so this means it is not training (`kyv7clfc` read 0.00 from 10k).
- **Residual, at 100k:** training iff the mean p of its evals at 60k–100k (5 × 50 episodes) clears
  B_sq = 0.582 by two standard errors, p − 0.582 ≥ 2·sqrt(p(1−p)/250 + 0.582·0.418/500): **at least
  165 of 250 (0.66).** Else stopped. For calibration: `oj5k0jqb`'s evals over the same window are
  0.56 0.64 0.70 0.64 0.76, 165 of 250, so ResFiT's own run meets this with no margin.
- **Tuned RLPD, at 100k:** training iff any eval is at least 0.04 or at least 5 training episodes
  have succeeded. Else stopped.
- **Convergence,** from 150k, at each eval: stopped once the mean of its last 5 evals no longer
  beats the mean of the 5 before by two standard errors (sqrt(p₁(1−p₁)/250 + p₂(1−p₂)/250));
  else run to 300k.
- **Seed 2 of an arm iff it passes its 100k check,** launched from this commit.
- **Verdict** (two seeds): an arm beats the base policy iff the mean over its last 5 evals, both
  seeds (500 episodes), clears B_sq by two standard errors: at least 0.644. The 110k–150k window is
  reported beside it, for comparison with `oj5k0jqb`.
- **A critic that diverges** (`debug/q_data_demo` or `debug/q_data_online` above 1.5; a 0/1-reward
  return cannot exceed 1, and `xm9cuo92`'s critic diverged while stuck, 2026-10-09 NOTE) is reported
  to Aden when seen. It is not a stop rule.
- **Readouts,** in NOTEs at 10k, 50k, the 100k decision and convergence: evals and training
  successes; tuned RLPD's online over demo value over (5k, 10k] (on Can, `gkck6eqp` 0.41 and
  `d047it6c` 0.45; `xm9cuo92`, stuck, 0.03); the residual's `residual_l2` against its action scale,
  and `dQ_da_mean_abs`.

**Predictions** (Claude's, before launch):
1. The residual's step-0 eval is within the base policy's spread (0.46–0.70, its ten evals for
   B_sq). High confidence: its `residual_l1` was 1e-4 at the smoke run's step 100.
2. The residual does not collapse by 30k, unlike `kyv7clfc`. Low-to-moderate confidence: its critic
   now sees every input and its demos match the env, but nothing equivariant has yet improved on a
   base policy in the matching env.
3. The residual passes its 100k check. Low confidence: ResFiT's own run meets it with no margin.
4. Tuned RLPD's online over demo value over (5k, 10k] is above 0.2. Moderate confidence: on Can,
   the frames matching was what kept it up.
5. Tuned RLPD passes its 100k check. Low confidence: no tuned RLPD has learned Square here.

**Checks before the commit.** A 1,000-step smoke run of each arm on GPU 1 exited 0 (packages in
`outputs/runs/equi-resrl-smoke/`: residual `rol02bhz`, tuned RLPD `twrhpz36`). The residual
started at `residual_l1` 1e-4 and its value near 0.18; tuned RLPD's values stayed below 0.7.
`tests/test_sweep_arms.py` passed in full, its GPU tests included (24). Its input-reach test now
reads the actor before the squash and the action scale. Read after them, at escnn's init, the
residual's change with the gripper was 6e-5, under the test's 1e-4. Before them it is 7.5e-4 and
grows in proportion to the change (twice the change, 1.5e-3; ten times, 7.2e-3). So the 6e-5 was the
squash at |mu| up to about 3 and the 0.1 scale, which shrink every input's change alike, not a
blind input.

**Resources.** Both on GPU 1, launched by hand in tmux (`sq-res`, `sq-trlpd`) through `submit.sh`,
which runs the gate, with consoles in `outputs/queue/`. Each first fills its 80,731-frame demo
buffer (about 25 min for the residual, which labels it with the base policy). In the smoke runs,
the residual took 0.35 s per step through its warm-up, alone on GPU 1. Tuned RLPD took 0.51 s per
step while sharing GPU 1 with the residual's buffer fill. Shared, 100k is expected in about a day.
GPU 0 keeps `gkck6eqp` (then its seed 2) and `resfit-residual-square` 2.

---

## 2026-10-09 — LAUNCHED — `square-rerendered-1009` seed 1 (`6ng62tpm`, `vc5o977f`); `resfit-residual-square` 2 (`s8opblsd`)

From the pre-registration above, at its commit `eb90a0c`, in the worktree `~/projects/equi-resrl-eb90a0c`,
by hand on GPU 1 through `submit.sh` (wrapper: console to `outputs/queue/`, then an `EXIT` line).
- **`equi-residual-square-rerendered` 1:** tmux `sq-res`, launched 09:49, gate GO 10:05, in wandb
  11:03 after labelling its demo buffer with the base policy (**`6ng62tpm`**,
  [run](https://wandb.ai/aden-mckinney10-university-of-central-florida/robomimic-square-final/runs/6ng62tpm)).
  Console `outputs/queue/20261009-094910_equi-residual-square-rerendered-1_gpu1.log`.
- **`equi-tuned-rlpd-square-rerendered` 1:** tmux `sq-trlpd`, launched 09:49, in wandb 10:17
  (**`vc5o977f`**,
  [run](https://wandb.ai/aden-mckinney10-university-of-central-florida/robomimic-square-final/runs/vc5o977f)).
  Console `outputs/queue/20261009-094919_equi-tuned-rlpd-square-rerendered-1_gpu1.log`. Step 0: 0 of
  50. At 6.6k, online value 0.15 against demo value 0.56.
- **The gate said GO** for both: Tier 1 206 passed and 2 skipped, Tier 2 372 passed, the render
  check passed. The residual's one warning is the BC policy not being in the worktree's
  `artifacts/` (TODO, the gate's BC-policy check); it downloaded `v9`. Both run packages record
  `eb90a0c`, clean, and mujoco 3.3.2. The cache keys each trainer built are preflight's: residual
  offline `c293a914`, online `1683475f`; tuned RLPD offline `be8abb63`, online `099fa3d6`. Neither
  read nor wrote a cache.
- **`resfit-residual-square` 2** (fixed-env baselines, from `9cf207d`, by hand on GPU 0 in tmux
  `resfit-sq2`, launched 08:37; DEVIATION and NOTE above): gate GO (Tier 1 188 passed and 2
  skipped, Tier 2 385 passed), in wandb 09:49 (**`s8opblsd`**). Evals 0.68 at step 0, 0.22 at 10k
  (seed 1, `oj5k0jqb`: 0.50, then 0.30 at 10k, then 0.800 over 110k–150k).
- **RAM.** Four trainers (three on Square) use 105 of 125 GB at 11:05. A Square run's main process
  grows from about 10 to about 25 GB by 150k (`oj5k0jqb`, `kyv7clfc`, from wandb's system
  metrics), so the machine is expected to run 5–10 GB into swap (225 GB) early on 2026-10-10,
  until `s8opblsd` ends at 150k. Aden was told and left the queue as it is.

---

## 2026-10-09 — AMENDMENT and CODE — the residual's 100k check at one standard error; a watcher enforces the seed group's rules

- **The amendment** (Aden, about 11:25, before the residual's step-0 eval was logged, and so before
  any of its results): the residual passes its 100k check iff its 60k–100k evals clear B_sq by one
  standard error, p − 0.582 ≥ sqrt(p(1−p)/250 + 0.582·0.418/500): **at least 155 of 250 (0.62)**,
  in place of two standard errors (165 of 250). Why: `oj5k0jqb`, ResFiT's own residual, scored
  exactly 165 there, so the pre-registered check would have stopped an equivariant residual that
  learns at all more slowly than ResFiT's. The cost: a residual that only holds the base policy
  passes with about a 16% chance, and gets a seed 2. Every other rule stands.
- **`tools/square_rerendered_watch.py`,** in tmux `sq-watch` from the main checkout at the commit
  that adds this entry, enforces the rules so no one has to watch: SIGINT on the residual's
  collapse, on a failed 100k check, and at convergence from 150k. It logs a passed 100k check as
  due its seed 2, which it does not launch: GPU and RAM need a human. It logs a critic value above
  1.5 once per run, and finalizes the package of a run it stopped. Every action goes to
  `outputs/queue/watch.log`. It reuses `tools/run_watch.py`'s trainer discovery, package lookup and
  log. `tests/test_square_rerendered_watch.py` checks its rules on `oj5k0jqb`'s and `kyv7clfc`'s eval
  sequences: `oj5k0jqb` passes the 100k check and is still rising at 150k, `kyv7clfc` is stopped at
  30k, tuned RLPD is never stopped for collapse, and 155 of 250 passes where 154 does not. Before
  launch it read both live records.

---

## 2026-10-10 — NOTE — `square-rerendered-1009` seed 1: the residual collapsed by 10k; tuned RLPD learned Square to 0.50, then its critic diverged

The NOTEs pre-registered at 10k, 50k and the 100k decision fell overnight and were not written as
they fell; this one gives each readout, from the run packages (`wandb_logs/metrics.jsonl`) and
`outputs/queue/watch.log`. Config caveats: none; every override is live at `eb90a0c` (provenance
table) and `tests/test_sweep_arms.py` pinned both arms.

**`6ng62tpm` (`equi-residual-square-rerendered` 1): collapsed, stopped at 30.3k.**
- Evals 0.62 at step 0, 0.02 at 10k, 0.00 at 20k and 30k. The watcher sent SIGINT at 17:18:50 on
  2026-10-09 by the collapse rule and finalized the package at 17:19:50.
- Training episodes: 8 of 35 succeeded over 0–10k (the critic warm-up, on the base policy), 0 of 68
  over 10k–30k.
- `residual_l1` 0.028 over (5k, 10k] and 0.050 over (25k, 30k], against the action scale 0.1: not
  at its bound. Over 10k–30k it averaged 0.043–0.048, below ResFiT's residual at the same steps
  (`oj5k0jqb` 0.061–0.067), which dipped and recovered. `dQ_da_mean_abs` 8e-5, then 7e-5 (`oj5k0jqb`
  1.0e-4 to 7.5e-5).
- Online value 0.005 over (5k, 10k], −0.007 over (15k, 20k], −0.020 over (25k, 30k], against demo
  value 0.23, 0.25, 0.26: online over demo 0.02 over (5k, 10k] (`oj5k0jqb` 0.14, `s8opblsd` 0.17).
- Predictions: 1 (step-0 within 0.46–0.70) holds; 2 (no collapse by 30k) fails; 3 (passes its 100k
  check) fails, as it never reached it. Same outcome as `kyv7clfc` (0.60, then 0.00 from 10k)
  despite five changes from it (critic norm, heads, depth, demos, length).

**`vc5o977f` (`equi-tuned-rlpd-square-rerendered` 1): learned, passed its 100k check, diverged, stopped at 150k.**
- Evals 0.00 0.00 0.06 0.20 0.26 0.28 0.30 0.44 **0.50** (80k) 0.24 (90k), then 0.00 at every
  eval from 100k to 150k. Training successes per 10k: 0/33, 1/33, 3/35, 9/38, 13/39, 13/42, 20/45,
  21/47, 22/47, 2/34, then 0 of about 33 in each 10k from 100k. **The first tuned RLPD to learn
  Square in our trainer.**
- 100k check passed (watcher, 22:59:48): best eval 0.50 against 0.04, 104 training successes
  against 5. **Seed 2 is due** by the pre-registered rule.
- **The critic diverged.** Demo value 0.58–0.62 and critic loss 2–4e-4 from 5k to 90k. A drift from
  about 86k: critic loss about 1.5×, online value 0.28 to 0.31 by 90k, action magnitude 0.48 to
  0.50. The 90k eval (0.24 against 0.50 at 80k, about 2.8 SE) came in the drift, before any value
  passed 1. Then a runaway from about 92k: a critic value first above 1 at 97.9k, above 1.5 at
  99.1k (the watcher's FLAG, `q_data_demo` 1.83), above 10 at 100.0k, above 100 at 108.7k. Peaks:
  `q_data_demo` 212 and `q_policy_demo` 222 (135.3k), `q_data_online` 25 and `q_policy_online` 27
  (146.2k), critic loss 165 (129.4k). A target here cannot exceed 1 (2026-10-10 MEASUREMENT, below).
  `dQ_da_mean_abs` 3e-5 to 90k, then 1.5–3.2e-3. Gradient-norm clipping at 1.0 was on for the
  actor, critic and encoder throughout (`q_agent.py:503-504`).
- Stopped at 150k by the convergence rule (03:40:01 on 2026-10-10: last five evals 0.000 against
  0.296 for the five before), the rule as written, though the run had collapsed, not converged.
  Package finalized at 03:41:01.
- Readouts: online over demo 0.26 over (5k, 10k] and 0.41 over (45k, 50k]. Prediction 4 (above 0.2
  over (5k, 10k]) holds; prediction 5 (passes its 100k check) holds.
- The second no-norm equivariant critic to diverge: `xm9cuo92` (Can, uncoded frames) diverged while
  stuck at zero (2026-10-09 NOTE); this one diverged after learning. `gkck6eqp` and `juhq4p36`
  (Can, coded frames, 10 heads of 2 layers) stayed bounded while learning. So the 2026-10-09
  NOTE's "a critic with no norm can stay bounded while it learns" held on Can, not here. Task, head
  count and depth, and demos all differ between them.

**The two arms fail differently.** The residual's critic carries no value to online states (its
online value goes negative) and the residual breaks the policy while smaller than ResFiT's, which
does not; tuned RLPD's critic carries value, learns, then runs away. A stabilizer for the second
says nothing about the first.

---

## 2026-10-10 — NOTE — `resfit-residual-square` 2 (`s8opblsd`) finished: the fixed-env baselines' ResFiT residual arm passes; `gkck6eqp` finished

- **`s8opblsd`** (from `9cf207d`, GPU 0, tmux `resfit-sq2`) exited 0 at 21:42 on 2026-10-09, at
  150k. Evals 0.68 0.22 0.28 0.20 0.40 0.44 0.32 0.44 0.68 0.54 0.48 0.60 0.68 0.66 0.58 0.62;
  110k–150k 157 of 250 (0.628). Its best (0.68) is its step-0.
- **The arm's verdict** (`tools/baseline_verdict.py`, the 2026-10-07 rule, both seeds' ten evals at
  110k–150k): **passes**, p = 0.714 (357 of 500), +0.132 on B_sq = 0.582 against the 0.060 of two
  SE. The pass rests on seed 1: `oj5k0jqb` 0.800, `s8opblsd` 0.628, a gap of about 4.3 SE of their
  difference, so the seeds differ by more than binomial noise. Alone, `s8opblsd` is about 1.2 SE
  above B_sq. The weaker seed also carried less value online: online over demo 0.25 over
  (140k, 150k], against 0.47 for `oj5k0jqb`.
- **`gkck6eqp`** (`equi-tuned-rlpd-can-nonorm-democodec` 1, from `e3793ce`) exited 0 at 17:00:31 on
  2026-10-09, at 150k. Evals 0.00 0.94 0.82 0.92 0.96 0.82 0.90 0.98 1.00 0.88 1.00 0.96 0.98 0.94
  0.98 0.96; 110k–150k 241 of 250 (0.964). Its critic stayed bounded: demo value at most 0.73,
  online at most 0.68, critic loss at most 0.0026.

---

## 2026-10-10 — DEVIATION — every other run cancelled: `juhq4p36` stopped at 122.6k, `resfit-tuned-rlpd-can` 2 never launched

Aden's instruction (2026-10-10): stop every running run and run only the four arms planned for
equivariant tuned RLPD on Square plus a fifth (PRE-REGISTERED below, `square-trlpd-stab-1010`).
- **`9cf207d resfit-tuned-rlpd-can 2`** commented out of `~/queues/gpu0.txt` at 09:02, with a dated
  note, before the run ahead of it was stopped, so the `gpu0` runner could not launch it. **The
  fixed-env baselines' ResFiT tuned-RLPD Can arm stays at one seed (`d047it6c`) and has no verdict.**
- **`juhq4p36`** (`equi-tuned-rlpd-can-nonorm-democodec` 2, from `e3793ce`, launched by the `gpu0`
  runner at 17:00:31 on 2026-10-09, in wandb 17:35): SIGINT at 09:02:25, exit 130 at 09:02:28
  (`gpu0.log`), at 122.6k. Package finalized by hand at 09:02:57 (2,578 history rows to 122.6k).
  Evals 0.00 0.68 0.88 0.98 0.94 0.90 0.96 0.82 0.86 0.90 0.94 0.80 0.92 (0–120k); 1,184 of 1,361
  training episodes succeeded; critic bounded (demo value at most 0.68, online at most 0.63, loss at
  most 0.0021). **`critic-norm-can-1008`'s democodec arm has no verdict:** the rule pools both
  seeds' 110k–150k evals against B_can = 0.852 (426 of 500). Seed 1 has 241 of 250; seed 2 had 86 of
  the 100 episodes it reached, and needed 99 of its last 150. Claude advised letting it finish
  (about 12:30; nothing planned could use the GPU before the evening); Aden chose to stop it.
- The `gpu0` and `gpu1` runners keep running with no active line.

---

## 2026-10-10 — AMENDMENT — `square-rerendered-1009`: a divergence stop, before tuned RLPD's seed 2 launches

Aden's decision (2026-10-10), made before `equi-tuned-rlpd-square-rerendered` 2 launched, and so
before any of its results:
- **Divergence stop:** a run is stopped at an eval once every logged critic value in the 5,000 steps
  before it is above 10 (the largest of `debug/q_data_demo`, `q_data_online`, `q_policy_demo` and
  `q_policy_online`, each row) and its last three evals are at or below 0.05. A target here cannot
  exceed 1. Replayed on `vc5o977f`, it fires at 120k (evals 0.00 at 100k, 110k and 120k), 30k steps
  before the convergence rule stopped it; that run never came back from the state.
- **The 1.5 flag** reads `q_policy_demo` and `q_policy_online` too, not only `q_data_*`.
- Every other rule stands. The watcher enforces these once it is rewritten (2026-10-10 CODE, below);
  until then it applies the rules it has.

---

## 2026-10-10 — MEASUREMENT — the TD target cannot exceed 1, on the demos or online

So a clamp of the target to [0, 1] (`agent.clip_q_target_to_reward_range`) never cuts a correct
target, and any critic value above 1 is the critic's own error.
- **The demos:** every re-rendered demo ends with several frames flagged done (Square: 5 in 289
  episodes, 6 in 11; Can: 5 in 292, 6 in 8), and the trainer gives each such transition reward 1.
  Through the trainer's own buffer and `MultiStepTransform` (n = 3, γ = 0.99; `tools/target_bound.py`,
  new), each done ends its own trajectory. The largest n-step reward is 1.0, no rewarded transition
  bootstraps (0 of 80,429 stored on Square, 0 of 62,454 on Can), and the largest target with every
  bootstrapped value at 1 is exactly 1.0. The n-step rewards seen are 0, 0.9801, 0.99 and 1.
- **Online:** robosuite's sparse 0/1 reward, a success terminates the episode, and a timeout cuts the
  bootstrap (`dexmg.py`, `MultiStepTransform`'s `nonterminal`).
- **The lower end:** every reward is at least 0, so a target below 0 is also the critic's error.

---

## 2026-10-10 — LAUNCHED — `equi-tuned-rlpd-square-rerendered` 2 (`6yh4l5mq`), the anchor's seed 2

Due by `square-rerendered-1009`'s rule (its seed 1, `vc5o977f`, passed its 100k check), from that
pre-registration's commit `eb90a0c`, in the worktree `~/projects/equi-resrl-eb90a0c`, by hand on
GPU 1 through `submit.sh` in tmux `sq-trlpd-a`: launched 09:11, gate GO 09:28, in wandb 09:33
(**`6yh4l5mq`**,
[run](https://wandb.ai/aden-mckinney10-university-of-central-florida/robomimic-square-final/runs/6yh4l5mq)).
Console `outputs/queue/20261010-091117_equi-tuned-rlpd-square-rerendered-2_gpu1.log`.
- **The gate said GO:** Tier 1 206 passed and 2 skipped, Tier 2 372 passed, the render check
  passed (Can 0.0103, Square 0.0093 against bars 0.0125 and 0.0129). Its one warning is the disk
  (40 GB free; no cache to write). The run package records `eb90a0c`, clean, and mujoco 3.3.2.
  Cache keys as `vc5o977f`'s (offline `be8abb63`, online `099fa3d6`), neither read nor written.
- **Its rules** are `square-rerendered-1009`'s as amended this morning (the divergence stop, the
  `q_policy_*` flag): the running watcher applies the earlier ones until it is restarted on the code
  of the 2026-10-10 CODE entry below. It is also the anchor of `square-trlpd-stab-1010` (PRE-REGISTERED
  below): whether `vc5o977f`'s runaway repeats.

---

## 2026-10-10 — CODE — a LayerNorm on each critic head's pooled features; the watcher by seed group, with the divergence stop; the critic's start, measured

- **`equivariance.critic_pooled_norm`**, `"none"` (the default) or `"layer_norm"`: `TrivialLayerNorm`
  (`equi_rl_utils.py`), torch's LayerNorm with its per-channel scale and shift, on each critic head's
  pooled features, between `GroupPooling` and the output Linear (`EquiHeadMLP`, `critic.py`).
  - Aden's design, preserved in TODO P3.4 from the deleted `rl_utils.py`, where it never ran. The
    field check is stricter: every field trivial, not merely one-dimensional, since C8 also has a
    one-dimensional sign irrep, which is not invariant.
  - The pooled features are invariant, so the critic stays invariant. The norm bounds each head's Q
    by its own and the output Linear's weights, whatever the input: |Q| ≤ ‖w ⊙ γ‖√d + |w·β| + |b|.
    That is RLPD's reason (Ball et al., 2023) for a LayerNorm before the critic's last layer.
  - At `"none"` the critic is every earlier run's, bit for bit: `test_regression_agent.py` and
    `test_regression_single_arm.py` pass (14). The scalar ablation refuses the field
    (`EQUIVARIANT_ONLY_FIELDS`). ARCHITECTURE.md (provenance table, head diagram) and EQUIVARIANCE.md
    ("Where a norm may go") updated.
- **Tests.**
  - `test_equivariance.py`: the critic is invariant to < 1e-4 with the field on, alone and with
    RegularLayerNorm, in train and eval mode, for one and two arms. `TrivialLayerNorm` is torch's
    LayerNorm and refuses a non-trivial field. The bound holds at input scales 1, 1e3 and 1e6, where
    without the norm Q grows past 1e3 times it.
  - `test_no_ops.py`: one `TrivialLayerNorm` per head, right after its pooling and right before its
    output Linear, and none at the default. And `agent.clip_q_target_to_reward_range` clamps the
    equivariant critic's TD target to [0, 1]: with the target critic made to say 100, every clamped
    target is exactly 1 and the unclamped ones are above 50. No test had covered the clamp on this
    path.
  - `test_config.py`: the default. `tools/agent_fingerprint.py`: two candidates, `pooled-ln` and
    `actor-critic-ln`; `test_sync_free.py`: their updates, and `clipq`'s, make no host sync (8
    passed).
- **`tools/square_rerendered_watch.py`, by seed group:** `--group square-rerendered-1009` or
  `--group square-trlpd-stab-1010` (`GROUPS`), one code path. The second group's four arms follow
  tuned RLPD's rules. It adds the morning's AMENDMENT: the divergence stop, and `q_policy_demo` and
  `q_policy_online` read for it and for the 1.5 flag. Each log line names its group.
  `test_square_rerendered_watch.py`: the stop on synthetic histories (three evals at or below 0.05,
  every row above 10 over the 5k before, the largest of the four values per row); `vc5o977f`'s whole
  record, replayed, is flagged at 99.1k and stopped at 120k; the new arms are never stopped for
  collapse.
- **`tools/critic_start.py`:** the critic's starting spread across heads and its min-of-two
  pessimism, the measurement behind `496bq48s`'s start on 2026-10-08 (its code lived in a
  `tests/test_sweep_arms.py` since rewritten). On a real batch (256 transitions of the re-rendered
  demos, through the trainer's normalizer) or the reference batch. `--match-to` prints the
  last-layer scale that matches a config's spread; scaling the last Linear by s scales every head's
  Q, and both numbers, by s exactly.
- **`tools/rerender_demos.py`: a gate on GPU 0 no longer hangs.**
  - The identity guard's online env runs in its own process (since `01f7c8c`) and inherits the
    wrapper's `MUJOCO_EGL_DEVICE_ID`, which was the tool's fixed `--gpu 1`. robosuite refuses at
    import a device outside `CUDA_VISIBLE_DEVICES`, so under `CUDA_VISIBLE_DEVICES=0`, as
    `submit.sh` sets it for GPU 0, that env died at import. The guard then waited on it forever:
    the parent kept its own copy of the child's end of the pipe, so it never saw the pipe close.
  - Every gate since `01f7c8c` ran on GPU 1, where the guard passes. Found when this entry's Tier 1,
    run on GPU 0, reached its 40-minute timeout in `test_every_guard_passes_on_cans_first_demos`.
  - Now the render device defaults to the trainer's (the first of `CUDA_VISIBLE_DEVICES`, else 1
    as before), an explicit `--gpu` outside it is refused, and a dead online env raises at once. On
    GPU 1 the guard runs as before. It changes only where a check renders, so the built datasets
    are untouched.
  - `test_rerender_demos.py`: `test_the_render_device_is_the_trainers`. On GPU 0 the guard passes
    on Can's first demos: 3 resets and 90 steps, the state exact and every frame bit-identical.
- **`tests/test_sweep_arms.py` and `submit.sh`,** rewritten for `square-trlpd-stab-1010` (PRE-REGISTERED
  below). Each arm composes to the anchor's launch but for its one change (and the two normed arms'
  start). The anchor is what `vc5o977f`'s package says it launched with. The normed arms start as
  spread as the anchor on the pre-registered measurement, and each arm's norms sit where they are
  meant to. 26 Tier-1 tests and the GPU tier pass.

---

## 2026-10-10 — PRE-REGISTERED — stabilizing equivariant tuned RLPD on Square

**Question** (Aden, 2026-10-10): `vc5o977f`, equivariant tuned RLPD on Square, learned to 0.50 at
80k and then its critic ran away (demo value 212 against a ceiling of 1; NOTE above), with
gradient-norm clipping already on. Does clamping the TD target, a LayerNorm on each critic head's
pooled features, or LayerNorm where ResFiT has it keep the critic bounded and the policy learning?
Does `gkck6eqp`'s larger critic, which stayed bounded on Can, avoid the runaway on its own? And does
the runaway repeat at the anchor's design (its seed 2, `6yh4l5mq`, LAUNCHED above)?

**Seed-group `square-trlpd-stab-1010`,** four arms, seed 1 each, from the commit that adds this
entry, in a worktree pinned to it. Names `<arm>-<seed>`, notes `square-trlpd-stab-1010/<arm>/seed<seed>`,
wandb `robomimic-square-final` / `tuned-rlpd`. Two seeds per arm, as every group since 2026-10-07
(STANDARDS 5.3 asks three).

**Shared, `vc5o977f`'s launch exactly:** `--config-name=equi_tuned_rlpd_square_config
equivariance.residual_squash=true offline_data.name=local/robomimic-mh-square-rerendered
algo.total_timesteps=300000 read_buffer_caches=false write_buffer_caches=false seed=<seed>`, and
`agent.critic.num_q=5` except in E: no norm, N = 8, one hidden layer per head, frames uncoded, every
other field at the config's default. `tests/test_sweep_arms.py` pins each arm to the anchor but for
its change, and the anchor to `vc5o977f`'s recorded launch.

| Arm | Change from the anchor | Why |
|---|---|---|
| `equi-tuned-rlpd-square-clipq` | `agent.clip_q_target_to_reward_range=true` | Every correct target lies in [0, 1] (MEASUREMENT above), so the clamp cuts only the critic's error out of what it bootstraps from |
| `equi-tuned-rlpd-square-pooledln` | `equivariance.critic_pooled_norm=layer_norm`, `equivariance.critic_last_layer_init_scale=0.8835` | Bounds each head's Q whatever the input, as RLPD's LayerNorm before the last layer does (CODE above) |
| `equi-tuned-rlpd-square-ln` | `equivariance.actor_norm=layer_norm equivariance.critic_norm=layer_norm`, `equivariance.critic_last_layer_init_scale=0.8168` | RegularLayerNorm wherever ResFiT's `use_layer_norm` puts LayerNorm (as `496bq48s`, on Can): after every hidden Linear of the actor (3) and the critic (visual projection, trunk, each head's one) |
| `equi-tuned-rlpd-square-10x2` | `agent.critic.num_q=10 equivariance.num_critic_layers=2` | `gkck6eqp`'s critic, which stayed bounded while it learned Can |

Aden's decisions (2026-10-10): D covers the actor and critic sites only. ResFiT's ViT encoders have
their own LayerNorms, and the equivariant CNN encoder has no LayerNorm option, so "D + encoder" is a
second-wave candidate if D helps. D keeps one hidden layer per head. The normed arms' starts are
matched to the anchor's. E is included.

**The starts,** `tools/critic_start.py`, real batch (256 transitions of the re-rendered Square
demos), train mode, 20 seeds; ± is the SD across seeds:

| Critic at init | Spread across heads | Min of two heads |
|---|---|---|
| Anchor: no norm, 5 heads × 1 layer (also `clipq`) | 0.731 ± 0.509 | −0.357 |
| `pooledln` unscaled | 0.827 ± 0.269 | −0.450 |
| **`pooledln`, scale 0.8835** | **0.731** | **−0.397** |
| `ln` unscaled | 0.895 ± 0.289 | −0.480 |
| **`ln`, scale 0.8168** | **0.731** | **−0.392** |
| `10x2` (default init, not matched) | 0.588 ± 0.390 | −0.277 |

- The plan named 5 seeds. Over 5 seeds the anchor's spread was 0.689, `pooledln` 0.933, `ln`
  1.089 and `10x2` 0.518, which would have set scales of 0.738 and 0.633. The anchor's spread varies
  by seed with an SD of about 0.5, so a 5-seed mean is uncertain by about ±0.23; 20 seeds were used
  instead.
- On Square the norms move the start far less than RegularLayerNorm did on Can's reference batch
  (2.1× there; 1.13× and 1.22× here). Both are matched, so that each normed arm differs from the
  anchor in its norm alone.
- **At the same seed, `clipq`'s initial weights are the anchor's bit for bit,** and `pooledln`'s and
  `ln`'s are the anchor's exactly but for their norms (identity at init: they draw no random numbers)
  and the scaled last layer. Checked tensor by tensor at seed 1, the seed `vc5o977f` ran with. So
  each of the three starts from `vc5o977f`'s own initial weights; the episodes still differ, since
  `seed` does not reach the envs.

**What varies:** each arm against the anchor (`vc5o977f`, `6yh4l5mq`): its one change, plus the
matched start for `pooledln` and `ln`. Between two arms, two changes: an observation, not a
comparison.

**Rules.**
- **Learning check at 100k,** `vc5o977f`'s: an eval to 100k of at least 0.04, or at least 5
  training successes to 100k. Fail: stopped, no seed 2.
- **Divergence stop:** at an eval, once every logged critic value in the 5,000 steps before it is
  above 10 (the largest of `q_data_demo`, `q_data_online`, `q_policy_demo` and `q_policy_online` in
  each row) and the last three evals are at or below 0.05 (the morning's AMENDMENT, the same rule).
- **Convergence from 150k:** stopped once the mean of its last 5 evals no longer beats the mean of
  the 5 before by two standard errors; else on to 300k.
- **Seed 2 of an arm iff its seed 1 passes the learning check,** from this commit, launched by hand
  into the first free slot. E's seed 1 takes the first slot that frees, then the seed 2s in the
  order their seed 1s passed. The watcher (`--group square-trlpd-stab-1010`) enforces the stops and
  logs which seed 2s are due.
- **Verdict per arm, two seeds.** It *stabilizes* iff both seeds pass the learning check, no
  `q_data_*` or `q_policy_*` value exceeds 1.5 on either seed through 150k, and its pooled
  110k–150k mean holds its pooled 60k–100k mean: p_late ≥ p_mid − 2·sqrt(p_late(1−p_late)/500 +
  p_mid(1−p_mid)/500), 500 episodes each. An arm whose seed 1 fails the learning check has no
  verdict: it did not learn, so there was nothing to stabilize.
- **Reported beside it, deciding nothing:**
  - its pooled 60k–100k mean against the anchor's two seeds (`vc5o977f` 0.296), i.e. whether the
    change slows learning;
  - best eval, and the first step at which any critic value exceeds 1 and 1.5;
  - peak `q_data_*` and `q_policy_*`, and online over demo value over (5k, 10k] (`vc5o977f` 0.26);
  - action magnitude (it rose 0.48 → 0.66 before `vc5o977f`'s runaway), and training successes per
    10k.
- **Reading it:** if the anchor's seed 2 does not diverge by 150k, the runaway is not reliable at
  this design, "no divergence" in an arm is weak evidence, and the window condition carries the
  verdict.

**Predictions** (Claude's, before launch):
1. The anchor's seed 2 (`6yh4l5mq`) has a critic value above 1.5 by 150k. Moderate confidence: one
   of one diverged; the mechanism is unknown.
2. `clipq`'s `q_data_*` stay at or below 1.5 through 150k. High confidence: every target is clamped
   to at most 1. Whether it keeps what it learns: no prediction.
3. `pooledln` has no critic value above 1.5 through 150k, and its pooled 60k–100k mean is within 2
   SE of the anchor's. Moderate confidence.
4. `ln` learns slower than the anchor (pooled 60k–100k below the anchor's by more than 2 SE). Low
   confidence: RegularLayerNorm has not learned in any run of ours, but has only run on uncoded
   frames.
5. `10x2` has no critic value above 1.5 through 150k. Low-to-moderate confidence.
6. At least one arm stabilizes by the verdict. Moderate confidence.

**Resources.**
- Two runs per GPU: about 9.7 GB of GPU memory each (the trainer about 3.8 GB, and each of its nine
  env workers about 0.66 GB, since they render on the GPU), against 23.5 GB.
- About 28 GB of RAM per run at start and 37 GB at 150k: `vc5o977f`'s main process went from 11 to
  19 GB, plus about 17 GB of env workers. So the anchor and three arms at once go about 25 GB into
  swap late on, mostly idle eval workers. A fifth run does not fit.
- Placement: GPU 1 has the anchor (since 09:33) and `clipq`; GPU 0 has `pooledln` and `ln`.
  `10x2` waits for the first slot to free, then the seed 2s.
- The gate's RAM check (`--concurrent-seeds`, 21 GB a run against 85% of MemAvailable) counts the
  runs not yet going, since those already running are out of MemAvailable. `submit.sh` takes it
  from `CONCURRENT` (default 1): 3, 2 and 1 for `clipq`, `pooledln` and `ln`, launched one after
  another, each once the one before is training, then 1 for every later launch. At 4, as
  `submit.sh` was first written, it counted the anchor twice and would have refused every launch
  (84 GB against 0.85 × 91). Its 21 GB a run is below the 28–37 GB measured above, so the swap
  estimate above, not the check, is the one to go by.
- Shared, about 0.55–0.6 s per step, so 100k in about 16 h and 150k in about a day.
- Launched by hand in tmux through `submit.sh`, which runs the gate, from the worktree pinned to
  this commit. Consoles go to `outputs/queue/`.

**Checks before the commit.**
- **Smoke runs:** a 1,000-step smoke run of each arm exited 0, from the main checkout with this
  entry's code uncommitted (packages in `outputs/runs/equi-resrl-smoke/`): `clipq` `prwxyasb`,
  `pooledln` `ya80bsff`, `ln` `0qhqq0p3`, `10x2` `061xa5q5`.
  - 0.38–0.73 s per step, by how many shared GPU 0; main-process RSS 10.2–10.3 GB.
  - Every critic value stayed below 0.7.
  - **`pooledln`'s online value stayed below 0 through the smoke's 700 updates** (−0.08 at step 100,
    −0.17 at 1,000). The anchor's own smoke (`twrhpz36`) and `10x2`'s also started negative but
    crossed 0 by step 500. One short run; its (5k, 10k] online-over-demo readout will say whether
    it lasts.
  - A first `pooledln` smoke (`nsytas18`) died of CUDA out-of-memory, with four smoke runs and the
    GPU tests sharing GPU 0. That is how the per-run GPU memory above was found.
- **Tests:** `tests/test_sweep_arms.py` passed in full (26 Tier-1 tests, and 23 on the GPU,
  including the matched starts and each arm's norm placement). So did the CODE entry's tests, the
  regression goldens (14), the sync-free census of the new candidates (8), and
  `test_square_rerendered_watch.py` (8). Then the gate's two pytest tiers in full on GPU 0: Tier 2
  418 passed (25 min, beside Tier 1), and Tier 1, with `test_square_rerendered_watch.py`, 215
  passed and 2 skipped (3 min), once the re-render guard's hang was fixed (CODE above). The first
  Tier 1 run had stopped at its 40-minute timeout in that guard. `tests/test_sweep_arms.py` passed
  again (49) with `submit.sh` final.

---

## 2026-10-10 — LAUNCHED — `square-trlpd-stab-1010` seed 1: `clipq` (`jdu1nkxw`), `pooledln` (`lclh933n`), `ln` (`0avs073d`)

From `6b2acc6` (PRE-REGISTERED above), in the worktree `~/projects/equi-resrl-6b2acc6`, by hand
through `submit.sh` in tmux, one after another, each once the gate before it had said GO
(`CONCURRENT` 3, 2 and 1; the pre-registration's resources). Consoles in `outputs/queue/`.

| Arm | GPU, tmux | Launched | Gate GO | In wandb | Run |
|---|---|---|---|---|---|
| `equi-tuned-rlpd-square-clipq` 1 | 1, `sq-trlpd-b` | 11:02 | 11:32 | 11:39 | [**`jdu1nkxw`**](https://wandb.ai/aden-mckinney10-university-of-central-florida/robomimic-square-final/runs/jdu1nkxw) |
| `equi-tuned-rlpd-square-pooledln` 1 | 0, `sq-trlpd-c` | 11:32 | 12:01 | 12:06 | [**`lclh933n`**](https://wandb.ai/aden-mckinney10-university-of-central-florida/robomimic-square-final/runs/lclh933n) |
| `equi-tuned-rlpd-square-ln` 1 | 0, `sq-trlpd-d` | 12:02 | 12:32 | 12:40 | [**`0avs073d`**](https://wandb.ai/aden-mckinney10-university-of-central-florida/robomimic-square-final/runs/0avs073d) |

- **The gate said GO for each:**
  - Tier 1: 207 passed and 2 skipped.
  - Tier 2: 417 passed and 1 skipped, in 25–27 min.
  - Render check passed: Can 0.0098–0.0102 and Square 0.0081–0.0089, against bars of 0.0125 and
    0.0129.
  - RAM headroom: 98, 70 and 46 GB available, against 63, 42 and 21 GB.
  - Its one warning each: the disk (39 GB free; no cache to write).
  - The Tier 2 skip is `test_the_anchor_is_what_vc5o977f_launched_with`. It looked for
    `vc5o977f`'s run package under the worktree, which has no `outputs/`. It passed from the main
    checkout at this code (PRE-REGISTERED, the checks before the commit), and from `ae39bd9` it
    looks under `CACHE_DIR`.
- **Run packages:** each records `6b2acc6`, clean, and mujoco 3.3.2, and its argv carries its arm's
  change: `clipq`'s run name ends `_clipT`; `pooledln` has `critic_pooled_norm=layer_norm` and
  scale 0.8835; `ln` has both norms and scale 0.8168. Each step-0 eval is 0.00, as for every tuned
  RLPD run (no base policy).
- **Watchers,** both run from the worktree:
  - `sq-watch` (`--group square-rerendered-1009`) was restarted at 11:01 on this commit's code, so
    the anchor (`6yh4l5mq`) now has the divergence stop and the `q_policy_*` flag.
  - `sq-watch2` (`--group square-trlpd-stab-1010`) has run from 11:01, and tracked each arm within a
    minute of its run package.
- **Resources at 12:40,** with the four runs up: 100 GB of RAM used, 23 GB available, swap
  untouched; GPU 0 16.6 GB, GPU 1 19.7 GB.
- **`10x2`** waits for the first slot to free, then the seed 2s, as pre-registered.

---

## 2026-10-11 — DEVIATION and NOTE — `6yh4l5mq` and `lclh933n` stopped by hand past 150k, for the residual's RAM; `0avs073d` converged; `10x2`'s gate said NO-GO

**The deviation.** Aden's decision (2026-10-11, about 19:10), to free RAM for two equivariant
residual runs (PRE-REGISTERED below), which need about 20 GB each at start and 35 GB by 150k, with
15 GB available. Claude laid out four options (stop pooledln and the anchor's seed 2, stop all
three, stop pooledln only, stop nothing and queue); Aden chose the first, which Claude recommended,
and `jdu1nkxw` (`clipq`) runs on.
- **`6yh4l5mq`** (`equi-tuned-rlpd-square-rerendered` 2): SIGINT at 19:11:36, at 194.3k; package
  finalized by the watcher at 19:12:32. Evals 0.00 0.00 0.14 0.06 0.28 0.24 0.18 0.28 0.28 0.28
  0.42 0.56 0.38 0.52 0.58 0.54 0.66 0.66 0.64 0.60 (0–190k). Every critic value at most 0.704.
- **`lclh933n`** (`equi-tuned-rlpd-square-pooledln` 1): SIGINT at 19:11:36, at 186.3k; finalized
  at 19:11:42. Evals 0.00 0.00 0.00 0.00 0.04 0.04 0.06 0.20 0.12 0.06 0.18 0.16 0.20 0.38 0.40
  0.30 0.38 0.24 0.40 (0–180k). Every critic value at most 0.708.
- **What it costs:** neither reached its convergence stop, so neither has a "converged at" step or
  a final value. Every number their verdicts use was logged by 150k: the learning check (both
  passed), the critic bound through 150k, and the 60k–100k and 110k–150k windows. Both stops are in
  `outputs/queue/watch.log` as `HUMAN STOP`.

**The rest of `square-trlpd-stab-1010` and `square-rerendered-1009` since launch** (`watch.log`):
- **The 100k checks:** `6yh4l5mq` passed at 02:00 (best 0.42, 86 training successes), `lclh933n`
  at 05:33 (best 0.20, 61), `jdu1nkxw` at 05:54 (best 0.52, 117), `0avs073d` at 06:36 (best 0.24,
  13). So `clipq`, `pooledln` and `ln` are each due a seed 2 by the rule. None has launched.
- **`0avs073d`** (`equi-tuned-rlpd-square-ln` 1) stopped by the convergence rule at 160k (17:21:30;
  last 5 0.172 against 0.152 for the 5 before, 2 SE 0.066). Evals 0.00 0.00 0.00 0.02 0.00 0.00
  0.02 0.06 0.04 0.24 0.18 0.24 0.14 0.14 0.16 0.18 0.24. Its critic's largest value was 1.10,
  under the 1.5 flag.
- **`10x2` did not launch.** Its gate (GPU 0, tmux `sq-trlpd-e`, 17:35) said NO-GO on RAM headroom
  alone: 24 GB available, against 21 GB for one run (0.85 × 24 = 20.4). Console
  `outputs/queue/20261011-173536_equi-tuned-rlpd-square-10x2-1_gpu0.log`.
- **No critic diverged.** No run in either group logged a value above 1.5, the runaway of
  `vc5o977f` did not repeat in `6yh4l5mq`, and `clipq`'s ceiling never had a target to cut.
- `10x2` and the three seed 2s wait for RAM after the residual runs below. The verdicts and the
  analysis of the group are not written here.

---

## 2026-10-11 — PRE-REGISTERED — the equivariant residual on Square: its actor's last layer at zero, and a 50k critic warm-up

**Question** (Aden, 2026-10-11): `6ng62tpm`, the equivariant residual on Square, scored 0.62 at
step 0 and 0.02 at 10k (2026-10-10 NOTE). Its critic carried no value to the online states: online
value 0.005 over (5k, 10k] against demo value 0.23, while its own online episodes over 0–5k earned a
discounted return of 0.127 (7 of 18 succeeded; `metrics.jsonl`). Is the actor moving before the
critic can tell a good residual from a bad one? Two arms, each one change from `6ng62tpm`:
1. **The actor's last layer at exactly zero,** in place of 1e-4. The 1e-4 start came from `z8yoqylh`
   (1e-4, learned Can) against `a3e3zylp` (0.0, collapsed), which also differed in host and
   software stack, so the start was never isolated. 0.0 is the config's default and ResFiT's, and
   makes step 0 exactly the base policy (ARCHITECTURE.md assumed it). The fixed-env baselines
   pre-registered this arm (`equi-residual-square-zero-init`, 2026-10-07). It was held on
   2026-10-08 with that batch's blind FieldNorm critic and never ran, so this is its first run.
2. **A longer critic warm-up:** 50,000 critic-only updates before the actor's first, in place of
   10,000. 

**Why 50k** (Claude's choice; Aden asked for "longer"):
- At the end of `6ng62tpm`'s 10k warm-up updates, the mean TD target was still rising: −0.036
  over its first 2,000 updates, 0.044 over 6k–8k, and 0.062 over its last 2,000 (means of the
  single-batch values printed every 100 updates, console log). ResFiT's residual warm-ups also
  ended still rising, from lower (`oj5k0jqb` −0.107, 0.036, 0.059; `s8opblsd` −0.322, −0.037,
  0.002), so this alone does not separate the two stacks.
- Value from a success has to reach states about 240 steps earlier, 3 steps per bootstrap, through a
  target that follows the critic with a time constant of 200 updates (τ = 0.005). That is about 80
  bootstraps, so on the order of 16k updates or more. 50k leaves room for that, at 0.6–1.6 h more
  warm-up: 0.148 s per update for `6ng62tpm`, sharing GPU 1, and 0.074 s alone on GPU 0 (the smoke
  below).
- **The risk:** the warm-up's online data is fixed: 10k transitions from 35 base-policy episodes with
  noise, 9 of them successes (`6ng62tpm`'s). 50k updates make about 640 passes over it (10k, about
  130), so a longer warm-up can also fit those 35 episodes too closely.

**Seed-group `square-residual-start-1011`,** two arms, seed 1 each, from the commit that adds this
entry, in a worktree pinned to it. Names `<arm>-<seed>`, notes
`square-residual-start-1011/<arm>/seed<seed>`, wandb `robomimic-square-final` / `equi-residual-rl`.

**Shared, `6ng62tpm`'s launch exactly:** `--config-name=residual_equi_td3_square_config
equivariance.residual_squash=true agent.critic.num_q=5 base_policy.wt_version=v9
offline_data.name=local/robomimic-mh-square-rerendered algo.total_timesteps=300000
read_buffer_caches=false write_buffer_caches=false seed=<seed>`, and
`agent.actor.actor_last_layer_init_scale=1e-4` but in `init0`. That is: no norm anywhere, 5 critic
heads of one hidden layer, N = 8, actor lr 1e-6, action scale 0.1, τ = 0.005, the actor updated once
per 4 critic updates, frames uncoded, every other field at the config's default.
`tests/test_sweep_arms.py` pins each arm to the anchor but for its change, and the anchor to
`6ng62tpm`'s recorded launch.

| Arm | Change from `6ng62tpm` |
|---|---|
| `equi-residual-square-init0` | `agent.actor.actor_last_layer_init_scale=0.0` |
| `equi-residual-square-warmup50k` | `algo.critic_warmup_steps=50000` |

- **What varies:** each arm against `6ng62tpm`, its one change. At this commit, `6ng62tpm`'s launch
  composes to its recorded config except for `equivariance.critic_pooled_norm`, a field added since
  `eb90a0c`, at its default `none`, which is bit-identical to every earlier run
  (`tests/test_no_ops.py`).
- **At seed 1, each arm starts from `6ng62tpm`'s initial weights.** `warmup50k`'s match tensor for
  tensor; `init0`'s match too, but for its actor's last Linear, whose weights and bias are zero in
  the actor and the actor target alike (`tests/test_sweep_arms.py`). The episodes still differ,
  since `seed` does not reach the envs.
- **Between the two arms:** two changes, so an observation, not a comparison.
- **Context, not a comparison:** ResFiT's residual on Square (`oj5k0jqb`, `s8opblsd`), whose last
  layer starts at 0.0 with a 10k warm-up, dipped by 10k (0.30, 0.22) and recovered. It differs in
  stack and in its demos.

**Rules,** `square-rerendered-1009`'s residual rules as amended, enforced by the watcher
(`--group square-residual-start-1011`):
- **Collapse:** stopped at the third eval in a row at or below 0.05.
- **Divergence stop:** the 2026-10-10 AMENDMENT's.
- **At 100k:** passes iff its evals at 60k–100k reach at least 155 of 250, one standard error over
  B_sq = 0.582. Else stopped.
- **Convergence from 150k:** stopped once the mean of its last 5 evals no longer beats the mean of
  the 5 before by two standard errors; else on to 300k.
- **Seed 2 of an arm iff it passes its 100k check,** from this commit.
- **Verdict (two seeds):** an arm beats the base policy iff the mean of its last 5 evals, both seeds
  (500 episodes), is at least 0.644. The 110k–150k window is reported beside it.
- **A critic value above 1.5** is reported to Aden. It is not a stop rule.

**Readouts,** in NOTEs at the 30k collapse decision and at 100k:
- Evals, and training successes per 10k.
- Online over demo value over (0, 5k] and (5k, 10k]: `6ng62tpm` 0.02 over (5k, 10k]; `oj5k0jqb`
  0.14, `s8opblsd` 0.17. Beside it, the realized discounted return of the training episodes over the
  same steps (`6ng62tpm` 0.127 and 0.015; `oj5k0jqb` 0.035 and 0.018).
- `warmup50k`'s online and demo values through its warm-up, at 10k, 20k, 30k, 40k and 50k updates,
  from its console (CODE, below). This says whether the online value was still rising at 10k, and
  where it levels off.
- `residual_l1` against the action scale (0.1), and `dQ_da_mean_abs`.

**Reading it** (decided now):
- **`warmup50k` holds the base policy past 30k and `init0` collapses:** the critic's readiness
  matters. Next: the actor gated on the critic's online value levelling off, rather than on a
  fixed update count, and `warmup50k`'s seed 2.
- **`warmup50k`'s online value rises through its warm-up, but the run still collapses:** readiness
  is not enough. Next: a penalty on the residual's size, which holds it near zero while the critic's
  gradient is small (Adam takes full-size steps on a gradient of any size).
- **`warmup50k`'s online value stops rising by about 10k updates:** more critic-only updates on this
  data do not carry value online. The problem is in the critic or its data, not the actor's timing.
- **`init0` holds the base policy past 30k:** the 1e-4 start matters, which is unexpected, and
  needs a seed 2 of both arms. **`init0` collapses:** the 1e-4 start is not the cause, and later
  residual arms start at 0.0, as ResFiT's do.

**Predictions** (Claude's, before launch):
1. `init0`'s step-0 eval is within the base policy's spread (0.46–0.70). High confidence: its
   residual is exactly zero (`tests/test_sweep_arms.py`).
2. `init0` collapses by 30k, as `6ng62tpm` did. Moderate-to-high confidence. Adam's step size does
   not depend on the gradient's size, so the start is forgotten within tens of actor updates.
   `6ng62tpm`'s `residual_l1` was 0.0006 over 0–1k and grew by 0.002–0.006 per 1k after that, to
   0.038 over 9k–10k.
3. `warmup50k`'s online value is higher at 50k warm-up updates than at 10k. Moderate confidence.
4. `warmup50k`'s online over demo value over (5k, 10k] of training is above 0.1. Low-to-moderate
   confidence.
5. `warmup50k` does not collapse by 30k. Low confidence.
6. Neither arm passes its 100k check. Moderate confidence.

**CODE** (in this commit): the warm-up's progress line (`train_residual_td3.py`, every 100 updates)
now prints the critic's value on each buffer, the `critic_diagnostics` the training loop logs as
`debug/q_*`. It is logging only: `tests/test_critic_diagnostics.py` checks that the call moves no
parameter and draws no random number. Console only, since wandb's step is 0 through the warm-up.
The run package keeps it in `wandb_logs/output.log`.

**Checks before the commit.**
- `tests/test_sweep_arms.py` passed in full: 14 Tier-1 tests, and 14 on the GPU. Three of the GPU
  tests are new: the starts above; that `init0`'s residual is exactly zero at init and moves off
  zero at its first actor update, with finite losses; and no norm and five one-layer heads in each
  arm.
- `tests/test_square_rerendered_watch.py` passed (9): the new group follows the residual's rules on
  `kyv7clfc`'s and `oj5k0jqb`'s evals. Every other Tier-1 test passed too (241, and 2 skipped).
- **Smoke run of `init0`** (`SMOKE=1`, GPU 0, from the main checkout with this entry's code
  uncommitted; console `outputs/queue/20261011-191614_smoke-equi-residual-square-init0-1_gpu0.log`).
  It labelled the demos and ran its critic warm-up with the new line: by update 1,100, demo value
  −0.03 to 0.04 and online −0.09 to −0.06. The online buffer holds only the smoke's 300 transitions,
  so these are not a readout. The data and policy values match, since the residual is zero. 0.074 s
  per update, alone on the GPU. `warmup50k` has no smoke of its own: it differs only in the
  warm-up's loop count.

**Resources.**
- After the DEVIATION above, `clipq` runs alone on GPU 1, GPU 0 is free, and 87 GB of RAM is
  available.
- **Placement:** `warmup50k` on GPU 0, alone through its long, GPU-bound warm-up. `init0` on GPU 1
  with `clipq`, as `6ng62tpm` shared GPU 1 with `vc5o977f`.
- **RAM:** about 20 GB per residual run at start and 35 GB by 150k, plus `clipq` at about 33 GB,
  so about 105 GB of 125 at 150k. Both launch at once, so each gate's RAM check counts both
  (`CONCURRENT=2`).
- **Timing:**
  - Each run labels its 80,731 demo frames with the base policy (about 27 min).
  - It then fills 10k warm-up steps, and runs its critic warm-up: about 25 min for `init0`, about
    1–2 h for `warmup50k`.
  - After that, about 0.69 s per step with its eval, sharing a GPU (`6ng62tpm`: 1.9 h per 10k).
    So the 30k collapse decision comes about 7.5 h after launch for `init0`, and about 9 h after
    for `warmup50k`.
- **What waits:** `10x2` and the seed 2s of `square-trlpd-stab-1010`.

---

## 2026-10-11 — LAUNCHED — `square-residual-start-1011` seed 1: `init0` (`eaqorqki`), `warmup50k` (`doevaarp`)

From `0af7655` (PRE-REGISTERED above), in the worktree `~/projects/equi-resrl-0af7655`, by hand
through `submit.sh` in tmux, both at 19:44, each gate counting both (`CONCURRENT=2`). Consoles in
`outputs/queue/`, `20261011-194445_<arm>-1_gpu<N>.log`.

| Arm | GPU, tmux | Gate GO | In wandb | Run |
|---|---|---|---|---|
| `equi-residual-square-init0` 1 | 1 (with `clipq`), `sq-res-init0` | about 20:10 | 21:04 | [**`eaqorqki`**](https://wandb.ai/aden-mckinney10-university-of-central-florida/robomimic-square-final/runs/eaqorqki) |
| `equi-residual-square-warmup50k` 1 | 0, `sq-res-wu50k` | about 20:10 | 20:40 | [**`doevaarp`**](https://wandb.ai/aden-mckinney10-university-of-central-florida/robomimic-square-final/runs/doevaarp) |

- **The gate said GO for both:**
  - Tier 1: 207 passed and 2 skipped. Tier 2: 397 passed, in 17–18 min.
  - Render check passed: Can 0.0103 and 0.0105, Square 0.0099 and 0.0088, against bars of 0.0125
    and 0.0129.
  - RAM headroom: 87 GB available against 42 GB.
  - Warnings: the BC policy is not in the worktree's `artifacts/` (it downloads `v9`), and the disk
    has 36 GB free (no cache to write).
- **Run packages:** each records `0af7655`, clean, and mujoco 3.3.2. Each argv is the
  pre-registered one: `init0` carries `actor_last_layer_init_scale=0.0`; `warmup50k` carries `1e-4`
  and `critic_warmup_steps=50000`.
- **The warm-up buffers differ by chance.** The envs are not seeded, and each run fills its 10k
  base-policy steps with noise afresh:
  - `init0` 7 successes of 35 episodes;
  - `warmup50k` 16 of 36;
  - `6ng62tpm` 9 of 35.

  So `warmup50k`'s critic fits about twice the online successes the other two do. This weakens
  the comparison of its online value with `6ng62tpm`'s, for the warm-up and the first 10k steps of
  training. The 30k NOTE reads both against each run's own realized return.
- **Speed:** critic warm-up updates at 0.074 s alone on GPU 0 (`warmup50k`), and 0.158 s on GPU 1
  sharing with `clipq` (`init0`).
- **The smoke run** (`qyqm14qf`, PRE-REGISTERED, checks) finished, exit 0, at 1,000 steps. Its
  residual left zero and grew to `residual_l1` 0.0013 by step 1,000, with every loss finite.
- **Watcher:** `sq-watch3` (`--group square-residual-start-1011`, from the worktree) started at
  19:44:50, and tracked `doevaarp` at 20:40:53 and `eaqorqki` at 21:05:55. `sq-watch` (its group
  `square-rerendered-1009` now has nothing running) and `sq-watch2` run on.

---

## 2026-10-12 — DEVIATION — every running experiment cancelled: `eaqorqki` stopped at 80.2k, and the waiting tuned-RLPD arms dropped

Aden's instruction (2026-10-12, about 08:45), with the plan that follows (FINDING and
PRE-REGISTERED below): cancel every other experiment, since its lessons are in.
- **`eaqorqki`** (`equi-residual-square-init0` 1, `square-residual-start-1011`): SIGINT at
  08:50:04, at 80.2k; package finalized by the watcher at 08:50:09. Evals 0.50 0.00 0.00 0.06 0.06
  0.04 0.16 0.24 0.30 (0–80k). **What the stop costs:** no verdict. The run's 60k–100k window had
  35 of the 150 episodes it reached, so even 100 of the last 100 would give 135, short of the 155
  its 100k check needs. It does leave the recovery after 80k unseen (0.16, 0.24, 0.30 at
  60k–80k).
- **`doevaarp`** (`equi-residual-square-warmup50k` 1) had already been stopped by its 100k check
  at 07:35:55 (30 of 250; NOTE to follow).
- **`square-trlpd-stab-1010`'s `10x2` 1 and the seed 2s of `clipq`, `pooledln` and `ln`,** which
  waited for RAM, are dropped: none will launch. That group's arms keep one seed each and no
  verdict. `jdu1nkxw` (`clipq` 1) was stopped by its convergence rule at 230k (04:17:23; last five
  0.628 against 0.588 for the five before); every other run of the group had stopped by 2026-10-11.
- **The watchers** `sq-watch` (`square-rerendered-1009`), `sq-watch2` (`square-trlpd-stab-1010`)
  and `sq-watch3` (`square-residual-start-1011`) have nothing left to watch, and are stopped.

---

## 2026-10-12 — FINDING — the equivariant residual's collapse on Square is a wrist that rolls after the grasp, carried by about three of its seven action components

From the three equivariant residual runs on the re-rendered Square demos, seed 1 each:
`6ng62tpm` (the 1e-4 start), `eaqorqki` (init0) and `doevaarp` (warmup50k). ResFiT's `oj5k0jqb` and
`s8opblsd` are context. Figures and the scripts that made them:
`outputs/diagnostics/residual-collapse-1012/` (outside git).

**Method.**
- **Splitting the videos.** Each eval video holds its 50 episodes in completion order, every
  frame labelled SUCCESS or FAIL, and a failure is exactly 300 frames. So each video was split at
  the label and, within runs of successes, at the reset's frame jump (`sheets.py`). Every split
  matched its eval's count, for example 7,200 FAIL frames, 24 × 300, for `doevaarp`'s step-0 0.52.
- **The contact sheets:** each episode's last frame before the reset (the done step's frame
  already shows the next episode), and frames through time.
- **The residual histogram.** `histograms/residual_actions` pools 256 × 7 values per logged
  update. Its mass is read as components' worth near zero (|r| < 0.02) and near the 0.1 bound
  (|r| > 0.06) (`hist.py`).
- **Training episodes** per 1k steps from `training/episode_return` (`bins.py`).

**What was found.**
- **The training curve's early peak is the base policy, not learning.**
  - Training episodes pooled over the three runs: 17 of 33 over 0–3k (0.52), 5 of 21 over 3–5k,
    and 3 of 51 over 5–10k.
  - Through 3k every residual value in the batch is under 0.02.
  - `residual_l1` grew at the same rate in all three to 5k, whatever the start or the warm-up:
    0.003 at 2k, 0.006–0.007 at 3k, 0.010–0.011 at 4k.
  - The critic's online value rose and fell with the episodes. `doevaarp`: 0.064–0.085 over
    1k–4k, then about 0 by 15k.
- **The collapse is a behavior the base never shows.**
  - In the 10k and 20k evals of all three runs, the approach and the grasp look like the base's
    (steps 0–40).
  - From about step 70 the wrist rolls a little more each step. By step 160 the gripper points
    sideways at the camera, holding the nut away from the peg.
  - In the step-0 evals the gripper stays vertical in every episode. The base's failures are
    missed or repeated grasps at the nut, and nuts held next to the peg.
- **About three of the seven components carry the residual.**
  - At 10k, components' worth near zero: 3.2, 3.9 and 2.5 (`6ng62tpm`, `eaqorqki`, `doevaarp`).
    Near the bound: 2.5, 2.8 and 3.1.
  - A gripper pointing sideways needs rotation about a horizontal axis, which only the tilt pair
    (rx, ry) commands; yaw cannot do it. That attribution is an inference, since no metric splits
    the residual by field.
- **The recovery tracks the residual pulling back.**
  - `doevaarp`: near the bound 3.1 at 10k, 0.5 at 30k; `residual_l1` 0.050 at 12k, 0.021 at 30k.
    Its training successes came back from 24k, and its 30k eval (0.22) has about half its
    episodes back on the base's vertical approach.
  - `eaqorqki` pulled back less (2.9 at 15k, 1.4 at 30k) and recovered later (0.16, 0.24, 0.30
    at 60k–80k).
  - `6ng62tpm` had not pulled back by 30k (0.052, still rising) when it was stopped.
- **A second failure mode, later.**
  - `doevaarp`'s residual grew again (0.028 at 40k, 0.049 at 100k; 2.75 components near the
    bound), at evals of 0.10–0.18.
  - At 100k the roll is rare, about 3 of 45 failures. Most fail at the grasp: the gripper reaches
    the nut, often turned differently from the base's approach, pushes it about at table height
    and never lifts it.
- **ResFiT's residual is larger and saturated, and works.**
  - `oj5k0jqb`'s `residual_l1` is 0.057 at 10k, and its components near the bound number 3.5 at
    10k and 5.8 of 7 at 100k, in its normalized units.
  - Its 10k dips (0.30 and 0.22) are mostly the base's own failures, with a few rolls.
  - So the size of the residual alone does not explain the collapse.

**Context, not measured here.**
- OSC_POSE actions are per-step deltas, so a rotation residual adds up every step: 0.1 env units
  is about 3° per step at robosuite's default output scale of 0.5 rad.
- The demos barely tilt: per-step tilt length median 0.030, 90th percentile 0.083.
- ResFiT's residual cap is 0.036 env units on rx (0.1 × 1.1 × its demos' half-range of rx) and
  0.11 on ry. The equivariant stack's tilt pair is a disk of radius 0.1 in env units.

**Side finding: the eval's Q is the critic at the residual alone, on the equivariant residual
path.**
- `run_dexmg_evaluation` scores the critic at `agent.act()`'s output
  (`evaluate_dexmg.py:190-200`). On that path the output is the residual: the wrapper adds the
  base action itself (`residual_env_wrapper.py:136`).
- So the "Q =" overlay and `eval_q_trajectories_*.png` of every equivariant residual run show the
  critic at about zero action, not at the action executed.
- Training is unaffected, and so are tuned RLPD and ResFiT. A CORRECTION follows with the fix.

**The penalty suggested overnight is held, and why.** `square-residual-start-1011`'s reading
"readiness is not enough" named a penalty on the residual's size as next.
- `debug/dQ_da_mean_abs` is the gradient of `-q.mean()` (`q_agent.py:546`, `:608`), so it is
  already divided by the batch of 256. The per-sample slope is about 256 × 7e-5 ≈ 0.018.
- The penalty `w · mean(sum(r²))` balances that slope at r = slope / (2w). Holding the residual
  near 0.005 takes w ≈ 1.8, not 0.007.
- ResFiT's slope is about the same size, so the same weight would hold ResFiT's useful residual
  near 0.005 as well, against the 0.06–0.08 it works at. The penalty filters by the gradient's
  size, which does not separate good from bad residuals here.

**Caveats.** One seed per arm; the contact sheets were read by eye; the per-field attribution is
an inference. The next two entries test it: the equivariant stack's tilt range given to ResFiT,
and then the equivariant residual without tilt, with a field-by-field test at a collapsed
checkpoint.

---

## 2026-10-12 — PRE-REGISTERED — ResFiT's residual on Square with rx at the equivariant stack's range

**Question** (Aden, 2026-10-12):
- The equivariant residual's collapse on Square rolls the wrist after the grasp (FINDING above).
- Its tilt pair is bounded at 0.1 env units in any direction. ResFiT's rx is bounded at 0.036,
  because its demos never use a wide rx range.
- Given the equivariant stack's rx range, does ResFiT's residual start rolling the wrist, or
  collapse?

**Seed group `resfit-rx-1012`.**
- **Arm** `resfit-residual-square-rx`, seed 1, from the commit that adds this entry, in a
  worktree pinned to it. Name `resfit-residual-square-rx-1`, notes
  `resfit-rx-1012/resfit-residual-square-rx/seed1`, wandb `robomimic-square-final` /
  `residual-rl`.
- **Launch:** `oj5k0jqb`'s, plus one override. `--config-name=residual_td3_square_config
  agent.critic.num_q=10 base_policy.wt_version=v9 offline_data.min_action_range=2.0
  algo.total_timesteps=150000 read_buffer_caches=false write_buffer_caches=false seed=1
  debug=false`.

**What varies: rx's normalization range, and nothing else.**
- `ActionScaler` floors each component's half-range at `min_action_range / 2`, then widens it by
  1.1 (`utils/normalization.py:63-71`). In these demos (`ankile/robomimic-mh-square-image`) every
  component but rx spans exactly [−1, 1], and rx spans [−0.448332, 0.208986].
- So a floor of 2.0 raises rx's half-range from 0.3287 to 1.0 and leaves every other component
  bit for bit (`tests/test_sweep_arms.py`). Its midpoint, which cancels out of a residual, stays
  at −0.120.
- In env units, that one change moves three things together, which is what the equivariant stack
  does for rx:

  | | `oj5k0jqb` | this arm |
  |---|---|---|
  | rx residual cap | 0.036 | 0.110 |
  | rx exploration and target noise (0.05 in normalized units) | 0.018 | 0.055 |
  | rx's normalization range | [−0.481, 0.242] | [−1.220, 0.980] |

**Held fixed.**
- Everything else, the demos included: the original coded ones, as the anchors'.
- At this commit, the anchor's launch composes to `oj5k0jqb`'s recorded config. The one field
  added since `9cf207d` is `encode_env_frames_like_demos`, at its no-op default `False` (test).
- ResFiT's code path has changed since `9cf207d` only by the warm-up's `critic_diagnostics` call
  (`train_residual_td3.py:1139`). `test_critic_diagnostics_changes_nothing` shows it moves no
  parameter and draws no random number, ResFiT's agent included.
- At the same seed the arm's initial weights are the anchor's, bit for bit (test). The episodes
  still differ, since `seed` does not reach the envs.

**Context: the anchors, the fixed-env baselines' ResFiT residual arm.**
- `oj5k0jqb` scored 0.800 over 110k–150k, and `s8opblsd` 0.628.
- At 10k, 20k and 30k: 0.30, 0.58, 0.54 (`oj5k0jqb`) and 0.22, 0.28, 0.20 (`s8opblsd`). Pooled,
  106 of 300 (0.353).
- The two seeds differ by about 4 SE, so one seed of this arm is an observation.

**Rules** (`tools/square_rerendered_watch.py --group resfit-rx-1012`):
- **Stop:** at the third eval in a row at or below 0.05. Otherwise the run goes to 150k, as its
  anchors did.
- **No 100k check:** `s8opblsd`'s 60k–100k evals (123 of 250) would fail the equivariant
  residual's.
- **Seed 2** iff it collapses, or its 10k–30k evals pooled (150 episodes) fall below the anchors'
  pooled 0.353 (300 episodes) by two standard errors.

**Readouts,** in NOTEs at 30k and at 150k:
- Evals and training successes per 10k.
- `residual_l1`, and the pooled histogram's components near zero and near the bound, as in the
  FINDING above.
- **The rolled share, scored blind.**
  - Material: the last frame before the reset of every failed episode in the 10k, 20k and 30k
    evals of this run and of both anchors.
  - Procedure: shuffled into one sheet with the run and step hidden, the key kept aside, scored
    by Claude as rolled or not, then unblinded.
  - "Rolled": the gripper visibly points off vertical, with the wrist's round cap facing the
    camera.

**Reading it** (decided now):
- **It rolls more than its anchors, and its evals fall below theirs:** the wide tilt range alone
  causes the roll, even with ResFiT's critic. The equivariant stack's tilt bound is then part of
  the fix.
- **It is like its anchors:** a wide tilt range is harmless with a critic that works. The
  equivariant residual's problem is then its critic's gradient on tilt, not the range.

**Predictions** (Claude's, before launch):
1. Its step-0 eval is within the base's spread, 0.46–0.70. High confidence: its residual starts
   at exactly zero.
2. It does not collapse by 30k. Moderate confidence: its ry is already at 0.11, and ResFiT does
   not roll.
3. Its rolled share is within its anchors' range. Low to moderate confidence.
4. Its 110k–150k evals fall within 0.63–0.80. Low confidence.

**Checks before the commit.**
- `tests/test_sweep_arms.py`: Tier 1 passed (10), and on the GPU 3 passed. The 4 equivariant-only
  tests skip, having no arm in this group.
- `tests/test_square_rerendered_watch.py` passed (10), with the new group's: `kyv7clfc`'s evals
  are stopped at 30k; `oj5k0jqb`'s, `s8opblsd`'s and a held base's are never stopped through
  150k.
- No smoke run: the code path is the anchor's, and only one scaler value changes.

**Resources.**
- GPU 0, alone until the next entry's arms join it.
- About 20–35 GB of RAM; 121 GB is free after the DEVIATION above.
- `oj5k0jqb` took 5.9 h and `s8opblsd` 11.7 h from step 0 to 150k, plus about 45 min of labelling
  the demos with the base policy before step 0.

---

## 2026-10-12 — NOTE — `square-residual-start-1011`: both arms lost the base policy by 10k; neither held it or reached its 100k check

The readouts pre-registered for the 30k collapse decision and the 100k check. They fell overnight
and were not written as they fell. From the run packages (`wandb_logs/metrics.jsonl`,
`output.log`) and `outputs/queue/watch.log`. Config caveats: none; both arms launched as
pre-registered from `0af7655` (LAUNCHED above).

**`eaqorqki` (`equi-residual-square-init0` 1): stopped by hand at 80.2k (DEVIATION above).**
- Evals 0.50 0.00 0.00 0.06 0.06 0.04 0.16 0.24 0.30 (0–80k).
- **The collapse rule did not fire, by one episode:** 0.00, 0.00, then 0.06 at 30k (3 of 50).
- Training successes per 10k: 9/35 over 0–10k, which is the critic warm-up on the base policy;
  then 0/33, 1/34, 6/36, 1/33, 3/34, 8/36 and 6/35.

**`doevaarp` (`equi-residual-square-warmup50k` 1): stopped by its 100k check** (07:35:55; 30 of
250 at 60k–100k, against 155).
- Evals 0.52 0.02 0.02 0.22 0.12 0.18 0.12 0.12 0.12 0.14 0.10 (0–100k).
- **The collapse rule did not fire:** 0.22 at 30k.
- Training successes to 100k: 65.

**Online over demo value, and the realized discounted return of the training episodes over the same
steps:**

| | (0, 5k] | (5k, 10k] |
|---|---|---|
| `6ng62tpm` (anchor) | 0.005 / 0.199 = 0.02; realized 0.127 | 0.005 / 0.229 = 0.02; realized 0.015 |
| `eaqorqki` (init0) | −0.014 / 0.190 = −0.07; realized 0.128 | −0.003 / 0.226 = −0.02; realized 0.035 |
| `doevaarp` (warmup50k) | 0.078 / 0.216 = 0.36; realized 0.147 | 0.051 / 0.236 = 0.22; realized 0.000 |

**`doevaarp`'s warm-up,** from its console: the means of the ten single-batch values printed in the
1,000 updates before each count.

| Updates | 2k | 10k | 20k | 30k | 40k | 50k |
|---|---|---|---|---|---|---|
| online value | −0.078 | 0.012 | 0.033 | 0.052 | 0.068 | 0.063 |
| demo value | 0.000 | 0.162 | 0.194 | 0.200 | 0.218 | 0.209 |
| TD target | −0.036 | 0.084 | 0.112 | 0.127 | 0.141 | 0.137 |

So the online value was still rising at 10k updates, and levelled off by about 40k.

**`residual_l1` against the action scale 0.1, and `dQ_da_mean_abs`:**

| | (5k, 10k] | (25k, 30k] | (75k, 80k] | (95k, 100k] |
|---|---|---|---|---|
| `6ng62tpm` | 0.028, 7.9e-5 | 0.050, 6.8e-5 | — | — |
| `eaqorqki` | 0.026, 8.7e-5 | 0.035, 8.0e-5 | 0.044, 6.3e-5 | — |
| `doevaarp` | 0.032, 7.6e-5 | 0.023, 7.1e-5 | 0.042, 5.9e-5 | 0.048, 5.2e-5 |

**Predictions:**
1. `init0`'s step-0 eval within 0.46–0.70: **held** (0.50).
2. `init0` collapses by 30k: **failed by the rule,** by one episode (0.06 at 30k). In substance it
   did collapse: 0.00 at 10k and 20k.
3. `warmup50k`'s online value higher at 50k warm-up updates than at 10k: **held** (0.063 against
   0.012).
4. `warmup50k`'s online over demo over (5k, 10k] above 0.1: **held** (0.22).
5. `warmup50k` does not collapse by 30k: **held by the rule** (0.22 at 30k). In substance it lost the
   base policy by 10k (0.02, 0.02).
6. Neither passes its 100k check: **held.** `doevaarp` failed it, and `eaqorqki` could not have
   passed it when stopped.

**Reading it, as decided at pre-registration:**
- **"`warmup50k`'s online value rises through its warm-up, but the run still collapses:
  readiness is not enough"** applies in substance, though the collapse rule did not fire. The
  critic's online value was better informed than `6ng62tpm`'s, and responded to outcomes (0.078
  over (0, 5k], 0.051 over (5k, 10k], as the episodes failed). The residual still grew at the same
  rate as in the other two runs.
- **"`init0` collapses: the 1e-4 start is not the cause"** applies. Later equivariant residual runs
  start at 0.0, as ResFiT's do.
- **The next step named then, a penalty on the residual's size, is held.** The FINDING above gives
  why: the per-sample slope is about 0.018, so the weight that would hold the residual near 0.005
  is about 1.8; ResFiT's slope is the same size, so that weight would hold its useful residual
  down too. The tilt diagnosis follows instead (PRE-REGISTERED above, and to follow).
- **Caveat:** `doevaarp`'s warm-up buffer held about twice the others' successes (16 of 36,
  against 7 and 9 of 35), which may account for some of its online value.

---

## 2026-10-12 — PRE-REGISTERED — the equivariant residual on Square with no tilt, and a diagnostic run with checkpoints

**Questions** (Aden, 2026-10-12, from the FINDING above):
1. Does the equivariant residual hold the base policy on Square if it cannot tilt the gripper?
2. At a collapsed checkpoint, which of the residual's fields causes the collapse?

**Seed group `residual-tilt-1012`,** seed 1 of each arm, from the commit that adds this entry, in a
worktree pinned to it. Names `<arm>-<seed>`, notes `residual-tilt-1012/<arm>/seed<seed>`, wandb
`robomimic-square-final` / `equi-residual-rl`.

**Both arms are `eaqorqki`'s launch** (`equi-residual-square-init0` 1, `square-residual-start-1011`,
`0af7655`) plus their changes:
- **The anchor:** `--config-name=residual_equi_td3_square_config
  equivariance.residual_squash=true agent.critic.num_q=5 base_policy.wt_version=v9
  offline_data.name=local/robomimic-mh-square-rerendered
  agent.actor.actor_last_layer_init_scale=0.0 algo.total_timesteps=300000
  read_buffer_caches=false write_buffer_caches=false seed=1 debug=false`.
- That is: no norm anywhere, 5 critic heads of one hidden layer, N = 8, actor lr 1e-6, action scale
  0.1, the actor's last layer at zero, a 10k critic warm-up.

| Arm | Plus | Length | Rules |
|---|---|---|---|
| `equi-residual-square-notilt` | `+equivariance.residual_field_bound.action_rx_ry=0.0`, `checkpoint_interval=10000` | to 300k | the residual's |
| `equi-residual-square-diag` | `checkpoint_interval=2000` | 60k | none |

**What varies.**
- **`notilt` against `eaqorqki`:** the tilt pair's bound on the actor's mean, 0.1 → 0, so the
  residual cannot tilt the gripper. It bounds the mean only, so exploration and target noise on
  tilt are as the anchor's, and the critic sees the same tilt data (Aden's decision 4 of the
  plan).
- **`diag` against `eaqorqki`:** nothing that computes. It is init0 with checkpoints, ending at
  60k.
- **What both arms add, which changes nothing a run computes** (CODE below, tests):
  - checkpoints every N steps;
  - the per-field residual and slope logging;
  - the bound's buffer, outside the state dict.
- **Between the two arms:** one change, the bound, besides checkpoint counts and length. So `diag`
  is also `notilt`'s control through 60k, on episodes that differ.
- **At the same seed both start from `eaqorqki`'s weights,** bit for bit (test). The episodes
  differ, since `seed` does not reach the envs.

**Rules.**
- **`notilt`:** `square-residual-start-1011`'s residual rules, enforced by the watcher (`--group
  residual-tilt-1012`):
  - the collapse stop, at the third eval in a row at or below 0.05;
  - the divergence stop;
  - the 100k check, at least 155 of 250 at 60k–100k;
  - convergence from 150k;
  - seed 2 iff it passes its 100k check.
- **"Holds the base",** a readout, not a stop: every eval through 30k at least 0.46, the low end of
  B_sq's ten evals.
- **`diag`:** no stop rule. The watcher logs its evals and never stops it; it ends at 60k. Its
  checkpoints feed the field test.

**The field test** (`tools/residual_field_ablation.py`, CODE to follow before it runs):
- **Precondition:** `diag` collapses as its anchors did. Its 10k eval, and the full-residual variant
  at its 10k checkpoint, are both at most 0.10. Otherwise the test is uninformative, and goes back
  to Aden.
- **The test:** at the 10k checkpoint, 100 episodes per variant, with the residual's whole field set
  to zero in `act()`. Whole fields keep the policy equivariant.
  - Variants: full; none (the base policy); −xy; −z; −tilt; −yaw; −gripper; tilt only.
  - At 100 episodes, SE is about 0.05 near 0.55.
- **The verdict:**
  - **Tilt is the cause** iff all three hold: −tilt is within 0.15 of none; tilt only is at most
    0.10; and no other single field's removal reaches 0.30.
  - **Field F is the cause** if F meets the same test in tilt's place.
  - **No single field is the cause** otherwise.
- **Reported, deciding nothing:**
  - the same test at 4k and 6k (the onset), and at 20k and 30k (the recovery);
  - at any 40k–60k checkpoint whose eval shows the grasp failures;
  - each variant's tilt readout and contact sheets.

**What happens at the verdict** (Aden's decision 5 of the plan):
- `notilt` runs on under its own rules, whatever the verdict.
- If the verdict names another field, or no single field, that goes to Aden. No arm is added and
  none is stopped for it.

**Readouts,** in NOTEs (`notilt` at 10k, 30k and 100k; `diag` at 10k, 30k and 60k; the field test
when it is run):
- Evals and training successes per 10k.
- Per field, on online and demo rows: the residual's mean |r| and signed means, and the critic's
  per-sample slope (`debug/residual_field/*`, `debug/dQ_da_field/*`).
- The contact sheets of each eval, and the tilt readout from the checkpoints.

**Predictions** (Claude's, before launch):
1. `notilt`'s step-0 eval is within 0.46–0.70. High confidence: its residual starts at zero.
2. `notilt` holds the base through 30k. Low to moderate confidence: `doevaarp`'s 100k failures
   suggest the drift may move to another field.
3. If `notilt` falls, its failures are at the grasp, not a roll. Moderate confidence.
4. `notilt` passes its 100k check. Low confidence.
5. `diag` collapses by 10k. High confidence: three of three anchors did.
6. At `diag`'s 10k checkpoint, −tilt restores at least 0.40, and tilt only is at most 0.10.
   Moderate confidence for each.
7. `diag`'s tilt signed means are near its mean |r|: a consistent push. Moderate confidence.

**CODE** (in this commit):
- **`equivariance.residual_field_bound`** (`config/residual_td3.py`):
  - The residual's bound per action field, in env units, in place of `agent.actor.action_scale`
    for each field named. The empty default is every earlier run's actor, bit for bit.
  - The actor builds one bound per component from the encoder's field names
    (`obs_encoder.action_fields`). Each field gets one number, so a pair is scaled as a whole and
    the actor stays equivariant (EQUIVARIANCE.md, "Clipping has to respect the representation").
  - Residual actor only. Unknown fields and the scalar ablation raise.
- **Per-field logging** (`utils/field_metrics.py`, both stacks). Both agents hand the trainer the
  actor loss's gradient on the action (`_dQ_da`) on logged updates. The trainer logs, per field
  and on the online and demo rows:
  - the residual's mean |r|, signed means, and a pair's mean length (`debug/residual_field/*`);
  - the critic's per-sample slope (`debug/dQ_da_field/*`), which is 256× the logged
    `debug/dQ_da_mean_abs` scale.
- **`checkpoint_interval`, read by nothing until now, is live** in `train_residual_td3.py`. It saves
  `models/step_<N>.pt` in the run package after each multiple's eval. `train_rlpd_dexmg.py` raises
  if it is set.
- **The launch side:** `submit.sh`, `tests/test_sweep_arms.py` (anchor `eaqorqki`), and the
  watcher's group `residual-tilt-1012`, whose `diag` is logged but never stopped.

**Checks before the commit.**
- **No change to what earlier runs computed:**
  - `tests/test_regression_agent.py`, `tests/test_no_ops.py` and
    `tests/test_regression_single_arm.py` passed (78). The empty mapping is the goldens, bit for
    bit, since `mu * 0.1` and `mu * bound` are bit-identical in float32, forward and backward,
    on GPU and CPU.
  - New in `test_no_ops.py`, for one and two arms: a field's bound replaces `action_scale` on
    that field's components alone, and adds no state-dict entry. Unknown fields and the vanilla
    actor raise.
- **The bound keeps the actor equivariant,** with and without the squash, in train and eval mode:
  new in `test_equivariance.py`, `{action_rx_ry: 0}` and `{action_rx_ry: 0.03, action_xy: 0.05}`.
  New in `test_layouts.py`: the field names match the layout, and match the logged fields.
- **`tests/test_sweep_arms.py`:** Tier 1 passed (15), and on the GPU 18. New tests:
  - `eaqorqki`'s saved model loads strictly into each arm;
  - each arm's residual starts at exactly zero, and every field moves off zero within three
    updates, but `notilt`'s tilt, which stays exactly zero;
  - the scalar ablation refuses the bound.
- **`tests/test_field_metrics.py`** (5, new, in the gate's Tier 1), `tests/test_run_package.py`
  (saving a checkpoint mid-training changes no parameter, buffer, mode or random state), and
  `tests/test_square_rerendered_watch.py` (11) passed.
- **The smoke run of `notilt` runs alongside the launch, not before it,** for speed (Aden: launch as
  soon as possible).
  - It started at 09:20 from the main checkout, with this commit's code uncommitted (`SMOKE=1`,
    GPU 1, console `outputs/queue/*_smoke-equi-residual-square-notilt-1_gpu1.log`).
  - It reaches the training loop well before either arm can: each arm first runs the gate, labels
    80,731 demo frames, fills 10k warm-up steps and runs 10k critic updates.
  - Its result goes in the LAUNCHED entry. If it finds a fault in the new code (the bound, the
    field logging, or the step-0 checkpoint), both arms are stopped before their training starts.

**Resources.**
- `diag` on GPU 1, alone: its 10k checkpoint gates the field test. `notilt` on GPU 0, beside
  `resfit-rx-1012`'s arm.
- **RAM:** about 20 GB per run at start, 35 GB by 150k. Three runs fit (121 GB free before
  `resfit-rx-1012`).
- **Disk:** checkpoints 87 MB each: `diag` 31 (2.7 GB) and `notilt` up to 31 (2.7 GB), of
  about 33 GB free.
- **Timing:**
  - About 1 h to step 0: labelling the demos (about 27 min), the warm-up's 10k steps, and 10k
    critic updates.
  - Then about 1 h per 10k for `diag` alone (`doevaarp`: 59 min between evals), and about 2 h per
    10k for `notilt` sharing a GPU.

---

## 2026-10-12 — CORRECTION and CODE — the eval's Q on the equivariant residual path; a tilt readout in every eval; the field-zeroing tool

**The correction** (the FINDING's side finding, above):
- `run_dexmg_evaluation` scored the critic at `agent.act()`'s output. On the equivariant residual
  path that output is the residual alone: the env wrapper adds the base action itself
  (`residual_env_wrapper.py:136`).
- So every equivariant residual run's "Q =" overlay on its eval videos, and every
  `eval_q_trajectories_*.png` (wandb `value/q_trajectories`), shows the critic at about zero
  action, not at the action executed. That covers every equivariant residual run so far, and
  `eaqorqki`, `doevaarp`, `6ng62tpm` and `kyv7clfc` among them.
- Training is unaffected: the actor loss, the TD target and `critic_diagnostics` always combined the
  actions. So are tuned RLPD, whose actor outputs the whole action, and ResFiT, whose branch already
  added the base.
- **The fix:** the evaluator scores Q at `agent.combine_actions(base, residual)`, the actor loss's
  own combination (made public, from `_combine_actions`).
  - `critic.q_value` draws the same random numbers as before, so an eval's random stream is
    unchanged.
  - It reaches runs launched from this commit on. `resfit-rx-1012` (`60fce72`) is ResFiT's path,
    which was right. `residual-tilt-1012` (`be0b7bb`) launched before it, so its eval Q plots keep
    the old error.

**The CODE** (logging and tooling; none changes what a run computes):
- **A tilt readout in every eval** (`utils/evaluate_dexmg.py`):
  - Per episode, the largest angle between arm 0's gripper axis and its axis at the episode's first
    step, from the raw end-effector quaternion. A yaw leaves the axis where it is.
  - Logged as `eval/tilt_max_deg_mean` and `eval/tilt_over_30deg_frac`.
  - ResFiT's path un-standardizes the state first (`StateStandardizer.unstandardize`, new).
  - `tests/test_eval_base_policy.py` scripts known tilts and yaws through the real evaluator.
- **`tools/residual_field_ablation.py`:**
  - Re-evaluates a run's `models/step_<N>.pt` with whole residual fields set to zero, on the
    trainer's own evaluator, for the field test pre-registered above.
  - It builds the agent with the trainer's own construction, moved verbatim into
    `train_residual_td3.build_agent`, which `main` calls where it built the agent before.
  - `tests/test_field_ablation.py` checks each variant's mask on one and two arms. On the GPU it
    checks that `eaqorqki`'s saved model loads strictly through `build_agent`, and that the unmasked
    agent acts exactly as loaded.
- `tests/README.md`, ARCHITECTURE.md and the gate's Tier 1 list are brought up to date for this
  commit and the two before it.

---

## 2026-10-12 — LAUNCHED — `resfit-rx-1012` seed 1: `resfit-residual-square-rx` (`fk9ayyco`)

From `60fce72` (PRE-REGISTERED above), in the worktree `~/projects/equi-resrl-60fce72`, by hand
through `submit.sh` in tmux, at 09:00. Console
`outputs/queue/20261012-090021_resfit-residual-square-rx-1_gpu0.log`.

| Arm | GPU, tmux | Gate GO | In wandb | Run |
|---|---|---|---|---|
| `resfit-residual-square-rx` 1 | 0, `resfit-rx` | about 09:19 | 09:56 | [**`fk9ayyco`**](https://wandb.ai/aden-mckinney10-university-of-central-florida/robomimic-square-final/runs/fk9ayyco) |

- **The gate said GO:**
  - Tier 1: 207 passed and 2 skipped. Tier 2: 382 passed and 4 skipped, in 15 min.
  - Render check passed: Can 0.0108, Square 0.0085, against bars of 0.0125 and 0.0129.
  - RAM headroom: 113 GB available against 21 GB.
  - Warnings: the BC policy is not in the worktree's `artifacts/` (it downloads `v9`), and the disk
    has 35 GB free (no cache to write).
- **Run package:** records `60fce72`, clean, and mujoco 3.3.2. Its argv is the pre-registered one,
  `oj5k0jqb`'s plus `offline_data.min_action_range=2.0` and `algo.total_timesteps=150000`, and its
  demos are `oj5k0jqb`'s (`ankile/robomimic-mh-square-image`).
- **The ActionScaler moved as pre-registered.** Its printed range over all components is
  [−1.2197, 1.1000]. The low end is the low end of rx's widened range, [−1.220, 0.980] in the
  table above. The high end is every other component's [−1, 1] widened by 1.1.
- **The warm-up buffer:** 7 successes of 35 episodes, as `eaqorqki`'s.
- **Step 0:** 0.46 (23 of 50), the base policy with a residual at zero: the low end of B_sq's
  ten evals.
- **Speed:** critic warm-up updates at 0.025 s on GPU 0, beside `residual-tilt-1012`'s gate.
- **Watcher:** `rx-watch` (`--group resfit-rx-1012`, from the worktree) started at 09:00:26, and
  tracked `fk9ayyco` at 09:56:30.

---

## 2026-10-12 — LAUNCHED — `residual-tilt-1012` seed 1: `notilt` (`dczf4k7v`), `diag` (`ls0thn4u`)

From `be0b7bb` (PRE-REGISTERED above), in the worktree `~/projects/equi-resrl-be0b7bb`, by hand
through `submit.sh` in tmux, both at 09:26, each gate counting both (`CONCURRENT=2`). Consoles in
`outputs/queue/`, `20261012-092609_<arm>-1_gpu<N>.log`.

| Arm | GPU, tmux | Gate GO | In wandb | Run |
|---|---|---|---|---|
| `equi-residual-square-notilt` 1 | 0 (with `fk9ayyco`), `sq-tilt-notilt` | about 10:03 | 10:39 | [**`dczf4k7v`**](https://wandb.ai/aden-mckinney10-university-of-central-florida/robomimic-square-final/runs/dczf4k7v) |
| `equi-residual-square-diag` 1 | 1, `sq-tilt-diag` | about 10:02 | 10:33 | [**`ls0thn4u`**](https://wandb.ai/aden-mckinney10-university-of-central-florida/robomimic-square-final/runs/ls0thn4u) |

- **The gate said GO for both:**
  - Tier 1: 214 passed and 2 skipped, in 16 min. Tier 2: 424 passed, in 20 min. Both slow from
    sharing the machine with each other, `fk9ayyco` and the smoke run.
  - Render check passed: Can 0.0100 and 0.0096, Square 0.0086 and 0.0090 (`notilt`, `diag`),
    against bars of 0.0125 and 0.0129.
  - RAM headroom: 98 and 95 GB available against 42 GB.
  - Warnings: the BC policy is not in the worktree's `artifacts/` (it downloads `v9`), and the disk
    has 34–35 GB free (no cache to write).
- **Run packages:** each records `be0b7bb`, clean, and mujoco 3.3.2. Each argv is the pre-registered
  one: the anchor, plus `+equivariance.residual_field_bound.action_rx_ry=0.0 checkpoint_interval=10000`
  and `algo.total_timesteps=300000` for `notilt`, and `checkpoint_interval=2000` and
  `algo.total_timesteps=60000` for `diag`.
- **The warm-up buffers differ by chance,** since the envs are not seeded: `notilt` 6 successes of
  34 episodes, `diag` 10 of 35, against `eaqorqki`'s 7 of 35. So `diag`'s critic fits somewhat more
  online successes than the other two. The 10k and 30k NOTEs read each run against its own
  realized return.
- **Speed:** critic warm-up updates at 0.076 s for `diag`, alone on GPU 1, and 0.10–0.11 s for
  `notilt`, beside `fk9ayyco` on GPU 0.
- **The smoke run** (`vxw74uzh`, offline, wandb project `equi-resrl-smoke`) passed every check the
  entry set it. It ran `notilt`'s overrides for 1,000 steps after learning starts at 300, with no
  eval.
  - **Its checks:**
    - `models/step_0.pt` was saved (87 MB).
    - All 28 `debug/residual_field/*` keys and all 24 `debug/dQ_da_field/*` keys were logged, every
      100 steps.
    - The tilt pair's residual was exactly 0 on every logged row, online and demo: its mean |r|,
      its length, and rx's and ry's signed means. Every other field left zero, to mean |r| of
      0.001–0.002 by step 1,000. The critic's slope on tilt was not zero (0.028 online, 0.035 on
      demos): the bound holds the actor's mean only, as pre-registered.
    - Every loss was finite. `residual_l1` was 0.0012 at step 1,000. It exited 0.
  - **What code it ran.** It launched at 09:13:47 from the main checkout, with the gate skipped
    (`SKIP_PREFLIGHT=1`), on `60fce72` with `be0b7bb`'s changes uncommitted (the console's "commit
    60fce72"). Its manifest records `be0b7bb` with `534cc71`'s edits uncommitted: the tree when
    wandb started, at 09:43:51, not at launch.
    - `be0b7bb`'s code files were last edited at 09:12:49, before the launch.
    - `534cc71`'s edits began at 09:26:20, so any module the run first imported after that may be
      `534cc71`'s. That commit changes nothing a training run computes (its CODE entry; the
      regression goldens passed at it).
  - **Speed:** critic warm-up updates at 0.075 s on GPU 1, beside `diag`'s gate; the 1,000 steps
    took about 6 min.
- **Watcher:** `tilt-watch` (`--group residual-tilt-1012`, from the worktree) started at 09:26:14,
  and tracked `ls0thn4u` at 10:33:18 and `dczf4k7v` at 10:40:18.

---

## 2026-10-12 — PRE-REGISTERED — ResFiT's residual on Square, given the equivariant stack's differences one at a time; the equivariant residual given ResFiT's layout

**Question** (Aden, 2026-10-12): ResFiT's residual lifts the Square base policy (`oj5k0jqb` 0.800,
`s8opblsd` 0.628 over 110k–150k), and every equivariant residual so far has lost it by 10k.
- Which of the equivariant stack's differences, given to ResFiT's residual one at a time, breaks
  it?
- Does the equivariant residual escape its collapse given ResFiT's layout and parameter count?

Aden chose the demos, LayerNorm and the action limits first, and dropped four arms the plan
proposed: the noise, the critic's size, the augmentation, and LayerNorm off in the actor alone. The
plan is `docs/plans/2026-10-12-resfit-toward-equi.md` (local). `fk9ayyco` (`resfit-rx-1012`, rx's
range alone) is context: 0.46, 0.16, 0.50, 0.48, 0.52 through 40k.

**Seed group `resfit-equi-diff-1012`**, seed 1 of each arm.
- **Names:** `<arm>-<seed>`; notes `resfit-equi-diff-1012/<arm>/seed<seed>`; wandb
  `robomimic-square-final` / `residual-rl`.
- **The anchor** is `oj5k0jqb`'s launch (`resfit-residual-square` 1, `9cf207d`):
  `--config-name=residual_td3_square_config agent.critic.num_q=10 base_policy.wt_version=v9
  algo.total_timesteps=150000 read_buffer_caches=false write_buffer_caches=false seed=<seed>
  debug=false`, on the original coded demos.

| Arm | Change from `oj5k0jqb` | Launched from |
|---|---|---|
| `resfit-residual-square-rerendered` | `offline_data.name=local/robomimic-mh-square-rerendered`, the demos every equivariant run trains on | this commit |
| `resfit-residual-square-noln` | `agent.actor.use_layer_norm=false agent.critic.use_layer_norm=false`; the ViT encoders keep theirs | this commit |
| `resfit-residual-square-limits` | the equivariant stack's action limits, in env units (below) | its CODE commit |
| `resfit-residual-square-noln-all` | `noln`, and the ViTs' LayerNorms too (a new switch) | its CODE commit |
| `resfit-residual-square-combined` | `rerendered` + `noln` + `limits`, **only if each of the three holds alone** (Rules) | `limits`' commit or later |

**`limits`, exactly** (Aden: all three). Three limits change from "the demos' range, per axis" to
the equivariant stack's. ResFiT's noise, its scaling, and the critic's view of actions are kept.
1. **The residual's cap.**
   - Now: 0.1 in scaled units per axis, which is 0.11 env units, and 0.036 on rx.
   - New: the xy pair and the tilt pair (rx, ry) each inside a circle of radius 0.1 env units, with
     `equi_tanh`; z, yaw and gripper within ±0.1 env units.
2. **What the critic scores**, in the actor loss and the TD target.
   - Now: `clamp(base + residual, -1, 1)` in scaled units (`off_policy/rl/q_agent.py:336`, `:450`),
     which is the demos' range per axis.
   - New: `equi_clip` in env units: circles of radius 1 on xy and tilt, and ±1 on the rest, as the
     equivariant agent's `combine_actions`.
3. **What is executed.**
   - Now: `ActionScaler.unscale` clamps to the demos' range ×1.1 per axis (rx [−0.48, 0.24]).
   - New: the env's box [−1, 1].

The buffer keeps storing the unclipped sum, as both stacks do now (TODO B13).

**Seed group `equi-resfit-arch-1013`**, seed 1 of each arm. Its code comes in its own CODE commit.
- **Names:** notes `equi-resfit-arch-1013/<arm>/seed<seed>`; wandb `robomimic-square-final` /
  `equi-residual-rl`.
- **The anchor** is `eaqorqki`'s launch (`square-residual-start-1011`, `0af7655`):
  `--config-name=residual_equi_td3_square_config equivariance.residual_squash=true
  agent.critic.num_q=5 base_policy.wt_version=v9
  offline_data.name=local/robomimic-mh-square-rerendered agent.actor.actor_last_layer_init_scale=0.0
  read_buffer_caches=false write_buffer_caches=false seed=<seed> debug=false`.
- **Run length:** `algo.total_timesteps=150000`, not `eaqorqki`'s 300000. Every schedule runs on
  fixed step counts, so through 150k the two are the same run.

| Arm | Change from `eaqorqki` |
|---|---|
| `equi-residual-square-arch` | ResFiT's layout, as one change: RegularLayerNorm in the actor and the critic where ResFiT has LayerNorm (`equivariance.actor_norm=layer_norm equivariance.critic_norm=layer_norm`), ResFiT's 10 heads of 2 hidden layers (`agent.critic.num_q=10 equivariance.num_critic_layers=2`), and a second policy layer in the actor (`equivariance.num_actor_layers=2`; that field is read by nothing today and is made live in the CODE commit, its default moved from 2 to 1 so every earlier run is unchanged). The critic keeps LayerNorm's own start, not matched to `eaqorqki`'s (Aden: the arm asks for ResFiT's layout). |
| `equi-residual-square-arch-wide` | `arch`, with the actor at 448 fields (`equivariance.actor_degree_channel=448`): 3.4M parameters, against ResFiT's 3.9M |

**Parameter counts as launched** (measured 2026-10-12, the sweep's test builder, Square):

| | Encoders | Actor | Critic |
|---|---|---|---|
| ResFiT's residual (`oj5k0jqb`) | 0.76M | 3.86M | 23.6M |
| The equivariant residual (`eaqorqki`, 5 × 1) | 0.63M | 0.30M | 0.84M |
| … with a 10 × 2 critic | | | 2.82M |
| … with the actor at 256, 362, 448 fields | | 1.13M, 2.22M, 3.36M | |

The channel widths match (1024 = 128 regular fields × 8). Each C8-equivariant layer shares its
weights, so it has about 1/8 of a dense layer's parameters at that width. Matching the critic's
23.6M would cost about 8× the compute, so the critic matches by width only.

**What varies, and what is held.**
- **Each arm against its anchor:** its change alone. `combined` has three changes and `arch`
  five, each declared as one.
- **The same starting weights:** at seed 1, `rerendered`'s initial weights are `oj5k0jqb`'s bit
  for bit. `noln`'s are too, less its LayerNorms, which draw no random number at init
  (`tests/test_sweep_arms.py`). The episodes still differ, since `seed` does not reach the envs.
- **`rerendered` changes the demos' frames and state, not their actions.**
  - The action statistics, and the ActionScaler built from them, are the original's bit for bit
    (test).
  - The state is observed one physics substep late, as the env observes it, so its statistics move
    by at most 4.2e-4 on a min or max.
  - The demos' base actions come from the diffusion base on the re-rendered frames, as in every
    equivariant run.

**Rules,** `fk9ayyco`'s rule for every arm of both groups. The watcher enforces them (`--group`
per seed group).
- **Stop** only at the third eval in a row at or below 0.05, or on the divergence stop. Otherwise
  run to 150k.
- **The main readout:** the 10k, 20k and 30k evals pooled (150 episodes).
  - **A ResFiT arm breaks** iff it collapses, or its readout falls below the anchors' pooled 106
    of 300 (0.353) by two standard errors, √(p(1−p)/150 + 0.353·0.647/300). That is **at most 39
    of 150**. By this rule `fk9ayyco` (57 of 150) holds.
  - **An equivariant arm escapes the collapse** iff its readout is above the equivariant
    anchors' (`eaqorqki` 3 of 150, `6ng62tpm` 1 of 150: 4 of 300) by two standard errors: **at
    least 8 of 150**. It **matches ResFiT** iff it is not below 0.353 by two standard errors: **at
    least 40 of 150**.
- **Seed 2** of a ResFiT arm iff it breaks; of an equivariant arm iff it escapes. From the arm's
  own commit, into the first free slot.
- **`combined` launches** iff `rerendered`, `noln` and `limits` each hold.
- **Reported, deciding nothing:**
  - 110k–150k pooled against the anchors' 0.714. They differ by about 4 SE, so this needs two
    seeds.
  - `residual_l1`, and critic values above 1.5.
  - `noln`'s actor: action size and gradient norm. `9phsy6sf`, ResFiT's tuned RLPD without
    LayerNorm, saturated within 200 updates.
  - The share of rolled wrists in the eval videos.

**Readouts,** in NOTEs at each arm's 30k and at its end.

**Predictions** (Claude's, before launch):
1. `rerendered` holds. Moderate-to-high confidence: ResFiT learned on frames that did not match
   its env's, and these do.
2. `noln` holds. Low confidence. The residual's actor learns at 1e-6 from a zero last layer, far
   from `9phsy6sf`'s saturation at 1e-4, but nothing has tested ResFiT's critic without its
   LayerNorm.
3. `limits` holds. Moderate confidence: rx's range alone holds through 40k (`fk9ayyco`). The
   circles lie inside the anchor's box except on rx, whose cap widens from 0.036 to 0.1.
4. `noln-all` breaks. Low confidence: a ViT with no LayerNorm may not train.
5. `equi-arch` does not escape the collapse. Moderate confidence: RegularLayerNorm slowed
   equivariant tuned RLPD (`0avs073d`), and no single field of the residual carries the collapse
   (the field test of `ls0thn4u` at 10k, NOTE below).
6. If any of `rerendered`, `noln` and `limits` breaks, it is `noln`. Low confidence.

**Checks before the commit.**
- `tests/test_sweep_arms.py`: 16 Tier-1 tests and 6 on the GPU passed. The 4 equivariant-only tests
  skip, with no equivariant arm in the table yet.
  - The anchor still composes to `oj5k0jqb`'s recorded config.
  - The re-rendered demos leave the ActionScaler bit for bit.
  - `noln` starts from the anchor's weights, less its LayerNorms. The critic's ensemble stacks its
    heads, so its stacked LayerNorm tensors are matched to their template layers.
  - `noln` has no LayerNorm in its actor or critic, and its ViTs keep their 6.
- `tests/test_square_rerendered_watch.py`: 12 passed. The new groups follow `fk9ayyco`'s rule on
  `kyv7clfc`'s, `oj5k0jqb`'s and `s8opblsd`'s evals.
- A smoke run of `rerendered`, the ResFiT stack's first load of the local re-rendered dataset
  (`kvcdljaf`, offline in `equi-resrl-smoke`; 14:18–15:21 on GPU 1 beside `diag`, from the main
  checkout at `8ce1642` with this entry's `submit.sh` and tests uncommitted):
  - It loaded the dataset, labelled its base actions, ran the 10k critic warm-up and 1,000 steps
    with updates, and exited 0.
  - The ActionScaler printed the original's range.
  - Peak RAM 10.6 GB (`notilt`'s smoke: 10.4 GB). About 0.26 s per online step: the warm-up took
    about 690 s of its 952 s.
  - Q on demo and online data at most 0.25 and 0.24.

**Resources and timeline** (the plan's estimates):
- **Speed:** a ResFiT run takes about 1 h per 10k sharing a GPU and 0.4 h alone, and about
  1 h 05 min from launch to wandb (the gate, then labelling the demos). The `arch` arms about 2 h
  per 10k, measured in their smoke.
- **Slots:** two runs per GPU.
  - `rerendered` takes GPU 1's free slot.
  - `noln` takes `notilt`'s slot on GPU 0 if `notilt` stops at its 30k collapse check (about
    15:10). Otherwise it takes `diag`'s on GPU 1, at 60k (about 17:30).
  - `limits` takes the next free slot. Overnight it may launch without a smoke (Aden), from a
    tmux script that waits for its slot. The gate still builds the arm and runs an update.
- **Expected:** 30k readouts from about 20:30 tonight. The ResFiT arms end from about 08:30 on
  2026-10-13; the `arch` arms on 2026-10-15 or 16 if they do not collapse.

---

## 2026-10-12 — NOTE — the field test at `ls0thn4u`'s 10k checkpoint: no single field of the residual is the cause

`residual-tilt-1012`'s field test.
- **The run:** `tools/residual_field_ablation.py`, at `8ce1642` from the main checkout, 11:51–12:36.
  The checkpoint is `models/step_10000.pt`; 100 episodes per variant, 8 envs.
- **Records:** in `outputs/diagnostics/residual-field-ablation/` (outside git):
  `ls0thn4u_step10000.json` and `.log`, and contact sheets for full, −tilt and tilt only.
- **Who records it:** the session that launched it ended before writing it, so this entry is by
  the session that pre-registered `resfit-equi-diff-1012`, from the JSON.
- **The precondition holds:** `ls0thn4u`'s 10k eval was 0.00, and the full residual here 0.00, both
  at most 0.10.

| Variant (fields zeroed) | Success of 100 | Mean of each episode's largest tilt |
|---|---|---|
| full | 0 | 23.6° |
| none (the base policy alone) | 57 | 12.5° |
| −xy | 0 | 22.6° |
| −z | 0 | 20.6° |
| −tilt | 1 | 14.4° |
| −yaw | 0 | 19.9° |
| −gripper | 0 | 22.9° |
| tilt only | 22 | 20.6° |

- **The verdict, by the pre-registered rule: no single field is the cause.**
  - Tilt fails all three tests: −tilt is 0.01 against none's 0.57, not within 0.15; tilt only is
    0.22, above 0.10; and every other single removal leaves 0 or 0.01, below 0.30.
  - No other field passes in tilt's place.
  - So, as pre-registered, this goes to Aden, and no arm is added or stopped for it.
- **Tilt alone does real damage** (0.57 to 0.22), but removing it restores nothing. That fits
  `notilt` (`dczf4k7v`), which, bounded to no tilt, scored 0.02 and 0.00 at 10k and 20k.
- **Not run:** the same test at 4k and 6k (the onset), and at 20k and 30k, pre-registered as
  reported only.

---

## 2026-10-12 — CODE — `agent.actor.action_bound`: ResFiT's residual with the equivariant stack's action limits, for `resfit-residual-square-limits`

The code for `resfit-equi-diff-1012`'s `limits` arm (PRE-REGISTERED above): one field, three
limits, as pre-registered.
- **`agent.actor.action_bound`**, `"box"` (the default) or `"equivariant"`. ResFiT's agent
  (`off_policy/rl/`) reads it, and so does the trainer's `ActionScaler`. The equivariant agent
  refuses anything but `"box"`, since it always bounds by its own `equi_tanh` and `equi_clip`.
- **Under `"equivariant"`:**
  1. **The residual's mean.** The actor's last Linear has no Tanh. Its output goes through
     `action_scale · equi_tanh`, in env units, as the equivariant actor's does. Then it is divided
     by the scaler's per-axis half-range, into the scaler's units, where the base action and
     ResFiT's noise are.
  2. **What the critic scores.** The TD target, the actor loss, `critic_diagnostics` and the
     evaluator's Q now all combine through one method, `QAgent.combine_actions`. Under this bound
     it maps the sum to env units (middle + sum × half-range), clips it with `equi_clip` (bound 1),
     and maps it back.
  3. **What is executed.** `ActionScaler(clamp_unscaled=False)`: `unscale` returns
     middle + action × half-range, unclamped. robosuite's controllers clip their input to ±1
     (`default_panda.json`), as they do for every equivariant run under `IdentityScaler`.
  - **Unchanged:**
    - ResFiT's noise: σ 0.05 in the scaler's units, the target's clip 0.3.
    - Its warm-up: base + U(±0.2) in those units.
    - The scaler's widening by `action_scale`, and the clamp `scale` applies to the base action.
    - The buffer still stores the unclipped sum.
  - **The action layout:** `arm_action_shape` and `build_action_layout` move out of `ResObsEnc`
    into module functions in `obs_encoder.py`, so both agents read one layout.
  - **Refused:** a vanilla actor, an action that is not one arm's 7, and a scaler that is missing
    or clamps.
- **On Square, in env units** (the original demos' range ×1.1 per axis). Every axis but rx spans
  the env's [−1, 1], so `limits` changes rx most.

  | | Anchor (`"box"`) | `limits` (`"equivariant"`) |
  |---|---|---|
  | Residual's cap | 0.11 per axis; 0.036 on rx | xy and (rx, ry) each a circle of radius 0.1; z, yaw, gripper ±0.1 |
  | What the critic scores | ±1.1 per axis; rx [−0.48, 0.24] | xy and (rx, ry) each inside the unit circle; z, yaw, gripper ±1 |
  | What is executed | ±1 per axis (robosuite's clip of ±1.1); rx [−0.48, 0.24] | ±1 per axis |

  The rx cap of 0.1 is 0.28 in the scaler's units, against 0.091 on the other axes.
- **`"box"` is every earlier ResFiT run, bit for bit.**
  - `tests/test_action_bound.py` fingerprints ResFiT's agent at `oj5k0jqb`'s config: parameters,
    the eval action, `critic_diagnostics` and one TD3 update, plus the scaler. It measures in
    PyTorch's deterministic mode, in a subprocess, against goldens captured from the tree at
    `8ce1642` before the field existed.
  - The tree with the field gives the same 26 values, exactly. So did both GPUs.
  - The evaluator's ResFiT branch now calls `combine_actions`, which under `"box"` is the same clamp.
- **Tests.**
  - `tests/test_action_bound.py`, 7 passed. They use demo statistics whose middle is off zero on
    every axis, so a missing offset would show.
    - **The residual's mean**, in env units, is `0.1 · equi_tanh` of the layer's output: inside each
      circle and within ±0.1, and it reaches the bound when the last layer is large.
    - **The critic's sum** in env units is `equi_clip`'s: a sum inside the bound is unchanged, and
      a pair outside keeps its direction at length 1. The clip commutes with rotating both pairs by
      π/4 and by 2 rad.
    - **The executed action** is the unclamped sum.
    - At one seed, both bounds start from the same weights.
    - The refusals.
  - `tests/test_sweep_arms.py`: the `limits` arm launches `agent.actor.action_bound=equivariant`
    alone, composes to the anchor but for it, builds and updates, and starts from the anchor's
    weights bit for bit. The anchor's composition still matches `oj5k0jqb`'s recorded config, with
    the new field at `"box"`.
  - With the modules the change reaches, on GPU 1: 265 passed, 6 skipped. Those are the
    equivariant regression goldens (the layout's move changes nothing), `test_sweep_arms.py`,
    `test_critic_diagnostics.py`, `test_env_wrapper.py`, `test_config.py`, `test_layouts.py`,
    `test_field_ablation.py`, `test_eval_base_policy.py` and `test_imports.py`.
- **Not tested:** that the trainer builds the scaler unclamped. It is not silent if it does not:
  the agent refuses a clamping scaler at startup, so such a run fails at launch.

---

## 2026-10-12 — CODE — `agent.vit.use_layer_norm`: the ViT encoders' LayerNorms, for `resfit-residual-square-noln-all`

The code for `resfit-equi-diff-1012`'s `noln-all` arm (PRE-REGISTERED above): `noln`, and the ViT
encoders' LayerNorms off too.
- **`agent.vit.use_layer_norm`** (`VitEncoderConfig`), `true` by default. At `false`, `MinVit` puts
  an Identity in place of each LayerNorm: two per transformer layer, before attention and before
  the MLP, and the final one. That is three per encoder at the anchor's depth of 1, and six for the
  two cameras.
  - With `agent.actor.use_layer_norm=false` and `agent.critic.use_layer_norm=false`, the arm has no
    norm anywhere. The patch embedding's GroupNorm is off in the anchor too
    (`agent.vit.embed_norm=0`).
  - The equivariant agent reads no `agent.vit` field, this one included.
- **At `true`, every earlier ResFiT run, bit for bit:** `tests/test_action_bound.py`'s fingerprint
  of ResFiT's agent (its ViTs included) is the tree before 2026-10-12's changes, exactly, in
  deterministic mode on GPU 1.
- **`tests/test_sweep_arms.py`:**
  - `noln-all` launches its three fields alone, composes to the anchor but for them, builds and
    updates.
  - It starts from the anchor's weights bit for bit, less every LayerNorm's. A LayerNorm draws no
    random number at init.
  - It has no LayerNorm, GroupNorm or BatchNorm in any module.
  - The anchor still composes to `oj5k0jqb`'s recorded config, with the field at `true`.
- **Run on GPU 1:** `test_sweep_arms.py`, `test_action_bound.py`, `test_config.py`, `test_imports.py`
  and `test_no_ops.py`: 228 passed, 6 skipped. `test_no_ops.py` also covers the equivariant agent
  after the `limits` commit, which it had not run on.

---

## 2026-10-12 — AMENDMENT — `equi-resfit-arch-1013`: the actor keeps its depth, which is ResFiT's already

Made before any arm of the group launched.
- **What changes:** `equi-residual-square-arch` no longer sets `equivariance.num_actor_layers=2`,
  and no CODE commit makes that field live. It stays dead (ARCHITECTURE.md's provenance table).
- **Why** (found while writing its code): the pre-registration counted the equivariant actor's
  policy MLP alone. At the anchor's config (`use_visual_proj=true`), the actor has three hidden
  layers before its output, and so does ResFiT's:

  | Actor | Hidden layers | Then |
  |---|---|---|
  | Equivariant | visual projection 288→1024, trunk 1042→1024, one policy layer 1042→1024 | 1024→7 |
  | ResFiT | `compress` 20,736→128, two policy layers 144→1024 and 1024→1024 | 1024→7 |

  A second policy layer would make the equivariant actor a layer deeper than ResFiT's. It would also
  put `arch-wide` at about 5.0M parameters (the layer adds 1.6M at 448 fields), not the 3.4M
  pre-registered.
- **Aden's decision (2026-10-12): keep the depth.**
  - The question the group asks, ResFiT's layout, stands, and so does its parameter count.
  - The second actor layer Aden had asked to test is not run: ResFiT's actor has no extra layer to
    match.
- **The arms now**, each on `eaqorqki`'s launch less its head count, for 150k:

  | Arm | Change from `eaqorqki` |
  |---|---|
  | `equi-residual-square-arch` | `equivariance.actor_norm=layer_norm equivariance.critic_norm=layer_norm agent.critic.num_q=10 equivariance.num_critic_layers=2` |
  | `equi-residual-square-arch-wide` | `arch`, and `equivariance.actor_degree_channel=448` |

- **Parameters,** measured on the arms' composed configs:
  - `arch`: encoder 0.63M, actor 0.31M, critic 2.82M.
  - `arch-wide`: actor 3.36M, against ResFiT's 3.86M.
- **Unchanged:** the anchor, the run length, the rules, the readouts, and the predictions
  (prediction 5 holds as written).
- **Launch:** config only, from this amendment's commit.
  - `arch` runs a smoke first, in its own slot: it is the first equivariant run with ResFiT's
    critic, so its speed and RAM are measured there.
  - `arch-wide` follows without one.
  - Both go through `outputs/queue/launchers/slot_queue.sh` into the first free slots after
    `noln-all`.
- **Tests:** `tests/test_sweep_arms.py` now covers both groups, each against its own anchor.
  - Tier 1, 42 passed: the arms launch the overrides above and compose to `eaqorqki`'s config but
    for them. `eaqorqki`'s anchor is its launch, and composes to its recorded config but for the two
    ResFiT fields added today, at their defaults.
  - Tier 2 on GPU 1, 21 passed.
    - Both arms build and update.
    - The actor is equivariant and the critic invariant at their sizes, in eval and train mode.
      The zero-initialized actor is checked at escnn's own init.
    - Every input reaches the critic and the actor: RegularLayerNorm is not blind as FieldNorm
      was.
    - Each arm has ResFiT's layout: 3 RegularLayerNorms in the actor, 22 in the critic, none in
      the encoder, and no other norm; 10 heads of 3 Linears. `arch-wide`'s hidden layers are 448
      fields.

---

## 2026-10-12 — LAUNCHED — `resfit-equi-diff-1012` seed 1: `rerendered` (`bue9dupf`); `noln`, `limits`, `noln-all` and `equi-resfit-arch-1013`'s arms queued

`rerendered` launched from `7db124d` (PRE-REGISTERED above), in the worktree
`~/projects/equi-resrl-7db124d`, at 15:25 through `submit.sh` in tmux `rfe-rerendered`, on GPU 1
beside `diag`, its gate counting itself alone (`CONCURRENT=1`). Its console is
`outputs/queue/20261012-152508_resfit-residual-square-rerendered-1_gpu1.log`.

| Arm | GPU | Gate GO | In wandb | Run |
|---|---|---|---|---|
| `resfit-residual-square-rerendered` 1 | 1 | about 15:50 | 16:43 | [**`bue9dupf`**](https://wandb.ai/aden-mckinney10-university-of-central-florida/robomimic-square-final/runs/bue9dupf) |

- **The gate said GO:**
  - Tier 1: 226 passed and 2 skipped, in 3.5 min. Tier 2: 413 passed and 4 skipped, in 18 min.
  - RAM: 36 GB available against about 21 GB.
  - Two warnings:
    - The BC policy is not in the worktree's `artifacts/`. It loaded from wandb's cache.
    - The disk had 28 GB free. It is 26.5 GB at 15:48, and each ResFiT run writes about 1–1.5 GB
      through 150k.
- **Run package:** records `7db124d`, clean, and mujoco 3.3.2. Its argv is the pre-registered one:
  the anchor plus `offline_data.name=local/robomimic-mh-square-rerendered`.
- **Labelling** the 300 demos' base actions on the re-rendered frames took 47 min, beside `diag` on
  GPU 1.
- **Its smoke** (`kvcdljaf`) is in the pre-registration's checks.

**Queued, each to launch without this session.** Each needs its own LAUNCHED entry with its run
ID, from a later session. The scripts are in `outputs/queue/launchers/`, and each logs what it waits
for.

| Arm | From | How | Expected |
|---|---|---|---|
| `resfit-residual-square-noln` | `7db124d` | `wait_then_launch.sh` on `diag`'s process, GPU 1 (tmux `rfe-noln`) | about 19:00, when `diag` ends at 60k |
| `resfit-residual-square-limits` | `304167e` (CODE) | `wait_then_launch.sh` on `fk9ayyco`'s process, GPU 0 (tmux `rfe-limits`), no smoke (Aden) | about 01:00 on 2026-10-13 |
| `resfit-residual-square-noln-all` | `9add9e6` (CODE) | `slot_queue.sh`, the first free slot (tmux `rfe-queue`), no smoke | 2026-10-13, when `rerendered`, `noln` or `notilt` ends |
| `equi-residual-square-arch` | `95490f9` (AMENDMENT) | `slot_queue.sh`, the next free slot: its smoke, then the arm if the smoke exits 0 | 2026-10-13 |
| `equi-residual-square-arch-wide` | `95490f9` | `slot_queue.sh`, the next free slot, no smoke | 2026-10-13 or 14 |

- **Watchers,** each enforcing `fk9ayyco`'s rule:
  - tmux `rfe-watch`, from `7db124d`, for `resfit-equi-diff-1012`. Its arm list holds the queued
    ResFiT arms.
  - tmux `arch-watch`, from `95490f9`, for `equi-resfit-arch-1013`.
- **The slot queue counts a slot as held** by a run's gate or trainer (by `CUDA_VISIBLE_DEVICES`),
  by a launcher from its start to its run's end, and by a waiting `wait_then_launch.sh`. Two runs
  per GPU. At its start it counted 3 on each GPU: two runs and one waiting launch.

---

## 2026-10-12 — PRE-REGISTERED — the equivariant residual's anchors again, to a fixed 150k (`equi-residual-long-1012`)

**Question** (Aden, 2026-10-12): does the equivariant residual win the base policy back if it is let
run? Every equivariant residual on the re-rendered demos lost the base by 10k. None was let run to
see whether it recovers:
- `6ng62tpm` was stopped by its collapse rule at 30.3k (0.62, 0.02, 0.00, 0.00).
- `eaqorqki` passed that rule by one success (0.06 at 30k), then climbed to 0.16, 0.24 and 0.30 at
  60k–80k, and was cancelled at 80.2k still climbing.
- `doevaarp` came back to 0.22 at 30k, then sat at 0.10–0.18 to its 100k check, which stopped it.
- `notilt` (`dczf4k7v`) is at 0.14, 0.18 and 0.28 at 30k–60k.

**Seed group `equi-residual-long-1012`**, seed 2 of each arm: seed 1's starts are on record to 30k
(`6ng62tpm`) and 80k (`eaqorqki`, and `ls0thn4u` to 60k). At seed 2 the two arms' weights are the
same but for the actor's last layer, so they make a pair.
- **Names:** `<arm>-2`; notes `equi-residual-long-1012/<arm>/seed2`; wandb
  `robomimic-square-final` / `equi-residual-rl`.

| Arm | Its launch, but for `algo.total_timesteps=150000` and `seed=2` |
|---|---|
| `equi-residual-square-long` | `6ng62tpm`'s (`square-rerendered-1009`, `eb90a0c`): `--config-name=residual_equi_td3_square_config equivariance.residual_squash=true agent.critic.num_q=5 agent.actor.actor_last_layer_init_scale=1e-4 base_policy.wt_version=v9 offline_data.name=local/robomimic-mh-square-rerendered read_buffer_caches=false write_buffer_caches=false debug=false` |
| `equi-residual-square-init0-long` | `eaqorqki`'s (`square-residual-start-1011`, `0af7655`): the same with `agent.actor.actor_last_layer_init_scale=0.0` |

- **What varies, against each original run:**
  - The seed, 2 not 1.
  - The run length, 150k not 300k. Every schedule runs on fixed step counts, so through 150k each
    is its original run.
  - The rules (below).
  - The code is this commit's. The equivariant agent's update is pinned by
    `tests/test_regression_agent.py`'s goldens, which pass here as at both originals' commits. The
    fields added since are at their no-op defaults (`tests/test_sweep_arms.py`).
- **Rules:** to a fixed 150k, stopped only on the divergence stop. No collapse stop, no 100k check,
  no convergence stop.
- **Readouts,** in NOTEs at 30k and at 150k:
  - **The dip:** the 10k–30k evals pooled (150 episodes).
  - **The late window:** the 110k–150k evals pooled (250 episodes), against the base rate
    B_sq = 0.582 (500 episodes) and ResFiT's anchors' 0.714 over the same window (`oj5k0jqb` and
    `s8opblsd`, 500 episodes).
    - **Past the base** iff it is above 0.582 by two standard errors,
      √(p(1−p)/250 + 0.582·0.418/500).
    - **Back to the base** iff it is not below 0.582 by two standard errors.
    - **ResFiT's level** iff it is not below 0.714 by two standard errors.
  - Reported: the first eval at or above 0.30, `residual_l1`, the tilt readouts, critic values
    above 1.5.
- **Seed 3** of an arm iff it is back to the base or past it at 150k.
- **Predictions** (Claude's, before launch):
  1. Both arms recover partly and stay below the base: under 0.582 by two standard errors at
     110k–150k. Moderate confidence: `eaqorqki`'s 0.30 at 80k and `doevaarp`'s plateau.
  2. The two arms do not differ by more than two standard errors at 110k–150k. Moderate confidence:
     the start did not set the collapse (`eaqorqki`).
- **Launch:** from this entry's commit, through `outputs/queue/launchers/slot_queue.sh`, first in
  its order (AMENDMENT below).

---

## 2026-10-12 — AMENDMENT — every run to a fixed 150k; `resfit-equi-diff-1012` gains `smallcritic` and `noise`; the queue's order

Aden's decisions (2026-10-12), before any run below reached a rule's step:
- **Every run to a fixed 150k, stopped only on the divergence stop.** A residual that loses the base
  can win it back late (PRE-REGISTERED above), so the collapse stop could stop one that would
  recover. Aden chose the same rule for both stacks.
  - **`resfit-equi-diff-1012`:** no collapse stop. `noln` (`c1xca5uk`, 0.00 at 10k) runs on. The
    watcher was stopped by hand at 20:07, before its 20k eval, and restarted with this rule.
  - **`equi-resfit-arch-1013`:** no collapse stop, and a late readout added.
  - **`residual-tilt-1012`'s `notilt`:** no collapse stop, no 100k check, no convergence stop. It
    was launched for 300k, so the watcher stops it at its 150k eval.
  - **The main readouts stand:**
    - A ResFiT arm breaks iff its 10k–30k evals pooled are at most 39 of 150, or three evals in a
      row are at or below 0.05 (it is no longer stopped for that).
    - An equivariant arm escapes the collapse iff its 10k–30k evals pooled are at least 8 of 150.
  - **Added for every arm:** the 110k–150k window against B_sq and ResFiT's 0.714, as in the
    PRE-REGISTERED entry above.
- **`resfit-equi-diff-1012` gains two arms,** each one change from `oj5k0jqb`'s launch:

  | Arm | Change | Code |
  |---|---|---|
  | `resfit-residual-square-smallcritic` | the equivariant residual's critic heads: `agent.critic.num_q=5 agent.critic.num_layers=1`, at ResFiT's width (1024) | none |
  | `resfit-residual-square-noise` | the equivariant stack's noise: `agent.actor.noise_units=env`, σ 0.05 in env units on every axis (ResFiT's own: 0.055 on most of Square's axes, 0.018 on rx) | CODE below |

  - **`combined`** (`rerendered` + `noln` + `limits`) launches iff each of the three holds by the
    rule above. `tools/arms_hold.py` decides it when its turn in the queue comes. It is skipped if
    any arm breaks, or if any lacks its 30k eval.
  - **Predictions:**
    - `smallcritic` holds. Low confidence: the critic is the stacks' largest difference, but
      ResFiT's demos-plus-online value learning may not need its size.
    - `noise` holds. Moderate confidence: the rx noise grows 2.75×, and rx's range alone held
      (`fk9ayyco`).
- **The queue,** in order, each into the first free GPU slot (`slot_queue.sh`, tmux `rfe-queue`):

  | # | Arm | Seed | From | Smoke |
  |---|---|---|---|---|
  | 1 | `equi-residual-square-long` | 2 | this commit | no |
  | 2 | `equi-residual-square-init0-long` | 2 | this commit | no |
  | 3 | `resfit-residual-square-limits` | 1 | `304167e` | no (Aden) |
  | 4 | `resfit-residual-square-noln-all` | 1 | `9add9e6` | no |
  | 5 | `equi-residual-square-arch` | 1 | `95490f9` | yes, in its own slot |
  | 6 | `equi-residual-square-arch-wide` | 1 | `95490f9` | no |
  | 7 | `resfit-residual-square-combined`, iff the three hold | 1 | this commit | no |
  | 8 | `resfit-residual-square-smallcritic` | 1 | this commit | no |
  | 9 | `resfit-residual-square-noise` | 1 | this commit | no |

  - **Two changes from Aden's table:** `diag` ended at 18:09, before this was wired, and its
    slot went to `noln` as wired then. So the baseline takes the next slot, `fk9ayyco`'s, at about
    00:45. The rest keep their order.
  - Seed 2s per the rules go in by hand, after the queue.

---

## 2026-10-12 — CODE — `agent.actor.noise_units`: ResFiT's noise in env units, for `resfit-residual-square-noise`; the watcher's fixed length; the queue's `combined` check

- **`agent.actor.noise_units`**, `"scaled"` (the default) or `"env"`. ResFiT's agent reads it, and
  the equivariant agent refuses anything but `"scaled"`.
  - **Under `"env"`:** the actor's σ (`algo.stddev_*`, 0.05) is in env units on every axis. It is
    divided by the scaler's per-axis half-range into the scaler's units, where ResFiT adds it. The
    clip on the noise (`agent.stddev_clip`, 0.3) is in env units too. It binds at 6σ, so never in
    effect; the equivariant stack has none. This covers both the exploration noise and the TD
    target's.
  - **Unchanged:** the residual's bound (`action_bound`, `"box"` in this arm), the warm-up's
    U(±0.2), and the truncation to [−1, 1] in the scaler's units, which binds at neither σ.
  - It reads the same env-unit frame as `action_bound=equivariant`, through
    `QAgent._set_env_frame`, which replaces `_set_equivariant_bound`.
  - **On Square:** σ goes from 0.055 env units to 0.05 on six axes, and from 0.018 to 0.05 on rx.
- **At `"scaled"`, every earlier ResFiT run bit for bit:** `tests/test_action_bound.py`'s fingerprint
  is the tree before 2026-10-12's changes, exactly, on GPU 1.
- **Tests,** `tests/test_action_bound.py`, 10 passed:
  - Under `"env"`, the actor's σ times the half-range is 0.05 on every axis, where ResFiT's own σ
    is 0.05 in the scaler's units.
  - With σ = 1, every draw's noise is within 0.3 env units, and reaches it on every axis.
  - Both settings start from the same weights.
  - `"env"` is refused on a vanilla actor, without a scaler, and by the equivariant agent.
- **The watcher** (`tools/square_rerendered_watch.py`):
  - `FIXED_LENGTH_ARMS`: every arm of `resfit-equi-diff-1012`, `equi-resfit-arch-1013` and
    `equi-residual-long-1012`, and `notilt`. These are stopped only on the divergence stop, and
    `notilt` at its 150k eval (`STOP_AT`).
  - `COLLAPSE_ONLY_ARMS` keeps `fk9ayyco` alone.
  - The new group `equi-residual-long-1012`.
  - `tests/test_square_rerendered_watch.py`: none of those arms is stopped on a 16-eval collapse or
    on `oj5k0jqb`'s, `s8opblsd`'s or a flat 0.58 history. `notilt` alone stops at 150k, each still
    stops on a runaway critic, and the diagnostic run never stops.
- **`tools/arms_hold.py`** decides `combined`, by the pre-registered break rule, from the three arms'
  live records.
  - It exits 0 iff all three hold, 1 if any breaks, and 2 if any is not decidable yet. The queue
    skips the job on anything but 0.
  - Its test pins the rule's edge: 39 of 150 breaks, 40 holds, and so does a collapse after 30k.
  - On the live runs it gives `fk9ayyco` 57 of 150, the pre-registration's own count.
- **The arms' tests,** `tests/test_sweep_arms.py`, now with three anchors (`oj5k0jqb`, `eaqorqki`,
  `6ng62tpm`):
  - Each anchor is its run's launch, and composes to its recorded config but for the fields added
    since, at their defaults. `6ng62tpm`'s adds `equivariance.critic_pooled_norm: none`.
  - `smallcritic` has 5 heads of one hidden layer at width 1024, and its encoders start as the
    anchor's.
  - `combined` and `noise` start from the anchor's weights, less `combined`'s LayerNorms.
  - The long arms are their anchors' launches, equivariant and invariant at their sizes.
- **Run on GPU 1:** the regression goldens, `test_action_bound.py`,
  `test_square_rerendered_watch.py`, `test_config.py`, `test_imports.py`,
  `test_critic_diagnostics.py`, `test_env_wrapper.py` and `test_sweep_arms.py` with its GPU tier:
  274 passed, 2 skipped. The `arms_hold` test, added after that run started, passed on its own
  (12 in its file).

---

## 2026-10-12 — LAUNCHED — `resfit-equi-diff-1012` seed 1: `noln` (`c1xca5uk`)

Launched by its waiting script, `wait_then_launch.sh` in tmux `rfe-noln`, when `diag`'s process
exited at 18:09. It launched at 18:09 from `7db124d`, worktree `~/projects/equi-resrl-7db124d`, on
GPU 1 beside `rerendered`. Its console is
`outputs/queue/20261012-180908_resfit-residual-square-noln-1_gpu1.log`.

| Arm | GPU | In wandb | Run |
|---|---|---|---|
| `resfit-residual-square-noln` 1 | 1 | 19:08 | [**`c1xca5uk`**](https://wandb.ai/aden-mckinney10-university-of-central-florida/robomimic-square-final/runs/c1xca5uk) |

- **The gate said GO:**
  - Tier 1: 226 passed and 2 skipped. Tier 2: 413 passed and 4 skipped, in 16 min.
  - Two warnings, as `rerendered`'s: the BC policy loads from wandb's cache, and the disk is low.
- **Run package:** records `7db124d`, clean, and mujoco 3.3.2. Its argv is the pre-registered one:
  the anchor plus `agent.actor.use_layer_norm=false agent.critic.use_layer_norm=false`.
- **Evals so far:** 0.54 at step 0, **0.00 at 10k**. ResFiT's anchors were at 0.30 and 0.22 at
  10k, `rerendered` at 0.16.
  - So this AMENDMENT's rule runs it on: no collapse stop.
  - Its watcher was stopped at 20:07, before its 20k eval, and restarted with that rule.

---

## 2026-10-13 — LAUNCHED — the queue overnight: `init0-long` (`oav5gk76`), `noln-all` (`ra6024xn`), `arch-wide` (`grvwhcfr`); three gates failed, and those jobs are queued again

The slot queue (`outputs/queue/launchers/slot_queue.sh`, tmux `rfe-queue`), as the 2026-10-12
AMENDMENT set it up. Consoles are in `outputs/queue/`, and the queue's log is
`outputs/queue/launchers/slot_queue.log`.

| Job | Launched | GPU | Gate | Run |
|---|---|---|---|---|
| `equi-residual-square-long` 2 | 00:42, as `fk9ayyco` exited | 0 | **NO-GO** | — |
| `equi-residual-square-init0-long` 2 | 01:11 | 0 | GO (Tier 2: 501 passed) | [**`oav5gk76`**](https://wandb.ai/aden-mckinney10-university-of-central-florida/robomimic-square-final/runs/oav5gk76), from `0b14c17`, in wandb 02:31 |
| `resfit-residual-square-limits` 1 | 03:31, as `rerendered` exited | 1 | **NO-GO** | — |
| `resfit-residual-square-noln-all` 1 | 03:52 | 1 | GO (Tier 2: 430 passed) | [**`ra6024xn`**](https://wandb.ai/aden-mckinney10-university-of-central-florida/robomimic-square-final/runs/ra6024xn), from `9add9e6`, in wandb 04:51 |
| `equi-residual-square-arch` 1, its smoke | 05:30, as `noln` exited | 1 | **NO-GO**, so not launched | — |
| `equi-residual-square-arch-wide` 1 | 05:58 | 1 | GO (Tier 2: 454 passed) | [**`grvwhcfr`**](https://wandb.ai/aden-mckinney10-university-of-central-florida/robomimic-square-final/runs/grvwhcfr), from `95490f9`, in wandb 07:02 |

- **Each run package** records its commit, clean, and mujoco 3.3.2. Each argv is its pre-registered
  one, the long arm at seed 2.
- **The three NO-GOs** each failed one Tier-1 test,
  `tests/test_rerender_demos.py::test_every_guard_passes_on_cans_first_demos`, and nothing else.
  - Each came in the minute a run exited. The next launch, 20–30 minutes later, passed the same
    test.
  - Run by hand at 07:58 from `0b14c17`, it passed three times in three.
  - So the likely cause is the exiting run still releasing the GPU or the CPU while the test
    renders Can's demos and compares pixels. The gate records only the test's name, not which
    assertion failed, so this is not confirmed (TODO).
- **Two consequences:**
  - Each failed job's slot went to the next job, as the queue is written. So `arch-wide` launched
    before `arch`, with no smoke run before it. Its own run measures the speed and RAM `arch`'s
    smoke would have.
  - **Fixed at 08:01:** a slot now counts as free once it has stayed free for 300 s
    (`slot_queue.sh`'s `SETTLE_S`).
- **Queued again at 08:01, first:** `long`, `limits`, `arch` (smoke first). Then the jobs not yet
  reached: `combined` (iff its three arms hold), `smallcritic`, `noise`.

---

## 2026-10-13 — NOTE — `rerendered` and `fk9ayyco` hold; `noln` breaks ResFiT's residual, and does not recover by 150k

`resfit-equi-diff-1012` (PRE-REGISTERED 2026-10-12) and `resfit-rx-1012`, seed 1, against the
anchors' pooled 10k–30k of 106 of 300 (0.353; an arm breaks at 39 of 150 or fewer) and their pooled
110k–150k of 0.714 (500 episodes). Evals are 50 episodes each, from the run packages.

| Run | Evals 0, 10k, …, 150k | 10k–30k | Verdict | 110k–150k |
|---|---|---|---|---|
| `fk9ayyco` rx range | 0.46, 0.16, 0.50, 0.48, 0.52, 0.50, 0.50, 0.62, 0.48, 0.60, 0.66, 0.70, 0.70, 0.76, 0.70, 0.72 | 57 of 150 | holds | 179 of 250, 0.716 |
| `bue9dupf` `rerendered` | 0.60, 0.16, 0.48, 0.60, 0.70, 0.58, 0.62, 0.64, 0.76, 0.88, 0.74, 0.74, 0.74, 0.90, 0.88, 0.88 | 62 of 150 | holds | 207 of 250, **0.828** |
| `c1xca5uk` `noln` | 0.54, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.02, 0.00, 0.02, 0.06, 0.04, 0.04 | **0 of 150** | **breaks** | 8 of 250, 0.032 |
| `ra6024xn` `noln-all` (running) | 0.48, 0.00, 0.00, 0.00, 0.02 (to 40k) | **0 of 150** | **breaks** | — |

- **LayerNorm in the actor and critic is what ResFiT's residual needs here,** of the differences
  tested so far.
  - Without it, the residual loses the base by 10k, as every equivariant residual does, and does
    not win it back by 150k: 0.032 at 110k–150k, against the equivariant `notilt`'s 0.16–0.36 over
    90k–140k.
  - `noln-all`, with the ViTs' LayerNorms off too, fails the same way so far.
  - One seed each; seed 2s are due by the rule.
- **`rerendered` beats its anchors late:** 0.828 against 0.714 at 110k–150k, about 3.6 standard
  errors. One seed, so reported, not decided on.
  - By its rule it holds, so the demos are not what breaks the equivariant residual.
  - `fk9ayyco` (rx's range alone) lands on the anchors: 0.716.
- **Predictions** (2026-10-12):
  - 1 (`rerendered` holds) and 6 (if one breaks, it is `noln`) were right.
  - 2 (`noln` holds, low confidence) was wrong.
- **What follows by the rules:**
  - `combined` will be skipped, since `noln` broke.
  - Seed 2s of `noln` and of `noln-all` are due (each broke). They are launched by hand, after the
    queue (AMENDMENT).
  - `limits` is queued again.
- **For the next session:** what LayerNorm does for ResFiT's critic and actor, and what that says
  about the equivariant stack's normalization. `arch` and `arch-wide` put RegularLayerNorm into the
  equivariant residual, and `arch-wide` is running (0.64 at step 0).

---

## 2026-10-13 — DEVIATION and AMENDMENT — `arch-wide` (`grvwhcfr`) stopped at 10k; `arch`, `smallcritic` and `combined` dropped; `noln-all`'s seed 2 skipped; the equivariant baseline's seed 3 now, and `noln`'s seed 2 into the queue

Aden's decisions (2026-10-13, about 09:10): the work narrows to the equivariant residual's reference
to 150k, LayerNorm in ResFiT's residual, init0, and the actor's bounds and noise. The actor's
architecture, the critic's size and the weight-init scheme are set aside for now. The question
order this serves, Aden's: (1) is LayerNorm an enabler of ResFiT's residual; (2a) which of its
placements; (2b) how to do what it does in the equivariant residual; then the actor's other
differences, if LayerNorm does not carry it.

- **`grvwhcfr` (`equi-residual-square-arch-wide` 1) stopped** by hand (SIGINT) just past its 10k
  eval. Evals 0.64 at step 0, **0.12 at 10k**.
  - Reported, deciding nothing: every other equivariant residual on the re-rendered demos scored
    0.00–0.02 at 10k (`6ng62tpm`, `eaqorqki`, `doevaarp`, `ls0thn4u`, `dczf4k7v`, `oav5gk76`: 3 of
    300 pooled). `arch-wide` is the only one with RegularLayerNorm in its actor and critic.
  - `equi-resfit-arch-1013`'s rules decide nothing on one eval. Its pre-registration stands, unrun
    past 10k.
- **Dropped from the queue, unrun:** `equi-residual-square-arch` (its smoke's gate failed overnight,
  so it never launched), `resfit-residual-square-smallcritic`, and `resfit-residual-square-combined`
  (its rule would have skipped it, since `noln` broke). Their pre-registrations stand.
- **`noln-all`'s seed 2 is skipped,** against its rule (it broke, so seed 2 was due). `noln` already
  breaks, so a second seed of `noln` with the ViTs' LayerNorms off too cannot tell anything `noln`'s
  own seed 2 cannot. `ra6024xn` runs on to 150k.
- **`noln`'s seed 2** goes into the queue now, third, from its launch commit (`7db124d`), instead
  of by hand after the queue (the 2026-10-12 AMENDMENT).
- **AMENDMENT to `equi-residual-long-1012`: seed 3 of `equi-residual-square-long` now,** from
  `0b14c17` as seed 2, not only if seed 2 is back to the base at 150k.
  - Why: no equivariant residual of this design has a record past 30k (`6ng62tpm` was stopped at
    30.3k), so `init0`'s and `notilt`'s late evals have no reference. One seed gives no spread:
    ResFiT's two anchors differ by 0.17 at 110k–150k (0.800, 0.628).
  - Its rules and readouts are seed 2's. `init0-long`'s seed 3 keeps its rule.
- **The queue,** in this order (`slot_queue.sh`, tmux `rfe-queue`, restarted with this file):

  | # | Job | Seed | From |
  |---|---|---|---|
  | 1 | `equi-residual-square-long` | 2 | `0b14c17` |
  | 2 | `equi-residual-square-long` | 3 | `0b14c17` |
  | 3 | `resfit-residual-square-noln` | 2 | `7db124d` |
  | 4–6 | LayerNorm off in the actor alone, the critic alone, the ViTs alone, seed 1 | 1 | the PRE-REGISTERED entry below |
  | 7 | `resfit-residual-square-limits` | 1 | `304167e` |
  | 8 | `resfit-residual-square-noise` | 1 | `0b14c17` |
  | 9–11 | the three LayerNorm arms, seed 2 | 2 | the PRE-REGISTERED entry below |

  Rows 4–6 and 9–11 join the jobs file once their pre-registration is committed and pinned.

---

## 2026-10-13 — PRE-REGISTERED — ResFiT's residual on Square with LayerNorm off in one place (`resfit-ln-place-1013`)

**Question** (Aden, 2026-10-13; (2a) of the question order in the DEVIATION above): `noln`, with
LayerNorm off in the actor and the critic, breaks ResFiT's residual (`c1xca5uk`: 0 of 150 at
10k–30k, 0.032 at 110k–150k; its seed 2 is queued). Which of its LayerNorms does the residual
need: the actor's, the critic's, or the ViT encoders'?

**Seed group `resfit-ln-place-1013`**, seeds 1 and 2 of each arm.
- **Names:** `<arm>-<seed>`; notes `resfit-ln-place-1013/<arm>/seed<seed>`; wandb
  `robomimic-square-final` / `residual-rl`.
- **The anchor** is `oj5k0jqb`'s launch, as for `resfit-equi-diff-1012`: `--config-name=residual_td3_square_config
  agent.critic.num_q=10 base_policy.wt_version=v9 algo.total_timesteps=150000
  read_buffer_caches=false write_buffer_caches=false seed=<seed> debug=false`, on the original coded
  demos.

| Arm | Change from `oj5k0jqb` | The LayerNorms it removes |
|---|---|---|
| `resfit-residual-square-noln-actor` | `agent.actor.use_layer_norm=false` | after the actor's compression layer and each of its hidden layers |
| `resfit-residual-square-noln-critic` | `agent.critic.use_layer_norm=false` | after the critic's input projection and each head's hidden layers |
| `resfit-residual-square-noln-vit` | `agent.vit.use_layer_norm=false` | the two ViT encoders' six (two per transformer layer and a final one) |

- With the anchors (both on) and `noln` (both off), `noln-actor` and `noln-critic` complete a 2×2
  on the actor's and the critic's LayerNorms. `noln-vit` is the third place; `noln-all` (all three
  off) is not needed to read it.
- No new code: each arm is one existing switch.

**What varies, and what is held.**
- Each arm's change alone. At a seed, its starting weights are the anchor's bit for bit, less the
  LayerNorms it removes, which draw no random number at init (`tests/test_sweep_arms.py`).
- The code is this commit's. It changes no agent code from `0b14c17`, where ResFiT's agent at these
  defaults is the tree before 2026-10-12's changes (`tests/test_action_bound.py`'s fingerprint).
- The episodes differ between runs, as always: `seed` does not reach the envs.

**Rules,** `resfit-equi-diff-1012`'s as amended on 2026-10-12. The watcher enforces them
(`--group resfit-ln-place-1013`).
- **To a fixed 150k,** stopped only on the divergence stop.
- **The main readout:** the 10k, 20k and 30k evals pooled (150 episodes). An arm **breaks** iff that
  is at most 39 of 150 (below the anchors' 106 of 300 by two standard errors), or three evals in a
  row are at or below 0.05. Otherwise it **holds**.
- **Both seeds of every arm,** whatever seed 1 shows: a hold is a claim here too (Aden: one or two
  150k runs per question).
- **Reading a place, at two seeds:**
  - **Needed** iff its arm breaks at both seeds.
  - **Not needed alone** iff its arm holds at both seeds. This is not "it does nothing": a hold
    admits 40 of 150 against the anchors' 0.353. So both windows are reported against the anchors'
    (0.353 at 10k–30k, 0.714 at 110k–150k), to show how close it comes.
  - **No verdict** if the seeds disagree; seed 3 by hand.
  - If `noln-actor` and `noln-critic` both hold while `noln` breaks, either LayerNorm is enough
    alone, and that is the reading.
- **Reported, deciding nothing:**
  - 110k–150k pooled against the anchors' 0.714 and B_sq = 0.582.
  - The residual's size early: `train/residual_l1_magnitude` and `debug/residual_max_abs` at
    1k–3k and 10k (`noln` reached 0.083 of its 0.1 cap by 3k, the anchors 0.04).
  - `train/actor_grad_norm`, `debug/dQ_da_mean_abs`, `debug/q_data_online`, `debug/q_advantage`,
    and critic values above 1.5.
- **No checkpoints:** a ResFiT checkpoint is 223 MB, and the disk is short.

**Predictions** (Claude's, before launch):
1. `noln-actor` breaks. Moderate confidence: `noln`'s residual reached its cap by 3k, the
   saturation `9phsy6sf` showed in tuned RLPD, which a one-batch test traced to the actor's
   LayerNorms (2026-10-08 NOTE).
2. `noln-critic` holds. Low confidence: nothing has tested ResFiT's critic without its LayerNorm
   beside a working actor.
3. `noln-vit` holds. Low confidence: a transformer without LayerNorm may train poorly, but the
   encoders learn from the critic's loss alone, and the actor reads their features detached.

**Launch:** through `outputs/queue/launchers/slot_queue.sh` (tmux `rfe-queue`), from this entry's
commit in its own pinned worktree. Seed 1 of each after `noln`'s seed 2, then `limits` and `noise`,
then seed 2 of each (DEVIATION above). Expected launches: seed 1 on Wednesday at about 02:00, 07:00
and 13:00; seed 2 Wednesday about 18:00 and 24:00 and Thursday about 01:30. Each takes about 11 h
from launch to 150k sharing a GPU, and about 20 GB of RAM at start, 35 GB by 150k.

**Checks before the commit:**
- `tests/test_sweep_arms.py` and `tests/test_square_rerendered_watch.py`, CPU tier: 104 passed.
  Each new arm launches exactly its pre-registered overrides and composes to `oj5k0jqb`'s config
  but for its one switch; the watcher runs the new group's arms to a fixed 150k, with the
  divergence stop.
- `tests/test_sweep_arms.py`'s GPU tier for the ResFiT arms (GPU 1, beside two runs): 27 passed.
  - Each placement arm has no LayerNorm in its one place and the anchor's count elsewhere (ViTs: 6).
  - Each starts from the anchor's weights bit for bit, less the LayerNorms it removes.
  - Each builds and runs one update with finite losses.

---

## 2026-10-13 — NOTE — `notilt` (`dczf4k7v`) finished at 150k: partial recovery, below the base

`residual-tilt-1012`'s `notilt` (the equivariant residual with its tilt pair bounded at zero), stopped
by its watcher at its 150k eval (09:23), as the 2026-10-12 AMENDMENT set. Package finalized.

- **Evals 0, 10k, …, 150k:** 0.54, 0.02, 0.00, 0.14, 0.18, 0.14, 0.28, 0.24, 0.26, 0.36, 0.34,
  0.26, 0.18, 0.24, 0.16, 0.16.
- **10k–30k pooled:** 8 of 150. **110k–150k pooled:** 50 of 250, **0.200**, below B_sq = 0.582 by
  far more than two standard errors (0.067), and below ResFiT's anchors' 0.714. Its peak, 0.36 at
  90k, did not hold.
- **No reference yet:** no equivariant residual of its design had run past 80k. The baseline's
  seeds 2 and 3 (`equi-residual-long-1012`, launched today) give one at 150k, about Wednesday
  15:00. Whether 0.20 late is better or worse than the baseline waits for them.

---

## 2026-10-13 — LAUNCHED — the equivariant baseline's seeds 2 and 3 (`equi-residual-long-1012`); the queue retries a launch its gate stops

The slot queue, restarted on the DEVIATION and AMENDMENT above. Consoles are in `outputs/queue/`.

| Job | Launched | GPU | Gate | Trainer from |
|---|---|---|---|---|
| `equi-residual-square-long` 2 | 09:17, `arch-wide`'s slot | 1 | GO (Tier 1: 226 passed; Tier 2: 501 passed) | 09:47, `0b14c17` |
| `equi-residual-square-long` 3 | 09:28, `notilt`'s slot | 0 | GO (Tier 1: 226 passed; Tier 2: 501 passed) | 09:58, `0b14c17` |

- Each argv is `6ng62tpm`'s launch at its seed, to 150k (the 2026-10-12 PRE-REGISTERED entry). Their
  wandb IDs appear after the demos are labelled, about 1 h 15 min after launch; the 30k NOTE
  records them.
- Both gates ran their GPU tier at once and took about 25 min, against 16 alone.
- **The queue now retries a launch its gate stops,** up to 3 tries 600 s apart, holding the slot
  between tries (`outputs/queue/launchers/retry/run_job.sh`, a copy of `run_job.sh`; the queue runs
  `retry/slot_queue.sh`, an identical copy, from 10:00). A trainer that started is never relaunched.
  Why: three of six gates failed the render-guard test on the night of 2026-10-12 and passed when
  relaunched, and a stopped job would give its slot to the next job, out of the pre-registered
  order. Tested on a stand-in `submit.sh` whose first gate fails: tried twice, launched once.
- The queue's remaining order is the DEVIATION's: `noln` 2, the three placement arms' seed 1 (from
  `25b22aa`, `resfit-ln-place-1013`, watched by tmux `ln-watch`), `limits`, `noise`, the placement
  arms' seed 2.

---

## 2026-10-15 — LAUNCHED — the queue's nine ResFiT launches, 2026-10-13 to 10-14 (`noln` 2, `resfit-ln-place-1013`'s six, `limits`, `noise`); the equivariant baseline's run IDs

The slot queue (`outputs/queue/launchers/retry/slot_queue.sh`, tmux `rfe-queue`) over the order the
2026-10-13 DEVIATION set. Consoles are in `outputs/queue/`. **Every gate said GO at its first try**,
so the retry never ran. Each run package records its commit, clean, and mujoco 3.3.2, and each
argv is its pre-registered one (checked against the manifests).

| Job | Launched | GPU | From | Gate (Tier 1; Tier 2 passed) | In wandb | Run | Ended |
|---|---|---|---|---|---|---|---|
| `resfit-residual-square-noln` 2 | 10-13 17:33 | 1 | `7db124d` | 226; 413 | 18:51 | [**`xjdldyvu`**](https://wandb.ai/aden-mckinney10-university-of-central-florida/robomimic-square-final/runs/xjdldyvu) | 10-14 09:27 |
| `resfit-residual-square-noln-actor` 1 | 10-14 06:29 | 0 | `25b22aa` | 226; 528 | 07:52 | [**`vsy97qbz`**](https://wandb.ai/aden-mckinney10-university-of-central-florida/robomimic-square-final/runs/vsy97qbz) | 10-14 19:26 |
| `resfit-residual-square-noln-critic` 1 | 10-14 08:24 | 1 | `25b22aa` | 226; 528 | 09:31 | [**`z8sj1nmt`**](https://wandb.ai/aden-mckinney10-university-of-central-florida/robomimic-square-final/runs/z8sj1nmt) | 10-14 19:44 |
| `resfit-residual-square-noln-vit` 1 | 10-14 09:32 | 1 | `25b22aa` | 226; 528 | 10:41 | [**`on6qdbuo`**](https://wandb.ai/aden-mckinney10-university-of-central-florida/robomimic-square-final/runs/on6qdbuo) | 10-14 20:47 |
| `resfit-residual-square-limits` 1 | 10-14 12:59 | 0 | `304167e` | 226; 421 | 13:59 | [**`uw2aa908`**](https://wandb.ai/aden-mckinney10-university-of-central-florida/robomimic-square-final/runs/uw2aa908) | 10-14 23:51 |
| `resfit-residual-square-noise` 1 | 10-14 19:30 | 0 | `0b14c17` | 226; 501 | 20:39 | [**`spfcsqte`**](https://wandb.ai/aden-mckinney10-university-of-central-florida/robomimic-square-final/runs/spfcsqte) | 10-15 06:19 |
| `resfit-residual-square-noln-actor` 2 | 10-14 19:49 | 1 | `25b22aa` | 226; 528 | 20:55 | [**`grkiw1l1`**](https://wandb.ai/aden-mckinney10-university-of-central-florida/robomimic-square-final/runs/grkiw1l1) | 10-15 07:13 |
| `resfit-residual-square-noln-critic` 2 | 10-14 20:52 | 1 | `25b22aa` | 226; 528 | 22:01 | [**`948yzbi3`**](https://wandb.ai/aden-mckinney10-university-of-central-florida/robomimic-square-final/runs/948yzbi3) | 10-15 08:09 |
| `resfit-residual-square-noln-vit` 2 | 10-14 23:56 | 0 | `25b22aa` | 226; 528 | 10-15 01:03 | [**`vqzx1ir9`**](https://wandb.ai/aden-mckinney10-university-of-central-florida/robomimic-square-final/runs/vqzx1ir9) | running: 0.86 at 130k (08:19) |

- **The equivariant baseline's seeds 2 and 3** (2026-10-13 LAUNCHED) are
  [**`od8u86i1`**](https://wandb.ai/aden-mckinney10-university-of-central-florida/robomimic-square-final/runs/od8u86i1)
  (seed 2, in wandb 10-13 10:24, ended 10-14 08:19) and
  [**`dja2kbb6`**](https://wandb.ai/aden-mckinney10-university-of-central-florida/robomimic-square-final/runs/dja2kbb6)
  (seed 3, 10:51, ended 10-14 12:54).
- A ResFiT run took 11–13 h from launch to 150k (`xjdldyvu` 16 h, beside an equivariant run); an
  equivariant one 23–29 h, two per GPU.
- The queue ran out at 10-14 23:57. From 10-15 06:19 slots stood idle; by 08:09 three of four.
- **Config caveats:** none known on ResFiT's agent at these commits (ARCHITECTURE.md's provenance
  table covers the equivariant config).

---

## 2026-10-15 — NOTE — `equi-residual-long-1012` at 150k: the equivariant baseline's first reference; neither start gets back to the base

The pre-registered readouts (2026-10-12), each run to 150k. Evals are 50 episodes; 10k–30k pooled is
150 episodes, 110k–150k pooled 250.

| Run | Evals 0, 10k, …, 150k | 10k–30k | 110k–150k | Readout |
|---|---|---|---|---|
| `od8u86i1` baseline (last layer 1e-4), seed 2 | 0.56, 0.00, 0.00, 0.00, 0.06, 0.00, 0.12, 0.24, 0.18, 0.30, 0.26, 0.34, 0.22, 0.24, 0.22, 0.34 | 0 of 150 | 68 of 250, **0.272** | below the base |
| `dja2kbb6` baseline, seed 3 | 0.60, 0.02, 0.02, 0.10, 0.00, 0.08, 0.08, 0.24, 0.18, 0.10, 0.32, 0.16, 0.16, 0.16, 0.30, 0.30 | 7 of 150 | 54 of 250, **0.216** | below the base |
| `oav5gk76` init0, seed 2 | 0.62, 0.00, 0.00, 0.20, 0.02, 0.12, 0.28, 0.18, 0.30, 0.28, 0.24, 0.24, 0.40, 0.24, 0.44, 0.34 | 10 of 150 | 83 of 250, **0.332** | below the base |

- **All three are below B_sq = 0.582 by far more than two standard errors** (back to the base needs
  about 127 of 250), so none is back to or past the base, nor at ResFiT's 0.714. By the rule no
  arm has a seed 3 due; the baseline's seed 3 ran by the 2026-10-13 AMENDMENT.
- **Predictions:** 1 (both starts recover partly and stay below the base) was right. 2 (the starts do
  not differ by more than two standard errors at 110k–150k) was right: 0.332 against 0.272, a
  difference of 0.060 against two standard errors of 0.082.
- **Reported:**
  - the first eval at or above 0.30: 90k (`od8u86i1`), 100k (`dja2kbb6`), 80k (`oav5gk76`);
  - `train/residual_l1_magnitude` 0.002–0.005 at 0–3k, 0.052–0.054 at 120k–150k;
  - the rolled-wrist share (`eval/tilt_over_30deg_frac`) peaks at 0.78 (`od8u86i1`, 20k), and is 0.06
    or less in every run from 110k;
  - no critic value above 1.5 (`train/critic_qt` at most 0.23).
- **What it gives the earlier runs:** the baseline pooled is 122 of 500 (0.244) at 110k–150k.
  - **Late partial recovery is this design's normal course:** each run climbs from about zero at
    10k–50k to 0.2–0.34 from 70k. So `eaqorqki`'s climb to 0.30 by 80k and `notilt`'s 0.16–0.36
    were not their changes' doing.
  - `notilt` (`dczf4k7v`, seed 1, 0.200) is within the baseline's two seeds (0.216, 0.272): removing
    tilt did not help late.
  - **init0 is ahead in both matched pairs** (same seed, weights alike but the last layer): seed 1
    10k–30k, 3 of 150 (`eaqorqki`) against 1 (`6ng62tpm`); seed 2, 10 against 0, and 0.332 against
    0.272 late. The gap is about the baseline's own seed spread (0.272 against 0.216): consistent in
    direction, not established.
- **The critic and the actor, against ResFiT's working runs** (both stacks log the same keys):
  - `debug/q_data_online` stays between −0.04 and +0.03 throughout, where ResFiT's climbs to about
    0.2 by 120k; `debug/q_data_demo` reaches 0.33–0.34 (ResFiT's 0.42–0.48);
  - `debug/q_advantage` 0.001–0.002, about 1% of `debug/q_base_only_mean` (ResFiT's about 2%);
  - `train/actor_grad_norm` 0.004–0.007, flat (ResFiT's rises from 0.06 to 0.33–0.50).
- **Config caveats:** `equivariance.num_actor_layers` is logged as 2 and is dead (the actor has one
  hidden layer); `equivariance.initialize` reaches the encoder only.

---

## 2026-10-15 — NOTE — `resfit-ln-place-1013`, `noln` seed 2 and `noln-all`: ResFiT's residual needs its actor's LayerNorm; the 10k–30k rule cannot separate the critic's or the ViTs'

The pre-registered readouts (2026-10-13 PRE-REGISTERED; `noln` and `noln-all` from 2026-10-12's).
An arm breaks iff its 10k–30k evals pooled are at most 39 of 150, or three evals in a row are at or
below 0.05. The anchors (`oj5k0jqb`, `s8opblsd`): 71 and 35 of 150 at 10k–30k, 0.800 and 0.628 at
110k–150k (0.714 pooled).

| Run | Change | Seed | 10k–30k | By the rule | 110k–150k |
|---|---|---|---|---|---|
| `c1xca5uk` | `noln` (actor + critic) | 1 | 0 | breaks | 8 of 250, 0.032 |
| `xjdldyvu` | `noln` | 2 | 0 | breaks | 8 of 250, 0.032 |
| `ra6024xn` | `noln-all` (+ the ViTs) | 1 | 0 | breaks | 2 of 250, 0.008 |
| `vsy97qbz` | `noln-actor` | 1 | 0 | breaks | 36 of 250, **0.144** |
| `grkiw1l1` | `noln-actor` | 2 | 11 | breaks | 66 of 250, **0.264** |
| `z8sj1nmt` | `noln-critic` | 1 | 37 | breaks | 194 of 250, **0.776** |
| `948yzbi3` | `noln-critic` | 2 | 34 | breaks | 180 of 250, **0.720** |
| `on6qdbuo` | `noln-vit` | 1 | 50 | holds | 198 of 250, **0.792** |
| `vqzx1ir9` | `noln-vit` | 2 | 37 | breaks | 120 of 150 to 130k, 0.800 (running) |

Evals 0, 10k, …, 150k:
- `xjdldyvu`: 0.38, then 0.00–0.04 at every eval.
- `vsy97qbz`: 0.54, 0.00 to 100k but 0.02 at 60k, then 0.10, 0.08, 0.20, 0.20, 0.14.
- `grkiw1l1`: 0.66, 0.00, 0.06, 0.16, 0.42, 0.30, 0.44, 0.34, 0.46, 0.40, 0.34, 0.16, 0.34, 0.24, 0.18, 0.40.
- `z8sj1nmt`: 0.34, 0.04, 0.30, 0.40, 0.52, 0.48, 0.78, 0.66, 0.70, 0.72, 0.82, 0.74, 0.66, 0.80, 0.82, 0.86.
- `948yzbi3`: 0.58, 0.04, 0.36, 0.28, 0.56, 0.54, 0.46, 0.48, 0.60, 0.60, 0.60, 0.64, 0.66, 0.80, 0.70, 0.80.
- `on6qdbuo`: 0.50, 0.10, 0.40, 0.50, 0.44, 0.62, 0.50, 0.62, 0.66, 0.62, 0.72, 0.78, 0.84, 0.80, 0.76, 0.78.
- `vqzx1ir9` to 130k: 0.56, 0.20, 0.20, 0.34, 0.30, 0.52, 0.56, 0.52, 0.62, 0.64, 0.76, 0.84, 0.70, 0.86.

- **By the rules as registered:**
  - `noln` breaks at both seeds: question (1) of Aden's order is answered at two seeds, as is
    `noln-all`'s break at one.
  - `noln-actor` breaks at both seeds: its place is **needed**.
  - `noln-critic` breaks at both seeds (37 and 34): by the rule its place is **needed** too.
  - `noln-vit`'s seeds disagree (50, 37): **no verdict**, seed 3 due (set aside: DEVIATION below).
- **The rule's bar sits inside the anchors' own spread.** `s8opblsd`, the anchor's seed 2, scored 35
  of 150 at 10k–30k: by this bar the anchor itself breaks. So the bar cannot tell `noln-critic`
  (37, 34) or `noln-vit`'s seed 2 (37) from the anchor.
- **The late window,** registered as reported, separates every arm. `noln-critic` (0.776, 0.720) and
  `noln-vit` (0.792; 0.800 to 130k) land at or above the anchors' 0.714. `noln-actor` (0.144, 0.264)
  and `noln` (0.032 at both) are far below even the base (0.582).
- **Claude's reading, deciding nothing formally:** the actor's LayerNorm is what ResFiT's residual
  needs here. The critic's and the ViTs' make the recovery from the 10k dip faster and are not
  needed for where the run ends. With the actor's and the critic's both off it ends worse (0.032)
  than with the actor's alone (0.144, 0.264): the critic's LayerNorm partly makes up for the
  actor's.
- **The mechanism, reported** (training-batch diagnostics, means over each window):
  - Without the actor's LayerNorm the residual is at once near its cap: `train/residual_l1_magnitude`
    0.066–0.070 at 0–3k, against the anchors' 0.029–0.030 and `noln-critic`'s 0.031–0.034. That is
    prediction 1's mechanism, the saturation `9phsy6sf` showed in tuned RLPD.
  - Its actor's gradient then falls (`train/actor_grad_norm` 0.063–0.100 at 6k–10k, 0.030–0.039 at
    120k–150k), where the anchors' rises to 0.33–0.37 and `noln-critic`'s to 0.11–0.13.
  - Its online value stays near zero (`debug/q_data_online` −0.008 and +0.046 at 120k–150k); in the
    placement arms that hold it ends at 0.17–0.22 (the anchors' at 0.21 and 0.12).
- **Predictions** (2026-10-13): 1 (`noln-actor` breaks) right. 2 (`noln-critic` holds) wrong by the
  rule, right by the late window. 3 (`noln-vit` holds) right at seed 1; at seed 2 wrong by the rule,
  right by the late window.
- **For the equivariant residual:** the same symptom is not there. Its residual is 0.002–0.005 at
  0–3k and its actor's gradient 0.004–0.007, flat (the NOTE above): too slow rather than saturated.
  So "ResFiT's actor needs LayerNorm" does not yet say what the equivariant actor lacks; the next
  experiment tests RegularLayerNorm there directly (DEVIATION below).
- **Config caveats:** none known on ResFiT's agent at these commits.

---

## 2026-10-15 — NOTE — `limits` and `noise` hold: the equivariant stack's action limits and noise do not break ResFiT's residual

`resfit-equi-diff-1012`'s last two arms, seed 1 each (2026-10-12 PRE-REGISTERED and AMENDMENT).

| Run | Change | Evals 0, 10k, …, 150k | 10k–30k | Verdict | 110k–150k |
|---|---|---|---|---|---|
| `uw2aa908` | `limits` (`agent.actor.action_bound=equivariant`) | 0.52, 0.40, 0.54, 0.48, 0.58, 0.44, 0.76, 0.60, 0.72, 0.62, 0.80, 0.76, 0.84, 0.74, 0.80, 0.86 | **71 of 150** | holds | 200 of 250, **0.800** |
| `spfcsqte` | `noise` (`agent.actor.noise_units=env`) | 0.44, 0.30, 0.58, 0.56, 0.44, 0.50, 0.52, 0.76, 0.78, 0.80, 0.78, 0.88, 0.88, 0.80, 0.80, 0.76 | **72 of 150** | holds | 206 of 250, **0.824** |

- Both hold, so no seed 2 is due. Their predictions (both hold, moderate confidence) were right.
  `combined` was dropped on 2026-10-13 and never ran.
- Their 10k–30k readouts (71, 72) are level with `oj5k0jqb`'s 71, the highest of any ResFiT residual
  run, and `limits`' 0.40 is the highest 10k eval of the ResFiT runs since the render fix. One seed each, and both late windows sit inside
  the anchors' own 0.63–0.80: **"no harm", not "helps"**.
- With `fk9ayyco` (rx's range alone, 0.716): the circle bounds, rx's cap at 0.1 env units instead of
  0.036, the critic scoring `equi_clip`'s sum, the env clipping what is executed, and σ 0.05 in env
  units (rx's from 0.018) are each carried by ResFiT's residual. **The equivariant residual's action
  space is not what breaks it.**
- `limits`' `debug/residual_max_abs` reads 0.276 in the scaler's units: 0.1 env units on rx over rx's
  half-range of 0.36, as its bound sets.
- **Config caveats:** none known on ResFiT's agent at these commits.

---

## 2026-10-15 — DEVIATION and DECISION — the work turns to the equivariant residual; `noln-vit`'s seed 3 set aside; the next experiment planned

Aden's decisions, 2026-10-15, after the NOTEs above:
- **The focus is the equivariant residual.** No further ResFiT ablations now.
- **`noln-vit`'s seed 3, due by its rule (its seeds disagree), is set aside.** The rule's bar sits
  inside the anchor's own spread, and both seeds' late windows are at ResFiT's level.
- **The next experiment** (planned, not yet pre-registered): `arch-wide` again, to 150k at seed 2,
  beside each of its four parts alone on the init0 anchor (`eaqorqki`'s launch: RegularLayerNorm in
  the actor; in the critic; ResFiT's 10 heads of 2 layers; the actor at 448 fields), and init0's
  seed 3 (an AMENDMENT to `equi-residual-long-1012`). init0 is the anchor. The main readout moves to
  the 110k–150k window, against `oav5gk76`'s 0.332, because of the bar problem above. The plan, with
  every step: `docs/plans/2026-10-15-equi-arch-parts.md` (local). Its pre-registration goes here
  before any of its runs launch.
- **Disk:** 11.6 GB free at 08:30. The wandb artifact cache holds 22 GB (+18 GB since 2026-10-13);
  the plan clears it first.

---

## 2026-10-15 — NOTE — `vqzx1ir9` (`noln-vit` 2) at 150k: `resfit-ln-place-1013` is complete; the wandb artifact cache cleared

`resfit-ln-place-1013`'s last run (2026-10-13 PRE-REGISTERED) reached 150k and exited at 09:03,
package complete, from `25b22aa`, clean.

- **Evals 0, 10k, …, 150k:** 0.56, 0.20, 0.20, 0.34, 0.30, 0.52, 0.56, 0.52, 0.62, 0.64, 0.76,
  0.84, 0.70, 0.86, 0.72, 0.94.
- **10k–30k:** 37 of 150, so it breaks by the rule, as the NOTE above recorded. **110k–150k:** 203 of
  250, **0.812**.
- **`noln-vit` at both seeds:** 0.792 and 0.812 late (401 of 500, 0.802), at or above the anchors'
  0.714. The anchors' own two seeds differ by about 4 SE (0.800, 0.628), so this is not read as a
  gain. The verdict stands as the NOTE above gave it: the seeds disagree by the 10k–30k rule, so no
  verdict, and seed 3 is set aside (DEVIATION above).
- **The group at 110k–150k, complete:** `noln-actor` 0.144 and 0.264; `noln-critic` 0.776 and 0.720;
  `noln-vit` 0.792 and 0.812. The NOTE's reading stands: the actor's LayerNorm is the one ResFiT's
  residual needs.
- **Reported:** `train/residual_l1_magnitude` 0.025 at 0–3k (the anchors' 0.029–0.030);
  `train/actor_grad_norm` 0.132 and `debug/q_data_online` 0.194 at 120k–150k (means); no critic value
  above 1.5 (the largest 0.513).
- **Watchers:** `ln-watch`, `rfe-watch` and `tilt-watch` stopped at 09:06, since their groups have no
  run left. `long-watch` stays for `equi-residual-long-1012`.
- **The wandb artifact cache was cleared** with `wandb artifact cache cleanup 2GB` at about 09:05,
  with Aden's approval, once `vqzx1ir9` had exited and no wandb process was left. It reclaimed 21.7 GB, leaving 1.8 GB, so the
  disk went from 11 GB free to 31 GB. What it held, checked against every run package first:
  - copies of each run's current best upload: 7.0 GB, each also in its run package's `wandb_best/`
    (the same files by md5, in every package since 2026-10-08) and on wandb;
  - the v9 base policy: 0.4 GB, also in `artifacts/`, where the trainer loads it;
  - about 16 GB that were only there: the best uploads a run's later, better eval superseded, which
    `upload_best` had already deleted from wandb by design. For example, wandb holds `bue9dupf`'s
    v4 alone.
  - No eval history, config, video or checkpoint was in it. The cache fills again with each new
    best upload: it grew 18 GB in the two days before.

---

## 2026-10-15 — PRE-REGISTERED — the equivariant residual on Square with each part of ResFiT's layout alone, and all four again (`equi-arch-parts-1015`)

**Question** (Aden, 2026-10-15; (2b) of the 2026-10-13 question order: how to do in the equivariant
residual what LayerNorm does for ResFiT's). ResFiT's residual needs its actor's LayerNorm (NOTEs
above), and the equivariant stack's action limits and noise do not break ResFiT's residual
(`limits` 0.800, `noise` 0.824). The one equivariant residual with ResFiT's layout, `arch-wide`
(`grvwhcfr`, four changes at once), scored 0.12 at 10k, where every other one on these demos
scored 0.00–0.02 (3 of 300 pooled). It was stopped there (2026-10-13 DEVIATION). Does any of its
four parts, alone, lift the equivariant residual above init0 late? Do all four together, run to
150k?

**Seed group `equi-arch-parts-1015`**, seed 2 of each arm, in two waves (Launch, below).
- **Names:** `<arm>-<seed>`; notes `equi-arch-parts-1015/<arm>/seed<seed>`; wandb
  `robomimic-square-final` / `equi-residual-rl`.
- **The anchor** is `eaqorqki`'s launch (`square-residual-start-1011`, `0af7655`), to 150k:
  `--config-name=residual_equi_td3_square_config equivariance.residual_squash=true
  agent.critic.num_q=5 base_policy.wt_version=v9
  offline_data.name=local/robomimic-mh-square-rerendered agent.actor.actor_last_layer_init_scale=0.0
  algo.total_timesteps=150000 read_buffer_caches=false write_buffer_caches=false seed=<seed>
  debug=false`.
- At seed 2 that launch is **`oav5gk76`** (`equi-residual-square-init0-long` 2, from `0b14c17`,
  150k: 0.332 at 110k–150k). Every arm is compared with it, and at seed 2 each pairs with it.

| Arm | Change from the anchor | Wave |
|---|---|---|
| `equi-residual-square-arch-wide` | all four below: `equivariance.actor_norm=layer_norm equivariance.critic_norm=layer_norm agent.critic.num_q=10 equivariance.num_critic_layers=2 equivariance.actor_degree_channel=448`, as `equi-resfit-arch-1013`'s AMENDMENT set it (its seed 1, `grvwhcfr`, stopped at 10k) | 1 |
| `equi-residual-square-actorln` | `equivariance.actor_norm=layer_norm`: RegularLayerNorm after each of the actor's three hidden Linears (its visual projection, its trunk and its one policy layer) | 1 |
| `equi-residual-square-criticln` | `equivariance.critic_norm=layer_norm`: RegularLayerNorm after the critic's two projections (visual, trunk) and each head's hidden Linear, 2 + 5 × 1 = 7 | 1 |
| `equi-residual-square-heads` | `agent.critic.num_q=10 equivariance.num_critic_layers=2`: ResFiT's 10 heads of 2 hidden layers | 1 |
| `equi-residual-square-actor448` | `equivariance.actor_degree_channel=448`: the actor's three hidden layers at 448 regular fields | 2 |

- **RegularLayerNorm** is in the places of ResFiT's LayerNorm. It normalizes each sample over all its
  channels, as torch's does, but its scale and shift are shared by a field's 8 channels, which keeps
  it equivariant; torch's per-channel affine would not be (EQUIVARIANCE.md). It starts at scale 1 and
  shift 0, LayerNorm's own start.
- **Parameter counts,** measured 2026-10-15 on the arms' composed configs (the sweep's test builder):

  | | Encoder | Actor | Critic |
  |---|---|---|---|
  | The anchor (`oav5gk76`) | 0.627M | 0.305M | 0.839M |
  | `actorln` | 0.627M | 0.306M | 0.839M |
  | `criticln` | 0.627M | 0.305M | 0.840M |
  | `heads` | 0.627M | 0.305M | 2.819M |
  | `actor448` | 0.627M | 3.361M | 0.839M |
  | `arch-wide` | 0.627M | 3.364M | 2.824M |
  | ResFiT's residual (`oj5k0jqb`, 2026-10-12) | 0.76M | 3.86M | 23.6M |

- **Config caveats:** `equivariance.num_actor_layers` is dead (logged as 2; the actor has one policy
  layer whatever its value), so no arm changes the actor's depth, which is ResFiT's already (the
  2026-10-12 AMENDMENT). `equivariance.initialize` reaches the encoder only.

**What varies, and what is held.**
- Each arm's change alone. `arch-wide`'s four are declared as one.
- **The starting weights,** at seed 2 (`tests/test_sweep_arms.py`). The agent builds its encoder,
  then its critic, then its actor.
  - `actorln` and `criticln`: the anchor's bit for bit, less their RegularLayerNorms', which draw no
    random number.
  - `actor448`: the anchor's encoder and critic; its larger actor is drawn after them.
  - `heads` and `arch-wide`: the anchor's encoder; their larger critics draw more numbers, so their
    critics and actors differ.
  - The episodes differ between runs, as always: `seed` does not reach the envs.
- **The code** is this commit's. `git diff 0b14c17` (`oav5gk76`'s commit) touches no file under
  `resfit/`, and the equivariant agent is pinned by `tests/test_regression_agent.py`'s goldens.
- **The slots:** two runs per GPU, as `oav5gk76` ran. Wave 1 is the first wave of four equivariant
  runs at once, and two of them are larger. That changes how long the runs take, not what they
  compute.

**Rules.** The watcher enforces them (`--group equi-arch-parts-1015`).
- **To a fixed 150k,** stopped only on the divergence stop.
- **The main readout:** the 110k–150k evals pooled (250 episodes), against `oav5gk76`'s 83 of 250
  (0.332).
  - **Better than init0** iff above 0.332 by two standard errors,
    √(p(1−p)/250 + 0.332·0.668/250): **at least 105 of 250** (0.42).
  - **Back to the base** iff not below B_sq = 0.582 (500 episodes) by two standard errors,
    √(p(1−p)/250 + 0.582·0.418/500): **at least 127 of 250**.
  - **ResFiT's level** iff not below the ResFiT anchors' 0.714 (`oj5k0jqb` and `s8opblsd`, 500
    episodes) by two standard errors: **at least 161 of 250**.
- **Why the late window, not 10k–30k:** the 10k–30k bar of `resfit-ln-place-1013` sat inside seed
  spread (the ResFiT anchor's own seed 2 scored under it), and the late window separated every ResFiT
  arm cleanly (the NOTEs above). The bar of 105 is 0.088 above init0. That is about 1.5 times the
  late gap between the baseline's two seeds (0.272 and 0.216) and between init0 and the baseline at
  seed 2 (0.332 and 0.272). One seed against one seed, so a pass is followed by a seed 3.
- **Seed 3** of each arm better than init0 at 150k, the two highest if more than two qualify, from
  this commit, into the first free slot.
- **What follows:**
  - One part alone better than init0: that part is the equivariant residual's answer to (2b). Its
    seed 3, then whether it reaches ResFiT's level.
  - `arch-wide` better, and no single part: a wave of `arch-wide` less one part at a time.
  - Nothing better than init0: the encoder (the equivariant CNN against ResFiT's ViT) is the next
    difference. Then the scalar ablation of this stack as a control (STATUS, "What is not").
- **Reported, deciding nothing:**
  - the 10k eval and 10k–30k pooled, against `oav5gk76`'s 0.00 and 10 of 150, the baseline's 0 and
    7 (`od8u86i1`, `dja2kbb6`), and `grvwhcfr`'s 0.12 at 10k;
  - the first eval at or above 0.30, and the best eval (no equivariant residual on these demos has
    scored above 0.44 after step 0: `oav5gk76` at 140k);
  - the signs ResFiT's working runs show (NOTEs above; ARCHITECTURE.md, "Reading the critic and actor
    diagnostics"):
    - `train/residual_l1_magnitude` at 0–3k: about 0.03 in ResFiT's anchors, 0.002–0.005 in the
      equivariant baseline, and 0.07 in ResFiT without its actor's LayerNorm;
    - `train/actor_grad_norm`: rising in ResFiT (0.06 to 0.13–0.42), against flat (equivariant,
      0.004–0.007) or falling (ResFiT without its actor's LayerNorm);
    - `debug/q_data_online`: positive by about 5k and climbing to about 0.2 in ResFiT, against −0.04
      to +0.03 throughout in the equivariant runs;
    - `debug/q_advantage`: about 2% of `debug/q_base_only_mean` in ResFiT, about 1% in the
      equivariant runs;
  - `eval/tilt_over_30deg_frac`, and critic values above 1.5 (the watcher's FLAG).
- **No checkpoints** (87 MB each for the anchor's agent, more for the larger arms), as in
  `equi-residual-long-1012`.

**Predictions** (Claude's, before launch):
1. `actorln` is better than init0. Low confidence. ResFiT's lesson is its actor's LayerNorm, but
   ResFiT without it fails the opposite way: a residual at its cap within 3k. The equivariant
   residual is too slow instead (0.002–0.005 at 3k, its actor's gradient flat).
2. `arch-wide` is better than init0. Low-to-moderate confidence: its 0.12 at 10k, one eval of one
   seed.
3. `criticln`, `heads` and `actor448` are each not better than init0. Low confidence.
4. No arm is back to the base (127 of 250). Moderate confidence: no equivariant residual has scored
   above 0.44 at any eval after step 0.

**Launch.** Through `outputs/queue/launchers/retry/slot_queue.sh` (tmux `rfe-queue`), from this
entry's commit in its own pinned worktree. Each job goes into the first slot free for 300 s, two per
GPU. A launch its gate stops is tried again, up to 3 times 600 s apart.

| # | Job | Seed | From | Wave |
|---|---|---|---|---|
| 1 | `equi-residual-square-arch-wide` | 2 | this commit | 1 |
| 2 | `equi-residual-square-actorln` | 2 | this commit | 1 |
| 3 | `equi-residual-square-criticln` | 2 | this commit | 1 |
| 4 | `equi-residual-square-heads` | 2 | this commit | 1 |
| 5 | `equi-residual-square-actor448` | 2 | this commit | 2 |
| 6 | `equi-residual-square-init0-long` | 3 | `0b14c17` | 2 (AMENDMENT below) |

- Wave 1 fills the four free slots. Wave 2 takes the first two slots wave 1 frees. Seed 3s join after
  wave 1's readouts.
- **Watchers:** tmux `parts-watch`, from this commit, for the group; `long-watch` (from `0b14c17`)
  for init0's seed 3.
- **No smoke runs:** `grvwhcfr`'s first 10k measured `arch-wide`'s speed (about 1 h 35 min per 10k,
  beside one run) and its RAM, and the single-part arms are smaller.

**Resources and timeline.**
- **Speed:** equivariant runs took 23–29 h from launch to 150k, two per GPU (`od8u86i1`, `dja2kbb6`,
  `oav5gk76`), and about 1 h 15 min from launch to training (the gate, then labelling the demos).
  The larger arms may be slower.
- **RAM: four runs fill the machine's 125 GB.** Each trainer's own process peaks at 23–24 GB
  (`proc.memory.rssMB`: `oav5gk76`, `od8u86i1`, `dja2kbb6`, and ResFiT's `xjdldyvu`), but that leaves
  out the rest of a run. The system's own count is the one that matters: in the last four-run
  overlap (three equivariant runs at about 110k–150k and `xjdldyvu`, 10-14 03:00–06:24), available
  memory sat at 0.3–0.7 GB for three hours, and about 1 GB went to swap (225 GB of it). No run was
  lost or stopped. Wave 1's four reach that point together, near 150k, so expect the same. The
  larger arms add only megabytes of weights; the buffers dominate.
- **GPU memory:** two runs on one 24.5 GB GPU have peaked at 21–23 GB (`oav5gk76`'s; `grvwhcfr`'s
  with a ResFiT run beside it). The queue fills GPU 0 first, so `arch-wide` and `actorln` share GPU
  0, and `criticln` and `heads` GPU 1.
- **Disk:** 31 GB free after the cache cleanup above. The cache refills with each new best upload,
  so clear it again if the disk falls below about 10 GB.
- **Expected:** wave 1 launches about 10:00–10:30 on 2026-10-15, so its 150k falls on 2026-10-16,
  from about 09:30 (at 23 h) to about 15:30 (at 29 h), later if the larger arms are slower. Wave 2
  starts as wave 1's first two runs end, and reaches 150k on 2026-10-17.

**Checks before the commit:**
- `tests/test_sweep_arms.py` and `tests/test_square_rerendered_watch.py`, CPU tier: 128 passed.
  - Each new arm launches exactly its pre-registered overrides, and composes to `eaqorqki`'s config
    but for its one change.
  - `arch-wide` launches into this group with `equi-resfit-arch-1013`'s overrides.
  - The watcher runs the new group's five arms to a fixed 150k, with the divergence stop.
- **Each arm's argv against `oav5gk76`'s recorded one** (its manifest, less naming): each adds its
  change alone, and `heads` and `arch-wide` replace `agent.critic.num_q=5`.
- **A test fixed before the commit.** On its first GPU run, `heads` failed
  `test_every_input_can_move_the_critic_and_the_actor`: Q moved by 6.2e-5 with the arm's height,
  under the bar of 1e-4.
  - **It is not blind.** The test took the change of the heads' mean Q. Per head, each of `heads`'
    ten moves by at least 1.9e-3 with the height, as the anchor's five do (at least 1.8e-3). Their
    signs differ, and at torch seed 0 the mean cancels to 6.2e-5. At seeds 1 and 2 the mean is 3.7e-4
    and 2.1e-3.
  - **The fix:** the test now takes each head's change and requires every head to see every input.
    That is stricter, since a critic blind by its structure is blind in every head.
  - **The bar is re-measured:** each head moves by at least 1.1e-4 in every arm (`heads`, with the
    gripper opening), the FieldNorm critic's heads by at most 1.7e-6 with a trivial input, and the
    actor by at least 7.4e-4. So the bar moves from 1e-4 to 1e-5, between the two.
  - No run's code changes, and every earlier arm passes under the new measure.
- `tests/test_sweep_arms.py`, GPU tier on GPU 1, after the fix: **71 passed** (17 min 47 s).
  - Every arm builds and runs one update.
  - Each equivariant arm's actor commutes with the group and its critic is invariant, at its own
    sizes. Every input reaches each of its critic's heads and its actor.
  - Each part arm has its one part and the anchor's design elsewhere (the counts in the arms table).
  - At seed 2, `actorln` and `criticln` start from the anchor's weights bit for bit, less their
    RegularLayerNorms. `actor448` keeps the anchor's encoder and critic, and `heads` and `arch-wide`
    its encoder.
  - The ResFiT arms' weight tests pass with the shared helper's new argument (the norm classes to
    drop).

---

## 2026-10-15 — AMENDMENT — `equi-residual-long-1012`: init0's seed 3 now

Made before the run launches.
- **What changes:** `equi-residual-square-init0-long` seed 3 launches now, from `0b14c17` as its seed
  2 did. Before, its rule ran seed 3 only if seed 2 was back to the base at 150k, and seed 2
  (`oav5gk76`, 0.332) was not.
- **Why** (Aden, 2026-10-15):
  - It pairs with the baseline's seed 3 (`dja2kbb6`, 0.216) for a third init0-against-baseline pair.
    init0 is ahead in the other two (seed 1 at 10k–30k: 3 of 150 against 1; seed 2: 10 against 0,
    and 0.332 against 0.272 late), by about the baseline's own seed spread.
  - It gives `equi-arch-parts-1015`'s anchor a second seed.
- **Its rules and readouts** are seed 2's (2026-10-12 PRE-REGISTERED). `long-watch` enforces them.
- **Launch:** sixth in `equi-arch-parts-1015`'s queue (above), in wave 2.
