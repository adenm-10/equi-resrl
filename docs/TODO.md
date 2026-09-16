# TODO

Priority order. Each item names the doc section it should update when finished.

Tags: `[SC]` scientific consistency · `[CS]` code simplicity and correctness ·
`[HI]` human interpretability. See the priority ranking in [CLAUDE.md](../CLAUDE.md).

---

## H — Housekeeping

### H1. Unreachable modules — RESOLVED 2026-09-16 `[CS]`

Five modules were reachable from neither entry point.

**Decision: `rl_utils.py` deleted.** It was untracked, imported by nothing, and — decisively —
**could not be imported at all**: `from __future__ import annotations` sat on line 15, which is a
hard `SyntaxError` in Python. The file had never once loaded in any run. Its only original content,
`TrivialLayerNorm`, is preserved verbatim in P3.4 below.

Still unreachable, still present, deferred to P3.5 (they are tracked, so deleting them is reversible
and there is no urgency):

| File | Note |
|---|---|
| `equi_off_policy/networks/ablate_equi_obs_encoder.py` | Superseded by `obs_encoder.py` in `caf83f3`. |
| `equi_off_policy/common_utils/field_norm.py` | 441 lines. `FieldNorm` also lives in `equi_rl_utils.py`. |
| `equi_off_policy/common_utils/rotation_transformer.py` | Imported by nothing. |
| `equi_off_policy/common_utils/rotation_utils.py` | Only imported by `rotation_transformer.py`, itself dead. |

No longer blocks R1.

### H2. Review two git stashes `[HI]`

`stash@{0}` on `aabde0f`, `stash@{1}` on `d9909ce`. Contents unreviewed. Worth five minutes of
`git stash show -p` before the reproduction, on the chance one holds relevant work.
**Updates:** STATUS.md.

### H3. `note.txt` and `log.txt` — RESOLVED 2026-09-16 `[HI]`

Both removed from tracking. Content migrated into the docs; recoverable via
`git show HEAD~1:note.txt` if ever needed.

## R — Reproduce the lost run

Current top priority. Pre-flight audit done 2026-09-16; most of the infrastructure is already in
place. Full reasoning in [EXPERIMENTS.md](EXPERIMENTS.md).

### Already verified ready — no action needed

| Item | State |
|---|---|
| GPUs | 2 × RTX 3090, 24 GB each, both idle. The 300k equi run ran on this exact machine. |
| Conda env | `residual`, Python 3.10.19. torch 2.9.1+cu128, torchrl 0.9.2, tensordict 0.9.1, escnn 1.0.13 — matches the run's `requirements.txt` exactly. CUDA sees both GPUs. |
| BC base policy | **No retraining needed.** `robomimic-can-bc/xhjdl8a7` still live in wandb (1072 MB, COMMITTED) and cached at `artifacts/run_xhjdl8a7_best:v7/`. Square's `3hzs5bz1` likewise. |
| Can offline buffer | Cache hash `cd03f9df` computed and present, 5.0 GB. Will be a cache hit. |
| Can online warmup buffer | Cache hash `9ff5ddd4` computed and present, 16 GB (Can, horizon 200). Will be a cache hit, so **no 10k-step env warmup and no cache write** — which also removes any parallel-seed write race. |
| Robot base probe | `env_probes/Can.json` present, `base_xy = [-0.5, -0.1]`. |
| wandb auth | Logged in as `aden-mckinney10`. |

### R0. Watch RAM, not VRAM `[SC]`

VRAM is not the constraint. **RAM is.** Each run holds the online buffer (16 GB, preallocated at
`buffer_size=200_000`) plus the offline buffer (5 GB) in CPU memory — roughly **21 GB per run**.
The machine has 62 GB total, ~58 GB available.

- 1 run: comfortable.
- **2 runs in parallel: safe (~42 GB).**
- 3 runs in parallel: ~63 GB, over budget. Do not.

So run **2 seeds in parallel, then the third** — about 2 × 35 h ≈ 3 days, versus 5 days serial.
Pin each to a different GPU with `CUDA_VISIBLE_DEVICES`. Disk has 155 GB free (83 % used); no large
new writes are expected since both caches hit, but keep an eye on it.

### R1. Tag `caf83f3` as the reproduction baseline — DONE `[SC]`

Tagged `repro-z8yoqylh-base`. The freeze stays in force until R8.

### R2. Reconstruct the config, with assumptions written down `[SC]`

`z8yoqylh`'s config is gone and confirmed unrecoverable. Rebuild from `ResidualEquiTD3CanConfig`
defaults. Two things narrow the search a lot: most `EquivarianceConfig` fields are inert, and the
three surviving 300k Can runs give a picture of what was being varied around that time
(`enc_degree_channel` 16 vs 32, `num_actor_layers` 2 vs 3 — the latter inert, `use_norms` false in
the last one).

Must be pinned and recorded: `N`, `enc_degree_channel`, `actor_degree_channel`,
`critic_degree_channel`, `action_scale`, `actor_lr`, `critic_lr`, `n_step`, `gamma`,
`num_updates_per_iteration`, `buffer_size`, `critic_warmup_steps`, `stddev_max`, `stddev_min`.

**Start from the config defaults, not from `a3e3zylp`'s overrides** — that run collapsed. Record in
EXPERIMENTS.md as a reconstruction with stated assumptions, flagged permanently as such.

### R3. Add a launch script and fix the naming `[SC]`

**There is no launch script for Can residual RL at all** — `shell/paper_runs/can/` contains only
`2_can_rlpd.sh`, and there are no equivariant scripts anywhere in the tree. Past runs were launched
from hand-typed commands kept in `note.txt`. Write:

- `shell/paper_runs/can/1_can_equi_residual_rl.sh` — takes a seed, sets `CUDA_VISIBLE_DEVICES`.
- Optionally a small wrapper that launches seed 1 and 2 together and seed 3 after.

Naming needs a **narrower** fix than first thought. The run name already includes task and seed
automatically (`{wandb.name}__{date}__{task}_n3_utd4_buf200000_off300ep_lr1e-06__seed{seed}`), so
seeds are already distinguishable. What is missing is the **variant**: equivariant and baseline runs
are both `wandb.name=residual-rl`, `group=residual-rl`. Set `wandb.name=equi-residual-rl`,
`group=equi-can-reproduction`, and put the real description in `notes` instead of the stale
`paper_runs/square/yet_to_come` that every run to date carries.

Also: `cfg.seed` defaults to `None`, which randomizes. **Pass explicit seeds** (e.g. 0, 1, 2) so the
runs are reproducible.

### R4. Pre-register the comparison — DONE `[SC]`

Written to EXPERIMENTS.md under "PRE-REGISTERED, NOT YET LAUNCHED", including six
numbered assumptions and the list of known defects deliberately left in place.

#### Original note

Write the EXPERIMENTS.md entry before launching: seed-group id, all three seeds, hypothesis, and
the metrics that decide it. A deleted or interrupted run then still leaves a record — the literal
lesson of `z8yoqylh`, now doubly so given `msfkjwab` is gone too.

**Record one caveat up front:** because both buffer caches hit, all three seeds share identical
warmup data and identical offline data. They differ only in network initialization and in online
exploration after warmup. That reduces variance, but it means the seeds are not fully independent
and the spread will understate true seed variance. It is also what `z8yoqylh` did, so it is the
right choice for a reproduction.

### R5. Smoke gate — BUILT `[SC]`

`preflight.py` implements it as a three-tier gate, wired into `submit.sh` so a launch is
blocked unless it passes. `python preflight.py` currently reports **GO**. See `tests/README.md`.

Still to do at launch: the live `debug=true` short run, whose key number is `dQ_da` — if it is
still ~2e-4 the `FieldNorm` hypothesis is wrong and the 3-day run should not start.

#### Original note

`debug=true`, a few thousand steps. Check: agent instantiates; both caches load (watch for
`found on disk`); step-0 eval reproduces Can's 0.82–0.92 base rate; `critic_loss` and `dQ_da` are in
range; no NaNs. Roughly 3 days of compute behind this gate, so spend the hour.

**Specifically check `dQ_da`.** If the `FieldNorm` hypothesis is right, `caf83f3` should show a
materially larger action gradient than the `~2e-4` that `a3e3zylp` logged. That is an early read on
whether the reproduction will work, available within minutes instead of 31 hours.

### R6. Launch the seeds `[SC]`

`residual_equi_td3_can_config`, 300k steps, three fixed seeds, identical config. Two in parallel
(one per GPU), third after. Expect ~31–33 h each based on the three comparable runs.

### R7. Baseline arm — the local copy is the only copy `[SC]`

**Revised: `msfkjwab` is deleted from wandb.** It cannot be re-pulled. The Can baseline
(0.92 → 1.00, 300k steps) survives only as `wandb/run-20260226_193630-msfkjwab/` on this disk, and
EXPERIMENTS.md already holds its numbers recomputed from that log.

Options: compare against the recorded single-seed baseline and state the asymmetry plainly, or
spend another ~3 days on 3 baseline seeds. Given Can's base policy is at 0.92 with a 1.00 ceiling,
the baseline is nearly pinned and the extra seeds buy little. **Recommend: use the recorded run,
flag the 3-vs-1 asymmetry, and spend the compute on Square instead**, where the base policy sits at
0.52 and the measurement actually discriminates.

### R8. Analyze against the pre-registration `[SC]` `[HI]`

Use `/analyze-experiment`. Success rate and episode length, mean and spread across seeds. The
question: is the flat equivariant episode length real, or was it one seed? Then lift the freeze.
**Updates:** STATUS.md, EXPERIMENTS.md, PROGRESS.md.

---

## P1 — Standards — DONE 2026-09-16

**P1.1 (extract conventions)** and **P1.2 (write STANDARDS.md)** complete.
[STANDARDS.md](STANDARDS.md) now has 28 rules across 6 areas, each citing an exemplar or a live
anti-pattern from this repo.

The significant finding from P1.1 is recorded as rule 3.3: **the scalar ablation is not a
controlled ablation.** See P4 below.

**P1.3 (Claude Code workflow)** complete. Three skills in `.claude/skills/` — `session-start`,
`session-end`, `analyze-experiment` — plus a read-only Bash allowlist in `.claude/settings.json`.

Remaining follow-up: revisit `analyze-experiment` when the wandb workflow changes (Aden flagged
this as upcoming). The skill reads run data from local `wandb/run-*/files/`, and ARCHITECTURE.md's
"reading a past run without wandb" section documents that layout.

---

## P2 — Equivariance tests — MOSTLY DONE 2026-09-16

Built as the pre-submission gate. `preflight.py` (3 tiers) plus `tests/` (7 files, 148 tests).
Full detail in [tests/README.md](../tests/README.md).

**Done**

- **P2.0 infrastructure** — pytest, fixtures at the reproduction's real config, `--gpu` marker,
  `CropRandomizer` disabled via `eval()`.
- **P2.0 import smoke test** — every module under `resfit/` imports. 2 justified skips.
- **P2.0b layout assertions** — widths agree at every `GeometricTensor` boundary;
  `action_layout` covers the action exactly once with no gaps or overlaps.
- **P2.1 group-action plumbing** — `tests/group_action.py`, derived from the physics rather than
  from the modules' own `FieldType` declarations. Both sign conventions verified against escnn.
- **P2.2 per-module tests** — encoder equivariance (exact at 90°, measured at 45°), actor
  equivariance, **critic invariance**, `GroupPooling` placement, `equi_clip` commutation. All 8
  group elements. Plus three anti-vacuity guards (nonzero actor output, action-dependent Q,
  clipping actually exercised).
- **P2.3 no-op confirmation** — all six claims in the provenance table are now executable fact,
  including the four-way `use_norms` split.
- **Measured tolerances** recorded in EQUIVARIANCE.md assumption 5.

**Result: the equivariant implementation is correct.** The declared representations match the
physics, the actor is equivariant and the critic invariant to `< 1e-4` at every group element. So
the collapse in the `7dae4925` runs was not a broken symmetry — which is what makes the `FieldNorm`
hypothesis the leading explanation.

**Still to do (P2.4, does not block the reproduction)**

- Full `QAgent.act` plus `_combine_actions`: assert the weaker property that holds when a
  non-equivariant base policy is summed with an equivariant residual.
- Negative control: `use_equivariant_model=False` must **break** equivariance — a test proving the
  tests can fail.
- Normalizer: assert isotropic scale and zero offset on every `irrep(1)` field in
  `build_equivariant_normalizer`. Currently guaranteed only by reading the code.

---

## P3 — Cleanup, gated on R8

Do not start before the reproduction completes. See the freeze in STATUS.md.

### P3.1 Wire or remove every dead config field `[CS]` `[SC]`

Make `use_norms`, `use_orth_init`, `num_actor_layers`, `num_critic_layers`, `dropout`, and
`initialize` either actually branch or not exist. Use P2.3's bit-identity tests to prove the
cleanup did not move behavior. Only after this do these become real, ablatable knobs.

### P3.2 Resolve the reachable-code rough edges `[CS]`

From ARCHITECTURE.md "Known rough edges": the `std=0` versus `std=None` divergence in
`Actor.forward`, the ignored `return_logits`, the stored-and-unread `loss_cfg`, the commented-out
`assert False` debug blocks.

### P3.3 Decide about the distributional critic `[CS]`

`agent.critic.loss` (C51 / HL-Gauss) is settable and unreachable on the equivariant path. Either
implement it or make the config reject it.

---

### P3.4 Restore normalization to the critic head, using `TrivialLayerNorm` `[CS]`

Every normalization layer in the equivariant critic is currently commented out, and its `use_norms`
flag does nothing — so the critic trains with no normalization at all, while the actor has active
batch norm. That asymmetry is a plausible contributor to the instability seen as `dQ_da ≈ 2e-4` and
residual saturation.

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

Belongs in `equi_rl_utils.py` alongside `FieldNorm`. Gated on R8 — it changes critic behavior, so it
cannot land before the reproduction. Any version of this must be covered by the P2.3 critic
invariance test, since normalization is exactly where equivariance tends to break.

### P3.5 Sweep the four remaining unreachable modules `[CS]`

`ablate_equi_obs_encoder.py`, `field_norm.py`, `rotation_transformer.py`, `rotation_utils.py`. All
tracked, so deletion is reversible. Check `field_norm.py` against `equi_rl_utils.py`'s `FieldNorm`
before removing — if the standalone copy is the better one, keep that instead. See H1.

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
  multiple runs. Is the critic providing any usable action gradient?
- **Expand to more tasks.** Square next — its base policy sits at 0.52, so it discriminates far
  better than Can's 0.92. Then the two-arm dexmg tasks, which need BC base policies trained
  (`wandb_id` is still `TODO` for TwoArmBoxCleanup and TwoArmCanSortRandom).
- **Equivariant SAC.** No SAC exists anywhere in the repo, so this is a genuine build: equivariant
  squashed-Gaussian actor (equivariant mean, invariant log-std), entropy temperature, twin critics.
  The SO(2) equivariant RL paper's headline result is equivariant SAC, so the architecture maps
  over — but it needs P2 in place to be trustworthy.
