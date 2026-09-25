# Coding and architecture standards

Rules derived from the conventions already in this repo, not invented. Where the repo is
inconsistent, the rule names the version to follow and the anti-pattern is cited from real code.

Every rule is tagged by which project priority it serves: `[SC]` scientific consistency,
`[CS]` code simplicity and correctness, `[HI]` human interpretability. Ranking is in
[CLAUDE.md](../CLAUDE.md) — when rules conflict, the higher priority wins.

---

## 1. Configuration

### 1.1 Config shape `[CS]`

Hydra plus nested frozen-style dataclasses, all under
[resfit/rl_finetuning/config/](../resfit/rl_finetuning/config/). Follow the existing pattern:

- Mutable defaults use `field(default_factory=...)`, never a bare literal.
- A task is a **subclass**, overriding only what differs. See `ResidualTD3CanConfig`.
- An optional subsystem is `T | None` with `None` meaning off. `equivariance: EquivarianceConfig | None`
  is the exemplar, and it is what `train_residual_td3.py:293` branches on.
- Register every runnable config in the `ConfigStore` block at the bottom of
  [residual_td3.py](../resfit/rl_finetuning/config/residual_td3.py#L338), and keep the registered
  name aligned with the class name.

### 1.2 A config field either branches or does not exist `[SC]` `[CS]`

**The most important rule in this document.** Architectural variants go behind a field that
actually switches at runtime. Never behind commented-out blocks.

The reason is scientific, not aesthetic: wandb logs the resolved config, so a field that is
accepted and ignored makes every run's recorded hyperparameters a false description of the network
that trained. Run `a3e3zylp` was launched with `equivariance.num_actor_layers=3`; the override did
nothing, and its wandb config says otherwise. Any comparison across runs varying such a field is
comparing identical networks.

Current violations are inventoried in the provenance table in
[ARCHITECTURE.md](ARCHITECTURE.md) and scheduled for repair in TODO P3.1, gated behind the
bit-identity tests in P2.3.

### 1.3 Beware the flag that half-survives `[SC]` `[CS]`

A worse case than a fully dead flag. When norm layers are commented out but the flag still feeds
something else, the flag looks inert and is not.

Anti-pattern, live in the code: `use_norms` in the encoder. Every `InnerBatchNorm` is commented
out, but [equi_encoder.py:125](../resfit/rl_finetuning/equi_off_policy/networks/equi_encoder.py#L125)
still sets `bias=not use_norms`. So `use_norms=True` gives a bias-free layer with no normalization
after it — the worst of both, and almost certainly not what was intended.

When commenting out a layer, audit every other use of its flag in the same file.

### 1.4 One knob, one home `[CS]` `[HI]`

The same hyperparameter must not exist in two config classes.

Anti-pattern, live: layer count and normalization are each defined twice.

| Knob | In `QAgentConfig` | In `EquivarianceConfig` | Which path reads which |
|---|---|---|---|
| Actor layers | `agent.actor.num_layers` | `equivariance.num_actor_layers` | Baseline reads the first; equivariant reads the second |
| Critic layers | `agent.critic.num_layers` | `equivariance.num_critic_layers` | Same split |
| Normalization | `agent.{actor,critic}.use_layer_norm` | `equivariance.use_norms` | Same split |

So `agent.actor.num_layers` is silently ignored on the equivariant path, and
`equivariance.num_actor_layers` is silently ignored inside the actor. Setting either looks
reasonable and does nothing.

### 1.5 Booleans are booleans `[CS]`

Anti-pattern: `layer_norm` in [off_policy/rl/actor.py:12](../resfit/rl_finetuning/off_policy/rl/actor.py#L12)
is an int used as a three-way enum — `1` means norm after every layer, `2` means after the last one
only, anything else means none. It is undocumented and only discoverable by reading `build_fc`.
Upstream code; do not copy the pattern. New three-way choices get a string enum with the options
named in the dataclass.

### 1.6 Changing a default breaks comparability `[SC]`

Changing any default in `residual_td3.py` or `rlpd.py` invalidates comparisons against every prior
run. It requires a note in [EXPERIMENTS.md](EXPERIMENTS.md) recording what changed and from which
commit.

---

## 2. Equivariant code (escnn)

### 2.1 Names match the math `[HI]` `[CS]`

If a module claims a representation, the escnn `FieldType` says so, and a test checks it. This
matters more here than usual: in `ResObsEnc`, the *grouping of tensor dimensions* is the
representation theory. The interleaved slicing that regroups a 6D rotation into three xy column
pairs is not index gymnastics — it is the claim that each pair transforms as `irrep(1)`. Get the
grouping wrong and shapes still line up, the network still trains, and the symmetry is silently
false. See [EQUIVARIANCE.md](EQUIVARIANCE.md).

Established naming, follow it: `vis_type`, `ih_type`, `prop_type`, `action_type`, `vis_ih_type`,
`vp_type` (visual projection out), `trunk_proj_type`, `enc_out_type_{critic,actor}`.

### 2.2 Equivariant versus invariant is a deliberate choice, stated per module `[HI]`

The actor is equivariant (`π(g·s) = g·π(s)`); the critic is invariant (`Q(g·s, g·a) = Q(s, a)`).
Any new module states which it is and how it achieves it. The critic achieves invariance with
`GroupPooling` at [critic.py:35-37](../resfit/rl_finetuning/equi_off_policy/rl/critic.py#L35) —
the single most load-bearing line in the equivariant code, and the change that enabled the first
successful run.

### 2.3 Use the existing `GeometricTensor` helpers `[CS]`

`slice_gt`, `cat_gts`, and `equi_clip` live in `equi_rl_utils.py`. Do not hand-roll slicing or
concatenation of `GeometricTensor`s — the helpers keep the field type attached, which is what makes
a mismatch detectable.

`equi_clip` in particular is not interchangeable with `torch.clamp`: it projects each `irrep(1)`
pair onto a disk (rotationally symmetric) rather than clamping per-axis onto a box (not
rotationally symmetric). Its docstring records why this preserves equivariance.

### 2.4 Normalization is per representation `[SC]`

Every `irrep(1)` field needs an **isotropic** scale and a **zero** offset, or the normalizer
breaks the symmetry it is feeding. `get_range_symmetric_normalizer_from_stat` is the one to use for
those fields; plain range normalizers are only safe on `trivial` fields. Full per-field table in
EQUIVARIANCE.md.

A corollary that trips people up: escnn's `FieldNorm` **cannot** be used on all-trivial features —
its own docstring says it will set them to zero. That is exactly the feature type after
`GroupPooling`. See TODO P3.4.

### 2.5 Fail at construction, not after 41 hours `[CS]`

Assert field-type and width agreement where a `GeometricTensor` is built or unwrapped. **43 of 197
failed runs in this project died with escnn types in the traceback** — shape mismatches were being
discovered by launching training runs. Construction-time assertions are TODO P2.0b.

---

## 3. The two agent families

### 3.1 Mirror or diverge, and say which `[HI]`

`off_policy/` holds the original ResFiT agent; `equi_off_policy/` holds the new work. Where a
module and its twin must differ, the divergence carries a comment naming the reason. Without that,
the correspondence table in ARCHITECTURE.md cannot be maintained and the two paths drift silently.

Good existing example: the `ResObsEnc` docstring explains why the in-hand camera is scalar in both
modes — its frame rotates with the gripper, so world-frame symmetry does not apply.

### 3.2 File prefixes mean something `[HI]`

- `equi_*` — the equivariant implementation.
- `ablate_*` — the scalar (non-equivariant) counterpart used as the controlled ablation.
- No prefix inside `equi_off_policy/` — shared by both paths, or the unified entry point
  (`obs_encoder.py`, `actor.py`, `critic.py`, `q_agent.py`).

### 3.3 An ablation must differ in exactly one thing `[SC]`

**Currently violated, and it matters.** `use_equivariant_model=False` is meant to be the control
that isolates equivariance from everything else. It is not, because the scalar and equivariant
modules differ in several ways at once:

| | equivariant | scalar ablation |
|---|---|---|
| Actor hidden layers | 1, hardcoded ([actor.py:68-72](../resfit/rl_finetuning/equi_off_policy/rl/actor.py#L68)) | `num_layers`, honored in a loop ([ablate_equi_actor.py:60](../resfit/rl_finetuning/equi_off_policy/rl/ablate_equi_actor.py#L60)) |
| Actor dropout | ignored | applied when `> 0` |
| Actor output squashing | **none** (`NormNonLinearity` commented out) | **`Tanh()`** |
| Actor return type | tensor, clipped by `equi_clip` | `TruncatedNormal` distribution, clipped on `.sample()` |
| Critic head | `Linear → GroupPooling → Linear`, **no ReLU** | `Linear → ReLU → Linear → ReLU → Linear` |
| Critic ensemble | `ModuleList` loop plus `torch.stack` | `vmap` over `stack_module_state` |

So running the ablation today would compare a 1-layer unsquashed equivariant actor against a
2-layer `Tanh`-squashed scalar one, and a 0-ReLU critic head against a 2-ReLU one. Any difference
in outcome could not be attributed to equivariance.

**Before the ablation is run as a control, the two stacks must be matched on depth, nonlinearity
count, output squashing, and clipping semantics** — with every remaining difference listed in the
EXPERIMENTS.md entry. Tracked as TODO P4. Until then the ablation produces an uninterpretable
result, and the "run the scalar ablation" task is blocked on this, not merely on compute.

### 3.4 Know where normalization happens on your path `[HI]`

Two different answers depending on the path, and confusing them produces silently wrong statistics:

- **Baseline:** `StateStandardizer` and `ActionScaler` built from dataset stats and applied
  *outside* the network.
- **Equivariant:** those become `IdentityStandardizer` / `IdentityScaler`, and normalization moves
  *inside* the encoder via `build_equivariant_normalizer` and `ResObsEnc.set_normalizer`. It has to
  move, because normalization must be applied per representation (rule 2.4).

---

## 4. Code hygiene

### 4.1 No dead parameters `[CS]`

A constructor parameter that is stored but never read gets wired up or removed. A keyword argument
that is accepted and ignored is a silent-wrong-answer bug.

Live violations: `dropout` and `loss_cfg` in the equivariant `Critic`; `dropout` in the equivariant
`Actor`; `return_logits` in `Critic.forward`, which a caller can pass and quietly have ignored.

### 4.2 Every module must import `[CS]`

`rl_utils.py` sat in the tree for roughly five months with `from __future__ import annotations` on
line 15 — a hard `SyntaxError`. Nothing caught it because nothing imported it. The import smoke
test in TODO P2.0 makes this enforced rather than aspirational.

`from __future__` imports go first, before anything else.

### 4.3 Unreachable means deleted or documented `[CS]`

A file nothing imports gets removed, or gets a header comment saying why it is being kept. Five
unreachable modules accumulated silently; three duplicated reachable code. One is now deleted, four
are pending in TODO P3.5.

### 4.4 No commented-out alternatives left in place `[CS]` `[HI]`

If a variant is worth keeping, it goes behind a config branch (rule 1.2) or into a doc. Commented
code has no tests, no type checking, and no way to tell whether it still works. The current
equivariant actor and critic carry several blocks of it, which is how the dead config fields came
to exist.

---

## 5. Running experiments

### 5.1 Every run gets a launch script `[SC]`

Under `resfit/rl_finetuning/shell/`, so the exact command is recoverable without digging through
wandb metadata.

### 5.2 Run names encode the variant `[SC]`

`wandb.name`, `group`, and `notes` must identify variant, task, seed, and what changed relative to
the comparison.

Anti-pattern: every run to date is `name=residual-rl`, `group=residual-rl`,
`notes=paper_runs/square/yet_to_come` — equivariant and baseline, Can and Square alike. Variants
can only be told apart by parsing `--config-name` out of `wandb-metadata.json`. Three seeds under
this scheme would be indistinguishable in the wandb UI.

### 5.3 A comparison is at least three seeds `[SC]`

A comparison is a **set** of ≥3 seeded runs sharing a seed-group id, recorded as a set. Single-seed
numbers may be recorded but never compared.

Why: in the project's one equivariant-versus-baseline comparison, the equivariant success rate
oscillates by roughly ±0.06 *within a single run*, while the gap under discussion was 0.08. That
gap is not separable from seed noise. Use fixed, recorded seeds rather than random ones.

### 5.4 Pre-register before launching `[SC]`

Write the EXPERIMENTS.md entry **before** the run starts: seed-group id, seeds, hypothesis, and
which metrics decide it. Two reasons — a deleted or interrupted run still leaves a record, which is
the literal lesson of the lost `z8yoqylh` run; and pre-registering stops the comparison being
reshaped after the numbers are in.

### 5.5 Report step-0 alongside best and final `[SC]`

The step-0 evaluation is **the base BC policy alone**, because
`actor_last_layer_init_scale = 0.0` makes the residual start at exactly zero. It is the bar, not a
result. A run whose best equals its step-0 learned nothing — its best moment was before training
began, and several runs in the log look successful if you read only `best`.

### 5.6 EXPERIMENTS.md is append-only `[SC]`

Never edit or delete a past entry. A wrong entry gets a new entry correcting it, with the original
left in place.

---

## 6. Logging

### 6.1 Use the existing namespaces `[HI]`

`train/` for per-update quantities, `eval/` for evaluation rollouts, `debug/` for diagnostics,
`timing/` for the `TrainingTimer` output, `value/` for Q-value artifacts.

Anti-pattern: `train/` and `training/` are both in use and mean nearly the same thing —
`train/actor_loss_base` and `train/residual_l2_magnitude` sit alongside `training/actor_lr`,
`training/global_step`, `training/episode_return`. Prefer `train/` for per-update metrics and
`training/` only for run-level state (step counters, schedules). Do not add a third spelling.

### 6.2 Time new stages through `TrainingTimer` `[HI]`

`with timer.time("stage_name"):` automatically produces `timing/<stage>_percentage`,
`_avg_ms`, and `_total_s`. Do not hand-roll timing.

### 6.3 Log the diagnostics that have caught real failures `[SC]`

These three found the residual-saturation failure mode and should stay logged on any new agent:

- `train/residual_l{1,2}_magnitude` — compare against `action_scale`. Equality means the residual is
  pinned at its clip boundary.
- `debug/dQ_da_mean_abs` — the critic's action gradient. Around `1e-4` means the actor is ascending
  a nearly flat surface.
- `train/critic_loss` — a low, stable loss alongside a flat `dQ_da` means the critic is fitting
  something with no useful action dependence.

### 6.4 One upload per run: the best, replaced in place `[SC]` `[HI]`

A run uploads exactly one wandb artifact, `run_<id>_best`: the best model and the video of the eval
that produced it, through `upload_best`. Everything else it produces goes in its run package under
`outputs/runs/` (ARCHITECTURE.md, "Run packages"). Scalars, histograms and `value/` plots still log
to wandb as before.

Why: per-eval videos and per-step model uploads filled wandb storage, while the end-of-run cleanup
deleted the only local copy — backwards for a project that has already lost runs to wandb deletion.
A new file a run should keep goes into the package, not into a new upload.
