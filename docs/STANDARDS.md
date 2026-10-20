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

Anti-pattern, in the code until 2026-09-29: `use_norms` in the encoder. Every `InnerBatchNorm` was
commented out, but `equi_encoder.py` still set `bias=not use_norms`, so `use_norms=True` gave a
bias-free layer with no normalization after it — the worst of both, and almost certainly not what
was intended. `encoder_norm` replaced it; every conv stays bias-free, as in every earlier run.

When commenting out a layer, audit every other use of its flag in the same file.

### 1.4 One knob, one home `[CS]` `[HI]`

The same hyperparameter must not exist in two config classes.

Anti-pattern, live: layer count and normalization are each defined twice.

| Knob | In `QAgentConfig` | In `EquivarianceConfig` | Which path reads which |
|---|---|---|---|
| Actor layers | `agent.actor.num_layers` | `equivariance.num_actor_layers` | ResFiT's actor reads the first; the equivariant one reads neither (`num_actor_layers` is dead) |
| Critic layers | `agent.critic.num_layers` | `equivariance.num_critic_layers` | ResFiT's critic reads the first; the equivariant and ablated ones the second |
| Normalization | `agent.{actor,critic}.use_layer_norm` | `equivariance.{actor,critic}_norm` | Same split |

So `agent.actor.num_layers` is silently ignored by the equivariant actor, and
`equivariance.num_actor_layers` is silently ignored inside the actor. Setting either looks
reasonable and does nothing. From 2026-10-20 the equivariant agent can host ResFiT's networks
(`equivariance.*_family=resfit`), and refuses an `equivariance` field off its default where only
ResFiT's network sits (`check_families`); the `agent.*` fields an equivariant network ignores are
not refused.

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
the single most load-bearing line in the equivariant code.

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
- `ablate_*` — the scalar counterpart, historically. Only `ablate_equi_encoder.py` is left, holding
  the wrist camera's CNN that the equivariant and ablated encoders share. The ablated family, the
  controlled ablation, is `rl/ablated.py`.
- No prefix inside `equi_off_policy/` — shared by both paths, or the unified entry point
  (`obs_encoder.py`, `actor.py`, `critic.py`, `q_agent.py`).

### 3.3 An ablation must differ in exactly one thing `[SC]`

The control that isolates equivariance from everything else is the **ablated family**
(`equivariance.*_family=ablated`, from 2026-10-20). It is defined by export, so it differs in the
weight constraint alone: the equivariant network is built exactly as the equivariant family builds
it, then each layer is exported to its plain torch form (`rl/ablated.py`), and the forward mirrors
the equivariant one's. So it has the same depth, nonlinearities, norms, squashing, clipping, noise
and features, and it starts as the same function at the same seed. Only its updates differ: every
dense weight moves on its own, where escnn ties them to the group.

The tests prove both halves (`tests/test_ablated.py`): at the same seed the ablated stack has the
equivariant stack's random state, outputs and first critic step; after a few updates it is no longer
equivariant, while the equivariant stack still is.

What an ablation like this does not hold fixed must still be listed in its EXPERIMENTS.md entry. For
the ablated family, the dense weights are 8 times as many, so at one learning rate each network
changes faster per step (the arch-swap plan's probe: about 3 times as much change in Q, and 6.6
times as fast an actor at the same rate).

**The violation this replaced.** `use_equivariant_model=False` was meant to be that control and was
not: its scalar modules differed in six ways at once (actor depth, dropout, output squashing, return
type, the critic head's nonlinearities, and the ensemble's implementation), so any difference in
outcome could not have been attributed to equivariance. It never ran, and was removed with its two
fields on 2026-10-20.

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

### 4.5 Test on the inputs the env produces `[SC]`

An observation reaches the agent by two routes. act() gets it from the env, and the update gets it
from the replay buffer. The two formats differ: the env's images are float in [0, 1], and the
buffer's copies are uint8.

Every test, golden and fingerprint fed uint8 only. So for seven months no test could see that the
equivariant encoder turned the env's images near-blank. Every equivariant run executed a residual
unrelated to the one it trained, and almost none of it in the two IID-batch-norm checkpoints
measured (EXPERIMENTS.md, 2026-10-02 CORRECTION).

- A test of either route feeds the format that route delivers.
- `tests/test_obs_formats.py` checks that the two give the same act().
- A new observation key, or a change to how the trainer stores one, extends that file.

---

## 5. Running experiments

### 5.1 Every run gets a launch script `[SC]`

Under `resfit/rl_finetuning/shell/`, so the exact command is recoverable without digging through
wandb metadata.

### 5.2 Run names encode the variant `[SC]`

`wandb.name`, `group`, and `notes` must identify variant, task, seed, and what changed relative to
the comparison.

Since 2026-09-30 the group is the task and the method, never the experiment (Aden's decision). The
config puts every Square equivariant RL run in `equi-residual-rl`, and `train_bc_dexmg.py` puts
every Square BC run in `robomimic-square-bc`. So the run name carries the variant and seed, and
the notes carry the experiment: `equi-residual-rl-<arm>-<seed>` and
`<seed-group id>/<arm>/seed<seed>`.

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

- `train/residual_l{1,2}_magnitude` — compare against `action_scale` with `residual_squash`, and
  against 1.0, the clip, without it. Equality means the residual is pinned at its bound.
  - It is the update batch's residual, computed from buffer images, not the one act() executes.
  - Until 2026-10-02 the two differed by about 200× in the two IID-batch-norm checkpoints measured
    (rule 4.5).
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

### 6.5 No host syncs in the update path `[CS]`

A host sync makes the CPU wait until the GPU has finished all the work queued so far. The update
runs four times per env step, so one sync there is paid on every step. Before 2026-09-30 there
were 487 per env step, and they roughly doubled its time (ARCHITECTURE.md, "Why the update used to
stall").

- Compute a value that is only logged under `compute_metrics` in `QAgent.update`, and behind the
  same condition in the trainer.
- Three syncs hide in code that looks free:
  - a GPU scalar used as a Python number, such as a slice bound `x[:, idx[0]:idx[1]]`;
  - a copy from ordinary host memory, such as `torch.arange(n).to(device)`. Build the tensor on
    the device, or pin it and copy with `non_blocking=True`;
  - `.item()` inside an `assert`, or a GPU tensor in an `if`.
- `tests/test_sync_free.py` runs every config in `tools/agent_fingerprint.py`'s `CANDIDATES`
  under `torch.cuda.set_sync_debug_mode("error")`. A config that becomes a default stays there.
