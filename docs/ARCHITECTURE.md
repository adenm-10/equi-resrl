# Architecture

A map of the repo: what each piece owns, how a training step flows, and where to look for things.
For the symmetry design specifically, see [EQUIVARIANCE.md](EQUIVARIANCE.md).

## The shape of the project

This repo is a fork of ResFiT (Amazon's "Residual Off-Policy RL for Finetuning Behavior Cloning
Policies") with an equivariant variant added alongside the original. The original code is mostly
untouched; the new work lives in `resfit/rl_finetuning/equi_off_policy/`.

Two stages of training:

1. **Behavior cloning.** Train a base policy (ACT or Diffusion) on a robomimic/dexmg dataset.
   Weights get uploaded to wandb as an artifact.
2. **Residual RL.** Freeze the base policy. Train a small residual policy with TD3 that adds a
   correction to the base policy's action. The residual is what we make equivariant.

Stage 1 is already done for the tasks in play and its checkpoints are cached locally, so day-to-day
work is entirely stage 2.

## Entry points

| Script | Stage | Notes |
|---|---|---|
| [resfit/lerobot/scripts/train_bc_dexmg.py](../resfit/lerobot/scripts/train_bc_dexmg.py) | BC | argparse CLI. Launchers in `resfit/lerobot/shell/`. |
| [resfit/rl_finetuning/scripts/train_residual_td3.py](../resfit/rl_finetuning/scripts/train_residual_td3.py) | Residual RL | Hydra CLI. **The main script.** 1316 lines. |
| [resfit/rl_finetuning/scripts/train_rlpd_dexmg.py](../resfit/rl_finetuning/scripts/train_rlpd_dexmg.py) | RLPD baseline | Upstream's RLPD trainer. Not used: tuned RLPD runs through the main script with `base_policy=None` ("Residual or vanilla", below). |

### The one fork that matters

[train_residual_td3.py:293](../resfit/rl_finetuning/scripts/train_residual_td3.py#L293):

```python
equivariant_run = not cfg.equivariance is None
```

If the config has an `equivariance` block, the script builds the equivariant agent from
`equi_off_policy/`. If not, it builds the original ResFiT agent from `off_policy/`. This single
boolean is the whole switch between the baseline and the new work. Everything downstream branches
off it.

There is a second, independent switch *inside* the equivariant path:
`cfg.equivariance.use_equivariant_model`. When false, `QAgent` builds the "scalar ablation" stack
(`ablate_equi_actor.py`, `ablate_equi_critic.py`) — same data pipeline, same feature layout, same
training loop, but plain tensors instead of group representations. The intent is a controlled
ablation isolating *equivariance* from everything else.

**It does not currently isolate it.** The two stacks differ in depth, nonlinearity count, output
squashing, and clipping semantics as well as in equivariance — six differences, listed in
[STANDARDS.md](STANDARDS.md) rule 3.3. Running it as a control today would give an
uninterpretable result. Matching the stacks is tracked in TODO P4 and blocks the ablation
experiment.

### Residual or vanilla

A third switch, independent of the other two, was added on 2026-10-04 (EXPERIMENTS.md, CODE):

```python
residual_run = cfg.base_policy is not None
```

`base_policy=None` runs vanilla RL. There is no base policy, and the actor outputs the whole action.
That is the ResFiT paper's "Tuned RLPD", which `TunedRLPDConfig` sets up. Both agents run it. A
vanilla run differs from a residual one at exactly these places, each a branch on `residual_run`:

| Where | Residual run | Vanilla run |
|---|---|---|
| Base policy | loaded from wandb, for training and for evals | none |
| `ActionScaler` (ResFiT path) | range widened by `action_scale`, room for base + residual | not widened, as upstream's RLPD |
| Env wrapper | `BasePolicyVecEnvWrapper`: adds the base action, steps base + residual | `NormalizedVecEnvWrapper`: normalizes, steps the action |
| Buffer observation keys | state and base action | state |
| Agent | `residual_actor=True` | `residual_actor=False`: the actor reads no base action |
| Offline labels | the base policy's action on each demo frame | none |
| Warm-up actions | base action + U(±0.2) | U(±1) |
| Cache keys | unchanged | add `base_policy: None` and `normalization` (`identity` or `dataset`) |
| Logging | `train/residual_l*`, `histograms/residual_actions` | `train/action_l*`, `histograms/actions`, as upstream |

- **At startup** a vanilla run refuses any residual-only field that is on
  (`residual_only_fields_set` in `config/residual_td3.py`).
- **The ResFiT agent's** non-residual mode is upstream's, the one `train_rlpd_dexmg.py` uses.
- **The equivariant agent's is new.** `ResObsEnc(base_action_input=False)` gives the actor the
  critic's features, and `Actor(residual_actor=False)` drops the base action from its trunk and
  policy inputs. Its output bound is `residual_squash`, which then bounds the whole action. The
  scalar ablation stays residual-only and refuses at construction.
- **Against upstream's `train_rlpd_dexmg.py`,** the training steps are the same: 4 updates per env
  step with the actor on the last, the same warm-up, buffers and exploration. Three things differ:
  - replay sampling is uniform through torchrl's prioritized buffer at α = 0, not a
    `RandomSampler`, so it draws other random numbers;
  - evals fall on every multiple of 10k, the last step included;
  - the run keeps its package instead of deleting it (STANDARDS 6.4).

## Config system

Hydra + nested dataclasses. All in [resfit/rl_finetuning/config/](../resfit/rl_finetuning/config/).

```
RLPDDexmgConfig                       (rlpd.py — task, cameras, envs, eval settings)
└── ResidualTD3DexmgConfig            (residual_td3.py — adds base_policy, offline_data, equivariance)
    ├── ResidualTD3CanConfig                 task=Can,    equivariance=None
    ├── ResidualTD3SquareConfig              task=Square, equivariance=None
    ├── ResidualEquiTD3CanConfig             task=Can,    equivariance=EquivarianceConfig(N=8)
    ├── ResidualEquiTD3SquareConfig          task=Square, equivariance=EquivarianceConfig(N=8)
    ├── ResidualTD3BoxCleanConfig            task=TwoArmBoxCleanup (two-arm, dexmg)
    │   ├── ResidualEquiTD3BoxCleanConfig    equivariance=EquivarianceConfig(N=8, n_arms=2)
    │   ├── ResidualTD3CoffeeConfig          task=TwoArmCoffee
    │   └── ResidualTD3TwoArmCanSortConfig   task=TwoArmCanSortRandom
    └── TunedRLPDConfig                      base_policy=None: vanilla RL, the residual design made vanilla
        ├── TunedRLPDSquareConfig            task=Square, ResFiT's stack (10 critic heads)
        │   └── EquiTunedRLPDSquareConfig    + equivariance=EquivarianceConfig(N=8), 2 critic heads
        └── TunedRLPDCanConfig               task=Can
            └── EquiTunedRLPDCanConfig       + equivariance=EquivarianceConfig(N=8), 2 critic heads
```

Sub-configs live inside these: `ResidualTD3AlgoConfig` (`algo.*`), `QAgentConfig` (`agent.*`),
`EquivarianceConfig` (`equivariance.*`), `OfflineDataConfig`, `BasePolicyConfig`, `WandBConfig`.

### `--config-name` to dataclass

Registered at [residual_td3.py:338-347](../resfit/rl_finetuning/config/residual_td3.py#L338):

| `--config-name=` | Class | Task | Equivariant |
|---|---|---|---|
| `residual_td3_can_config` | `ResidualTD3CanConfig` | Can | no |
| `residual_td3_square_config` | `ResidualTD3SquareConfig` | Square | no |
| `residual_equi_td3_can_config` | `ResidualEquiTD3CanConfig` | Can | yes |
| `residual_equi_td3_square_config` | `ResidualEquiTD3SquareConfig` | Square | yes |
| `residual_td3_dexmg_config` | `ResidualTD3DexmgConfig` | (base) | no |
| `residual_td3_box_clean_config` | `ResidualTD3BoxCleanConfig` | TwoArmBoxCleanup | no |
| `residual_equi_td3_box_clean_config` | `ResidualEquiTD3BoxCleanConfig` | TwoArmBoxCleanup | yes |
| `residual_td3_coffee_config` | `ResidualTD3CoffeeConfig` | TwoArmCoffee | no |
| `residual_td3_two_arm_cansort_config` | `ResidualTD3TwoArmCanSortConfig` | TwoArmCanSortRandom | no |
| `tuned_rlpd_square_config` | `TunedRLPDSquareConfig` | Square | no; vanilla RL |
| `tuned_rlpd_can_config` | `TunedRLPDCanConfig` | Can | no; vanilla RL |
| `equi_tuned_rlpd_square_config` | `EquiTunedRLPDSquareConfig` | Square | yes; vanilla RL |
| `equi_tuned_rlpd_can_config` | `EquiTunedRLPDCanConfig` | Can | yes; vanilla RL |

## Config field provenance

**This is the most important table in this document.** Several `EquivarianceConfig` fields have
been accepted, logged to wandb, and then ignored. A run's wandb config then does not describe the
network that trained. Verified by reading the code at the 2026-09-29 sweep commit; `tests/test_no_ops.py`
makes each row executable.

| Field | Default | Encoder | Critic | Actor | Verdict |
|---|---|---|---|---|---|
| `use_equivariant_model` | `True` | — | — | — | **Live** — picks equi vs. scalar-ablation stack |
| `N` | `8` | used | used (via group) | used (via group) | **Live** |
| `enc_degree_channel` | `32` | `n_out` | — | — | **Live** (last run overrode to 16) |
| `ih_out_channel` | `32` | wrist width | — | — | **Live** from 2026-09-29. Before, the wrist width was `enc_degree_channel` |
| `actor_degree_channel` | `128` | — | — | `hidden_dim` | **Live** |
| `critic_degree_channel` | `128` | — | `hidden_dim` | — | **Live** |
| `initialize` | `True` | passed through | hardcoded `True` | hardcoded `True` | **Partial** — only reaches the encoder |
| `use_norms` | `True` | — | — | — | **Scalar ablation only** from 2026-09-29. Before, it reached two of the actor's three norm sites and set the encoder's last conv to `bias=not use_norms` |
| `actor_norm` | `none` | — | — | all 3 sites | **Live** from 2026-09-29. Every run before 2026-09-30 had `iid_bn`, the default until then |
| `critic_norm` | `none` | — | every hidden layer | — | **Live** from 2026-09-29 |
| `critic_pooled_norm` | `none` | — | each head, between `GroupPooling` and its output Linear | — | **Live** from 2026-10-10 |
| `encoder_norm` | `none` | every agentview conv | — | — | **Live** from 2026-09-29. Every conv is bias-free whatever its value |
| `use_orth_init` | `True` | — | — | live | **Live**, actor only |
| `critic_orth_init` | `False` | — | live | — | **Live** from 2026-09-29 |
| `encoder_orth_init` | `False` | agentview only | — | — | **Live** from 2026-09-29 |
| `critic_last_layer_init_scale` | `None` | — | each head's output Linear | — | **Live** from 2026-09-29 |
| `use_visual_proj` | `True` | — | live | live | **Live** from 2026-09-29 |
| `residual_squash` | `False` | — | — | live | **Live** from 2026-09-29 |
| `residual_field_bound` | `{}` | — | — | the mean's bound per field (`Actor.bound`) | **Live** from 2026-10-12. A field named gets its own bound in env units in place of `agent.actor.action_scale`; the empty default is every earlier run, bit for bit. Residual actor only; the scalar ablation refuses it |
| `frozen_target_critic` | `False` | — | target critic's mode | — | **Live** from 2026-09-29. `True` reproduces every earlier equivariant run, whose target critic never changed (EXPERIMENTS.md, 2026-09-29 CORRECTION) |
| `float_images_as_uint8` | `False` | act()'s image scaling, both branches | — | — | **Live** from 2026-10-02. `True` reproduces every earlier equivariant and scalar-ablation run, whose act() saw near-blank images (EXPERIMENTS.md, 2026-10-02 CORRECTION) |
| `num_actor_layers` | `2` | — | — | ignored | **DEAD** — actor is hardcoded to 1 hidden layer |
| `num_critic_layers` | `1` | — | hidden layers per head | — | **Live** from 2026-09-29. Before, it was dead with default `2`, and every head had 1 hidden layer |

A field added on 2026-09-29 is absent from earlier runs' configs; read it at its default, except
`frozen_target_critic`, which those runs had in effect as `True`, and `actor_norm`, which every run
before 2026-09-30 had as `iid_bn` (EXPERIMENTS.md, 2026-09-30 CODE). `float_images_as_uint8`, added on
2026-10-02, was in effect `True` in every run before then.

Also dead: `agent.actor.dropout` and `agent.critic.dropout` are threaded into both modules and
never used. `agent.critic.loss` reaches the equivariant critic as `loss_cfg`, is stored, and is
never read — so the distributional critic (C51 / HL-Gauss) is unreachable on the equivariant path
even though it is still settable in config. `Critic.forward`'s `return_logits` is accepted and
ignored.

**The actor's BC term** (`agent.bc_loss_coef`, live in `train_residual_td3.py` since 2026-10-06,
TODO B14). At a positive coefficient the trainer passes each actor update the batch's demo half,
as upstream's `train_rlpd_dexmg.py` does, and `QAgent.update_actor_rft` adds
`bc_loss_coef × Σ_axes (π(s) − a_demo)²`, averaged over the demo rows, to the actor's loss. The BC
loss sees the encoder's features detached (`bc_backprop_encoder`, upstream's default False). At 0
the actor takes `QAgent.update_actor`, as every run before 2026-10-06 did. The trainer refuses a
positive coefficient for a residual run (upstream's BC loss asserts a vanilla actor), for the
equivariant agent (untested), and with no demos in the batch. It refuses `agent.bc_loss_dynamic`
outright: its weight needs IBRL's `act_method`, which `QAgentConfig` does not have.

**The env's frames through the demos' codec** (`encode_env_frames_like_demos`, top level, from
2026-10-08, default `False`). `True` passes every camera frame the env returns through the AV1
codec that stored the demos (`resfit/dexmg/environments/demo_codec.py`), in
`RobosuiteGymWrapper._process_obs`, so that rollouts, evals and the online buffer see coded frames;
`render()` does not. It reaches both envs through `get_envs`, and adds a field to the online cache
key only when on. `False` is every earlier run, bit for bit (EXPERIMENTS.md, 2026-10-08 CODE).

**Re-rendered demos** (`offline_data.name=local/robomimic-mh-{can,square}-rerendered`, from
2026-10-08). Local LeRobot datasets in the LeRobot home, built by `tools/rerender_demos.py` from
robomimic's v1.5 state files: each demo state set into our env's scene and observed through
`RobosuiteGymWrapper._process_obs`, the frames stored as PNG. Every frame after an episode's first,
and its observation.state, is taken as robosuite's `step()` leaves the state, with the body
positions one 2 ms physics substep behind it, as every online observation is; actions, done flags
and timestamps are the source datasets'. The trainer loads them as it loads the source datasets,
and the offline cache key records the name; the env and the codec switch are untouched
(EXPERIMENTS.md, 2026-10-08 CODE, "re-rendered demos"). Provenance: each dataset's
`meta/rerender.json`.

The scalar ablation (`use_equivariant_model=False`) raises at construction if any field that only
the equivariant networks read is off its default.

### What the networks actually are at HEAD, at the defaults

Critic Q-head, per ensemble member
([EquiHeadMLP](../resfit/rl_finetuning/equi_off_policy/rl/critic.py#L11)):

```
[Linear(→ hidden, regular repr) → critic_norm → ReLU] × (num_critic_layers − 1)
→ Linear(→ hidden, regular repr) → critic_norm → GroupPooling [→ TrivialLayerNorm]
→ Linear(pooled → 1, trivial repr)
```

`TrivialLayerNorm` is there only with `critic_pooled_norm=layer_norm` (from 2026-10-10): torch's own
LayerNorm on the pooled features, which are invariant, so it needs no equivariant form. It bounds
each head's Q by its output Linear's and its own weights, whatever the input (RLPD's reason for a
LayerNorm before the critic's last layer).

At the default (one hidden layer, no norm) that is `Linear → GroupPooling → Linear`, with no
activation before the pooling. `GroupPooling` (a max over the group channels of each field) is
itself nonlinear, so the head is not purely linear. This `GroupPooling` layer is what makes the Q
output invariant rather than equivariant, and must be preserved. **This head is not the one
`7dae4925` had**: there `FieldNorm` and the ReLUs were still active. Why that matters, and why the
earlier "`FieldNorm` broke the critic" reasoning no longer holds, is in
[EQUIVARIANCE.md](EQUIVARIANCE.md), "What actually changed".

Actor policy head ([actor.py](../resfit/rl_finetuning/equi_off_policy/rl/actor.py)):

```
Linear(in → hidden) → actor_norm → ReLU → Linear(hidden → action reps) [→ equi_tanh] → × action_scale
```

One hidden layer. `actor_norm` also follows the visual and trunk projections. Its default, `none`
since 2026-09-30, normalizes at none of the three sites, as in the SO(2) equivariant RL code this
project follows. Every run before then had `IIDBatchNorm1d` at all three. `equi_tanh` is there only with
`residual_squash=True`; without it the mean is not bounded by `action_scale`.

Normalization options (`equi_rl_utils.hidden_norm`): `iid_bn` is escnn's `IIDBatchNorm1d`;
`layer_norm` is `RegularLayerNorm`, one mean and variance over all of a sample's channels, as
ResFiT's LayerNorm, with a scale and shift shared by each field's channels; `field_norm` is
`FieldNorm`, which normalizes each field by its own statistics. The encoder's `inner_bn` is escnn's
`InnerBatchNorm`. After pooling (`equi_rl_utils.pooled_feature_norm`), `critic_pooled_norm=layer_norm`
is `TrivialLayerNorm`. The ResFiT baseline has LayerNorm after every hidden layer of its actor and
critic, and inside its ViT encoders, where the equivariant CNN encoder has no LayerNorm option.

## The two stacks side by side

As the fixed-env baselines of 2026-10-07 run them (TODO B15; EXPERIMENTS.md, the pre-registration
of that date): ResFiT's stack from `off_policy/`, and the equivariant stack from `equi_off_policy/`
at Aden's design (`residual_squash=true`, 10 heads, `num_critic_layers=2`,
`critic_norm=field_norm`, `actor_norm=none`). Within a task both take the same recipe, so a pair of
arms compares everything in this table, not equivariance alone (STANDARDS 3.3).

| | ResFiT | Equivariant |
|---|---|---|
| Front camera | a ViT (patch 8, depth 1, width 128) | `EquivariantResEncoder76Cyclic`, 32 regular fields of C8 |
| Wrist camera | its own ViT | `ResEncoder76InHand`, 32 scalar channels: not equivariant, as a wrist view turns with the hand |
| Augmentation | `RandomShiftsAug`, pad 4 | `CropRandomizer` to 76×76, centred in eval mode |
| Normalization | dataset statistics outside the agent (`StateStandardizer`, `ActionScaler`) | per representation inside the encoder, xy about the robot base; the outer scalers are identity |
| Critic | per-camera features → spatial-embedding trunk → 10 heads × 2 hidden layers (1024, LayerNorm) | visual projection → trunk → 10 heads × 2 hidden layers (128 regular fields, `FieldNorm`) → `GroupPooling` → Q |
| Actor | projection (128) → 2 hidden layers (1024, LayerNorm) → Tanh | visual projection → trunk → 1 hidden layer (128 regular fields, no norm) → `equi_tanh` |
| Action bound | per axis: a box | per pair that rotates: a disk (below) |
| Residual's last layer at init | scaled by 0 | scaled by 1e-4, or 0 in the zero-init arm |

**The disk bound.** The equivariant action has two pairs that turn with the group, translation xy
(dims 0:2) and rotation rx-ry (3:5), and three components that do not (z, rz, gripper). A bound on
a pair has to be the same in every direction to commute with the rotation, so `equi_tanh` and
`equi_clip` bound each pair's length by 1, a disk, and each other component to [-1, 1]. The env's
action space is the box [-1, 1]^7, and ResFiT's Tanh and clamp bound each axis to it. So:
- **The equivariant actor cannot output a box corner.** In the demos, measured on 2026-10-07 over
  every frame (Can 62,756, Square 80,731), the xy pair is longer than 1 in **14.9% of Can's frames
  and 3.2% of Square's**, up to √2 (both axes at 1: a fast diagonal move); the rx-ry pair in 0.0% and
  0.1%. Within the disk a diagonal move is at most 0.71 per axis, against 1 for ResFiT.
- **Tuned RLPD** (vanilla) bounds the whole action this way. Its critic still trains on the demos'
  actions as recorded, corners included; only the actor's own actions are held to the disk.
- **The residual** bounds `base + residual` this way where the critic scores it
  (`_combine_actions`), but the buffer stores the unclipped sum and the env clips that per axis, to
  the box. So the critic is trained on actions that differ from the ones executed whenever the sum
  leaves the disk (TODO B13).

Kept as a stack difference: a box bound would break equivariance, and changing the bound would
change every equivariant run's updates.

**The actor's depth.** `equivariance.num_actor_layers` is dead (provenance table above): whatever
its value, the actor is visual projection → trunk → 1 hidden layer, each 128 regular fields. That
is ResFiT's actor depth counting the projection (a projection, then two layers), and one layer
fewer per head than this critic at `num_critic_layers=2`. Kept as built (Aden, 2026-10-07).

**Shared by both stacks:** the trainer and its replay (n-step 3, γ 0.99, batch 256 with half demos,
4 updates per env step, a 200k buffer), the cameras, the env, the base policy, the reward, and the
eval.

## Module map

### `resfit/rl_finetuning/equi_off_policy/` — the new work

| File | Role | Reachable from entry point? |
|---|---|---|
| `rl/q_agent.py` | `QAgent`. Owns encoder + actor + critic + targets, all TD3 update logic. The hub. | yes |
| `rl/actor.py` | Equivariant residual actor. Outputs `action_scale * mu` in the action representation. | yes |
| `rl/critic.py` | Equivariant Q-ensemble. Group-pools to an invariant scalar. | yes |
| `rl/equi_rl_utils.py` | Shared helpers: `slice_gt`, `cat_gts`, `equi_clip`, `apply_deltaortho_init`, `scale_final_equi_layer`, `FieldNorm`, `HLGaussLoss`, `C51Loss`, `schedule`, `TruncatedNormal`. Imported with `*` by actor and critic. | yes |
| `rl/ablate_equi_actor.py` | Scalar (non-equivariant) actor for the controlled ablation. | yes |
| `rl/ablate_equi_critic.py` | Scalar critic for the ablation. | yes |
| `networks/obs_encoder.py` | `ResObsEnc`. Builds field types, encodes images, normalizes and lays out proprioception and base action. Handles both equivariant and scalar paths. | yes |
| `networks/equi_encoder.py` | `EquivariantResEncoder76Cyclic` — the C_N-equivariant vision backbone for the agentview camera. | yes |
| `networks/ablate_equi_encoder.py` | `ResEncoder76` (scalar agentview) and `ResEncoder76InHand` (in-hand, used in **both** paths). | yes |
| `networks/equi_normalizer.py` | `LinearNormalizer`, `build_equivariant_normalizer`, `detect_robot_bases`, and the per-arm field names. | yes |
| `networks/ablate_equi_obs_encoder.py` | **Unreachable.** Superseded by `obs_encoder.py`. | **no** |
| `common_utils/crop_randomizer.py` | Random crop 84×84 → 76×76. | yes |
| `common_utils/escnn_fixes.py` | Two fixes to escnn 1.0.11 that remove its host syncs, exact to its outputs: basis expansions hash by identity, and a `GroupPooling` that slices with Python ints. See "Why the update used to stall". | yes |
| `common_utils/module_attr_mixin.py` | Device/dtype helper base class. | yes |
| `common_utils/tensor_util.py` | Tensor helpers used by the crop randomizer. | yes |
| `common_utils/field_norm.py` | **Unreachable.** `FieldNorm` also exists in `equi_rl_utils.py`. | **no** |
| `common_utils/rotation_transformer.py` | **Unreachable.** | **no** |
| `common_utils/rotation_utils.py` | **Unreachable** (only imported by `rotation_transformer.py`). | **no** |

Four unreachable files, all tracked, deferred to TODO P3.5. Check `field_norm.py` against
`equi_rl_utils.py`'s `FieldNorm` before removing it — if the standalone copy is the better one, keep
that instead.

A fifth, `rl/rl_utils.py`, was deleted on 2026-09-16: untracked, imported by nothing, and
unimportable (`from __future__ import annotations` on line 15, a hard `SyntaxError`), so it had
never loaded in any run. Its one original class, `TrivialLayerNorm`, is preserved verbatim in
TODO P3.4.

### `resfit/rl_finetuning/off_policy/` — the original ResFiT agent

| File | Equivariant counterpart | Relationship |
|---|---|---|
| `rl/q_agent.py` | `equi_off_policy/rl/q_agent.py` | Diverged. Same update structure, different feature types and clipping. |
| `rl/actor.py` | `equi_off_policy/rl/actor.py` | Diverged. Original returns a distribution object; equivariant returns a tensor. |
| `rl/critic.py` | `equi_off_policy/rl/critic.py` | Diverged. Original supports C51/HL-Gauss; equivariant does not (dead `loss_cfg`). |
| `networks/encoder.py` | `networks/obs_encoder.py` + `networks/equi_encoder.py` | Restructured. |
| `networks/min_vit.py` | *(none)* | Original only. |
| `common_utils/*` | `equi_off_policy/common_utils/*` | Partly copied, partly extended. `off_policy/common_utils` is still imported by the equivariant `q_agent.py`. |

### Shared, path-independent

| File | Role |
|---|---|
| `wrappers/residual_env_wrapper.py` | `BasePolicyVecEnvWrapper`. Runs the base policy, adds `observation.base_action` to the observation, sums base + residual before stepping. |
| `wrappers/normalized_env_wrapper.py` | `NormalizedVecEnvWrapper`, vanilla RL's: the residual wrapper less the base policy. Standardizes the state, unscales the action, records it in `info["scaled_action"]`. |
| `utils/normalization.py` | `StateStandardizer`, `ActionScaler`, and the identity versions used on the equivariant path. |
| `utils/hugging_face.py` | Downloads BC checkpoints from wandb and buffer caches from HF. |
| `utils/checkpoint.py`, `utils/evaluate_dexmg.py`, `utils/rb_transforms.py`, `utils/dtype.py` | Checkpointing, evaluation rollouts, replay-buffer transforms, dtype helpers. From 2026-10-12 each eval also logs the gripper's tilt (`eval/tilt_max_deg_mean`, `eval/tilt_over_30deg_frac`: per episode, the largest angle between arm 0's gripper axis and its axis at the start), and scores Q at the base action plus the residual on both stacks (before, the equivariant residual's Q was scored at the residual alone; EXPERIMENTS.md, 2026-10-12 CORRECTION). |
| `utils/field_metrics.py` | From 2026-10-12: the residual and the critic's per-sample slope per action field, on the logged update's online and demo rows (`debug/residual_field/*`, `debug/dQ_da_field/*`), both stacks. |
| `dexmg/environments/dexmg.py` | `create_vectorized_env` — robosuite/mimicgen env construction. |

## How a training step flows

```
BasePolicyVecEnvWrapper.step(residual_naction)
  ├─ combined = last_base_naction + residual_naction
  ├─ env.step(combined)
  ├─ base_policy.select_action(next_raw_obs)      # frozen ACT or Diffusion
  └─ obs["observation.base_action"] = base_naction

ResObsEnc.forward(obs)                             # equi_off_policy/networks/obs_encoder.py
  ├─ images     uint8 (buffer) ÷ 255, float (env) as is → ×2 − 1 → [−1, 1]
  ├─ agentview  84×84 → random crop 76×76 → EquivariantResEncoder76Cyclic → regular repr
  ├─ in-hand    84×84 → random crop 76×76 → ResEncoder76InHand            → trivial repr
  ├─ state → ee_pos, ee_quat, ee_q
  │    ├─ pos_xy  = normalize(ee_pos[:2] - robot_base_xy)   → irrep(1)
  │    ├─ pos_z   = normalize(ee_pos[2])                    → trivial
  │    ├─ ee_rot  = quat → 6D rotation, split into 3 xy column pairs → 3 × irrep(1)
  │    └─ ee_q    = normalize(gripper state)                → 2 × trivial
  ├─ base_action → xy (irrep 1), z (trivial), rx_ry (irrep 1), rz (trivial), gripper (trivial)
  └─ returns (actor_features, critic_features) as GeometricTensors
        critic_features = [vis | in-hand | prop]
        actor_features  = [vis | in-hand | prop | base_action]

Actor.forward(actor_feat, std)
  ├─ slice into (vis+ih, prop, base_action)
  ├─ visual_proj → cat → input_proj → cat → policy
  ├─ mu * action_scale
  └─ if std given: + noise, then equi_clip(·, bound=1.0)

Critic.forward(critic_feat, act)
  ├─ slice into (vis+ih, prop); wrap act as GeometricTensor
  ├─ visual_proj → cat(v, prop, act) → input_proj → cat(z, prop, act)
  └─ Q-ensemble heads → GroupPooling → scalar Q   [num_q, B, 1]

QAgent.combine_actions(base, residual)
  └─ equi_clip(base + residual, layout, bound=1.0)
```

Note the skip connections: proprioception and action are re-concatenated at every stage rather
than only entering at the input. Both actor and critic do this.

In a vanilla run (`base_policy=None`) `NormalizedVecEnvWrapper` steps the actor's action itself,
the encoder has no base-action block, so `actor_features` is `critic_features`, and the actor's
skip connections carry proprioception only.

### Metrics only on the update that is logged

The trainer logs every `log_freq` env steps, and reads the metrics of the last update before it:
one update in 400 at the defaults. So `QAgent.update(..., compute_metrics=False)` skips every value
that is only logged, and the trainer passes `True` on that one update. The update itself is the same
bit for bit either way (`tests/test_regression_agent.py`). `_td_errors`, which prioritized replay
reads, is always returned, and stays on the GPU. The ResFiT agent takes no such argument, and
computes its metrics on every update, as upstream does.

**`timing/*` and `Time%`.** Without metrics nothing in the update reads a GPU value, so the CPU
queues all four updates and moves on. The GPU wait would then be timed under `env_step`, whose
`act()` is the next thing to read a GPU value. One `torch.cuda.synchronize()` at the end of the
last update's `gradient_update` timer puts it back, so `timing/gradient_update_*` still measures
the updates. It costs one sync per env step, which `act()` would have made anyway.

### Why the update used to stall

The CPU queues GPU work and moves on, so while the GPU runs one layer the CPU can already queue the
next. A *host sync* breaks that: anything that needs a GPU value on the CPU makes the CPU wait until
the GPU has finished everything queued so far. Before 2026-09-30 one env step's learning (one
`act()` and four updates, the last with the actor) made 487 of them, counted with
`torch.cuda.set_sync_debug_mode` at `target-fix`'s config on `1e33b8f`:

| Where | Syncs | Fix |
|---|---|---|
| escnn basis expansions' `__hash__`/`__eq__`, which compare sampled bases with `.cpu()` and `allclose`. torch hashes modules on every walk: `parameters()`, `train()`, `modules()` | 177 | `escnn_fixes`: hash by identity |
| random crop: offsets and index grids built on the CPU and copied from pageable memory, and range asserts on the GPU | 128 | offsets drawn and checked on the CPU, copied from pinned memory |
| critic `GroupPooling`: GPU scalars as slice bounds | 80 | `escnn_fixes.GroupPooling` |
| `.item()` and `.cpu()` on metrics | 65 | computed only on the logged update (above) |
| actor `IIDBatchNorm1d`: escnn asserts on its own running stats | 30 | not fixed: `actor_norm` defaults to `none` from 2026-09-30 |
| `get6DRotation`: indexing with a Python list | 9 | slicing |

The hashing was also slow in itself: 24 ms per `enc.parameters()`, 12 ms per critic soft update,
and 21 ms for the train/eval switch around each `act()`.

After the fixes the loop is GPU-bound, and the rest of the gap to ResFiT is compute: the
equivariant encoder does about 10× the FLOPs of ResFiT's per sample. `tests/test_sync_free.py`
keeps every config in `tools/agent_fingerprint.py`'s `CANDIDATES` sync-free (STANDARDS 6.5), and
`tools/bench_update.py` times an env step's learning.

## Normalization lives in two different places

This trips people up, so it is worth stating plainly.

**Baseline path.** `StateStandardizer` and `ActionScaler` are built from dataset statistics in
`train_residual_td3.py` and applied *outside* the network, in the env wrapper and buffer.

**ResFiT's action bound** (`agent.actor.action_bound`, from 2026-10-12). `"box"`, the default, is
ResFiT's own, and every ResFiT run before it ran this:
- the actor's final Tanh × `action_scale`, per axis, in the scaler's units;
- `QAgent.combine_actions` clamps base + residual per axis to [-1, 1]. The actor loss, the TD target,
  `critic_diagnostics` and the evaluator's Q all combine through it.
- `ActionScaler.unscale` clamps the executed action to the demos' range widened by `action_scale`.

`"equivariant"` (`resfit-residual-square-limits`) gives this agent the equivariant stack's three
limits in env units, reached through the scaler's per-axis middle and half-range:
- `action_scale · equi_tanh` in place of the Tanh;
- `equi_clip` in `combine_actions`;
- `ActionScaler(clamp_unscaled=False)`, so the env clips to its box, as under `IdentityScaler`.

The layout is the equivariant encoder's (`arm_action_shape`, `build_action_layout` in
`obs_encoder.py`). It is for residual runs with one arm and a one-dimensional gripper; the
equivariant agent refuses it. `tests/test_action_bound.py` pins `"box"` to the agent before the
field, and tests the bound.

**Equivariant path.** Those two become `IdentityStandardizer` and `IdentityScaler`
([train_residual_td3.py:305-307](../resfit/rl_finetuning/scripts/train_residual_td3.py#L305)), and
normalization moves *inside* the encoder via `build_equivariant_normalizer` and
`ResObsEnc.set_normalizer`. It has to move, because normalization must be applied per
representation to preserve equivariance — see [EQUIVARIANCE.md](EQUIVARIANCE.md).

So "where is the data normalized" has two different answers depending on which path you are on,
and comparing statistics across the two paths requires knowing which.

**Images arrive in two formats on both paths.** act() gets the env's float images in [0, 1]; the
update gets the replay buffer's uint8 copies. ResFiT's `_encode` and `ResObsEnc` both convert by
dtype. Until 2026-10-02 `ResObsEnc` divided both by 255, so every equivariant run acted on
near-blank images (EXPERIMENTS.md, 2026-10-02 CORRECTION). `tests/test_obs_formats.py` feeds both.

## Training phases in `train_residual_td3.py`

Roughly in order:

1. Download and freeze the base BC policy from wandb (`base_policy.wandb_id`).
2. Load the LeRobot dataset; build normalizers (path-dependent, as above).
3. Build train and eval vectorized envs, each wrapped in `BasePolicyVecEnvWrapper`.
4. Build the agent: `QAgent` from either `off_policy` or `equi_off_policy`.
5. For equivariant runs only: probe the env for the robot base XY (cached in `env_probes/<task>.json`), build the equivariant normalizer, attach it to the encoder.
6. Create offline and online replay buffers; populate the offline buffer from the dataset (cached in `offline_buffer_cache/`).
7. Environment warmup: fill the online buffer to `algo.learning_starts` with base policy + noise.
8. Critic warmup: `algo.critic_warmup_steps` critic-only updates, actor frozen.
9. Main loop to `algo.total_timesteps`: step env, add to buffer, do `num_updates_per_iteration` gradient updates, evaluate every `eval_interval_every_steps`. On a new best success rate, save the residual weights and upload them with that eval's video (see "Run packages" below).

A vanilla run skips step 1, wraps its envs in `NormalizedVecEnvWrapper` (3), stores no base action
(6), warms up with uniform random actions (7), and runs no critic warm-up (8), since
`TunedRLPDConfig` sets none. It builds its buffers in memory: `write_buffer_caches=false`.

**The buffer caches** (steps 6 and 7) are addressed by a hash of what filled them, printed at
startup and mirrored by `preflight.cache_hashes`. Two top-level fields decide whether a run uses
them:
- `write_buffer_caches` (default true): false keeps both buffers in memory. Existing caches are
  still read.
- `read_buffer_caches` (default true; from 2026-10-07): false fills both buffers in the run and
  loads no cache, from disk or the Hub. The trainer refuses it with `write_buffer_caches=true`.

The reason a run may not read a cache: a key does not record everything that filled it. It
records torchrl's and tensordict's versions, and, since 2026-10-07, mujoco's in the online key,
whose frames mujoco renders. It does not record a residual buffer's normalization path (TODO B7,
S9). Square's 6.4 GB offline cache takes about 25 min to rebuild (BC inference on 80,731 frames,
measured 2026-10-07).

Each run also logs `versions` in its wandb config and run manifest, the packages its commit does
not pin: mujoco, robosuite, torch, torchrl, tensordict, escnn (`software_versions()` in
`load_policy.py`). It is not a config field: nothing reads it.

`eval_first: bool = True` means there is an evaluation at step 0. At the config's default,
`agent.actor.actor_last_layer_init_scale = 0.0`, the residual starts at exactly zero, so **the
step-0 evaluation measures the base BC policy alone.** That number is the baseline every run must
be compared against, and it is why an experiment record that only reports "best success rate" can
be misleading. The equivariant residual runs of `square-rerendered-1009` (`6ng62tpm`) and of the
fixed-env baselines (`kyv7clfc`) overrode the scale to 1e-4. Their step 0 is the base policy plus a
residual of about 1e-4 (`residual_l1`), within noise of it, but not exactly it.

## Where to find things

| Thing | Location |
|---|---|
| BC checkpoints (cached) | `artifacts/run_<wandb_id>_best:v<n>/policy/model.safetensors` |
| BC checkpoints (source) | wandb artifact, id in `base_policy.wandb_id` |
| Offline datasets | HuggingFace, name in `offline_data.name`. Lars Ankile's: https://huggingface.co/ankile |
| Offline buffer cache | `offline_buffer_cache/<hash>/` (12 GB) |
| Online buffer cache | `online_buffer_cache/` (32 GB) |
| Robot base XY probes | `env_probes/<task>.json` |
| Local wandb run data | `wandb/run-<date>-<run_id>/files/{config.yaml,output.log,wandb-metadata.json}` |
| Run packages | `<CACHE_DIR>/outputs/runs/<project>/<run_id>/`, or `.../local/<timestamp>/` without wandb. See below. |
| Re-evaluating a saved residual with fields zeroed | `tools/residual_field_ablation.py <run_id> --step N`, from a run's `models/step_<N>.pt`; it builds the agent with the trainer's own `build_agent`. Results in `<CACHE_DIR>/outputs/diagnostics/residual-field-ablation/` |
| Past local run outputs | `local_runs/run_<timestamp>_.../`; before run packages, `run_<timestamp>_*` and `bc_run_*` in the launch directory, deleted when the run completed |
| Run launchers | `resfit/rl_finetuning/shell/paper_runs/<task>/` and `.../ablations/<task>/<knob>/` |
| Vendored dependencies | `deps/{lerobot,robosuite,mimicgen,dexmimicgen}/` (gitignored, 1.7 GB) |

### Reading a past run without wandb

Everything needed is on disk under `wandb/run-*-<run_id>/files/`:

- `wandb-metadata.json` — the exact command line, and the git commit it ran at.
- `config.yaml` — the fully resolved config. Remember the provenance table above: some of these
  values did not affect the network.
- `output.log` — per-step training lines and the evaluation markers, one `✓`/`✗` per episode.
- `requirements.txt` — the Python environment at run time.

### Run packages

Both trainers write everything a run saves into one folder and keep it after the run:
`<CACHE_DIR>/outputs/runs/<project>/<wandb_run_id>/`, or `local/<timestamp>/` when wandb is off.
`CACHE_DIR` defaults to the launch directory.

**Every record lands in the main checkout** (from 2026-10-07). A run launched from a worktree
pinned to its commit, as `tools/gpu_queue.sh` launches, sets `CACHE_DIR` to the main checkout, and
writes all it records there: its package (`RUNS_ROOT`), wandb's local record (`wandb.init(dir=...)`
in all three trainers, so `<CACHE_DIR>/wandb/`), the BC checkpoints it downloads
(`<CACHE_DIR>/artifacts/`), and Hydra's folder (`submit.sh` passes `hydra.run.dir`, under
`<CACHE_DIR>/outputs/<date>/<time>/`). The worktree holds only the code, so it can be removed once
its runs end. Before then, each worktree kept its own `wandb/` and `outputs/`; their 34 records
were moved into main on 2026-10-07 (EXPERIMENTS.md, CODE). The helpers are at the bottom of
`resfit/lerobot/utils/load_policy.py`.

| Path | Contents |
|---|---|
| `manifest.json` | wandb ids and URL, host, argv, git commit **plus the list of uncommitted files**, one entry per resume, the best step and success rate, `completed` |
| `wandb_best/` | An exact copy of the run's one wandb upload: `best.json`, `best_eval.mp4`, and `policy/` (BC) or `best_model.pt` (residual) |
| `wandb_logs/` | wandb's `config.yaml`, `output.log`, summary, metadata and `requirements.txt`, the `run-<id>.wandb` record, and `metrics.jsonl` |
| everything else | What the trainer already saved locally: BC `checkpoints/`, `latest/`, `best/` and eval videos; residual `models/` (`best_model.pt`, and `step_<N>.pt` every `checkpoint_interval` steps from 2026-10-12) and `outputs/` |

**The upload rule.** A run uploads one artifact, `run_<id>_best`: the best model plus the video of
the eval that produced it. Each new best uploads a new version, waits for it to commit, then deletes
the older versions, so wandb holds exactly one and a failed upload never leaves the run without a
best. No per-eval videos, per-step models or `latest` uploads. At the end of a completed run the best
video is posted once to the existing panel (`eval/rollout_video` for BC, `eval/video` for residual),
captioned with its step and success rate. Every other panel is unchanged.

**Read `metrics.jsonl`, not the wandb API.** `Run.scan_history()` drops the final history row, which
is where the last eval and the best-video entry land. `metrics.jsonl` is parsed from the local
`.wandb` record, which has every row; `wandb sync` on that record restores a deleted run. A crashed
or killed run keeps its package with `completed: false` and no `wandb_logs/`; the same files are
still under `wandb/`.

**`best_model.pt` is for re-evaluation, not resuming.** It holds the agent's weights, the resolved
config, the step and the success rate, and no optimizer state. `load_residual_model` loads it into an
agent built from that config, and the equivariant normalizer must be attached with `set_normalizer`
first. Save and load both run in eval mode, because escnn registers its cached basis buffers only
there. A BC run resumes from its package with `--resume_ckpt <package>/latest`.

**`step_<N>.pt`, from 2026-10-12, is the same for a fixed step.** With `checkpoint_interval` above 0,
the trainer saves one after each multiple's eval, so it is the policy that eval measured, with
`success_rate` left `None`. It is what `tools/residual_field_ablation.py` re-evaluates. Each is about
87 MB on Square, so a run at 2,000 to 60k keeps 31 (2.7 GB).

### Observation and action layout per task

`ResObsEnc` reads `observation.state` and `observation.base_action` as `n_arms` contiguous blocks.
Per arm the state is `[eef_pos 3, eef_quat 4, gripper_qpos G]` and the action is
`[delta_pos 3, delta_rot 3, hand H]`. Every value below was measured from the environment, not read
off a config.

| Task | arms | G | H | `observation.state` | action | prop (encoded) | cameras |
|---|---|---|---|---|---|---|---|
| Can, Square | 1 | 2 | 1 | 9 | 7 | 11 | agentview + `robot0_eye_in_hand` |
| TwoArmBoxCleanup | 2 | 12 | 6 | 38 | 24 | 42 | agentview + `robot0/1_eye_in_hand` |

`prop` is wider than `observation.state` because the encoder expands each `eef_quat` (4) into a 6D
rotation (6). `ResObsEnc.state_dim` is the raw width and `prop_dim` the encoded one; `QAgent`
asserts the raw width against the env, so mixing the two up fails at construction.

Set by `equivariance.n_arms`, `gripper_dim` and `hand_dof`. Wrong values do not crash — they
mis-slice state and action into the wrong representations — so `test_config.py` pins them per task.

Three further BoxCleanup facts, all measured: `gripper_qvel` exists in the raw observation but is
**not** part of `observation.state`; `robot0` is `PandaDexRH` and carries the right hand; the
horizon is 300. The robot bases are at `(-0.56, -0.25)` and `(-0.56, +0.25)`, both with identity
orientation, giving a rotation center of `(-0.56, 0.0)`.

### Task, dataset, and base policy

| Task | Dataset | BC run | Base policy | Cached at |
|---|---|---|---|---|
| Can | `ankile/robomimic-mh-can-image` | `robomimic-can-bc/xhjdl8a7` | **Diffusion** (n_obs 2, chunk 16) | `artifacts/run_xhjdl8a7_best:v7/` |
| Square | `ankile/robomimic-mh-square-image` | `robomimic-square-bc/3hzs5bz1` | **Diffusion** (n_obs 2, chunk 16) | `artifacts/run_3hzs5bz1_best:v9/` |
| TwoArmCoffee | `ankile/dexmg-two-arm-coffee` | `dexmg-bc/tjsr8xad` | ACT (n_obs 1, chunk 20) | `artifacts/run_tjsr8xad_best:v2/` |
| TwoArmBoxCleanup | `ankile/dexmg-two-arm-box-cleanup` | *(`TODO` in config)* | — | — |
| TwoArmCanSortRandom | `ankile/dexmg-two-arm-can-sort-random` | *(`TODO` in config)* | — | — |

**The base policy for Can and Square is a Diffusion policy, not ACT.** This matters: the README's
example command and much of the upstream ResFiT paper work use ACT, so the equivariant experiments
are not running the same base-policy type as the reference material. It is deliberate — the stated
research goal is comparing equivariant versus non-equivariant residual finetuning *of a diffusion
policy*. Verified by reading `policy/config.json` in each cached artifact.

Two more artifacts are cached whose configs no longer reference them: `run_6qgo2g5k_best:v1` (an
earlier TwoArmCoffee BC policy, now commented out in the config) and `run_e14vlvap_best:v3`.

To point a residual run at a different BC policy: find the run at wandb.ai, take the 8-character
run id from the URL, and set `base_policy.wandb_id = "<project>/<run_id>"` in the task config.
`base_policy.wt_type="best"` works for every BC run. `"latest"` or a step number resolves only for BC
runs from before run packages, the last ones to upload those artifacts.

## Known rough edges

Recorded so they are not rediscovered. None are blocking.

- `train_residual_td3.py` has several commented-out `assert False` debugging blocks around dataset
  stats printing.
- `load_policy.py` infers the policy type by checking for the substring `diffusion` in the config's
  `type` field, and otherwise by the *presence of a `use_vae` key* to detect ACT. The second test is
  incidental rather than meaningful. Carries its own `TODO` upstream.
- `QAgent._act_default_equi` calls `actor.forward(feat, std=0)` for evaluation. In
  `Actor.forward`, `std=None` returns the unclipped mean while `std=0` returns a clipped mean.
  Two different code paths for what reads as the same intent. **And `std=0` draws a random
  number** (`torch.randn_like(mean) * 0`; found 2026-10-07): the action is exact, but torch's RNG
  moves. Every eval's `act()` and every actor loss take that path, so an equivariant run's random
  stream depends on its eval schedule. Left as it is: changing it changes every equivariant run's
  updates. `critic_diagnostics` avoids it (`std=None`, then the same clip).
- `networks/ablate_equi_obs_encoder.py` is imported by nothing — the scalar ablation goes through
  `ResObsEnc(equivariant=False)`. It still looks up the pre-bimanual normalizer key
  `action_gripper`, which no longer exists, so wiring it up would raise. It also has the image bug
  (EXPERIMENTS.md, 2026-10-02 CORRECTION). Left in place; delete it in the Phase 3 cleanup.
- `Critic.forward` accepts a `return_logits` keyword and ignores it.
- **`action_scale` does not bound the residual** unless `equivariance.residual_squash=True`
  (added 2026-09-29, off by default, as every earlier run had it). `Actor.forward` computes
  `scaled_mu = mu * action_scale`, and without the squash `|mu|` is not bounded by 1, so the mean
  can exceed `action_scale`. Measured at 1.27x on the two-arm actor.
  The executed action is bounded instead by `equi_clip(..., bound=1.0)`, which is ten times
  looser than `action_scale=0.1` suggests. Read `residual_l1` / `residual_l2` against 1.0, not
  against `action_scale`. Both are the update batch's residual, not the executed one. Surfaced 2026-09-25 by parameterising the equivariance tests over arm
  count; the single-arm actor happened to stay under the bound, so the old test passed by luck.
- **The base policy is reset only on success, and for diffusion it resets every env.**
  `BasePolicyVecEnvWrapper.step` resets on `terminated` but not `truncated`, so after a timeout the
  auto-reset env keeps executing the old episode's action chunk. The reset also comes after
  `select_action`, and diffusion's `reset()` clears the queues of all envs, not just the finished one.
  Fixing either changes Can and Square behaviour, so it breaks comparability with the runs that are
  still comparable: the ResFiT runs, the base rates and the image-fix pair.
- **The ACT branch of that reset crashes.** It passes `env_ids=terminated_envs`, a name that does not
  exist, so the first success with an ACT base policy raises `NameError`. Can and Square use
  diffusion and never reach it; BoxCleanup does. TODO B3.
- Two git stashes exist: `stash@{0}` on `aabde0f`, `stash@{1}` on `d9909ce`. Contents unreviewed.
