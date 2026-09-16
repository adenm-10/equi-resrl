# The equivariance design

Why this project has a symmetry in it, what exactly is assumed to be symmetric, and which of
those assumptions actually hold. For where the code lives, see
[ARCHITECTURE.md](ARCHITECTURE.md).

## The idea in one paragraph

A tabletop manipulation task has a rotational symmetry: if you rotate the whole scene about the
vertical axis, the correct behavior rotates with it. Reaching for a can at 3 o'clock is the same
problem as reaching for a can at 9 o'clock. A generic neural network has to learn that twice. A
network built to respect the symmetry gets it for free, which should mean better sample efficiency
— the central claim of the SO(2) equivariant RL paper. This project tests whether that benefit
survives when the equivariant network is only learning a *residual* correction on top of a
non-equivariant base policy.

## Group

We use **C8**, the cyclic group of 8 rotations about the vertical (z) axis, as a discrete stand-in
for the continuous group SO(2).

Set by `EquivarianceConfig.N = 8`, adopted from the SO(2) equivariant RL paper's parameters in
commit `a948786`. Built at [q_agent.py:97](../resfit/rl_finetuning/equi_off_policy/rl/q_agent.py#L97):

```python
self.group = gspaces.no_base_space(CyclicGroup(N))
```

`no_base_space` means the group acts only on feature channels, not on spatial dimensions. The
network layers are ordinary linear layers with constrained weights, not steerable convolutions.
The one exception is the vision backbone, which does act on image space.

**Consequence:** equivariance is exact only for the 8 multiples of 45°. Rotations in between are
approximated. Raising `N` tightens the approximation and costs compute and memory linearly.

We use escnn throughout. Three representation types matter:

| Representation | Size | Behaves like |
|---|---|---|
| `trivial_repr` | 1 | A number that does not change when the world rotates (a height, a gripper width) |
| `irrep(1)` | 2 | A vector in the horizontal plane that rotates with the world (an xy position, an xy velocity) |
| `regular_repr` | 8 | One value per group element; the general-purpose hidden feature type |

## What is declared symmetric

All in [obs_encoder.py:96-120](../resfit/rl_finetuning/equi_off_policy/networks/obs_encoder.py#L96).

### Proprioception — 11 dimensions

| Slice | Content | Representation | Reasoning |
|---|---|---|---|
| 0:2 | End-effector xy, minus robot base xy | `irrep(1)` | A horizontal position rotates with the world |
| — | End-effector rotation, as 3 xy column pairs | 3 × `irrep(1)` | See below |
| — | End-effector z | `trivial` | Height is unchanged by a z-rotation |
| — | Gripper state, 2 dims | 2 × `trivial` | Gripper opening does not rotate |

Total: 4 × `irrep(1)` + 3 × `trivial` = 11.

**The rotation encoding is the subtle part.** The quaternion is converted to a 6D rotation
representation (the first two rows of the rotation matrix) by `quaternion_to_rotation_6d`. Under a
world z-rotation `g`, the matrix becomes `R_g · R`. Because `R_g` only mixes the x and y axes, the
`(x, y)` part of each *column* of `R` rotates as a plane vector. So the 6 numbers are regrouped
into three xy pairs, one per column, and each pair is declared `irrep(1)`:

```
6D vector layout:  [R00 R01 R02 R10 R11 R12]
regrouped as:      (R00,R10) (R01,R11) (R02,R12)
                    column 0   column 1  column 2
```

This regrouping is what the interleaved slicing at
[obs_encoder.py:275-280](../resfit/rl_finetuning/equi_off_policy/networks/obs_encoder.py#L275)
is doing. It looks like index gymnastics; it is the representation theory.

### Action — 7 dimensions

Delta end-effector pose in an OSC_POSE controller, plus gripper.

| Slice | Content | Representation |
|---|---|---|
| 0:2 | `action_xy` | `irrep(1)` |
| 2:3 | `action_z` | `trivial` |
| 3:5 | `action_rx_ry` | `irrep(1)` |
| 5:6 | `action_rz` | `trivial` |
| 6:7 | `action_gripper` | `trivial` |

The base policy's action enters the network as an observation (`observation.base_action`, added by
`BasePolicyVecEnvWrapper`), so it carries the same representation as the action.

### Vision

| Camera | Representation | Reasoning |
|---|---|---|
| `agentview` | `regular_repr` via `EquivariantResEncoder76Cyclic` | Treated as a view that rotates with the world |
| `robot0_eye_in_hand` | `trivial_repr` via the scalar `ResEncoder76InHand` | The in-hand camera frame rotates *with the gripper*, so the image content does not change when the world rotates. Encoding it as invariant is the correct choice. |

The in-hand decision is right and is documented in the `ResObsEnc` docstring. The agentview
decision gives approximate rather than exact equivariance, because the camera is tilted — a
reviewed and accepted trade-off, explained in assumption 1 below.

### The Q function is invariant, not equivariant

This distinction is the crux of the whole design.

- **The actor is equivariant.** Rotate the scene, and the action it produces rotates with it:
  `π(g·s) = g·π(s)`.
- **The critic is invariant.** Rotate both the scene and the action, and the value is unchanged:
  `Q(g·s, g·a) = Q(s, a)`. A value is a number; it has no direction.

The critic achieves invariance with a `GroupPooling` layer
([critic.py:35-37](../resfit/rl_finetuning/equi_off_policy/rl/critic.py#L35)):

```python
gpool = nn.GroupPooling(hidden_type)          # max over the 8 group channels of each field
layers.append(gpool)
layers.append(nn.Linear(gpool.out_type, out_type, initialize=escnn_init))
```

Features up to this point live in the regular representation (8 numbers per field, one per group
element). `GroupPooling` takes the max over those 8, which produces a quantity that does not
change when the group permutes them. After pooling, the feature type is all-trivial, so the final
`Linear` maps invariants to an invariant scalar.

`GroupPooling` is load-bearing and must be preserved when refactoring the critic. But it is **not**
the change that fixed the collapse — see below.

### What actually changed between the collapsed runs and the successful one

Verified 2026-09-16 by diffing `7dae4925` against `caf83f3`. `GroupPooling` was **already present**
at `7dae4925`, and all three surviving runs at that commit collapsed to zero success. The critic
head there was:

```
Linear → FieldNorm → ReLU → [Linear → FieldNorm → ReLU] × (num_layers-1) → GroupPooling → Linear
```

At `caf83f3` it became simply `Linear → GroupPooling → Linear`, with every `FieldNorm` and `ReLU`
commented out. So the difference is the **removal of `FieldNorm` (and the ReLUs) from the critic
head**, not the addition of group pooling.

There is a plausible mechanism. `FieldNorm` was applied to `hidden_type`, which is
`regular_repr` — and per escnn's definition, `FieldNorm` computes each field's mean as *the
projection onto the trivial-representation subspace* and subtracts it. The trivial subspace of a
regular representation is exactly its group-invariant component. So `FieldNorm` was stripping the
invariant part of every field, and the `GroupPooling` immediately after was then trying to extract
an invariant from features whose invariant content had just been normalized away.

That is consistent with the diagnostics: `dQ_da_mean_abs ≈ 2e-4` (a critic with almost no usable
action dependence) alongside a low, stable `critic_loss` (it fit *something* well). It is a
hypothesis, not a proven cause — but it is a specific and testable one.

**It is also not the only change.** `caf83f3` doubled the encoder depth in the same commit (4 -> 8
residual blocks; the second block in each stage had been commented out) and raised
`enc_degree_channel` from 16 to 32. A deeper, wider encoder giving better features is an equally
plausible reason a previously-signalless critic started working. So a successful reproduction will
confirm *that* the commit works without identifying *which* of the three changes did it. See
EXPERIMENTS.md, "three candidates, not one".

**Consequence for TODO P3.4:** restoring normalization to the critic head is higher-risk than it
looks. Whatever goes back in must not touch the invariant subspace of a regular representation.
`TrivialLayerNorm` operating *after* pooling is safe in exactly this respect; `FieldNorm` *before*
pooling is the thing that appears to have broken it.

### Clipping has to respect the representation

Clamping each action dimension independently to `[-1, 1]` would break equivariance: a per-axis box
is not rotationally symmetric, so clipping before and after a rotation give different answers.
`equi_clip` (in `equi_rl_utils.py`) instead clips each `irrep(1)` pair by its *norm* — a circle
rather than a square — and clips `trivial` components as ordinary scalars. It is used in
`Actor.forward` and in `QAgent._combine_actions`, in both cases with `bound=1.0`.

The non-equivariant path uses plain `torch.clamp(-1, 1)`. This is a real, intended difference
between the two paths, not an inconsistency.

## Normalization must be done per representation

An ordinary per-dimension normalizer (subtract a mean, divide by a per-dimension std) breaks
equivariance on any `irrep(1)` field. Scaling x and y differently is a shear; shears do not commute
with rotations. Adding a nonzero offset moves the rotation center.

So for every `irrep(1)` field the normalizer must be **isotropic** (one shared scale for both
components) and **offset-free**. This is handled by `get_range_symmetric_normalizer_from_stat` in
[equi_normalizer.py:307](../resfit/rl_finetuning/equi_off_policy/networks/equi_normalizer.py#L307):
it takes a single `abs_max` over both components and sets `input_min = -abs_max`,
`input_max = +abs_max`, which makes the offset exactly zero.

Per-field assignment, from `build_equivariant_normalizer`:

| Field | Normalizer | Why |
|---|---|---|
| `pos_xy` | symmetric, shifted by robot base xy | `irrep(1)`, needs isotropic scale about the rotation center |
| `pos_z` | plain range | `trivial`, free to scale per-dimension |
| `ee_rot` | identity | already unit-norm from the rotation matrix |
| `ee_q` | plain range | `trivial` |
| `action_xy` | symmetric, **no** shift | `irrep(1)`; it is a delta, so it is already centered on zero |
| `action_z` | plain range | `trivial` |
| `action_rx_ry` | identity | `irrep(1)`; identity is trivially isotropic |
| `action_rz` | identity | `trivial` |
| `action_gripper` | plain range | `trivial` |

This is all correct as written. It is also why the equivariant path sets the outer
`StateStandardizer` and `ActionScaler` to identity and normalizes inside the encoder instead.

## The rotation center

Rotations are about the robot's base, not the world origin. If you normalize `pos_xy` about the
world origin but rotate about the base, the symmetry does not hold.

Two matching pieces:

1. `detect_robot_base_xy` ([equi_normalizer.py:460](../resfit/rl_finetuning/equi_off_policy/networks/equi_normalizer.py#L460))
   spins up one env instance, walks the MuJoCo body list for a name matching `*_base` containing
   `robot`, and reads its xy position. Cached in `env_probes/<task>.json` so this only happens
   once. For Can: `[-0.5, -0.1]`.
2. `ResObsEnc.forward` subtracts that value from `ee_pos[:, 0:2]` before normalizing, and
   `build_equivariant_normalizer` shifts the `pos_xy` statistics by the same amount so the scale is
   computed about the right center.

Both must agree. If `set_normalizer` is called without `robot_base_xy`, it prints a warning and
leaves the center at the world origin — which silently breaks the symmetry rather than raising.

`detect_robot_base_xy` also warns if the base orientation is not the identity quaternion, because
the subtraction assumes the base frame is aligned with the world frame.

## Assumptions

Documented so they can be tested rather than argued about.

### 1. The agentview camera is not top-down — accepted approximation

**Status: reviewed and accepted by Aden, 2026-09-16.** The judgment is that a tilted camera still
gives approximate equivariance, which is good enough for the symmetry prior to be useful. Recorded
here with the reasoning so the decision is visible rather than implicit, not as an open problem.

The equivariant vision encoder treats the agentview image as something that rotates in-plane when
the world rotates about the vertical axis. That correspondence is exact only for a top-down camera,
and agentview is not top-down.

From `deps/robosuite/robosuite/models/assets/arenas/table_arena.xml:47`:

```xml
<camera mode="fixed" name="agentview" pos="0.5 0 1.35" quat="0.653 0.271 0.271 0.653"/>
```

Working that quaternion out, the camera's viewing direction is about `(-0.71, 0, -0.71)` — a 45°
downward tilt looking back across the table, from a fixed position. Rotating the scene about the
robot base's z-axis does *not* produce an in-plane rotation of that image. It produces a
perspective change that no 2D image rotation reproduces.

The SO(2) equivariant RL results this project builds on use top-down observations, where the
correspondence is exact.

So the proprioception and action branches are **exactly** equivariant, and the agentview branch is
**approximately** equivariant — a scene rotation produces an image change that is close to, but not
identical to, an in-plane image rotation. The perspective error grows with distance from the
rotation center and with the tilt angle.

The accepted position is that approximate equivariance on the vision branch is still a useful
prior: the network is not being handed a false invariance so much as a slightly noisy one, and the
exactly-equivariant proprioception and action branches carry the parts of the state where the
symmetry is precise.

If equivariant results ever need explaining and the exact branches have been ruled out, the cheap
diagnostic is still available: render a scene before and after a 45° rotation about the base and
measure the distance between the second render and a 45° in-plane rotation of the first. That
quantifies the approximation rather than arguing about it. Not currently scheduled.

### 2. Random cropping breaks exact equivariance

`CropRandomizer` takes a random 76×76 crop of the 84×84 image. A random translation does not
commute with rotation, so it injects a small equivariance error on every forward pass. This is
standard practice in the papers and probably fine, but the equivariance tests must run with
cropping disabled or they will fail for the wrong reason.

### 3. The combined action is not equivariant, by construction

The base BC policy is an ordinary ACT or Diffusion policy with no symmetry constraint. The final
action is `base_action + residual`. Even with a perfectly equivariant residual, the sum is not
equivariant, because the first term is not.

This is inherent to residual RL, not a bug, and it is the most scientifically interesting thing
about the project — it is the question the project exists to answer. But it does mean the symmetry
prior constrains only the correction term, which bounds how much it can possibly help. Any
equivariance test of the full agent must assert the correct weaker property, not `π(g·s) = g·π(s)`.

### 4. C8 is a coarse approximation of SO(2)

45° granularity. Known, accepted, adjustable via `N`.

### 5. Numerical exactness — now measured

Measured 2026-09-16 by `tests/test_equivariance.py`. Tolerances come from these numbers rather
than being picked to make tests pass.

| What | Measured error | Asserted |
|---|---|---|
| Actor equivariance, all 8 elements | `< 1e-4` | yes, `< 1e-4` |
| Critic invariance, all 8 elements | `< 1e-4` | yes, `< 1e-4` |
| `equi_clip` commutation, all 8 elements | `< 1e-4` | yes, `< 1e-4` |
| Vision encoder at 90° multiples | `< 1e-3` | yes, `< 1e-3` |
| Vision encoder at 45° multiples | **≈ 0.32–0.34 relative** | reported only |

The actor, critic and clip act on flat feature vectors, where the group action is exact matrix
multiplication — those are tight and they hold. The vision encoder is exact at right angles
(`torch.rot90` is a pixel permutation) but **about a third off at the diagonal group elements**,
because rotating a pixel grid by 45° needs bilinear resampling.

So the C8 vision branch is genuinely equivariant for only half its group elements, and
approximately equivariant for the other half. This is a second, independent source of approximation
on top of the tilted camera in assumption 1, and it is larger than one might guess. Worth keeping in
mind if the equivariant advantage turns out smaller than the SO(2) paper's.

## Open questions

- Is the `GroupPooling` head better placed before or after the skip re-concatenation of
  proprioception and action?
- The critic head currently has no activation before pooling (the ReLU is commented out) and
  `GroupPooling` is doing double duty as the head's only nonlinearity. Is that deliberate?
- The critic head currently has **no normalization at all**, while the actor has active batch norm.
  `TrivialLayerNorm` — written by Aden, never run, preserved in TODO P3.4 — is the correct layer for
  the all-trivial features that `GroupPooling` produces, since escnn's `FieldNorm` would zero them.
  Restoring critic normalization is scheduled for P3.4, gated on the reproduction.
- The in-hand camera is invariant, which is correct, but it means half the visual input carries no
  symmetry information. How much does the equivariant agentview branch actually contribute?
