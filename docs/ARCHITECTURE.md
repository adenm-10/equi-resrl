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
| [resfit/rl_finetuning/scripts/train_rlpd_dexmg.py](../resfit/rl_finetuning/scripts/train_rlpd_dexmg.py) | RLPD baseline | Alternative baseline from the paper. Not currently used. |

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
    │   ├── ResidualTD3CoffeeConfig          task=TwoArmCoffee
    │   └── ResidualTD3TwoArmCanSortConfig   task=TwoArmCanSortRandom
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
| `residual_td3_coffee_config` | `ResidualTD3CoffeeConfig` | TwoArmCoffee | no |
| `residual_td3_two_arm_cansort_config` | `ResidualTD3TwoArmCanSortConfig` | TwoArmCanSortRandom | no |

## Config field provenance

**This is the most important table in this document.** Several `EquivarianceConfig` fields are
accepted, logged to wandb, and then ignored. A run's wandb config therefore does not describe the
network that actually trained. Verified by reading HEAD (`caf83f3`).

| Field | Default | Encoder | Critic | Actor | Verdict |
|---|---|---|---|---|---|
| `use_equivariant_model` | `True` | — | — | — | **Live** — picks equi vs. scalar-ablation stack |
| `N` | `8` | used | used (via group) | used (via group) | **Live** |
| `enc_degree_channel` | `32` | `n_out` | — | — | **Live** (last run overrode to 16) |
| `actor_degree_channel` | `128` | — | — | `hidden_dim` | **Live** |
| `critic_degree_channel` | `128` | — | `hidden_dim` | — | **Live** |
| `initialize` | `True` | passed through | hardcoded `True` | hardcoded `True` | **Partial** — only reaches the encoder |
| `use_norms` | `True` | commented out | commented out | 2 of 3 sites | **Partial** — see below |
| `use_orth_init` | `True` | — | commented out | live | **Partial** — actor only |
| `num_actor_layers` | `2` | — | — | ignored | **DEAD** — actor is hardcoded to 1 hidden layer |
| `num_critic_layers` | `2` | — | ignored | — | **DEAD** — critic head is hardcoded |

Also dead: `agent.actor.dropout` and `agent.critic.dropout` are threaded into both modules and
never used. `agent.critic.loss` reaches the equivariant critic as `loss_cfg`, is stored, and is
never read — so the distributional critic (C51 / HL-Gauss) is unreachable on the equivariant path
even though it is still settable in config.

Where the dead code is:

- `num_critic_layers` — [critic.py:30-33](../resfit/rl_finetuning/equi_off_policy/rl/critic.py#L30) commented out.
- `num_actor_layers` — [actor.py:75-84](../resfit/rl_finetuning/equi_off_policy/rl/actor.py#L75) commented out.
- `use_norms` in the critic — [critic.py:26,31,106,115](../resfit/rl_finetuning/equi_off_policy/rl/critic.py#L26) all commented out.
- `use_norms` in the encoder — [equi_encoder.py:35,37,58,93,128](../resfit/rl_finetuning/equi_off_policy/networks/equi_encoder.py#L35) commented out. The flag still controls `bias=not use_norms`, so the encoder is currently bias-free with no normalization following it.
- `use_orth_init` in the critic — [critic.py:68-70,140-142](../resfit/rl_finetuning/equi_off_policy/rl/critic.py#L68) commented out.

**Do not fix these yet.** See the freeze note in [STATUS.md](STATUS.md). They are documented here
so that experiment records can be read correctly in the meantime, and they are scheduled for
cleanup in Phase 3 behind bit-identity regression tests.

### What the networks actually are at HEAD

Critic Q-head, per ensemble member
([EquiHeadMLP](../resfit/rl_finetuning/equi_off_policy/rl/critic.py#L11)):

```
Linear(in → hidden, regular repr) → GroupPooling → Linear(pooled → 1, trivial repr)
```

No activation between the first Linear and the pooling — the ReLU on line 28 is commented out.
`GroupPooling` (a max over the group channels of each field) is itself nonlinear, so the head is
not purely linear, but it is shallower and has fewer nonlinearities than the config implies.
This `GroupPooling` layer is what makes the Q output invariant rather than equivariant, and must be
preserved. **Note this head is HEAD's, not the successful run's** — at `7dae4925`, where `z8yoqylh`
ran, `FieldNorm` and the ReLUs were still active here. Why that matters, and why the earlier
"`FieldNorm` broke the critic" reasoning no longer holds, is in
[EQUIVARIANCE.md](EQUIVARIANCE.md), "What actually changed".

Actor policy head ([actor.py:68-72](../resfit/rl_finetuning/equi_off_policy/rl/actor.py#L68)):

```
Linear(in → hidden) → IIDBatchNorm1d → ReLU → Linear(hidden → action reps)
```

One hidden layer, and that batch norm is unconditional (it ignores `use_norms`).

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
| `networks/equi_normalizer.py` | `LinearNormalizer`, `build_equivariant_normalizer`, `detect_robot_base_xy`. | yes |
| `networks/ablate_equi_obs_encoder.py` | **Unreachable.** Superseded by `obs_encoder.py`. | **no** |
| `common_utils/crop_randomizer.py` | Random crop 84×84 → 76×76. | yes |
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
| `utils/normalization.py` | `StateStandardizer`, `ActionScaler`, and the identity versions used on the equivariant path. |
| `utils/hugging_face.py` | Downloads BC checkpoints from wandb and buffer caches from HF. |
| `utils/checkpoint.py`, `utils/evaluate_dexmg.py`, `utils/rb_transforms.py`, `utils/dtype.py` | Checkpointing, evaluation rollouts, replay-buffer transforms, dtype helpers. |
| `dexmg/environments/dexmg.py` | `create_vectorized_env` — robosuite/mimicgen env construction. |

## How a training step flows

```
BasePolicyVecEnvWrapper.step(residual_naction)
  ├─ combined = last_base_naction + residual_naction
  ├─ env.step(combined)
  ├─ base_policy.select_action(next_raw_obs)      # frozen ACT or Diffusion
  └─ obs["observation.base_action"] = base_naction

ResObsEnc.forward(obs)                             # equi_off_policy/networks/obs_encoder.py
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

QAgent._combine_actions(base, residual)
  └─ equi_clip(base + residual, layout, bound=1.0)
```

Note the skip connections: proprioception and action are re-concatenated at every stage rather
than only entering at the input. Both actor and critic do this.

## Normalization lives in two different places

This trips people up, so it is worth stating plainly.

**Baseline path.** `StateStandardizer` and `ActionScaler` are built from dataset statistics in
`train_residual_td3.py` and applied *outside* the network, in the env wrapper and buffer.

**Equivariant path.** Those two become `IdentityStandardizer` and `IdentityScaler`
([train_residual_td3.py:305-307](../resfit/rl_finetuning/scripts/train_residual_td3.py#L305)), and
normalization moves *inside* the encoder via `build_equivariant_normalizer` and
`ResObsEnc.set_normalizer`. It has to move, because normalization must be applied per
representation to preserve equivariance — see [EQUIVARIANCE.md](EQUIVARIANCE.md).

So "where is the data normalized" has two different answers depending on which path you are on,
and comparing statistics across the two paths requires knowing which.

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
9. Main loop to `algo.total_timesteps`: step env, add to buffer, do `num_updates_per_iteration` gradient updates, evaluate every `eval_interval_every_steps`, checkpoint on best success rate.

`eval_first: bool = True` means there is an evaluation at step 0. Because
`agent.actor.actor_last_layer_init_scale = 0.0`, the residual starts at exactly zero, so **the
step-0 evaluation measures the base BC policy alone.** That number is the baseline every run must
be compared against, and it is why an experiment record that only reports "best success rate" can
be misleading.

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
| Past local run outputs | `local_runs/run_<timestamp>_.../` |
| Run launchers | `resfit/rl_finetuning/shell/paper_runs/<task>/` and `.../ablations/<task>/<knob>/` |
| Vendored dependencies | `deps/{lerobot,robosuite,mimicgen,dexmimicgen}/` (gitignored, 1.7 GB) |

### Reading a past run without wandb

Everything needed is on disk under `wandb/run-*-<run_id>/files/`:

- `wandb-metadata.json` — the exact command line, and the git commit it ran at.
- `config.yaml` — the fully resolved config. Remember the provenance table above: some of these
  values did not affect the network.
- `output.log` — per-step training lines and the evaluation markers, one `✓`/`✗` per episode.
- `requirements.txt` — the Python environment at run time.

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

## Known rough edges

Recorded so they are not rediscovered. None are blocking.

- `train_residual_td3.py` has several commented-out `assert False` debugging blocks around dataset
  stats printing.
- `load_policy.py` infers the policy type by checking for the substring `diffusion` in the config's
  `type` field, and otherwise by the *presence of a `use_vae` key* to detect ACT. The second test is
  incidental rather than meaningful. Carries its own `TODO` upstream.
- `QAgent._act_default_equi` calls `actor.forward(feat, std=0)` for evaluation. In
  `Actor.forward`, `std=None` returns the unclipped mean while `std=0` returns a clipped mean.
  Two different code paths for what reads as the same intent.
- `Critic.forward` accepts a `return_logits` keyword and ignores it.
- Two git stashes exist: `stash@{0}` on `aabde0f`, `stash@{1}` on `d9909ce`. Contents unreviewed.
