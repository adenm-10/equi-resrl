# Tests

The pre-submission gate. Before Phase 2 this repo had **zero** tests, and escnn
field-type mismatches were being discovered by launching training runs — 43 of
197 failed runs died with escnn types in a traceback.

## Running them

```bash
conda activate residual
export PYTHONPATH="$PWD:$PYTHONPATH"

python preflight.py                 # the full gate: use this before a launch
python -m pytest tests/ -q          # just the tests
python -m pytest tests/ -q -s       # with measured equivariance errors printed
```

`./submit.sh <arm> <seed> [gpu]` runs `preflight.py` automatically and refuses to launch if a
blocking check fails. `SKIP_PREFLIGHT=1` overrides that; don't.

## What each file does

| File | Tier | Needs GPU | Checks |
|---|---|---|---|
| `test_imports.py` | 1 | no | Every module under `resfit/` imports. Catches the `SyntaxError` class of bug that sat undetected in `rl_utils.py` for ~5 months. |
| `test_config.py` | 1 | no | Hydra configs resolve; the reproduction config still has the hyperparameters the experiment log claims. The four tuned-RLPD configs are the residual design made vanilla, with each stack's critic heads; a vanilla run refuses any residual-only field that is on; their cache keys are pinned and share none with a residual key. The online keys record mujoco's version from 2026-10-07; the offline keys did not move. |
| `test_regression_single_arm.py` | 1 | no | Fingerprints the single-arm encoder output against goldens captured at `b8e1de7`. Pins Can/Square behaviour across the bimanual generalisation. |
| `test_regression_agent.py` | 1 | no | Fingerprints the whole equivariant agent at the Square reference config (parameters, actor and critic outputs, one TD3 update) via `tools/agent_fingerprint.py`. Golden captured at `f3e6315`, identical at `b8e1de7`. Makes new-commit Square runs comparable to the reference seeds in their update; act() on the env's images changed at `e15d36c` (`test_obs_formats.py`). A second golden pins the same agent at `actor_norm=none`, the default from 2026-09-30, identical at `1e33b8f`. Also checks that the target critic computes with its soft-updated weights (the 2026-09-29 fix), that `frozen_target_critic` reproduces the old bug, that skipping an unlogged update's metrics (`compute_metrics=False`) leaves 8 updates bit-identical, and that so does logging the critic's Q by buffer after each, in eval mode, as the trainer does from 2026-10-07. |
| `test_run_package.py` | 1 | no | Run packages persist locally, wandb keeps exactly one best upload (checked against a fake wandb), and a saved residual model reloads to identical actions, in each network family; saving one mid-training (`checkpoint_interval`, 2026-10-12) changes no parameter, buffer, mode or random state. Records and BC downloads land under `CACHE_DIR`, not the working directory, so a launch worktree holds only code. The manifest records the package versions the commit does not pin, mujoco among them. |
| `test_render_check.py` | 1 | no | The arithmetic of the gate's render check (`tools/render_check.py`), on synthetic frames: the gap between average frames, its bar of 2.5x the gap between the even and the odd demos, a band rendered differently failing it, and frames outside [0, 1] refused. The check itself, on the dataset and the env, runs in `preflight.py` ("Known gaps"). |
| `test_gpu_queue.py` | 1 | no | `tools/gpu_queue.sh`, run against a scratch queue and a stub `submit.sh`: a queued line launches once and STOP ends the runner; a line it cannot delete (as on the full disk of 2026-10-07) is not launched, and neither is any line below the disk floor; a log line cut off without its newline does not swallow the next entry; rewriting the script under a running runner changes nothing. All but the first fail on the script before 2026-10-07. |
| `test_demo_codec.py` | 1 | no | `encode_env_frames_like_demos` (2026-10-08 CODE): the codec's settings are LeRobot's and the Can dataset's, keyframe pattern included; its decode gives the dataset loader's frames bit for bit; the env wrapper codes its frames when asked and is bit-identical to before when not; the switch moves the online cache key and no other, and the trainer passes it to both envs. Rendered, in a subprocess (`tools/codec_check.py`, EGL), coded env reset frames cannot be told from demo first frames by colour or detail (ROC area within [0.15, 0.85]) where raw ones are told apart by colour at 0.95 or more. |
| `test_rerender_demos.py` | 1 | no | The re-rendered demos (2026-10-08 CODE): after a MuJoCo step the body positions are the previous substep's, and `as_stepped` recovers them from the state (a small model, no renderer); the stepped observation.state may move only the end-effector pose. Rendered, in a subprocess (`tools/rerender_demos.py --check-only`, EGL), every guard passes on Can's first two demos, the identity guard against the trainer's own step path in its own process: the state exactly, frames up to the GPU's residue (1 level on at most 32 pixels; 2026-10-09 CODE). A built dataset is lossless (PNG, no video), complete, and records passing guards and a clean commit, and `tools/codec_check.py --rerendered` cannot tell raw env frames from its demo frames (skipped until built). The dataset name moves the offline cache key and no other, and the gate reads it from the composed config. |
| `test_env_wrapper.py` | 1 | no | When the env wrapper resets the base policy: ACT only the finished envs, diffusion all of them, and never on a timeout (a pinned limitation). Guards the TODO B3 fix. `NormalizedVecEnvWrapper`, vanilla RL's, steps exactly as the residual wrapper does when the base action normalizes to zero: same env action, buffer action, observations and rewards. |
| `test_eval_base_policy.py` | 1 | no | The base-rate script's zero-residual stub drives the trainer's real evaluator on a scripted env: same episode counting (in-flight episodes dropped at the last one), residual exactly zero. The evaluator's tilt readout (2026-10-12) reports each episode's largest tilt from its start, not its yaw, and starts afresh after the auto-reset. |
| `test_square_verdict.py` | 1 | no | The stage-1 verdict of 2026-10-01 (`tools/square_verdict.py`) and the watcher's decisions (`tools/run_watch.py`), on runs built to sit either side of each rule: a rising run advances, a flat one or one climbing back to its start does not, a collapse fails, nothing after 100k counts; the watcher stops only runs launched past 100k, and any collapse; the squash and depth arms follow the 2026-09-30 table. |
| `test_field_metrics.py` | 1 | no | `utils/field_metrics.py` (2026-10-12): the action's fields in both stacks' order, for one arm, two arms with six-dof hands, and any other shape one component per field; the metrics split the online and demo rows, keep the signed mean's sign, report the critic's slope per sample (the actor loss's gradient times the batch), and skip an empty demo half. |
| `test_field_ablation.py` | 1 | partly | `tools/residual_field_ablation.py` (2026-10-12): each pre-registered variant keeps exactly its components, on every arm; the masked agent is the agent but for its masked residual. On the GPU, a saved model loads through the trainer's `build_agent`, and the unmasked agent acts as saved. |
| `test_encoder_probe.py` | 1 | no | `tools/encoder_probe.py` (2026-10-23): the nut pose it reads from a demo state set into the demos' scene is robosuite's own `NutAssemblySquare` env's from that state (and the grip site its `robot0_eef_pos`); its targets' layout; the episode split never puts one episode on two sides; ridge on principal scores, by the covariance and by the kernel, recovers a linear map; the AUROC is the pairwise count. |
| `test_interface.py` | 2 | partly | The encoder's interface to the critic (2026-10-23 CODE): the encoders in two slices around their 9x9 maps compute what they compute whole, bit for bit, equivariant and exported; a window needs the visual projection, and the ResFiT encoder refuses the other encoders' fields. On the GPU, each of `enc-iface-1023`'s arms hands its critic and its actor the numbers the plan lists, and `iface-window` starts as `iface-eee` at seed 2: the same random state after construction, every shared tensor bit for bit, the same features, Q and first critic loss, a window share of 0; after two updates every window weight has moved. A model saved before the interface (`wk45zp27`) loads into `iface-eee`. |
| `test_actor_design.py` | 2 | partly | The actor's design (2026-10-26 CODE): `ResFiTShapedActor` at `actor-resfit-shape`'s config has ResFiT's widths in channels (128, 1,024, 1,024, then the action) and reads the state and base action once, at its first hidden layer; at 362 fields its hidden-to-hidden Linear has ResFiT's free parameters; it starts at zero; it is equivariant with and without its norm and squash, in eval and train (GPU); the agent refuses the design in the ablated family, behind ResFiT's actor and in vanilla RL, and its width where no actor reads it. |
| `test_layouts.py` | 2 | yes | Declared `FieldType`s match the physics; widths agree at every `GeometricTensor` boundary. The encoder's `action_fields` name each layout entry and match the fields `field_metrics.py` logs. Vanilla RL's encoder gives the actor exactly the critic's features and needs no base action; its actor takes one action block fewer at each stage and refuses features that carry one; any network of the ablated or ResFiT family refuses vanilla mode. |
| `test_equivariance.py` | 2 | yes | Actor is equivariant, critic is **invariant**, `equi_clip` and `equi_tanh` commute with the group. Every sweep option (each norm, squash, no visual projection, deeper heads, the inits) keeps the symmetry in train and eval mode. So does the vanilla actor, with no base action, for every actor norm; with `equi_tanh` its output, the whole action, is bounded per block, and without it only the explored action is. `RegularInnerBatchNorm` gives escnn's InnerBatchNorm's outputs, gradients and running statistics, flat and spatial. A per-field residual bound (`equivariance.residual_field_bound`, 2026-10-12) keeps the actor equivariant, with and without the squash. The interface's options (2026-10-23): the agentview encoder's output at `enc_out_channel`, at `iface-wide-all`'s width, and with the critic's window, equivariant at right angles (the window's 45° error reported); the critic with a window, its projection moved off zero, invariant. |
| `test_ablated.py` | 2 | yes | The ablated family (`equivariance.*_family=ablated`, 2026-10-20): at init0's and `arch-wide`'s configs, seed 2, the ablated stack starts as the equivariant stack's function: the same torch and numpy random state after construction (exporting draws nothing), the same encoder features bit for bit, the same actor means (at escnn's init, not zero) and Q values, and the same first critic loss. Its parameters are exactly the equivariant layers' expanded weights (3.73M, 26.9M, 22.6M at `arch-wide`). **The negative control:** after four updates at `arch-wide`'s rates, the ablated actor and critic are not equivariant (errors 2.3 and 0.41) while the equivariant ones still are (below 1e-6), so the equivariance tests can fail. A layer with no plain form (the batch norms, FieldNorm) refuses to export. At the three `enc-iface-1023` interfaces the ablated stack starts as the equivariant one too, window included, and the ablated critic reads a window moved off zero as the equivariant one does. |
| `test_resfit_heads.py` | 2 | partly | The ResFiT family (`equivariance.*_family=resfit`, 2026-10-20): at one seed the ResFiT encoder's ViTs are ResFiT's own agent's, bit for bit, and their tokens are ResFiT's own `_encode`'s from uint8 and float images; in train mode it shifts them as ResFiT's `augment=True` does at the same seed, and in eval mode not at all. The critic is off_policy's `Critic` on the same weights and inputs. The actor has off_policy's `Actor`'s parameters at the same seed less its Tanh, and its mean is `equi_tanh` of the same pre-activation times the bound, where ResFiT's is tanh. Every refusal of `check_families` raises, and the field it names at its default does not (vanilla RL's is `test_layouts.py`'s). |
| `test_no_ops.py` | 2 | yes | Pins which config fields actually do nothing, and which sites each live one reaches. `equivariance.residual_field_bound` (2026-10-12): the empty mapping is the actor bit for bit; a field's bound replaces `action_scale` on that field's components alone, on every arm, and adds no state-dict entry; unknown fields and the vanilla actor raise. |
| `test_square_rerendered_watch.py` | 1 | no | `tools/square_rerendered_watch.py` applies `square-rerendered-1009`'s rules as pre-registered and amended (2026-10-09), on `oj5k0jqb`'s and `kyv7clfc`'s eval sequences: the residual is stopped at its collapse (`kyv7clfc`, 30k) and tuned RLPD never for collapse; the residual's 100k check passes at 155 of 250 and not 154 (`oj5k0jqb`, 165, passes); tuned RLPD's passes at an eval of 0.04 or 5 training successes; convergence from 150k (`oj5k0jqb` still rising); a 100k window without five evals is left to a human. The later groups: `square-trlpd-stab-1010` under tuned RLPD's rules, `square-residual-start-1011` under the residual's, `resfit-rx-1012` stopped only at its collapse (neither ResFiT anchor nor a held base is stopped through 150k), and `residual-tilt-1012`, whose `notilt` follows the residual's rules and whose `diag` is never stopped. From the 2026-10-12 AMENDMENT, every arm of `resfit-equi-diff-1012`, `equi-resfit-arch-1013`, `equi-residual-long-1012`, `resfit-ln-place-1013`, `equi-arch-parts-1015` and `equi-actor-speed-1016`, and `notilt`, runs to a fixed 150k: never stopped for collapse, still stopped on the divergence stop (`notilt` at 150k, being launched for 300k). `equi-speed-matched-1017`'s four arms are watch-only (2026-10-17 DEVIATION, no automatic stop of any kind): never stopped on any eval sequence, a diverged critic included. So are `arch-swap-1021`'s six, and each one's early speed (`train/residual_l1_magnitude` over steps 1–3,000, once it has logged past 3k) is logged once: `SPEED` inside 0.025–0.045, a `FLAG` outside; `h216n857`'s record reads 0.0367. `enc-iface-1023`'s four arms are watch-only and log their early speed too. |
| `test_sweep_arms.py` | 2 | partly | Every arm in `submit.sh` is a pre-registered one and launches its pinned overrides, which compose through the trainer's Hydra configs to the pre-registered recipe: its anchor's launch but for its listed changes, the anchor being what its run launched with and composing today to that run's recorded config. Rewritten with `submit.sh` for each group: at `60fce72` for `resfit-rx-1012` (anchor `oj5k0jqb`; the rx arm moves the `ActionScaler` on rx alone), at `be0b7bb` for `residual-tilt-1012` (anchor `eaqorqki`), at `7db124d` for `resfit-equi-diff-1012` (anchor `oj5k0jqb`; `limits` and `noln-all` join with their code: `noln` and `noln-all` start from the anchor's weights less its LayerNorms, and `noln-all` has no norm anywhere), and with `equi-resfit-arch-1013` (anchor `eaqorqki`, less its run length) in the same file, each group against its own anchor: its arms have ResFiT's layout, RegularLayerNorm after every hidden layer and 10 heads of 2 hidden layers, and a zero-initialized actor is checked for equivariance at escnn's own init. At `25b22aa` `resfit-ln-place-1013` joins (anchor `oj5k0jqb`): each placement arm has no LayerNorm in its one place and the anchor's elsewhere. With `equi-arch-parts-1015` (anchor `eaqorqki`; `arch-wide` moves into it), each of `arch-wide`'s four parts alone has that part and the anchor's design elsewhere; at seed 2, `actorln` and `criticln` start from the anchor's weights bit for bit less their RegularLayerNorms, `actor448` from its encoder and critic, `heads` and `arch-wide` from its encoder. With `equi-actor-speed-1016` (anchor `eaqorqki`), `actorln448` has both actor parts and the anchor's critic and starts from its encoder and critic; `actorlr1e-5` and `actorlr3e-5` start from the anchor's weights bit for bit, and each arm's actor AdamW steps at its rate (1e-6 in the anchor), its critic's and encoder's at 1e-4, with no scheduler and weight decay 0.01. With `equi-speed-matched-1017` (anchor `eaqorqki`), each arm is its parent arm (`actorln`, `actor448`, `actorln448`, `init0-long`) but for `agent.actor_lr`: its overrides are its parent's and its parent's seed-2 run's recorded argv (`8sgfkskv`, `vnpqtnva`, `x3xopn1y`, `oav5gk76`) plus the rate, its composed config differs from its parent's in the rate alone, it has its parent's design, it starts from its parent's weights bit for bit at seed 2, and its actor steps at its rate. With `arch-swap-1021` (anchor `eaqorqki`; 2026-10-20), each of the six arms launches its overrides, composes to the anchor but for its listed changes (its families off their default and its rate), and `arch-swap-eee` is `arch-wide`'s launch exactly; on the GPU, each has the network families it names, every input reaches its critic and its actor, the pairs that differ in one network (`ere`/`eee`, `ara`/`aaa`, `ara`/`ere`, `rra`/`rrr`) start from the same weights in the others at seed 2, and each actor's AdamW steps at its rate (ResFiT's 1e-6, `aaa`'s 2.5e-7), the critic's and encoder's at 1e-4. With `enc-iface-1023` (anchor `eaqorqki`; 2026-10-23), `iface-eee` is `arch-swap-eee`'s launch exactly and each other arm is it but for its interface fields; on the GPU each is equivariant and invariant at its own sizes, window included, every input reaches it, and its rates are the anchor's (`tests/test_interface.py` checks what each builds). On the GPU, each arm builds its agent, runs one TD3 update and starts from the anchor's weights bit for bit; for an equivariant arm, its actor is equivariant and its critic invariant at the arm's own sizes, every input reaches each of the critic's heads (from 2026-10-15; the heads' mean before) and the actor, it has the anchor's no-norm, five-head design, its residual starts at zero and learns (`notilt`'s tilt staying exactly zero), the anchor's saved model loads into it, and the scalar ablation refuses a field bound. The FieldNorm critic of the fixed-env baselines stays on record as blind to every trivial input. |
| `test_sync_free.py` | 2 | partly | The update makes no host sync (STANDARDS 6.5): for every config in `tools/agent_fingerprint.py`'s `CANDIDATES` without the actor's IID batch norm (the 2026-09-30 schedule's arms and their combinations, `actor-fn`, `actor-inner`), for the encoder's InnerBatchNorm, for the two-arm layout, for vanilla RL's agent at the `actor_norm` default, for the five mixes of network families `arch-swap-1021` runs (the ablated layers, ResFiT's ViTs without their `VitEncoder.forward` syncs, its random shifts, actor and critic), and for `enc-iface-1023`'s three interfaces and the ablated window. On CPU too: escnn is 1.0.11, and each fix in `escnn_fixes.py`, the crops and the rotation reorder give exactly what they replaced. |
| `test_critic_diagnostics.py` | 2 | yes | Both agents' `critic_diagnostics`, the Q split by buffer the trainer logs (`debug/q_{data,policy}_{online,demo}`, `debug/policy_data_gap_*`): it reports what it says, in vanilla and residual mode, skips an empty demo half, and moves no parameter and draws no random number, so logging leaves a run's numbers alone. The equivariant agent's (from 2026-10-07) is checked at the baselines' design, and its policy action equals the one its actor loss scores, bit for bit. `debug/dQ_da_*` is nonzero after an actor update; it read 0 in every ResFiT run before 2026-10-06. |
| `test_action_bound.py` | 2 | yes | `agent.actor.action_bound` and `agent.actor.noise_units` (from 2026-10-12). Under `noise_units=env`, σ and its clip are in env units on every axis; both settings start from the same weights; the field is refused where it means nothing. At `box`, ResFiT's agent and `ActionScaler` are as they were: parameters, the eval action, `critic_diagnostics` and one TD3 update, measured in deterministic mode in a subprocess, against goldens captured at `8ce1642` before the field existed (the tree with it gave the same values exactly, on both GPUs). At `equivariant`, on demo statistics off-centre on every axis: the residual's mean, in env units, is `action_scale · equi_tanh` of the layer's output, inside a circle of radius 0.1 on xy and on tilt and within ±0.1 on z, yaw and gripper, and reaches the bound; the critic scores the sum clipped as `equi_clip` clips it in env units, which commutes with rotating both pairs; the executed action is the unclamped sum, for robosuite's ±1 input range to clip; both bounds start from the same weights; the field is refused on a vanilla actor, without the run's unclamped scaler, and by the equivariant agent. |
| `test_bc_term.py` | 2 | yes | The actor's BC term (`agent.bc_loss_coef > 0`, TODO B14): the update with the demo half logs `rft/bc_loss` and still logs the critic's own slope `debug/dQ_da_*`; the term pulls the actor toward the demos' actions where the same updates without it push the actor away; and the BC step moves the encoder iff `agent.bc_backprop_encoder` is set. It does not run the trainer: that the trainer passes the demo half was checked by a smoke run, not by this file. |
| `test_obs_formats.py` | 2 | yes | The agent acts on the observations the env produces, not only on buffer batches: the env's float images and the buffer's uint8 copies give the same encoder features (each encoder family: the equivariant, the ablated, and ResFiT's own preprocessing) and the same `act()` at the Square reference config, and for vanilla RL's agent, whose observations have no base action. The critic's window reads both formats alike. `float_images_as_uint8` still reproduces the bug of every run before 2026-10-02 and leaves the buffer format, all the update sees, bit-identical. Every other test feeds uint8 only, which is how that bug went unseen. |
| `group_action.py` | — | — | Independent implementation of the C_N action. Not a test file. |

## Arm count

The `obs_enc` fixture is parameterised over one-arm (Can, Square) and two-arm (TwoArmBoxCleanup)
layouts, so every Tier 2 assertion runs twice. Test ids are suffixed `[1arm]` / `[2arm]`.

Both `test_regression_*.py` modules build on **CPU** against CPU-captured goldens. escnn caches basis
tensors per representation, so once a GPU module exists a later CPU build fails on a device
mismatch. `conftest.py` sorts those modules first for this reason; preflight is unaffected because it
runs Tier 1 in its own subprocess. The agent's goldens also hold only at the default CPU thread count:
at `OMP_NUM_THREADS=6`, 7 of its 22 values move (2026-10-21), so run Tier 1 without it.

## Two design decisions worth knowing

**1. The group action is implemented independently.** `group_action.py` derives the
action from the physics and from `docs/EQUIVARIANCE.md`, not by importing the
`FieldType` declarations from the modules under test. If the tests read the same
declarations the code uses, a wrong declaration would pass its own test.

This splits the checking into two honest halves:

- `test_layouts.py` compares the **declared** representation against escnn's own
  representation matrices for the independently-built layout. This catches a wrong
  irrep assignment — the failure mode where shapes still line up, the network still
  trains, and the symmetry is silently false.
- `test_equivariance.py` checks the **implementation** commutes with the declared
  action. This catches a misplaced `GroupPooling` or a non-commuting normalizer.

Conflating them into one test would catch neither reliably.

**2. Several tests exist only to stop a vacuous pass.** An equivariance test on a
network that outputs zeros passes trivially. So:

- `test_actor_output_is_nonzero` — the reproduction runs with
  `actor_last_layer_init_scale=0.0`, which makes the actor output exactly zero. The
  fixture deliberately uses `None` instead, and this asserts it.
- `test_critic_q_depends_on_action` — a critic that ignores its action is also
  "invariant". The residual-saturation failure mode showed `dQ_da ≈ 2e-4`, i.e.
  almost no action dependence, which an invariance test alone would not notice.
- `test_equi_clip_actually_clips` — asserts the test input exceeded the bound, so
  the clipping path was actually exercised.

## Measured equivariance error

Recorded rather than tuned. Tolerances come from these numbers, not the reverse.

| What | Error | Why |
|---|---|---|
| Actor, critic, `equi_clip`, all 8 group elements | `< 1e-4` | These act on flat feature vectors, so the group action is exact matrix multiplication. Asserted tightly. |
| Vision encoder at 90° multiples (k = 0, 2, 4, 6) | `< 1e-3` | `torch.rot90` is an exact pixel permutation. Asserted. |
| Vision encoder at 45° multiples (k = 1, 3, 5, 7) | **≈ 0.32–0.34 relative** | Rotating a pixel grid by 45° requires bilinear resampling, which is not exact. Reported, not asserted tightly. |

That 45° figure is a property of resampling a discrete grid, not a bug — but it is
large, and it is worth knowing that the equivariant vision branch is only
approximately equivariant at the diagonal group elements. This is separate from,
and additional to, the tilted-camera approximation discussed in
`docs/EQUIVARIANCE.md` assumption 1.

## Known gaps

Honest list. None blocks the reproduction.

- **No test of the full `QAgent`.** The composed `act()` and `_combine_actions`
  path is untested for equivariance, including the weaker property that holds
  when a non-equivariant base policy is added to an equivariant residual. TODO
  P2.4. `test_obs_formats.py` checks act() only for agreement between the env's
  and the buffer's image formats.
- ~~**No negative control.**~~ Closed 2026-10-20: `test_ablated.py` shows the ablated
  family, which starts as the equivariant function, no longer equivariant after a few
  updates, so the equivariance tests can fail.
- **The normalizer is untested.** `build_equivariant_normalizer` should be checked
  for isotropic scale and zero offset on every `irrep(1)` field; right now that
  property is only guaranteed by reading the code. TODO P2.2.
- **Two modules are skipped** in `test_imports.py`, both justified in-file:
  `dexmg.py` needs a live simulator, and `train_bc_dexmg.py` calls
  `parse_args()` at module level (line 217, outside the `__main__` guard at 905).
  The second is a real defect, documented by
  `test_bc_script_parses_args_at_import_time` so it cannot be forgotten.
- **Tier 3 is not a test.** Resource and artifact checks live in `preflight.py`
  and are not pytest cases.
- **Nor is the render check.** `preflight.py` runs `tools/render_check.py` on the real dataset
  and env, Can and Square (about a minute), and fails on either; with it, a mujoco other than
  3.3.2 fails Tier 1 outright. Its bar was set by measurement (the tool's docstring): mujoco 3.8.1
  fails it on both tasks, 3.3.2 passes with 7 standard deviations to spare on Can and 9 on
  Square. Env resets are unseeded, so the gap it reports varies by about 0.0005 between calls.
