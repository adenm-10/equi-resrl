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

`./submit.sh <seed>` runs `preflight.py` automatically and refuses to launch if a
blocking check fails. `SKIP_PREFLIGHT=1` overrides that; don't.

## What each file does

| File | Tier | Needs GPU | Checks |
|---|---|---|---|
| `test_imports.py` | 1 | no | Every module under `resfit/` imports. Catches the `SyntaxError` class of bug that sat undetected in `rl_utils.py` for ~5 months. |
| `test_config.py` | 1 | no | Hydra configs resolve; the reproduction config still has the hyperparameters the experiment log claims. |
| `test_regression_single_arm.py` | 1 | no | Fingerprints the single-arm encoder output against goldens captured at `b8e1de7`. Pins Can/Square behaviour across the bimanual generalisation. |
| `test_layouts.py` | 2 | yes | Declared `FieldType`s match the physics; widths agree at every `GeometricTensor` boundary. |
| `test_equivariance.py` | 2 | yes | Actor is equivariant, critic is **invariant**, `equi_clip` commutes with the group. |
| `test_no_ops.py` | 2 | yes | Pins which config fields actually do nothing. |
| `group_action.py` | — | — | Independent implementation of the C_N action. Not a test file. |

## Arm count

The `obs_enc` fixture is parameterised over one-arm (Can, Square) and two-arm (TwoArmBoxCleanup)
layouts, so every Tier 2 assertion runs twice. Test ids are suffixed `[1arm]` / `[2arm]`.

`test_regression_single_arm.py` builds on **CPU** against CPU-captured goldens. escnn caches basis
tensors per representation, so once a GPU module exists a later CPU build fails on a device
mismatch. `conftest.py` sorts that module first for this reason; preflight is unaffected because it
runs Tier 1 in its own subprocess.

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
  path is untested, including the weaker property that holds when a
  non-equivariant base policy is added to an equivariant residual. TODO P2.4.
- **No negative control.** Nothing yet asserts that
  `use_equivariant_model=False` *breaks* equivariance, i.e. that these tests can
  fail. TODO P2.4.
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
