# Status

**Last updated:** 2026-09-16 · **Last verified commit:** `17e4f8a`

This file describes what is true right now. Rewrite it in place; do not append. For history, see
[EXPERIMENTS.md](EXPERIMENTS.md) and [PROGRESS.md](PROGRESS.md).

## Where the project is

**The lost run was found.** `z8yoqylh` was never deleted from disk — it was on a second workstation
the earlier audits never checked. Its full record survives: resolved config, launch arguments, git
commit, environment, and evaluation log. So the project is no longer reconstructing a run from
circumstantial evidence; it is replicating a documented one.

**A 3-seed replication is running right now** on that second machine. Until it reports, the
successful result is still a single seed.

## The two machines

Both are needed to read the history, because host correlates perfectly with outcome.

| | `ZXP-S-works` | `boce-WS-01` (`10.188.60.163`, user `aden`) |
|---|---|---|
| GPUs | 2 × RTX 3090 | 2 × RTX 4090 |
| RAM | 62 GB | 125 GB |
| Stack | torch 2.9.1, torchrl 0.9.2, escnn 1.0.13 | torch 2.6.0, torchrl 0.7.0, **escnn 1.0.11** |
| Repo | `~/Desktop/projects/equi-resrl` | `~/projects/equi-resrl` |
| Role | **every collapsed equivariant run** | **the one success, `z8yoqylh`** |
| Now | idle | running the 3-seed replication |

`boce-WS-01` matches `z8yoqylh`'s recorded `requirements.txt` field for field. Key auth works from
`ZXP-S-works`. Buffer caches exist on both but hash differently, because torchrl and tensordict
versions are part of the cache key — that difference is benign.

## Running now

Launched 2026-09-16 16:00 EDT on `boce-WS-01`, from a git worktree at `7dae4925`
(`~/projects/equi-resrl-7dae4925`), via `~/launch_repro.sh <seed> <gpu>`. Seed-group
`equi-repro-7dae4925`, pre-registered in [EXPERIMENTS.md](EXPERIMENTS.md).

| Seed | wandb | GPU | Role | ETA |
|---|---|---|---|---|
| 2114495708 | `9qgsaxsr` | 0, alone | exact replication | ~47 h |
| 1 | `4sseggkv` | 1, shared | seed robustness | ~85–90 h |
| 2 | `dxbept4u` | 1, shared | seed robustness | ~85–90 h |

Startup verified on all three: seeds match, both caches hit (`8edf618e` offline, `43637c10`
online), no rebuild, no errors. Seed 2114495708's step-0 eval is **0.84** against `z8yoqylh`'s 0.82,
consistent within 50-episode noise.

**Two operational notes.** Each run holds ~25 GB resident, not the ~21 GB previously assumed, so
three is the ceiling on that machine — do not add a fourth. And stdout is block-buffered into
`~/repro_s<seed>.log`, so those logs lag reality by thousands of steps; read
`wandb/run-*/files/output.log` instead, or check CPU time.

## Freeze in effect

**Do not modify anything under `resfit/rl_finetuning/equi_off_policy/` until the replication
reports.**

The reason has changed but still holds. It is no longer that the target config is unknown — it is
known. It is that three runs are in flight and the analysis compares them against `z8yoqylh`. The
running worktree is pinned at `7dae4925`, so edits at HEAD cannot disturb them mechanically, but
they would make "did HEAD reproduce it?" a different question from the one being answered.

## What the recovered metadata settled

Full detail in [EXPERIMENTS.md](EXPERIMENTS.md), 2026-09-16 correction entry.

| | Believed before | Actually |
|---|---|---|
| Commit | `caf83f3` | **`7dae4925`** |
| `actor_last_layer_init_scale` | 0.0, no override known | **1e-4**, passed on the CLI |
| Stack | torch 2.9.1 / escnn 1.0.13 | torch 2.6.0 / escnn 1.0.11 |
| Seed | unknown | **2114495708** |
| best | ~0.92, from a screenshot | **0.94**, from the log |
| Runtime | estimated 58 h | **47.3 h**, measured |

`enc_degree_channel = 32` was guessed correctly. Everything else above was wrong.

## Three things that shape everything downstream

### 1. `7dae4925` produced both the collapses and the success

So it is not an architectural failure at that commit, and **the `FieldNorm` hypothesis is dead** —
`FieldNorm` and the ReLUs were present in the critic head of the run that reached 0.94. The tag
`repro-z8yoqylh-base` points at `caf83f3` and is wrong; `repro-z8yoqylh-actual` points at
`7dae4925`. Both are pushed; the wrong one is left in place rather than moved.

### 2. Two candidate causes, confounded

`z8yoqylh` and `a3e3zylp` share a commit and a config and differ only in the init-scale override and
the host/stack. Isolating them takes two more runs and is deliberately deferred until the
replication confirms the result. See [EQUIVARIANCE.md](EQUIVARIANCE.md).

### 3. The successful run had robot-base centering disabled

At `7dae4925` every robot-base call site is commented out, so `z8yoqylh` rotated about the **world
origin**, not the robot base — the case EQUIVARIANCE.md describes as silently breaking the symmetry.
It reached 0.94 anyway. The feature is active at HEAD, so **HEAD is not the architecture that
produced 0.94.** Recorded in [EQUIVARIANCE.md](EQUIVARIANCE.md), "The rotation center".

## What works

- **Baseline residual TD3.** Square 0.52 → 0.90 (`870ws2c2`), Can 0.92 → 1.00 (`msfkjwab`). Both
  300k steps, both single-seed.
- **BC base policies.** Cached on both machines for Can and Square; no retraining needed.
- **The equivariant implementation is correct.** 148 tests: declared representations match the
  physics at all 8 group elements, actor equivariant and critic invariant to `< 1e-4`. The gate
  passes on both machines.
- **`preflight.py`** now validates the environment against `z8yoqylh`. It previously validated
  against `a3e3zylp`, a run that collapsed to 0.00.

## What does not work, or is unknown

- **The success is still n=1.** The replication is in flight.
- **Nothing has error bars.** Every completed result in the project is single-seed.
- **The provenance table is only verified at `caf83f3`.** [ARCHITECTURE.md](ARCHITECTURE.md) lists
  which `EquivarianceConfig` fields are logged and ignored, and the no-op tests pin them at HEAD.
  **Which fields are inert at `7dae4925` is not known** — and one prior assumption about that commit
  (that `use_norms`, `use_orth_init` and `use_equivariant_model` did not exist there) turned out to
  be false. The three running runs therefore have an unverified config-caveats field.
- **Why the equivariant agent does not improve episode length.** In the lost comparison the baseline
  learned to finish Can faster (≈150 → ≈105 steps) while the equivariant version stayed flat
  (≈152). The success-rate gap is within single-seed noise; the episode-length divergence is the
  more interesting signal, if it survives three seeds.
- **The scalar ablation is still not usable as a control.** The two stacks differ in six ways beyond
  equivariance. See [STANDARDS.md](STANDARDS.md) rule 3.3 and TODO P4.

## Active question

**Does `z8yoqylh`'s 0.94 replicate across three seeds at `7dae4925`?** Everything else waits on it.

The honest read is the 10k and 20k evaluations — every collapse in the record was unambiguous by
then (0.00, 0.02, 0.18). Use `/analyze-experiment` once the seeds report.

## One thing that should not wait

`z8yoqylh`'s only copies are `boce-WS-01:~/projects/equi-resrl/wandb/run-20260522_083555-z8yoqylh`
and a backup at `~/equi-resrl-preserve/`. It is not in wandb — it was deleted there. `wandb sync` on
that directory would restore it from local data. Two runs have already been lost this way, and
`msfkjwab`, the Can baseline, survives only as `ZXP-S-works:wandb/run-20260226_143630-msfkjwab/`.
**Do not delete either directory.**
