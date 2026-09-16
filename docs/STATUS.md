# Status

**Last updated:** 2026-09-16 · **Last verified commit:** `caf83f3`

This file describes what is true right now. Rewrite it in place; do not append. For history, see
[EXPERIMENTS.md](EXPERIMENTS.md).

## Where the project is

The ResFiT baseline reproduces on both Can and Square. An equivariant TD3 residual agent is
implemented and has trained successfully once, on Can, to 300k steps — **but that run's data was
deleted and cannot be recovered.** Every equivariant run that still exists on this machine either
collapsed to zero success or never left zero.

So the immediate task is not new science. It is recreating the one good result so there is
something solid to build on.

## Freeze in effect

**Do not modify anything under `resfit/rl_finetuning/equi_off_policy/` until the `z8yoqylh` run has
been reproduced.**

Reasoning: the successful run's config and code are only known through commit `caf83f3` plus a
screenshot. Any edit to the equivariant modules makes "did we reproduce it?" unanswerable, because
a failure to reproduce could not be distinguished from a behavior change we introduced. This is
priority 1 (scientific consistency) overriding priority 2 (code simplicity) — there are known
defects in these files that we are deliberately leaving in place for now.

The cleanup is planned, scheduled, and gated. See Phase 3 in [TODO.md](TODO.md).

The pre-launch blocker is cleared: the untracked `rl_utils.py` has been deleted (TODO H1), so the
tree now matches the commit and the reproduction is reproducible from the tag alone.

## What works

- **Baseline residual TD3.** Reproduced on Square (0.52 → 0.90, run `870ws2c2`) and Can
  (0.92 → 1.00, run `msfkjwab`). Both 300k steps.
- **BC base policies.** Trained and cached locally for Can, Square, and TwoArmCoffee. **No BC
  retraining is needed** — see the artifacts table in [ARCHITECTURE.md](ARCHITECTURE.md).
- **The equivariant agent runs end to end.** It instantiates, trains for 300k steps, and does not
  produce NaNs. The escnn plumbing, field types, normalizer, and frame conversion are all wired up.
- **Environment is intact and matches the last run exactly.** conda env `residual`, Python 3.10.19,
  torch 2.9.1+cu128, torchrl 0.9.2, tensordict 0.9.1, escnn 1.0.13 — identical to the
  `requirements.txt` of the 300k run. CUDA sees both GPUs.
- **All inputs for the Can reproduction are on disk.** BC artifact cached and still live in wandb;
  Can offline buffer (`cd03f9df`, 5.0 GB) and Can online warmup buffer (`9ff5ddd4`, 16 GB) both
  present and will be cache-hits; `env_probes/Can.json` present.

## What does not work, or is unknown

- **The successful result is unreproduced.** One run, one seed, data lost.
- **Pre-group-pooling equivariant runs collapse.** 14 runs across 5 commits, one consistent failure
  mode: the residual saturates at `action_scale` and drives success to zero. Whether group pooling
  fully fixes this or merely delayed it in one seed is unknown.
- **Nothing has error bars.** Every result in the project is single-seed.
- ~~No tests exist.~~ **Resolved 2026-09-16.** 148 tests in `tests/`, plus a three-tier
  pre-submission gate (`preflight.py`) wired into `submit.sh`. The equivariant implementation is
  verified correct: declared representations match the physics at all 8 group elements, the actor is
  equivariant and the critic invariant to `< 1e-4`, and every claim in the provenance table is now
  executable fact.
- **The scalar ablation has never been run, and is not currently usable as a control.**
  `use_equivariant_model=False` is implemented but the two stacks differ in six ways beyond
  equivariance — actor depth, dropout, output squashing, return type, critic head nonlinearity
  count, ensemble implementation. So "equivariance is wrong" still cannot be separated from
  "residual RL is misconfigured on this path". Matching the stacks is a prerequisite; see
  [STANDARDS.md](STANDARDS.md) rule 3.3 and TODO P4.

## Three things that shape everything downstream

### 1. Config fields that silently do nothing

At `caf83f3`, these `EquivarianceConfig` fields are accepted, logged to wandb, and ignored:
`num_actor_layers`, `num_critic_layers`, both `dropout` fields, the critic's `use_norms` and
`use_orth_init`, and `initialize` everywhere except the encoder. The critic's `loss` config is
stored and never read, so the distributional critic is unreachable on the equivariant path.

Two consequences:

- **Every past run's wandb config misdescribes the network that trained.** The `a3e3zylp` run was
  launched with `equivariance.num_actor_layers=3` and that override did nothing. Any comparison
  across runs that varied these fields was comparing identical architectures.
- **Reproduction is easier than it looks.** The architecture is pinned in code, so config drift
  cannot have changed it. The space of configs consistent with the lost run is much smaller than
  the config file suggests.

Full table in [ARCHITECTURE.md](ARCHITECTURE.md).

### 2. Group pooling is not what fixed the collapse

Corrected 2026-09-16. `nn.GroupPooling` was **already present** in the critic head at `7dae4925`,
and all three surviving runs at that commit collapsed to exactly 0.00 with three different seeds and
three different configs. Diffing `7dae4925` against `caf83f3` shows the real change: the critic head
went from `Linear → FieldNorm → ReLU → … → GroupPooling → Linear` to plain
`Linear → GroupPooling → Linear`, with every `FieldNorm` and `ReLU` commented out.

`FieldNorm` subtracts each field's projection onto the trivial-representation subspace, which for a
`regular_repr` field is exactly its group-invariant component — the thing the following
`GroupPooling` exists to extract. That is consistent with `dQ_da ≈ 2e-4` alongside a low, stable
`critic_loss`. Hypothesis, not proven, but specific and testable.

**This is good news for the reproduction:** `caf83f3` differs from the three collapsed runs in
precisely the way that would explain a different outcome. Full reasoning in
[EQUIVARIANCE.md](EQUIVARIANCE.md).

### 3. The agentview camera is tilted, so vision equivariance is approximate

The equivariant vision encoder assumes rotating the scene about the vertical axis corresponds to
rotating the agentview image in-plane. The agentview camera is fixed at a 44.9° downward tilt
(`table_arena.xml:47`), so that correspondence is approximate rather than exact.

**Reviewed and accepted 2026-09-16.** Approximate equivariance on the vision branch is considered a
useful prior. The proprioception and action branches are exactly equivariant and carry the parts of
the state where the symmetry is precise. Recorded in [EQUIVARIANCE.md](EQUIVARIANCE.md) assumption
1 so the decision is visible; not an open problem and not scheduled for work.

## wandb is thinner than assumed

Audited 2026-09-16. `robomimic-can-final` contains only 4 runs, all collapsed.

- **`msfkjwab`, the Can baseline, is also deleted** — not just `z8yoqylh`. It survives *only* as the
  local directory `wandb/run-20260226_193630-msfkjwab/`. **Do not delete that directory; it is the
  only copy of the Can baseline (0.92 → 1.00).** TODO R7's plan to re-pull it from the API does not
  work.
- Two 300k runs were found that have no local copy: `czjqzg0b` (2026-05-07) and `qe2by47h`
  (2026-05-16). Both collapsed. `qe2by47h` is the most recent equivariant run of any kind.
- `z8yoqylh` is confirmed gone from every project. It is not hiding elsewhere.

## Ready to launch

`python preflight.py` reports **GO**. Verified: environment matches the reference run, BC policy
cached and resolvable, both Can buffer caches hit, robot-base probe present, 60 GB RAM available
against ~21 GB per run, 166 GB disk free.

- **Commit tagged** `repro-z8yoqylh-base` at `caf83f3`.
- **Pre-registered** in [EXPERIMENTS.md](EXPERIMENTS.md) with six numbered assumptions.
- **Launch with** `./submit.sh <seed> [gpu]` — seeds 1, 2, 3, sequentially.

One step remains: a live `debug=true` smoke run. Its key number is `debug/dQ_da_mean_abs`. The
collapsed runs showed ~2e-4; if `caf83f3` still shows that, the `FieldNorm` hypothesis is wrong and
the 3-day run should not start.

## Active question

Two, in priority order:

1. **Does the `z8yoqylh` result reproduce from HEAD across 3+ seeds?** Everything else waits on
   this.
2. **Why does the equivariant agent not improve episode length?** In the lost comparison, the
   baseline learned to finish Can faster over training (≈150 → ≈105 steps) while the equivariant
   version stayed flat (≈152). The success-rate gap (≈0.92 vs 1.00) is within single-seed noise and
   should not be treated as a finding yet; the episode-length divergence is the more interesting
   signal, if it survives multiple seeds.

## Standards are now written

[STANDARDS.md](STANDARDS.md) has 28 rules derived from the existing code. The ones most likely to
bite during the reproduction: run names must encode the variant (rule 5.2), a comparison is ≥3
seeds (5.3), pre-register before launching (5.4), and always report step-0 alongside best and final
(5.5).

## Known upcoming change

Aden plans to change the wandb workflow. The `analyze-experiment` skill currently reads run data
from local `wandb/run-*/files/` directories, and ARCHITECTURE.md's "reading a past run without
wandb" section documents that layout. Both will need revisiting when the new workflow lands. Noted
in TODO P1.3.

## Note on task choice

Can has a base policy at 0.92 and a ceiling at 1.00 — about 0.08 of headroom. It is a poor task for
detecting whether equivariance helps. Square starts at 0.52 and discriminates much better. Can is
the right task for the reproduction, because that is what the lost run used, but Square is probably
the better place to spend compute afterward.

## Housekeeping done 2026-09-16

- Papers moved to `docs/papers/` and gitignored (18 MB, versus 1.5 MB for the entire tracked repo).
- 24 root-level `run_2026-*` directories moved to `local_runs/` and gitignored.
- `.gitignore` extended: `docs/papers/`, `local_runs/`, `run_20*/`, `log.txt`,
  `.claude/settings.local.json`.
- `note.txt` and `log.txt` removed from tracking; content migrated into these docs. Recoverable
  via `git show HEAD~1:note.txt`.
- `rl_utils.py` deleted — untracked, unimported, and unimportable (`SyntaxError` on line 15). Its
  one original idea, `TrivialLayerNorm`, is preserved in TODO P3.4.
- Four unreachable modules remain, deferred to TODO P3.5.
- Three Claude Code skills added in `.claude/skills/`: `session-start`, `session-end`,
  `analyze-experiment`.
- `docs/PROGRESS.md` created — the chronological session log, distinct from STATUS (now),
  TODO (next), and EXPERIMENTS (runs).
