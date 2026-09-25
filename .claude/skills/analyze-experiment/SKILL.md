---
name: analyze-experiment
description: Analyze the results of training runs — the most recent by default, or specific runs if the user names them. Pulls each run's config and evaluation history, presents a side-by-side table of the key metrics, explains in short plain English what each run was testing and what the numbers mean, calls out the best and worst results and any common failure modes, and then suggests how to move forward given the rest of the project. Use when the user asks about experiment results, how a run went, or wants runs compared.
---

# Analyze experiment results

Goal: turn run logs into a clear read on what happened and what to do next.

## 1. Decide which runs

If the user named runs (wandb ids, dates, "the three Can seeds"), use those. Otherwise take the most
recent substantive runs — a run that never got past ~1000 steps or a second evaluation is a launch
failure, not a result, and should be mentioned in a line rather than analyzed.

```bash
ls -1t wandb/run-* | head -10
```

Include the natural comparison arm even if the user did not ask for it. A seed group is analyzed as
a group; a variant is analyzed against its baseline. Say which comparison you chose and why.

## 2. Pull the data from disk

Everything needed is local, so this works without wandb access. Per run, under
`wandb/run-<date>-<run_id>/files/`:

- `wandb-metadata.json` — the exact command line and the **git commit** the run used.
- `config.yaml` — the fully resolved config.
- `output.log` — per-step training lines and evaluation markers.

Get success rates by counting the `✓`/`✗` markers in each `Evaluating N episodes:` line rather than
trusting `wandb-summary.json`, which holds only the last logged value and can read as 0 for a run
that did fine earlier.

Runs launched with run packages also have `outputs/runs/<project>/<run_id>/`. Check it first:
`manifest.json` lists the uncommitted files as well as the commit, and `wandb_logs/metrics.jsonl`
has every logged metric row. If you query the wandb API instead, `Run.scan_history()` drops the final
row, which is where the last eval lands.

**Two things to get right, or the analysis will mislead:**

- **The step-0 evaluation is the base BC policy alone**, because the residual initializes to exactly
  zero (`actor_last_layer_init_scale = 0.0`). It is the bar each run must clear. A run whose best
  score equals its step-0 score learned nothing — its best moment was before training began. Always
  report step-0 alongside best and final.
- **Check the config-field provenance table** in [ARCHITECTURE.md](../../../docs/ARCHITECTURE.md)
  for each run's commit. Several config fields are logged to wandb and ignored by the code, so an
  override in the command line may have had no effect. If two runs differ only in a dead field, they
  ran the same network and must not be presented as a comparison.

## 3. Build the table

One row per run, metrics as columns. Include at least:

| Run | Commit | Task | Variant | Seed | Steps | step-0 | Best | Final | Verdict |
|---|---|---|---|---|---|---|---|---|---|

Add whatever else is relevant to what was being tested — episode length, `critic_loss`,
`residual_l1` / `residual_l2`, `actor_grad_norm`, `dQ_da`, wall-clock time. Mark anything you could
not recover as unknown rather than guessing.

For a seed group, also give mean and spread, and state plainly whether the differences are larger
than the spread. A gap smaller than the seed noise is not a finding.

## 4. Explain, in plain English

Keep each of these short — a few sentences, not paragraphs.

- **What each run was testing.** The hypothesis, and what varied against what was held fixed. Take
  it from EXPERIMENTS.md if the run was pre-registered; otherwise infer it from the config diff and
  say that you inferred it.
- **What the numbers mean.** Not a restatement of the table — the interpretation.
- **Best and worst**, and what separates them.
- **Common failure modes.** Look for the patterns this project has hit before: residual saturating
  at `action_scale` (check whether `residual_l2` sits at the action scale), a near-flat critic
  gradient (`dQ_da` around 1e-4), collapse to zero success after a promising start, escnn field-type
  errors, and manual interrupts. Name the pattern and cite the number that shows it.

Avoid jargon where an ordinary word works. The reader wants to know what happened, not to be shown
the terminology.

## 5. Say what to do next

Ground this in the whole project, not just these runs — read [STATUS.md](../../../docs/STATUS.md)
and [TODO.md](../../../docs/TODO.md) first so the suggestion fits the plan rather than restarting it.

Cover: does this change the active question in STATUS.md? Does it confirm or undercut a hypothesis
already on record? What is the cheapest next experiment that would distinguish the remaining
explanations? Is anything now worth *not* doing?

Be honest about ambiguity. Single-seed results, runs that differ in more than one variable, and
metrics recovered from a partial log all warrant a stated caveat rather than a confident reading.

## 6. Offer to log it

Ask whether to append these runs to [EXPERIMENTS.md](../../../docs/EXPERIMENTS.md) using the schema
at the top of that file. Do not write the entries without being asked — but do offer, because an
analyzed run that never gets logged is the failure mode this doc set exists to prevent.
