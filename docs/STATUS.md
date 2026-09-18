# Status

**Last updated:** 2026-09-18 · **Last verified commit:** `f7aa873`, plus uncommitted edits to
EXPERIMENTS.md and TODO.md from this session

This file describes what is true right now. Rewrite it in place; do not append. For history, see
[EXPERIMENTS.md](EXPERIMENTS.md) and [PROGRESS.md](PROGRESS.md).

## Where the project is

**The replication failed, and it failed because it was aimed at the wrong commit.** All three
`equi-repro-7dae4925` seeds sat at 0.00 for every evaluation after step 0 — 71 evaluations, not one
successful episode. The exact-replication seed completed all 300k steps at 0.00.

**`z8yoqylh` did not run at `7dae4925`. It ran `caf83f3`'s code.** The 2026-09-16 retargeting read
the `git.commit` field in the recovered wandb metadata, which records `HEAD` at launch and not the
working tree — and that tree was dirty. The original target was right all along.

**A corrected replication at `caf83f3` is running**, two seeds, launched 2026-09-18. Until it
reports, the project still has exactly one successful equivariant run, and it is still n=1.

## The evidence for `caf83f3`

Full detail in [EXPERIMENTS.md](EXPERIMENTS.md), 2026-09-18 correction. Three independent lines:

1. **Five config fields.** `z8yoqylh` logged `critic.dropout`, `enc_degree_channel=32`, and
   `use_equivariant_model` / `use_norms` / `use_orth_init`. At `7dae4925` the critic field is named
   `drop`, the encoder default is 16, and **the three `use_*` fields do not exist.** At `caf83f3`
   all five match.
2. **The full config composes exactly.** `caf83f3` + `z8yoqylh`'s recorded launch arguments +
   `seed=2114495708` reproduces its logged config on **all 117 fields**. The only differences are
   int-versus-float YAML rendering and `actor_name`, which is derived at runtime.
3. **The timing.** `z8yoqylh` started 2026-05-22 08:35 EDT; `caf83f3` was committed the same
   morning at 10:28 EDT, message *"got equivariant agent working for the first time."*

This is the strongest provenance any run in this project has had. It is also the reason for a new
standing rule: **a recorded commit hash does not determine the code that ran. The resolved config
does.** Where they disagree, the config wins.

## The two machines

| | `ZXP-S-works` | `boce-WS-01` (`10.188.60.163`, user `aden`) |
|---|---|---|
| GPUs | 2 × RTX 3090 | 2 × RTX 4090 |
| RAM | 62 GB | 125 GB |
| Stack | torch 2.9.1, torchrl 0.9.2, escnn 1.0.13 | torch 2.6.0, torchrl 0.7.0, **escnn 1.0.11** |
| Repo | `~/Desktop/projects/equi-resrl` | `~/projects/equi-resrl` |
| Role | **every collapsed equivariant run** | **the one success, `z8yoqylh`** |
| Now | idle | **running the 2-seed `caf83f3` replication** |

Host still correlates with outcome across the whole project history, and no experiment has yet
tested it. `boce-WS-01` matches `z8yoqylh`'s recorded `requirements.txt` field for field.

## Running now

**Nothing.** Seeds 1 and 2 were killed 2026-09-18 07:48 EDT at 209,500 steps, after 20 consecutive
0.00 evaluations each and after the exact-replication arm had already finished at 0.00.

## The driver problem, and what it left behind

An unattended upgrade installed `nvidia-driver-580` 580.178.04 over a loaded 580.173.02 module,
which broke CUDA for any new process (`error 804`). The runs already going survived, because they
held the old module. **Resolved by rebooting** on 2026-09-18 around 07:30 EDT; both 4090s are back
and torch sees them.

It left a deviation worth remembering. The machine is now on kernel 6.8.0-138 and driver 580.178.04,
where `z8yoqylh` ran on 6.8.0-111 and 580.173.02. The python stack is unchanged and still matches
the recovered `requirements.txt` exactly, which is what `preflight.py` checks. But host and stack
are live candidate causes in this project, so this is the one variable the replication cannot hold
fixed — and the first thing to suspect if both seeds collapse.

## The `caf83f3` replication

Pre-registered in [EXPERIMENTS.md](EXPERIMENTS.md) as seed-group `equi-repro-caf83f3`, with the
launch commands, GPU layout and startup checks written out. Verified before launch:

- Worktree at `~/projects/equi-resrl-caf83f3`, clean, `artifacts/` symlinked to the main checkout
- Config verified against `z8yoqylh`'s 117 fields; seeding path verified at
  `train_residual_td3.py:345-352`
- `~/launch_repro.sh` rewritten for `caf83f3` — rewritten in place per convention, so it no longer
  launches `7dae4925`
- Buffer caches hit (`8edf618e` offline, `43637c10` online), BC policy `xhjdl8a7` cached, 127 GB RAM
  and 71 GB disk free

**Two seeds, not three** — 2114495708 on GPU 0 and seed 1 on GPU 1, each alone, so both finish in
~35-47 h instead of the full set taking four days. The trade is that n=2 has no tiebreaker if the
seeds split; the reasoning is recorded in the pre-registration.

## Freeze

The freeze on `resfit/rl_finetuning/equi_off_policy/` has met its stated condition: the replication
reported, and the three runs it protected are dead. **Whether it continues through the `caf83f3`
runs is Aden's call.** The argument for keeping it is unchanged — edits at HEAD cannot mechanically
disturb a pinned worktree, but they make "did this reproduce?" a different question. The argument
against is that R13 and R14 both want code changes.

## What the corrected target changes

| Claim | Now |
|---|---|
| `7dae4925` produced both the collapses and the success | It produced **only collapses** |
| The `FieldNorm` hypothesis is dead | **Reopened** — it was ruled out on a premise that is gone |
| The successful run had robot-base centering disabled | **Probably enabled.** The centering block is live at `caf83f3` (`equi_normalizer.py:425`); it is commented out only at `7dae4925`. So `z8yoqylh` likely rotated about the robot base and the symmetry was *not* silently broken. |
| `use_*` fields exist at `7dae4925` | They do not. That inference was circular. |
| `enc_degree_channel = 32` was guessed right | The guess was right; the three runs nonetheless ran **16** |

Two git tags now point at non-targets: `repro-z8yoqylh-base` at `caf83f3` — which is, by accident,
the correct commit under a misleading name — and `repro-z8yoqylh-actual` at `7dae4925`. Neither is
being moved.

## What works

- **Baseline residual TD3.** Square 0.52 → 0.90 (`870ws2c2`), Can 0.92 → 1.00 (`msfkjwab`). Both
  300k steps, both single-seed.
- **BC base policies.** Cached on both machines for Can and Square; no retraining needed.
- **The equivariant implementation is correct.** 148 tests: declared representations match the
  physics at all 8 group elements, actor equivariant and critic invariant to `< 1e-4`.
- **`preflight.py`** validates the environment against `z8yoqylh`, and caught the driver failure.

## What does not work, or is unknown

- **The success is still n=1**, and one replication attempt has now failed at a different commit.
- **Nothing has error bars.** Every completed result in the project is single-seed.
- **The gate cannot validate the commit it launches.** `preflight.py` and `tests/` do not exist at
  `caf83f3` or `7dae4925`, so the 148 tests run against HEAD while the launch runs older code. True
  of both replication attempts. TODO R14.
- **Why the equivariant agent does not improve episode length.** In the lost comparison the baseline
  learned to finish Can faster (≈150 → ≈105 steps) while the equivariant version stayed flat
  (≈152). Still the more interesting signal, if it survives three seeds.
- **The scalar ablation is still not usable as a control.** See [STANDARDS.md](STANDARDS.md) rule
  3.3 and TODO P4.
- **`m0ylcivk`'s local directory is gone**, despite a do-not-delete note. It was the `caf83f3` run
  killed on 2026-09-16 for not being the reproduction — aimed at the right commit after all. TODO
  R15.

## Active question

**Does `z8yoqylh`'s 0.94 replicate across three seeds at `caf83f3`?** Everything else waits on it.

Note that `torch_deterministic = False` in the recovered config, so even the exact-replication arm
is not bit-reproducible. The `7dae4925` arms' step-0 evaluations came in at 0.84 / 0.82 / 0.84
against `z8yoqylh`'s 0.82 — that spread is the noise floor. Read the 10k and 20k evaluations:
`z8yoqylh` was at 0.80 by its second evaluation and never below 0.80 again, and no run in the record
has ever recovered from 0.00 at 20k. Use `/analyze-experiment` once the seeds report.

## One thing that should not wait

`z8yoqylh`'s only copies are `boce-WS-01:~/projects/equi-resrl/wandb/run-20260522_083555-z8yoqylh`
and a backup at `~/equi-resrl-preserve/`. It is not in wandb — it was deleted there. `wandb sync` on
that directory would restore it from local data. Three runs have now been lost or partly lost this
way (`z8yoqylh`, `msfkjwab`, `m0ylcivk`), and `msfkjwab`, the Can baseline, survives only as
`ZXP-S-works:wandb/run-20260226_143630-msfkjwab/`. **Do not delete either directory.** The three
dead `7dae4925` runs are also still on disk under `~/projects/equi-resrl-7dae4925/wandb/`.
