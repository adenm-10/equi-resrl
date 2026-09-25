# equi-resrl — working agreement

Equivariant residual RL: take the residual off-policy RL algorithm from ResFiT and make the
residual policy SO(2)-equivariant, following the SO(2) equivariant RL paper. Papers are in
`docs/papers/` (local only, not in git).

## Read these first

| Doc | What it's for |
|---|---|
| [docs/STATUS.md](docs/STATUS.md) | What is true right now. Read this every session. |
| [docs/TODO.md](docs/TODO.md) | What to do next, in priority order. |
| [docs/PROGRESS.md](docs/PROGRESS.md) | Session-by-session record of what was done and decided. |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Where code lives and how a training step flows. |
| [docs/EQUIVARIANCE.md](docs/EQUIVARIANCE.md) | The symmetry design: group, representations, assumptions. |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | Every run ever done, including the failures. Append-only. |
| [docs/STANDARDS.md](docs/STANDARDS.md) | Coding and architecture rules. |
| [tests/README.md](tests/README.md) | The pre-submission gate: what it checks and what it does not. |

## The three priorities

These are ranked. When they conflict, the higher one wins.

**1. Scientific consistency.** A run's recorded config must fully determine the code that ran.
Every comparison states what varied and what was held fixed. No claim without a run ID and a
seed count behind it. Failures get logged as carefully as successes.

**2. Code simplicity and correctness.** One code path per behavior. A config field either
branches at runtime or does not exist. No commented-out alternatives left in place. Equivariance
claims are backed by tests, not by comments.

**3. Human interpretability.** Someone should be able to reconstruct the reasoning behind any
architectural or experimental decision months later, from the docs alone. Names match the math.
Docs explain *why*; code shows *what*.

Concretely, priority 1 beating priority 2 means: **do not refactor for cleanliness if it breaks
comparability with existing runs.** See the freeze note in STATUS.md.

## Environment

```bash
conda activate residual
export PYTHONPATH="$PWD:$PYTHONPATH"   # required in every new shell
```

Always launch as a module (`python -m resfit.rl_finetuning.scripts.train_residual_td3`), not by
file path. Hydra config names are registered at the bottom of
[residual_td3.py](resfit/rl_finetuning/config/residual_td3.py).

Setup beyond the README: `robomimic` has to be installed without `egl_probe`, which does not build
here. Install it with `--no-deps` and then add its dependencies by hand:

```bash
pip install robomimic --no-deps
pip install h5py psutil tqdm termcolor tensorboard tensorboardX imageio imageio-ffmpeg matplotlib
```

## Session protocol

Three skills in `.claude/skills/` cover this:

| Skill | When |
|---|---|
| `/session-start` | Beginning of a session. Reads the state docs, checks them against the repo, reports where things stand, asks what you want to work on. |
| `/session-end` | End of a session. Writes the session into the docs, takes safe condensing opportunities, asks about committing. |
| `/analyze-experiment` | After runs finish. Comparison table, plain-English read on the results, suggested next steps. |

Doing it by hand instead, the end-of-session protocol is, in the same commit as the code change:

1. Update STATUS.md if what-is-true changed. Rewrite in place, do not append.
2. Prepend a dated entry to PROGRESS.md.
3. Append to EXPERIMENTS.md if a run was launched or finished. Never edit or delete past entries.
4. Update TODO.md.
5. Check doc size: `docs/`, `.claude/` and this file should stay under 150 KB combined.
   Test and tool code does not count against that budget.
   `cat CLAUDE.md docs/*.md .claude/skills/*/SKILL.md .claude/settings.json | wc -c`

## Before launching a training run

```bash
python preflight.py          # three-tier gate; exits non-zero on a blocking failure
./submit.sh <seed> [gpu]     # runs the gate, then launches
```

`submit.sh` is deliberately **rewritten in place** for each new experiment rather than copied — the
durable record is the pre-registered entry in EXPERIMENTS.md plus the resolved config in wandb.
Pre-register before launching (STANDARDS.md rule 5.4): two runs have already been lost to wandb
deletion, and a deleted run takes its config with it.

## Repository size

The tracked repo is small (about 200 files, 1.5 MB) and should stay that way. Everything heavy is
gitignored: `wandb/`, `artifacts/`, `outputs/` (including the run packages in `outputs/runs/`),
`local_runs/`, both buffer caches, `deps/`, and `docs/papers/`. Docs are plain markdown, no images, no CSVs, no checkpoints. Experiment records
cite wandb URLs and numbers as text.

## Writing style for docs

Plain English wherever plain English does the job. Use the technical term when it is the precise
one (representation, irrep, equivariance) and not when it is decoration. These are handoff
documents first, so completeness beats brevity when the two conflict.
