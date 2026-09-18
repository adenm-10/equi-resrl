# TODO

Priority order. Each item names the doc section it should update when finished.

Tags: `[SC]` scientific consistency · `[CS]` code simplicity and correctness ·
`[HI]` human interpretability. See the priority ranking in [CLAUDE.md](../CLAUDE.md).

---

## H — Housekeeping

### H1, H3 — RESOLVED 2026-09-16 `[CS]` `[HI]`

`rl_utils.py` deleted: untracked, unimported, and a hard `SyntaxError` on line 15, so it had never
once loaded. Its only original content, `TrivialLayerNorm`, is preserved verbatim in P3.4 below.
`note.txt` and `log.txt` removed from tracking, content migrated into these docs and recoverable via
`git show 8442ba2^:note.txt`. Four unreachable modules remain, deferred to P3.5.

### H2. Review two git stashes `[HI]`

`stash@{0}` on `aabde0f`, titled *"transferring to new compute"*. `stash@{1}` on `d9909ce`. Contents
still unreviewed. That first title is more interesting than it looked: work has moved between the
two workstations before, and that is exactly how `z8yoqylh` came to be invisible from
`ZXP-S-works`. Worth five minutes of `git stash show -p`.
**Updates:** STATUS.md.

### H4. Restore `z8yoqylh` to wandb `[SC]`

It exists only as two local directories on `boce-WS-01` — see the last section of STATUS.md.
`wandb sync ~/projects/equi-resrl/wandb/run-20260522_083555-z8yoqylh` would put it back in the cloud
from local data. Aden's call, because it publishes; recommended, because this project has already
lost two runs to wandb deletion and this is the one that matters.
**Updates:** STATUS.md, EXPERIMENTS.md.

### H5. The 150 KB doc budget no longer fits, and cannot `[HI]`

Needs a decision from Aden. The budget in CLAUDE.md is 150 KB for `CLAUDE.md` + `docs/*.md` +
skills + `settings.json`. It is currently **about 152.6 KB, roughly 1.7% over** — this entry
included — after cutting TODO.md by 5.4 KB, rewriting STATUS.md 3.2 KB shorter, and condensing four
superseded PROGRESS sections down to pointers in this session.

The structural problem: **EXPERIMENTS.md is 40.4 KB, 27% of the budget, and is append-only by
project rule** — and rule 5.4 requires a pre-registration for every run, plus a correcting entry
whenever a past claim turns out wrong. It can only grow, so the ceiling will be breached further
every session that launches anything. This session alone added ~10.6 KB there, all of it mandated.

Three options:

1. **Exclude EXPERIMENTS.md from the budget** and apply the ceiling to the docs that are rewritable.
   Most consistent with the append-only rule. Recommended.
2. **Raise the ceiling** to 200 KB and revisit later. Simple, delays the question.
3. **Keep cutting the rewritable docs.** Not recommended — the remaining content is load-bearing,
   and the next cuts would trade correctness for size, which the session-end skill forbids.

**Updates:** CLAUDE.md.

---

## R — Replicate the recovered run

Current top priority. **The `7dae4925` replication ran and failed on all three seeds, because it was
aimed at the wrong commit.** `z8yoqylh` ran `caf83f3`'s code with `HEAD` reading `7dae4925`; the
proof is a 117-field config match plus five schema fields that only exist at `caf83f3`. Both the
results and the correction are in [EXPERIMENTS.md](EXPERIMENTS.md), 2026-09-18 entries.

### R1–R6 — DONE 2026-09-16, then INVALIDATED 2026-09-18 `[SC]`

The recovery and the tooling work stand. The retargeting does not: R2 corrected the target from
`caf83f3` to `7dae4925`, which was backwards, and the tag `repro-z8yoqylh-actual` at `7dae4925` is
wrong for the same reason. Two tags now point at non-targets; neither is being moved, on the same
grounds as last time.

### R7 — DONE 2026-09-18 `[SC]`

All three `7dae4925` seeds sat at 0.00 for every evaluation after step 0. Seed 2114495708 completed
300k at 0.00 in 34.6 h; seeds 1 and 2 were killed at 209,500 steps after 20 consecutive 0.00
evaluations. Numbers in EXPERIMENTS.md.

### R8 — RESOLVED 2026-09-18, against the runs `[SC]`

The config caveats are settled: the three runs' config differs from `z8yoqylh`'s in five fields
(`critic.drop` vs `dropout`, `enc_degree_channel` 16 vs 32, and the three `use_*` flags absent).
`use_norms`, `use_orth_init` and `use_equivariant_model` genuinely **do not exist** at `7dae4925` —
the prior claim that they did was circular, assuming the commit in order to conclude the fields
existed.

### R12. Launch the `caf83f3` replication `[SC]`

**Next action, staged and blocked only on a reboot.** Pre-registered in EXPERIMENTS.md as seed-group
`equi-repro-caf83f3`, with the exact commands, the GPU layout and the startup checks to verify.

Ready: worktree at `~/projects/equi-resrl-caf83f3` with `artifacts/` symlinked, config verified to
compose to `z8yoqylh`'s 117 logged fields, seeding path verified, `~/launch_repro.sh` rewritten for
`caf83f3`. Preflight passes every check except CUDA.

**Unblocked** by the 2026-09-18 reboot. Two seeds, one per GPU. The reboot moved the machine to
kernel 6.8.0-138 and driver 580.178.04, away from `z8yoqylh`'s 6.8.0-111 / 580.173.02 — recorded in
the pre-registration as the one variable this replication cannot hold fixed.
**Updates:** EXPERIMENTS.md, STATUS.md.

### R13. Re-examine the four conclusions that rested on `7dae4925` `[SC]` `[HI]`

The correction entry lists them. Two need real work rather than a doc edit:

- **The `FieldNorm` hypothesis is reopened.** It was ruled out on the premise that `FieldNorm` sat in
  the successful run's critic head. That premise is gone.
- **Robot-base centering was probably enabled**, not disabled. At `caf83f3` the centering block is
  live (`equi_normalizer.py:425`); the "disabled" finding came from reading `7dae4925`, where every
  call site is commented out. So `z8yoqylh` likely rotated about the robot base and the symmetry was
  not silently broken — which removes a standing worry rather than adding one.

**Updates:** EQUIVARIANCE.md, STATUS.md, ARCHITECTURE.md.

### R14. The gate cannot validate the commit it launches `[CS]`

`preflight.py` and `tests/` do not exist at `caf83f3` or `7dae4925`, so the 148 tests and the
equivariance checks run against HEAD while the launch runs older code. True of both replication
attempts. Worth fixing before the next one, or at minimum stating in every entry.
**Updates:** tests/README.md, STANDARDS.md.

### R15. `m0ylcivk`'s local directory is gone `[SC]`

It was the `caf83f3` run killed on 2026-09-16 for "not being the reproduction" — aimed at the right
commit after all. EXPERIMENTS.md:349 says do not delete that directory; it is not on disk. Check
whether it survives in wandb, and if so whether its early evaluations agree with `z8yoqylh`.
**Updates:** EXPERIMENTS.md.

### R9. Baseline arm — the local copy is the only copy `[SC]`

`msfkjwab` is deleted from wandb and survives only as
`ZXP-S-works:wandb/run-20260226_143630-msfkjwab/`. EXPERIMENTS.md already holds its numbers
recomputed from that log, so it cannot be re-pulled. **Recommend: compare against the recorded
single-seed baseline, state the 3-vs-1 asymmetry plainly, and spend compute on Square instead**,
where the base policy sits at 0.52 and the measurement actually discriminates.

### R10. Analyze against the pre-registration, then lift the freeze `[SC]` `[HI]`

Use `/analyze-experiment`. Success rate and episode length, mean and spread across the three seeds.
The question the lost comparison raised: is the flat equivariant episode length real, or was it one
seed?
**Updates:** STATUS.md, EXPERIMENTS.md, PROGRESS.md.

### R11. Isolate the two candidate causes `[SC]`

Deferred until R10 confirms the result. Then two runs at ~35-47 h each, **at `caf83f3`** — the
commit references here were `7dae4925` until the 2026-09-18 correction:

- `caf83f3` + `actor_last_layer_init_scale=0.0` on `boce-WS-01`, same seed — isolates the override.
- `caf83f3` + `actor_last_layer_init_scale=1e-4` on `ZXP-S-works` — isolates the software stack.

The second is the more interesting one. Host correlates perfectly with outcome across the entire
project history — every collapsed equivariant run on `ZXP-S-works`, the one success on
`boce-WS-01` — and no experiment has ever tested it.

---

## P1 — Standards — DONE 2026-09-16

28 rules in [STANDARDS.md](STANDARDS.md), three skills in `.claude/skills/`. The significant finding
is rule 3.3: the scalar ablation is not a controlled ablation. See P4.

Remaining follow-up: revisit `analyze-experiment` when the wandb workflow changes. It reads run data
from local `wandb/run-*/files/`, and now has to cope with **two machines** — the runs it needs to
compare live on `boce-WS-01`, not here.

---

## P2 — Equivariance tests — MOSTLY DONE 2026-09-16

`preflight.py` (3 tiers) plus `tests/` (7 files, 148 tests), passing on both machines. Full detail in
[tests/README.md](../tests/README.md). **Result: the equivariant implementation is correct** —
declared representations match the physics at all 8 group elements, the actor is equivariant and the
critic invariant to `< 1e-4`. So the collapses were not a broken symmetry.

**Still to do (P2.4, does not block the replication)**

- Full `QAgent.act` plus `_combine_actions`: assert the weaker property that holds when a
  non-equivariant base policy is summed with an equivariant residual.
- Negative control: `use_equivariant_model=False` must **break** equivariance — a test proving the
  tests can fail.
- Normalizer: assert isotropic scale and zero offset on every `irrep(1)` field in
  `build_equivariant_normalizer`. Currently guaranteed only by reading the code.
- **New:** the suite runs against HEAD only. The runs that matter are at `7dae4925`, and that tree
  differs in at least robot-base centering, so decide whether the suite should be runnable against a
  worktree. Relates to R8.

---

## P3 — Cleanup, gated on R10

Do not start before the reproduction completes. See the freeze in STATUS.md.

### P3.1 Wire or remove every dead config field `[CS]` `[SC]`

Make `use_norms`, `use_orth_init`, `num_actor_layers`, `num_critic_layers`, `dropout`, and
`initialize` either actually branch or not exist. Use P2.3's bit-identity tests to prove the
cleanup did not move behavior. Only after this do these become real, ablatable knobs.

### P3.2 Resolve the reachable-code rough edges `[CS]`

From ARCHITECTURE.md "Known rough edges": the `std=0` versus `std=None` divergence in
`Actor.forward`, the ignored `return_logits`, the stored-and-unread `loss_cfg`, the commented-out
`assert False` debug blocks.

### P3.3 Decide about the distributional critic `[CS]`

`agent.critic.loss` (C51 / HL-Gauss) is settable and unreachable on the equivariant path. Either
implement it or make the config reject it.

---

### P3.4 Restore normalization to the critic head, using `TrivialLayerNorm` `[CS]`

**Scope note added 2026-09-16:** this describes the critic head **at HEAD / `caf83f3`**, where every
normalization layer is commented out and `use_norms` does nothing. At `7dae4925`, where the only
successful run lives, the head still had `FieldNorm` and the ReLUs active. So "the critic trains
with no normalization" is a property of HEAD, not of the run that reached 0.94.

At HEAD, then: the critic trains with no normalization at all while the actor has active batch norm.
That asymmetry is a plausible contributor to the instability seen as `dQ_da ≈ 2e-4` and residual
saturation — though note that the run which worked had `FieldNorm` present, so the previous
reasoning that `FieldNorm` was itself the problem no longer holds. See
[EQUIVARIANCE.md](EQUIVARIANCE.md), "What actually changed".

The obvious fix has a trap. escnn's `FieldNorm` cannot be used after `GroupPooling`, because its own
docstring warns: *"If a field is only containing trivial irreps, this layer will just set its values
to zero."* Post-pooling features are exactly all-trivial, so `FieldNorm` there would zero the Q
values.

Aden had already written the correct workaround, in the now-deleted `rl_utils.py`. It never ran (the
file had a `SyntaxError`), and it was never confirmed whether the critic head was its intended
destination — but the docstring says so explicitly. Preserved here verbatim:

```python
class TrivialLayerNorm(torch.nn.Module):
    """LayerNorm for GeometricTensors with all-trivial field type.

    Trivial reps act as identity under the group, so a plain LayerNorm
    is equivariant on these features. Use this AFTER GroupPooling, where
    FieldNorm would zero the signal (mean == feature for size-1 fields).

    Operates on flat [B, C] GeometricTensors (post-pooling, pre-final-Linear).
    """

    def __init__(self, in_type: nn.FieldType):
        super().__init__()
        for rep in in_type.representations:
            assert rep.size == 1, (
                f"TrivialLayerNorm requires all-trivial reps; "
                f"got a rep of size {rep.size}"
            )
        self.in_type = in_type
        self.out_type = in_type
        self.norm = torch.nn.LayerNorm(in_type.size)

    def forward(self, x: nn.GeometricTensor) -> nn.GeometricTensor:
        assert x.type == self.in_type, (
            f"TrivialLayerNorm input type mismatch: "
            f"expected {self.in_type}, got {x.type}"
        )
        return nn.GeometricTensor(self.norm(x.tensor), self.out_type)
```

Belongs in `equi_rl_utils.py` alongside `FieldNorm`. Gated on R10 — it changes critic behavior, so it
cannot land before the reproduction. Any version of this must be covered by the P2.3 critic
invariance test, since normalization is exactly where equivariance tends to break.

### P3.5 Sweep the four remaining unreachable modules `[CS]`

`ablate_equi_obs_encoder.py`, `field_norm.py`, `rotation_transformer.py`, `rotation_utils.py`. All
tracked, so deletion is reversible. Check `field_norm.py` against `equi_rl_utils.py`'s `FieldNorm`
before removing — if the standalone copy is the better one, keep that instead. See H1.

---

## P4 — Science, after reproduction

Not yet planned in detail. Recorded so it is not forgotten.

- ~~Test the top-down camera hypothesis.~~ **Closed 2026-09-16** — reviewed and accepted as an
  approximation. The cheap diagnostic (render before/after a 45° rotation, compare against a 45°
  image rotation) stays documented in EQUIVARIANCE.md assumption 1 should it ever be needed, but is
  not scheduled.
- **Match the two stacks, then run the scalar ablation.** `use_equivariant_model=False` is meant to
  be the control separating "equivariance is wrong" from "residual RL is misconfigured on this
  path". It cannot serve that role yet: the scalar and equivariant modules differ in six ways at
  once — actor depth, actor dropout, output squashing (`Tanh` vs none), actor return type
  (distribution vs tensor), critic head nonlinearity count (2 ReLUs vs 0), and ensemble
  implementation (`vmap` vs loop). Full table in [STANDARDS.md](STANDARDS.md) rule 3.3.
  **Blocked on matching depth, nonlinearity count, squashing, and clipping semantics**, not on
  compute. Until then the ablation yields an uninterpretable result.
- **Diagnose the residual saturation.** `residual_l2 ≈ action_scale` with `dQ_da ≈ 2e-4` across
  multiple runs. Is the critic providing any usable action gradient?
- **Expand to more tasks.** Square next — its base policy sits at 0.52, so it discriminates far
  better than Can's 0.92. Then the two-arm dexmg tasks, which need BC base policies trained
  (`wandb_id` is still `TODO` for TwoArmBoxCleanup and TwoArmCanSortRandom).
- **Equivariant SAC.** No SAC exists anywhere in the repo, so this is a genuine build: equivariant
  squashed-Gaussian actor (equivariant mean, invariant log-std), entropy temperature, twin critics.
  The SO(2) equivariant RL paper's headline result is equivariant SAC, so the architecture maps
  over — but it needs P2 in place to be trustworthy.
