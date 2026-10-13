"""The arms submit.sh launches, exactly as it launches them.

A queued arm that fails at launch loses its GPU slot. So every arm in
submit.sh's case table is checked here first. Three seed groups, and each arm
against its anchor's launch, less the anchor's run length:
  - resfit-equi-diff-1012: oj5k0jqb's (resfit-residual-square 1, 9cf207d);
  - equi-resfit-arch-1013: eaqorqki's (square-residual-start-1011, 0af7655);
  - equi-residual-long-1012: each arm is its anchor's launch, again: 6ng62tpm's
    (square-rerendered-1009, eb90a0c) and eaqorqki's.

  Tier 1  its overrides are the pre-registered ones and set no key twice; it
          composes through train_residual_td3's Hydra configs to its anchor's
          config but for its listed changes; each anchor is what its run
          launched with, and composes today to that run's recorded config but
          for fields added since at their no-op defaults; the re-rendered demos
          leave the action scaler as it was; the dataset each arm loads is
          there; the smoke mode is the arm, briefly; and every arm launches.
  Tier 2  its agent builds on the GPU and runs one TD3 update, with finite
          losses. A ResFiT arm starts from the anchor's weights, bit for bit,
          but for any norm it drops (smallcritic: its encoders); noln has no
          LayerNorm in its actor or critic and keeps its ViTs', noln-all has no
          norm anywhere, and smallcritic has 5 heads of 1 hidden layer. What
          limits' bound and noise's units do is tested in
          tests/test_action_bound.py. An equivariant arm is equivariant (its
          actor) and invariant (its critic) at its own sizes, and every input
          reaches its critic and its actor; the arch arms have ResFiT's layout:
          RegularLayerNorm after every hidden layer of the actor and the
          critic, and 10 heads of 2 hidden layers. A test here keeps on record
          that the fixed-env baselines' FieldNorm critic is blind to every
          trivial input (2026-10-08 FINDING).

The arm list is pinned to the pre-registrations (EXPERIMENTS.md, 2026-10-12
PRE-REGISTERED, "ResFiT's residual on Square, given the equivariant stack's
differences one at a time; the equivariant residual given ResFiT's layout", and
"the equivariant residual's anchors again, to a fixed 150k") and the AMENDMENTs
of that date.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import re
import subprocess

import pytest

from resfit.lerobot.utils.load_policy import RUNS_ROOT

REPO = pathlib.Path(__file__).resolve().parent.parent
SUBMIT = REPO / "submit.sh"
LEROBOT_HOME = pathlib.Path.home() / ".cache/huggingface/lerobot"

RESFIT_GROUP = "resfit-equi-diff-1012"
EQUI_GROUP = "equi-resfit-arch-1013"
LONG_GROUP = "equi-residual-long-1012"

RERENDERED = "resfit-residual-square-rerendered"
NOLN = "resfit-residual-square-noln"
LIMITS = "resfit-residual-square-limits"
NOLN_ALL = "resfit-residual-square-noln-all"
COMBINED = "resfit-residual-square-combined"
SMALLCRITIC = "resfit-residual-square-smallcritic"
NOISE = "resfit-residual-square-noise"
ARCH = "equi-residual-square-arch"
ARCH_WIDE = "equi-residual-square-arch-wide"
LONG = "equi-residual-square-long"
INIT0_LONG = "equi-residual-square-init0-long"
DEMOS = "ankile/robomimic-mh-square-image"  # ResFiT's config's own, the original coded demos
RERENDERED_DEMOS = "local/robomimic-mh-square-rerendered"  # every equivariant run's

# What every arm launches, less the run's name, notes, project and group.
COMMON = {"algo.total_timesteps=150000", "read_buffer_caches=false", "write_buffer_caches=false",
          "seed=1", "debug=false"}
# oj5k0jqb's launch at 9cf207d, the anchor every ResFiT arm is one change from.
RESFIT_BASE = {"--config-name=residual_td3_square_config", "base_policy.wt_version=v9", *COMMON}
RESFIT_ANCHOR = RESFIT_BASE | {"agent.critic.num_q=10"}
# eaqorqki's launch at 0af7655 and 6ng62tpm's at eb90a0c, each but for its run length, 300k: every
# schedule runs on fixed step counts, so through 150k each is the same run.
EQUI_BASE = {"--config-name=residual_equi_td3_square_config", "equivariance.residual_squash=true",
             "base_policy.wt_version=v9", f"offline_data.name={RERENDERED_DEMOS}", *COMMON}
EQUI_ANCHOR = EQUI_BASE | {"agent.critic.num_q=5", "agent.actor.actor_last_layer_init_scale=0.0"}
BASELINE_ANCHOR = EQUI_BASE | {"agent.critic.num_q=5", "agent.actor.actor_last_layer_init_scale=1e-4"}
NOLN_OVERRIDES = {"agent.actor.use_layer_norm=false", "agent.critic.use_layer_norm=false"}
# ResFiT's layout in the equivariant residual, as amended (Aden, 2026-10-12): the actor keeps its depth.
RESFIT_LAYOUT = {"equivariance.actor_norm=layer_norm", "equivariance.critic_norm=layer_norm",
                 "agent.critic.num_q=10", "equivariance.num_critic_layers=2"}
LAUNCHED = {
    RERENDERED: RESFIT_ANCHOR | {f"offline_data.name={RERENDERED_DEMOS}"},
    NOLN: RESFIT_ANCHOR | NOLN_OVERRIDES,
    LIMITS: RESFIT_ANCHOR | {"agent.actor.action_bound=equivariant"},
    NOLN_ALL: RESFIT_ANCHOR | NOLN_OVERRIDES | {"agent.vit.use_layer_norm=false"},
    COMBINED: RESFIT_ANCHOR | {f"offline_data.name={RERENDERED_DEMOS}"} | NOLN_OVERRIDES
              | {"agent.actor.action_bound=equivariant"},
    SMALLCRITIC: RESFIT_BASE | {"agent.critic.num_q=5", "agent.critic.num_layers=1"},
    NOISE: RESFIT_ANCHOR | {"agent.actor.noise_units=env"},
    ARCH: (EQUI_ANCHOR - {"agent.critic.num_q=5"}) | RESFIT_LAYOUT,
    ARCH_WIDE: (EQUI_ANCHOR - {"agent.critic.num_q=5"}) | RESFIT_LAYOUT | {"equivariance.actor_degree_channel=448"},
    LONG: BASELINE_ANCHOR,
    INIT0_LONG: EQUI_ANCHOR,
}
# Each arm's changes from its anchor, as composed paths, and its values there.
NOLN_PATHS = {"agent.actor.use_layer_norm", "agent.critic.use_layer_norm"}
LAYOUT_PATHS = {"equivariance.actor_norm", "equivariance.critic_norm", "agent.critic.num_q",
                "equivariance.num_critic_layers"}
FROM_ANCHOR = {
    RERENDERED: {"offline_data.name"},
    NOLN: NOLN_PATHS,
    LIMITS: {"agent.actor.action_bound"},
    NOLN_ALL: NOLN_PATHS | {"agent.vit.use_layer_norm"},
    COMBINED: {"offline_data.name", "agent.actor.action_bound"} | NOLN_PATHS,
    SMALLCRITIC: {"agent.critic.num_q", "agent.critic.num_layers"},
    NOISE: {"agent.actor.noise_units"},
    ARCH: LAYOUT_PATHS,
    ARCH_WIDE: LAYOUT_PATHS | {"equivariance.actor_degree_channel"},
    LONG: set(),
    INIT0_LONG: set(),
}
NOLN_VALUES = {"agent.actor.use_layer_norm": False, "agent.critic.use_layer_norm": False}
LAYOUT_VALUES = {"equivariance.actor_norm": "layer_norm", "equivariance.critic_norm": "layer_norm",
                 "agent.critic.num_q": 10, "equivariance.num_critic_layers": 2}
ARM_VALUES = {
    RERENDERED: {"offline_data.name": RERENDERED_DEMOS},
    NOLN: NOLN_VALUES,
    LIMITS: {"agent.actor.action_bound": "equivariant"},
    NOLN_ALL: {**NOLN_VALUES, "agent.vit.use_layer_norm": False},
    COMBINED: {"offline_data.name": RERENDERED_DEMOS, **NOLN_VALUES, "agent.actor.action_bound": "equivariant"},
    SMALLCRITIC: {"agent.critic.num_q": 5, "agent.critic.num_layers": 1},
    NOISE: {"agent.actor.noise_units": "env"},
    ARCH: LAYOUT_VALUES,
    ARCH_WIDE: {**LAYOUT_VALUES, "equivariance.actor_degree_channel": 448},
    LONG: {},
    INIT0_LONG: {},
}
# The demos each arm loads.
ARM_DEMOS = {RERENDERED: RERENDERED_DEMOS, NOLN: DEMOS, LIMITS: DEMOS, NOLN_ALL: DEMOS,
             COMBINED: RERENDERED_DEMOS, SMALLCRITIC: DEMOS, NOISE: DEMOS, ARCH: RERENDERED_DEMOS,
             ARCH_WIDE: RERENDERED_DEMOS, LONG: RERENDERED_DEMOS, INIT0_LONG: RERENDERED_DEMOS}
# The fixed-env baselines' equi-tuned-rlpd-can (eyyi99q4) as 9cf207d launched it, less naming.
BASELINES = {
    "equi-tuned-rlpd-can": ["--config-name=equi_tuned_rlpd_can_config", "equivariance.residual_squash=true",
                            "agent.critic.num_q=10", "equivariance.num_critic_layers=2",
                            "equivariance.critic_norm=field_norm", "equivariance.actor_norm=none",
                            "algo.total_timesteps=150000", "read_buffer_caches=false",
                            "write_buffer_caches=false", "seed=1", "debug=false"],
}
NAMING = ("wandb.name", "wandb.notes", "wandb.project", "wandb.group")
WANDB = {RESFIT_GROUP: ("robomimic-square-final", "residual-rl"),
         EQUI_GROUP: ("robomimic-square-final", "equi-residual-rl"),
         LONG_GROUP: ("robomimic-square-final", "equi-residual-rl")}

# The composed config every arm must have: its anchor's, as its recorded config holds it.
TRAINING = {
    "task": "Square", "algo.total_timesteps": 150_000, "read_buffer_caches": False,
    "write_buffer_caches": False, "algo.n_step": 3, "algo.gamma": 0.99, "algo.batch_size": 256,
    "algo.offline_fraction": 0.5, "algo.num_updates_per_iteration": 4,
    "algo.actor_updates_per_iteration": 1, "algo.buffer_size": 200_000, "algo.learning_starts": 10_000,
    "algo.critic_warmup_steps": 10_000, "num_envs": 1, "eval_first": True,
    "eval_interval_every_steps": 10_000, "eval_num_episodes": 50, "save_video": True, "seed": 1,
    "debug": False, "wandb.mode": "online", "encode_env_frames_like_demos": False,
    "base_policy.wt_version": "v9", "offline_data.use_base_policy_for_base_actions": True,
    "agent.actor_lr": 1e-6, "agent.critic_lr": 1e-4, "agent.critic_target_tau": 0.005,
    "algo.stddev_min": 0.05, "algo.stddev_max": 0.05, "agent.stddev_clip": 0.3,
    "agent.actor.action_scale": 0.1, "algo.random_action_noise_scale": 0.2,
    "algo.use_base_policy_for_warmup": True, "algo.actor_lr_warmup_steps": 0, "agent.bc_loss_coef": 0.0,
    "agent.critic.min_q_heads": 2, "agent.critic.policy_gradient_type": "ensemble_mean",
    "agent.critic_grad_clip_norm": 1.0, "agent.actor.action_bound": "box",
    "agent.actor.noise_units": "scaled", "agent.vit.use_layer_norm": True,
}
EQUI_TRAINING = {
    **TRAINING, "offline_data.name": RERENDERED_DEMOS, "agent.critic.num_q": 5, "equivariance.N": 8,
    "equivariance.use_equivariant_model": True, "equivariance.residual_squash": True,
    "equivariance.encoder_norm": "none", "equivariance.actor_norm": "none",
    "equivariance.critic_norm": "none", "equivariance.critic_pooled_norm": "none",
    "equivariance.num_critic_layers": 1, "equivariance.critic_last_layer_init_scale": None,
    "equivariance.frozen_target_critic": False, "equivariance.float_images_as_uint8": False,
    "equivariance.actor_degree_channel": 128, "equivariance.critic_degree_channel": 128,
    "equivariance.enc_degree_channel": 32, "equivariance.use_visual_proj": True,
}
# Each anchor: its launch (at 150k), its run package, the run length it launched with, the composed
# config its arms share, and the fields added since its commit, at the defaults that leave its run
# as it was ("box", "scaled": ResFiT's own, as tests/test_action_bound.py's goldens pin them; the
# equivariant agent reads no ResFiT field).
ANCHORS = {
    "oj5k0jqb": {
        "launch": RESFIT_ANCHOR, "steps": 150_000,
        "shared": {**TRAINING, "offline_data.name": DEMOS, "equivariance": None,
                   "offline_data.min_state_std": 0.1, "agent.critic.num_q": 10,
                   "agent.actor.actor_last_layer_init_scale": 0.0, "agent.critic.num_layers": 2,
                   "agent.actor.use_layer_norm": True, "agent.critic.use_layer_norm": True},
        "added": {"encode_env_frames_like_demos": False, "agent.actor.action_bound": "box",
                  "agent.actor.noise_units": "scaled", "agent.vit.use_layer_norm": True},
    },
    "eaqorqki": {
        "launch": EQUI_ANCHOR, "steps": 300_000,
        "shared": {**EQUI_TRAINING, "agent.actor.actor_last_layer_init_scale": 0.0},
        "added": {"agent.actor.action_bound": "box", "agent.actor.noise_units": "scaled",
                  "agent.vit.use_layer_norm": True},
    },
    "6ng62tpm": {
        "launch": BASELINE_ANCHOR, "steps": 300_000,
        "shared": {**EQUI_TRAINING, "agent.actor.actor_last_layer_init_scale": 1e-4},
        "added": {"agent.actor.action_bound": "box", "agent.actor.noise_units": "scaled",
                  "agent.vit.use_layer_norm": True, "equivariance.critic_pooled_norm": "none"},
    },
}
# Keys a run writes into its own recorded config, which no composition holds.
RUNTIME_STAMPED = {"actor_name"}


def case_table_arms(script: str) -> list[str]:
    """The arm names in a submit.sh's case table, in order."""
    table = script[script.index('case "$ARM" in'):script.index("\nesac")]
    return re.findall(r"^ {4}([a-z0-9][a-z0-9-]*)\)", table, flags=re.M)


ARMS = case_table_arms(SUBMIT.read_text())
EQUI_ARMS = [arm for arm in ARMS if arm.startswith("equi-")]
ARCH_ARMS = [arm for arm in EQUI_ARMS if arm in (ARCH, ARCH_WIDE)]
RESFIT_ARMS = [arm for arm in ARMS if arm.startswith("resfit-")]


def group_of(arm: str) -> str:
    if arm in RESFIT_ARMS:
        return RESFIT_GROUP
    return EQUI_GROUP if arm in ARCH_ARMS else LONG_GROUP


def anchor_of(arm: str) -> str:
    if arm in RESFIT_ARMS:
        return "oj5k0jqb"
    return "6ng62tpm" if arm == LONG else "eaqorqki"


def _launch_order(overrides) -> list[str]:
    return sorted(overrides, key=lambda o: not o.startswith("--config-name"))


def _overrides(arm: str, tmp_path: pathlib.Path, **env_extra) -> list[str]:
    """What submit.sh would launch `arm` with, read from its dry run.

    PYTHON=true passes its interpreter check: the dry run exits before using it.
    """
    (tmp_path / "offline_buffer_cache").mkdir(exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k != "SMOKE"}
    env.update({"DRY_RUN": "1", "SKIP_PREFLIGHT": "1", "PYTHON": "true", "CACHE_DIR": str(tmp_path),
                **env_extra})
    out = subprocess.run([str(SUBMIT), arm, "1"], env=env, capture_output=True,
                         text=True, check=True).stdout
    return [line.strip() for line in out.splitlines() if line.startswith("    ")]


def _compose(overrides: list[str]):
    from hydra import compose, initialize

    import resfit.rl_finetuning.config.residual_td3  # noqa: F401  registers the configs

    flag, config_name = overrides[0].split("=", 1)
    assert flag == "--config-name"
    with initialize(version_base=None, config_path=None):
        return compose(config_name=config_name, overrides=overrides[1:])


def _flat(cfg) -> dict[str, object]:
    """A composed config as dotted path -> value."""
    from omegaconf import OmegaConf

    out = {}

    def walk(prefix, node):
        if isinstance(node, dict):
            for key, value in node.items():
                walk(f"{prefix}.{key}" if prefix else key, value)
        else:
            out[prefix] = node

    walk("", OmegaConf.to_container(cfg, resolve=True))
    return out


def test_case_table_is_the_preregistered_arms():
    assert len(ARMS) == len(set(ARMS)), "an arm is listed twice"
    assert set(ARMS) == set(LAUNCHED)


@pytest.mark.parametrize("arm", ARMS)
def test_arm_launches_its_preregistered_overrides(arm, tmp_path):
    overrides = _overrides(arm, tmp_path)
    keys = [o.split("=", 1)[0] for o in overrides]
    assert len(keys) == len(set(keys)), f"{arm} sets a key twice: {keys}"
    assert {o for o in overrides if o.split("=", 1)[0] not in NAMING} == LAUNCHED[arm]


@pytest.mark.parametrize("arm", ARMS)
def test_arm_composes_to_its_preregistered_recipe(arm, tmp_path):
    flat = _flat(_compose(_overrides(arm, tmp_path)))
    project, group = WANDB[group_of(arm)]
    expected = {**ANCHORS[anchor_of(arm)]["shared"], **ARM_VALUES[arm], "wandb.project": project,
                "wandb.group": group, "wandb.name": f"{arm}-1", "wandb.notes": f"{group_of(arm)}/{arm}/seed1"}
    for path, value in expected.items():
        assert flat.get(path, "absent") == value, f"{arm}: {path} = {flat.get(path, 'absent')}"
    assert "," not in flat["wandb.notes"], "hydra splits on commas"


def _differing(a: dict, b: dict) -> set[str]:
    return {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)} - set(NAMING)


@pytest.mark.parametrize("arm", ARMS)
def test_arm_is_the_anchor_but_for_its_change(arm, tmp_path):
    ours = _flat(_compose(_overrides(arm, tmp_path)))
    anchor = _flat(_compose(_launch_order(ANCHORS[anchor_of(arm)]["launch"])))
    assert _differing(ours, anchor) == FROM_ANCHOR[arm]


def _package(name: str) -> pathlib.Path:
    # Under CACHE_DIR, where every run package lives: a launch worktree has no outputs/.
    return RUNS_ROOT / "robomimic-square-final" / name


def _anchor_as_launched(name: str) -> set[str]:
    """The anchor run's own launch: eaqorqki and 6ng62tpm ran to 300k."""
    anchor = ANCHORS[name]
    return (anchor["launch"] - {"algo.total_timesteps=150000"}) | {f"algo.total_timesteps={anchor['steps']}"}


@pytest.mark.parametrize("name", list(ANCHORS))
def test_each_anchor_is_what_its_run_launched_with(name):
    if not _package(name).exists():
        pytest.skip(f"{name}'s run package is not on this machine")
    argv = json.loads((_package(name) / "manifest.json").read_text())["argv"][1:]
    launched = {o for o in argv if o.split("=", 1)[0] not in (*NAMING, "hydra.run.dir")}
    assert launched == _anchor_as_launched(name)


def _number(value):
    """wandb records some floats as strings ('1e-06'); compare them as the numbers they are."""
    if isinstance(value, str) and re.fullmatch(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?", value):
        return float(value)
    return value


def _flat_recorded(path: pathlib.Path) -> dict[str, object]:
    """A run's recorded config.yaml as dotted path -> value, less wandb's own keys, the versions the
    run stamped in, and its naming."""
    import yaml

    record = yaml.safe_load(path.read_text())
    values = {k: v["value"] for k, v in record.items() if not k.startswith("_") and k != "versions"}
    out = {}

    def walk(prefix, node):
        if isinstance(node, dict):
            for key, value in node.items():
                walk(f"{prefix}.{key}" if prefix else key, value)
        else:
            out[prefix] = _number(node)

    walk("", values)
    return {k: v for k, v in out.items() if not k.startswith("wandb.")}


@pytest.mark.parametrize("name", list(ANCHORS))
def test_each_anchor_composes_to_its_runs_recorded_config(name):
    """So each anchor at this commit is the config its run ran with. A field added since is allowed
    only at the default that leaves every earlier run as it was."""
    if not _package(name).exists():
        pytest.skip(f"{name}'s run package is not on this machine")
    recorded = _flat_recorded(_package(name) / "wandb_logs/config.yaml")
    ours = {k: _number(v) for k, v in _flat(_compose(_launch_order(_anchor_as_launched(name)))).items()
            if not k.startswith("wandb.")}
    added = {k: ours[k] for k in ours.keys() - recorded.keys()}
    assert added == ANCHORS[name]["added"]
    changed = {k for k in ours.keys() & recorded.keys() if ours[k] != recorded[k]}
    assert changed <= RUNTIME_STAMPED, {k: (recorded[k], ours[k]) for k in changed}
    assert recorded.keys() - ours.keys() <= RUNTIME_STAMPED


def test_the_rerendered_demos_leave_the_action_scaler_as_it_was():
    """The re-rendered demos keep every demo's actions, so the action statistics and the ActionScaler
    built from them are the original's bit for bit: the arm's change reaches the frames and the
    state, not the residual's units. The state is observed one physics substep late, as the env
    observes it, so its statistics move a little (at most 4.2e-4 on a min or max)."""
    import torch
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

    from resfit.rl_finetuning.utils.normalization import ActionScaler

    original, rerendered = (LeRobotDatasetMetadata(name).stats for name in (DEMOS, RERENDERED_DEMOS))
    for key in ("min", "max", "mean", "std"):
        assert torch.equal(torch.as_tensor(original["action"][key]), torch.as_tensor(rerendered["action"][key])), key
    old, new = (ActionScaler.from_dataset_stats(action_stats=stats["action"], action_scale=0.1,
                                                min_range_per_dim=0.1).limits
                for stats in (original, rerendered))
    assert torch.equal(old.min, new.min) and torch.equal(old.max, new.max)
    for key in ("min", "max"):
        drift = (torch.as_tensor(original["observation.state"][key]) -
                 torch.as_tensor(rerendered["observation.state"][key])).abs().max().item()
        assert 0 < drift < 5e-4, (key, drift)


@pytest.mark.parametrize("arm", ARMS)
def test_the_dataset_each_arm_loads_is_built(arm):
    """A dataset the trainer cannot find is a launch that dies at startup."""
    root = LEROBOT_HOME / ARM_DEMOS[arm]
    assert (root / "meta/info.json").exists() and (root / "meta/episodes_stats.jsonl").exists(), root
    if ARM_DEMOS[arm] == RERENDERED_DEMOS:
        assert (root / "meta/rerender.json").exists(), root


@pytest.mark.parametrize("arm", ARMS)
def test_smoke_mode_is_the_arm_briefly(arm, tmp_path):
    real = _overrides(arm, tmp_path)
    smoke = _overrides(arm, tmp_path, SMOKE="1")
    keys = [o.split("=", 1)[0] for o in smoke]
    assert len(keys) == len(set(keys)), f"{arm} smoke sets a key twice: {keys}"
    brief = {"algo.total_timesteps=1000", "algo.learning_starts=300", "eval_first=false",
             "eval_interval_every_steps=100000", "wandb.mode=offline",
             "wandb.project=equi-resrl-smoke", "wandb.group=smoke", f"wandb.name=smoke-{arm}-1",
             f"wandb.notes=smoke/{group_of(arm)}/{arm}/seed1"}
    replaced = {o.split("=", 1)[0] for o in brief}
    assert set(smoke) == {o for o in real if o.split("=", 1)[0] not in replaced} | brief
    assert _compose(smoke).algo.total_timesteps == 1000


@pytest.mark.parametrize("arm", ARMS)
def test_every_arm_launches(arm, tmp_path):
    """PYTHON=true stands in for the trainer."""
    (tmp_path / "offline_buffer_cache").mkdir()
    env = {**os.environ, "SKIP_PREFLIGHT": "1", "PYTHON": "true", "ALLOW_NO_TMUX": "1",
           "CACHE_DIR": str(tmp_path)}
    env.pop("DRY_RUN", None)
    env.pop("SMOKE", None)
    proc = subprocess.run([str(SUBMIT), arm, "1"], env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert f"Launching {arm}-1" in proc.stdout


# ---------------------------------------------------------------------------
# Tier 2: each arm's agent, as its composed config builds it
# ---------------------------------------------------------------------------


def _agent(cfg, device):
    """The agent the trainer builds from `cfg`, on the reference batch's shapes."""
    import torch
    from omegaconf import open_dict

    from tools.agent_fingerprint import ACTION_DIM, CAMERAS, OBS_SHAPE, ROTATION_CENTRE, STATE_DIM, _stats

    with open_dict(cfg):
        cfg.agent.device = str(device)
    shapes = dict(obs_shape=OBS_SHAPE, prop_shape=(STATE_DIM,), action_dim=ACTION_DIM,
                  rl_cameras=list(CAMERAS), residual_actor=cfg.base_policy is not None)
    if cfg.equivariance is None:
        from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
        from resfit.rl_finetuning.utils.normalization import ActionScaler

        # The trainer's scaler, on stand-in demo statistics.
        scaler = ActionScaler.from_dataset_stats(
            _stats(ACTION_DIM), action_scale=cfg.agent.actor.action_scale,
            min_range_per_dim=cfg.offline_data.min_action_range, device=device,
            clamp_unscaled=cfg.agent.actor.action_bound == "box")
        return QAgent(cfg=cfg.agent, action_scaler=scaler, **shapes)
    from resfit.rl_finetuning.equi_off_policy.networks.equi_normalizer import build_equivariant_normalizer
    from resfit.rl_finetuning.equi_off_policy.rl.q_agent import QAgent

    agent = QAgent(agent_cfg=cfg.agent, equi_cfg=cfg.equivariance, equivariant=True, **shapes)
    normalizer = build_equivariant_normalizer(
        stats={"observation.state": _stats(STATE_DIM), "action": _stats(ACTION_DIM)},
        robot_base_xy=torch.tensor(ROTATION_CENTRE),
    )
    agent.enc.set_normalizer(normalizer, robot_base_xy=list(ROTATION_CENTRE))
    return agent.to(device)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("arm", ARMS)
def test_arm_builds_and_updates(arm, tmp_path, device):
    import torch

    from tools.agent_fingerprint import ACTION_DIM, BATCH, STDDEV, fixed_batch

    cfg = _compose(_overrides(arm, tmp_path))
    residual = cfg.base_policy is not None
    agent = _agent(cfg, device)
    agent.train(True)
    metrics = agent.update(fixed_batch(base_action=residual).to(device), stddev=STDDEV, update_actor=True)
    for key in ("train/critic_loss", "train/actor_loss_base"):
        assert math.isfinite(metrics[key]), f"{arm}: {key} = {metrics[key]}"

    agent.train(False)
    with torch.no_grad():
        action = agent.act(fixed_batch(base_action=residual)["obs"].to(device), eval_mode=True, cpu=False)
    assert action.shape == (BATCH, ACTION_DIM)
    assert torch.isfinite(action).all(), f"{arm}: non-finite eval action"




@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("training", [False, True], ids=["eval", "train"])
@pytest.mark.parametrize("arm", EQUI_ARMS)
def test_equivariant_arm_commutes_with_the_group(arm, training, tmp_path, device):
    """The arm's own networks, at its sizes: pi(g.s) = g.pi(s), and Q(g.s, g.a) = Q(s, a). A
    zero-initialized last layer makes the actor output zero, which commutes trivially, so the
    actor's layers are checked at escnn's own init."""
    import torch
    from escnn import nn as enn

    from tests import group_action as ga
    from tests.conftest import N, ONE_ARM, TOL
    from tests.test_equivariance import _assert_actor_equivariant, _feat, _rel_err

    cfg = _compose(_overrides(arm, tmp_path))
    assert cfg.equivariance.N == N
    agent = _agent(cfg, device)
    agent.train(training)  # QAgent.train returns None
    eq, spec = cfg.equivariance, ONE_ARM
    critic_layout = ga.enc_out_layout_critic(eq.enc_degree_channel, eq.ih_out_channel,
                                             spec["n_arms"], spec["gripper_dim"])
    if cfg.base_policy is None:
        actor_type, actor_layout = agent.enc.enc_out_type_critic, critic_layout
    else:
        actor_type = agent.enc.enc_out_type_actor
        actor_layout = ga.enc_out_layout_actor(eq.enc_degree_channel, eq.ih_out_channel,
                                               spec["n_arms"], spec["gripper_dim"], spec["hand_dof"])
    actor = agent.actor
    if cfg.agent.actor.actor_last_layer_init_scale == 0.0:
        from omegaconf import open_dict

        with open_dict(cfg):
            cfg.agent.actor.actor_last_layer_init_scale = None
        unzeroed = _agent(cfg, device)
        unzeroed.train(training)
        actor = unzeroed.actor
    _assert_actor_equivariant(actor, actor_type, actor_layout, spec, device, arm, seed=21)

    act_layout = ga.action_layout(spec["n_arms"], spec["hand_dof"])
    feat = _feat(None, critic_layout, device, seed=22)
    act = _feat(None, act_layout, device, seed=23)
    with torch.no_grad():
        q = agent.critic(enn.GeometricTensor(feat, agent.enc.enc_out_type_critic), act)
        assert q.shape[0] == cfg.agent.critic.num_q
        for k in range(1, N):
            got = agent.critic(enn.GeometricTensor(ga.act_on_layout(feat, critic_layout, k, N),
                                                   agent.enc.enc_out_type_critic),
                               ga.act_on_layout(act, act_layout, k, N))
            err = _rel_err(got, q)
            assert err < TOL, f"{arm}: critic not invariant at k={k}: {err:.2e}"


# Changes to one input of the reference batch, each large against that input's range. Every input
# reaches the critic and the actor; the trivial-type ones are the wrist camera, the arm's height,
# the gripper opening, and the z, yaw and gripper actions.
SEES = 1e-4  # every input moves Q by at least 1.1e-3 without FieldNorm, a trivial one by at most
             # 6.7e-6 with it (2026-10-08, on this batch at init)
TRIVIAL_INPUTS = ("wrist image", "state z", "state gripper", "action z", "action yaw", "action gripper")


def _input_changes():
    from tools.agent_fingerprint import CAMERAS

    def image(cam):
        return lambda obs, act: obs.__setitem__(cam, obs[cam].new_zeros(obs[cam].shape))

    def state(i, by):
        return lambda obs, act: obs.__setitem__("observation.state", obs["observation.state"].index_add(
            1, obs["observation.state"].new_tensor([i], dtype=int),
            obs["observation.state"].new_full((obs["observation.state"].shape[0], 1), by)))

    def action(i):
        return lambda obs, act: act.__setitem__((slice(None), i), act[:, i] + 0.5)

    return {
        "front image": image(CAMERAS[0]), "wrist image": image(CAMERAS[1]),
        "state x": state(0, 0.1), "state z": state(2, 0.2), "state gripper": state(7, 0.02),
        "action x": action(0), "action rx": action(3), "action z": action(2),
        "action yaw": action(5), "action gripper": action(6),
    }


def _output_changes(cfg, device) -> dict[str, tuple[float, float]]:
    """Per input change, the largest change in Q (mean over heads) and in the actor's mean action.

    A residual actor's last layer starts scaled down (1e-4), and its mean is squashed and multiplied
    by its action scale (0.1); each is a fixed map on the output that shrinks every input's change
    alike. What reaches the actor is a property of its layers, so it is built at escnn's own init and
    read before the squash and the scale. (Read after them, at escnn's init, the residual's change
    with the gripper was 6e-5: the layers' 7.5e-4, which grows with the gripper's change, squashed at
    |mu| up to about 3 and scaled by 0.1; 2026-10-09.)
    """
    import torch
    from omegaconf import open_dict

    from tools.agent_fingerprint import fixed_batch

    residual = cfg.base_policy is not None
    with open_dict(cfg):
        cfg.agent.actor.actor_last_layer_init_scale = None
        cfg.agent.actor.action_scale = 1.0
        cfg.equivariance.residual_squash = False
    torch.manual_seed(0)
    agent = _agent(cfg, device)
    agent.train(False)  # QAgent.train returns None

    def outputs(change=None):
        batch = fixed_batch(base_action=residual).to(device)
        obs = {k: v.clone() for k, v in batch["obs"].items()}
        act = batch["action"].clone()
        if change is not None:
            change(obs, act)
        with torch.no_grad():
            actor_feat, critic_feat = agent.enc(obs)
            return agent.critic(critic_feat, act).mean(dim=0), agent.actor(actor_feat, std=None)

    q0, pi0 = outputs()
    changes = {}
    for name, change in _input_changes().items():
        q, pi = outputs(change)
        changes[name] = ((q - q0).abs().max().item(), (pi - pi0).abs().max().item())
    return changes


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("arm", EQUI_ARMS)
def test_every_input_can_move_the_critic_and_the_actor(arm, tmp_path, device):
    """Invariance tests pass on a network that ignores an input, so this checks what they cannot."""
    changes = _output_changes(_compose(_overrides(arm, tmp_path)), device)
    blind_q = {k: f"{dq:.1e}" for k, (dq, _) in changes.items() if dq <= SEES}
    assert not blind_q, f"{arm}: Q does not move with {blind_q}"
    blind_pi = {k: f"{dpi:.1e}" for k, (_, dpi) in changes.items()
                if not k.startswith("action") and dpi <= SEES}
    assert not blind_pi, f"{arm}: the actor does not move with {blind_pi}"


@pytest.mark.gpu
@pytest.mark.parametrize("arm", ARCH_ARMS)
def test_each_arch_arm_has_resfits_layout(arm, tmp_path, device):
    """As amended (2026-10-12): RegularLayerNorm after each hidden Linear of the actor and of the
    critic, and no other norm, none in the encoder.
    - The actor keeps its depth, ResFiT's: three hidden layers (its visual projection, its trunk and
      one policy layer), then the output; arch-wide's are 448 fields wide.
    - The critic has ResFiT's 10 heads of two hidden layers: three Linears each, the output after
      pooling. With its two projections, 22 norms."""
    from escnn import nn as enn

    from resfit.rl_finetuning.equi_off_policy.rl.equi_rl_utils import (
        FieldNorm,
        RegularInnerBatchNorm,
        RegularLayerNorm,
        TrivialLayerNorm,
    )

    def count(module, cls):
        return sum(isinstance(m, cls) for m in module.modules())

    agent = _agent(_compose(_overrides(arm, tmp_path)), device)
    others = (enn.IIDBatchNorm1d, RegularInnerBatchNorm, FieldNorm, TrivialLayerNorm)
    for part in (agent.enc, agent.actor, agent.critic):
        assert count(part, others) == 0, f"{arm}: another norm in {type(part).__name__}"
    assert count(agent.enc, RegularLayerNorm) == 0

    assert count(agent.actor, enn.Linear) == 4 and count(agent.actor, RegularLayerNorm) == 3
    fields = 448 if arm == ARCH_WIDE else 128
    for hidden in (agent.actor.visual_proj[0], agent.actor.input_proj[0], agent.actor.policy[0]):
        assert hidden.out_type.size == fields * 8, (arm, hidden.out_type.size)

    heads = agent.critic.q_ensemble.heads
    assert len(heads) == 10
    assert all(count(head, enn.Linear) == 3 for head in heads)
    assert count(agent.critic, RegularLayerNorm) == 2 + 10 * 2


def _seeded_agent(cfg, device, seed: int = 1):
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return _agent(cfg, device)


def _weights_in_order(module, drop_layer_norms: bool) -> list:
    """Every entry of the module's state, in order, less LayerNorms' if asked. Dropping a LayerNorm
    renumbers the Sequentials after it, so names cannot be matched; the order can. The critic's
    ensemble also keeps its heads' parameters stacked, as `_vmap_param_net_<i>_*` for layer i of
    its head template, so those of a template LayerNorm are dropped too."""
    import re

    import torch

    owners = dict(module.named_modules())
    out = []
    for name, tensor in module.state_dict().items():
        owner = owners.get(name.rsplit(".", 1)[0]) if "." in name else module
        stacked = re.search(r"(.*)_vmap_param_net_(\d+)_\w+$", name)
        if stacked:
            ensemble = owners[stacked.group(1).rstrip(".")]
            owner = ensemble._head_template.net[int(stacked.group(2))]
        if drop_layer_norms and isinstance(owner, torch.nn.LayerNorm):
            continue
        out.append(tensor)
    return out


@pytest.mark.gpu
@pytest.mark.parametrize("arm", RESFIT_ARMS)
def test_each_arm_starts_from_the_anchors_weights(arm, tmp_path, device):
    """At the same seed, the arm's initial weights are the anchor's bit for bit: rerendered changes
    the data alone; limits the bound and noise the noise's units, neither of which draws a random
    number or adds a weight; noln, noln-all and combined drop LayerNorms, which draw no random number
    at init, so every other weight is drawn as the anchor's. smallcritic's critic is smaller, and is
    built before the actor, so only its encoders are the anchor's. The episodes still differ, since
    `seed` does not reach the envs. An equivariant arch arm's weights are not its anchor's: its ten
    deeper heads draw more numbers before the actor's."""
    import torch

    anchor = _seeded_agent(_compose(_launch_order(RESFIT_ANCHOR)), device)
    ours = _seeded_agent(_compose(_overrides(arm, tmp_path)), device)
    if arm in (RERENDERED, LIMITS, NOISE):
        theirs, mine = anchor.state_dict(), ours.state_dict()
        assert theirs.keys() == mine.keys()
        for key, value in mine.items():
            assert torch.equal(value, theirs[key]), f"{arm}: {key} differs from the anchor's"
        return
    if arm == SMALLCRITIC:
        theirs, mine = anchor.encoders.state_dict(), ours.encoders.state_dict()
        assert theirs.keys() == mine.keys() and all(torch.equal(mine[k], theirs[k]) for k in mine)
        return
    # noln, combined: the anchor's actor and critic less their LayerNorms, its encoders as they were.
    # noln-all: its encoders less their LayerNorms too.
    for part in ("actor", "critic", "encoders"):
        if part == "encoders" and arm in (NOLN, COMBINED):
            theirs, mine = anchor.encoders.state_dict(), ours.encoders.state_dict()
            assert theirs.keys() == mine.keys() and all(torch.equal(mine[k], theirs[k]) for k in mine)
            continue
        theirs = _weights_in_order(getattr(anchor, part), drop_layer_norms=True)
        mine = _weights_in_order(getattr(ours, part), drop_layer_norms=False)
        assert len(theirs) == len(mine), (part, len(theirs), len(mine))
        for i, (a, b) in enumerate(zip(theirs, mine)):
            assert a.shape == b.shape and torch.equal(a, b), f"{arm}: {part} tensor {i} differs"


@pytest.mark.gpu
def test_smallcritic_has_the_equivariant_residuals_five_heads_of_one_hidden_layer(tmp_path, device):
    """eaqorqki's head count and depth (agent.critic.num_q=5, equivariance.num_critic_layers=1), at
    ResFiT's width: each head a hidden Linear, then the output."""
    import torch

    ensemble = _agent(_compose(_overrides(SMALLCRITIC, tmp_path)), device).critic.q_ensemble
    assert ensemble.num_heads == 5
    linears = [m for m in ensemble._head_template.net if isinstance(m, torch.nn.Linear)]
    assert len(linears) == 2 and linears[0].out_features == 1024
    assert ensemble._vmap_param_net_0_weight.shape[0] == 5


@pytest.mark.gpu
def test_noln_has_no_layernorm_in_its_actor_or_critic_and_its_vits_keep_theirs(tmp_path, device):
    """Where ResFiT's use_layer_norm puts LayerNorm: after the actor's compression and each of its
    hidden layers, and after the critic's input projection and each head's hidden layers. The ViT
    encoders' own (two per transformer layer and a final one) are another switch, noln-all's."""
    import torch

    def count(module):
        return sum(isinstance(m, torch.nn.LayerNorm) for m in module.modules())

    anchor = _agent(_compose(_launch_order(RESFIT_ANCHOR)), device)
    ours = _agent(_compose(_overrides(NOLN, tmp_path)), device)
    assert count(anchor.actor) > 0 and count(anchor.critic) > 0
    assert count(ours.actor) == count(ours.critic) == 0
    assert count(ours.encoders) == count(anchor.encoders) == 2 * 3


@pytest.mark.gpu
def test_noln_all_has_no_norm_anywhere(tmp_path, device):
    """No LayerNorm in the actor, the critic or the ViTs, and no other norm either: the patch embedding's
    GroupNorm is off in the anchor too (agent.vit.embed_norm=0)."""
    import torch

    ours = _agent(_compose(_overrides(NOLN_ALL, tmp_path)), device)
    norms = (torch.nn.LayerNorm, torch.nn.GroupNorm, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)
    found = [name for name, m in ours.named_modules() if isinstance(m, norms)]
    assert not found, found


@pytest.mark.gpu
@pytest.mark.slow
def test_the_field_norm_critic_these_arms_replace_is_blind_to_every_trivial_input(device):
    """eyyi99q4's critic: FieldNorm subtracts each regular field's trivial part, which is all a
    trivial-type input contributes to it (2026-10-08 FINDING). The rotating inputs still reach Q."""
    changes = _output_changes(_compose(BASELINES["equi-tuned-rlpd-can"]), device)
    for name, (dq, _) in changes.items():
        if name in TRIVIAL_INPUTS:
            assert dq < SEES, f"field norm's Q moved by {dq:.1e} with {name}"
        else:
            assert dq > SEES, f"field norm's Q did not move with {name}: {dq:.1e}"
