"""Tier 1: run packages persist locally, and wandb keeps exactly one best upload.

wandb is replaced by a fake that records calls, so nothing here touches the
network. The residual round trip builds a small agent on the gate's device.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from resfit.lerobot.utils import load_policy as lp
from tests.conftest import ENC_HIDDEN, N, OBS_SHAPE
from tests.test_regression_single_arm import BASE_XY, _inputs, _stats


class _FakeArtifact:
    def __init__(self, name, type, metadata):  # noqa: A002 - mirrors wandb.Artifact
        self.name, self.type, self.metadata = name, type, metadata
        self.files = []

    def add_dir(self, path):
        root = Path(path)
        self.files = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


class _FakeWandb:
    """Records uploads, waits and deletes. `versions` is what the server holds."""

    def __init__(self, tmp_path: Path, *, offline: bool = False):
        files_dir = tmp_path / "wandb" / "run-20260925_000000-abc123" / "files"
        files_dir.mkdir(parents=True)
        self.run = SimpleNamespace(
            id="abc123", project="proj", entity="ent", name="name", url="https://wandb.ai/ent/proj/runs/abc123",
            dir=str(files_dir), disabled=False, offline=offline, log_artifact=self._log_artifact,
        )
        self.events: list[tuple[str, str]] = []
        self.versions: list[str] = []
        self.uploads: dict[str, list[str]] = {}
        self.logged: list[tuple[dict, int]] = []
        self.fail_delete = False
        self.Artifact = _FakeArtifact

    def _log_artifact(self, artifact, aliases):
        assert aliases == ["best", "latest"]
        version = f"v{len(self.uploads)}"
        self.uploads[version] = artifact.files
        self.events.append(("log", version))

        def wait():
            self.events.append(("wait", version))
            self.versions.append(version)
            return SimpleNamespace(version=version)

        return SimpleNamespace(wait=wait)

    def _delete(self, version):
        if self.fail_delete:
            raise RuntimeError("network down")
        self.events.append(("delete", version))
        self.versions.remove(version)

    def Api(self):  # noqa: N802 - mirrors wandb.Api
        self.events.append(("api", ""))
        return SimpleNamespace(
            artifacts=lambda type_name, name: [
                SimpleNamespace(version=v, delete=lambda delete_aliases, v=v: self._delete(v))
                for v in list(self.versions)
            ],
        )

    def Video(self, path, format, caption):  # noqa: N802, A002 - mirrors wandb.Video
        return SimpleNamespace(path=path, caption=caption)

    def log(self, data, step):
        self.logged.append((data, step))


@pytest.fixture
def fake(tmp_path, monkeypatch):
    fake = _FakeWandb(tmp_path)
    monkeypatch.setattr(lp, "wandb", fake)
    monkeypatch.setattr(lp, "RUNS_ROOT", tmp_path / "runs")
    return fake


def _best_files(tmp_path: Path, tag: str, *, video: bool = True) -> dict[str, Path]:
    policy = tmp_path / tag / "policy"
    policy.mkdir(parents=True)
    (policy / "config.json").write_text(json.dumps({"tag": tag}))
    files = {"policy": policy}
    if video:
        (tmp_path / tag / "eval.mp4").write_text(tag)
        files[lp.BEST_VIDEO] = tmp_path / tag / "eval.mp4"
    return files


def _manifest(package: Path) -> dict:
    return json.loads((package / "manifest.json").read_text())


# ---------------------------------------------------------------------------
# create_run_package
# ---------------------------------------------------------------------------

def test_package_is_named_by_run_id(fake, tmp_path):
    package = lp.create_run_package()
    assert package == tmp_path / "runs" / "proj" / "abc123"
    manifest = _manifest(package)
    assert manifest["wandb"]["id"] == "abc123"
    assert manifest["uploads"] is True
    assert manifest["completed"] is False
    assert len(manifest["git"]["commit"]) == 40
    assert isinstance(manifest["git"]["dirty"], list)


def test_resumed_run_keeps_its_folder_and_appends_the_launch(fake):
    first = lp.create_run_package()
    started = _manifest(first)["started"]
    second = lp.create_run_package()
    assert second == first
    manifest = _manifest(second)
    assert manifest["started"] == started
    assert len(manifest["resumes"]) == 1


def test_run_without_wandb_is_packaged_under_local(fake, tmp_path):
    fake.run = None
    package = lp.create_run_package()
    assert package.parent == tmp_path / "runs" / "local"
    assert _manifest(package)["wandb"] is None
    assert _manifest(package)["uploads"] is False


# ---------------------------------------------------------------------------
# upload_best
# ---------------------------------------------------------------------------

def test_only_the_newest_best_survives(fake, tmp_path):
    package = lp.create_run_package()
    for step, tag in enumerate("abc"):
        lp.upload_best(package, _best_files(tmp_path, tag), step=step, success_rate=0.1 * step)

    assert fake.versions == ["v2"]
    staging = package / "wandb_best"
    assert json.loads((staging / "policy" / "config.json").read_text()) == {"tag": "c"}
    assert json.loads((staging / "best.json").read_text()) == {"step": 2, "success_rate": pytest.approx(0.2)}
    for files in fake.uploads.values():
        assert files == ["best.json", lp.BEST_VIDEO, "policy/config.json"]


def test_a_version_is_deleted_only_after_its_successor_commits(fake, tmp_path):
    package = lp.create_run_package()
    for step, tag in enumerate("ab"):
        lp.upload_best(package, _best_files(tmp_path, tag), step=step, success_rate=0.5)

    assert fake.events.index(("delete", "v0")) > fake.events.index(("wait", "v1"))


def test_a_failed_delete_leaves_an_extra_version_and_does_not_raise(fake, tmp_path):
    fake.fail_delete = True
    package = lp.create_run_package()
    for step, tag in enumerate("ab"):
        lp.upload_best(package, _best_files(tmp_path, tag), step=step, success_rate=0.5)

    assert fake.versions == ["v0", "v1"]
    assert json.loads((package / "wandb_best" / "policy" / "config.json").read_text()) == {"tag": "b"}


def test_a_commit_without_a_version_deletes_nothing(fake, tmp_path):
    package = lp.create_run_package()
    lp.upload_best(package, _best_files(tmp_path, "a"), step=0, success_rate=0.5)
    real_log = fake.run.log_artifact
    fake.run.log_artifact = lambda artifact, aliases: SimpleNamespace(
        wait=lambda: (real_log(artifact, aliases).wait(), SimpleNamespace(version=None))[1]
    )
    lp.upload_best(package, _best_files(tmp_path, "b"), step=1, success_rate=0.6)

    assert fake.versions == ["v0", "v1"]
    assert ("delete", "v0") not in fake.events


def test_offline_run_is_packaged_but_uploads_nothing(tmp_path, monkeypatch):
    fake = _FakeWandb(tmp_path, offline=True)
    monkeypatch.setattr(lp, "wandb", fake)
    monkeypatch.setattr(lp, "RUNS_ROOT", tmp_path / "runs")

    package = lp.create_run_package()
    lp.upload_best(package, _best_files(tmp_path, "a"), step=0, success_rate=0.5)
    lp.log_best_video_to_panel(package, "eval/video", step=10)

    assert (package / "wandb_best" / lp.BEST_VIDEO).exists()
    assert fake.events == []
    assert fake.logged == []


# ---------------------------------------------------------------------------
# log_best_video_to_panel
# ---------------------------------------------------------------------------

def test_best_video_is_posted_once_at_the_given_step(fake, tmp_path):
    package = lp.create_run_package()
    lp.upload_best(package, _best_files(tmp_path, "a"), step=40, success_rate=0.75)
    lp.log_best_video_to_panel(package, "eval/video", step=300)

    [(data, step)] = fake.logged
    assert step == 300
    assert data["eval/video"].path == str(package / "wandb_best" / lp.BEST_VIDEO)
    assert data["eval/video"].caption == "best eval: step 40, success 0.75"


def test_no_video_means_no_panel_entry(fake, tmp_path):
    package = lp.create_run_package()
    lp.upload_best(package, _best_files(tmp_path, "a", video=False), step=0, success_rate=0.5)
    lp.log_best_video_to_panel(package, "eval/video", step=10)
    assert fake.logged == []


# ---------------------------------------------------------------------------
# finalize_run_package
# ---------------------------------------------------------------------------

# Last row mirrors the real final row: an eval plus the panel video, a nested media item.
HISTORY = [
    {"_step": 0, "train/loss": 1.5},
    {"_step": 10, "eval/success_rate": 0.7,
     "eval/video": {"_type": "video-file", "caption": "best eval: step 0, success 0.50"}},
]


def _write_wandb_files(fake, history=HISTORY):
    """wandb's local files for the fake run, with a genuine run-<id>.wandb record."""
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal import datastore

    files_dir = Path(fake.run.dir)
    for name in lp.WANDB_LOG_FILES:
        (files_dir / name).write_text(name)
    store = datastore.DataStore()
    store.open_for_write(str(files_dir.parent / "run-abc123.wandb"))
    for row in history:
        entry = wandb_internal_pb2.Record()
        for key, value in row.items():
            if isinstance(value, dict):
                for sub, sub_value in value.items():
                    entry.history.item.add(nested_key=[key, sub], value_json=json.dumps(sub_value))
            else:
                entry.history.item.add(key=key, value_json=json.dumps(value))
        store.write(entry)
    store.close()


def test_finalize_copies_wandb_record_and_completes_manifest(fake, tmp_path):
    _write_wandb_files(fake)
    package = lp.create_run_package()
    lp.upload_best(package, _best_files(tmp_path, "a"), step=40, success_rate=0.75)
    lp.finalize_run_package(package)

    logs = package / "wandb_logs"
    for name in (*lp.WANDB_LOG_FILES, "run-abc123.wandb"):
        assert (logs / name).exists(), name
    rows = [json.loads(line) for line in (logs / "metrics.jsonl").read_text().splitlines()]
    assert rows == HISTORY

    manifest = _manifest(package)
    assert manifest["completed"] is True
    assert manifest["best"] == {"step": 40, "success_rate": 0.75}


def test_finalize_survives_an_unreadable_record(fake):
    _write_wandb_files(fake)
    (Path(fake.run.dir).parent / "run-abc123.wandb").write_text("not a record")
    package = lp.create_run_package()
    lp.finalize_run_package(package)

    assert (package / "wandb_logs" / "output.log").exists()
    assert not (package / "wandb_logs" / "metrics.jsonl").exists()
    assert _manifest(package)["completed"] is True


# ---------------------------------------------------------------------------
# save_residual_model / load_residual_model
# ---------------------------------------------------------------------------

def _agent(device, *, seed: int, stats_scale: float):
    """Single-arm equivariant QAgent, small but built exactly as training builds it."""
    from resfit.rl_finetuning.config.residual_td3 import EquivarianceConfig
    from resfit.rl_finetuning.config.rlpd import QAgentConfig
    from resfit.rl_finetuning.equi_off_policy.networks.equi_normalizer import (
        build_equivariant_normalizer,
    )
    from resfit.rl_finetuning.equi_off_policy.rl.q_agent import QAgent

    torch.manual_seed(seed)
    agent = QAgent(
        obs_shape=OBS_SHAPE, prop_shape=(9,), action_dim=7,
        rl_cameras=["observation.images.agentview", "observation.images.robot0_eye_in_hand"],
        agent_cfg=QAgentConfig(device=str(device)),
        equi_cfg=EquivarianceConfig(
            N=N, enc_degree_channel=ENC_HIDDEN, actor_degree_channel=8, critic_degree_channel=8,
        ),
        residual_actor=True,
    )
    stats = {
        "observation.state": {k: v * stats_scale for k, v in _stats(9).items()},
        "action": {k: v * stats_scale for k, v in _stats(7).items()},
    }
    normalizer = build_equivariant_normalizer(stats=stats, robot_base_xy=torch.tensor(BASE_XY))
    agent.enc.set_normalizer(normalizer, robot_base_xy=BASE_XY)
    agent.to(device)
    return agent


def _act(agent, obs):
    from resfit.rl_finetuning.off_policy.common_utils import utils

    with torch.no_grad(), utils.eval_mode(agent):
        return agent.act(obs, eval_mode=True, cpu=False)


def test_residual_model_round_trip(tmp_path, device):
    """Weights and normalizer come back from the file: the loading agent starts with
    different ones, and its actions then match the saving agent's exactly."""
    from resfit.rl_finetuning.utils.checkpoint import load_residual_model, save_residual_model

    obs = {k: v.to(device) for k, v in _inputs().items()}
    src = _agent(device, seed=0, stats_scale=1.0)
    dst = _agent(device, seed=1, stats_scale=2.0)
    path = tmp_path / "models" / "best_model.pt"

    save_residual_model(src, path, config={"task": "Can"}, global_step=7, success_rate=0.5)
    assert src.training and src.actor.training, "saving must leave the agent in train mode"

    expected = _act(src, obs)
    assert not torch.allclose(_act(dst, obs), expected)

    rest = load_residual_model(dst, path)
    torch.testing.assert_close(_act(dst, obs), expected, rtol=0, atol=1e-6)
    assert rest == {"config": {"task": "Can"}, "global_step": 7, "success_rate": 0.5}
