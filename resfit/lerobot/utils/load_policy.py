# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch
import wandb

from resfit.lerobot.policies.act.modeling_act import ACTPolicy
from resfit.lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy


def download_policy_from_wandb(
    run_id: str,
    *,
    step: str | None = None,
    artifact_version: str = "latest",
) -> tuple[Path, str]:
    """Download a policy checkpoint logged on W&B and return its folder.

    The policy is expected to have been created with the training utilities in
    `train_hf.py` and therefore to contain a `config.json` in the root of the
    downloaded artifact.
    """
    api = wandb.Api()
    project, id_ = run_id.split("/")

    if step is None or str(step).lower() == "latest":
        artifact_name = f"run_{id_}_latest:{artifact_version}"
        checkpoint_step = "latest"
    elif str(step).lower() == "best":
        artifact_name = f"run_{id_}_best:{artifact_version}"
        checkpoint_step = "best"
    else:
        artifact_name = f"run_{id_}_model_step_{step}:{artifact_version}"
        checkpoint_step = str(step)

    artifact_path = f"{project}/{artifact_name}"
    artifact = api.artifact(artifact_path)

    art_dir = Path(artifact.download())
    policy_dir = art_dir / "policy"  # The artifact root already contains the policy files.

    if not (policy_dir / "config.json").exists():
        raise FileNotFoundError(f"Policy directory not found inside downloaded artifact: {policy_dir}")

    return policy_dir, checkpoint_step


def load_policy(policy_dir: Path) -> ACTPolicy:
    """Infer policy type (diffusion / act) from `config.json` and load weights."""

    with (policy_dir / "config.json").open() as f:
        cfg_dict = json.load(f)

    policy_name_field = str(cfg_dict.get("type", "")).lower()

    # TODO: improve policy-type inference logic when additional policies are added
    if "diffusion" in policy_name_field:
        # raise NotImplementedError("Diffusion policy not implemented")
        return DiffusionPolicy.from_pretrained(policy_dir)
    if "use_vae" in cfg_dict:
        return ACTPolicy.from_pretrained(policy_dir)

    raise ValueError(f"Unknown policy type: {policy_name_field}")


def save_checkpoint(ckpt_dir: Path, step: int, policy, optimizer) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # Save model weights + config
    policy.save_pretrained(ckpt_dir / "policy")
    # Save optimizer & misc state
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
        },
        ckpt_dir / "trainer_state.pt",
    )


def load_checkpoint(ckpt_dir: Path, policy, optimizer):
    state_pth = ckpt_dir / "trainer_state.pt"
    if not state_pth.exists():
        raise FileNotFoundError(state_pth)
    state = torch.load(state_pth, map_location="cpu")
    policy_loaded = policy.from_pretrained(ckpt_dir / "policy")
    optimizer.load_state_dict(state["optimizer"])
    return state["step"], policy_loaded, optimizer


# -----------------------------------------------------------------------------
# Run packages ------------------------------------------------------------------
# -----------------------------------------------------------------------------
# One folder per run under outputs/runs/, persisted: everything the trainer saves
# locally, a copy of the run's single best upload (wandb_best/), and a copy of
# wandb's own record of the run (wandb_logs/). Kept here because both trainers
# import this module and download_policy_from_wandb reads the upload written below.

# Resolved at import, like the trainers' _CACHE_ROOT, so a Hydra chdir cannot move it.
RUNS_ROOT = Path(os.environ.get("CACHE_DIR", ".")).expanduser().resolve() / "outputs" / "runs"

BEST_VIDEO = "best_eval.mp4"
WANDB_LOG_FILES = ("config.yaml", "output.log", "wandb-summary.json", "wandb-metadata.json", "requirements.txt")


def _live_run():
    """The current wandb run, or None when wandb is off or disabled (debug mode)."""
    run = wandb.run
    return None if run is None or run.disabled else run


def uploads_enabled() -> bool:
    run = _live_run()
    return run is not None and not run.offline


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))


def _git_state() -> dict:
    """HEAD plus uncommitted changes, from the checkout this code was imported from.

    HEAD alone does not identify the code that ran: z8yoqylh logged 7dae4925 while
    running caf83f3's tree.
    """
    here = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=here, capture_output=True, text=True, check=True
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"commit": commit, "dirty": dirty}


def create_run_package() -> Path:
    """Create this run's package and record how it was launched.

    Call after wandb.init. Named by wandb run id so a folder maps to its run; runs
    without wandb go to local/<timestamp>. A resumed run reuses its folder and has
    the new launch appended.
    """
    run = _live_run()
    started = datetime.now()
    if run is None:
        package = RUNS_ROOT / "local" / started.strftime("%Y-%m-%d_%H-%M-%S")
    else:
        package = RUNS_ROOT / run.project / run.id
    package.mkdir(parents=True, exist_ok=True)

    launch = {
        "started": started.isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "argv": sys.argv,
        "git": _git_state(),
    }
    manifest_path = package / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest["resumes"].append(launch)
    else:
        manifest = {
            "wandb": None if run is None else {
                "entity": run.entity, "project": run.project, "id": run.id,
                "name": run.name, "url": run.url, "dir": run.dir,
            },
            "uploads": uploads_enabled(),
            **launch,
            "resumes": [],
            "completed": False,
        }
    _write_json(manifest_path, manifest)
    return package


def upload_best(package: Path, files: dict[str, Path], *, step: int, success_rate: float) -> None:
    """Make wandb_best/ hold exactly `files`, then make it the run's only best upload.

    Keys are names inside the upload; "policy" keeps the layout download_policy_from_wandb
    expects. Older versions are deleted only after the new one has committed, so a
    failed upload never leaves the run without a best. A failed delete is retried at
    the next call.
    """
    staging = package / "wandb_best"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    for name, src in files.items():
        if src.is_dir():
            shutil.copytree(src, staging / name)
        else:
            shutil.copy2(src, staging / name)
    best = {"step": step, "success_rate": success_rate}
    _write_json(staging / "best.json", best)

    if not uploads_enabled():
        return

    run = wandb.run
    name = f"run_{run.id}_best"
    artifact = wandb.Artifact(name=name, type="model", metadata=best)
    artifact.add_dir(str(staging))
    logged = run.log_artifact(artifact, aliases=["best", "latest"]).wait()
    if logged.version is None:
        # Without the new version's name, every version would match the delete below.
        print(f"[upload_best] {name} committed without a version; superseded versions kept")
        return

    try:
        for version in wandb.Api().artifacts("model", f"{run.entity}/{run.project}/{name}"):
            if version.version != logged.version:
                version.delete(delete_aliases=True)
    except Exception as exc:
        print(f"[upload_best] superseded versions of {name} not deleted: {type(exc).__name__}: {exc}")


def log_best_video_to_panel(package: Path, key: str, step: int) -> None:
    """Post the uploaded best eval's video to the run's panel, once.

    `step` must be the run's last step: wandb drops anything logged at an earlier one.
    """
    video = package / "wandb_best" / BEST_VIDEO
    if not video.exists() or not uploads_enabled():
        return
    best = json.loads((package / "wandb_best" / "best.json").read_text())
    caption = f"best eval: step {best['step']}, success {best['success_rate']:.2f}"
    wandb.log({key: wandb.Video(str(video), format="mp4", caption=caption)}, step=step)


def _history_from_record(record: Path) -> list[dict]:
    """Every history row in a run-<id>.wandb record, in logged order.

    Read from the local record rather than the API: the API's full-history scan drops
    the final row, which is where the last eval and the best-video panel entry land.
    """
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal import datastore

    store = datastore.DataStore()
    store.open_for_scan(str(record))
    rows = []
    while (data := store.scan_data()) is not None:
        entry = wandb_internal_pb2.Record()
        entry.ParseFromString(data)
        if entry.WhichOneof("record_type") != "history":
            continue
        row: dict = {}
        for item in entry.history.item:
            keys = [item.key] if item.key else list(item.nested_key)
            target = row
            for key in keys[:-1]:
                target = target.setdefault(key, {})
            target[keys[-1]] = json.loads(item.value_json)
        rows.append(row)
    return rows


def finalize_run_package(package: Path) -> None:
    """Copy wandb's record of the run into wandb_logs/ and mark the package complete.

    Call after wandb.finish(), so output.log, the summary and the record are flushed.
    The run-<id>.wandb record is what `wandb sync` restores a deleted run from, and
    metrics.jsonl is its history rows as JSON lines.
    """
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    run_info = manifest["wandb"]

    if run_info is not None:
        logs = package / "wandb_logs"
        logs.mkdir(exist_ok=True)
        files_dir = Path(run_info["dir"])
        for name in WANDB_LOG_FILES:
            if (files_dir / name).exists():
                shutil.copy2(files_dir / name, logs / name)
        for record in files_dir.parent.glob("*.wandb"):
            shutil.copy2(record, logs / record.name)
            try:
                history = _history_from_record(record)
                (logs / "metrics.jsonl").write_text(
                    "".join(json.dumps(row, default=str) + "\n" for row in history)
                )
            except Exception as exc:
                print(f"[finalize_run_package] metrics.jsonl not written: {type(exc).__name__}: {exc}")

    best_path = package / "wandb_best" / "best.json"
    manifest["best"] = json.loads(best_path.read_text()) if best_path.exists() else None
    manifest["finished"] = datetime.now().isoformat(timespec="seconds")
    manifest["completed"] = True
    _write_json(manifest_path, manifest)
