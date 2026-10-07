"""Tier 1: tools/gpu_queue.sh launches a queued line only when it is safe to.

Each test runs the real script against a scratch HOME, queue and CACHE_DIR. The
"worktree" is a one-commit git repo whose submit.sh records its arguments, so a
launch is visible without training anything.

What is pinned, from the full disk of 2026-10-07:
  - a line that cannot be deleted is not launched (the runner relaunched one arm
    44 times when `sed -i` failed);
  - nothing launches below the disk floor;
  - a log line cut short without its newline does not swallow the next entry;
  - editing the script while it runs does not change the running runner.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import time

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = REPO / "tools" / "gpu_queue.sh"

STUB_SUBMIT = """#!/bin/bash
echo "$@" >> "$LAUNCHES"
"""


@pytest.fixture
def scratch(tmp_path):
    """A HOME holding one pinned 'worktree', a queue file, and a CACHE_DIR."""
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "submit.sh").write_text(STUB_SUBMIT)
    (stage / "submit.sh").chmod(0o755)
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(stage)]
    subprocess.run(["git", "init", "-q", str(stage)], check=True)
    subprocess.run([*git, "add", "submit.sh"], check=True)
    subprocess.run([*git, "commit", "-qm", "stub"], check=True)
    commit = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True,
                            check=True).stdout.strip()
    home = tmp_path / "home"
    (home / "projects").mkdir(parents=True)
    stage.rename(home / "projects" / f"equi-resrl-{commit[:7]}")

    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    cache = tmp_path / "cache"
    cache.mkdir()
    env = {**os.environ, "HOME": str(home), "TMUX": "test", "CACHE_DIR": str(cache),
           "LAUNCHES": str(tmp_path / "launches.txt"), "QUEUE_POLL_SECONDS": "0.1",
           "QUEUE_MIN_FREE_GB": "0"}
    return {"commit": commit[:7], "queue": queue_dir / "q.txt", "env": env,
            "log": cache / "outputs" / "queue" / "gpu0.log",
            "launches": tmp_path / "launches.txt", "tmp": tmp_path}


def _run(s, *, timeout=20, script=SCRIPT):
    """Run the runner to its STOP line; return its exit code."""
    return subprocess.run(["bash", str(script), "0", str(s["queue"])], env=s["env"],
                          capture_output=True, text=True, timeout=timeout).returncode


def _run_for(s, seconds):
    """Run the runner for `seconds`, then kill it: for a runner that must not move on."""
    proc = subprocess.Popen(["bash", str(SCRIPT), "0", str(s["queue"])], env=s["env"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(seconds)
    proc.kill()
    proc.wait()


def _launches(s):
    return s["launches"].read_text().splitlines() if s["launches"].exists() else []


def test_a_queued_line_launches_once_and_stop_ends_the_runner(scratch):
    scratch["queue"].write_text(f"# a comment\n{scratch['commit']} arm-a 1\nSTOP\n")
    assert _run(scratch) == 0
    assert _launches(scratch) == ["arm-a 1 0"]
    assert scratch["queue"].read_text() == "# a comment\n"
    log = scratch["log"].read_text()
    assert "LAUNCH arm-a seed 1" in log and "EXIT 0 for arm-a seed 1" in log
    assert "STOP read" in log


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write to a read-only folder")
def test_a_line_that_cannot_be_deleted_is_not_launched(scratch):
    scratch["queue"].write_text(f"{scratch['commit']} arm-a 1\n")
    scratch["queue"].parent.chmod(0o555)  # sed -i writes a temp file beside the queue: it fails
    try:
        _run_for(scratch, 2.0)
    finally:
        scratch["queue"].parent.chmod(0o755)
    assert _launches(scratch) == []
    assert scratch["queue"].read_text() == f"{scratch['commit']} arm-a 1\n"
    assert "could not delete it" in scratch["log"].read_text()


def test_nothing_launches_below_the_disk_floor(scratch):
    scratch["env"]["QUEUE_MIN_FREE_GB"] = str(10**9)
    scratch["queue"].write_text(f"{scratch['commit']} arm-a 1\n")
    _run_for(scratch, 1.0)
    assert _launches(scratch) == []
    assert scratch["queue"].read_text() == f"{scratch['commit']} arm-a 1\n", "the line stays queued"
    assert "the line stays queued" in scratch["log"].read_text()


def test_a_cut_off_log_line_does_not_swallow_the_next_entry(scratch):
    scratch["log"].parent.mkdir(parents=True)
    scratch["log"].write_text("2026-10-07 08:23:45 gpu0 LAUNCH cut short ->")
    scratch["queue"].write_text("STOP\n")
    assert _run(scratch) == 0
    lines = scratch["log"].read_text().splitlines()
    assert lines[0] == "2026-10-07 08:23:45 gpu0 LAUNCH cut short ->"
    assert all(line[:4].isdigit() for line in lines[1:]), lines


def test_editing_the_script_does_not_change_a_running_runner(scratch):
    copy = scratch["tmp"] / "gpu_queue.sh"
    shutil.copy(SCRIPT, copy)
    scratch["queue"].write_text("")
    proc = subprocess.Popen(["bash", str(copy), "0", str(scratch["queue"])], env=scratch["env"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            if scratch["log"].exists() and "runner started" in scratch["log"].read_text():
                break
            time.sleep(0.05)
        # Rewrite the file under the running runner, longer and shorter than the original.
        copy.write_text("echo replaced; exit 7\n" * 400)
        scratch["queue"].write_text(f"{scratch['commit']} arm-b 2\nSTOP\n")
        assert proc.wait(timeout=20) == 0
    finally:
        proc.kill()
    assert _launches(scratch) == ["arm-b 2 0"]
    assert "STOP read" in scratch["log"].read_text()
