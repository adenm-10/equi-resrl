"""tools/rerender_demos.py: the demos' frames rendered again in our env, losslessly, every frame after
an episode's first one physics substep late, as the env observes.

After a MuJoCo step the body positions are one substep behind the state, and the tool's `as_stepped`
recovers them from the state alone (a small model, no renderer). Rendered in a subprocess, as the
gate runs tools/render_check.py, every guard passes on Can's first demos: frames and
observation.state rendered from a state are those the trainer's step path returned in it, bit for
bit; the demos pair with the LeRobot episodes and replay their observation.state. A built dataset,
where one exists, is lossless, complete, and records passing guards and the commit it was built at.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

from tools import rerender_demos as rr

REPO = Path(__file__).resolve().parents[1]
LEROBOT_HOME = Path.home() / ".cache/huggingface/lerobot"
CAMERAS = ("observation.images.agentview", "observation.images.robot0_eye_in_hand")


def _have_sources(task: str) -> bool:
    from huggingface_hub import try_to_load_from_cache

    _, fname, _, _ = rr.SOURCES[task]
    cached = try_to_load_from_cache(rr.ROBOMIMIC_REPO, fname, repo_type="dataset", revision=rr.ROBOMIMIC_REVISION)
    return isinstance(cached, str) and (LEROBOT_HOME / rr.SOURCES[task][0]).exists()


# A tumbling free body and a two-link arm under random torques, on robosuite's timestep and integrator.
MODEL = """
<mujoco>
  <option timestep="0.002" integrator="Euler"/>
  <worldbody>
    <body name="box" pos="0 0 1"><freejoint/><geom type="box" size=".05 .05 .05" mass="1"/></body>
    <body name="link1" pos="0.5 0 1">
      <joint name="j1" type="hinge" axis="0 0 1" damping="0.1"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02"/>
      <body name="link2" pos="0.3 0 0">
        <joint name="j2" type="hinge" axis="0 1 0" damping="0.1"/>
        <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02"/>
        <site name="tip" pos="0.3 0 0"/>
      </body>
    </body>
  </worldbody>
  <actuator><motor joint="j1" gear="5"/><motor joint="j2" gear="5"/></actuator>
</mujoco>
"""


def test_after_a_step_the_positions_are_the_previous_substeps_and_as_stepped_recovers_them():
    m = mujoco.MjModel.from_xml_string(MODEL)
    d, e = mujoco.MjData(m), mujoco.MjData(m)
    rng = np.random.default_rng(0)
    d.qvel[:6] = rng.normal(0, 1, 6)
    mujoco.mj_forward(m, d)
    lag = 0.0
    for _ in range(50):
        d.ctrl[:] = rng.uniform(-1, 1, m.nu)
        mujoco.mj_step(m, d)
        e.qpos[:], e.qvel[:] = d.qpos, d.qvel
        rr.as_stepped(m, e)
        np.testing.assert_allclose(e.xpos, d.xpos, rtol=0, atol=1e-12)
        np.testing.assert_allclose(e.site_xpos, d.site_xpos, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(e.qpos, d.qpos)  # the joint angles stay the state's
        mujoco.mj_forward(m, e)  # positions from the state itself, as reset() leaves them
        lag = max(lag, float(np.abs(e.xpos - d.xpos).max()))
    assert lag > 1e-4, lag  # the gap as_stepped reproduces is there to reproduce


def test_the_stepped_state_may_move_only_the_end_effector_pose():
    exact = np.tile(np.array([0.1, 0.2, 0.9, 0, 0, 0, 1, 0.02, -0.02], np.float32), (4, 1))
    stepped = exact.copy()
    stepped[1:, rr.EEF_POS] += 1e-3
    stepped[2, rr.EEF_QUAT] *= -1  # the same rotation
    pos, quat = rr.stepped_change(stepped, exact)
    assert pos == pytest.approx(np.sqrt(3) * 1e-3, rel=1e-3) and quat == 0.0
    moved = stepped.copy()
    moved[3, rr.GRIPPER] += 1e-4
    with pytest.raises(rr.GuardFailed, match="gripper"):
        rr.stepped_change(moved, exact)
    far = stepped.copy()
    far[3, rr.EEF_POS] += 2 * rr.LAG_POS_TOL
    with pytest.raises(rr.GuardFailed, match="moved"):
        rr.stepped_change(far, exact)


@pytest.mark.slow
@pytest.mark.skipif(not _have_sources("Can"), reason="Can's robomimic file or LeRobot dataset is not cached")
def test_every_guard_passes_on_cans_first_demos():
    proc = subprocess.run(
        [sys.executable, "-m", "tools.rerender_demos", "--task", "Can", "--check-only", "--episodes", "2", "--json"],
        cwd=REPO, capture_output=True, text=True, check=False,
        env={**os.environ, "PYTHONPATH": f"{REPO}:{os.environ.get('PYTHONPATH', '')}"},
    )
    found = [json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{")]
    assert len(found) == 1, proc.stderr[-2000:]
    r = found[0]
    assert "guard_failed" not in r, r
    assert r["identity"]["bit_identical"] and r["identity"]["resets"] == 3 and r["identity"]["steps"] == 90
    assert r["pairing"] == {"episodes": 2, "frames": r["pairing"]["frames"], "actions_exact": True}
    assert r["replay"]["max_proprio_diff"] <= rr.PROPRIO_TOL
    assert r["stepped_state"]["gripper_unchanged"]
    assert 0 < r["stepped_state"]["eef_pos_max_m"] <= rr.LAG_POS_TOL


@pytest.mark.parametrize("task", sorted(rr.SOURCES))
def test_a_built_dataset_is_lossless_complete_and_records_passing_guards(task):
    root = LEROBOT_HOME / rr.SOURCES[task][3]
    if not (root / "meta/rerender.json").exists():
        pytest.skip(f"{rr.SOURCES[task][3]} is not built")
    report = json.loads((root / "meta/rerender.json").read_text())
    info = json.loads((root / "meta/info.json").read_text())
    source = json.loads((LEROBOT_HOME / rr.SOURCES[task][0] / "meta/info.json").read_text())
    assert report["commit"]["dirty"] == [], report["commit"]
    assert report["versions"]["mujoco"] == mujoco.__version__
    assert report["identity"]["bit_identical"] and report["pairing"]["actions_exact"]
    assert report["replay"]["max_proprio_diff"] <= rr.PROPRIO_TOL
    assert report["columns"]["copied_columns_equal"] and report["columns"]["state_is_rendered"]
    for counts in ((report["pairing"]["episodes"], report["pairing"]["frames"]),
                   (info["total_episodes"], info["total_frames"])):
        assert counts == (source["total_episodes"], source["total_frames"])
    for cam in CAMERAS:
        assert info["features"][cam]["dtype"] == "image", cam  # PNG in the parquet, no video


def test_the_dataset_name_moves_the_offline_cache_key_and_no_other():
    import preflight

    for method, stack in (("residual", "equi"), ("tuned-rlpd", "equi"), ("tuned-rlpd", "resfit")):
        for task, (source, _, _, rerendered) in rr.SOURCES.items():
            off, on = preflight.cache_hashes(task, method, stack)
            assert preflight.cache_hashes(task, method, stack, dataset_name=source) == (off, on)
            off_r, on_r = preflight.cache_hashes(task, method, stack, dataset_name=rerendered)
            assert off_r != off and on_r == on, (task, method, stack)


def test_the_gate_reads_the_runs_dataset_name():
    import preflight

    ok, flags = preflight.tier1_compose(["--config-name=equi_tuned_rlpd_can_config",
                                         f"offline_data.name={rr.SOURCES['Can'][3]}"])
    assert ok and flags["offline_data.name"] == rr.SOURCES["Can"][3]
