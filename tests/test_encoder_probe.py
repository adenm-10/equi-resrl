"""Tier 1: tools/encoder_probe.py (2026-10-23), the probe of what a frozen encoder's taps hold about
the nut.

The targets read from a demo state set into the demos' own scene are the nut pose robosuite's own
env reports from that state (and its end-effector position, which pairs each state with its frame);
the episode split never puts one episode on two sides; ridge on principal scores recovers a linear
map, by the covariance and by the kernel; the AUROC is the pairwise count it stands for.
"""

from __future__ import annotations

import numpy as np
import pytest

from tools import encoder_probe as probe


def _states_file():
    try:
        from tools.rerender_demos import states_file

        return states_file("Square")
    except Exception as e:  # not downloaded on this machine
        pytest.skip(f"robomimic's Square states are not here: {e}")


def test_the_targets_are_robosuites_own_nut_pose():
    """From demo_0's states, in the demos' scene with no renderer, against robosuite's
    NutAssemblySquare env set to the same state: the nut's position and yaw, and the grip site's
    position (robot0_eef_pos)."""
    import h5py
    import mujoco
    import robosuite

    from tools.rerender_demos import demo_keys

    with h5py.File(_states_file(), "r") as f:
        g = f[f"data/{demo_keys(f)[0]}"]
        xml, states = g.attrs["model_file"], g["states"][()]
    m = probe.demo_scene(xml)
    d = mujoco.MjData(m)
    env = robosuite.make("NutAssemblySquare", robots="Panda", has_renderer=False, has_offscreen_renderer=False,
                         use_camera_obs=False, ignore_done=True)
    try:
        env.reset()
        assert env.sim.model.nq == m.nq
        for t in (0, len(states) // 2, len(states) - 1):
            probe.set_flat_state(m, d, states[t])
            mujoco.mj_forward(m, d)
            ours = probe.read_pose(m, d)
            env.sim.set_state_from_flattened(states[t])
            env.sim.forward()
            obs = env._get_observations(force_update=True)
            np.testing.assert_allclose(ours["nut"], obs["SquareNut_pos"], atol=1e-6)
            np.testing.assert_allclose(ours["grip"], obs["robot0_eef_pos"], atol=1e-6)
            x, y, z, w = obs["SquareNut_quat"]  # robosuite's xyzw
            yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            assert abs(np.angle(np.exp(1j * (ours["nut_yaw"] - yaw)))) < 1e-5
    finally:
        env.close()


def test_target_row_layout():
    pose = {"nut": np.array([0.1, 0.2, 0.9]), "peg": np.array([0.0, 0.1, 0.8]), "grip": np.array([0.1, 0.0, 1.0]),
            "nut_yaw": 0.5, "grip_yaw": 0.2}
    row = probe.target_row(pose, rest_z=0.88)
    assert len(row) == sum(s.stop - s.start for s in probe.TARGETS.values()) == 11
    np.testing.assert_allclose(row[probe.TARGETS["nut_grip_xyz"]], [0.0, 0.2, -0.1])
    np.testing.assert_allclose(row[probe.TARGETS["nut_grip_yaw"]], [np.sin(0.3), np.cos(0.3)])
    np.testing.assert_allclose(row[probe.TARGETS["nut_peg_xyz"]], [0.1, 0.1, 0.1])
    np.testing.assert_allclose(row[probe.TARGETS["nut_world_xy"]], [0.1, 0.2])
    assert row[probe.TARGETS["lifted"]][0] == 1.0  # 2 cm above its start
    assert probe.target_row(pose, rest_z=0.895)[probe.TARGETS["lifted"]][0] == 0.0


def test_the_episode_split_never_shares_an_episode():
    episodes = np.repeat(np.arange(300), 7)
    split = probe.episode_split(episodes)
    sets = {k: set(episodes[v]) for k, v in split.items()}
    assert (len(sets["test"]), len(sets["val"]), len(sets["fit"])) == (60, 48, 192)
    assert not (sets["test"] & sets["val"] or sets["test"] & sets["fit"] or sets["val"] & sets["fit"])
    assert sorted(np.concatenate(list(split.values()))) == list(range(len(episodes)))


@pytest.mark.parametrize("n, d", [(400, 40), (150, 600)], ids=["covariance", "kernel"])
def test_ridge_on_principal_scores_recovers_a_linear_map(n, d):
    rng = np.random.default_rng(0)
    w = rng.normal(size=(d, 3))
    x = rng.normal(size=(n + 200, d)).astype(np.float32)
    y = x @ w + 0.01 * rng.normal(size=(n + 200, 3))
    fit, val, test = np.arange(n), np.arange(n, n + 100), np.arange(n + 100, n + 200)
    z_fit, (z_val, z_test), s = probe.principal_scores(x, fit, [val, test])
    np.testing.assert_allclose(z_fit.T @ z_fit, np.diag(s), atol=1e-2 * s.max())  # orthogonal scores
    pred, _ = probe.ridge(z_fit, s, y[fit], z_val, y[val], z_test)
    r2 = probe.r2(y[test], pred)
    if n > d:  # determined: recovered
        assert r2.min() > 0.99, r2
    else:      # more weights than frames: ridge gets the part of the map the frames span
        assert r2.min() > 0.1, r2


def test_auroc_is_the_pairwise_count():
    rng = np.random.default_rng(1)
    score = np.round(rng.normal(size=60), 1)  # with ties
    label = (rng.random(60) < 0.4).astype(float)
    pos, neg = score[label == 1], score[label == 0]
    pairs = (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()
    assert probe.auroc(score, label) == pytest.approx(pairs)
