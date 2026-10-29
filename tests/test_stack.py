"""Stack, the new task (TODO B25): robosuite's own `Stack`, the scene of MimicGen's Stack_D0.

The wrapper builds it as a single-arm Panda task (agentview and wrist cameras, the 9-number state) with
a 300-step horizon, the cubes placed within +-8 cm of the table's middle at any yaw, and a stack ends the
episode as a success. None of it touches Square: the goldens pin that.
"""

from __future__ import annotations

import os

import numpy as np
import pytest


def test_stack_is_a_single_arm_panda_task():
    from resfit.dexmg.environments.dexmg import ENV_ROBOTS

    assert ENV_ROBOTS["Stack"] == ["Panda"]


@pytest.mark.gpu
@pytest.mark.slow
def test_the_stack_env_observes_places_and_succeeds_as_square_does():
    os.environ.setdefault("MUJOCO_GL", "egl")
    from robosuite.utils.placement_samplers import UniformRandomSampler

    from resfit.dexmg.environments.dexmg import RobosuiteGymWrapper

    gpu = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or 0)
    w = RobosuiteGymWrapper("Stack", render_gpu_device_id=gpu)
    try:
        env = w.env
        assert type(env).__name__ == "Stack" and w.horizon == 300

        # The observation: Square's single-arm layout.
        obs, _ = w.reset()
        assert set(obs) == {"observation.state", "observation.images.agentview",
                            "observation.images.robot0_eye_in_hand"}
        assert obs["observation.state"].shape == (9,)
        assert obs["observation.images.agentview"].shape == (3, 84, 84)

        # The placement: MimicGen's Stack_D0 bounds, any yaw, about the table's middle.
        sampler = env.placement_initializer
        assert isinstance(sampler, UniformRandomSampler)
        assert list(sampler.x_range) == [-0.08, 0.08] and list(sampler.y_range) == [-0.08, 0.08]
        assert sampler.rotation is None
        middle = np.asarray(env.table_offset[:2])
        for _ in range(5):
            w.reset()
            for body in (env.cubeA_body_id, env.cubeB_body_id):
                assert np.abs(env.sim.data.body_xpos[body][:2] - middle).max() <= 0.08 + 1e-6

        # A stack: cube A set on cube B, out of the gripper, ends the episode as a success.
        assert not env._check_success()
        qpos = env.sim.data.qpos
        a = env.sim.model.get_joint_qpos_addr("cubeA_joint0")[0]
        b = env.sim.model.get_joint_qpos_addr("cubeB_joint0")[0]
        qpos[a:a + 3] = qpos[b:b + 3] + np.array([0.0, 0.0, 0.025 + 0.02 - 0.001])  # B's and A's half-sizes
        qpos[a + 3:a + 7] = qpos[b + 3:b + 7]
        env.sim.data.qvel[:] = 0.0
        env.sim.forward()
        _, reward, terminated, truncated, info = w.step(np.zeros(env.action_dim))
        assert reward == 1.0 and terminated and not truncated and info["success"]
    finally:
        w.close()


def test_the_demo_builders_replay_gap_reads_a_quaternion_up_to_sign():
    """tools/build_stack_demos.py's replay guard: q and -q are one rotation; a moved gripper is a gap."""
    from tools.build_stack_demos import replay_gap

    ours = np.array([[0.1, 0.2, 0.9, 0.0, 0.0, 0.6, 0.8, 0.02, -0.02]], dtype=np.float32)
    flipped = ours.copy()
    flipped[:, 3:7] *= -1
    assert replay_gap(ours, flipped) == (0.0, 0.0, 0.0)
    moved = ours.copy()
    moved[:, 0] += 0.003
    moved[:, 7] += 0.001
    pos, quat, grip = replay_gap(ours, moved)
    assert pos == pytest.approx(0.003, abs=1e-7) and quat == 0.0 and grip == pytest.approx(0.001, abs=1e-7)
