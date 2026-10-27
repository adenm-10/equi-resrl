"""Tier 2: the task objects' poses, observation.object (O1; resfit/dexmg/environments/object_obs.py).

What the switch adds, checked from outside the code that declares it:
  * the key's 14 numbers are the nut's robosuite observation then the peg's body pose, in one
    convention, and the env returns them, declared, through the vector env;
  * a run is refused where the switch is not built for;
  * the normalizer gives each object its own scale, so the nut's numbers are the same whether or not
    the peg is read, and adds nothing when off;
  * the encoder's object block has the layout docs/EQUIVARIANCE.md states (tests/group_action.py), and
    a raw observation turned as the physics turns it (the gripper's and the objects' positions about the
    base, every quaternion turned, the base action's xy and tilt turned) gives that representation
    applied to the original's: the rotation centre, the quaternion convention, the ordering and the
    field types, end to end. No test checked this for the gripper before O1;
  * the actor stays equivariant and the critic invariant with the objects in;
  * every family builds, updates and logs with them, `rrr` included;
  * the cache keys move only when the switch is on, the same for either object set, and the trainer
    gives the switch to both its envs.

With the switch off nothing changes, bit for bit: the regression goldens (test_regression_agent.py,
test_regression_single_arm.py, test_action_bound.py) pin that.
"""

from __future__ import annotations

import inspect
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from resfit.dexmg.environments.object_obs import (
    OBJECT_DIM,
    OBJECT_KEY,
    OBJECT_SETS,
    check_object_obs,
    objects_read,
    square_object_obs,
)
from tests import group_action as ga
from tests.conftest import BATCH, ENC_HIDDEN, IH_OUT, N, OBS_SHAPE, TOL

BASE = (-0.56, 0.0)  # env_probes/Square.json, the rotation centre
SETS = [objs for objs in OBJECT_SETS.values() if objs]
SET_IDS = ["nut", "nut+peg"]


# ---------------------------------------------------------------------------
# The key
# ---------------------------------------------------------------------------


def test_the_key_is_the_nuts_observation_then_the_pegs_body_pose():
    """Nut position and quaternion as robosuite observed them, then the peg's body position and its
    MuJoCo quaternion (w x y z) reordered to robosuite's x y z w."""
    obs = {"SquareNut_pos": np.array([0.1, 0.2, 0.83]), "SquareNut_quat": np.array([0.0, 0.0, 0.6, 0.8])}
    xpos = np.zeros((5, 3)); xpos[3] = (0.23, 0.10, 0.85)
    xquat = np.zeros((5, 4)); xquat[3] = (0.9, 0.1, 0.2, 0.3)  # w x y z
    sim = SimpleNamespace(data=SimpleNamespace(body_xpos=xpos, body_xquat=xquat))
    got = square_object_obs(obs, sim, peg_body_id=3)
    assert got.dtype == np.float32 and got.shape == (OBJECT_DIM,)
    np.testing.assert_array_equal(got, np.float32([0.1, 0.2, 0.83, 0, 0, 0.6, 0.8, 0.23, 0.10, 0.85, 0.1, 0.2, 0.3, 0.9]))


def test_object_obs_names_its_objects():
    assert objects_read("none") == ()
    assert objects_read("nut") == ("nut",)
    assert objects_read("nut+peg") == ("nut", "peg")
    with pytest.raises(ValueError, match="object_obs"):
        objects_read("peg")


FEATURES = {OBJECT_KEY: {"dtype": "float32", "shape": [OBJECT_DIM]}}
OK = dict(task="Square", equivariant=True, residual=True, n_arms=1, dataset_features=FEATURES)


@pytest.mark.parametrize("change, match", [
    ({"task": "Can"}, "Square only"),
    ({"equivariant": False}, "ResFiT's own agent"),
    ({"residual": False}, "residual run on one arm"),
    ({"n_arms": 2}, "residual run on one arm"),
    ({"dataset_features": {}}, "needs demos"),
    ({"dataset_features": {OBJECT_KEY: {"dtype": "float32", "shape": [7]}}}, "needs demos"),
])
def test_a_run_the_switch_is_not_built_for_is_refused(change, match):
    with pytest.raises(ValueError, match=match):
        check_object_obs("nut", **{**OK, **change})


def test_off_is_never_refused():
    assert check_object_obs("none", task="Can", equivariant=False, residual=False, n_arms=2, dataset_features={}) == ()
    assert check_object_obs("nut+peg", **OK) == ("nut", "peg")


@pytest.mark.gpu
@pytest.mark.slow
def test_the_env_returns_the_key_declared_and_through_the_vector_env():
    """The wrapper's key is the definition's, unbounded in its observation space; without the space
    entry AsyncVectorEnv would drop it with no error. Off, there is no key."""
    import os

    os.environ.setdefault("MUJOCO_GL", "egl")
    from resfit.dexmg.environments.dexmg import RobosuiteGymWrapper, create_vectorized_env
    from resfit.dexmg.environments.object_obs import PEG_BODY

    gpu = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or 0)
    w = RobosuiteGymWrapper("Square", render_gpu_device_id=gpu, object_obs=True)
    try:
        space = w.observation_space[OBJECT_KEY]
        assert space.shape == (OBJECT_DIM,) and np.isneginf(space.low).all() and np.isposinf(space.high).all()
        assert w.env.sim.model.body_id2name(w.env.peg1_body_id) == PEG_BODY
        obs, _ = w.reset()
        raw = w.env._get_observations()
        np.testing.assert_array_equal(obs[OBJECT_KEY][:7], np.concatenate([raw["SquareNut_pos"], raw["SquareNut_quat"]]).astype(np.float32))
        obs, *_ = w.step(np.zeros(w.env.action_dim))
        assert obs[OBJECT_KEY].shape == (OBJECT_DIM,)
    finally:
        w.close()
    off = RobosuiteGymWrapper("Square", render_gpu_device_id=gpu)
    try:
        assert OBJECT_KEY not in off.observation_space.spaces and OBJECT_KEY not in off.reset()[0]
    finally:
        off.close()
    with pytest.raises(ValueError, match="Square only"):
        RobosuiteGymWrapper("Can", render_gpu_device_id=gpu, object_obs=True)
    vec = create_vectorized_env("Square", num_envs=1, debug=False, object_obs=True)
    try:
        obs, _ = vec.reset()
        assert obs[OBJECT_KEY].shape == (1, OBJECT_DIM)
    finally:
        vec.close()


# ---------------------------------------------------------------------------
# The normalizer
# ---------------------------------------------------------------------------


def _stats(n: int) -> dict:
    from tools.agent_fingerprint import _stats as stand_in

    return stand_in(n)


def _normalizer(objects):
    from resfit.rl_finetuning.equi_off_policy.networks.equi_normalizer import build_equivariant_normalizer

    stats = {"observation.state": _stats(9), "action": _stats(7), OBJECT_KEY: _stats(OBJECT_DIM)}
    return build_equivariant_normalizer(stats, robot_base_xy=torch.tensor(BASE), objects=objects)


def test_each_object_has_its_own_scale_and_off_adds_nothing():
    off, nut, both = _normalizer(()), _normalizer(("nut",)), _normalizer(("nut", "peg"))
    assert set(nut.params_dict) - set(off.params_dict) == {"obj_pos_xy_nut", "obj_pos_z_nut", "obj_rot_nut"}
    assert set(both.params_dict) - set(nut.params_dict) == {"obj_pos_xy_peg", "obj_pos_z_peg", "obj_rot_peg"}
    for key in nut.params_dict:
        for stat in ("scale", "offset"):
            assert torch.equal(nut.params_dict[key][stat], both.params_dict[key][stat]), key
    xy = both.params_dict["obj_pos_xy_peg"]
    assert xy["scale"][0] == xy["scale"][1] and not xy["offset"].any(), "one scale for x and y, no offset"
    with pytest.raises(AssertionError):
        _normalizer(("peg",))


# ---------------------------------------------------------------------------
# The encoder: layout, and a raw observation turned by the physics
# ---------------------------------------------------------------------------


def _encoder(objects, encoder="equivariant"):
    from resfit.rl_finetuning.config.rlpd import VitEncoderConfig
    from resfit.rl_finetuning.equi_off_policy.networks.obs_encoder import ResObsEnc

    torch.manual_seed(0)
    enc = ResObsEnc(obs_shape=OBS_SHAPE, N=N, n_hidden=ENC_HIDDEN, ih_out_channel=IH_OUT, encoder=encoder,
                    vit_cfg=VitEncoderConfig() if encoder == "resfit" else None, objects=objects)
    enc.set_normalizer(_normalizer(objects), robot_base_xy=list(BASE))
    return enc.eval()


def test_off_builds_no_object_block():
    enc = _encoder(())
    assert enc.objects == () and enc.obj_type is None and enc.lowdim_type is enc.prop_type
    assert enc.lowdim_dim == enc.prop_dim


@pytest.mark.parametrize("objects", SETS, ids=SET_IDS)
def test_the_object_block_acts_as_the_physics_says(objects):
    enc, n = _encoder(objects), len(objects)
    lowdim = ga.prop_layout() + ga.object_layout(n)
    critic = ga.enc_out_layout_critic(ENC_HIDDEN, IH_OUT, n_objects=n)
    actor = ga.enc_out_layout_actor(ENC_HIDDEN, IH_OUT, n_objects=n)
    assert enc.lowdim_dim == ga.layout_width(lowdim, N) == 11 + 9 * n
    for k, g in enumerate(enc.group.fibergroup.elements):
        for declared, layout in ((enc.lowdim_type, lowdim), (enc.enc_out_type_critic, critic),
                                 (enc.enc_out_type_actor, actor)):
            ours = ga.act_on_layout(torch.eye(ga.layout_width(layout, N), dtype=torch.float64), layout, k, N).numpy().T
            np.testing.assert_allclose(declared.representation(g), ours, atol=1e-6)


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of quaternions stored x y z w, as robosuite stores them."""
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack([aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw,
                        aw * bw - ax * bx - ay * by - az * bz], dim=-1)


def _turn(obs: dict, k: int) -> dict:
    """The scene turned by 2*pi*k/N about the robot base's vertical axis, as the raw observation sees
    it: positions about the base, quaternions left-multiplied by the turn, the base action's xy and
    tilt (axis-angle x, y) turned, every height, gripper reading and z part held."""
    theta = 2 * math.pi * k / N
    c, s = math.cos(theta), math.sin(theta)
    rot = torch.tensor([[c, -s], [s, c]])
    q_turn = torch.tensor([0.0, 0.0, math.sin(theta / 2), math.cos(theta / 2)])
    base = torch.tensor(BASE)

    def position(p):
        return torch.cat([base + (p[:, 0:2] - base) @ rot.T, p[:, 2:3]], dim=1)

    out = dict(obs)
    st = obs["observation.state"]
    out["observation.state"] = torch.cat([position(st[:, 0:3]), _quat_mul(q_turn, st[:, 3:7]), st[:, 7:]], dim=1)
    a = obs["observation.base_action"]
    out["observation.base_action"] = torch.cat([a[:, 0:2] @ rot.T, a[:, 2:3], a[:, 3:5] @ rot.T, a[:, 5:]], dim=1)
    o = obs[OBJECT_KEY]
    out[OBJECT_KEY] = torch.cat([torch.cat([position(o[:, j:j + 3]), _quat_mul(q_turn, o[:, j + 3:j + 7])], dim=1)
                                 for j in (0, 7)], dim=1)
    return out


def _raw_obs(seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)

    def unit(n):
        q = torch.randn(n, 4, generator=g)
        return q / q.norm(dim=1, keepdim=True)

    def position(n):
        return torch.cat([torch.rand(n, 2, generator=g) * 0.4 - 0.2, 0.8 + torch.rand(n, 1, generator=g) * 0.2], dim=1)

    images = {cam: torch.randint(0, 256, (BATCH, *OBS_SHAPE), generator=g, dtype=torch.uint8)
              for cam in ("observation.images.agentview", "observation.images.robot0_eye_in_hand")}
    return {**images,
            "observation.state": torch.cat([position(BATCH), unit(BATCH), torch.rand(BATCH, 2, generator=g) * 0.04], dim=1),
            "observation.base_action": torch.rand(BATCH, 7, generator=g) * 2 - 1,
            OBJECT_KEY: torch.cat([position(BATCH), unit(BATCH), position(BATCH), unit(BATCH)], dim=1)}


@pytest.mark.parametrize("encoder", ["equivariant", "resfit"])
@pytest.mark.parametrize("objects", [(), *SETS], ids=["off", *SET_IDS])
@pytest.mark.parametrize("k", range(N))
def test_a_turned_raw_observation_gives_the_declared_representation(encoder, objects, k):
    """The encoder's low-dimensional blocks, critic's [proprioception | objects] and actor's
    [proprioception | objects | base action], from a raw observation turned as the physics turns it,
    equal the declared representation applied to the original's. The images are not turned: they are
    the same in both, and these blocks do not read them."""
    enc = _encoder(objects, encoder)
    obs = _raw_obs()
    lowdim = ga.prop_layout() + ga.object_layout(len(objects))
    images = enc.vis_ih_type.size
    with torch.no_grad():
        actor, critic = enc(obs)
        actor_t, critic_t = enc(_turn(obs, k))
    as_tensor = (lambda f: f.tensor) if encoder == "equivariant" else (lambda f: f)
    actor, critic, actor_t, critic_t = map(as_tensor, (actor, critic, actor_t, critic_t))
    np.testing.assert_allclose(critic_t[:, -enc.lowdim_dim:], ga.act_on_layout(critic[:, -enc.lowdim_dim:], lowdim, k, N),
                               atol=2e-5)
    layout = lowdim + ga.action_layout()
    np.testing.assert_allclose(actor_t[:, images:], ga.act_on_layout(actor[:, images:], layout, k, N), atol=2e-5)


def test_the_turn_is_a_turn():
    """The test's own physics: two quarter turns are a half turn, and positions keep their distance
    from the base."""
    obs = _raw_obs()
    twice, half = _turn(_turn(obs, 2), 2), _turn(obs, 4)
    for key in ("observation.state", OBJECT_KEY, "observation.base_action"):
        np.testing.assert_allclose(twice[key], half[key], atol=1e-5)
    distance = lambda o: (o[OBJECT_KEY][:, 0:2] - torch.tensor(BASE)).norm(dim=1)  # noqa: E731
    np.testing.assert_allclose(distance(_turn(obs, 1)), distance(obs), atol=1e-6)


# ---------------------------------------------------------------------------
# The networks, with the objects in
# ---------------------------------------------------------------------------


def _rel_err(a, b):
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-12)).item()


@pytest.mark.parametrize("objects", SETS, ids=SET_IDS)
def test_the_actor_stays_equivariant_and_the_critic_invariant(objects):
    """At the feature level, as test_equivariance.py checks them without objects."""
    from escnn import nn as enn

    from tests.conftest import build_actor, build_critic

    enc, n = _encoder(objects), len(objects)
    cpu = torch.device("cpu")
    torch.manual_seed(1)
    actor = build_actor(enc, cpu, prop_type=enc.lowdim_type).eval()
    critic = build_critic(enc, cpu, prop_type=enc.lowdim_type).eval()
    a_layout = ga.enc_out_layout_actor(ENC_HIDDEN, IH_OUT, n_objects=n)
    c_layout = ga.enc_out_layout_critic(ENC_HIDDEN, IH_OUT, n_objects=n)
    g = torch.Generator().manual_seed(2)
    fa = torch.randn(BATCH, ga.layout_width(a_layout, N), generator=g)
    fc = torch.randn(BATCH, ga.layout_width(c_layout, N), generator=g)
    act = torch.randn(BATCH, 7, generator=g)
    with torch.no_grad():
        mu = actor(enn.GeometricTensor(fa, enc.enc_out_type_actor), std=None)
        q = critic(enn.GeometricTensor(fc, enc.enc_out_type_critic), act)
        for k in range(1, N):
            mu_k = actor(enn.GeometricTensor(ga.act_on_layout(fa, a_layout, k, N), enc.enc_out_type_actor), std=None)
            assert _rel_err(mu_k, ga.act_on_layout(mu, ga.action_layout(), k, N)) < TOL, k
            q_k = critic(enn.GeometricTensor(ga.act_on_layout(fc, c_layout, k, N), enc.enc_out_type_critic),
                         ga.act_on_layout(act, ga.action_layout(), k, N))
            assert _rel_err(q_k, q) < TOL, k
        # The objects reach both: change them alone and both outputs move.
        lo = slice(-9 * n, None)
        fc2 = fc.clone(); fc2[:, lo] += 1.0
        assert not torch.allclose(critic(enn.GeometricTensor(fc2, enc.enc_out_type_critic), act), q)
        fa2 = fa.clone(); fa2[:, -7 - 9 * n:-7] += 1.0
        assert not torch.allclose(actor(enn.GeometricTensor(fa2, enc.enc_out_type_actor), std=None), mu)


FAMILY_MIXES = {"eee": ("equivariant",) * 3, "aaa": ("ablated",) * 3, "ere": ("equivariant", "resfit", "equivariant"),
                "rrr": ("resfit",) * 3}


def _agent(families, objects):
    """The agent as the trainer builds it with the objects (build_agent), at Square's config."""
    from resfit.rl_finetuning.config.residual_td3 import ResidualEquiTD3SquareConfig
    from resfit.rl_finetuning.equi_off_policy.rl.q_agent import QAgent
    from tools.agent_fingerprint import ACTION_DIM, CAMERAS, STATE_DIM

    cfg = ResidualEquiTD3SquareConfig()
    cfg.agent.device = "cpu"
    cfg.agent.critic.num_q = 2
    for part, family in zip(("encoder", "actor", "critic"), families):
        setattr(cfg.equivariance, f"{part}_family", family)
    torch.manual_seed(0)
    agent = QAgent(obs_shape=OBS_SHAPE, prop_shape=(STATE_DIM,), action_dim=ACTION_DIM, rl_cameras=list(CAMERAS),
                   agent_cfg=cfg.agent, equi_cfg=cfg.equivariance, residual_actor=True, objects=objects)
    agent.enc.set_normalizer(_normalizer(objects), robot_base_xy=list(BASE))
    return agent


@pytest.mark.slow
@pytest.mark.parametrize("mix", FAMILY_MIXES)
def test_every_family_updates_and_logs_with_the_objects(mix):
    """Each mix builds with the objects, acts, takes an update and logs its critic diagnostics, whose
    hand-written key list must carry observation.object (else the first log step crashes). The objects
    reach the networks' features."""
    from resfit.rl_finetuning.off_policy.common_utils import utils
    from tools.agent_fingerprint import STDDEV, fixed_batch

    agent = _agent(FAMILY_MIXES[mix], ("nut", "peg"))
    assert agent.enc.lowdim_dim == 29
    obs = dict(fixed_batch(objects=True)["obs"].items())
    moved = {**obs, OBJECT_KEY: obs[OBJECT_KEY] + 1.0}
    with utils.eval_mode(agent):
        assert agent.act(obs, eval_mode=True, cpu=True).shape == (BATCH, 7)
        assert "debug/q_data_online" in agent.critic_diagnostics(fixed_batch(objects=True), 2)
        with torch.no_grad():
            feats = [f.tensor if hasattr(f, "tensor") else f for f in (*agent.enc(obs), *agent.enc(moved))]
    assert not torch.equal(feats[0], feats[2]) and not torch.equal(feats[1], feats[3])
    metrics = agent.update(fixed_batch(objects=True), STDDEV, update_actor=True)
    assert np.isfinite(metrics["train/critic_loss"])


# ---------------------------------------------------------------------------
# The trainer and the gate
# ---------------------------------------------------------------------------


def test_the_cache_keys_move_only_when_on_and_alike_for_either_set():
    import preflight

    off = preflight.cache_hashes("Square", dataset_name="local/robomimic-mh-square-rerendered")
    assert preflight.cache_hashes("Square", dataset_name="local/robomimic-mh-square-rerendered",
                                  object_obs="none") == off
    nut = preflight.cache_hashes("Square", dataset_name="local/robomimic-mh-square-rerendered", object_obs="nut")
    both = preflight.cache_hashes("Square", dataset_name="local/robomimic-mh-square-rerendered", object_obs="nut+peg")
    assert nut == both and nut[0] != off[0] and nut[1] != off[1]


def test_the_trainer_keys_the_caches_and_threads_the_switch_into_both_envs():
    """The trainer's own source: one env factory, called for the training and the eval env, passes the
    switch; both cache keys gain it only when on; the buffers store the key; the agent and its
    normalizer get the objects."""
    from resfit.rl_finetuning.scripts import train_residual_td3 as t

    main, build = inspect.getsource(t.main), inspect.getsource(t.build_agent)
    assert main.count("create_vectorized_env(") == 1 and "object_obs=bool(objects)" in main
    assert main.count("env = get_envs(") == 2  # env and eval_env
    assert 'online_cache_meta["object_obs"] = True' in main and 'offline_cache_meta["object_obs"] = True' in main
    assert "lowdim_keys.append(OBJECT_KEY)" in main and "curr_obs[OBJECT_KEY] = sample[OBJECT_KEY]" in main
    assert "check_object_obs(cfg.object_obs" in main
    assert build.count("objects=objects") == 2


def test_the_gate_reads_object_obs_from_the_composed_config():
    import re

    import preflight

    src = inspect.getsource(preflight.tier1_compose)
    assert 'flags["object_obs"]' in src
    assert re.search(r"^object_obs: (\S+)$", "seed: 2\nobject_obs: nut+peg\n", flags=re.M).group(1) == "nut+peg"
