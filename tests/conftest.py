"""Shared fixtures for the pre-submission test gate.

Modules are built at the configs the reproduction actually uses, so the gate
tests the thing that will run rather than a toy.
"""

from __future__ import annotations

import pytest
import torch

# Config used by the Can reproduction. Kept small where it does not change
# behaviour (enc_degree_channel) so the gate stays fast on CPU.
N = 8
ENC_HIDDEN = 16          # equivariance.enc_degree_channel
ACTOR_HIDDEN = 128       # equivariance.actor_degree_channel
CRITIC_HIDDEN = 128      # equivariance.critic_degree_channel
OBS_SHAPE = (3, 84, 84)
CROP = 76
BATCH = 4
ACTION_SCALE = 0.2

TOL = 1e-4               # see docs/EQUIVARIANCE.md, measured tolerances

# Per-arm observation layouts. TWO_ARM is TwoArmBoxCleanup, measured from the env:
# state is [eef_pos 3, eef_quat 4, gripper_qpos 12] x 2, action [delta_pose 6, hand 6] x 2.
ONE_ARM = {"n_arms": 1, "gripper_dim": 2, "hand_dof": 1}
TWO_ARM = {"n_arms": 2, "gripper_dim": 12, "hand_dof": 6}


def wrist_keys(n_arms: int) -> tuple[str, ...]:
    return tuple(f"observation.images.robot{i}_eye_in_hand" for i in range(n_arms))


def pytest_addoption(parser):
    parser.addoption(
        "--gpu", action="store_true", default=False,
        help="Run the tiers that need a GPU (construction + equivariance).",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires CUDA")
    config.addinivalue_line("markers", "slow: takes more than a few seconds")


def pytest_collection_modifyitems(config, items):
    # The single-arm regression module builds on CPU against CPU-captured goldens.
    # escnn caches basis tensors per representation, so once a GPU module exists
    # those caches live on cuda and a later CPU build dies on a device mismatch.
    # Sorting is stable, so everything else keeps its collection order.
    items.sort(key=lambda i: 0 if "test_regression_single_arm" in str(i.fspath) else 1)

    if config.getoption("--gpu"):
        return
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="needs CUDA; pass --gpu once a GPU is free")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session", params=[ONE_ARM, TWO_ARM], ids=["1arm", "2arm"])
def arm_spec(request):
    """Every equivariance and layout test runs for both arm counts."""
    return request.param


@pytest.fixture(scope="session")
def obs_enc(device, arm_spec):
    """The unified observation encoder, equivariant branch, in eval mode.

    eval() matters: it makes CropRandomizer deterministic and switches the
    actor's batch norms onto running statistics. Random cropping is a random
    translation, which does not commute with rotation, so it would inject
    equivariance error on every forward pass.
    """
    from resfit.rl_finetuning.equi_off_policy.networks.obs_encoder import ResObsEnc

    enc = ResObsEnc(
        obs_shape=OBS_SHAPE, crop_shape=(CROP, CROP), N=N, n_hidden=ENC_HIDDEN,
        equivariant=True, initialize=True,
        wrist_keys=wrist_keys(arm_spec["n_arms"]), **arm_spec,
    ).to(device)
    enc.eval()
    return enc


@pytest.fixture(scope="session")
def vision_encoder(device):
    """The C_N-equivariant agentview backbone, tested directly.

    Bypasses ResObsEnc so no cropping is involved.
    """
    from resfit.rl_finetuning.equi_off_policy.networks.equi_encoder import (
        EquivariantResEncoder76Cyclic,
    )

    enc = EquivariantResEncoder76Cyclic(
        obs_channel=OBS_SHAPE[0], n_out=ENC_HIDDEN, N=N, initialize=True, use_norms=True,
    ).to(device)
    enc.eval()
    return enc


@pytest.fixture(scope="session")
def critic(obs_enc, device):
    from resfit.rl_finetuning.config.rlpd import CriticLossCfg
    from resfit.rl_finetuning.equi_off_policy.rl.critic import Critic

    c = Critic(
        group=obs_enc.group,
        vis_ih_type=obs_enc.vis_ih_type,
        prop_type=obs_enc.prop_type,
        action_type=obs_enc.action_type,
        hidden_dim=CRITIC_HIDDEN,
        num_layers=2,
        num_q=2,
        dropout=0.0,
        use_norms=True,
        use_orth_init=True,
        loss_cfg=CriticLossCfg(),
        min_q_heads=2,
        policy_gradient_type="ensemble_mean",
    ).to(device)
    c.eval()
    return c


@pytest.fixture(scope="session")
def actor(obs_enc, device):
    from resfit.rl_finetuning.equi_off_policy.rl.actor import Actor

    a = Actor(
        group=obs_enc.group,
        vis_ih_type=obs_enc.vis_ih_type,
        prop_type=obs_enc.prop_type,
        action_type=obs_enc.action_type,
        action_layout=obs_enc.action_layout,
        hidden_dim=ACTOR_HIDDEN,
        action_shape=obs_enc.action_shape,
        num_layers=2,
        dropout=0.0,
        use_norms=True,
        use_orth_init=True,
        last_layer_scale=None,   # None, not 0.0: a zeroed final layer makes
                                 # every equivariance test pass trivially
        action_scale=ACTION_SCALE,
        residual_actor=True,
    ).to(device)
    a.eval()
    return a


@pytest.fixture
def rng():
    return torch.Generator().manual_seed(0)
