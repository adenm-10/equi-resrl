"""The hand's heading frame (equivariance.action_frame=heading; EXPERIMENTS.md, 2026-11-01 CODE, the wrist
sweep; docs/EQUIVARIANCE.md, "The hand's frame").

psi is the heading of the gripper's x axis in the table plane, from its rotation R: column 0 of R,
(R00, R10), normalized. R_psi is the turn by psi about the vertical. In the heading frame every world
vector the networks read or write is turned by R_psi^T: the base action's and the residual's xy and
rx_ry pairs, and the gripper's rotation, R_psi^T R; z, rz and the hand are unchanged, as is gravity.

Turning the scene and the hand together about the hand by any angle then leaves every input of a
residual without the agentview camera unchanged (the wrist image turns with the hand, and the height,
opening and tilt do not move), so any network reading them is symmetric under that turn by
construction. The networks after it are ResFiT's, unconstrained (q_agent.py, check_families).

The encoder appends (cos psi, sin psi) to the actor's and the critic's features. The wrappers here
strip them and turn actions between the frames, so every caller keeps passing and receiving world-frame
actions: the buffer, combine_actions, the env and every call site of the agent are unchanged. The
actor's bound (a disk per pair) and its noise (isotropic per pair) commute with the turn, so the turned
action is bounded and explored as before.
"""

from __future__ import annotations

import torch

from resfit.rl_finetuning.equi_off_policy.rl.equi_rl_utils import feature_tensor


def heading_of(rot6d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(cos psi, sin psi), each [B, 1], from the 6D rotation [R00, R01, R02, R10, R11, R12] (ResObsEnc's
    get6DRotation, R's first two rows). Defined unless the gripper's x axis is vertical."""
    x, y = rot6d[:, 0:1], rot6d[:, 3:4]
    r = torch.sqrt(x * x + y * y)
    return x / r, y / r


def turn_pair(v: torch.Tensor, c: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """2-vectors [B, 2] turned by the angle whose cosine and sine are c and s."""
    return torch.cat([c * v[:, 0:1] - s * v[:, 1:2], s * v[:, 0:1] + c * v[:, 1:2]], dim=1)


def turn_action(a: torch.Tensor, c: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """One arm's action [xy 2, z, rx_ry 2, rz, hand] with its xy and rx_ry pairs turned by (c, s)."""
    return torch.cat([turn_pair(a[:, 0:2], c, s), a[:, 2:3], turn_pair(a[:, 3:5], c, s), a[:, 5:]], dim=1)


def turn_rot6d(rot6d: torch.Tensor, c: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """R_psi^T R as its first two rows, from R's: row 0 c row0 + s row1, row 1 -s row0 + c row1."""
    row0, row1 = rot6d[:, 0:3], rot6d[:, 3:6]
    return torch.cat([c * row0 + s * row1, -s * row0 + c * row1], dim=1)


class HeadingFrameActor(torch.nn.Module):
    """The actor in the heading frame: reads the features less their last two numbers, (cos psi, sin
    psi), and turns its action by R_psi back to the world, after its bound and noise."""

    def __init__(self, actor: torch.nn.Module):
        super().__init__()
        self.actor = actor
        self.action_layout = actor.action_layout

    def forward(self, feat, std=None) -> torch.Tensor:
        x = feature_tensor(feat)
        return turn_action(self.actor(x[:, :-2], std), x[:, -2:-1], x[:, -1:])


class HeadingFrameCritic(torch.nn.Module):
    """The critic in the heading frame: reads the features less (cos psi, sin psi), and the world-frame
    action turned by R_psi^T."""

    def __init__(self, critic: torch.nn.Module):
        super().__init__()
        self.critic = critic

    def _split(self, feat, act):
        x = feature_tensor(feat)
        return x[:, :-2], turn_action(act, x[:, -2:-1], -x[:, -1:])

    def forward(self, feat, act: torch.Tensor) -> torch.Tensor:
        return self.critic(*self._split(feat, act))

    def q_value(self, feat, act: torch.Tensor) -> torch.Tensor:
        return self.critic.q_value(*self._split(feat, act))

    def q_value_for_policy(self, feat, act: torch.Tensor) -> torch.Tensor:
        return self.critic.q_value_for_policy(*self._split(feat, act))
