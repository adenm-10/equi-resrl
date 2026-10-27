"""The task objects' poses, `observation.object`: what the env returns and the demos store when the
run's `object_obs` is on (EXPERIMENTS.md, the 2026-10-27 evening DECISION and O1's CODE entry).

The object-centric paper's shape (docs/papers/obj_cent_resrl.pdf, eq. 2 and 4): each task object's
world position (3, m) and world orientation as a quaternion (4), 7 numbers an object, concatenated.
Square has two: the nut, which the robot carries, and the square peg (`peg1`), which is fixed to the
table and is the nut's target. The env and the demos always store both, the nut's then the peg's, 14
numbers; `object_obs` picks which the networks read, the nut alone (the first 7) or both.

The nut's are robosuite's own observations `SquareNut_pos` and `SquareNut_quat` (x, y, z, w), so
they come from the same observation pass as observation.state, with its one-substep lag
(tools/rerender_demos.py). robosuite has no peg sensor, so the peg's are read from its body in the
simulator, in the nut's convention: MuJoCo stores a body's quaternion w first.

This module is the one definition: the env (dexmg.py) and the demo builder
(tools/add_object_obs.py) both call `square_object_obs`, and the encoder slices by `OBJECTS` and
`OBJECT_STRIDE`. It imports nothing heavy, so the encoder can read its constants.
"""

from __future__ import annotations

import numpy as np

OBJECT_KEY = "observation.object"
OBJECTS = ("nut", "peg")  # their order in OBJECT_KEY
OBJECT_STRIDE = 7  # position 3, quaternion 4 (x, y, z, w)
OBJECT_DIM = len(OBJECTS) * OBJECT_STRIDE
# object_obs, the run's config field: which objects the networks read.
OBJECT_SETS = {"none": (), "nut": ("nut",), "nut+peg": ("nut", "peg")}
SQUARE_TASKS = ("Square", "NutAssemblySquare")  # the config's name and robosuite's
NUT = "SquareNut"
PEG_BODY = "peg1"  # the square peg, SquareNut's target (robosuite's nut_assembly.py)


def objects_read(object_obs: str) -> tuple[str, ...]:
    """The objects a run's networks read, by its object_obs."""
    if object_obs not in OBJECT_SETS:
        raise ValueError(f"object_obs={object_obs!r}: expected one of {tuple(OBJECT_SETS)}")
    return OBJECT_SETS[object_obs]


def square_object_obs(obs: dict, sim, peg_body_id: int) -> np.ndarray:
    """OBJECT_KEY's 14 numbers from one robosuite observation of Square and its simulator: the nut's
    position and quaternion as robosuite observed them, then the peg's body pose, as float32."""
    peg_quat_wxyz = np.asarray(sim.data.body_xquat[peg_body_id])
    return np.concatenate([
        obs[f"{NUT}_pos"], obs[f"{NUT}_quat"],
        sim.data.body_xpos[peg_body_id], peg_quat_wxyz[[1, 2, 3, 0]],
    ]).astype(np.float32)


def check_object_obs(object_obs: str, *, task: str, equivariant: bool, residual: bool, n_arms: int,
                     dataset_features: dict) -> tuple[str, ...]:
    """Refuse, at startup, a run this switch is not built for; return the objects its networks read.

    On, it needs Square (the only task with an object definition), this repo's agent (ResFiT's own
    has no input for it), a residual run on one arm (O1's arms; nothing else is tested), and demos that
    store OBJECT_KEY (tools/add_object_obs.py), at the width the env returns."""
    objects = objects_read(object_obs)
    if not objects:
        return objects
    if task not in SQUARE_TASKS:
        raise ValueError(f"object_obs={object_obs!r} is defined for Square only, not {task!r}")
    if not equivariant:
        raise ValueError("object_obs needs this repo's agent (equivariance set): ResFiT's own agent "
                         "does not read observation.object")
    if not residual or n_arms != 1:
        raise ValueError("object_obs is built for a residual run on one arm")
    feature = dataset_features.get(OBJECT_KEY)
    if feature is None or tuple(feature["shape"]) != (OBJECT_DIM,):
        raise ValueError(f"object_obs={object_obs!r} needs demos that store {OBJECT_KEY} ({OBJECT_DIM} "
                         f"numbers; tools/add_object_obs.py); the dataset has {feature}")
    return objects
