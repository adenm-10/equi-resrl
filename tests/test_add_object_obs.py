"""Tier 1: tools/add_object_obs.py's writing helpers, on a small table of the source's shape.

The tool's guards run on the real demos when it builds (its docstring); this checks the pieces they
rely on, on the CPU with no simulator: the column goes in after observation.state, in the table and in
the Hugging Face features the parquet carries, with every source column untouched; the per-episode
statistics are LeRobot's own; and the feature order follows the column order.
"""

from __future__ import annotations

import json

import numpy as np
import pyarrow as pa

from resfit.dexmg.environments.object_obs import OBJECT_DIM, OBJECT_KEY
from tools import add_object_obs as tool


def _table(n: int = 5) -> pa.Table:
    state = pa.FixedSizeListArray.from_arrays(pa.array(np.arange(9 * n, dtype=np.float32)), 9)
    action = pa.FixedSizeListArray.from_arrays(pa.array(np.ones(7 * n, dtype=np.float32)), 7)
    features = {"action": {"feature": {"dtype": "float32", "_type": "Value"}, "length": 7, "_type": "Sequence"},
                "observation.state": {"feature": {"dtype": "float32", "_type": "Value"}, "length": 9, "_type": "Sequence"},
                "index": {"dtype": "int64", "_type": "Value"}}
    table = pa.table({"action": action, "observation.state": state, "index": pa.array(range(n), pa.int64())})
    return table.replace_schema_metadata({b"huggingface": json.dumps({"info": {"features": features}}).encode()})


def test_the_column_follows_observation_state_and_leaves_the_rest_as_it_was():
    src = _table()
    objects = np.random.default_rng(0).standard_normal((5, OBJECT_DIM)).astype(np.float32)
    new = tool.add_column(src, objects)
    assert new.column_names == ["action", "observation.state", OBJECT_KEY, "index"]
    for name in src.column_names:
        assert new.column(name).equals(src.column(name))
    np.testing.assert_array_equal(np.asarray(new.column(OBJECT_KEY).to_pylist(), np.float32), objects)
    hf = json.loads(new.schema.metadata[b"huggingface"])["info"]["features"]
    assert list(hf) == new.column_names
    assert hf[OBJECT_KEY] == {"feature": {"dtype": "float32", "_type": "Value"}, "length": OBJECT_DIM, "_type": "Sequence"}
    assert new.column_names == tool._after_state_list(src.column_names)


def test_the_statistics_are_lerobots_own():
    objects = np.arange(3 * OBJECT_DIM, dtype=np.float32).reshape(3, OBJECT_DIM)
    stats = tool.episode_stats(objects)
    assert set(stats) == {"min", "max", "mean", "std", "count"}
    np.testing.assert_array_equal(stats["min"], objects[0])
    np.testing.assert_array_equal(stats["max"], objects[2])
    np.testing.assert_allclose(stats["std"], objects.std(0))
    assert stats["count"] == [3]
    json.dumps(stats)  # as episodes_stats.jsonl stores it


def test_the_names_are_the_definitions_layout():
    assert len(tool.OBJECT_NAMES) == OBJECT_DIM
    assert tool.OBJECT_NAMES[:7] == ["nut_pos_0", "nut_pos_1", "nut_pos_2", "nut_quat_0", "nut_quat_1", "nut_quat_2",
                                     "nut_quat_3"]
    assert tool.OBJECT_NAMES[7] == "peg_pos_0"
