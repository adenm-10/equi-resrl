"""Tier 1: every module under resfit/ must import.

Cheap, no GPU, seconds. This is the test that would have caught rl_utils.py's
`from __future__ import annotations` on line 15 -- a hard SyntaxError -- on the
day it was written instead of five months later. See STANDARDS.md rule 4.2.
"""

from __future__ import annotations

import importlib
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent

# Modules that pull in heavy optional deps or need a live MuJoCo context.
# Keep this list short and justified; it is an admission of a gap, not a default.
SKIP_SUBSTRINGS = (
    "resfit.dexmg.environments.dexmg",       # constructs robosuite envs on import path
    "resfit.lerobot.scripts.train_bc_dexmg", # calls parse_args() at module level, see below
)


def _module_names() -> list[str]:
    names = []
    for path in sorted((REPO / "resfit").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(REPO).with_suffix("")
        parts = list(rel.parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        names.append(".".join(parts))
    return names


ALL_MODULES = _module_names()


def test_found_modules():
    assert len(ALL_MODULES) > 30, f"only found {len(ALL_MODULES)} modules; glob is wrong"


@pytest.mark.parametrize("name", ALL_MODULES)
def test_module_imports(name):
    if any(s in name for s in SKIP_SUBSTRINGS):
        pytest.skip("needs a live simulator context")
    importlib.import_module(name)


def test_no_future_import_after_first_statement():
    """`from __future__` must be the first statement or it is a SyntaxError.

    Compiling catches it, but this gives a targeted message instead of a
    traceback from deep inside an import chain.
    """
    offenders = []
    for path in sorted((REPO / "resfit").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        src = path.read_text(errors="replace")
        if "from __future__" not in src:
            continue
        try:
            compile(src, str(path), "exec")
        except SyntaxError as exc:  # pragma: no cover - only on regression
            offenders.append(f"{path.relative_to(REPO)}:{exc.lineno}: {exc.msg}")
    assert not offenders, "files that will not compile:\n  " + "\n  ".join(offenders)


def test_bc_script_parses_args_at_import_time():
    """Documents a known defect rather than silently skipping it.

    train_bc_dexmg.py calls `args_cli = parser.parse_args()` at module level
    (line 217), outside the `if __name__ == "__main__"` guard at line 905. So
    importing the module tries to parse *pytest's* argv and exits. It is the
    stage-1 BC script and not on the reproduction path, so it is skipped above
    rather than fixed under the freeze.

    When it gets fixed (move the parse into main), this test fails and the skip
    entry above should be removed in the same change.
    """
    src = (REPO / "resfit" / "lerobot" / "scripts" / "train_bc_dexmg.py").read_text()
    guard = src.index('if __name__ == "__main__"')
    parse = src.index("args_cli = parser.parse_args()")
    assert parse < guard, (
        "train_bc_dexmg.py no longer parses args at import time. Remove it from "
        "SKIP_SUBSTRINGS and delete this test."
    )
