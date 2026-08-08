"""Guards on the test suite itself.

Both rules here exist because a test that quietly stops testing anything still
reports success, which is worse than no test at all. Deliberately free of any
Qt dependency, so these keep running in the degraded environment that
``BLAB_SKIP_GUI_TESTS=1`` describes — the one where a missing GUI stack has
already removed most of the suite.
"""

from __future__ import annotations

import ast
from pathlib import Path

from gui_stack import PYSIDE6_MODULE, SKIP_GUI_ENV
from repo_paths import REPO_ROOT

TESTS_DIR = Path(__file__).resolve().parent

#: Directory names that, handed to ``Path`` as the head of a literal, mean the
#: path was written relative to the working directory rather than the repo.
REPO_RELATIVE_ROOTS = {"src", "tests", "examples", "assets", "docs", "scripts"}


def _test_modules() -> list[Path]:
    modules = sorted(TESTS_DIR.glob("test_*.py"))
    assert modules, f"no test modules found under {TESTS_DIR} - the scan is vacuous"
    return modules


def test_tests_do_not_address_source_by_relative_path() -> None:
    """Repo-relative paths in tests are a false-green generator.

    Such a path resolves against the working directory. *Reading* one fails
    loudly from the wrong directory, but *globbing* one returns empty - and an
    empty result satisfies both ``assert offenders == []`` and every negative
    substring assertion. Measured 2026-08-07: with real violations planted in
    the tree, two guard tests reported PASS when run from ``repo/src``.
    Everything goes through tests/repo_paths.py now.

    Matched on the parse tree rather than the source text, so a docstring that
    describes the antipattern does not count as committing it.
    """
    offenders = []
    for path in _test_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Path"):
                continue
            first = node.args[0] if node.args else None
            if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
                continue
            if first.value.replace("\\", "/").split("/")[0] in REPO_RELATIVE_ROOTS:
                offenders.append(f"{path.name}:{node.lineno}")

    assert offenders == [], f"these must anchor paths via tests/repo_paths.py: {sorted(offenders)}"
    assert (REPO_ROOT / "pyproject.toml").is_file(), f"REPO_ROOT resolved to {REPO_ROOT}"


def test_the_gui_suite_cannot_silently_skip_itself() -> None:
    """A broken GUI stack must stop the run, not shrink it.

    ``importorskip`` gated 154 tests - about a third of the suite - behind an
    import that can fail for reasons that are environment faults rather than
    reasons the tests do not apply. A bad wheel would have turned all of them
    into skips and still reported a green run. The gate lives in
    tests/gui_stack.py now: a hard error by default, downgraded to an explicit
    collection-ignore only when somebody sets the opt-out.
    """
    offenders = []
    for path in _test_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            if node.func.attr != "importorskip":
                continue
            first = node.args[0] if node.args else None
            if isinstance(first, ast.Constant) and first.value == PYSIDE6_MODULE:
                offenders.append(f"{path.name}:{node.lineno}")

    assert offenders == [], (
        f"the GUI stack must not be importorskip-gated; it is resolved once in "
        f"tests/gui_stack.py, with {SKIP_GUI_ENV} as the deliberate opt-out: {sorted(offenders)}"
    )
