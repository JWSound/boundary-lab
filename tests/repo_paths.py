"""The one filesystem anchor for tests that read repository source.

Several tests assert against source text or scan the tree for violations, and
they addressed files as ``Path("src/blab/...")`` — relative to the *current
working directory*, not to the repository.

That is a false-green generator, not merely a portability wart. A test that
reads a file this way fails loudly from the wrong directory, but one that
*globs* a directory gets an empty result, and an empty result satisfies both
``assert offenders == []`` and every ``assert "..." not in source``. Measured
2026-08-07: with real violations planted in the tree,
``test_no_module_derives_its_own_app_root`` and
``test_qt_file_dialog_usage_is_centralized_in_service`` both reported PASS when
run from ``repo/src`` instead of ``repo``.

This mirrors :data:`blab.paths.APP_ROOT`, which exists for the same reason on
the application side. ``test_tests_do_not_address_source_by_relative_path``
keeps it at one anchor.
"""

from __future__ import annotations

from pathlib import Path

#: Repository root. This module is ``tests/repo_paths.py``, so the root is one
#: level up. Test modules must not recompute this.
REPO_ROOT = Path(__file__).resolve().parents[1]

#: Application source root, ``src/blab``.
BLAB_SRC = REPO_ROOT / "src" / "blab"

#: The decomposed main-window package.
MAIN_WINDOW_PKG = BLAB_SRC / "ui" / "main_window"


def source_text(*relative_parts: str) -> str:
    """Read a source file under ``src/blab`` by path parts.

    Raises rather than returning empty when the file is missing, so a rename
    is a loud failure instead of an assertion that silently stops testing
    anything.
    """
    path = BLAB_SRC.joinpath(*relative_parts)
    if not path.is_file():
        raise FileNotFoundError(f"expected source file not found: {path}")
    return path.read_text(encoding="utf-8")
