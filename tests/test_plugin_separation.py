"""Guards the separation between the main package and the bundled Generator Plugins.

Plugins live entirely in their folder under ``geometry_providers``. The core refers to them only by plugin ID
(allowlist, enable-once setting); it never imports their code and never registers a plugin command in ``blab``.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

from blab import cli

REPO = Path(__file__).resolve().parents[1]
MAIN_PACKAGE = REPO / "src" / "blab"
BUNDLED_PLUGINS = ("cad_import", "vented_enclosure")


def _forbidden() -> set[str]:
    """Dotted names that reach the code of a bundled plugin, by any module path it could be mounted under."""
    names = {"geometry_providers"}
    for folder in BUNDLED_PLUGINS:
        names |= {folder, folder.replace("_", ""), f"blab.{folder}"}
    return names


def _imported_names(tree: ast.AST) -> list[str]:
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            found.append(base)
            found += [f"{base}.{alias.name}".lstrip(".") for alias in node.names]
        elif isinstance(node, ast.Call):
            callee = ast.unparse(node.func)
            if callee.endswith(("import_module", "__import__")) and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    found.append(first.value)
    return found


def _reaches_plugin(name: str) -> bool:
    parts = name.split(".")
    forbidden = _forbidden()
    return any(".".join(parts[:count]) in forbidden for count in range(1, len(parts) + 1))


def main_package_plugin_imports() -> list[str]:
    offenders = []
    for path in sorted(MAIN_PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        offenders += [
            f"{path.relative_to(REPO).as_posix()} imports {name}"
            for name in _imported_names(tree)
            if _reaches_plugin(name)
        ]
    return offenders


def test_no_module_of_the_main_package_imports_bundled_plugin_code():
    assert main_package_plugin_imports() == []


def test_the_scan_recognizes_plugin_imports():
    for source in (
        "import blab.cad_import",
        "from blab.cad_import import pipeline",
        "from blab import cad_import",
        "from blab.vented_enclosure.design import X",
        "import importlib\nimportlib.import_module('blab.cad_import.sizing')",
        "from geometry_providers.cad_import import pipeline",
    ):
        assert any(_reaches_plugin(name) for name in _imported_names(ast.parse(source))), source
    for source in ("import blab.cli", "from blab.generators import catalog", "from . import cli"):
        assert not any(_reaches_plugin(name) for name in _imported_names(ast.parse(source))), source


def test_core_command_list_holds_only_generic_commands():
    words = {w for folder in BUNDLED_PLUGINS for w in (folder, folder.replace("_", "-"), folder.replace("_", ""))}
    for command, module in cli.COMMAND_MODULES.items():
        assert not any(word in command or word in module for word in words), command
        spec = importlib.util.find_spec(module)
        assert spec is not None and spec.origin is not None, command
        assert Path(spec.origin).resolve().is_relative_to(MAIN_PACKAGE), command
