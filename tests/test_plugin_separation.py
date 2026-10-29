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


PLUGIN_WORDS = {
    word for folder in BUNDLED_PLUGINS for word in (folder, folder.replace("_", ""), folder.replace("_", "-"))
} | {"geometry_providers"}


def _is_plugin_segment(segment: str) -> bool:
    return segment in PLUGIN_WORDS


def _string_parts(node: ast.AST) -> list[str]:
    """Literal text of a string expression (constant, f-string or concatenation); dynamic parts are skipped."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.JoinedStr | ast.BinOp | ast.Call):
        return [part.value for part in ast.walk(node) if isinstance(part, ast.Constant) and isinstance(part.value, str)]
    return []


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
                found += [piece for part in _string_parts(node.args[0]) for piece in [part]]
    return found


def _reaches_plugin(name: str) -> bool:
    """True when a plugin name appears in any dotted segment of an imported name."""
    return any(_is_plugin_segment(segment) for segment in name.replace("/", ".").split("."))


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
        "from blab.generators import cad_import",
        "import blab.generators.cad_import",
        "import importlib\nimportlib.import_module(f'blab.generators.{x}.cad_import')",
        "import importlib\nimportlib.import_module('blab.generators.' + 'cad_import')",
        "import importlib\nimportlib.import_module('blab.' + name + '.vented_enclosure.design')",
        "__import__('blab.generators.cad_import')",
    ):
        assert any(_reaches_plugin(name) for name in _imported_names(ast.parse(source))), source
    for source in ("import blab.cli", "from blab.generators import catalog", "from . import cli"):
        assert not any(_reaches_plugin(name) for name in _imported_names(ast.parse(source))), source


def test_the_plugin_id_string_outside_an_import_call_is_allowed():
    source = 'IDS = {"cad_import": "blab.cad_import"}'
    assert not any(_reaches_plugin(name) for name in _imported_names(ast.parse(source)))


def test_core_command_list_holds_only_generic_commands():
    words = {w for folder in BUNDLED_PLUGINS for w in (folder, folder.replace("_", "-"), folder.replace("_", ""))}
    for command, module in cli.COMMAND_MODULES.items():
        assert not any(word in command or word in module for word in words), command
        spec = importlib.util.find_spec(module)
        assert spec is not None and spec.origin is not None, command
        assert Path(spec.origin).resolve().is_relative_to(MAIN_PACKAGE), command
