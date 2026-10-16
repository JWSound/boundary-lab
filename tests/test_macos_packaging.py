import json
import runpy
import struct
from pathlib import Path

import pytest


@pytest.fixture
def packaging_tools(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    return runpy.run_path(str(scripts / "build_macos_installer.py"))


def test_signing_inventory_rejects_modified_and_added_files(tmp_path, packaging_tools):
    (tmp_path / "payload").write_text("original")
    files, links = packaging_tools["inventory"](tmp_path)
    manifest = {"files": files, "symlinks": links}
    (tmp_path / "runtime-manifest.json").write_text(json.dumps(manifest))
    packaging_tools["verify_resources"](tmp_path, manifest)
    (tmp_path / "payload").write_text("tampered")
    with pytest.raises(RuntimeError, match="inventory"):
        packaging_tools["verify_resources"](tmp_path, manifest)
    (tmp_path / "payload").write_text("original")
    (tmp_path / "extra").touch()
    with pytest.raises(RuntimeError, match="inventory"):
        packaging_tools["verify_resources"](tmp_path, manifest)


def test_native_detection_excludes_debug_and_objects(tmp_path, packaging_tools):
    file = tmp_path / "native"
    for kind in (1, 2, 6, 8, 10):
        file.write_bytes(b"\xcf\xfa\xed\xfe" + b"\0" * 8 + struct.pack("<I", kind) + b"\0" * 16)
        assert packaging_tools["macho_type"](file) == kind
    file.write_text("not native code")
    assert packaging_tools["macho_type"](file) is None


def test_nested_helper_is_sealed_before_framework_binary(tmp_path, packaging_tools):
    framework = tmp_path / "QtWebEngineCore.framework"
    helper = framework / "Versions/A/Helpers/QtWebEngineProcess.app"
    helper_binary = helper / "Contents/MacOS/QtWebEngineProcess"
    framework_binary = framework / "Versions/A/QtWebEngineCore"
    for binary in (helper_binary, framework_binary):
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.write_bytes(b"\xcf\xfa\xed\xfe" + b"\0" * 8 + struct.pack("<I", 2) + b"\0" * 16)
    native, order = packaging_tools["signing_order"](tmp_path)
    assert set(native) == {helper_binary, framework_binary}
    assert order.index(helper_binary) < order.index(helper)
    assert order.index(helper) < order.index(framework_binary) < order.index(framework)


def test_staging_omits_qt_linker_objects_but_keeps_runtime_libraries(tmp_path, packaging_tools):
    from build_macos_runtime import remove_qt_build_objects

    qt = tmp_path / "runtime/python/lib/python3.13/site-packages/PySide6/Qt"
    qt.mkdir(parents=True)
    linker_input = qt / "plugin.cpp.o"
    library = qt / "plugin.dylib"
    for path in (linker_input, library):
        path.write_bytes(b"contents")
    remove_qt_build_objects(tmp_path)
    assert not linker_input.exists()
    assert library.read_bytes() == b"contents"
