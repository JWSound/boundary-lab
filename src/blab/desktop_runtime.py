"""Isolated Windows distribution startup, deliberately independent of Qt."""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
import sys
import traceback
import uuid
from pathlib import Path


def documents_directory() -> Path:
    """Resolve redirected/OneDrive Documents through the Windows Known Folder API."""
    if sys.platform != "win32":
        return Path.home() / "Documents"
    from ctypes import wintypes

    shell = ctypes.WinDLL("shell32", use_last_error=True)
    ole = ctypes.WinDLL("ole32")
    folder_id = (ctypes.c_byte * 16).from_buffer_copy(uuid.UUID("FDD39AD0-238F-46AF-ADB4-6C85480369C7").bytes_le)
    pointer = ctypes.c_void_p()
    shell.SHGetKnownFolderPath.argtypes = [
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.HANDLE,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    shell.SHGetKnownFolderPath.restype = ctypes.c_long
    ole.CoTaskMemFree.argtypes = [ctypes.c_void_p]
    result = shell.SHGetKnownFolderPath(ctypes.byref(folder_id), 0, None, ctypes.byref(pointer))
    if result != 0:
        raise OSError(f"Windows could not locate Documents (HRESULT {result:#x}).")
    try:
        return Path(ctypes.wstring_at(pointer))
    finally:
        ole.CoTaskMemFree(pointer)


def seed_documents(source: Path, destination: Path) -> None:
    """Install missing examples/docs only; never replace user-edited files."""
    for folder in ("examples", "documentation"):
        for path in (source / folder).rglob("*"):
            if path.is_file():
                target = destination / path.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with target.open("xb") as output, path.open("rb") as input_file:
                        shutil.copyfileobj(input_file, output)
                except FileExistsError:
                    pass
    (destination / "runs").mkdir(parents=True, exist_ok=True)


def configure_runtime(root: Path, *, documents: Path | None = None, local_data: Path | None = None) -> dict:
    root = root.resolve()
    manifest = json.loads((root / "runtime-manifest.json").read_text(encoding="utf-8"))
    user = (documents if documents is not None else documents_directory()) / "Boundary Lab"
    local = (local_data if local_data is not None else Path(os.environ["LOCALAPPDATA"])) / "Boundary Lab"
    cache = local / "runtime" / manifest["runtime_id"]
    for folder in (cache / "julia-depot", local / "logs", local / "tmp", local / "matplotlib"):
        folder.mkdir(parents=True, exist_ok=True)
    # Keep the bundled interpreter and solver independent of developer installs.
    for key in list(os.environ):
        if key.upper().startswith(("JULIA_", "PYTHON", "BLAB_", "QT_", "QML", "CUDA_PATH", "CUDA_HOME")):
            del os.environ[key]
    os.environ.update(
        BLAB_RESOURCE_ROOT=str(root / "resources"),
        BLAB_USER_ROOT=str(user),
        BLAB_PACKAGED_BACKENDS=",".join(manifest["backends"]),
        BLAB_JULIA_EXE=str(root / "runtime/julia/bin/julia.exe"),
        JULIA_DEPOT_PATH=f"{cache / 'julia-depot'};{root / 'runtime/julia-depot'};",
        JULIA_LOAD_PATH="@;@stdlib",
        JULIA_PKG_OFFLINE="true",
        JULIA_PKG_PRECOMPILE_AUTO="0",
        JULIA_CPU_TARGET="generic",
        JULIA_NUM_PRECOMPILE_TASKS="2",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUTF8="1",
        MPLCONFIGDIR=str(local / "matplotlib"),
        TEMP=str(local / "tmp"),
        TMP=str(local / "tmp"),
        PATH=os.pathsep.join(
            (
                str(root / "runtime/python"),
                str(root / "runtime/julia/bin"),
                str(Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32"),
            )
        ),
    )
    sys.dont_write_bytecode = True
    seed_documents(root / "user-files", user)
    os.chdir(user)
    return manifest


def check_cuda(root: Path) -> bool:
    """Check the bundled toolkit against the installed driver, without downloads."""
    script = 'using CUDA; CUDA.functional(true) || error("CUDA unavailable"); CUDA.versioninfo()'
    command = [
        os.environ["BLAB_JULIA_EXE"],
        "--startup-file=no",
        f"--project={root / 'runtime/python/Lib/site-packages/beat_engine/julia_cuda'}",
        "-e",
        script,
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        detail = result.stdout + result.stderr
        success = result.returncode == 0
    except (OSError, subprocess.TimeoutExpired) as exc:
        success, detail = False, str(exc)
    log = Path(os.environ["LOCALAPPDATA"]) / "Boundary Lab/logs/cuda-check.log"
    log.write_text(detail, encoding="utf-8")
    message = (
        "NVIDIA CUDA is ready. Select BEAT Engine CUDA in Preferences."
        if success
        else "NVIDIA CUDA is unavailable. CPU solving is still available. Install a compatible NVIDIA driver "
        "from nvidia.com, then run this check again."
    )
    ctypes.windll.user32.MessageBoxW(
        None, message + f"\n\nDetails: {log}", "Boundary Lab CUDA check", 0x40 if success else 0x30
    )
    return success


def _main() -> None:
    root = Path(sys.executable).resolve().parents[2]
    manifest = configure_runtime(root)
    args = sys.argv[1:]
    if args == ["--initialize"]:
        return
    if args == ["--check-cuda"]:
        if "cuda" not in manifest["backends"]:
            raise RuntimeError("This is the CPU package. Install the CPU + CUDA package to use NVIDIA acceleration.")
        raise SystemExit(0 if check_cuda(root) else 1)
    if args and args[0] == "--cli":
        from blab.cli import main as cli_main

        cli_main(args[1:])
    else:
        from blab.gui import main as gui_main

        gui_main()


def main() -> None:
    log = None
    try:
        if sys.stderr is None:
            path = Path(os.environ["LOCALAPPDATA"]) / "Boundary Lab/logs/launcher.log"
            path.parent.mkdir(parents=True, exist_ok=True)
            log = path.open("a", encoding="utf-8", buffering=1)
            sys.stdout = sys.stderr = log
        _main()
    except Exception:
        traceback.print_exc()
        if log is not None:
            ctypes.windll.user32.MessageBoxW(
                None, f"Boundary Lab could not start.\n\nDetails: {log.name}", "Boundary Lab", 0x10
            )
        raise SystemExit(1) from None
    finally:
        if log is not None:
            sys.stdout = sys.stderr = None
            log.close()


if __name__ == "__main__":
    main()
