"""Manifest-only discovery and explicit, lazy loading of trusted local packages."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from types import ModuleType

from blab.paths import APP_ROOT

_ENTRY = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*", re.ASCII)
_ID = re.compile(r"[a-z][a-z0-9_.-]*", re.ASCII)


def provider_roots() -> tuple[Path, ...]:
    return (APP_ROOT / "geometry_providers",)


@dataclass(frozen=True)
class ProviderManifest:
    id: str
    name: str
    version: str
    backend: str
    editor: str | None
    source_schema_version: int
    default_source: dict = field(default_factory=dict)
    mesh_scale_factor: float = 1.0

    @classmethod
    def read(cls, path: Path):
        data = json.loads(path.read_text(encoding="utf-8"))
        for key in ("manifest_version", "provider_api"):
            if type(data.get(key)) is not int or data[key] != 1:
                raise ValueError(f"Unsupported {key}; this host supports version 1.")
        if not isinstance(data.get("id"), str) or not _ID.fullmatch(data["id"]):
            raise ValueError("Invalid provider ID; use lowercase letters, digits, dots, hyphens or underscores.")
        for key in ("name", "version"):
            if not isinstance(data.get(key), str) or not data[key].strip():
                raise ValueError(f"Missing {key}.")
        for key in ("backend", "editor"):
            entry = data.get(key)
            if key == "editor" and entry is None:
                continue
            if not isinstance(entry, str) or not _ENTRY.fullmatch(entry):
                raise ValueError(f"Invalid {key} entry point; expected module:factory.")
        schema = data.get("source_schema_version")
        if type(schema) is not int or schema < 1:
            raise ValueError("source_schema_version must be a positive integer.")
        source = data.get("default_source", {})
        if not isinstance(source, dict):
            raise ValueError("default_source must be an object.")
        json.dumps(source, allow_nan=False)
        scale = data.get("mesh_scale_factor", 1.0)
        if type(scale) not in (float, int) or not math.isfinite(scale) or scale <= 0:
            raise ValueError("mesh_scale_factor must be positive and finite.")
        return cls(
            data["id"], data["name"], data["version"], data["backend"], data.get("editor"), schema, source, float(scale)
        )


@dataclass
class ProviderPackage:
    path: Path
    manifest: ProviderManifest | None = None
    error: str = ""
    fingerprint: str = ""


class ProviderCatalog:
    """Scanning never imports package code. Enabling permits lazy imports.

    Modules remain loaded for the process lifetime. A changed loaded package
    requires a restart; it is never mixed with new code or silently replaced.
    """

    def __init__(self, roots=None, *, enabled=(), reserved=("ath", "ath4")):
        self.roots = tuple(Path(root).resolve() for root in (provider_roots() if roots is None else roots))
        self.enabled = set(enabled)
        self.reserved = set(reserved)
        self.packages: list[ProviderPackage] = []
        self._loaded: dict[str, tuple[Path, str, str]] = {}
        self._failures: dict[str, str] = {}
        self._lock = RLock()

    def scan(self):
        with self._lock:
            packages = []
            paths = sorted(
                {path.resolve() for root in self.roots if root.is_dir() for path in root.glob("*/provider.json")}
            )
            for path in paths:
                package = ProviderPackage(path.parent)
                try:
                    package.manifest = ProviderManifest.read(path)
                    digest = hashlib.sha256(path.read_bytes())
                    for source in sorted(path.parent.rglob("*.py")):
                        if not source.resolve().is_relative_to(path.parent):
                            raise ValueError("Package source must stay within its package directory.")
                        digest.update(str(source.relative_to(path.parent)).encode())
                        digest.update(source.read_bytes())
                    package.fingerprint = digest.hexdigest()
                except Exception as exc:
                    package.error = str(exc)
                packages.append(package)
            ids = [p.manifest.id for p in packages if p.manifest]
            for package in packages:
                if package.manifest:
                    key = package.manifest.id
                    if key in self._failures:
                        package.error = self._failures[key]
                    if key in self.reserved or ids.count(key) > 1:
                        package.error = f"Conflicting provider ID: {key}"
                    loaded = self._loaded.get(key)
                    if loaded and loaded[:2] != (package.path, package.fingerprint):
                        package.error = "Package changed after loading; restart Boundary Lab."
            self.packages = packages
        return tuple(packages)

    def status(self, package, *, enabled=None):
        return package.error or (
            "Enabled" if package.manifest.id in (self.enabled if enabled is None else enabled) else "Disabled"
        )

    def package(self, provider_id):
        matches = [p for p in self.packages if p.manifest and p.manifest.id == provider_id]
        if len(matches) != 1:
            raise ValueError(f"Provider {provider_id!r} is missing or has conflicting packages.")
        package = matches[0]
        if package.error:
            raise ValueError(package.error)
        if provider_id not in self.enabled:
            raise ValueError(f"Provider {provider_id!r} is disabled. Enable it in Preferences.")
        return package

    def factory(self, provider_id, kind):
        with self._lock:
            package = self.package(provider_id)
            entry = getattr(package.manifest, kind)
            if entry is None:
                return None
            loaded = self._loaded.get(provider_id)
            if loaded is None:
                namespace = "_blab_provider_" + hashlib.sha256(str(package.path).encode()).hexdigest()[:20]
                root = ModuleType(namespace)
                root.__path__ = [str(package.path)]
                sys.modules[namespace] = root
                loaded = (package.path, package.fingerprint, namespace)
                self._loaded[provider_id] = loaded
            module, name = entry.split(":")
            try:
                factory = getattr(importlib.import_module(f"{loaded[2]}.{module}"), name)
                if not callable(factory):
                    raise TypeError(f"{entry} is not callable.")
                return factory
            except Exception as exc:
                package.error = f"Could not load {kind}: {exc}. Restart after repairing the package."
                self._failures[provider_id] = package.error
                raise ValueError(package.error) from exc

    def source_defaults(self, provider_id):
        return deepcopy(self.package(provider_id).manifest.default_source)


_catalog: ProviderCatalog | None = None


def provider_catalog() -> ProviderCatalog:
    global _catalog
    if _catalog is None:
        _catalog = ProviderCatalog()
        _catalog.scan()
    return _catalog
