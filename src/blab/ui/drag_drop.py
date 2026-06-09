"""Shared Qt drag/drop helpers."""

from __future__ import annotations

from pathlib import Path


def local_drop_paths(event) -> list[Path]:
    mime_data = event.mimeData()
    if not mime_data.hasUrls():
        return []
    return [Path(url.toLocalFile()) for url in mime_data.urls() if url.isLocalFile()]
