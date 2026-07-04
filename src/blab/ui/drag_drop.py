"""Shared Qt drag/drop helpers."""

from __future__ import annotations

from pathlib import Path


def local_drop_paths(event) -> list[Path]:
    mime_data = event.mimeData()
    paths = [Path(url.toLocalFile()) for url in mime_data.urls() if url.isLocalFile()] if mime_data.hasUrls() else []
    if paths or not mime_data.hasText():
        return paths

    text_paths = []
    for line in mime_data.text().splitlines():
        text = line.strip().strip("'\"")
        if not text:
            continue
        path = Path(text).expanduser()
        if path.exists():
            text_paths.append(path)
    return text_paths
