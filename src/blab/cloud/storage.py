"""Cloud artifact storage adapters."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class PresignedPutTarget:
    url: str
    method: str = "PUT"
    headers: dict[str, str] = field(default_factory=dict)
    key: str = ""


class BundleStore(Protocol):
    def bundle_key(self, job_id: str) -> str:
        """Return the artifact key used for a job's input bundle."""

    def create_upload_target(self, job_id: str, *, expires_in_s: int = 3600) -> PresignedPutTarget:
        """Return a client-usable upload target for a job bundle."""

    def download_bundle(self, job_id: str, destination: str | Path) -> Path:
        """Download a job bundle to a local path."""


class LocalBundleStore:
    """Filesystem-backed bundle store for development and tests."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def bundle_key(self, job_id: str) -> str:
        return f"jobs/{job_id}/input/solve.blabsolve.zip"

    def bundle_path(self, job_id: str) -> Path:
        return self.root / self.bundle_key(job_id)

    def create_upload_target(self, job_id: str, *, expires_in_s: int = 3600) -> PresignedPutTarget:
        del expires_in_s
        path = self.bundle_path(job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        return PresignedPutTarget(url=path.resolve().as_uri(), key=self.bundle_key(job_id))

    def put_bytes(self, job_id: str, data: bytes) -> Path:
        path = self.bundle_path(job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def download_bundle(self, job_id: str, destination: str | Path) -> Path:
        source = self.bundle_path(job_id)
        if not source.is_file():
            raise FileNotFoundError(f"No bundle exists for job {job_id}: {source}")
        output = Path(destination)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, output)
        return output


class S3BundleStore:
    """S3-backed bundle store used by the Fargate path."""

    def __init__(self, bucket: str, *, prefix: str = "", client=None):
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self._client = client

    @property
    def client(self):
        if self._client is None:
            try:
                import boto3
            except ImportError as exc:  # pragma: no cover - optional AWS extra
                raise RuntimeError('Install AWS dependencies with: python -m pip install -e ".[aws]"') from exc
            self._client = boto3.client("s3")
        return self._client

    def bundle_key(self, job_id: str) -> str:
        key = f"jobs/{job_id}/input/solve.blabsolve.zip"
        return f"{self.prefix}/{key}" if self.prefix else key

    def create_upload_target(self, job_id: str, *, expires_in_s: int = 3600) -> PresignedPutTarget:
        key = self.bundle_key(job_id)
        url = self.client.generate_presigned_url(
            ClientMethod="put_object",
            Params={
                "Bucket": self.bucket,
                "Key": key,
                "ContentType": "application/zip",
            },
            ExpiresIn=expires_in_s,
            HttpMethod="PUT",
        )
        return PresignedPutTarget(
            url=url,
            headers={"Content-Type": "application/zip"},
            key=key,
        )

    def download_bundle(self, job_id: str, destination: str | Path) -> Path:
        return self.download_key(self.bundle_key(job_id), destination)

    def download_key(self, key: str, destination: str | Path) -> Path:
        output = Path(destination)
        output.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, key, str(output))
        return output
