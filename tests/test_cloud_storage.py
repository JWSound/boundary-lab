from pathlib import Path

from blab.cloud.storage import LocalBundleStore, S3BundleStore


class FakeS3Client:
    def __init__(self) -> None:
        self.presign_calls = []
        self.download_calls = []

    def generate_presigned_url(self, **kwargs):
        self.presign_calls.append(kwargs)
        return "https://example.test/presigned"

    def download_file(self, bucket, key, destination):
        self.download_calls.append((bucket, key, destination))
        Path(destination).write_text("bundle", encoding="utf-8")


def test_local_bundle_store_puts_and_downloads_bundle(tmp_path: Path) -> None:
    store = LocalBundleStore(tmp_path / "artifacts")

    written = store.put_bytes("job_test", b"bundle")
    copied = store.download_bundle("job_test", tmp_path / "copy.zip")

    assert written.read_bytes() == b"bundle"
    assert copied.read_bytes() == b"bundle"
    assert store.create_upload_target("job_test").url.startswith("file:///")


def test_s3_bundle_store_creates_presigned_put_target() -> None:
    client = FakeS3Client()
    store = S3BundleStore("bucket", prefix="dev", client=client)

    target = store.create_upload_target("job_test", expires_in_s=60)

    assert target.url == "https://example.test/presigned"
    assert target.method == "PUT"
    assert target.headers == {"Content-Type": "application/zip"}
    assert target.key == "dev/jobs/job_test/input/solve.blabsolve.zip"
    assert client.presign_calls[0]["Params"] == {
        "Bucket": "bucket",
        "Key": "dev/jobs/job_test/input/solve.blabsolve.zip",
        "ContentType": "application/zip",
    }
    assert client.presign_calls[0]["ExpiresIn"] == 60
    assert client.presign_calls[0]["HttpMethod"] == "PUT"


def test_s3_bundle_store_downloads_key(tmp_path: Path) -> None:
    client = FakeS3Client()
    store = S3BundleStore("bucket", client=client)

    output = store.download_key("jobs/job_test/input/solve.blabsolve.zip", tmp_path / "bundle.zip")

    assert output.read_text(encoding="utf-8") == "bundle"
    assert client.download_calls == [
        ("bucket", "jobs/job_test/input/solve.blabsolve.zip", str(tmp_path / "bundle.zip"))
    ]
