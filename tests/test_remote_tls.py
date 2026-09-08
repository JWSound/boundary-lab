"""Authentication and verified TLS across real client/server sockets."""

import shutil
import socket
import ssl
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace
from urllib.error import URLError

import pytest

from blab.remote import RemoteBackend, _NoRedirect
from blab.server import SolveService, create_http_server

TOKEN = "test-only-token-" + "a" * 32


@pytest.fixture(scope="module")
def certificate(tmp_path_factory):
    openssl = shutil.which("openssl")
    if not openssl and Path("C:/Program Files/Git/usr/bin/openssl.exe").exists():
        openssl = "C:/Program Files/Git/usr/bin/openssl.exe"
    if not openssl:
        pytest.skip("OpenSSL required to generate an ephemeral TLS test certificate")
    directory = tmp_path_factory.mktemp("remote-tls")
    cert, key = directory / "cert.pem", directory / "key.pem"
    subprocess.run(
        [
            openssl,
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-days",
            "2",
            "-subj",
            "/CN=localhost",
            "-addext",
            "subjectAltName=DNS:localhost,IP:127.0.0.1",
            "-keyout",
            str(key),
            "-out",
            str(cert),
        ],
        check=True,
        capture_output=True,
    )
    return cert, key


@pytest.fixture
def tls_service(tmp_path, certificate):
    cert, key = certificate
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    service = SolveService(tmp_path / "jobs", backend=SimpleNamespace())
    server = create_http_server(service, 0, token=TOKEN, tls_context=context)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield service, f"https://127.0.0.1:{server.server_port}", cert
    finally:
        server.shutdown()
        server.server_close()
        service.close()
        thread.join(5)


def test_authenticated_tls_capabilities(tls_service):
    _service, url, cert = tls_service
    assert RemoteBackend(url, token=TOKEN, ca_file=cert).check_capabilities()["backend_ids"] == ["beat_cpu"]


@pytest.mark.parametrize("token", [None, "incorrect-token"])
@pytest.mark.parametrize(
    "method,path,data",
    [
        ("GET", "/v1/capabilities", None),
        ("POST", "/v1/jobs", b"invalid ZIP must not be parsed"),
        ("GET", "/v1/jobs/" + "a" * 32, None),
        ("DELETE", "/v1/jobs/" + "a" * 32, None),
    ],
)
def test_authentication_precedes_upload_and_job_access(tls_service, token, method, path, data):
    service, url, cert = tls_service
    with pytest.raises(RuntimeError, match="HTTP 401"):
        RemoteBackend(url, token=token, ca_file=cert).call(path, method=method, data=data)
    assert not list(service.root.iterdir())


def test_untrusted_certificate_is_rejected(tls_service):
    _service, url, _cert = tls_service
    with pytest.raises(URLError, match="CERTIFICATE_VERIFY_FAILED"):
        RemoteBackend(url, token=TOKEN).check_capabilities()


def test_certificate_hostname_mismatch_is_rejected(tls_service):
    _service, url, cert = tls_service
    port = int(url.rsplit(":", 1)[1])
    context = ssl.create_default_context(cafile=cert)
    with socket.create_connection(("127.0.0.1", port), timeout=5) as connection:
        with pytest.raises(ssl.SSLCertVerificationError, match="Hostname mismatch"):
            context.wrap_socket(connection, server_hostname="wrong.example")


def test_lan_requires_token_and_tls_before_binding(tmp_path):
    service = SolveService(tmp_path)
    for options in ({}, {"token": TOKEN}, {"tls_context": ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)}):
        with pytest.raises(ValueError, match="LAN binding requires"):
            create_http_server(service, 0, host="0.0.0.0", **options)
    with pytest.raises(ValueError, match="LAN connections require"):
        RemoteBackend("http://solver.example:8765", token=TOKEN)
    with pytest.raises(ValueError, match="LAN connections require"):
        RemoteBackend("https://solver.example:8765")


def test_redirects_cannot_forward_credentials():
    with pytest.raises(RuntimeError, match="redirects are not allowed"):
        _NoRedirect().redirect_request(None, None, 302, "Found", {}, "https://another.example")
