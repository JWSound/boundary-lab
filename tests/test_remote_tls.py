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
def tls_service(tmp_path, certificate, monkeypatch):
    cert, key = certificate
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    service = SolveService(tmp_path / "jobs", backend=SimpleNamespace())
    # Test-only TLS front end represents provider-managed HTTPS. The application
    # exposes no certificate configuration; clients use their normal trust store.
    server = create_http_server(service, 0, token=TOKEN, mode="hosted")
    get_request = server.get_request

    def accept_tls():
        connection, address = get_request()
        return context.wrap_socket(connection, server_side=True, do_handshake_on_connect=False), address

    server.get_request = accept_tls
    server.handle_error = lambda *_args: None
    monkeypatch.setenv("SSL_CERT_FILE", str(cert))
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
    assert RemoteBackend(url, token=TOKEN).check_capabilities()["backend_ids"] == ["beat_cpu"]


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
        RemoteBackend(url, token=token).call(path, method=method, data=data)
    assert not list(service.root.iterdir())


def test_untrusted_certificate_is_rejected(tls_service, monkeypatch):
    monkeypatch.delenv("SSL_CERT_FILE")
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


def test_hosted_mode_requires_key_before_binding(tmp_path):
    service = SolveService(tmp_path)
    with pytest.raises(ValueError, match="Hosted mode requires"):
        create_http_server(service, 0, host="0.0.0.0", mode="hosted")


@pytest.mark.parametrize("mode,token", [("private-network", None), ("private-network", TOKEN), ("hosted", TOKEN)])
def test_private_network_and_hosted_access_modes(tmp_path, mode, token):
    service = SolveService(tmp_path)
    server = create_http_server(service, 0, token=token, mode=mode)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        assert RemoteBackend(url, token=token).check_capabilities()["state"] == "ready"
        if token:
            with pytest.raises(RuntimeError, match="401"):
                RemoteBackend(url).check_capabilities()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


def test_client_accepts_private_lan_and_vpn_http():
    assert RemoteBackend("http://192.168.1.20:8765").url == "http://192.168.1.20:8765"
    assert RemoteBackend("http://workstation.internal:8765", token=TOKEN).token == TOKEN


def test_redirects_cannot_forward_credentials():
    with pytest.raises(RuntimeError, match="redirects are not allowed"):
        _NoRedirect().redirect_request(None, None, 302, "Found", {}, "https://another.example")
