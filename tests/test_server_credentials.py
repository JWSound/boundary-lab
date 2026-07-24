import blab.ui.server_credentials as credentials


class MemoryKeyring:
    def __init__(self):
        self.values: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str) -> str | None:
        return self.values.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self.values[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        self.values.pop((service, username), None)


def test_server_tokens_use_normalized_url_and_os_keyring(monkeypatch) -> None:
    keyring = MemoryKeyring()
    monkeypatch.setattr(credentials, "_load_keyring", lambda: keyring)
    credentials.clear_session_tokens()

    assert credentials.save_server_access_token("HTTPS://Example.COM:8765/", "secret-token") is True
    credentials.clear_session_tokens()

    assert credentials.load_server_access_token("https://example.com:8765") == "secret-token"
    assert "secret-token" not in credentials.normalize_server_credential_url("https://example.com:8765")

    assert credentials.save_server_access_token("https://example.com:8765", "") is True
    credentials.clear_session_tokens()
    assert credentials.load_server_access_token("https://example.com:8765") == ""


def test_server_token_remains_available_for_session_without_keyring(monkeypatch) -> None:
    monkeypatch.setattr(credentials, "_load_keyring", lambda: None)
    credentials.clear_session_tokens()

    assert credentials.save_server_access_token("https://pod.proxy.runpod.net", "session-token") is False
    assert credentials.load_server_access_token("https://pod.proxy.runpod.net") == "session-token"
