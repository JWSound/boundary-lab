import blab.ui.server_tokens as tokens


def test_server_tokens_are_remembered_by_normalized_url_for_the_session() -> None:
    tokens.clear_session_tokens()

    tokens.remember_server_access_token("HTTPS://Example.COM:8765/", "secret-token")

    assert tokens.load_server_access_token("https://example.com:8765") == "secret-token"
    assert "secret-token" not in tokens.normalize_server_token_url("https://example.com:8765")


def test_empty_server_token_clears_the_session_value() -> None:
    tokens.clear_session_tokens()
    tokens.remember_server_access_token("https://pod.proxy.runpod.net", "session-token")

    tokens.remember_server_access_token("https://pod.proxy.runpod.net/", "")

    assert tokens.load_server_access_token("https://pod.proxy.runpod.net") == ""
