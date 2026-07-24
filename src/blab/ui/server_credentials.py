"""OS-backed storage for solve-server access tokens."""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

KEYRING_SERVICE = "Boundary Lab Solve Server"
_SESSION_TOKENS: dict[str, str] = {}


def normalize_server_credential_url(server_url: str) -> str:
    text = str(server_url or "").strip().rstrip("/") or "http://127.0.0.1:8765"
    parsed = urlsplit(text)
    if not parsed.scheme or not parsed.netloc:
        return text
    hostname = (parsed.hostname or "").lower()
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    try:
        parsed_port = parsed.port
    except ValueError:
        return text
    port = "" if parsed_port is None else f":{parsed_port}"
    return urlunsplit((parsed.scheme.lower(), f"{hostname}{port}", parsed.path.rstrip("/"), "", ""))


def load_server_access_token(server_url: str) -> str:
    credential_url = normalize_server_credential_url(server_url)
    if credential_url in _SESSION_TOKENS:
        return _SESSION_TOKENS[credential_url]
    keyring = _load_keyring()
    if keyring is None:
        return ""
    try:
        token = keyring.get_password(KEYRING_SERVICE, credential_url) or ""
    except Exception:
        return ""
    if token:
        _SESSION_TOKENS[credential_url] = token
    return token


def save_server_access_token(server_url: str, access_token: str) -> bool:
    credential_url = normalize_server_credential_url(server_url)
    token = str(access_token or "").strip()
    if token:
        _SESSION_TOKENS[credential_url] = token
    else:
        _SESSION_TOKENS.pop(credential_url, None)

    keyring = _load_keyring()
    if keyring is None:
        return False
    try:
        if token:
            keyring.set_password(KEYRING_SERVICE, credential_url, token)
        else:
            try:
                keyring.delete_password(KEYRING_SERVICE, credential_url)
            except Exception:
                pass
    except Exception:
        return False
    return True


def clear_session_tokens() -> None:
    """Clear process-memory tokens; intended for tests and application shutdown."""
    _SESSION_TOKENS.clear()


def _load_keyring():
    try:
        import keyring
    except ImportError:
        return None
    return keyring
