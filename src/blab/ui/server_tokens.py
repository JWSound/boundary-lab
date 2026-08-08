"""Session-only storage for solve-server access tokens."""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

_SESSION_TOKENS: dict[str, str] = {}


def normalize_server_token_url(server_url: str) -> str:
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
    return _SESSION_TOKENS.get(normalize_server_token_url(server_url), "")


def remember_server_access_token(server_url: str, access_token: str) -> None:
    token_url = normalize_server_token_url(server_url)
    token = str(access_token or "").strip()
    if token:
        _SESSION_TOKENS[token_url] = token
    else:
        _SESSION_TOKENS.pop(token_url, None)


def clear_session_tokens() -> None:
    """Clear process-memory tokens; intended for tests and application shutdown."""
    _SESSION_TOKENS.clear()
