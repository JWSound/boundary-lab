"""Readiness check using the same authentication as normal clients."""

import json
import os
import sys
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener

try:
    settings = json.loads(Path("/tmp/blab-health.json").read_text())
    request = Request(f"http://127.0.0.1:{int(settings['port'])}/v1/capabilities")
    if token := os.environ.get(settings["token-env"]):
        request.add_header("Authorization", f"Bearer {token}")
    with build_opener(ProxyHandler({})).open(request, timeout=3) as response:
        ready = json.load(response).get("state") == "ready"
except Exception:
    ready = False
sys.exit(0 if ready else 1)
