"""Container defaults; all ordinary server CLI overrides remain available."""

import json
import os
import signal
import sys
from pathlib import Path

from blab.server import main


def terminate(_signum, _frame):
    raise KeyboardInterrupt


arguments = [
    "--host",
    "0.0.0.0",
    "--root",
    os.environ["BLAB_SERVER_ROOT"],
    "--port",
    os.environ["BLAB_SERVER_PORT"],
    "--mode",
    os.environ["BLAB_SERVER_MODE"],
    *sys.argv[1:],
]
# Record only routing information for the local authenticated health check.
settings = {"port": os.environ["BLAB_SERVER_PORT"], "token-env": "BLAB_SERVER_TOKEN"}
for index, argument in enumerate(arguments):
    for key in settings:
        if argument == f"--{key}" and index + 1 < len(arguments):
            settings[key] = arguments[index + 1]
        elif argument.startswith(f"--{key}="):
            settings[key] = argument.split("=", 1)[1]
Path("/tmp/blab-health.json").write_text(json.dumps(settings))
signal.signal(signal.SIGTERM, terminate)
main(arguments)
