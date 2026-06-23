# Copyright (c) 2026 Jinan Kordab
# SPDX-License-Identifier: MIT

"""Debug logging for pipeline tracing."""
import os
from datetime import datetime

DEBUG_LOG = "/tmp/piste_debug.log"

def log(msg: str):
    """Write timestamped debug message to log file."""
    ts = datetime.utcnow().strftime("%H:%M:%S.%f")[:-3]
    with open(DEBUG_LOG, "a") as f:
        f.write(f"[{ts}] {msg}\n")
    # Also print for docker logs
    print(f"[DEBUG] {msg}", flush=True)
