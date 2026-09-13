"""Independent process deadlines for explicitly authorized live pilot operations."""

from __future__ import annotations

import math
import os
import select
import subprocess
import sys
import time
from contextlib import contextmanager

from crypto_ai.sentiment.exceptions import NetworkSafetyError

# A fresh interpreter does not inherit Python thread locks or signing-key memory.
# Stdin is solely a parent-liveness/cancellation pipe; no secret enters it.
_WATCHDOG = """
import os, select, signal, sys, time
parent, deadline = int(sys.argv[1]), float(sys.argv[2])
try:
    os.write(1, b'1')
    while os.getppid() == parent:
        remaining = deadline - time.monotonic()
        readable, _, _ = select.select([0], [], [], max(0, min(remaining, 0.1)))
        if readable:
            break
        if remaining <= 0:
            if os.getppid() == parent:
                os.kill(parent, signal.SIGKILL)
            break
except BaseException:
    if os.getppid() == parent:
        os.kill(parent, signal.SIGKILL)
"""


@contextmanager
def hard_deadline(seconds: float):
    """Kill the calling process at the monotonic deadline, even during blocked I/O.

    Each nested bound owns an exec'd watchdog, so a ten-second HTTP operation
    cannot reset an outer 900-second/slot deadline. The parent pipe and live parent
    relationship prevent a detached watchdog from targeting a reused PID. The
    operation starts only after a startup acknowledgement. No Python runs after
    fork in a multi-threaded parent, avoiding inherited interpreter-lock deadlocks.
    """
    if type(seconds) not in (int, float) or not 0 < seconds <= 900 or not math.isfinite(seconds):
        raise NetworkSafetyError("invalid hard process deadline")
    deadline = time.monotonic() + seconds
    try:
        child = subprocess.Popen(
            [sys.executable, "-I", "-c", _WATCHDOG, str(os.getpid()), repr(deadline)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )
    except OSError:
        raise NetworkSafetyError("hard process watchdog could not start") from None
    try:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([child.stdout], [], [], remaining)[0]:
            raise NetworkSafetyError("hard watchdog startup exceeded the deadline")
        if child.stdout.read(1) != b"1" or child.poll() is not None:
            raise NetworkSafetyError("hard process watchdog did not become ready")
        yield
    finally:
        child.stdin.close()
        child.stdout.close()
        try:
            child.wait(timeout=1)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
