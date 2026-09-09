"""Shared-server stdio shim — ``python -m noesis.mcp --shared`` (ADR-86).

Why this exists, in one number: a Noesis process that has loaded the embedding
model measures **1342 MB resident, of which 1140 MB is unshared** with a second
process running the same model (``Shared_File`` measured at 0 MB — the weights
land in private tensors, not in a shared file mapping). Ten agents each
spawning ``python -m noesis.mcp`` is therefore roughly 11 GB of RAM and ten
model loads, and no amount of caching changes that: the on-disk cache is
already shared (523 MB, one copy; a warm second process loads in 11.7 s and
downloads nothing), so the duplication is in memory, not on disk.

The fix is topology. Each agent runs this shim — a thin stdio-to-HTTP MCP
proxy holding no model, no Qdrant pool and no state DB handle — and they all
talk to one server process that holds those things once. This is the same
daemon-plus-thin-client shape a language server or ``docker`` uses.

That trade only works because the shared server is safe to share, which is what
ADR-83/84/85 are for: one process now serving every agent's query is precisely
the reader-vs-reader concurrency that used to corrupt results. Shipping this
without those would have been worse than shipping neither.

**Electing the server.** N agents starting at once must produce exactly one
server, and the losers must wait for it rather than fail while it is still
starting. The mechanism, in order:

1. *Fast path.* Probe the configured port. A warm start takes no lock at
   all — measured at 0.07 s, against 5.8-8.1 s for a cold election.
2. *Elect.* Take an exclusive advisory lock (``filelock``). Chosen over a PID
   file or a row in SQLite for one property neither has: **the OS releases it
   when the holder dies**, so a killed starter cannot wedge every future shim.
   Verified by SIGKILLing a holder — the lock file survives on disk, the lock
   does not, and the next starter won in 1.32 s.
3. *Re-check under the lock.* Double-checked locking: whoever was ahead of us
   may already have started it. This is the branch 9 of 10 shims take.
4. *Spawn and wait.* The winner starts the server detached and polls
   readiness with bounded exponential backoff.

The port bind is the final arbiter behind all of that: two servers on one port
cannot both exist, so even a broken election degrades to one server plus a
loud failure rather than to silent double-serving. Verified: a second
``bind()`` on the same address is refused with ``EADDRINUSE`` even with
``SO_REUSEADDR`` set.

There is deliberately no endpoint/discovery file. An earlier draft published
one, and it introduced a failure this design does not otherwise have: asked for
port 8199 while the file named 8123, it returned the live 8123 — silently
proxying an operator to a server they did not ask for. The port is not
discovered, it is configured (``[server] port``), and both the shim and the
server read the same value; liveness is the probe. Removing the file removed
the bug along with the stale-file and partial-write handling it needed. The
probe checks the *body* for the same reason: on a port as contested as 8000,
"something answered 200" and "Noesis is running" are different facts, and
proxying an agent to whatever else is listening is that same bug by another
route.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
from filelock import FileLock, Timeout

logger = logging.getLogger(__name__)

# How long a loser will wait for the elected starter's server to answer
# /healthz. Generous on purpose: the winner pays Qdrant connection setup,
# schema init and the orphan sweep before it serves. The embedding model is
# NOT on that path — warm-up is a background task (ADR-77) — so this bounds
# process startup, not model loading.
READY_TIMEOUT_S = 60.0
_PROBE_TIMEOUT_S = 1.0
_BACKOFF_START_S = 0.05
_BACKOFF_MAX_S = 1.0
# How long a server we started but cannot use gets to exit on SIGTERM before
# it is killed. See _terminate_spawned for why a bounded wait, not a bare
# terminate().
_TERMINATE_GRACE_S = 5.0


def default_runtime_dir() -> Path:
    """Where the election lock and the spawned server's log live.

    ``XDG_RUNTIME_DIR`` first — it is the standard home for per-user runtime
    state and is cleaned up on logout, which is exactly the lifetime of a
    server that may still be running. Falls back to the cache dir, matching
    ``prefetch.default_fastembed_cache``'s anchoring so a shim spawned with an
    agent host's arbitrary cwd resolves the same path as every other one."""
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if runtime:
        return Path(runtime) / "noesis"
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base).expanduser() / "noesis"


def lock_path(runtime_dir: Path) -> Path:
    return runtime_dir / "server.lock"


def server_log_path(runtime_dir: Path) -> Path:
    return runtime_dir / "server.log"


def health_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/healthz"


def mcp_url(port: int) -> str:
    # Trailing slash matters: without it the mount 404s or hangs on connect
    # (docs/getting-started/connecting-agents.md troubleshooting table).
    return f"http://127.0.0.1:{port}/mcp/"


def probe(port: int, timeout: float = _PROBE_TIMEOUT_S) -> bool:
    """Is a *Noesis* server answering on this port? Never raises.

    A 200 on the port is not enough. 127.0.0.1:8000 is the most contested
    port on a developer's machine, and treating whatever answers there as
    Noesis is the same class of bug the endpoint file was deleted for
    (module docstring): silently proxying an operator to a server they did
    not ask for. So the body has to identify itself, using the shape
    ``plugin/.../healthcheck.py`` already treats as the contract —
    ``status == "ok"`` in a JSON object.

    Not stricter than that on purpose: ``assets``/``embedder_ready`` are a
    fail-loud *reporting* surface (ADR-77/78) whose values legitimately
    include ``"unknown"``, and making the probe a second consumer of their
    schema would turn a change there into a shim that spawns a duplicate
    server onto an occupied port. A malformed or non-JSON body is simply
    not-Noesis: ``.json()`` raising must return False, not propagate out of
    a function whose contract is that it never raises."""
    try:
        response = httpx.get(health_url(port), timeout=timeout)
        if response.status_code != 200:
            return False
        payload = response.json()
    except (httpx.HTTPError, OSError, ValueError):
        return False
    return isinstance(payload, dict) and payload.get("status") == "ok"


def _wait_ready(
    process: subprocess.Popen[bytes],
    port: int,
    deadline: float,
    runtime_dir: Path,
) -> bool:
    """Poll for readiness until *deadline*, watching the child as we go.

    Watching the child is the point of taking it as an argument: a server
    that dies on startup — port already taken, an import error, Qdrant
    down — never answers /healthz, so a loop that only probes the port sits
    out the entire readiness budget (60 s by default) before reporting a
    failure the child announced in its first second. The exit is detectable
    and the diagnosis is already written to the log, so both go in the
    error instead of being waited out."""
    delay = _BACKOFF_START_S
    while time.monotonic() < deadline:
        if probe(port):
            return True
        # Checked after the probe, never before: a child that exits the
        # instant after it served a good /healthz still counts as ready.
        returncode = process.poll()
        if returncode is not None:
            # Our child died — but that does not mean the port is unserved.
            # The commonest way to get here is losing a race we never saw:
            # another shim (one that resolved a different runtime_dir, so it
            # elected independently) already has a healthy server up, and our
            # duplicate died on EADDRINUSE. Probing once more distinguishes
            # "nothing is there" from "someone beat us to it", and only the
            # first is an error. Without this the second case fails a shim
            # that had a perfectly good server to talk to.
            if probe(port):
                return True
            raise RuntimeError(
                f"the Noesis server started for port {port} exited with code "
                f"{returncode} before becoming ready, and nothing else is "
                f"serving that port; the reason is in "
                f"{server_log_path(runtime_dir)}"
            )
        time.sleep(delay)
        delay = min(delay * 1.6, _BACKOFF_MAX_S)
    return False


def _terminate_spawned(process: subprocess.Popen[bytes]) -> None:
    """Stop a server this shim started but could not hand to anyone.

    Without this the failure path leaves a detached uvicorn running that the
    operator never started and will not think to look for — and, because it
    holds the port, one that makes every later shim's probe fail its identity
    check or, worse, succeed against a half-started process.

    SIGTERM first, because that is the signal uvicorn's graceful shutdown
    listens for. But the process being terminated here is by definition one
    that failed to come up: it may be wedged in an import, blocked in Qdrant
    connection setup, or otherwise not yet running the signal handler that
    makes SIGTERM mean anything. A bounded wait then SIGKILL is what turns
    "asked it to stop" into "it stopped"; ``wait()`` in both branches also
    reaps the child rather than leaving a zombie behind."""
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=_TERMINATE_GRACE_S)
        except subprocess.TimeoutExpired:
            logger.warning(
                "spawned server ignored SIGTERM; killing pid %d", process.pid
            )
            process.kill()
    try:
        # Bounded, because this runs while holding the election lock: a child
        # that will not die must not wedge every other shim on the machine.
        # Reaping is worth a moment, never worth the lock — the docstring
        # above promises a total bounded at 2 x ready_timeout and an
        # unbounded wait() here would have quietly made that false.
        process.wait(timeout=_TERMINATE_GRACE_S)
    except subprocess.TimeoutExpired:
        logger.warning(
            "spawned server pid %d did not exit after SIGKILL; abandoning it "
            "rather than holding the election lock",
            process.pid,
        )


def _spawn_server(port: int, runtime_dir: Path) -> subprocess.Popen[bytes]:
    """Start the HTTP server detached, so it outlives the shim that won the
    election — the next agent must find it already running.

    The log handle is closed as soon as Popen returns. ``Popen`` dup()s it
    into the child before that, and the child's copy is what keeps the file
    open, so closing here costs the server nothing and saves the shim — which
    lives as long as the agent does — from holding a write handle to a file
    it never writes to again."""
    runtime_dir.mkdir(parents=True, exist_ok=True)
    with open(server_log_path(runtime_dir), "ab") as log:
        return subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "noesis.app:app",
                # 127.0.0.1 only, never a wildcard (CLAUDE.md rule 2).
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            stdout=log,
            stderr=log,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )


def ensure_server(
    port: int,
    *,
    runtime_dir: Path | None = None,
    spawn: bool = True,
    ready_timeout: float = READY_TIMEOUT_S,
) -> int:
    """Return the port of a live server, starting one if needed.

    Exactly one caller starts a server however many call this at once; see
    the module docstring for the mechanism and what was verified."""
    runtime_dir = runtime_dir or default_runtime_dir()
    runtime_dir.mkdir(parents=True, exist_ok=True)

    # 1. Fast path — no lock in the common warm case.
    if probe(port):
        return port

    if not spawn:
        raise RuntimeError(
            f"no Noesis server answering at {health_url(port)} and --no-spawn "
            f"was given. Start one with: uvicorn noesis.app:app "
            f"--host 127.0.0.1 --port {port}"
        )

    try:
        # 2. Elect. Released by the OS if this process dies holding it.
        #    The lock wait is itself bounded by ready_timeout, so the total
        #    this call can take stays bounded (at 2 x ready_timeout) even
        #    though the readiness budget below is measured from here.
        with FileLock(str(lock_path(runtime_dir)), timeout=ready_timeout):
            # The deadline is deliberately taken *after* the lock, not before.
            # Computed before, every second spent queued behind another shim
            # is deducted from the time this server gets to start — so the
            # shim after a starter that burned the whole budget acquires the
            # lock with none left, spawns anyway, and fails instantly with a
            # timeout message about a server that had no time to answer.
            deadline = time.monotonic() + ready_timeout

            # 3. Re-check under the lock — someone ahead of us may have won.
            if probe(port):
                return port

            # 4. We are the elected starter.
            logger.info("starting Noesis server on port %d", port)
            process = _spawn_server(port, runtime_dir)
            try:
                ready = _wait_ready(process, port, deadline, runtime_dir)
            except Exception:
                # Includes the child-died error from _wait_ready: whatever
                # went wrong with the SERVER, this shim owns the process it
                # started. Deliberately Exception and not BaseException: a
                # KeyboardInterrupt or SystemExit means the host is stopping
                # THIS shim, which is no evidence at all that the server is
                # bad. Killing it there would undo the start_new_session that
                # exists precisely so the server outlives the shim, and a host
                # that interrupts during cold start would loop forever, never
                # leaving a server behind for the next attempt.
                _terminate_spawned(process)
                raise
            if not ready:
                # Same distinction the child-died branch makes: our server
                # never came up, but someone else's may have during the wait.
                # Probe before concluding the port is unserved — then stop our
                # useless child either way, since it is ours and it is not the
                # one answering.
                if probe(port):
                    _terminate_spawned(process)
                    return port
                # Never leave a detached server the operator did not start and
                # cannot see. It holds the port, so the alternative is every
                # later shim inheriting this one's half-started process.
                _terminate_spawned(process)
                raise RuntimeError(
                    f"Noesis server did not become ready on port {port} within "
                    f"{ready_timeout:.0f}s (pid {process.pid}, since terminated); "
                    f"see {server_log_path(runtime_dir)}"
                )
            return port
    except Timeout as exc:
        raise RuntimeError(
            f"timed out after {ready_timeout:.0f}s waiting to elect a Noesis "
            f"server; another process holds {lock_path(runtime_dir)}"
        ) from exc


def run_shim(
    port: int,
    *,
    runtime_dir: Path | None = None,
    spawn: bool = True,
) -> None:
    """Serve stdio MCP by proxying to the shared server."""
    from fastmcp.server import create_proxy

    live_port = ensure_server(port, runtime_dir=runtime_dir, spawn=spawn)
    logger.info("proxying stdio MCP to %s", mcp_url(live_port))
    create_proxy(mcp_url(live_port), name="noesis").run()
