"""Real-process coverage for the shared-server shim's spawn path (issue #54,
gaps 2 and 4; ADR-86). Opt-in, like every ``@pytest.mark.server`` test:
run with ``docker compose up -d`` then ``uv run pytest -m server``.

``tests/test_shim.py`` proves the election logic with ``shim._spawn_server``
stubbed in every single test — a recorder that never spawns. That leaves the
real spawn path (the ``uvicorn`` argv, ``start_new_session=True``, ``stdin=
DEVNULL``, and the log-fd inheritance that lets the child keep writing after
the parent closes its handle) exercised only by manual runs, per the module
docstring in shim.py itself. This file is the real thing: a real ``Popen``
running real ``uvicorn`` against a real Qdrant, with nothing stubbed except
an observation side-channel (see ``_recording_spawn`` below, which wraps —
never replaces — the real ``_spawn_server``).

Why this needs a live Qdrant and the other ``server``-tier tests don't quite:
``uvicorn`` runs the ASGI lifespan — which is where ``noesis.app`` connects
to Qdrant, calls ``ensure_collection`` and sweeps orphan points — BEFORE it
binds the listening socket. Verified empirically while writing this file: a
second real ``uvicorn noesis.app:app`` pointed at an already-bound port does
NOT fail fast at bind() before touching Qdrant. It fully runs the lifespan
(Qdrant connect ~0.6s), THEN fails to bind with ``EADDRINUSE``, THEN spends
~9s more in shutdown (an embedder worker daemon thread does not stop within
uvicorn's 5s grace and is abandoned with a warning) before the process
actually exits — exit code 1, ~9.3s wall time from bind failure to exit,
measured 3x. So both the winner and the "loser" in gap 4's scenario need a
real, reachable Qdrant to reach that point at all; without one, neither
process would reach ``EADDRINUSE`` in a way that reproduces the scenario
issue #54 describes, they'd both fail identically at the Qdrant connection
instead.

Isolation: every test here writes its own ``config.toml`` (``NOESIS_CONFIG``)
pointing ``db_path`` at a throwaway file and ``[qdrant] collection`` at a
``uuid4``-suffixed name, and deletes that collection afterward — the same
discipline ``test_qdrant_server_smoke.py``'s ``store`` fixture uses, for the
same reason: this talks to whatever server the developer has running, which
is very likely the one holding their real index, and must never touch the
default ``noesis_chunks`` collection or the operator's real state DB.

Deliberately NOT covered here:

* The independent-election unit-level branch logic (probe-before-blaming-
  the-dead-child) — that's ``tests/test_shim.py::
  test_a_dead_child_does_not_fail_a_shim_that_has_a_server_to_talk_to``,
  with everything but the branch itself stubbed. This file proves the same
  property end-to-end with two real elections instead.
* Model loading. ``/healthz`` reports ``"status": "ok"`` as soon as the
  lifespan completes — the embedder loads lazily on first use (ADR-77) — so
  reaching a healthy real server here needs Qdrant but not a downloaded
  model.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import threading
import time
import uuid
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from noesis.mcp import shim

from .test_qdrant_server_smoke import QDRANT_URL, _server_version_or_skip

pytestmark = pytest.mark.server

# Generous over the measured numbers in the module docstring (~6-8s for a
# winner, ~9-15s for a loser's full connect-then-fail-then-shutdown cycle),
# with margin for a slower CI runner.
_READY_TIMEOUT_S = 60.0
_JOIN_TIMEOUT_S = 90.0


@pytest.fixture(scope="module", autouse=True)
def _require_live_qdrant() -> None:
    # Skip, not fail, on a dev machine without `docker compose up -d` — same
    # convention as test_qdrant_server_smoke.py. Under the `server-compat` CI
    # job a real Qdrant service container is always present, so this never
    # skips there.
    _server_version_or_skip()


def _free_port() -> int:
    """A port nothing is listening on. Bind-then-close, matching
    tests/test_shim.py's own helper (duplicated rather than imported: this
    file is meant to stand alone, like every other ``server``-tier test)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _write_config(base: Path, collection: str) -> Path:
    """A throwaway config.toml: isolated state DB, isolated Qdrant
    collection. Never the operator's real ``noesis_chunks``."""
    config_path = base / "config.toml"
    db_path = base / "state.sqlite"
    config_path.write_text(
        f'db_path = "{db_path.as_posix()}"\n'
        f"\n[qdrant]\n"
        f'url = "{QDRANT_URL}"\n'
        f'collection = "{collection}"\n'
    )
    return config_path


def _cleanup_collection(name: str) -> None:
    client = QdrantClient(url=QDRANT_URL)
    try:
        if client.collection_exists(name):
            client.delete_collection(name)
    finally:
        client.close()


def _kill_and_reap(process: subprocess.Popen[bytes]) -> None:
    """Best-effort teardown for a real detached server this test started.

    SIGKILL, not SIGTERM: these tests don't care about a graceful uvicorn
    shutdown (that ~9s drain is exactly what the module docstring measured),
    only about not leaking a process holding a port and ~1.5GB RSS."""
    if process.poll() is None:
        try:
            os.killpg(os.getsid(process.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


# --------------------------------------------------------------------------
# Gap 2: the real _spawn_server path
# --------------------------------------------------------------------------


def test_spawn_server_starts_a_real_detached_process_with_a_written_log(
    tmp_path, monkeypatch
):
    """Calls the real ``shim._spawn_server`` — nothing stubbed — and checks
    the three properties the module docstring in shim.py claims and issue
    #54 says were verified only by hand:

    1. Detached: ``start_new_session=True`` makes the child its own session
       leader, so a killed shim cannot take the server down with it.
    2. ``stdin=DEVNULL``: the child does not inherit the shim's stdin (which,
       for the real ``python -m noesis.mcp --shared`` entry point, is the
       stdio MCP transport itself — inheriting it would be a second reader
       racing the shim for those bytes).
    3. The log fd survives ``_spawn_server`` closing its own handle: real
       ``uvicorn`` startup output must land on disk after that close, not
       before it.
    """
    port = _free_port()
    collection = f"noesis_shim_test_{uuid.uuid4().hex[:12]}"
    monkeypatch.setenv("NOESIS_CONFIG", str(_write_config(tmp_path, collection)))
    runtime_dir = tmp_path / "runtime"

    process = shim._spawn_server(port, runtime_dir)
    try:
        deadline = time.monotonic() + _READY_TIMEOUT_S
        became_ready = False
        while time.monotonic() < deadline:
            if shim.probe(port):
                became_ready = True
                break
            assert process.poll() is None, (
                f"spawned server exited early with code {process.returncode} "
                f"before answering /healthz; see {shim.server_log_path(runtime_dir)}"
            )
            time.sleep(0.1)
        assert became_ready, (
            f"real server never answered /healthz within {_READY_TIMEOUT_S:.0f}s; "
            f"see {shim.server_log_path(runtime_dir)}"
        )

        assert os.getsid(process.pid) != os.getsid(0), (
            "spawned server shares this test process's session — "
            "start_new_session did not detach it"
        )

        stdin_link = Path(f"/proc/{process.pid}/fd/0")
        if not stdin_link.exists():
            pytest.skip("no /proc on this host to inspect the child's stdin fd")
        target = os.readlink(stdin_link)
        assert target == "/dev/null", (
            f"spawned server's stdin fd points at {target!r}, not /dev/null — "
            f"it would read from whatever this test's own stdin is"
        )

        log_text = shim.server_log_path(runtime_dir).read_text()
        assert "Uvicorn running" in log_text, (
            f"server.log has no uvicorn startup line after _spawn_server closed "
            f"its own handle to it — the child's dup'd fd did not survive: "
            f"{log_text!r}"
        )
    finally:
        _kill_and_reap(process)
        _cleanup_collection(collection)


# --------------------------------------------------------------------------
# Gap 4: two real runtime dirs, two real elections, one real server
# --------------------------------------------------------------------------


def test_two_independently_electing_runtime_dirs_converge_on_one_real_server(
    tmp_path, monkeypatch
):
    """The scenario round 8 fixed, end to end instead of stubbed.

    Two shims that resolved different ``runtime_dir``s (e.g. one process saw
    ``XDG_RUNTIME_DIR`` set, the other didn't) elect independently — each
    against its own real ``FileLock``, with no mutex between them — and both
    spawn a real ``uvicorn`` at nearly the same instant, targeting the SAME
    port. Exactly one can bind it. The other's real server, per the module
    docstring's measurements, still fully connects to Qdrant and completes
    its own ASGI startup before it discovers the port is taken, then takes
    on the order of 10-15s total to exit. ``_wait_ready``'s job is to notice
    that exit, re-probe, find the healthy incumbent, and return success
    rather than raising — so both ``ensure_server`` calls below must return
    the same port with no exception, exactly as
    ``tests/test_shim.py::test_a_dead_child_does_not_fail_a_shim_that_has_a_server_to_talk_to``
    asserts with everything but that branch stubbed.

    One thing this test caught empirically that the unit test's stubbed
    timing cannot: ``_wait_ready``'s FIRST check each loop is ``probe(port)``,
    checked before ``process.poll()``. So the loser's own ``ensure_server``
    call can return success as soon as the winner answers /healthz — without
    ever looking at whether the loser's OWN spawned process has exited yet.
    Both ``ensure_server`` calls therefore typically return well before the
    loser's process has gone through its own connect-then-EADDRINUSE-then-
    shutdown cycle (measured ~9-15s). The loser process is not leaked — it
    is still headed for its own exit, unmanaged, in the background — but
    "exactly one server alive" is an EVENTUAL property here, not one true
    the instant both threads join. Asserted below with a bounded poll for
    exactly that reason.

    ``_spawn_server`` is wrapped, not replaced, so every property gap 2
    checks in isolation is still real here — the wrapper only records the
    ``Popen`` handles this test needs for cleanup, since a winning
    ``ensure_server`` call deliberately returns just the port, never the
    process (the whole point of ``start_new_session`` is that the server
    outlives the caller, so nothing hands that handle back)."""
    port = _free_port()
    collection = f"noesis_shim_test_{uuid.uuid4().hex[:12]}"
    monkeypatch.setenv("NOESIS_CONFIG", str(_write_config(tmp_path, collection)))

    dir_a = tmp_path / "runtime_a"
    dir_b = tmp_path / "runtime_b"

    real_spawn = shim._spawn_server
    spawned: list[subprocess.Popen[bytes]] = []
    spawned_lock = threading.Lock()

    def recording_spawn(spawn_port: int, runtime_dir: Path) -> subprocess.Popen[bytes]:
        process = real_spawn(spawn_port, runtime_dir)
        with spawned_lock:
            spawned.append(process)
        return process

    monkeypatch.setattr(shim, "_spawn_server", recording_spawn)

    results: dict[str, object] = {}

    def run(name: str, runtime_dir: Path) -> None:
        try:
            results[name] = shim.ensure_server(
                port, runtime_dir=runtime_dir, ready_timeout=_READY_TIMEOUT_S
            )
        except BaseException as exc:  # recorded, not swallowed — asserted below
            results[name] = exc

    thread_a = threading.Thread(target=run, args=("a", dir_a))
    thread_b = threading.Thread(target=run, args=("b", dir_b))
    thread_a.start()
    thread_b.start()
    thread_a.join(timeout=_JOIN_TIMEOUT_S)
    thread_b.join(timeout=_JOIN_TIMEOUT_S)

    try:
        assert not thread_a.is_alive() and not thread_b.is_alive(), (
            "a starter never finished within the join timeout"
        )
        assert results.get("a") == port, f"runtime dir a: {results.get('a')!r}"
        assert results.get("b") == port, f"runtime dir b: {results.get('b')!r}"
        assert len(spawned) == 2, (
            f"expected exactly one real spawn per independent election "
            f"(2 total), got {len(spawned)}"
        )

        # Eventual, not immediate — see the docstring above: a losing
        # ensure_server call can return before its own duplicate process has
        # gone through its connect/EADDRINUSE/shutdown cycle. Bounded well
        # past the ~9-15s measured for that cycle to complete.
        settle_deadline = time.monotonic() + 30.0
        alive = [p for p in spawned if p.poll() is None]
        while len(alive) != 1 and time.monotonic() < settle_deadline:
            time.sleep(0.2)
            alive = [p for p in spawned if p.poll() is None]
        assert len(alive) == 1, (
            f"expected exactly one real server left holding the port after "
            f"letting the loser's own process settle, found {len(alive)} "
            f"still running 30s after both ensure_server calls returned"
        )
    finally:
        for process in spawned:
            _kill_and_reap(process)
        _cleanup_collection(collection)
