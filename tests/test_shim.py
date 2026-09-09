"""Default-suite regression coverage for the shared-server shim (ADR-86).

The shim is the one place in Noesis that spawns a detached OS process, and
every property that makes that safe is a property of failure paths that a
happy-path run never touches. What this proves:

* **The election really elects.** N threads calling ``ensure_server`` at once
  produce exactly ONE spawn, against a real ``FileLock`` in a real temp
  runtime dir. The stub server is held un-ready until every caller has passed
  the fast-path probe, so the losers are genuinely racing rather than
  arriving after the winner has already finished — without that, a shim with
  no lock at all would pass.
* **The warm path costs nothing.** With a server already answering,
  ``ensure_server`` returns while another thread *holds* the election lock.
  That is the 0.07 s claim in the module docstring stated as behaviour: if
  the fast path ever moved below the lock, every warm start would queue
  behind every cold one.
* **The readiness budget is not spent waiting for the lock** (B6). Measured
  as the budget ``_wait_ready`` is actually handed, because the bug is
  arithmetic, not timing: a starter that burns the full 60 s leaves the next
  shim with a deadline already in the past, so it spawns a server and
  declares it dead in the same instant.
* **A server this shim starts and cannot use is stopped** (B7), and **a child
  that dies is noticed in the wait loop rather than waited out** (B8), with
  the log path in the message. Both are about what the operator is left with:
  an orphaned uvicorn nobody started, or a 60 s stall over a process that
  announced its own failure in its first second.
* **``probe`` identifies Noesis, not "port 8000 answered"** (B9). Checked
  against a real loopback HTTP server, so the JSON parse, the status code and
  the connection-refused path are the real ones. Both directions are probed:
  a genuine ``/healthz`` body must still return True, or the "fix" is just a
  probe that never succeeds.

Why this tier: none of it needs Qdrant, a model, a network or uvicorn.
``_spawn_server`` is stubbed everywhere a spawn would happen — a test that
started a real server would be testing the server, and would leave one
running when it failed. Ports are only ever bound to be released again, to
get a number nothing is listening on.

Deliberately NOT tested here:

* ``run_shim`` and the ``fastmcp`` proxy it constructs. Past
  ``ensure_server`` the shim is two lines of library wiring; asserting them
  would mean stubbing ``create_proxy`` to check we called it.
* That the spawned command line actually boots Noesis (``-m uvicorn
  noesis.app:app``). That is an integration property of the app, and the one
  argument that matters here for safety — ``--host 127.0.0.1``, CLAUDE.md
  rule 2 — is a literal in the source, not something a stubbed spawn can
  observe.
* ``default_runtime_dir``'s XDG resolution: environment plumbing shared with
  ``prefetch``, and every test here passes ``runtime_dir`` explicitly so that
  no test can ever touch the operator's real lock file.
* Real SIGTERM/SIGKILL delivery in ``_terminate_spawned``. The fake process
  records which of ``terminate``/``kill`` it was sent; that a live uvicorn
  honours SIGTERM is uvicorn's contract, and testing it would mean starting
  the very server this file refuses to start.
"""

from __future__ import annotations

import contextlib
import json
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from filelock import FileLock

from noesis.mcp import shim


def _free_port() -> int:
    """A port number nothing is listening on.

    Bind-then-close rather than a hardcoded number: a literal port would make
    these tests fail on a machine that happens to run something there, which
    is the exact confusion B9 is about."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextlib.contextmanager
def _stub_health(
    body: bytes, *, status: int = 200, content_type: str = "application/json"
):
    """Serve *body* on 127.0.0.1 for the life of the block; yields the port.

    A real loopback server rather than a monkeypatched ``httpx.get`` so that
    the status code, the transport error and the JSON decode in ``probe`` are
    the real ones — a stubbed response object could not tell a body that
    fails to parse from one that parses to the wrong thing."""

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib naming
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield int(server.server_address[1])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class _FakeProcess:
    """Stands in for the detached ``uvicorn`` ``Popen``.

    ``exit_code`` None means "still running"; set it to a number to simulate
    a server that died on startup. Records the signals it was sent, which is
    the only observable B7 has."""

    def __init__(self, exit_code: int | None = None) -> None:
        self.pid = 424242
        self.exit_code = exit_code
        self.terminated = False
        self.killed = False
        self.waited = False

    def poll(self) -> int | None:
        return self.exit_code

    def terminate(self) -> None:
        self.terminated = True
        self.exit_code = -15

    def kill(self) -> None:
        self.killed = True
        self.exit_code = -9

    def wait(self, timeout: float | None = None) -> int:
        self.waited = True
        if self.exit_code is None:
            raise AssertionError("wait() on a fake process that never exits")
        return self.exit_code


class _SpawnRecorder:
    """A ``_spawn_server`` replacement that counts calls and never spawns."""

    def __init__(self, process_factory=lambda: _FakeProcess()) -> None:
        self._factory = process_factory
        self._lock = threading.Lock()
        self.calls: list[int] = []
        self.processes: list[_FakeProcess] = []

    def __call__(self, port: int, runtime_dir: Path) -> _FakeProcess:
        process = self._factory()
        with self._lock:
            self.calls.append(port)
            self.processes.append(process)
        return process

    @property
    def count(self) -> int:
        with self._lock:
            return len(self.calls)


# --------------------------------------------------------------------------
# probe: what counts as "Noesis is running here" (B9)
# --------------------------------------------------------------------------


def test_probe_accepts_a_real_healthz_body():
    """Non-vacuity for the three refusals below: the identity check must
    still say yes to what ``api/routes.py`` actually returns. A probe that
    refused everything would pass every negative test in this file and break
    every cold start in production.

    Watched failing against a probe made over-strict (demanding a ``service``
    key ``/healthz`` does not return): ``assert False is True``."""
    body = json.dumps(
        {"status": "ok", "assets": "unknown", "embedder_ready": "unknown"}
    ).encode()
    with _stub_health(body) as port:
        assert shim.probe(port) is True


def test_probe_is_false_when_the_connection_is_refused():
    """The contract is "never raises", so nothing listening is a False, not
    an OSError escaping into the election.

    This one passes against the pre-fix probe too — it guards the older
    never-raises contract, not B9. Watched failing with ``httpx.HTTPError``
    dropped from the except clause: ``httpx.ConnectError: [Errno 111]
    Connection refused`` propagating out of ``probe``."""
    assert shim.probe(_free_port(), timeout=1.0) is False


def test_probe_is_false_for_a_200_that_is_not_noesis():
    """The bug B9 fixes: any HTTP 200 on the port was taken as Noesis. A
    Grafana, a Vite dev server or another project's API on 8000 would have
    had this shim proxy an agent's searches to it — the same silent
    wrong-server bug the endpoint file was deleted to prevent.

    Watched failing against the pre-fix ``return response.status_code == 200``:
    ``assert True is False``."""
    with _stub_health(
        json.dumps({"service": "definitely-not-noesis"}).encode()
    ) as port:
        assert shim.probe(port) is False


def test_probe_is_false_for_a_200_with_a_non_json_body():
    """A body that does not parse must be a refusal, not an exception: this
    runs inside the election, where a raise means no server gets started at
    all rather than one being started twice.

    Watched failing against the pre-fix probe: ``assert True is False`` — the
    HTML body was accepted as a Noesis server."""
    with _stub_health(
        b"<html>hello from somebody else</html>", content_type="text/html"
    ) as port:
        assert shim.probe(port) is False


def test_probe_is_false_for_a_non_200():
    """A Noesis-shaped body behind a 503 is not a server to proxy to. Like
    the refused connection, this held pre-fix as well; watched failing with
    the status-code check deleted: ``assert True is False``."""
    with _stub_health(json.dumps({"status": "ok"}).encode(), status=503) as port:
        assert shim.probe(port) is False


# --------------------------------------------------------------------------
# ensure_server: the election
# --------------------------------------------------------------------------


def test_concurrent_starters_produce_exactly_one_server(tmp_path, monkeypatch):
    """Ten threads, one spawn — against the real ``FileLock``.

    The stub server is held un-ready until every one of the ten has passed
    the fast-path probe, so all ten are inside the election at once. Without
    that gate the winner could finish before the others even looked, and a
    build with no lock at all would still spawn once.

    Watched failing with the ``FileLock`` replaced by a ``nullcontext``:
    "10 servers were spawned by 10 concurrent starters"."""
    starters = 10
    port = _free_port()
    ready = threading.Event()
    all_arrived = threading.Event()
    entered = threading.Semaphore(0)
    seen: set[int] = set()
    seen_lock = threading.Lock()

    def fake_probe(probe_port: int, timeout: float = 1.0) -> bool:
        assert probe_port == port
        tid = threading.get_ident()
        with seen_lock:
            first = tid not in seen
            seen.add(tid)
        if first:
            entered.release()
        return ready.is_set()

    recorder = _SpawnRecorder()

    def fake_spawn(spawn_port: int, runtime_dir: Path) -> _FakeProcess:
        # Refuse to become ready until every caller is past the fast path,
        # so "one spawn" cannot be an artifact of the losers arriving late.
        assert all_arrived.wait(timeout=10), "callers never all reached the election"
        process = recorder(spawn_port, runtime_dir)
        ready.set()
        return process

    monkeypatch.setattr(shim, "probe", fake_probe)
    monkeypatch.setattr(shim, "_spawn_server", fake_spawn)

    results: list[object] = []
    results_lock = threading.Lock()

    def call() -> None:
        try:
            value: object = shim.ensure_server(
                port, runtime_dir=tmp_path, ready_timeout=10.0
            )
        except BaseException as exc:  # recorded, not swallowed - asserted below
            value = exc
        with results_lock:
            results.append(value)

    threads = [threading.Thread(target=call) for _ in range(starters)]
    for thread in threads:
        thread.start()
    for _ in range(starters):
        assert entered.acquire(timeout=10), "a starter never probed the port"
    all_arrived.set()
    for thread in threads:
        thread.join(timeout=20)
        assert not thread.is_alive()

    assert results == [port] * starters, f"not every starter got the port: {results!r}"
    assert recorder.count == 1, (
        f"{recorder.count} servers were spawned by {starters} concurrent "
        f"starters; the election is not electing"
    )
    assert shim.lock_path(tmp_path).exists(), "the election used the real lock file"


def test_warm_start_returns_without_taking_the_election_lock(tmp_path, monkeypatch):
    """The fast path is asserted by contention, not by a stopwatch: another
    thread holds the lock for the whole call, so a shim that probed below the
    lock could not return at all. ``ready_timeout`` is short so that such a
    shim fails in a second instead of hanging the suite.

    Watched failing with the fast-path probe deleted: ``RuntimeError: timed
    out after 1s waiting to elect a Noesis server``."""
    port = _free_port()
    recorder = _SpawnRecorder()
    monkeypatch.setattr(shim, "probe", lambda p, timeout=1.0: True)
    monkeypatch.setattr(shim, "_spawn_server", recorder)

    holding = threading.Event()
    release = threading.Event()

    def hold_the_lock() -> None:
        with FileLock(str(shim.lock_path(tmp_path)), timeout=10):
            holding.set()
            release.wait(timeout=15)

    holder = threading.Thread(target=hold_the_lock)
    holder.start()
    try:
        assert holding.wait(timeout=10), "helper never acquired the election lock"
        started = time.monotonic()
        assert shim.ensure_server(port, runtime_dir=tmp_path, ready_timeout=1.0) == port
        elapsed = time.monotonic() - started
    finally:
        release.set()
        holder.join(timeout=10)

    assert recorder.count == 0, "a warm start must not spawn anything"
    assert elapsed < 0.5, (
        f"warm start took {elapsed:.2f}s while another thread held the "
        f"election lock — it is queueing behind the lock instead of "
        f"answering from the probe"
    )


def test_no_spawn_refuses_with_the_health_url_and_starts_nothing(tmp_path, monkeypatch):
    """``--no-spawn`` is for operators running the server themselves, so the
    refusal has to say which URL was tried — otherwise the one actionable
    fact (they configured a different port than the server listens on) is
    exactly what the message omits.

    Watched failing with the URL dropped from the message: ``assert
    'http://127.0.0.1:60423/healthz' in 'no Noesis server answering and
    --no-spawn was given. ...'``."""
    port = _free_port()
    recorder = _SpawnRecorder()
    monkeypatch.setattr(shim, "probe", lambda p, timeout=1.0: False)
    monkeypatch.setattr(shim, "_spawn_server", recorder)

    with pytest.raises(RuntimeError) as excinfo:
        shim.ensure_server(port, runtime_dir=tmp_path, spawn=False)

    message = str(excinfo.value)
    assert shim.health_url(port) in message, (
        f"the refusal must name the health URL it probed: {message!r}"
    )
    assert "--no-spawn" in message
    assert recorder.count == 0, "--no-spawn spawned a server"
    assert not shim.lock_path(tmp_path).exists(), (
        "the refusal took the election lock on the way out"
    )


# --------------------------------------------------------------------------
# ensure_server: the failure paths (B6, B7, B8)
# --------------------------------------------------------------------------


def test_the_readiness_budget_is_measured_from_the_lock_not_from_entry(
    tmp_path, monkeypatch
):
    """B6: time queued behind another shim must not be deducted from the time
    the server gets to start.

    Asserted on the budget ``_wait_ready`` is handed rather than on a wall
    clock, because the defect is arithmetic and a timing assertion would be
    measuring the machine. The lock is held for 0.5 s against a 5 s budget —
    a 10x margin over the 0.05 s epsilon below — so the pre-fix ~4.5 s and
    the post-fix ~5.0 s cannot be confused. The lower-bound assertion on
    ``waited`` keeps the test honest: if the call never blocked on the lock,
    a full budget would prove nothing.

    Watched failing with the deadline moved back above the lock: "the server
    was given 4.49s of a 5s readiness budget after 0.51s spent waiting for
    the election lock"."""
    port = _free_port()
    lock_hold_s = 0.5
    ready_timeout = 5.0
    budget: list[float] = []

    def recording_wait_ready(process, wait_port, deadline, runtime_dir) -> bool:
        budget.append(deadline - time.monotonic())
        return True

    monkeypatch.setattr(shim, "probe", lambda p, timeout=1.0: False)
    monkeypatch.setattr(shim, "_spawn_server", _SpawnRecorder())
    monkeypatch.setattr(shim, "_wait_ready", recording_wait_ready)

    holding = threading.Event()

    def hold_the_lock() -> None:
        with FileLock(str(shim.lock_path(tmp_path)), timeout=10):
            holding.set()
            time.sleep(lock_hold_s)

    holder = threading.Thread(target=hold_the_lock)
    holder.start()
    try:
        assert holding.wait(timeout=10), "helper never acquired the election lock"
        started = time.monotonic()
        assert (
            shim.ensure_server(port, runtime_dir=tmp_path, ready_timeout=ready_timeout)
            == port
        )
        waited = time.monotonic() - started
    finally:
        holder.join(timeout=10)

    assert waited >= lock_hold_s * 0.8, (
        f"the call returned in {waited:.2f}s without ever contending for the "
        f"lock, so the budget below proves nothing"
    )
    assert budget, "_wait_ready was never called"
    assert budget[0] > ready_timeout - 0.05, (
        f"the server was given {budget[0]:.2f}s of a {ready_timeout:.0f}s "
        f"readiness budget after {waited:.2f}s spent waiting for the election "
        f"lock — the lock wait is being charged to the server's startup"
    )


def test_a_server_that_never_answers_is_terminated_before_the_error(
    tmp_path, monkeypatch
):
    """B7: the failure path must not leave a detached uvicorn behind.

    Whatever went wrong, this shim started that process, it is holding the
    port, and the operator has no reason to look for it. Raising while it
    runs also poisons every later shim: the next one probes a half-started
    server it did not start.

    Watched failing with both ``_terminate_spawned`` calls removed: "the
    spawned server was left running after ensure_server gave up on it"."""
    port = _free_port()
    process = _FakeProcess(exit_code=None)  # alive, just never ready
    recorder = _SpawnRecorder(process_factory=lambda: process)
    monkeypatch.setattr(shim, "probe", lambda p, timeout=1.0: False)
    monkeypatch.setattr(shim, "_spawn_server", recorder)

    with pytest.raises(RuntimeError) as excinfo:
        shim.ensure_server(port, runtime_dir=tmp_path, ready_timeout=0.2)

    message = str(excinfo.value)
    assert "did not become ready" in message
    assert str(shim.server_log_path(tmp_path)) in message
    assert process.terminated, (
        "the spawned server was left running after ensure_server gave up on "
        "it — a detached process the operator never started and cannot see"
    )
    assert process.waited, "the terminated child was never reaped"
    assert not process.killed, (
        "a process that exits on SIGTERM must not also be SIGKILLed"
    )


def test_a_child_that_dies_fails_fast_and_names_the_log(tmp_path, monkeypatch):
    """B8: a server that exits on startup is noticed, not waited out.

    EADDRINUSE, an import error or a Qdrant that is down all produce a child
    that is gone in under a second and a port that never answers. Polling
    only the port turns that into the full readiness budget of silence and
    then a timeout message that blames slowness. The 3 s budget against a
    1 s assertion is the margin: pre-fix this test takes the whole 3 s.

    Watched failing with the child-poll removed from the wait loop: "took
    3.16s to report a child that was already dead when the first probe
    failed"."""
    port = _free_port()
    process = _FakeProcess(exit_code=1)  # died immediately, as on EADDRINUSE
    recorder = _SpawnRecorder(process_factory=lambda: process)
    monkeypatch.setattr(shim, "probe", lambda p, timeout=1.0: False)
    monkeypatch.setattr(shim, "_spawn_server", recorder)

    started = time.monotonic()
    with pytest.raises(RuntimeError) as excinfo:
        shim.ensure_server(port, runtime_dir=tmp_path, ready_timeout=3.0)
    elapsed = time.monotonic() - started

    message = str(excinfo.value)
    assert elapsed < 1.0, (
        f"took {elapsed:.2f}s to report a child that was already dead when "
        f"the first probe failed; the wait loop is not watching the process"
    )
    assert "exited" in message, f"the message must say the child died: {message!r}"
    assert "1" in message, f"the exit code is the diagnosis: {message!r}"
    assert str(shim.server_log_path(tmp_path)) in message, (
        f"the only place the reason exists is the server log, so the error "
        f"has to name it: {message!r}"
    )
    assert not process.killed, "a child that is already dead must not be signalled"


def test_a_server_that_ignores_sigterm_is_killed(tmp_path, monkeypatch):
    """The bounded wait in ``_terminate_spawned`` exists because the process
    being stopped is one that failed to start: it may not have reached the
    signal handler that makes SIGTERM mean anything. Terminate-and-hope would
    leave exactly the orphan B7 is about.

    Watched failing with the cleanup removed: neither ``terminate`` nor
    ``kill`` was ever sent."""
    port = _free_port()

    class _Stubborn(_FakeProcess):
        def terminate(self) -> None:
            self.terminated = True  # ignores it: still running afterwards

        def wait(self, timeout: float | None = None) -> int:
            self.waited = True
            if self.exit_code is None:
                raise subprocess.TimeoutExpired(cmd="uvicorn", timeout=timeout or 0)
            return self.exit_code

    process = _Stubborn(exit_code=None)
    monkeypatch.setattr(shim, "probe", lambda p, timeout=1.0: False)
    monkeypatch.setattr(shim, "_TERMINATE_GRACE_S", 0.05)
    monkeypatch.setattr(
        shim, "_spawn_server", _SpawnRecorder(process_factory=lambda: process)
    )

    with pytest.raises(RuntimeError):
        shim.ensure_server(port, runtime_dir=tmp_path, ready_timeout=0.2)

    assert process.terminated and process.killed, (
        "a spawned server that ignores SIGTERM must be killed, or the "
        "failure path still leaks the process it meant to clean up"
    )
