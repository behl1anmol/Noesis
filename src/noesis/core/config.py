"""Runtime settings, read once at startup from ``config.toml`` (§3.7).

M2 added ``[embedder]`` and ``[qdrant]`` plus the state-DB path; M4 added
``[reranker]``; M5 adds ``[structural]``; M7 adds ``[git]``. Everything has
a working default so the service runs with no config file at all.

Path defaults are anchored, never cwd-relative: the HTTP server (run from a
checkout) and the stdio MCP server (spawned with the agent host's cwd) must
resolve the SAME state DB, or the MCP tools silently answer "unknown
project_id" for every project registered via the dashboard. Same failure
class — and same cure — as prefetch.default_fastembed_cache.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_QUERY_URL = "http://127.0.0.1:6333"

# Explicit config-file override for hosts that can't control their cwd
# (e.g. an agent host's MCP server entry): NOESIS_CONFIG=/path/to/config.toml
CONFIG_ENV = "NOESIS_CONFIG"


def default_db_path() -> Path:
    """Absolute, cwd-independent default location for the state DB
    (user data dir, ``~/.local/share/noesis/noesis.sqlite``)."""
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base).expanduser() / "noesis" / "noesis.sqlite"


def default_config_path() -> Path:
    """Config lookup when neither an explicit path nor $NOESIS_CONFIG is
    given: ``./config.toml`` when present (deliberate dev override for
    running from a checkout), else the anchored user config dir."""
    cwd_cfg = Path("config.toml")
    if cwd_cfg.is_file():
        return cwd_cfg
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base).expanduser() / "noesis" / "config.toml"


@dataclass(frozen=True)
class EmbedderSettings:
    model: str = "nomic-ai/CodeRankEmbed"
    dim: int = 768
    batch_size: int = 32
    # None → auto-detect (cuda→mps→cpu); set e.g. "cuda"/"cpu" to pin. The
    # model runner resolves and logs the device (core/compute.py).
    device: str | None = None


@dataclass(frozen=True)
class RerankerSettings:
    """§3.7 ``[reranker]``. ``enabled`` is both the kill switch and the
    per-request default: ``enabled=false`` (the pre-gate default, Finding 2)
    never loads the model and requests cannot opt in; ``enabled=true`` makes
    ``rerank`` default on with per-request opt-out. The M4 gate decision
    flips the shipped default from measured NDCG@10 data."""

    model: str = "BAAI/bge-reranker-v2-m3"
    enabled: bool = False
    preload: bool = False
    candidates: int = 50
    batch_size: int = 16
    # None → auto-detect (cuda→mps→cpu); pin to force. On CPU the cross-encoder
    # cannot meet the p95 budget (M4 gate), which is why it ships default-off.
    device: str | None = None


@dataclass(frozen=True)
class StructuralSettings:
    """§3.7 ``[structural]``. ``max_results`` caps matches per query (the
    request may ask for less, never more); ``timeout_s`` is the wall-clock
    scan budget — on expiry the scan stops and returns partial results with
    ``timed_out: true`` rather than erroring, since partial matches are
    still actionable to an iterating agent."""

    max_results: int = 100
    timeout_s: float = 10.0


@dataclass(frozen=True)
class GitSettings:
    """§3.7 ``[git]``. ``fast_path=false`` disables candidate narrowing
    entirely — every run full hash-walks (the correctness baseline the
    fast path must match, §3.2 rule 1)."""

    fast_path: bool = True


@dataclass(frozen=True)
class IndexingSettings:
    """§3.7 ``[indexing]`` — the automatic path back to a full, anchored walk
    (ADR-56/57). Every automated trigger in the service is *scoped*, and only a
    full run drains the H1 dirty set or advances the anchor, so without these a
    project has no way out of a latched state (issue #25).

    ``promote_after_scoped_runs``: promote a scoped run to a full walk once
    this many scoped runs have started since the last completed full one.
    ``promote_candidate_fraction``: promote when the effective candidate set
    (pending ∪ dirty) reaches this fraction of the indexed file count — the
    point where a scoped run stops being cheaper than the full walk it already
    pays for, since discovery stats and binary-sniffs every file either way.
    ``unwalkable_quarantine_runs``: consecutive runs a directory must fail to
    be walked before the paths it hides stop being re-queued for retry.

    ``0`` disables each independently, restoring the pre-ADR-56 behaviour.

    ``max_concurrent_index_runs`` (ADR-85) caps index runs across the whole
    machine, not per project — ``try_start_run`` already guarantees one run
    per project, and this bounds how many projects may run at once. The
    default of 4 is a judgment call, NOT a measurement: concurrent runs
    serialize on the single embedder worker (ADR-20) regardless, so a larger
    number buys no indexing throughput and only multiplies simultaneous file
    walks, open descriptors and executor pressure; a value of 1 would be a
    visible behaviour change, since today a second project need not wait
    behind a long run. Set it explicitly if your machine says otherwise.
    """

    promote_after_scoped_runs: int = 20
    promote_candidate_fraction: float = 0.25
    unwalkable_quarantine_runs: int = 5
    max_concurrent_index_runs: int = 4


@dataclass(frozen=True)
class WatcherSettings:
    """§3.7 ``[watcher]``. ``poll_interval_s`` is the PollingObserver
    snapshot cadence used only for watched roots on inotify-blind
    filesystems (9p/cifs/nfs/fuse/…, e.g. WSL2's ``/mnt/c``); roots watched
    natively via inotify never poll and ignore it."""

    poll_interval_s: float = 1.0


@dataclass(frozen=True)
class ServerSettings:
    """§3.8 ``[server]``. Only the port is configurable: the host is fixed at
    127.0.0.1 by CLAUDE.md rule 2, and exposing it as a knob would make a
    wildcard bind one config edit away. The shim (ADR-86) reads this to know
    where to find — or start — the shared server, so the port has to live
    somewhere both it and the operator agree on."""

    port: int = 8000


@dataclass(frozen=True)
class QdrantSettings:
    """§3.3 ``[qdrant]``.

    ``query_connections`` sizes the search connection pool and, 1:1, the
    search executor's slots (ADR-83/84). ``None`` means derive from the
    machine — the same "None means auto" convention ``embedder.device``
    already uses — because the right value is a property of the host, not
    of the project. See ``derive_query_connections`` for the rule and the
    measurements behind it. Set it explicitly to override on a machine
    whose shape the rule guesses wrong.

    ``query_queue_depth`` bounds how many searches may WAIT for a slot
    before the service starts rejecting instead of queueing without limit.
    ``None`` derives it from ``query_connections``."""

    url: str = DEFAULT_QUERY_URL
    collection: str = "noesis_chunks"
    query_connections: int | None = None
    query_queue_depth: int | None = None


def available_cpus() -> int:
    """CPUs this process may actually run on.

    ``os.cpu_count()`` reports the host's CPUs, which is wrong inside a
    cgroup-limited container — it would size the pool for hardware the
    process cannot use. ``sched_getaffinity`` reports the real allowance
    where it exists (Linux); everything else falls back."""
    getaffinity = getattr(os, "sched_getaffinity", None)
    if getaffinity is not None:
        try:
            return max(1, len(getaffinity(0)))
        except OSError:
            pass
    return max(1, os.cpu_count() or 1)


# Bounds for the derived pool size. The floor is measured: K=1 costs ~33%
# throughput (258 q/s vs 384) because a single connection cannot overlap a
# query with the next one's round trip. The ceiling is a judgment call, not
# a measurement — see derive_query_connections.
_MIN_QUERY_CONNECTIONS = 2
_MAX_QUERY_CONNECTIONS = 8
# Waiters allowed per slot before rejecting. Four keeps a burst absorbable
# while bounding the wait a caller can be made to sit through to roughly
# four service times — tens of milliseconds at measured p50 — rather than
# letting an unbounded queue turn overload into an unbounded stall, which
# is the failure mode this whole change exists to remove.
_QUEUE_DEPTH_PER_CONNECTION = 4


def derive_query_connections(configured: int | None = None) -> int:
    """Resolve the search pool size; *configured* wins when given.

    Rule: ``clamp(available_cpus(), 2, 8)``.

    Measured on a 4-CPU box, 16 concurrent searches, 5,000 points, hybrid,
    prefetch 50, against a live Qdrant 1.18.3 — throughput saturates
    exactly at ``cpu_count`` and stays flat well past it while latency
    climbs monotonically:

        K=1  258 q/s  p95  4.7ms     K=6   375 q/s  p95 23.7ms
        K=2  353 q/s  p95  8.1ms     K=8   375 q/s  p95 33.3ms
        K=4  384 q/s  p95 16.0ms     K=12  379 q/s  p95 48.4ms

    The knee sits at the CPU count because Qdrant is co-resident and
    competing for the same cores: past it, queries queue inside Qdrant
    instead of in our pool, buying latency for no throughput.

    The ceiling of 8 is explicitly NOT measured — a bigger box was not
    available — which is precisely why the override exists."""
    if configured is not None:
        return configured
    return max(
        _MIN_QUERY_CONNECTIONS, min(_MAX_QUERY_CONNECTIONS, available_cpus())
    )


def derive_query_queue_depth(connections: int, configured: int | None = None) -> int:
    """Resolve how many searches may queue for a slot; *configured* wins."""
    if configured is not None:
        return configured
    return connections * _QUEUE_DEPTH_PER_CONNECTION


@dataclass(frozen=True)
class Settings:
    db_path: Path = field(default_factory=default_db_path)
    embedder: EmbedderSettings = field(default_factory=EmbedderSettings)
    reranker: RerankerSettings = field(default_factory=RerankerSettings)
    structural: StructuralSettings = field(default_factory=StructuralSettings)
    git: GitSettings = field(default_factory=GitSettings)
    watcher: WatcherSettings = field(default_factory=WatcherSettings)
    indexing: IndexingSettings = field(default_factory=IndexingSettings)
    qdrant: QdrantSettings = field(default_factory=QdrantSettings)
    server: ServerSettings = field(default_factory=ServerSettings)


def _require_bool(value: object, key: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(
            f"config field {key!r} must be a boolean (true/false), got {value!r}"
        )
    return value


def _require_positive_int(value: object, key: str) -> int:
    n = int(value)
    if n <= 0:
        raise ValueError(f"config field {key!r} must be > 0, got {n!r}")
    return n


def _require_positive_float(value: object, key: str) -> float:
    x = float(value)
    if x <= 0:
        raise ValueError(f"config field {key!r} must be > 0, got {x!r}")
    return x


def _require_non_negative_int(value: object, key: str) -> int:
    n = int(value)
    if n < 0:
        raise ValueError(f"config field {key!r} must be >= 0, got {n!r}")
    return n


def _optional_positive_int(value: object, key: str) -> int | None:
    """A knob whose absence means "derive it". Distinguishes "not set" from
    a set-but-invalid value, which must still be rejected loudly rather
    than silently falling back to the derived default."""
    if value is None:
        return None
    return _require_positive_int(value, key)


def _require_fraction(value: object, key: str) -> float:
    """A 0..1 proportion. The upper bound is real, not cosmetic: a value above
    1 can never be reached by a candidate set bounded by the index size, so it
    would silently mean "never promote" while reading as a tuned threshold.
    Use 0 to disable, which says so."""
    x = float(value)
    if not 0.0 <= x <= 1.0:
        raise ValueError(f"config field {key!r} must be between 0 and 1, got {x!r}")
    return x


def load_settings(config_path: str | Path | None = None) -> Settings:
    """Load settings, falling back to defaults per key.

    Resolution order: explicit *config_path* → ``$NOESIS_CONFIG`` →
    ``./config.toml`` (dev override) → ``$XDG_CONFIG_HOME/noesis/config.toml``.
    A relative ``db_path`` inside the file resolves against the file's own
    directory, never the process cwd — the cwd belongs to whoever spawned us.
    """
    if config_path is None:
        config_path = os.environ.get(CONFIG_ENV) or default_config_path()
    # expanduser first: an MCP host commonly sets NOESIS_CONFIG (or passes an
    # explicit path) as a literal "~/..." that no shell expanded. Without this
    # the is_file() check misses the real config and silently falls back to
    # zero-config defaults — reopening the default DB and reintroducing the
    # exact silent divergence ADR-44 closes.
    path = Path(config_path).expanduser()
    if not path.is_file():
        return Settings()
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    emb = raw.get("embedder", {})
    rrk = raw.get("reranker", {})
    stru = raw.get("structural", {})
    git = raw.get("git", {})
    wat = raw.get("watcher", {})
    idx = raw.get("indexing", {})
    qdr = raw.get("qdrant", {})
    srv = raw.get("server", {})
    raw_db = raw.get("db_path")
    if raw_db is None:
        db_path = default_db_path()
    else:
        db_path = Path(raw_db).expanduser()
        if not db_path.is_absolute():
            db_path = (path.resolve().parent / db_path).resolve()
    return Settings(
        db_path=db_path,
        embedder=EmbedderSettings(
            model=emb.get("model", EmbedderSettings.model),
            dim=_require_positive_int(
                emb.get("dim", EmbedderSettings.dim), "embedder.dim"
            ),
            batch_size=_require_positive_int(
                emb.get("batch_size", EmbedderSettings.batch_size),
                "embedder.batch_size",
            ),
            device=emb.get("device", EmbedderSettings.device),
        ),
        reranker=RerankerSettings(
            model=rrk.get("model", RerankerSettings.model),
            enabled=_require_bool(
                rrk.get("enabled", RerankerSettings.enabled), "reranker.enabled"
            ),
            preload=_require_bool(
                rrk.get("preload", RerankerSettings.preload), "reranker.preload"
            ),
            candidates=_require_positive_int(
                rrk.get("candidates", RerankerSettings.candidates),
                "reranker.candidates",
            ),
            batch_size=_require_positive_int(
                rrk.get("batch_size", RerankerSettings.batch_size),
                "reranker.batch_size",
            ),
            device=rrk.get("device", RerankerSettings.device),
        ),
        structural=StructuralSettings(
            max_results=_require_positive_int(
                stru.get("max_results", StructuralSettings.max_results),
                "structural.max_results",
            ),
            timeout_s=_require_positive_float(
                stru.get("timeout_s", StructuralSettings.timeout_s),
                "structural.timeout_s",
            ),
        ),
        git=GitSettings(
            fast_path=_require_bool(
                git.get("fast_path", GitSettings.fast_path), "git.fast_path"
            ),
        ),
        watcher=WatcherSettings(
            poll_interval_s=_require_positive_float(
                wat.get("poll_interval_s", WatcherSettings.poll_interval_s),
                "watcher.poll_interval_s",
            ),
        ),
        indexing=IndexingSettings(
            promote_after_scoped_runs=_require_non_negative_int(
                idx.get(
                    "promote_after_scoped_runs",
                    IndexingSettings.promote_after_scoped_runs,
                ),
                "indexing.promote_after_scoped_runs",
            ),
            promote_candidate_fraction=_require_fraction(
                idx.get(
                    "promote_candidate_fraction",
                    IndexingSettings.promote_candidate_fraction,
                ),
                "indexing.promote_candidate_fraction",
            ),
            unwalkable_quarantine_runs=_require_non_negative_int(
                idx.get(
                    "unwalkable_quarantine_runs",
                    IndexingSettings.unwalkable_quarantine_runs,
                ),
                "indexing.unwalkable_quarantine_runs",
            ),
            max_concurrent_index_runs=_require_positive_int(
                idx.get(
                    "max_concurrent_index_runs",
                    IndexingSettings.max_concurrent_index_runs,
                ),
                "indexing.max_concurrent_index_runs",
            ),
        ),
        qdrant=QdrantSettings(
            url=qdr.get("url", QdrantSettings.url),
            collection=qdr.get("collection", QdrantSettings.collection),
            query_connections=_optional_positive_int(
                qdr.get("query_connections"), "qdrant.query_connections"
            ),
            query_queue_depth=_optional_positive_int(
                qdr.get("query_queue_depth"), "qdrant.query_queue_depth"
            ),
        ),
        server=ServerSettings(
            port=_require_positive_int(
                srv.get("port", ServerSettings.port), "server.port"
            ),
        ),
    )
