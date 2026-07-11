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
class QdrantSettings:
    url: str = DEFAULT_QUERY_URL
    collection: str = "noesis_chunks"


@dataclass(frozen=True)
class Settings:
    db_path: Path = field(default_factory=default_db_path)
    embedder: EmbedderSettings = field(default_factory=EmbedderSettings)
    reranker: RerankerSettings = field(default_factory=RerankerSettings)
    structural: StructuralSettings = field(default_factory=StructuralSettings)
    git: GitSettings = field(default_factory=GitSettings)
    qdrant: QdrantSettings = field(default_factory=QdrantSettings)


def _require_bool(value: object, key: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(
            f"config field {key!r} must be a boolean (true/false), got {value!r}"
        )
    return value


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
    qdr = raw.get("qdrant", {})
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
            dim=int(emb.get("dim", EmbedderSettings.dim)),
            batch_size=int(emb.get("batch_size", EmbedderSettings.batch_size)),
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
            candidates=int(rrk.get("candidates", RerankerSettings.candidates)),
            batch_size=int(rrk.get("batch_size", RerankerSettings.batch_size)),
            device=rrk.get("device", RerankerSettings.device),
        ),
        structural=StructuralSettings(
            max_results=int(stru.get("max_results", StructuralSettings.max_results)),
            timeout_s=float(stru.get("timeout_s", StructuralSettings.timeout_s)),
        ),
        git=GitSettings(
            fast_path=_require_bool(
                git.get("fast_path", GitSettings.fast_path), "git.fast_path"
            ),
        ),
        qdrant=QdrantSettings(
            url=qdr.get("url", QdrantSettings.url),
            collection=qdr.get("collection", QdrantSettings.collection),
        ),
    )
