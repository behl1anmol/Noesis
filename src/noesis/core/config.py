"""Runtime settings, read once at startup from ``config.toml`` (§3.7).

M2 added ``[embedder]`` and ``[qdrant]`` plus the state-DB path; M4 adds
``[reranker]``; later milestones add ``[structural]`` and ``[git]`` sections
to the same file. Everything has a working default so the service runs with
no config file at all.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_QUERY_URL = "http://127.0.0.1:6333"


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
class QdrantSettings:
    url: str = DEFAULT_QUERY_URL
    collection: str = "noesis_chunks"


@dataclass(frozen=True)
class Settings:
    db_path: Path = Path("data/noesis.sqlite")
    embedder: EmbedderSettings = field(default_factory=EmbedderSettings)
    reranker: RerankerSettings = field(default_factory=RerankerSettings)
    qdrant: QdrantSettings = field(default_factory=QdrantSettings)


def load_settings(config_path: str | Path = "config.toml") -> Settings:
    """Load settings from *config_path*, falling back to defaults per key."""
    path = Path(config_path)
    if not path.is_file():
        return Settings()
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    emb = raw.get("embedder", {})
    rrk = raw.get("reranker", {})
    qdr = raw.get("qdrant", {})
    return Settings(
        db_path=Path(raw.get("db_path", Settings.db_path)),
        embedder=EmbedderSettings(
            model=emb.get("model", EmbedderSettings.model),
            dim=int(emb.get("dim", EmbedderSettings.dim)),
            batch_size=int(emb.get("batch_size", EmbedderSettings.batch_size)),
            device=emb.get("device", EmbedderSettings.device),
        ),
        reranker=RerankerSettings(
            model=rrk.get("model", RerankerSettings.model),
            enabled=bool(rrk.get("enabled", RerankerSettings.enabled)),
            preload=bool(rrk.get("preload", RerankerSettings.preload)),
            candidates=int(rrk.get("candidates", RerankerSettings.candidates)),
            batch_size=int(rrk.get("batch_size", RerankerSettings.batch_size)),
            device=rrk.get("device", RerankerSettings.device),
        ),
        qdrant=QdrantSettings(
            url=qdr.get("url", QdrantSettings.url),
            collection=qdr.get("collection", QdrantSettings.collection),
        ),
    )
