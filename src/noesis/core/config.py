"""Runtime settings, read once at startup from ``config.toml`` (§3.7).

M2 needs only the ``[embedder]`` and ``[qdrant]`` sections plus the state-DB
path; later milestones add ``[reranker]``, ``[structural]`` and ``[git]``
sections to the same file. Everything has a working default so the service
runs with no config file at all.
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


@dataclass(frozen=True)
class QdrantSettings:
    url: str = DEFAULT_QUERY_URL
    collection: str = "noesis_chunks"


@dataclass(frozen=True)
class Settings:
    db_path: Path = Path("data/noesis.sqlite")
    embedder: EmbedderSettings = field(default_factory=EmbedderSettings)
    qdrant: QdrantSettings = field(default_factory=QdrantSettings)


def load_settings(config_path: str | Path = "config.toml") -> Settings:
    """Load settings from *config_path*, falling back to defaults per key."""
    path = Path(config_path)
    if not path.is_file():
        return Settings()
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    emb = raw.get("embedder", {})
    qdr = raw.get("qdrant", {})
    return Settings(
        db_path=Path(raw.get("db_path", Settings.db_path)),
        embedder=EmbedderSettings(
            model=emb.get("model", EmbedderSettings.model),
            dim=int(emb.get("dim", EmbedderSettings.dim)),
            batch_size=int(emb.get("batch_size", EmbedderSettings.batch_size)),
        ),
        qdrant=QdrantSettings(
            url=qdr.get("url", QdrantSettings.url),
            collection=qdr.get("collection", QdrantSettings.collection),
        ),
    )
