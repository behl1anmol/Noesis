"""Indexing pipeline: discover → hash-diff → chunk → embed → upsert (§3.2).

Orchestrates the M1 spine (discovery, hashdiff, state) with the M2 pieces
(chunker, Embedder, VectorStore). Dense channel only in M2 — the sparse/BM25
channel and RRF fusion land in M3; the git fast-path narrows the candidate
set in M7. Embedding batches go through the Embedder Protocol at LOW
priority so live queries preempt indexing (§3.8).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from . import hashdiff, state
from .chunker import chunk_file
from .discovery import DiscoveryConfig, discover_files
from .embedder import Embedder
from .languages import detect_language
from .vectorstore import VectorStore


@dataclass(frozen=True)
class IndexResult:
    project_id: str
    run_id: str
    files_total: int
    files_indexed: int
    files_deleted: int
    chunks_written: int


def prepare_run(
    conn: sqlite3.Connection, embedder: Embedder, root_path: str
) -> tuple[str, str]:
    """Register (or re-open) the project and open a run row.

    Split from :func:`execute_run` so the API can hand back
    ``202 Accepted + run_id`` before indexing starts (§3.2).
    """
    project_id = state.register_project(conn, root_path, embedder.model_id)
    run_id = state.start_run(conn, project_id)
    return project_id, run_id


async def index_project(
    conn: sqlite3.Connection,
    store: VectorStore,
    embedder: Embedder,
    root_path: str,
    *,
    batch_size: int = 32,
    discovery_config: DiscoveryConfig | None = None,
) -> IndexResult:
    """Register, open a run, and index in one call (tests / CLI use)."""
    project_id, run_id = prepare_run(conn, embedder, root_path)
    return await execute_run(
        conn,
        store,
        embedder,
        root_path,
        project_id,
        run_id,
        batch_size=batch_size,
        discovery_config=discovery_config,
    )


async def execute_run(
    conn: sqlite3.Connection,
    store: VectorStore,
    embedder: Embedder,
    root_path: str,
    project_id: str,
    run_id: str,
    *,
    batch_size: int = 32,
    discovery_config: DiscoveryConfig | None = None,
) -> IndexResult:
    """Index changes for an already-registered project under an open run.

    Idempotent: chunk ids are content-derived and file state is only written
    after that file's chunks are safely in Qdrant, so an interrupted run
    re-processes only what is still out of date (Overview §5).
    """
    try:
        discovered = discover_files(root_path, discovery_config)
        stored = state.get_file_states(conn, project_id)
        diff = hashdiff.partition(root_path, discovered, stored)

        chunks_written = 0
        to_index = [*diff.new, *diff.changed]
        for rel in to_index:
            text = _read_text(root_path, rel)
            language = detect_language(rel)
            file_hash = diff.hashes[rel]
            chunks = chunk_file(
                text, language=language, file_path=rel, file_hash=file_hash
            )
            if chunks:
                vectors: list[list[float]] = []
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i : i + batch_size]
                    vectors.extend(
                        await embedder.embed_documents([c.text for c in batch])
                    )
                store.upsert_chunks(
                    project_id, chunks, vectors, embedding_model=embedder.model_id
                )
            # New points first, stale points after: chunk ids embed the file
            # hash, so old content lives at different ids and must be pruned —
            # but only once the replacement is searchable. A failure above
            # leaves the old chunks serving; a failure below leaves brief
            # duplicates that the next (self-healing) run prunes.
            store.delete_file_chunks(project_id, [rel], exclude_file_hash=file_hash)
            state.upsert_file(
                conn,
                project_id,
                rel,
                file_hash,
                language=language,
                chunk_count=len(chunks),
            )
            chunks_written += len(chunks)

        if diff.deleted:
            store.delete_file_chunks(project_id, diff.deleted)
            state.delete_files(conn, project_id, diff.deleted)

        state.finish_run(
            conn,
            run_id,
            "done",
            files_total=len(discovered),
            files_changed=len(to_index),
            chunks_written=chunks_written,
        )
        return IndexResult(
            project_id=project_id,
            run_id=run_id,
            files_total=len(discovered),
            files_indexed=len(to_index),
            files_deleted=len(diff.deleted),
            chunks_written=chunks_written,
        )
    except BaseException as exc:
        # BaseException: CancelledError (e.g. server shutdown) must also mark
        # the run failed, or it would sit "running" forever.
        state.finish_run(conn, run_id, "failed", error=str(exc) or type(exc).__name__)
        raise


def _read_text(root_path: str, rel: str) -> str:
    from pathlib import Path

    return (Path(root_path) / rel).read_text(encoding="utf-8", errors="replace")
