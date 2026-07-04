"""SQLite state store — projects, per-file state, index runs.

WAL-mode single-file DB per the approved Overview §7 schema plus the
`last_indexed_commit` column from the expanded doc §3.7. All mutating
functions commit before returning; callers never manage transactions.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

_SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
  id                  TEXT PRIMARY KEY,
  root_path           TEXT NOT NULL UNIQUE,
  embedding_model     TEXT NOT NULL,
  created_at          TEXT NOT NULL,
  updated_at          TEXT NOT NULL,
  last_indexed_commit TEXT
);

CREATE TABLE IF NOT EXISTS files (
  id              TEXT PRIMARY KEY,
  project_id      TEXT NOT NULL REFERENCES projects(id),
  path            TEXT NOT NULL,
  language        TEXT,
  content_hash    TEXT NOT NULL,
  chunk_count     INTEGER NOT NULL DEFAULT 0,
  last_indexed_at TEXT,
  UNIQUE(project_id, path)
);

CREATE TABLE IF NOT EXISTS index_runs (
  id             TEXT PRIMARY KEY,
  project_id     TEXT NOT NULL REFERENCES projects(id),
  status         TEXT NOT NULL CHECK(status IN ('queued','running','done','failed')),
  files_total    INTEGER,
  files_changed  INTEGER,
  chunks_written INTEGER,
  started_at     TEXT,
  finished_at    TEXT,
  error          TEXT
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect(db_path: str | Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    conn.commit()


def register_project(
    conn: sqlite3.Connection, root_path: str | Path, embedding_model: str
) -> str:
    resolved = str(Path(root_path).resolve())
    row = conn.execute(
        "SELECT id, embedding_model FROM projects WHERE root_path = ?", (resolved,)
    ).fetchone()
    if row is not None:
        if row["embedding_model"] != embedding_model:
            raise ValueError(
                f"project at {resolved} is indexed with model "
                f"{row['embedding_model']!r}; refusing to serve mixed-model state. "
                f"Re-register requires a full re-index with {embedding_model!r}."
            )
        return row["id"]
    project_id = uuid.uuid4().hex
    now = _now()
    conn.execute(
        "INSERT INTO projects (id, root_path, embedding_model, created_at, updated_at)"
        " VALUES (?, ?, ?, ?, ?)",
        (project_id, resolved, embedding_model, now, now),
    )
    conn.commit()
    return project_id


def get_project(conn: sqlite3.Connection, project_id: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM projects WHERE id = ?", (project_id,)
    ).fetchone()


def list_projects(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM projects ORDER BY created_at").fetchall()


def get_file_states(conn: sqlite3.Connection, project_id: str) -> dict[str, str]:
    rows = conn.execute(
        "SELECT path, content_hash FROM files WHERE project_id = ?", (project_id,)
    ).fetchall()
    return {row["path"]: row["content_hash"] for row in rows}


def upsert_file(
    conn: sqlite3.Connection,
    project_id: str,
    path: str,
    content_hash: str,
    language: str | None = None,
    chunk_count: int = 0,
) -> None:
    conn.execute(
        """
        INSERT INTO files (id, project_id, path, language, content_hash,
                           chunk_count, last_indexed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(project_id, path) DO UPDATE SET
          language = excluded.language,
          content_hash = excluded.content_hash,
          chunk_count = excluded.chunk_count,
          last_indexed_at = excluded.last_indexed_at
        """,
        (uuid.uuid4().hex, project_id, path, language, content_hash, chunk_count, _now()),
    )
    conn.commit()


def delete_files(
    conn: sqlite3.Connection, project_id: str, paths: Iterable[str]
) -> None:
    conn.executemany(
        "DELETE FROM files WHERE project_id = ? AND path = ?",
        [(project_id, p) for p in paths],
    )
    conn.commit()


def start_run(conn: sqlite3.Connection, project_id: str) -> str:
    run_id = uuid.uuid4().hex
    conn.execute(
        "INSERT INTO index_runs (id, project_id, status, started_at) VALUES (?, ?, 'running', ?)",
        (run_id, project_id, _now()),
    )
    conn.commit()
    return run_id


def get_latest_run(
    conn: sqlite3.Connection, project_id: str
) -> sqlite3.Row | None:
    """Most recent index run for a project — the M6 ``get_index_status``
    surface. Ordered by started_at (ISO-8601 UTC, lexicographically
    sortable); ties broken by rowid (insertion order)."""
    return conn.execute(
        "SELECT * FROM index_runs WHERE project_id = ?"
        " ORDER BY started_at DESC, rowid DESC LIMIT 1",
        (project_id,),
    ).fetchone()


def finish_run(
    conn: sqlite3.Connection,
    run_id: str,
    status: str,
    *,
    files_total: int | None = None,
    files_changed: int | None = None,
    chunks_written: int | None = None,
    error: str | None = None,
) -> None:
    conn.execute(
        """
        UPDATE index_runs SET
          status = ?, files_total = ?, files_changed = ?, chunks_written = ?,
          finished_at = ?, error = ?
        WHERE id = ?
        """,
        (status, files_total, files_changed, chunks_written, _now(), error, run_id),
    )
    conn.commit()
