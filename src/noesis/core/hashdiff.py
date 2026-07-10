"""SHA-256 change detection: hash files, partition against stored state.

Hash is the source of truth for change detection (Overview §4.9). The M7 git
fast-path may later shrink the candidate set fed to `partition`, but never
replaces the hash comparison itself (expanded doc §3.2 rule 1).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import AbstractSet, Iterable, Mapping

_READ_CHUNK = 1 << 20  # 1 MiB


def hash_file(path: str | Path) -> str:
    """SHA-256 hex digest of a file's content, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_READ_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class DiffResult:
    """Partition of discovered files vs stored state (relative POSIX paths)."""

    new: tuple[str, ...] = field(default=())
    changed: tuple[str, ...] = field(default=())
    unchanged: tuple[str, ...] = field(default=())
    deleted: tuple[str, ...] = field(default=())
    hashes: Mapping[str, str] = field(default_factory=dict)
    """Current content hash for every discovered file (new/changed/unchanged)."""
    errored: tuple[tuple[str, str], ...] = field(default=())
    """(path, error) for files whose hash attempt failed non-fatally (the H7
    carry-forward). The file's true state is UNKNOWN this run — callers must
    treat these like per-file failures: never advance the git anchor past
    them and re-queue them for retry, or a committed-then-errored file is
    carried forward as unchanged on every future fast-path run (stale
    forever)."""


def partition(
    root: str | Path,
    discovered: Iterable[str],
    stored: Mapping[str, str],
    *,
    candidates: AbstractSet[str] | None = None,
) -> DiffResult:
    """Partition `discovered` (relative POSIX paths under `root`) against
    `stored` (path -> content_hash from the files table).

    Files that vanish between discovery and hashing (FileNotFoundError /
    NotADirectoryError) are treated as deleted — the filesystem is ground
    truth at the moment it is read. A file that still exists but fails to
    open for another reason (permissions, transient network-fs error) is
    NOT treated as deleted: its stored hash is carried forward so its chunks
    are never purged (H7).

    When `candidates` is given (git fast-path, §3.2), only discovered files
    that are candidates — or unknown to `stored` — get hashed; the rest
    carry their stored hash forward as unchanged. Candidacy only shrinks
    the hashing work, never decides changed/unchanged by itself (rule 1):
    every hashed file is still compared against `stored`, and deletions
    fall out of discovery exactly as in the full walk.
    """
    root = Path(root)
    new: list[str] = []
    changed: list[str] = []
    unchanged: list[str] = []
    hashes: dict[str, str] = {}
    errored: list[tuple[str, str]] = []
    seen: set[str] = set()

    for rel in discovered:
        prior_hash = stored.get(rel)
        if candidates is not None and prior_hash is not None and rel not in candidates:
            seen.add(rel)
            hashes[rel] = prior_hash
            unchanged.append(rel)
            continue
        try:
            current = hash_file(root / rel)
        except (FileNotFoundError, NotADirectoryError):
            continue  # genuinely vanished mid-run; falls through to deleted
        except OSError as exc:
            # Transient/permission failure (EACCES after a chmod, EIO/ESTALE
            # on a network fs) on a file that still exists — NOT a deletion
            # (H7). Conflating the two would purge a live file's chunks. Carry
            # the stored hash forward as unchanged so its chunks keep serving;
            # a file unknown to `stored` we simply skip this run (nothing to
            # preserve, and it is not "deleted"). Either way the path is
            # surfaced in `errored`: "the next run retries" holds only if the
            # caller keeps the git anchor back / re-queues it — under the fast
            # path a committed file that errored here would otherwise never be
            # a candidate again and its stale hash would be carried forward on
            # every future run.
            errored.append((rel, str(exc) or type(exc).__name__))
            if prior_hash is not None:
                seen.add(rel)
                hashes[rel] = prior_hash
                unchanged.append(rel)
            continue
        seen.add(rel)
        hashes[rel] = current
        prior = stored.get(rel)
        if prior is None:
            new.append(rel)
        elif prior != current:
            changed.append(rel)
        else:
            unchanged.append(rel)

    deleted = sorted(p for p in stored if p not in seen)
    return DiffResult(
        new=tuple(sorted(new)),
        changed=tuple(sorted(changed)),
        unchanged=tuple(sorted(unchanged)),
        deleted=tuple(deleted),
        hashes=hashes,
        errored=tuple(errored),
    )
