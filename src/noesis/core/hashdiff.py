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


def partition(
    root: str | Path,
    discovered: Iterable[str],
    stored: Mapping[str, str],
    *,
    candidates: AbstractSet[str] | None = None,
) -> DiffResult:
    """Partition `discovered` (relative POSIX paths under `root`) against
    `stored` (path -> content_hash from the files table).

    Files that vanish between discovery and hashing are treated as deleted —
    the filesystem is ground truth at the moment it is read.

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
    seen: set[str] = set()

    for rel in discovered:
        prior_hash = stored.get(rel)
        if (
            candidates is not None
            and prior_hash is not None
            and rel not in candidates
        ):
            seen.add(rel)
            hashes[rel] = prior_hash
            unchanged.append(rel)
            continue
        try:
            current = hash_file(root / rel)
        except OSError:
            continue  # vanished mid-run; falls through to deleted if stored
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
    )
