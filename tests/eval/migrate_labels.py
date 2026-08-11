"""One-shot migration: positional golden labels -> content anchors (ADR-64).

Committed rather than run-and-discarded so the rewrite is reproducible and
reviewable: every anchor was chosen by this script, not by hand, and anyone
can re-derive the same 46 labels — 43 mechanically, 3 from OVERRIDES.

**Both ends are pinned to a commit, so the reproduction is deterministic
forever rather than only on the day it ran.**

- ``LABEL_COMMIT`` (``1d2bf06``) is where the labeled *spans* come from: the
  commit that introduced golden.yaml, whose line numbers describe that tree.
- ``MIGRATION_BASE`` (``216450e``) is the tree the anchors were resolved
  *against*, and the source of the pre-migration ``golden.yaml`` — the last
  version that still carried ``lines:``. Reading the live tree instead would
  make the output drift with every commit, so "anyone can re-derive the same
  anchors" would be true only on the day of the migration. It is also why an
  earlier version of this script could not run at all: it re-parsed the
  ``golden.yaml`` it had itself rewritten, and died on ``item["lines"]``,
  a key that no longer exists (PR #42 review).

What it does. For each ``relevant`` item in the pre-migration ``golden.yaml``
it reads the labeled span as it stood at ``LABEL_COMMIT``, then picks the
first line of that span which (a) still exists verbatim in the file at
``MIGRATION_BASE`` and (b) occurs exactly once there as a substring — the same
predicate ``harness.resolve_anchor`` applies at load time, so a chosen anchor
is one the loader is guaranteed to resolve.

Why the labeled span and not the later code: the anchor has to preserve what
the original labeler meant by that label. Re-deriving it from a newer tree
would be re-labeling, which is a content judgement and not a migration.

Anchors are chosen in **span order**, not by preferring definition-shaped
lines — see ``pick_anchor``, which records why definition-preference was tried
and rejected. Three labels cannot be migrated mechanically and carry explicit
overrides — see OVERRIDES. The rewrite is line-based rather than a YAML
round-trip so the file's header and inline rationale comments survive
untouched.

This reproduces commit ``3307997``'s golden.yaml (46 labels). It is NOT
today's golden.yaml: ``structural-08`` gained a second endpoint label in
``13fcd9d`` (47 labels), and four anchors were hardened in the PR #42 review —
``nl-07`` and ``nl-13``'s ``anchor`` and ``structural-04``'s ``anchor_end``
moved off docstring prose onto signature and field lines (round 1), and
``nl-09``'s ``anchor_end`` moved off ``WHERE id = ?``, which was unique only by
its indentation (round 2). Those are signed-off edits made after the migration,
not part of it — which is why this script can no longer write ``golden.yaml``
and refuses an ``--out`` that resolves to it.

Usage:  python tests/eval/migrate_labels.py [--out PATH]

Reproduce and check::

    python tests/eval/migrate_labels.py --out /tmp/golden.migrated.yaml
    git show 3307997:tests/eval/golden.yaml | diff - /tmp/golden.migrated.yaml

``tests/eval/test_migrate_labels.py`` runs exactly that comparison.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parents[1]
GOLDEN_REL = "tests/eval/golden.yaml"

# The commit that added golden.yaml; the labeled `lines` describe this tree.
LABEL_COMMIT = "1d2bf06"

# The tree the anchors were resolved against: the parent of the migration
# commit 3307997, i.e. the last revision whose golden.yaml still carried
# `lines:`. Pinned rather than "the working tree" so the output is a fixed
# artifact — see the module docstring.
MIGRATION_BASE = "216450e"

# What this script reproduces. Its golden.yaml is the migration's own output.
MIGRATION_COMMIT = "3307997"

# Labels whose target cannot be recovered from the labeled span. Each was
# verified individually and is called out in the PR body for sign-off, per
# golden.yaml's own rule that the stakeholder's labels are authoritative.
#
#   nl-10, symbol-12: config.py was 58 lines at LABEL_COMMIT and load_settings
#     sat at line 38, but the labels claim [63,107] and [72,107]. Those ranges
#     never matched anything — the labels were invalid the day they were
#     written, so both queries have scored 0 since M2, inside the stored M2
#     baseline itself. Re-pointed at load_settings, which is what both the
#     query text ("where are runtime settings loaded from the toml config
#     file") and the query id ("load_settings") ask for.
#   symbol-13: `class AppContext` moved from app.py to runtime.py in 34f2408
#     (2026-07-06), so this label's PATH has been wrong for a month, not just
#     its lines. Re-pointed at the definition's new home.
OVERRIDES: dict[tuple[str, str], tuple[str, str, str | None]] = {
    # (query id, labeled path) -> (new path, anchor, anchor_end)
    ("nl-10", "src/noesis/core/config.py"): (
        "src/noesis/core/config.py",
        "def load_settings(",
        'collection=qdr.get("collection", QdrantSettings.collection),',
    ),
    ("symbol-12", "src/noesis/core/config.py"): (
        "src/noesis/core/config.py",
        "def load_settings(",
        'collection=qdr.get("collection", QdrantSettings.collection),',
    ),
    ("symbol-13", "src/noesis/app.py"): (
        "src/noesis/runtime.py",
        "class AppContext:",
        "config_reranker_device_pin: str | None = None",
    ),
}

def blob_at(commit: str, rel_path: str) -> str | None:
    """File contents at *commit*, or None when the object is not reachable.

    None covers both "the path did not exist there" and "this is a shallow
    clone" — callers turn the second into a loud, named failure.
    """
    proc = subprocess.run(
        ["git", "show", f"{commit}:{rel_path}"],
        cwd=REPO_ROOT,
        capture_output=True,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.decode("utf-8", "replace")


def file_at(commit: str, rel_path: str) -> list[str] | None:
    blob = blob_at(commit, rel_path)
    return None if blob is None else blob.splitlines()


def unique_substring(candidate: str, body: list[str]) -> bool:
    """True iff *candidate* occurs on exactly one line — resolve_anchor's rule."""
    return sum(1 for line in body if candidate in line) == 1


def pick_anchor(span: list[str], current: list[str]) -> str | None:
    """First surviving, uniquely-resolving line of *span*, in span order.

    Span order, deliberately, and NOT "prefer a def/class line". Preferring
    definition-shaped lines was tried and reordered the search, which made it
    reach past the labeled code into a structurally similar neighbour: nl-11
    ("how do I check the status of a running index job") landed on
    ``@router.post("/search")`` — the adjacent endpoint — and nl-12 ("where
    are secret files kept out of the index") on a ``@dataclass`` decorator
    instead of ``SECRET_SKIP_PATTERNS``. The top of a labeled span is what the
    labeler pointed at, so walking it in order is what preserves their intent.
    Body lines are likelier to be edited than signatures; that only costs a
    loud failure, which is the whole design.
    """

    def usable(raw: str) -> str | None:
        stripped = raw.strip()
        # Short lines ("else:", ")") repeat everywhere; comments get deleted.
        if len(stripped) < 12 or stripped.startswith("#"):
            return None
        for candidate in (raw.rstrip(), stripped):
            if unique_substring(candidate, current):
                return candidate
        return None

    for raw in span:
        if (found := usable(raw)) is not None:
            return found
    return None


def pick_symbol_anchor(
    span: list[str], current: list[str], identifier: str
) -> str | None:
    """Anchor for a `symbol` query: a line that actually contains the identifier.

    Span order alone is wrong here and the label test proved it — it chose
    ``cfg = config or DiscoveryConfig()`` for the query "discover_files" and
    ``".js": "javascript",`` for "EXT_TO_LANGUAGE", both merely the first
    surviving line of the old span. For an identifier lookup the answer is the
    definition site, so the anchor must contain the identifier; definition
    forms win, and the search widens to the whole file because the definition
    may have moved out of the old span entirely (which is the rot itself).
    """
    definition_first = (
        f"def {identifier}(",
        f"async def {identifier}(",
        f"class {identifier}",
        f"{identifier} =",
        f"{identifier}:",
        f"{identifier} ",
        identifier,
    )
    for haystack in (span, current):
        for form in definition_first:
            hits = [line for line in haystack if form in line]
            for raw in hits:
                for candidate in (raw.rstrip(), raw.strip()):
                    if identifier in candidate and unique_substring(candidate, current):
                        return candidate
    return None


def pick_anchor_end(span: list[str], current: list[str], start: str) -> str | None:
    """Last surviving, uniquely-resolving line of *span* — the span's far end.

    Anchoring BOTH ends reproduces the original label's extent instead of
    collapsing it to a point, and that width is load-bearing rather than
    decorative.

    The rationale AS IT STOOD AT MIGRATION TIME, stated in the past tense
    because the scorer has since changed underneath it (PR #42 review round 4):
    ``dedupe_by_path`` then kept only the best-ranked chunk per file, so a
    point label did not ask "was the relevant region retrieved" — it asked
    "did the chunk containing this one line out-rank every other chunk of the
    same file", and a file whose right region ranked 4th behind another of its
    own chunks scored 0 while the region had in fact been retrieved. ADR-67
    removed that particular trap by grouping a file's chunks instead of
    collapsing them.

    The width rule survives it on narrower grounds, which is why nothing here
    changes: matching is still a span-overlap test, so a one-line label counts
    only when the chunk holding that exact line comes back, while a
    region-wide label counts whichever chunk of the region does. The original
    labeling rule ("`lines` is optional and generous (whole function/class)")
    existed for this; content-addressing the two ends keeps it while making
    both ends rot-proof.
    """

    def usable(raw: str) -> str | None:
        stripped = raw.strip()
        if len(stripped) < 12 or stripped.startswith("#"):
            return None
        for candidate in (raw.rstrip(), stripped):
            if candidate != start and unique_substring(candidate, current):
                return candidate
        return None

    for raw in reversed(span):
        if (found := usable(raw)) is not None:
            return found
    return None


def pre_migration_golden() -> str:
    """The last golden.yaml that still carried ``lines:``, verbatim.

    Read from git rather than from disk: this script rewrote the file on disk,
    so the on-disk copy is this function's *output*, not its input.
    """
    blob = blob_at(MIGRATION_BASE, GOLDEN_REL)
    if blob is None:
        raise SystemExit(
            f"cannot read {GOLDEN_REL} at {MIGRATION_BASE} — the object is not "
            f"reachable. Either this is a shallow clone (CI uses fetch-depth: 0), "
            f"or {MIGRATION_BASE} no longer names a commit in this history."
        )
    return blob


def compute_anchors(source: str | None = None) -> tuple[list[dict], list[dict]]:
    """Return (resolved, unresolved) label records in document order.

    *source* is the pre-migration golden.yaml text; it defaults to the copy at
    MIGRATION_BASE. Every file consulted is read at MIGRATION_BASE too, so the
    result is a function of two commits and nothing else.
    """
    raw = yaml.safe_load(source if source is not None else pre_migration_golden())
    resolved: list[dict] = []
    unresolved: list[dict] = []
    for query in raw["queries"]:
        for item in query["relevant"]:
            rel_path = item["path"]
            record = {"id": query["id"], "path": rel_path, "new_path": rel_path}
            override = OVERRIDES.get((query["id"], rel_path))
            if override is not None:
                new_path, anchor, anchor_end = override
                body = file_at(MIGRATION_BASE, new_path)
                if body is None:
                    record["reason"] = f"override target {new_path} absent at {MIGRATION_BASE}"
                    unresolved.append(record)
                    continue
                bad = [
                    text
                    for text in (anchor, anchor_end)
                    if text is not None and not unique_substring(text, body)
                ]
                if bad:
                    record["reason"] = f"override anchors not unique in {new_path}: {bad}"
                    unresolved.append(record)
                    continue
                record.update(
                    new_path=new_path,
                    anchor=anchor,
                    anchor_end=anchor_end,
                    source="override",
                )
                resolved.append(record)
                continue

            old = file_at(LABEL_COMMIT, rel_path)
            start, end = item["lines"]
            if old is None or start > len(old):
                record["reason"] = (
                    f"labeled range [{start},{end}] is past EOF at {LABEL_COMMIT} "
                    f"(file had {0 if old is None else len(old)} lines)"
                )
                unresolved.append(record)
                continue
            body = file_at(MIGRATION_BASE, rel_path)
            if body is None:
                record["reason"] = f"labeled file absent at {MIGRATION_BASE}"
                unresolved.append(record)
                continue
            span = old[start - 1 : end]
            if query["category"] == "symbol":
                anchor = pick_symbol_anchor(span, body, query["query"].strip())
            else:
                anchor = pick_anchor(span, body)
            if anchor is None:
                record["reason"] = "no line of the labeled span survives uniquely"
                unresolved.append(record)
                continue
            # Width is decided by what the query asks for.
            #
            # `symbol` queries are bare identifiers, so the answer IS the
            # definition site: a point anchor on the signature is exactly
            # right, and the chunker forces a hard chunk boundary at a
            # definition's start, so that signature always heads a chunk. It
            # also avoids absurdity — symbol-05's old 82-line span maps onto
            # today's `execute_run`, which is 1017 lines, and a span that wide
            # would let almost any chunk of indexer.py score as correct.
            #
            # `nl` and `structural` queries ask about a region, and there
            # the width is load-bearing: matching is a span-overlap test, so a
            # point label counts only when the chunk holding that exact line is
            # retrieved, while a region-wide label counts whichever chunk of
            # the region comes back. (At migration time the sharper form of
            # this was `dedupe_by_path` keeping only each file's best-ranked
            # chunk; ADR-67 removed that. See `pick_anchor_end`.)
            anchor_end = None
            if query["category"] != "symbol":
                anchor_end = pick_anchor_end(span, body, anchor)
            if anchor_end is not None:
                first = next(i for i, l in enumerate(body) if anchor in l)
                last = next(i for i, l in enumerate(body) if anchor_end in l)
                # The span's two ends may have been reordered by refactoring;
                # a label whose end precedes its start is not a narrower label,
                # it is a wrong one. Collapse to the start rather than guess.
                if last < first:
                    anchor_end = None
                else:
                    record["span_lines"] = last - first + 1
            record.update(anchor=anchor, anchor_end=anchor_end, source="span")
            resolved.append(record)
    return resolved, unresolved


def rewrite(records: list[dict], source: str | None = None) -> str:
    """Replace each `lines:` line with an `anchor:` line, in document order.

    Only the ``queries`` section carries ``lines:``; ``structural_patterns``
    pins match counts and is left alone. *source* is the pre-migration text —
    the on-disk golden.yaml has no ``lines:`` left to replace.
    """
    by_order = list(records)
    out: list[str] = []
    index = 0
    for line in (source if source is not None else pre_migration_golden()).splitlines():
        stripped = line.strip()
        if stripped.startswith("- path:") and index < len(by_order):
            record = by_order[index]
            indent = line[: len(line) - len(line.lstrip())]
            out.append(f"{indent}- path: {record['new_path']}")
            continue
        if stripped.startswith("lines:"):
            record = by_order[index]
            indent = line[: len(line) - len(line.lstrip())]
            out.append(f"{indent}anchor: {json.dumps(record['anchor'])}")
            if record.get("anchor_end"):
                out.append(f"{indent}anchor_end: {json.dumps(record['anchor_end'])}")
            index += 1
            continue
        out.append(line)
    if index != len(by_order):
        raise SystemExit(
            f"rewrote {index} labels but computed {len(by_order)} — aborting "
            f"rather than writing a partially migrated file"
        )
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # Deliberately NOT `--write`: golden.yaml has carried signed-off edits since
    # the migration (structural-08's second endpoint, three hardened anchors),
    # and a script whose only remaining job is reproduction must not be able to
    # revert them. It writes where you point it, and nowhere else.
    parser.add_argument(
        "--out",
        metavar="PATH",
        help=f"write the reproduced golden.yaml here (compare with {MIGRATION_COMMIT})",
    )
    args = parser.parse_args()

    # Validate the destination BEFORE the ~12 s of git reads below. Three
    # defects, all from the PR #42 round-2 review:
    #   * `--out ""` did the whole run and then silently degraded to a dry run,
    #     because the old test was `if args.out:` and "" is falsy;
    #   * a missing parent directory surfaced as a bare FileNotFoundError after
    #     the work, rather than as an argument error before it;
    #   * nothing refused `--out tests/eval/golden.yaml`, so the reason given
    #     for dropping `--write` — "a reproduction tool able to revert
    #     signed-off edits is a footgun" — was only half delivered.
    out_path: Path | None = None
    if args.out is not None:
        if not args.out.strip():
            parser.error("--out needs a path; omit the flag entirely for a dry run")
        out_path = Path(args.out).expanduser()
        if out_path.resolve() == (REPO_ROOT / GOLDEN_REL).resolve():
            parser.error(
                f"--out must not be {GOLDEN_REL}. This script reproduces the "
                f"MIGRATION, and golden.yaml has carried signed-off edits since "
                f"(structural-08's second endpoint, four hardened anchors) — "
                f"writing there would revert them. Write elsewhere and diff."
            )

    source = pre_migration_golden()
    resolved, unresolved = compute_anchors(source)
    print(f"resolved {len(resolved)} labels, unresolved {len(unresolved)}")
    for record in resolved:
        moved = "" if record["new_path"] == record["path"] else f"  (path -> {record['new_path']})"
        print(f"  {record['id']:<14} {record['path']:<32} {record['source']:<9} {record['anchor']!r}{moved}")
        if record.get("anchor_end"):
            print(f"  {'':<14} {'':<32} {'..end':<9} {record['anchor_end']!r}")
    for record in unresolved:
        print(f"  UNRESOLVED {record['id']:<14} {record['path']}: {record['reason']}")
    if unresolved:
        print("\nrefusing to write: every label must resolve", file=sys.stderr)
        return 1
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rewrite(resolved, source), encoding="utf-8")
        print(f"\nwrote {out_path}")
        print(f"compare: git show {MIGRATION_COMMIT}:{GOLDEN_REL} | diff - {out_path}")
    else:
        print("\ndry run — pass --out PATH to write the reproduced file")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
