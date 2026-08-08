"""One-shot migration: positional golden labels -> content anchors (ADR-64).

Committed rather than run-and-discarded so the rewrite is reproducible and
reviewable: every anchor below was chosen by this script, not by hand, and
anyone can re-derive the same 43 anchors from the same two commits.

What it does. For each ``relevant`` item in ``golden.yaml``'s ``queries``
section it reads the labeled span **as it stood at the labeling commit**
(``1d2bf06``, the commit that introduced golden.yaml), then picks the first
line of that span which (a) still exists verbatim in today's file and (b)
occurs exactly once there as a substring — the same predicate
``harness.resolve_anchor`` applies at load time, so a chosen anchor is one
the loader is guaranteed to resolve.

Why the labeled span and not today's code: the anchor has to preserve what
the original labeler meant by that label. Re-deriving it from today's tree
would be re-labeling, which is a content judgement and not a migration.

Definition-shaped lines (``def``/``class``/decorator/assignment) are preferred
over docstring prose, because prose is reworded more often than signatures and
every reword costs someone a loud failure.

Three labels cannot be migrated mechanically and carry explicit overrides —
see OVERRIDES. The rewrite is line-based rather than a YAML round-trip so the
file's header and inline rationale comments survive untouched.

Usage:  python tests/eval/migrate_labels.py [--write]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parents[1]
GOLDEN_PATH = EVAL_DIR / "golden.yaml"

# The commit that added golden.yaml; the labels describe this tree.
LABEL_COMMIT = "1d2bf06"

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

_DEFINITION = re.compile(r"^\s*(async\s+def\s|def\s|class\s|@)")
_ASSIGNMENT = re.compile(r"^\s*[A-Za-z_][A-Za-z0-9_.\[\]\"']*\s*(:[^=]+)?=\s*\S")


def file_at(commit: str, rel_path: str) -> list[str] | None:
    proc = subprocess.run(
        ["git", "show", f"{commit}:{rel_path}"],
        cwd=REPO_ROOT,
        capture_output=True,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.decode("utf-8", "replace").splitlines()


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
    decorative: ``dedupe_by_path`` keeps only the best-ranked chunk per file,
    so a point label does not ask "was the relevant region retrieved" — it
    asks "did the chunk containing this one line out-rank every other chunk of
    the same file". A file whose right region ranked 4th behind another of its
    own chunks would score 0 on a point label while the region was in fact
    retrieved. The original labeling rule ("`lines` is optional and generous
    (whole function/class)") existed for this reason; content-addressing the
    two ends keeps it while making both ends rot-proof.
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


def compute_anchors() -> tuple[list[dict], list[dict]]:
    """Return (resolved, unresolved) label records in document order."""
    raw = yaml.safe_load(GOLDEN_PATH.read_bytes())
    resolved: list[dict] = []
    unresolved: list[dict] = []
    for query in raw["queries"]:
        for item in query["relevant"]:
            rel_path = item["path"]
            record = {"id": query["id"], "path": rel_path, "new_path": rel_path}
            override = OVERRIDES.get((query["id"], rel_path))
            if override is not None:
                new_path, anchor, anchor_end = override
                body = (REPO_ROOT / new_path).read_text(errors="replace").splitlines()
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
            target = REPO_ROOT / rel_path
            if not target.is_file():
                record["reason"] = "labeled file no longer exists"
                unresolved.append(record)
                continue
            body = target.read_text(errors="replace").splitlines()
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
            # `nl` and `structural` queries ask about a region, and there the
            # width is load-bearing: `dedupe_by_path` keeps only the
            # best-ranked chunk per file, so a point label would score 0
            # whenever the right region was retrieved but ranked behind
            # another chunk of the same file.
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


def rewrite(records: list[dict]) -> str:
    """Replace each `lines:` line with an `anchor:` line, in document order.

    Only the ``queries`` section carries ``lines:``; ``structural_patterns``
    pins match counts and is left alone.
    """
    by_order = list(records)
    out: list[str] = []
    index = 0
    for line in GOLDEN_PATH.read_text().splitlines():
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
    parser.add_argument("--write", action="store_true", help="rewrite golden.yaml")
    args = parser.parse_args()

    resolved, unresolved = compute_anchors()
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
    if args.write:
        GOLDEN_PATH.write_text(rewrite(resolved))
        print(f"\nwrote {GOLDEN_PATH}")
    else:
        print("\ndry run — pass --write to apply")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
