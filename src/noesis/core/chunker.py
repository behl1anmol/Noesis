"""cAST split-then-merge chunking: AST nodes -> size-bounded chunks (Overview §4.4).

tree-sitter parses each file; adjacent sibling nodes are greedily merged into
chunks within a token budget, any node over the max budget is split by
recursing into its children, and over-budget leaves are split by lines.
Missing language or grammar degrades to a line-based fallback — never an
error. Canonical language names (core/languages.py) are passed straight to
tree-sitter-language-pack; they resolve 1:1 today, and M5's LANGUAGE_MAP is
the seam if that ever changes.

Hard invariant: concatenating ``chunk.text`` over a file's chunks, in order,
reproduces the file text exactly — byte-for-byte, no overlap. Whitespace and
comments between siblings attach to the following chunk, and every chunk
boundary is snapped to a line start, so ``start_line``/``end_line`` slice the
file's lines exactly.

Never split a function below its signature: a definition either fits whole in
a chunk, or — only when it alone exceeds the max budget — is split via its
own children behind a hard chunk boundary forced at the definition's start.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

from tree_sitter_language_pack import get_parser

MIN_CHUNK_TOKENS = 300
"""Soft floor: greedy merging aims to fill chunks at least this far."""

MAX_CHUNK_TOKENS = 800
"""Hard ceiling: only an indivisible single line may estimate above this."""

_WS_BYTES = frozenset(b" \t\n\r\x0b\x0c")


@dataclass(frozen=True)
class Chunk:
    """One retrieval unit of a file, with 1-based inclusive line bounds."""

    text: str
    file_path: str
    start_line: int
    end_line: int
    language: str | None
    node_type: str
    symbol_name: str | None
    file_hash: str


@dataclass
class _Piece:
    """Working span (byte offsets) carrying AST identity through the pipeline."""

    start: int
    end: int
    kind: str
    symbol: str | None
    named: bool
    hard_before: bool = False
    hard_after: bool = False


def chunk_file(
    text: str, *, language: str | None, file_path: str, file_hash: str
) -> list[Chunk]:
    """Chunk one file's content; empty text yields an empty list."""
    if not text:
        return []
    data = text.encode("utf-8")
    prefix = _nws_prefix(data)
    pieces = _ast_pieces(text, data, prefix, language)
    if pieces is None:  # no language / grammar / parse — degrade, never fail
        pieces = _line_pieces(data, 0, len(data), "text", named=True)
    pieces = _snap_to_lines(pieces, data, prefix)
    chunks: list[Chunk] = []
    for window in _merge(pieces, prefix):
        start, end = window[0].start, window[-1].end
        source = next(
            (p for p in window if p.symbol is not None),
            next((p for p in window if p.named and p.kind != "comment"), window[0]),
        )
        chunks.append(
            Chunk(
                text=data[start:end].decode("utf-8"),
                file_path=file_path,
                start_line=data.count(b"\n", 0, start) + 1,
                end_line=data.count(b"\n", 0, end - 1) + 1,
                language=language,
                node_type=source.kind,
                symbol_name=source.symbol,
                file_hash=file_hash,
            )
        )
    return chunks


def _token_estimate(text: str) -> int:
    """Decided estimator: ceil(non-whitespace chars / 4). Do not change."""
    nws = sum(1 for ch in text if not ch.isspace())
    return math.ceil(nws / 4)


def _nws_prefix(data: bytes) -> list[int]:
    """prefix[i] = count of non-whitespace characters in data[:i].

    Counted per character (matching _token_estimate's char-based budget) but
    indexed by byte offset, since piece/line bounds are byte offsets. Every
    boundary queried lands on a character start; byte positions interior to a
    multi-byte character are filled monotonically but never read."""
    prefix = [0] * (len(data) + 1)
    total = 0
    pos = 0
    for ch in data.decode("utf-8"):
        nxt = pos + len(ch.encode("utf-8"))
        if not ch.isspace():
            total += 1
        for j in range(pos + 1, nxt + 1):
            prefix[j] = total
        pos = nxt
    return prefix


def _span_tokens(prefix: list[int], start: int, end: int) -> int:
    return math.ceil((prefix[end] - prefix[start]) / 4)


_local = threading.local()


def _cached_parser(language: str | None):
    """Per-thread parser cache: the pack's Parser is pyo3-unsendable, so a
    parser created on one thread must never be reused from another."""
    if language is None:
        return None
    cache: dict[str, object | None] | None = getattr(_local, "parsers", None)
    if cache is None:
        cache = _local.parsers = {}
    if language not in cache:
        try:
            cache[language] = get_parser(language)
        except Exception:  # grammar unavailable — remember and degrade
            cache[language] = None
    return cache[language]


def _ast_pieces(
    text: str, data: bytes, prefix: list[int], language: str | None
) -> list[_Piece] | None:
    """Split phase: document-ordered pieces, or None if parsing is unavailable."""
    parser = _cached_parser(language)
    if parser is None:
        return None
    try:
        tree = parser.parse(text)
    except Exception:
        return None
    if tree is None:
        return None
    pieces: list[_Piece] = []
    _emit(tree.root_node(), data, prefix, pieces)
    return pieces or None


def _emit(node, data: bytes, prefix: list[int], out: list[_Piece]) -> None:
    """Emit pieces for *node*: whole if within budget, else via its children."""
    start, end = node.start_byte(), node.end_byte()
    if _span_tokens(prefix, start, end) <= MAX_CHUNK_TOKENS:
        out.append(
            _Piece(start, end, node.kind(), _symbol_name(node, data), node.is_named())
        )
        return
    first = len(out)
    count = node.child_count()
    if count:
        for i in range(count):
            child = node.child(i)
            if child is not None:
                _emit(child, data, prefix, out)
    else:  # over-budget leaf (giant string/comment): split by lines
        out.extend(_line_pieces(data, start, end, node.kind(), node.is_named()))
    if len(out) == first:  # defensive: no emittable children
        out.append(_Piece(start, end, node.kind(), None, node.is_named()))
    kind = node.kind()
    if _is_definition(kind):
        # The definition's head piece keeps its identity, and hard boundaries
        # stop its atoms from half-merging with preceding/following siblings —
        # a signature is never orphaned from its own body.
        head = out[first]
        head.kind = kind
        head.symbol = _symbol_name(node, data)
        head.named = True
        head.hard_before = True
        out[-1].hard_after = True


def _is_definition(kind: str) -> bool:
    return kind == "decorated_definition" or kind.endswith(
        ("_definition", "_declaration")
    )


def _symbol_name(node, data: bytes) -> str | None:
    """Name identifier for definition-like nodes, else None."""
    if not _is_definition(node.kind()):
        return None
    target = node
    if node.kind() == "decorated_definition":
        target = node.child_by_field_name("definition") or node
    name = target.child_by_field_name("name")
    if name is None:
        return None
    return data[name.start_byte() : name.end_byte()].decode("utf-8")


def _line_pieces(
    data: bytes, start: int, end: int, kind: str, named: bool
) -> list[_Piece]:
    """One piece per line of data[start:end]; merging regroups them later."""
    out: list[_Piece] = []
    pos = start
    while pos < end:
        nl = data.find(b"\n", pos, end)
        nxt = end if nl == -1 else nl + 1
        out.append(_Piece(pos, nxt, kind, None, named))
        pos = nxt
    return out


def _snap_to_lines(
    pieces: list[_Piece], data: bytes, prefix: list[int]
) -> list[_Piece]:
    """Turn piece starts into contiguous, line-aligned chunk boundaries.

    Each boundary lands at the start of the line where the piece's content
    begins, then walks back over whole blank lines so inter-sibling blank
    space attaches to the following chunk. Leading/trailing trivia joins the
    first/last piece via the fixed outer bounds. A piece squeezed to zero
    width (siblings sharing a line) is dropped; the survivor inherits the
    first dropped piece's identity and any hard flags, since the dropped
    content now opens the survivor's first line.
    """
    total = len(data)
    bounds = [0]
    for piece in pieces[1:]:
        boundary = data.rfind(b"\n", 0, piece.start) + 1
        while boundary > 0:  # pull whole blank lines into this piece
            line_start = data.rfind(b"\n", 0, boundary - 1) + 1
            if prefix[boundary] - prefix[line_start] == 0:
                boundary = line_start
            else:
                break
        bounds.append(max(boundary, bounds[-1]))
    bounds.append(total)

    out: list[_Piece] = []
    pending_meta: tuple[str, str | None, bool] | None = None
    pending_hard = False
    for piece, start, end in zip(pieces, bounds, bounds[1:]):
        if start == end:
            if pending_meta is None:
                pending_meta = (piece.kind, piece.symbol, piece.named)
            pending_hard = pending_hard or piece.hard_before or piece.hard_after
            continue
        piece.start, piece.end = start, end
        if pending_meta is not None:
            piece.kind, piece.symbol, piece.named = pending_meta
            pending_meta = None
        piece.hard_before = piece.hard_before or pending_hard
        pending_hard = False
        out.append(piece)
    if not out:  # degenerate (e.g. whitespace-only file parsed to width 0)
        out = [_Piece(0, total, "text", None, True)]
    return out


def _merge(pieces: list[_Piece], prefix: list[int]) -> list[list[_Piece]]:
    """Merge phase: greedily pack pieces up to the budget, honoring hard cuts."""
    windows: list[list[_Piece]] = []
    current: list[_Piece] = []
    current_tokens = 0
    for piece in pieces:
        piece_tokens = _span_tokens(prefix, piece.start, piece.end)
        if current and (
            piece.hard_before or current_tokens + piece_tokens > MAX_CHUNK_TOKENS
        ):
            windows.append(current)
            current, current_tokens = [], 0
        current.append(piece)
        current_tokens += piece_tokens
        if piece.hard_after:
            windows.append(current)
            current, current_tokens = [], 0
    if current:
        windows.append(current)
    return windows
