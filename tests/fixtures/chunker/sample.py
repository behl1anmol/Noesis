"""Sample module used to exercise the cAST chunker on Python source."""

from __future__ import annotations

import re

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def normalize_name(raw: str) -> str:
    """Lower-case an identifier and collapse repeated separators."""
    cleaned = raw.strip().replace("-", "_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.lower()


def split_words(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


class Cursor:
    """Tracks a position while scanning a buffer."""

    def __init__(self, buffer: str) -> None:
        self.buffer = buffer
        self.pos = 0

    def peek(self) -> str:
        if self.pos >= len(self.buffer):
            return ""
        return self.buffer[self.pos]

    def advance(self, count: int = 1) -> str:
        start = self.pos
        self.pos = min(self.pos + count, len(self.buffer))
        return self.buffer[start : self.pos]

    def eof(self) -> bool:
        return self.pos >= len(self.buffer)


def classify_line(line: str) -> str:
    stripped = line.strip()
    if not stripped:
        return "blank"
    if stripped.startswith("#"):
        return "comment"
    if stripped.endswith(":"):
        return "block-opener"
    return "code"


def render_report(rows: list[dict], *, title: str = "Report") -> str:
    """Render rows into a padded text table (deliberately over the budget).

    This function is intentionally larger than MAX_CHUNK_TOKENS so the
    chunker must split it via its own children rather than keep it whole.
    """
    if not rows:
        return f"{title}\n(no rows)\n"
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    widths = {column: len(column) for column in columns}
    for row in rows:
        for column in columns:
            value = str(row.get(column, ""))
            if len(value) > widths[column]:
                widths[column] = len(value)
    header_cells = [column.ljust(widths[column]) for column in columns]
    separator_cells = ["-" * widths[column] for column in columns]
    lines = [title, " | ".join(header_cells), "-+-".join(separator_cells)]
    for row in rows:
        cells = []
        for column in columns:
            value = str(row.get(column, ""))
            cells.append(value.ljust(widths[column]))
        lines.append(" | ".join(cells))
    numeric_totals: dict[str, float] = {}
    for row in rows:
        for column in columns:
            value = row.get(column)
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                numeric_totals[column] = numeric_totals.get(column, 0.0) + value
    if numeric_totals:
        total_cells = []
        for column in columns:
            if column in numeric_totals:
                total_cells.append(f"{numeric_totals[column]:g}".ljust(widths[column]))
            else:
                total_cells.append("".ljust(widths[column]))
        lines.append(" | ".join(total_cells))
    blank_counts: dict[str, int] = {}
    for row in rows:
        for column in columns:
            value = str(row.get(column, "")).strip()
            if not value:
                blank_counts[column] = blank_counts.get(column, 0) + 1
    for column in columns:
        missing = blank_counts.get(column, 0)
        if missing:
            share = 100.0 * missing / len(rows)
            lines.append(
                f"note: column {column!r} blank in {missing} rows ({share:.1f}%)"
            )
    widest_column = max(columns, key=lambda column: widths[column])
    narrowest_column = min(columns, key=lambda column: widths[column])
    lines.append(f"widest column: {widest_column!r} ({widths[widest_column]} chars)")
    lines.append(
        f"narrowest column: {narrowest_column!r} ({widths[narrowest_column]} chars)"
    )
    duplicate_count = 0
    seen_signatures: set[tuple] = set()
    for row in rows:
        signature = tuple(str(row.get(column, "")) for column in columns)
        if signature in seen_signatures:
            duplicate_count += 1
        else:
            seen_signatures.add(signature)
    if duplicate_count:
        lines.append(f"warning: {duplicate_count} duplicate rows detected")
    longest_cell = ""
    for row in rows:
        for column in columns:
            value = str(row.get(column, ""))
            if len(value) > len(longest_cell):
                longest_cell = value
    if len(longest_cell) > 40:
        preview = longest_cell[:37] + "..."
        lines.append(f"longest cell preview: {preview}")
    per_row_widths = [sum(len(str(value)) for value in row.values()) for row in rows]
    average_width = sum(per_row_widths) / len(per_row_widths)
    lines.append(
        f"average row payload: {average_width:.1f} chars over {len(rows)} rows"
    )
    numeric_columns = sorted(numeric_totals)
    for column in numeric_columns:
        values = [
            row[column] for row in rows if isinstance(row.get(column), (int, float))
        ]
        values = [value for value in values if not isinstance(value, bool)]
        if not values:
            continue
        smallest, largest = min(values), max(values)
        mean_value = sum(values) / len(values)
        lines.append(
            f"stats {column!r}: min={smallest:g} max={largest:g} mean={mean_value:.2f}"
        )
    sparse_columns = [
        column for column in columns if blank_counts.get(column, 0) * 2 > len(rows)
    ]
    if sparse_columns:
        joined = ", ".join(repr(column) for column in sparse_columns)
        lines.append(f"warning: sparse columns ({joined}) exceed 50% blanks")
    checksum = 0
    for line in lines:
        for char in line:
            checksum = (checksum * 31 + ord(char)) % 1_000_000_007
    lines.append(f"checksum: {checksum:09d}")
    footer = f"{len(rows)} rows x {len(columns)} columns"
    lines.append(footer)
    return "\n".join(lines) + "\n"


def summarize(rows: list[dict]) -> dict:
    report = render_report(rows)
    return {
        "lines": report.count("\n"),
        "rows": len(rows),
        "words": len(split_words(report)),
    }
