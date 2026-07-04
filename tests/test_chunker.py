"""cAST chunker tests: concatenation invariant, budgets, metadata, fallback."""

from __future__ import annotations

from pathlib import Path

import pytest

from noesis.core.chunker import (
    MAX_CHUNK_TOKENS,
    MIN_CHUNK_TOKENS,
    _token_estimate,
    chunk_file,
)
from noesis.core.languages import detect_language

FIXTURES = Path(__file__).parent / "fixtures" / "chunker"
FIXTURE_NAMES = ["sample.py", "sample.js", "sample.go", "Sample.java", "notes.txt"]


def load_fixture(name: str):
    text = (FIXTURES / name).read_text()
    chunks = chunk_file(
        text, language=detect_language(name), file_path=name, file_hash="deadbeef"
    )
    return text, chunks


def make_pathological() -> str:
    """Deeply nested classes, one huge function, one indivisible giant line."""
    nested = []
    for depth in range(25):
        indent = "    " * depth
        nested.append(f"{indent}class Level{depth}:")
        nested.append(f"{indent}    tag = {depth}")
    huge = [
        "def enormous(seed):",
        '    """One function far past the max chunk budget."""',
        "    acc = seed",
    ]
    for i in range(220):
        huge.append(f"    acc = (acc * {i} + {i * 7}) % 99991  # keep mixing state {i}")
    huge.append("    return acc")
    giant_line = 'PAYLOAD = "' + "x" * 6000 + '"'
    return "\n".join(nested) + "\n\n\n" + "\n".join(huge) + "\n\n" + giant_line + "\n"


def make_uniform() -> str:
    """Many similar mid-size functions, so greedy merging fills every chunk."""
    parts = []
    for i in range(30):
        body = "\n".join(
            f"    value_{j} = value_{j - 1} * {i} + {j}" for j in range(1, 12)
        )
        parts.append(f"def step_{i}(value_0):\n{body}\n    return value_11\n\n")
    return "".join(parts)


GENERATED = {
    "pathological.py": make_pathological(),
    "uniform.py": make_uniform(),
}


def all_cases():
    for name in FIXTURE_NAMES:
        yield name, *load_fixture(name)
    for name, text in GENERATED.items():
        chunks = chunk_file(
            text, language="python", file_path=name, file_hash="deadbeef"
        )
        yield name, text, chunks


def test_concatenation_invariant() -> None:
    for name, text, chunks in all_cases():
        assert chunks, name
        assert "".join(c.text for c in chunks) == text, name


def test_no_overlap_and_ordered_lines() -> None:
    for name, _, chunks in all_cases():
        for prev, cur in zip(chunks, chunks[1:]):
            assert cur.start_line == prev.end_line + 1, name


def test_line_ranges_slice_file_exactly() -> None:
    for name, text, chunks in all_cases():
        lines = text.splitlines(keepends=True)
        for chunk in chunks:
            assert 1 <= chunk.start_line <= chunk.end_line, name
            sliced = "".join(lines[chunk.start_line - 1 : chunk.end_line])
            assert sliced == chunk.text, (name, chunk.start_line, chunk.end_line)


def content_lines(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip())


def test_max_budget() -> None:
    for name, _, chunks in all_cases():
        for chunk in chunks:
            if content_lines(chunk.text) <= 1:
                continue  # indivisible single-line remnant may exceed the cap
            assert _token_estimate(chunk.text) <= MAX_CHUNK_TOKENS, (
                name,
                chunk.start_line,
            )


def test_min_budget_mostly_met() -> None:
    # Strict on the uniform file: every non-tail chunk reaches the floor.
    chunks = chunk_file(
        GENERATED["uniform.py"], language="python", file_path="u.py", file_hash="x"
    )
    assert len(chunks) >= 2
    for chunk in chunks[:-1]:
        assert _token_estimate(chunk.text) >= MIN_CHUNK_TOKENS
    # Lenient on fixtures: the hard boundaries around an oversized definition
    # legitimately close neighbouring chunks early, so only require that some
    # non-tail chunk reaches the floor in every multi-chunk fixture.
    for name in FIXTURE_NAMES:
        _, chunks = load_fixture(name)
        heads = chunks[:-1]
        if not heads:
            continue
        assert any(_token_estimate(c.text) >= MIN_CHUNK_TOKENS for c in heads), name


def test_python_symbols_and_node_types() -> None:
    body = "\n".join(f"    total = (total * {j} + {j}) % 1000003" for j in range(1, 90))
    text = (
        f"def alpha(total):\n{body}\n    return total\n\n\n"
        f"def beta(total):\n{body}\n    return total\n"
    )
    chunks = chunk_file(text, language="python", file_path="two.py", file_hash="x")
    assert len(chunks) == 2
    assert chunks[0].node_type == "function_definition"
    assert chunks[0].symbol_name == "alpha"
    assert chunks[1].node_type == "function_definition"
    assert chunks[1].symbol_name == "beta"
    # Fixture: the over-budget function opens its own chunk, symbol intact.
    _, chunks = load_fixture("sample.py")
    render = [c for c in chunks if c.symbol_name == "render_report"]
    assert len(render) == 1
    assert render[0].node_type == "function_definition"
    assert render[0].text.lstrip().startswith("def render_report")
    assert all(isinstance(c.node_type, str) and c.node_type for c in chunks)


def test_go_symbols_and_node_types() -> None:
    body = "\n".join(f"\tv{j} := (v{j - 1}*{j} + {j}) % 1000003" for j in range(1, 80))
    text = (
        "package main\n\n"
        f"func Alpha(v0 int) int {{\n{body}\n\treturn v79\n}}\n\n"
        f"func Beta(v0 int) int {{\n{body}\n\treturn v79\n}}\n"
    )
    chunks = chunk_file(text, language="go", file_path="two.go", file_hash="x")
    assert len(chunks) == 2
    assert chunks[1].node_type == "function_declaration"
    assert chunks[1].symbol_name == "Beta"
    assert chunks[1].text.lstrip().startswith("func Beta")
    # Fixture-level checks on the go sample.
    _, go_chunks = load_fixture("sample.go")
    assert all(c.language == "go" for c in go_chunks)
    assert go_chunks[0].node_type == "source_file"


def test_function_never_split_below_signature() -> None:
    text = GENERATED["pathological.py"]
    chunks = chunk_file(text, language="python", file_path="p.py", file_hash="x")
    holders = [c for c in chunks if "def enormous(seed):" in c.text]
    assert len(holders) == 1
    holder = holders[0]
    assert holder.text.lstrip().startswith("def enormous")
    assert holder.symbol_name == "enormous"
    # The 6000-char single line survives as one indivisible over-budget chunk.
    giant = [c for c in chunks if "PAYLOAD" in c.text]
    assert all(content_lines(c.text) == 1 for c in giant)
    assert len(giant) == 1
    assert _token_estimate(giant[0].text) > MAX_CHUNK_TOKENS


def test_fallback_unsupported_language() -> None:
    text, chunks = load_fixture("notes.txt")
    assert detect_language("notes.txt") is None
    assert chunks
    assert all(c.node_type == "text" for c in chunks)
    assert all(c.symbol_name is None for c in chunks)
    assert "".join(c.text for c in chunks) == text
    # A language string with no grammar degrades the same way, never errors.
    weird = chunk_file(text, language="klingon", file_path="n", file_hash="x")
    assert all(c.node_type == "text" for c in weird)
    assert "".join(c.text for c in weird) == text


def test_empty_and_degenerate_input() -> None:
    assert chunk_file("", language="python", file_path="e.py", file_hash="x") == []
    assert chunk_file("", language=None, file_path="e", file_hash="x") == []
    blank = "\n\n   \n"
    chunks = chunk_file(blank, language="python", file_path="b.py", file_hash="x")
    assert "".join(c.text for c in chunks) == blank


def test_oracle_astchunk() -> None:
    astchunk = pytest.importorskip("astchunk")
    # astchunk only supports python/java/csharp/typescript; its chunks also
    # do not guarantee exact concatenation, so the comparison is coverage-
    # shaped: both chunkers produce output, ours reproduces the file exactly.
    astchunk_language = {"sample.py": "python", "Sample.java": "java"}
    for name in FIXTURE_NAMES:
        text, ours = load_fixture(name)
        assert len(ours) > 0, name
        assert "".join(c.text for c in ours) == text, name
        language = astchunk_language.get(name)
        if language is None:
            continue
        builder = astchunk.ASTChunkBuilder(
            max_chunk_size=MAX_CHUNK_TOKENS * 4,  # its budget is non-ws chars
            language=language,
            metadata_template="default",
        )
        theirs = builder.chunkify(text)
        assert len(theirs) > 0, name
        assert all(window["content"] for window in theirs), name
