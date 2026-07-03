"""Canonical language names, keyed by file extension.

This module is the single source for the project's canonical language
identifiers (stored in ``files.language``). In M5 it grows the LANGUAGE_MAP
seam mapping canonical names to tree-sitter-language-pack grammar names and
ast-grep language strings (architecture §3.5) — keep it a plain dict module.
"""

from __future__ import annotations

from pathlib import Path

EXT_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "scala",
    ".sh": "bash",
    ".bash": "bash",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".md": "markdown",
    ".sql": "sql",
}


def detect_language(path: str | Path) -> str | None:
    """Return the canonical language for *path*'s extension, or None."""
    return EXT_TO_LANGUAGE.get(Path(path).suffix.lower())
