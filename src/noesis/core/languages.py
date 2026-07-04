"""Canonical language names, keyed by file extension.

This module is the single source for the project's canonical language
identifiers (stored in ``files.language``). LANGUAGE_MAP (M5, §3.5) is the
explicit seam mapping each canonical name to (a) its tree-sitter-language-pack
grammar name and (b) its ast-grep language string — the two engines do not
share identifiers by contract, even where they coincide today. Keep this a
plain dict module.
"""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class LanguageMapping:
    grammar: str  # tree-sitter-language-pack grammar name
    ast_grep: str  # ast-grep language string (SgRoot's second argument)


# Canonical name → engine identifiers. Grammar names resolve 1:1 today (the
# chunker passes canonical names straight to get_parser), but the mapping is
# kept explicit because the two engines' identifiers are independent (§3.5,
# risk 11). ast-grep strings were probe-verified against ast-grep-py 0.44.0:
# an unsupported string does not raise — it panics (pyo3 PanicException), so
# membership here is the *only* gate before SgRoot is called. ``toml`` and
# ``sql`` are canonical languages with no ast-grep 0.44 support and are
# deliberately absent: structural search reports them unsupported.
LANGUAGE_MAP: dict[str, LanguageMapping] = {
    "python": LanguageMapping("python", "python"),
    "javascript": LanguageMapping("javascript", "javascript"),
    "typescript": LanguageMapping("typescript", "typescript"),
    "tsx": LanguageMapping("tsx", "tsx"),
    "go": LanguageMapping("go", "go"),
    "rust": LanguageMapping("rust", "rust"),
    "java": LanguageMapping("java", "java"),
    "c": LanguageMapping("c", "c"),
    "cpp": LanguageMapping("cpp", "cpp"),
    "csharp": LanguageMapping("csharp", "csharp"),
    "ruby": LanguageMapping("ruby", "ruby"),
    "php": LanguageMapping("php", "php"),
    "swift": LanguageMapping("swift", "swift"),
    "kotlin": LanguageMapping("kotlin", "kotlin"),
    "scala": LanguageMapping("scala", "scala"),
    "bash": LanguageMapping("bash", "bash"),
    "html": LanguageMapping("html", "html"),
    "css": LanguageMapping("css", "css"),
    "json": LanguageMapping("json", "json"),
    "yaml": LanguageMapping("yaml", "yaml"),
    "markdown": LanguageMapping("markdown", "markdown"),
}
