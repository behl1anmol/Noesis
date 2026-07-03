"""Discovery filter tests: gitignore semantics, skip-list, binary/size caps."""

from __future__ import annotations

from pathlib import Path

from code_index.core.discovery import (
    DiscoveryConfig,
    discover_files,
    is_secret_path,
)
from code_index.core.languages import detect_language


def make_tree(root: Path, files: dict[str, bytes | str]) -> None:
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            p.write_bytes(content)
        else:
            p.write_text(content)


def test_gitignore_excludes_and_negation_reincludes(tmp_path: Path) -> None:
    make_tree(
        tmp_path,
        {
            ".gitignore": "*.log\n!keep.log\n",
            "app.py": "print('hi')\n",
            "debug.log": "x\n",
            "keep.log": "x\n",
        },
    )
    found = discover_files(tmp_path)
    assert "app.py" in found
    assert "debug.log" not in found
    assert "keep.log" in found


def test_nested_gitignore_applies_only_under_its_subdir(tmp_path: Path) -> None:
    make_tree(
        tmp_path,
        {
            "sub/.gitignore": "*.txt\n",
            "sub/ignored.txt": "x\n",
            "sub/code.py": "pass\n",
            "top.txt": "kept: root has no ignore rule\n",
        },
    )
    found = discover_files(tmp_path)
    assert "top.txt" in found
    assert "sub/code.py" in found
    assert "sub/ignored.txt" not in found


def test_excluded_dirs_pruned(tmp_path: Path) -> None:
    make_tree(
        tmp_path,
        {
            "node_modules/pkg/index.js": "x\n",
            ".git/config": "x\n",
            "src/main.py": "pass\n",
        },
    )
    found = discover_files(tmp_path)
    assert found == ["src/main.py"]


def test_secret_skip_list_applies_without_gitignore(tmp_path: Path) -> None:
    make_tree(
        tmp_path,
        {
            ".env": "TOKEN=abc\n",
            "certs/server.pem": "-----BEGIN-----\n",
            "config.py": "pass\n",
        },
    )
    found = discover_files(tmp_path)
    assert ".env" not in found
    assert "certs/server.pem" not in found
    assert "config.py" in found


def test_is_secret_path_patterns() -> None:
    assert is_secret_path(".env")
    assert is_secret_path("deploy/.env.production")
    assert is_secret_path("keys/id_rsa")
    assert is_secret_path(".ssh/known_hosts")
    assert not is_secret_path("src/env.py")


def test_binary_and_oversize_excluded(tmp_path: Path) -> None:
    make_tree(
        tmp_path,
        {
            "blob.bin": b"abc\x00def",
            "big.py": "# " + "x" * 100 + "\n",
            "ok.py": "pass\n",
        },
    )
    found = discover_files(tmp_path, DiscoveryConfig(max_file_bytes=50))
    assert "blob.bin" not in found
    assert "big.py" not in found
    assert "ok.py" in found


def test_output_sorted_posix(tmp_path: Path) -> None:
    make_tree(
        tmp_path,
        {
            "b/two.py": "pass\n",
            "a/one.py": "pass\n",
            "zed.py": "pass\n",
        },
    )
    found = discover_files(tmp_path)
    assert found == sorted(found)
    assert "a/one.py" in found and "b/two.py" in found and "zed.py" in found
    assert all("\\" not in p for p in found)


def test_detect_language() -> None:
    assert detect_language("src/app.py") == "python"
    assert detect_language("web/App.TSX") == "tsx"
    assert detect_language("main.go") == "go"
    assert detect_language("README.md") == "markdown"
    assert detect_language("mystery.xyz") is None
