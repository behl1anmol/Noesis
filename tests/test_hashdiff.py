import hashlib
from pathlib import Path

from code_index.core.hashdiff import DiffResult, hash_file, partition


def write(root: Path, rel: str, content: str) -> str:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return hashlib.sha256(content.encode()).hexdigest()


def test_hash_file_matches_hashlib(tmp_path: Path) -> None:
    expected = write(tmp_path, "a.py", "print('hi')\n")
    assert hash_file(tmp_path / "a.py") == expected


def test_partition_all_four_buckets(tmp_path: Path) -> None:
    h_same = write(tmp_path, "same.py", "same\n")
    write(tmp_path, "edited.py", "v2\n")
    write(tmp_path, "brand_new.py", "new\n")
    h_old_edited = hashlib.sha256(b"v1\n").hexdigest()
    h_gone = hashlib.sha256(b"gone\n").hexdigest()

    stored = {"same.py": h_same, "edited.py": h_old_edited, "gone.py": h_gone}
    discovered = ["same.py", "edited.py", "brand_new.py"]

    result = partition(tmp_path, discovered, stored)
    assert result.new == ("brand_new.py",)
    assert result.changed == ("edited.py",)
    assert result.unchanged == ("same.py",)
    assert result.deleted == ("gone.py",)
    assert set(result.hashes) == set(discovered)
    assert result.hashes["same.py"] == h_same


def test_partition_empty_state_everything_new(tmp_path: Path) -> None:
    write(tmp_path, "a.py", "a")
    write(tmp_path, "sub/b.py", "b")
    result = partition(tmp_path, ["a.py", "sub/b.py"], {})
    assert result.new == ("a.py", "sub/b.py")
    assert result.changed == result.unchanged == result.deleted == ()


def test_partition_vanished_file_counts_deleted(tmp_path: Path) -> None:
    h = write(tmp_path, "a.py", "a")
    # discovered claims b.py exists, but it does not on disk
    stored = {"a.py": h, "b.py": "deadbeef"}
    result = partition(tmp_path, ["a.py", "b.py"], stored)
    assert result.unchanged == ("a.py",)
    assert result.deleted == ("b.py",)


def test_rerun_is_idempotent(tmp_path: Path) -> None:
    write(tmp_path, "a.py", "a")
    first = partition(tmp_path, ["a.py"], {})
    second = partition(tmp_path, ["a.py"], dict(first.hashes))
    assert second == DiffResult(unchanged=("a.py",), hashes=first.hashes)
