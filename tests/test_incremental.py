"""M1 exit-criterion test: incremental detection on a fixture repo.

Full loop: register -> discover -> partition -> persist state -> mutate repo
-> rerun -> only the mutations are detected.
"""

from pathlib import Path

from noesis.core import state
from noesis.core.discovery import discover_files
from noesis.core.hashdiff import partition
from noesis.core.languages import detect_language


def build_repo(root: Path) -> None:
    (root / "src").mkdir(parents=True)
    (root / "src" / "main.py").write_text("def main():\n    return 1\n")
    (root / "src" / "util.py").write_text("def util():\n    return 2\n")
    (root / "README.md").write_text("# fixture\n")
    (root / ".gitignore").write_text("*.log\n")
    (root / "noise.log").write_text("ignored\n")
    (root / ".env").write_text("SECRET=1\n")


def run_index(conn, project_id: str, root: Path):
    discovered = discover_files(root)
    stored = state.get_file_states(conn, project_id)
    diff = partition(root, discovered, stored)
    for rel in (*diff.new, *diff.changed):
        state.upsert_file(
            conn,
            project_id,
            rel,
            content_hash=diff.hashes[rel],
            language=detect_language(rel),
        )
    state.delete_files(conn, project_id, diff.deleted)
    return diff


def test_incremental_detection_end_to_end(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    build_repo(repo)
    conn = state.connect(tmp_path / "state.sqlite")
    state.init_db(conn)
    project_id = state.register_project(conn, repo, "fake-embedder-v1")

    # First run: everything is new; ignored/secret files never enter state.
    first = run_index(conn, project_id, repo)
    assert set(first.new) == {".gitignore", "README.md", "src/main.py", "src/util.py"}
    assert first.changed == first.deleted == ()
    assert "noise.log" not in first.new and ".env" not in first.new

    # No-op rerun: everything unchanged.
    second = run_index(conn, project_id, repo)
    assert second.new == second.changed == second.deleted == ()
    assert set(second.unchanged) == set(first.new)

    # Mutate: edit one, add one, delete one.
    (repo / "src" / "main.py").write_text("def main():\n    return 99\n")
    (repo / "src" / "fresh.py").write_text("fresh = True\n")
    (repo / "src" / "util.py").unlink()

    third = run_index(conn, project_id, repo)
    assert third.changed == ("src/main.py",)
    assert third.new == ("src/fresh.py",)
    assert third.deleted == ("src/util.py",)
    assert set(third.unchanged) == {".gitignore", "README.md"}

    # State table reflects the final truth.
    final_state = state.get_file_states(conn, project_id)
    assert "src/util.py" not in final_state
    assert final_state["src/fresh.py"]
    conn.close()
