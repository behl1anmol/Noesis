"""Mechanics tests for the cold-start harness (tests/perf/cold_start_harness.py).

These run in the default suite: no model, no network, no Qdrant. They pin the
two things that would silently make the harness lie — byte accounting and the
delete guard — plus the scenario-ordering rule that gives 'warm' its meaning,
and the --model/--dim/--qdrant-url plumbing PR #49 review found broken.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from .cold_start_harness import (
    Meter,
    _prefetch_command,
    _remote_collection_name,
    _wipe,
    _worker_settings,
    dir_bytes,
    format_verdict,
    scenario_complaint,
    unique_labels,
)


def test_dir_bytes_counts_a_blob_once_despite_a_symlink(tmp_path: Path) -> None:
    """huggingface_hub stores a weight file in blobs/ and symlinks it into
    snapshots/. Following those symlinks reports every model twice — an
    early draft of this harness claimed a 1.10 GB download for a 548 MB
    model that way. dir_bytes must skip symlinks."""
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    blob = blobs / "abc123"
    blob.write_bytes(b"x" * 1000)
    snapshots = tmp_path / "snapshots" / "main"
    snapshots.mkdir(parents=True)
    (snapshots / "model.safetensors").symlink_to(blob)

    assert dir_bytes(tmp_path) == 1000


def test_dir_bytes_is_zero_for_a_missing_directory(tmp_path: Path) -> None:
    assert dir_bytes(tmp_path / "never-created") == 0


def test_meter_attributes_growth_to_the_phase_that_caused_it(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    meter = Meter({"cache": cache})

    with meter.phase("quiet"):
        pass
    with meter.phase("downloads"):
        (cache / "weights").write_bytes(b"y" * 4096)
    with meter.phase("quiet_again"):
        pass

    quiet, downloads, quiet_again = meter.phases
    assert quiet["fetched_bytes_total"] == 0
    assert downloads["fetched_bytes_total"] == 4096
    assert downloads["fetched_bytes"] == {"cache": 4096}
    assert quiet_again["fetched_bytes_total"] == 0


def test_meter_closes_the_phase_even_when_the_body_raises(tmp_path: Path) -> None:
    """A worker that dies mid-download must still leave an attributable
    record; a phase with no 'seconds' key would crash the report renderer."""
    meter = Meter({"cache": tmp_path})
    with pytest.raises(RuntimeError):
        with meter.phase("boom"):
            raise RuntimeError("boom")
    assert "seconds" in meter.phases[0]
    assert meter.phases[0]["fetched_bytes_total"] == 0


def test_wipe_refuses_a_path_outside_the_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    outside = tmp_path / "real-cache"
    outside.mkdir()
    (outside / "keep").write_text("precious")

    with pytest.raises(SystemExit):
        _wipe(outside, workspace)
    assert (outside / "keep").exists()


def test_wipe_refuses_the_workspace_itself(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    with pytest.raises(SystemExit):
        _wipe(workspace, workspace)
    assert workspace.exists()


def test_wipe_removes_a_directory_inside_the_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    target = workspace / "cache" / "hf"
    target.mkdir(parents=True)
    (target / "blob").write_text("stale")

    _wipe(target, workspace)
    assert not target.exists()


def test_warm_cannot_be_the_first_scenario() -> None:
    """'warm' is defined as 'the caches the previous scenario left behind'.
    Run first, it would silently measure whatever the workspace happened to
    hold, which is exactly the stale-artifact trap lesson 8 records.

    Asserted against `scenario_complaint`, not `main`: the first version of
    this test called `main(["--scenarios", "warm"])` and still passed with
    the rule deleted, because `main` went on to run a real 138-second
    workload that eventually raised SystemExit for an unrelated reason.
    """
    assert scenario_complaint(["warm"]) is not None
    assert scenario_complaint(["cold", "warm"]) is None


def test_unknown_scenario_is_rejected() -> None:
    complaint = scenario_complaint(["cold", "tepid"])
    assert complaint is not None and "tepid" in complaint


def test_an_empty_scenario_list_is_rejected() -> None:
    assert scenario_complaint([]) is not None


def test_unique_labels_disambiguates_repeated_scenarios() -> None:
    """Repeats are how the harness gets a noise floor. Without distinct
    labels they would share one run directory, so each repeat would wipe
    and overwrite the previous one's result."""
    assert unique_labels(["cold", "warm", "warm", "warm"]) == [
        "cold",
        "warm",
        "warm-2",
        "warm-3",
    ]
    assert unique_labels(["cold", "warm"]) == ["cold", "warm"]


def _result(scenario: str, label: str, seconds: float, fetched: int) -> dict:
    return {
        "scenario": scenario,
        "label": label,
        "sequence": "query-first",
        "phases": [
            {
                "phase": "first_search",
                "seconds": seconds,
                "fetched_bytes": {"hf": fetched} if fetched else {},
                "fetched_bytes_total": fetched,
            }
        ],
    }


def test_verdict_refuses_a_seconds_delta_inside_the_warm_spread() -> None:
    """Identical warm work on a contended 4-core box measured 11.6s and
    18.2s; the same box quiet measured 10.72-11.33s. A verdict that read a
    delta smaller than that spread as signal would be reporting contention
    as a finding."""
    verdict = format_verdict(
        [
            _result("cold", "cold", 21.8, 548_000_000),
            _result("warm", "warm", 18.2, 0),
            _result("warm", "warm-2", 11.6, 0),
        ]
    )
    assert "not separable" in verdict
    assert "548 MB fetched" in verdict


def test_verdict_claims_a_seconds_delta_that_clears_the_spread() -> None:
    verdict = format_verdict(
        [
            _result("cold", "cold", 300.0, 548_000_000),
            _result("warm", "warm", 12.0, 0),
            _result("warm", "warm-2", 12.5, 0),
        ]
    )
    assert "beyond the spread" in verdict
    assert "not separable" not in verdict


def test_verdict_makes_no_seconds_claim_from_a_single_warm_run() -> None:
    verdict = format_verdict(
        [_result("cold", "cold", 21.8, 548_000_000), _result("warm", "warm", 18.2, 0)]
    )
    assert "no floor" in verdict
    assert "not separable" not in verdict and "beyond the spread" not in verdict


def test_prefetch_command_carries_the_selected_model() -> None:
    """PR #49 review (r3942607710): the 'prefetched' scenario invoked
    `noesis.prefetch` with no --model, so it always downloaded prefetch's
    own default (CodeRankEmbed) regardless of --model. A run with a
    non-default --model then reported the SELECTED model's full cold-start
    cost as first_search time, reading as 'prefetch did not help' when
    prefetch was never asked to fetch that model."""
    cmd = _prefetch_command("some-org/some-model")
    assert "--model" in cmd
    assert cmd[cmd.index("--model") + 1] == "some-org/some-model"
    assert "--skip-reranker" in cmd


def test_worker_settings_passes_through_model_and_dim() -> None:
    """PR #49 review (r3942607711): --dim had no effect at all — the worker
    always built EmbedderSettings with its 768 default, so a --model with a
    different vector size created a 768-dim collection and then failed on
    its first real vector, deep inside the workload rather than at the
    --dim/--model mismatch that caused it."""
    settings = _worker_settings({
        "db_path": "/tmp/does-not-need-to-exist.sqlite",
        "model": "some-org/some-model",
        "dim": 1024,
        "device": "",
        "qdrant_url": "",
        "collection": "irrelevant-here",
    })
    assert settings.embedder.model == "some-org/some-model"
    assert settings.embedder.dim == 1024


def test_worker_settings_uses_the_namespaced_collection() -> None:
    settings = _worker_settings({
        "db_path": "/tmp/does-not-need-to-exist.sqlite",
        "model": "m",
        "dim": 768,
        "device": "",
        "qdrant_url": "http://127.0.0.1:6333",
        "collection": "noesis_perf_cold_start_warm_2",
    })
    assert settings.qdrant.collection == "noesis_perf_cold_start_warm_2"
    assert settings.qdrant.url == "http://127.0.0.1:6333"


def test_remote_collection_name_is_namespaced_per_label() -> None:
    """PR #49 review (r3942607713): every scenario shared one fixed
    collection name against a real --qdrant-url server, with nothing
    deleting its points afterward — so cold/warm repeats permanently
    appended to the same corpus and no run was isolated from the last."""
    cold = _remote_collection_name("cold")
    warm2 = _remote_collection_name("warm-2")
    assert cold != warm2
    assert cold == _remote_collection_name("cold")


def test_remote_collection_name_has_no_bare_hyphens() -> None:
    """Labels like 'warm-2' must not produce a hyphenated collection name
    verbatim — kept boring/portable rather than relying on Qdrant accepting
    hyphens in every deployment."""
    assert "-" not in _remote_collection_name("warm-2")
