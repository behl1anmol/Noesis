"""Device resolution (M4 follow-up, lesson 4)."""

from __future__ import annotations

from noesis.core.compute import resolve_device


def test_configured_device_wins_verbatim():
    # Explicit operator choice is never second-guessed and skips torch import.
    assert resolve_device("cuda") == "cuda"
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("mps") == "mps"


def test_auto_detect_returns_a_known_device():
    # Whatever this test host has, it resolves to one of the known strings
    # (no cuda/mps in CI → "cpu"); the point is it resolves explicitly.
    assert resolve_device(None) in {"cuda", "mps", "cpu"}
    assert resolve_device("") in {"cuda", "mps", "cpu"}  # empty == unset
