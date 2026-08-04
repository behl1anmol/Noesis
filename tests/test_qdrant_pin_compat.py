"""The two Qdrant pins must stay a supported pair (issue #37, ADR-62).

`pyproject.toml` pins the *client*; `docker-compose.yml` pins the *server*.
Nothing connected them, and they drifted three minor versions apart — the app
warned on every connection while the whole suite stayed green, because
`runtime.py` is the only remote-client construction in `src/` and every
store-touching test uses `QdrantClient(":memory:")`, which skips the version
handshake entirely.

This test closes that gap offline: no server, no container, no network. It
reads both pins off disk and applies the client's own compatibility predicate
to them. It runs in the default suite, so CI sees the skew the moment either
pin moves alone.

`is_compatible` is imported from the installed client rather than
reimplemented. That is deliberate: this test's job is to enforce *the client's*
rule, and a local paraphrase would silently drift from it — the same class of
bug as the pins themselves drifting. If a future client relocates the
predicate, the ImportError is the point: a project whose pins are coupled to
that rule should notice when the rule moves. The `<1.19` cap bounds when that
can happen to a deliberate client bump.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from qdrant_client.common.version_check import is_compatible, parse_version

from importlib.metadata import version as dist_version

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPOSE_PATH = REPO_ROOT / "docker-compose.yml"
CI_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "ci.yml"

# Native BM25 sparse vectors, the M3 tech-stack floor recorded in the compose
# comment and Appendix C of the expanded architecture doc.
BM25_SERVER_FLOOR = (1, 15, 2)


def _image_tag_version(image: str) -> str:
    """`qdrant/qdrant:v1.18.3` -> `1.18.3`. Fails loudly on a floating tag:
    `latest` is exactly what the compose comment pins away from, so it must
    not slip through as an unparseable version."""
    _, _, tag = image.partition(":")
    assert tag, f"image {image!r} carries no tag — pin a version, never latest"
    return tag.removeprefix("v")


def _compose_server_version() -> str:
    compose = yaml.safe_load(COMPOSE_PATH.read_text())
    return _image_tag_version(compose["services"]["qdrant"]["image"])


def test_pinned_client_and_server_versions_are_compatible():
    server = _compose_server_version()
    client = dist_version("qdrant-client")
    assert is_compatible(client, server), (
        f"qdrant-client {client} (pyproject.toml) and Qdrant server {server} "
        f"(docker-compose.yml) are not a supported pair: major versions must "
        f"match and the minor difference must not exceed 1. Move both pins in "
        f"one commit — see ADR-62."
    )


def test_pinned_server_meets_the_bm25_floor():
    server = _compose_server_version()
    parsed = parse_version(server)
    as_tuple = (parsed.major, parsed.minor, int(parsed.rest[0]) if parsed.rest else 0)
    assert as_tuple >= BM25_SERVER_FLOOR, (
        f"docker-compose.yml pins Qdrant {server}, below the "
        f"{'.'.join(map(str, BM25_SERVER_FLOOR))} floor required for the native "
        f"BM25 sparse channel the M3 hybrid retrieval path depends on."
    )


def test_ci_service_container_uses_the_compose_pin():
    """The CI `server-compat` job runs its own copy of the image tag, so the
    version now appears in two files. Pin them to each other here rather than
    trusting a comment — a CI job testing a *different* server than the one
    developers run would verify nothing about the documented setup."""
    workflow = yaml.safe_load(CI_WORKFLOW_PATH.read_text())
    ci_image = workflow["jobs"]["server-compat"]["services"]["qdrant"]["image"]
    assert _image_tag_version(ci_image) == _compose_server_version(), (
        f"CI service image {ci_image!r} does not match docker-compose.yml. "
        f"Both must move together or CI stops testing what `docker compose "
        f"up -d` actually starts."
    )


def test_compose_publishes_qdrant_on_localhost_only():
    """Rule 2 applies to what this project tells people to run, and the
    compose file is that instruction. Bundled here because the ports move in
    the same edit as the image tag — a version bump is exactly when a
    published port would slip in — and because an unauthenticated vector
    database reachable off-host is not a defect anyone should learn about
    from a server-tier test they may never run."""
    compose = yaml.safe_load(COMPOSE_PATH.read_text())
    published = compose["services"]["qdrant"]["ports"]
    assert published, "compose publishes no Qdrant ports — did the service change?"
    for mapping in published:
        assert str(mapping).startswith("127.0.0.1:"), (
            f"compose publishes {mapping!r} — every Qdrant port must bind "
            f"127.0.0.1 only, never 0.0.0.0 or a bare port (CLAUDE.md rule 2)"
        )


def test_compose_disables_qdrant_outbound_telemetry():
    """ADR-25 promises nothing leaves the machine; the Qdrant image ships
    `telemetry_disabled: false` and reports to telemetry.qdrant.io on start.
    Port binding only governs inbound traffic, so this is the only thing
    stopping the datastore from phoning home (ADR-63)."""
    compose = yaml.safe_load(COMPOSE_PATH.read_text())
    env = compose["services"]["qdrant"].get("environment") or {}
    if isinstance(env, list):  # compose also accepts KEY=VALUE strings
        env = dict(item.split("=", 1) for item in env)
    assert str(env.get("QDRANT__TELEMETRY_DISABLED", "")).lower() == "true", (
        "docker-compose.yml must set QDRANT__TELEMETRY_DISABLED=true — the "
        "image defaults it to false and reports usage statistics outbound, "
        "which contradicts ADR-25 (see ADR-63)."
    )
