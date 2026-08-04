"""The two Qdrant pins must stay a supported pair (issue #37, ADR-62).

`pyproject.toml` pins the *client*; `docker-compose.yml` pins the *server*.
Nothing connected them, and they drifted three minor versions apart — the app
warned on every connection while the whole suite stayed green, because
`runtime.py` is the only remote-client construction in `src/` and every
store-touching test uses `QdrantClient(":memory:")`, which skips the version
handshake entirely.

This test closes that gap offline: no server, no container, no network. It
reads the server pin off disk, takes the client version from the *resolved
environment*, and holds them to ADR-62's rule. It runs in the default suite,
so CI sees a skew the moment either end moves alone.

Two predicates, deliberately, because they answer different questions.
`is_compatible` is the client's own rule — imported rather than
reimplemented, since a local paraphrase would silently drift from it, the
same class of bug as the pins themselves drifting. But that rule tolerates a
minor difference of 1, and ADR-62 decided something stricter: both pins land
on the *same* minor, zero skew. Asserting only the vendor's predicate means a
green that says "qdrant-client tolerates this", not "we decided this" — and
issue #37 began at exactly one minor of skew before it grew to three, so the
looser check is blind to the state the bug starts in. The same-minor test is
the one that enforces the decision; the `is_compatible` test stays so that a
future client *tightening* its rule is also caught. If a future client
relocates the predicate, the ImportError is the point: a project whose pins
are coupled to that rule should notice when the rule moves. The `<1.19` cap
bounds when that can happen to a deliberate client bump.

On "the client version": `importlib.metadata` reports what is installed, not
what `pyproject.toml` declares. That is the stronger check — it is the client
that will actually open the socket — but it means this file verifies the
*resolved* pair. `uv sync` re-locks when `pyproject.toml` changes, so in CI
the two agree; in a stale local venv they can disagree, and it is the venv
that is wrong.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from qdrant_client.common.version_check import is_compatible, parse_version

from importlib.metadata import version as dist_version

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPOSE_PATH = REPO_ROOT / "docker-compose.yml"
CI_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "ci.yml"
INSTALL_DOC_PATH = REPO_ROOT / "docs" / "getting-started" / "installation.md"
COLAB_DOC_PATH = REPO_ROOT / "architecture-docs" / "m6-agent-connection-guide.md"

# Native BM25 sparse vectors, the M3 tech-stack floor recorded in the compose
# comment and Appendix C of the expanded architecture doc.
BM25_SERVER_FLOOR = (1, 15, 2)

# Every way this repo writes the server version down outside compose/CI: the
# image reference in the install guide, and the release-tarball URL the Colab
# guide wgets. Both must name the compose pin.
DOC_VERSION_PATTERNS = (
    re.compile(r"qdrant/qdrant:v([0-9][0-9A-Za-z.\-]*)"),
    re.compile(r"qdrant/releases/download/v([0-9][0-9A-Za-z.\-]*)/"),
)


def _image_tag(image: str) -> str:
    """`qdrant/qdrant:v1.18.3` -> `v1.18.3`. Fails loudly on a floating tag:
    `latest` is exactly what the compose comment pins away from, so it must
    not slip through as an unparseable version."""
    _, _, tag = image.partition(":")
    assert tag, f"image {image!r} carries no tag — pin a version, never latest"
    return tag


def _tag_version(tag: str) -> str:
    """`v1.18.3-unprivileged` -> `1.18.3`.

    Qdrant publishes variant tags (`-unprivileged`, `-gpu-nvidia`) that are the
    same server build. The variant is not part of the version, and leaving it
    attached made `parse_version` hand a non-numeric patch to `int()` — an
    opaque ValueError from a test whose whole value is its message.
    """
    version, _, _variant = tag.removeprefix("v").partition("-")
    return version


def _image_tag_version(image: str) -> str:
    return _tag_version(_image_tag(image))


def _compose_server_version() -> str:
    compose = yaml.safe_load(COMPOSE_PATH.read_text())
    return _image_tag_version(compose["services"]["qdrant"]["image"])


def _ci_qdrant_service() -> dict:
    workflow = yaml.safe_load(CI_WORKFLOW_PATH.read_text())
    return workflow["jobs"]["server-compat"]["services"]["qdrant"]


def _env_mapping(raw) -> dict[str, str]:
    """compose and Actions both accept a mapping or a `KEY=VALUE` list.

    A bare `- KEY` is valid compose — it passes the value through from the
    host environment — and carries no value in the file, so it maps to `""`
    and fails the caller's assertion with that test's own message. Splitting
    on `=` unconditionally would raise a bare ValueError from `dict()`
    instead, which is the same "opaque error where a verdict belongs" the
    variant-tag parse was split out to remove.
    """
    if isinstance(raw, list):
        return {key: value for key, _sep, value in (str(i).partition("=") for i in raw)}
    return {str(k): str(v) for k, v in (raw or {}).items()}


def test_pinned_client_and_server_versions_are_compatible():
    server = _compose_server_version()
    client = dist_version("qdrant-client")
    assert is_compatible(client, server), (
        f"qdrant-client {client} (resolved from pyproject.toml) and Qdrant "
        f"server {server} (docker-compose.yml) are not a supported pair: major "
        f"versions must match and the minor difference must not exceed 1. Move "
        f"both pins in one commit — see ADR-62."
    )


def test_pinned_client_and_server_are_the_same_minor():
    """ADR-62's actual rule, which is stricter than `is_compatible`.

    The client tolerates one minor of drift; ADR-62 does not tolerate any. A
    server bumped one minor ahead of the client passes every vendor check and
    is still the state issue #37 grew out of, so this is the assertion that
    makes "both pins move together" mechanical rather than remembered.
    """
    server = _compose_server_version()
    client = dist_version("qdrant-client")
    parsed_client, parsed_server = parse_version(client), parse_version(server)
    assert (parsed_client.major, parsed_client.minor) == (
        parsed_server.major,
        parsed_server.minor,
    ), (
        f"qdrant-client {client} and Qdrant server {server} are on different "
        f"minors. qdrant-client tolerates this, ADR-62 does not: the two pins "
        f"are one decision and land on the same minor. Move docker-compose.yml, "
        f".github/workflows/ci.yml and pyproject.toml in one commit."
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
    ci_image = _ci_qdrant_service()["image"]
    compose_image = yaml.safe_load(COMPOSE_PATH.read_text())["services"]["qdrant"][
        "image"
    ]
    # Full tag, not just the numeric version: `-unprivileged` and `-gpu-nvidia`
    # are the same version and a different image, and CI must run the image
    # `docker compose up -d` starts.
    assert _image_tag(ci_image) == _image_tag(compose_image), (
        f"CI service image {ci_image!r} does not match docker-compose.yml's "
        f"{compose_image!r}. Both must move together or CI stops testing what "
        f"`docker compose up -d` actually starts."
    )


def test_documented_server_versions_match_the_compose_pin():
    """The version is also written down in prose, and prose drifts.

    `installation.md` names the image users run and `m6-agent-connection-guide.md`
    wgets a release tarball by URL. The Colab guide asks the reader in a comment
    to keep its version equal to compose's — which is the mechanism
    `test_ci_service_container_uses_the_compose_pin` exists to replace. A stale
    version in either doc points someone at a server the client does not vouch
    for, or walks them into the cross-1.17 storage migration installation.md
    itself documents.
    """
    pinned = _compose_server_version()
    for path in (INSTALL_DOC_PATH, COLAB_DOC_PATH):
        text = path.read_text()
        found = [m for p in DOC_VERSION_PATTERNS for m in p.findall(text)]
        assert found, (
            f"{path.relative_to(REPO_ROOT)} names no Qdrant server version — if "
            f"the reference moved, move this test with it; a doc that silently "
            f"stops naming the version cannot be checked against the pin."
        )
        for version in found:
            assert _tag_version(version) == pinned, (
                f"{path.relative_to(REPO_ROOT)} documents Qdrant {version}, but "
                f"docker-compose.yml pins {pinned}. Every place this repo writes "
                f"the server version moves in one commit (ADR-62)."
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
    env = _env_mapping(compose["services"]["qdrant"].get("environment"))
    assert env.get("QDRANT__TELEMETRY_DISABLED", "").lower() == "true", (
        "docker-compose.yml must set QDRANT__TELEMETRY_DISABLED=true — the "
        "image defaults it to false and reports usage statistics outbound, "
        "which contradicts ADR-25 (see ADR-63)."
    )


def test_ci_service_container_matches_the_compose_hardening():
    """The CI job runs the same image, so it inherits the same two defaults.

    ADR-63's own rejected alternatives say a prose warning is not a default —
    "the default is the thing that ships". A `server-compat` service container
    with no `env:` ships exactly that default and posts to telemetry.qdrant.io
    on every CI run, which is the behaviour the compose file eleven lines away
    disables. The port binding is the same argument one step out: a bare
    `6333:6333` publishes on all interfaces, which is harmless on an ephemeral
    hosted runner and an unauthenticated vector database on the LAN of a
    self-hosted one. Both are asserted here so the two files cannot tell
    different stories about the same image.
    """
    service = _ci_qdrant_service()
    env = _env_mapping(service.get("env"))
    assert env.get("QDRANT__TELEMETRY_DISABLED", "").lower() == "true", (
        "the CI `server-compat` service must set QDRANT__TELEMETRY_DISABLED: "
        "'true' under `env:`, matching docker-compose.yml — otherwise every CI "
        "run starts the image's phone-home default (ADR-63)."
    )
    published = service.get("ports") or []
    assert published, "the CI service publishes no ports — the job cannot reach it"
    for mapping in published:
        assert str(mapping).startswith("127.0.0.1:"), (
            f"CI publishes {mapping!r} — bind 127.0.0.1 only, matching compose "
            f"and CLAUDE.md rule 2 (a bare mapping listens on all interfaces, "
            f"which matters on a self-hosted runner)"
        )
