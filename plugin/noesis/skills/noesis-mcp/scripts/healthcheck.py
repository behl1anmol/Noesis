#!/usr/bin/env python3
"""Diagnose connectivity to a running Noesis service.

Probes ``/healthz`` and ``/projects`` and prints a short report: whether the
service is reachable, and the registered projects with their index status. Use
this when the ``noesis:`` MCP tools are missing or misbehaving — it separates
"service is down" from "query is wrong".

Standard library only: runs under any ``python3`` without the Noesis venv.

Exit codes: 0 healthy; 1 unhealthy (service unreachable, /healthz not ok, or a
model whose assets are missing from the local cache — ADR-78, issue #47
finding 4: green here previously meant nothing about whether the next search
pays a multi-minute silent download). The reranker is checked the same way
when it is switched on (issue #52); when it is off — the shipped default —
its fields read "disabled" and nothing is reported about it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

DEFAULT_BASE_URL = "http://127.0.0.1:8000"


def resolve_base_url(cli_value: str | None) -> str:
    raw = (
        cli_value
        or os.environ.get("NOESIS_BASE_URL")
        or os.environ.get("CLAUDE_PLUGIN_OPTION_BASE_URL")
        or DEFAULT_BASE_URL
    )
    return raw.rstrip("/")


def _get(url: str, timeout: float):
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read() or b"{}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--base-url",
        default=None,
        help=f"Noesis base URL (default {DEFAULT_BASE_URL}).",
    )
    parser.add_argument(
        "--timeout", type=float, default=10.0, help="Per-request timeout in seconds."
    )
    args = parser.parse_args()

    base_url = resolve_base_url(args.base_url)
    print(f"Noesis health check → {base_url}")

    # 1. /healthz
    try:
        status, payload = _get(f"{base_url}/healthz", args.timeout)
    except urllib.error.URLError as exc:
        print(f"  [FAIL] service unreachable: {exc.reason}")
        print(
            "\nStart it:\n"
            "  docker compose up -d                                    # Qdrant\n"
            "  uv run uvicorn noesis.app:app --host 127.0.0.1 --port 8000\n"
            f"If it runs on another port, pass --base-url or set the plugin's\n"
            f"base_url config to match. MCP endpoint = base_url + /mcp/.",
        )
        sys.exit(1)

    if status != 200 or payload.get("status") != "ok":
        print(f"  [FAIL] /healthz returned {status} {payload}")
        sys.exit(1)
    print("  [ OK ] service is up (/healthz ok)")
    print(f"         MCP endpoint: {base_url}/mcp/")

    # Asset problems are COLLECTED, not exited on. This script is a
    # diagnostic: an operator running it wants the whole picture, and exiting
    # at the first missing model dropped the project listing below — the other
    # half of what they ran it for — and hid a second missing model behind the
    # first (issue #52 review round 8). The exit code is decided at the end.
    problems = 0

    def report(
        label: str, assets, ready, size: str, next_call: str, extra: str = ""
    ) -> int:
        """One model's asset verdict. Only "missing" and "ready" are reported
        on: "disabled" (the reranker kill switch, off by default), "unknown"
        (a bare app with no lifespan wired) and an absent field (an older
        service, or a proxy that strips it) are all things an operator cannot
        act on, and warning about a feature nobody turned on is a false
        alarm."""
        if assets == "missing":
            # `ready is True` means the running process already holds the
            # model, so nothing is about to block — but the weights really are
            # gone from the cache and the next RESTART re-downloads them, so
            # this is still a failure, just not the one the other branch
            # describes.
            consequence = (
                f"the model is already loaded in the running service, but the "
                f"cache is empty — a restart re-downloads {size}"
                if ready is True
                else f"the next {next_call} will block for minutes "
                f"downloading {size}"
            )
            print(
                f"  [FAIL] {label} model assets not found in the local cache — "
                f"{consequence}.\n"
                f"         Fetch them now: uv run python -m noesis.prefetch{extra}"
            )
            return 1
        if assets == "ready":
            print(f"  [ OK ] {label} model assets are cached")
        return 0

    problems += report(
        "embedding",
        payload.get("assets"),
        payload.get("embedder_ready"),
        "~550 MB",
        "search_code call",
    )
    problems += report(
        "reranker",
        payload.get("reranker_assets"),
        payload.get("reranker_ready"),
        "~2.3 GB",
        "reranked search",
        extra="  (or turn reranking off in config.toml)",
    )

    # 2. /projects
    try:
        _, projects = _get(f"{base_url}/projects", args.timeout)
    except urllib.error.URLError as exc:
        print(f"  [WARN] could not list projects: {exc.reason}")
        sys.exit(1 if problems else 0)

    if not projects:
        print("  [WARN] no projects registered.")
        print(
            "         Register one: scripts/register_project.py <abs-repo-path> --wait"
        )
        sys.exit(1 if problems else 0)

    print(f"  [ OK ] {len(projects)} project(s) registered:")
    for p in projects:
        print(
            f"         - id={p.get('id')}  root={p.get('root_path')}  model={p.get('embedding_model')}"
        )
    # Decided here, after everything has been said: 1 if any model's assets
    # are missing (ADR-78's fail-loud posture), 0 otherwise.
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
