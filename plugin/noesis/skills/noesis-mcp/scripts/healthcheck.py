#!/usr/bin/env python3
"""Diagnose connectivity to a running Noesis service.

Probes ``/healthz`` and ``/projects`` and prints a short report: whether the
service is reachable, and the registered projects with their index status. Use
this when the ``noesis:`` MCP tools are missing or misbehaving — it separates
"service is down" from "query is wrong".

Standard library only: runs under any ``python3`` without the Noesis venv.

Exit codes: 0 healthy; 1 unhealthy (service unreachable or /healthz not ok).
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
    parser.add_argument("--base-url", default=None, help=f"Noesis base URL (default {DEFAULT_BASE_URL}).")
    parser.add_argument("--timeout", type=float, default=10.0, help="Per-request timeout in seconds.")
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

    # 2. /projects
    try:
        _, projects = _get(f"{base_url}/projects", args.timeout)
    except urllib.error.URLError as exc:
        print(f"  [WARN] could not list projects: {exc.reason}")
        sys.exit(0)

    if not projects:
        print("  [WARN] no projects registered.")
        print("         Register one: scripts/register_project.py <abs-repo-path> --wait")
        sys.exit(0)

    print(f"  [ OK ] {len(projects)} project(s) registered:")
    for p in projects:
        print(f"         - id={p.get('id')}  root={p.get('root_path')}  model={p.get('embedding_model')}")


if __name__ == "__main__":
    main()
