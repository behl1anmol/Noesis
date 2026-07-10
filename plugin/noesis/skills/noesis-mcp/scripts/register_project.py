#!/usr/bin/env python3
"""Register a repository with a running Noesis service and start its first index.

Registration is a REST-only operator step — there is no MCP tool for it (the MCP
surface exposes retrieval + ``reindex`` only). This helper POSTs to
``/projects`` and, with ``--wait``, polls ``/projects/{id}/status`` until the
index run finishes.

Standard library only: runs under any ``python3`` without the Noesis venv.

Examples
--------
    python register_project.py /abs/path/to/repo
    python register_project.py /abs/path/to/repo --wait
    python register_project.py /abs/path/to/repo --base-url http://127.0.0.1:9000

Exit codes: 0 success (and index done, if --wait); 2 usage/argument error;
3 registration rejected by the service; 4 could not reach the service;
5 index run failed (only reachable with --wait).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_BASE_URL = "http://127.0.0.1:8000"


def resolve_base_url(cli_value: str | None) -> str:
    """Precedence: --base-url > NOESIS_BASE_URL > CLAUDE_PLUGIN_OPTION_BASE_URL
    > default. Trailing slash stripped so ``base + "/path"`` is well-formed."""
    raw = (
        cli_value
        or os.environ.get("NOESIS_BASE_URL")
        or os.environ.get("CLAUDE_PLUGIN_OPTION_BASE_URL")
        or DEFAULT_BASE_URL
    )
    return raw.rstrip("/")


def _request(
    method: str, url: str, body: dict | None, timeout: float
) -> tuple[int, dict]:
    """Return (status_code, parsed_json). Raises URLError on transport failure.
    HTTP error responses (4xx/5xx) are returned, not raised, so callers can read
    the service's error detail."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("content-type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        payload = exc.read()
        try:
            parsed = json.loads(payload or b"{}")
        except json.JSONDecodeError:
            parsed = {"detail": payload.decode(errors="replace")}
        return exc.code, parsed


def register(base_url: str, root_path: str, timeout: float) -> dict:
    status, payload = _request(
        "POST", f"{base_url}/projects", {"root_path": root_path}, timeout
    )
    if status == 202:
        return payload
    detail = payload.get("detail", payload)
    if status == 409:
        print(
            f"error: registration conflict (409): {detail}\n"
            "The path is already registered under a different embedding model. "
            "Switching models needs a full re-index — see the skill's "
            "transports.md (mixed-model guard).",
            file=sys.stderr,
        )
    else:  # 400 and anything else the service rejected
        print(
            f"error: service rejected registration ({status}): {detail}",
            file=sys.stderr,
        )
    sys.exit(3)


def wait_for_index(
    base_url: str, project_id: str, timeout: float, interval: float
) -> str:
    """Poll status until it leaves ``running``/``never_indexed``. Returns the
    terminal status string."""
    url = f"{base_url}/projects/{project_id}/status"
    while True:
        _, payload = _request("GET", url, None, timeout)
        status = payload.get("status", "unknown")
        if status in ("done", "failed"):
            if status == "failed":
                print(f"index failed: {payload.get('error')}", file=sys.stderr)
            else:
                print(
                    f"index done: {payload.get('files_total')} files, "
                    f"{payload.get('chunks_written')} chunks."
                )
            return status
        print(f"  indexing… status={status}", file=sys.stderr)
        time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root_path", help="Absolute path to the repository to index.")
    parser.add_argument(
        "--base-url",
        default=None,
        help=f"Noesis base URL (default {DEFAULT_BASE_URL}).",
    )
    parser.add_argument(
        "--wait", action="store_true", help="Poll until the first index finishes."
    )
    parser.add_argument(
        "--timeout", type=float, default=15.0, help="Per-request timeout in seconds."
    )
    parser.add_argument(
        "--poll-interval", type=float, default=5.0, help="Seconds between status polls."
    )
    args = parser.parse_args()

    root = Path(args.root_path).expanduser()
    if not root.is_dir():
        print(f"error: not a directory: {root}", file=sys.stderr)
        sys.exit(2)
    base_url = resolve_base_url(args.base_url)

    try:
        result = register(base_url, str(root.resolve()), args.timeout)
    except urllib.error.URLError as exc:
        print(
            f"error: cannot reach Noesis at {base_url} ({exc.reason}).\n"
            "Is the service running? `uv run uvicorn noesis.app:app "
            "--host 127.0.0.1 --port 8000` (and Qdrant: `docker compose up -d`).",
            file=sys.stderr,
        )
        sys.exit(4)

    project_id = result["project_id"]
    print(f"registered: project_id={project_id} run_id={result.get('run_id')}")

    if args.wait:
        status = wait_for_index(base_url, project_id, args.timeout, args.poll_interval)
        if status == "failed":
            sys.exit(5)
    else:
        print("index started in the background; poll get_index_status (or add --wait).")


if __name__ == "__main__":
    main()
