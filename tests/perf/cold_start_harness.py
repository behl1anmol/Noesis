"""Cold-start harness — reproduces the first-use model download stall.

Noesis loads its embedding weights lazily: ``LocalSTEmbedder`` starts its
worker thread on the first ``embed_*`` call and the model is constructed
inside that worker on its first job (``core/embedder.py``). On a machine
where ``python -m noesis.prefetch`` was never run, that first job is also
the first HTTP fetch of the weights — so the download lands *inside* a
user-facing MCP tool call, with no progress output on the tool channel.

This harness makes that reproducible. It runs one **workload** (the real
first-use sequence: register → ``reindex`` → ``search_code`` → drain →
``search_code``) inside a **scenario** (a controlled asset-cache state),
in a fresh subprocess, and reports per phase:

* wall seconds, and
* bytes that appeared in each asset cache during that phase.

Bytes are the machine-independent invariant; seconds are not — they scale
with the operator's link and CPU. Read the byte columns to decide *what*
is being fetched and *which call pays for it*; read the seconds only
against another scenario measured on the same machine in the same run.

Scenarios
---------
``cold``        every asset cache empty — a fresh install that skipped prefetch.
``warm``        caches as the preceding scenario left them — the control.
``prefetched``  caches empty, then ``python -m noesis.prefetch`` runs and is
                timed as its own phase, then the workload — the remedy.

Running ``cold,warm`` (the default) in one invocation is the experiment:
identical code, identical query, one variable (cache state).

Usage
-----
    uv run python tests/perf/cold_start_harness.py
    uv run python tests/perf/cold_start_harness.py --scenarios cold,prefetched
    uv run python tests/perf/cold_start_harness.py --sequence mcp
    uv run python tests/perf/cold_start_harness.py --qdrant-url http://127.0.0.1:6333

Every asset cache is redirected into ``--workspace`` (default
``dev/perf/cold-start``), so a run never touches the operator's real
``~/.cache`` and never has to delete anything outside its own workspace.
The Python environment itself is NOT rebuilt — this measures asset fetch,
not ``uv sync``.

By default the vector store is qdrant-client's embedded mode, the same
choice every store-touching test makes, so the harness runs with no
server. Pass ``--qdrant-url`` to measure against a real Qdrant instead;
that path builds the context through ``noesis.runtime.build_runtime_context``
exactly as both transports do.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

# Bump when a change makes new numbers incomparable to stored ones.
HARNESS_VERSION = 1

REPO_ROOT = Path(__file__).resolve().parents[2]

# The asset caches this harness controls, and the env var that steers each.
#
# HF_HOME              CodeRankEmbed / bge-reranker weights (huggingface_hub).
# FASTEMBED_CACHE_PATH Qdrant/bm25 tokenizer assets (prefetch.FASTEMBED_CACHE_ENV).
# XDG_CACHE_HOME       tree-sitter grammars — the language pack has no env var
#                      of its own and falls back to the system cache dir, so
#                      redirecting XDG_CACHE_HOME is the only handle on it.
#                      It is also what noesis' own default fastembed path is
#                      anchored to, which keeps the two consistent.
CACHE_SPECS: tuple[tuple[str, str], ...] = (
    ("hf", "HF_HOME"),
    ("fastembed", "FASTEMBED_CACHE_PATH"),
    ("xdg", "XDG_CACHE_HOME"),
)

SCENARIOS = ("cold", "warm", "prefetched")
SEQUENCES = ("query-first", "mcp")

DEFAULT_QUERY = "how are dense and sparse search results fused"


# --------------------------------------------------------------------------
# byte accounting
# --------------------------------------------------------------------------


def dir_bytes(path: Path) -> int:
    """Bytes on disk under *path*, symlinks excluded.

    Excluding symlinks is not a detail: huggingface_hub stores each file
    once under ``blobs/`` and symlinks it into ``snapshots/``, so counting
    followed symlinks reports every weight file twice. An early draft of
    this harness did exactly that and claimed a 1.10 GB download for a
    548 MB model. ``tests/perf/test_cold_start_harness.py`` pins the rule.
    """
    if not path.exists():
        return 0
    total = 0
    for entry in path.rglob("*"):
        if entry.is_symlink() or not entry.is_file():
            continue
        try:
            total += entry.stat().st_size
        except OSError:
            # A cache file can vanish under us (concurrent eviction); a
            # missing file contributes nothing rather than killing the run.
            continue
    return total


class Meter:
    """Times phases and attributes cache growth to the phase that caused it."""

    def __init__(self, caches: dict[str, Path]) -> None:
        self._caches = caches
        self.phases: list[dict] = []

    def _sizes(self) -> dict[str, int]:
        return {name: dir_bytes(path) for name, path in self._caches.items()}

    @contextmanager
    def phase(self, name: str, tolerate_errors: bool = False):
        """Time *name* and attribute cache growth to it.

        With *tolerate_errors*, an exception raised in the body is recorded
        on the phase and swallowed, so one broken call is reported as data
        instead of aborting a run that has already paid for a download.
        """
        before = self._sizes()
        started = time.perf_counter()
        record: dict = {"phase": name}
        self.phases.append(record)
        try:
            yield record
        except Exception as exc:  # noqa: BLE001 — recorded, see docstring
            record["error"] = f"{type(exc).__name__}: {exc}"
            if not tolerate_errors:
                raise
        finally:
            record["seconds"] = round(time.perf_counter() - started, 3)
            after = self._sizes()
            record["fetched_bytes"] = {
                key: after[key] - before[key]
                for key in after
                if after[key] != before[key]
            }
            record["fetched_bytes_total"] = sum(record["fetched_bytes"].values())


# --------------------------------------------------------------------------
# worker — runs one workload in this process; always spawned as a subprocess
# --------------------------------------------------------------------------


def _worker(config_path: Path) -> int:
    config = json.loads(config_path.read_text())
    caches = {name: Path(p) for name, p in config["caches"].items()}
    meter = Meter(caches)
    result: dict = {
        "scenario": config["scenario"],
        "label": config["label"],
        "sequence": config["sequence"],
    }

    # Import cost is a phase of its own: torch + sentence_transformers is
    # seconds of CPU before any noesis code runs, and it is NOT fixed by
    # prefetching, so keeping it separate stops it inflating the download
    # number the issue is about.
    with meter.phase("import"):
        import asyncio

        from qdrant_client import QdrantClient

        from noesis.core import jobs, retriever, state
        from noesis.core.config import EmbedderSettings, QdrantSettings, Settings
        from noesis.core.embedder import LocalSTEmbedder
        from noesis.core.vectorstore import VectorStore
        from noesis.runtime import (
            AppContext,
            build_runtime_context,
            close_runtime_context,
        )

    cfg = Settings(
        db_path=Path(config["db_path"]),
        embedder=EmbedderSettings(
            model=config["model"], device=config["device"] or None
        ),
        qdrant=QdrantSettings(
            url=config["qdrant_url"] or QdrantSettings.url,
            collection="noesis_perf_cold_start",
        ),
    )

    client = None

    async def run() -> None:
        nonlocal client
        with meter.phase("startup"):
            if config["qdrant_url"]:
                # Full-fidelity path: identical wiring to both transports.
                ctx = await build_runtime_context(cfg)
            else:
                # Embedded store — the convention every store-touching test
                # follows, so the harness runs with no server. Only the
                # Qdrant transport differs; the embedder and the client-side
                # fastembed BM25 encode — the two things that fetch assets —
                # are constructed exactly as build_runtime_context does.
                conn = state.connect(cfg.db_path)
                state.init_db(conn)
                embedder = LocalSTEmbedder(
                    model_id=cfg.embedder.model,
                    dim=cfg.embedder.dim,
                    batch_size=cfg.embedder.batch_size,
                    device=cfg.embedder.device,
                )
                client = QdrantClient(":memory:")
                store = VectorStore(client, collection_name=cfg.qdrant.collection)
                store.ensure_collection(embedder)
                ctx = AppContext(conn=conn, store=store, embedder=embedder)

        corpus = config["corpus"]
        query = config["query"]

        async def search(label: str, tolerate_errors: bool = False) -> None:
            with meter.phase(label, tolerate_errors=tolerate_errors) as record:
                hits = await retriever.search_code(
                    ctx.store, ctx.embedder, query, project_id, top_k=5
                )
                record["hits"] = len(hits["hits"])

        if config["sequence"] == "query-first":
            # Isolation mode (the default): nothing else is queued on the
            # embedder worker, so `first_search` is the model load and its
            # download, alone. The project has no chunks yet, so it returns
            # zero hits by construction — the latency is the measurement,
            # not the hits.
            with meter.phase("register"):
                project_id = state.register_project(
                    ctx.conn, corpus, ctx.embedder.model_id
                )
            await search("first_search")
            with meter.phase("reindex_call"):
                accepted = jobs.launch_index_run(ctx, corpus)
            with meter.phase("index_drain"):
                await ctx.jobs[accepted["run_id"]]
            await search("second_search")
        else:
            # Realistic MCP sequence. `reindex` returns a run_id immediately
            # (mcp/server.py) and indexing proceeds on a background task, so
            # the agent's very next `search_code` is what blocks on the
            # weight download — the reported symptom.
            #
            # This sequence overlaps a hybrid search with an index run on
            # one QdrantClient, which qdrant-client 1.18 does not survive.
            # Two distinct failures were observed and neither is the defect
            # being measured, so both searches record the error rather than
            # abort a run that has already paid for the download:
            #
            #   RuntimeError: dictionary changed size during iteration
            #     — qdrant_client/embed/model_embedder.py, whose
            #       _batch_accumulator has no lock. Transport-agnostic:
            #       client-side inference runs before dispatch, so a real
            #       server does not protect against it.
            #   IndexError: index N is out of bounds
            #     — qdrant_client/local/local_collection.py reading the
            #       deleted mask while an upsert grows the points. Embedded
            #       store only, and the damage PERSISTS: the following
            #       non-concurrent search fails too.
            #
            # Because of the second one, `mcp` numbers from the embedded
            # store are not trustworthy past the first search. Use
            # --qdrant-url for this sequence.
            with meter.phase("reindex_call"):
                accepted = jobs.launch_index_run(ctx, corpus)
            project_id = accepted["project_id"]
            await search("first_search", tolerate_errors=True)
            with meter.phase("index_drain"):
                await ctx.jobs[accepted["run_id"]]
            await search("second_search", tolerate_errors=True)

        try:
            result["indexed_chunks"] = ctx.store.count_project_points(project_id)
        except Exception as exc:  # noqa: BLE001 — see the mcp note above
            result["indexed_chunks"] = f"unavailable: {type(exc).__name__}"
        result["resolved_device"] = getattr(ctx.embedder, "resolved_device", None)
        if config["qdrant_url"]:
            await close_runtime_context(ctx)
        else:
            ctx.embedder.close()
            ctx.conn.close()
            if client is not None:
                client.close()

    asyncio.run(run())
    result["phases"] = meter.phases
    result["cache_bytes_final"] = {
        name: dir_bytes(path) for name, path in caches.items()
    }
    Path(config["result_path"]).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return 0


# --------------------------------------------------------------------------
# orchestrator
# --------------------------------------------------------------------------


def scenario_complaint(names: list[str]) -> str | None:
    """Why *names* is not a runnable scenario list, or None if it is.

    Kept as a pure function rather than inline argparse checks so it can be
    tested without ``main`` reaching the part that spawns workers and
    downloads weights: an earlier version of this rule was 'covered' by a
    test that called ``main`` and passed even with the rule deleted,
    because 138 seconds later something else raised SystemExit.
    """
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        return f"unknown scenario(s) {unknown}; choose from {list(SCENARIOS)}"
    if not names:
        return f"no scenarios given; choose from {list(SCENARIOS)}"
    if names[0] == "warm":
        return (
            "'warm' means 'the caches the previous scenario left behind' — "
            "it cannot be first; put 'cold' or 'prefetched' before it"
        )
    return None


def _wipe(path: Path, workspace: Path) -> None:
    """Delete *path*, refusing anything outside *workspace*.

    The harness only ever removes directories it created itself; the guard
    is here so a mistyped ``--workspace`` cannot turn this into a command
    that deletes a real model cache.
    """
    workspace = workspace.resolve()
    resolved = path.resolve()
    if resolved == workspace or workspace not in resolved.parents:
        raise SystemExit(f"refusing to delete {resolved} — outside {workspace}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _env_for(caches: dict[str, Path]) -> dict[str, str]:
    env = dict(os.environ)
    for name, var in CACHE_SPECS:
        env[var] = str(caches[name])
    # Never let a stray operator config.toml change what is measured.
    env.pop("NOESIS_CONFIG", None)
    return env


def _provenance(args: argparse.Namespace) -> dict:
    def git(*cmd: str) -> str:
        return subprocess.run(
            ["git", *cmd], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        import torch

        torch_version = torch.__version__
        cuda = torch.cuda.is_available()
    except Exception:  # noqa: BLE001 — provenance must never fail the run
        torch_version, cuda = None, None
    return {
        "harness_version": HARNESS_VERSION,
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch_version,
        "cuda_available": cuda,
        "model": args.model,
        "corpus": args.corpus,
        "query": args.query,
        "sequence": args.sequence,
        "store": args.qdrant_url or "embedded",
    }


def run_scenario(
    name: str, label: str, args: argparse.Namespace, workspace: Path
) -> dict:
    caches = {key: workspace / "cache" / key for key, _ in CACHE_SPECS}
    run_dir = workspace / "run" / label
    prefetch_phase: dict | None = None

    if name in ("cold", "prefetched"):
        for path in caches.values():
            _wipe(path, workspace)
    _wipe(run_dir, workspace)
    for path in caches.values():
        path.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    env = _env_for(caches)

    if name == "prefetched":
        before = {key: dir_bytes(path) for key, path in caches.items()}
        started = time.perf_counter()
        cmd = [sys.executable, "-m", "noesis.prefetch", "--skip-reranker"]
        proc = subprocess.run(
            cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True
        )
        elapsed = round(time.perf_counter() - started, 3)
        if proc.returncode != 0:
            raise SystemExit(
                f"prefetch failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
            )
        after = {key: dir_bytes(path) for key, path in caches.items()}
        fetched = {k: after[k] - before[k] for k in after if after[k] != before[k]}
        prefetch_phase = {
            "phase": "prefetch",
            "seconds": elapsed,
            "fetched_bytes": fetched,
            "fetched_bytes_total": sum(fetched.values()),
        }

    config = {
        "scenario": name,
        "label": label,
        "sequence": args.sequence,
        "caches": {key: str(path) for key, path in caches.items()},
        "db_path": str(run_dir / "state.sqlite"),
        "qdrant_url": args.qdrant_url,
        "model": args.model,
        "device": args.device,
        "corpus": str((REPO_ROOT / args.corpus).resolve()),
        "query": args.query,
        "result_path": str(run_dir / "result.json"),
    }
    config_path = run_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    print(f"[{label}] running workload ({args.sequence})...", flush=True)
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker", str(config_path)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    log_path = run_dir / "worker.log"
    log_path.write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        raise SystemExit(
            f"[{label}] worker failed ({proc.returncode}); see {log_path}\n"
            f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        )
    result = json.loads(Path(config["result_path"]).read_text())
    result["label"] = label
    if prefetch_phase is not None:
        result["phases"] = [prefetch_phase, *result["phases"]]
    result["worker_log"] = str(log_path)
    return result


def _mb(value: int) -> str:
    return f"{value / 1_000_000:8.1f}" if value else "       ·"


def format_table(results: list[dict]) -> str:
    lines = []
    for result in results:
        lines.append("")
        lines.append(
            f"### scenario: {result['label']}  (sequence: {result['sequence']})"
        )
        lines.append("")
        lines.append("| phase | seconds | fetched MB | detail |")
        lines.append("| --- | ---: | ---: | --- |")
        for phase in result["phases"]:
            detail = ", ".join(
                f"{k}+{v / 1_000_000:.1f}MB" for k, v in phase["fetched_bytes"].items()
            )
            if "hits" in phase:
                detail = (detail + " " if detail else "") + f"({phase['hits']} hits)"
            lines.append(
                f"| {phase['phase']} | {phase['seconds']:8.2f} | "
                f"{_mb(phase['fetched_bytes_total'])} | {detail} |"
            )
        lines.append(
            f"| **total** | **{sum(p['seconds'] for p in result['phases']):8.2f}** | "
            f"**{_mb(sum(p['fetched_bytes_total'] for p in result['phases']))}** | "
            f"{result.get('indexed_chunks', '?')} chunks indexed, "
            f"device={result.get('resolved_device')} |"
        )
    return "\n".join(lines)


def format_verdict(results: list[dict]) -> str:
    """The comparison the harness exists to make: same call, cold vs warm.

    Two claims are NOT equally strong and are not presented as if they were.
    The byte delta is exact — a cache either grew or it did not. The seconds
    delta is a single sample: on a quiet 4-core box three identical warm
    loads landed in 10.72-11.33s, but on the same box under contention two
    landed 6.6s apart. So it is reported against the spread of the warm
    repeats and is refused outright when it falls inside that spread. Run `--scenarios cold,warm,warm,warm`
    to give this section a noise floor to judge against.
    """
    cold = next((r for r in results if r["scenario"] == "cold"), None)
    warms = [r for r in results if r["scenario"] == "warm"]
    if cold is None or not warms:
        return (
            "\n### verdict\n\n"
            "No cold/warm pair in this run — the penalty is a *difference*, "
            "so run at least `--scenarios cold,warm` to state one."
        )

    def phase(result: dict, name: str) -> dict | None:
        return next((p for p in result["phases"] if p["phase"] == name), None)

    lines = ["", "### verdict", ""]
    if len(warms) < 2:
        lines.append(
            "> Only one warm run: there is no noise floor, so no seconds "
            "delta below is claimed as signal. Re-run with "
            "`--scenarios cold,warm,warm,warm`.\n"
        )
    else:
        lines.append(f"> Noise floor from {len(warms)} warm repeats.\n")

    for name in ("import", "startup", "first_search", "reindex_call", "index_drain", "second_search"):
        cold_phase = phase(cold, name)
        warm_phases = [p for p in (phase(w, name) for w in warms) if p is not None]
        if cold_phase is None or not warm_phases:
            continue
        warm_seconds = sorted(p["seconds"] for p in warm_phases)
        spread = warm_seconds[-1] - warm_seconds[0]
        # Against the SLOWEST warm sample, not the fastest: measuring from
        # the fastest silently adds the whole spread to the delta and turns
        # ordinary variance into a finding.
        delta = cold_phase["seconds"] - warm_seconds[-1]
        fetched = cold_phase["fetched_bytes_total"]

        if len(warm_seconds) < 2:
            time_claim = f"warm {warm_seconds[0]:.2f}s (1 sample, no floor)"
        elif delta <= spread:
            time_claim = (
                f"warm {warm_seconds[0]:.2f}–{warm_seconds[-1]:.2f}s; "
                f"the {delta:+.2f}s is inside the warm spread, not separable"
            )
        else:
            time_claim = (
                f"warm {warm_seconds[0]:.2f}–{warm_seconds[-1]:.2f}s; "
                f"{delta:+.2f}s beyond the spread"
            )
        byte_claim = (
            f"**{fetched / 1_000_000:.0f} MB fetched**" if fetched else "0 MB fetched"
        )
        lines.append(f"- `{name}`: cold {cold_phase['seconds']:.2f}s, {byte_claim} — {time_claim}")

    total_cold = sum(p["fetched_bytes_total"] for p in cold["phases"])
    paying = [
        p["phase"] for p in cold["phases"] if p["fetched_bytes_total"] > 0
    ]
    lines.append("")
    lines.append(
        f"**{total_cold / 1_000_000:.0f} MB** is fetched on a cold install, "
        f"inside these phases: {', '.join(paying) or 'none'}. "
        f"Every warm repeat fetched 0 MB."
    )
    return "\n".join(lines)


def unique_labels(names: list[str]) -> list[str]:
    """Label repeated scenario names so each run gets its own directory and
    its own row: ['cold', 'warm', 'warm'] -> ['cold', 'warm', 'warm-2']."""
    seen: dict[str, int] = {}
    labels = []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        labels.append(name if seen[name] == 1 else f"{name}-{seen[name]}")
    return labels


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--scenarios",
        default="cold,warm",
        help=f"comma-separated, in order, from {SCENARIOS} (default: cold,warm)",
    )
    parser.add_argument(
        "--sequence",
        default="query-first",
        choices=SEQUENCES,
        help="query-first: search before indexing, so the download is "
        "isolated on one call (default). mcp: reindex then search, the real "
        "agent path, which also overlaps a search with an index run",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=REPO_ROOT / "dev" / "perf" / "cold-start",
        help="where caches, state and reports live (default: dev/perf/cold-start)",
    )
    parser.add_argument(
        "--corpus",
        default="src/noesis/mcp",
        help="repo-relative path to index (default: src/noesis/mcp). Small on "
        "purpose: indexing is not what this harness measures, and embedding "
        "the whole tree on CPU costs ~20 min per scenario. Pass "
        "'src/noesis' for a fuller index phase",
    )
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--model", default="nomic-ai/CodeRankEmbed")
    parser.add_argument(
        "--device", default="", help="force a device (cpu/cuda/mps); default auto"
    )
    parser.add_argument(
        "--qdrant-url",
        default="",
        help="measure against a live Qdrant (full build_runtime_context path) "
        "instead of the embedded store",
    )
    args = parser.parse_args(argv)

    if args.worker is not None:
        return _worker(args.worker)

    names = [n.strip() for n in args.scenarios.split(",") if n.strip()]
    complaint = scenario_complaint(names)
    if complaint is not None:
        parser.error(complaint)
    corpus = (REPO_ROOT / args.corpus).resolve()
    if not corpus.is_dir():
        parser.error(f"--corpus is not a directory: {corpus}")

    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    # Marker so a human reading the tree knows what may be deleted.
    (workspace / "README.txt").write_text(
        "Scratch workspace for tests/perf/cold_start_harness.py.\n"
        "Everything here is regenerated on each run and safe to delete.\n"
    )

    provenance = _provenance(args)
    results = [
        run_scenario(name, label, args, workspace)
        for name, label in zip(names, unique_labels(names))
    ]

    report = {"provenance": provenance, "scenarios": results}
    json_path = workspace / "report_latest.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    body = "\n".join(
        [
            "# Noesis cold-start report",
            "",
            "Seconds are machine- and link-specific; bytes are not. Compare",
            "scenarios within one run, never across machines.",
            "",
            "```json",
            json.dumps(provenance, indent=2, sort_keys=True),
            "```",
            format_table(results),
            format_verdict(results),
            "",
        ]
    )
    md_path = workspace / "report_latest.md"
    md_path.write_text(body)
    print(body)
    print(f"\nwrote {json_path}\nwrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
