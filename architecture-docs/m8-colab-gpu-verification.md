# M8 — Colab GPU verification guide (dashboard device setting + real-model reindex)

Purpose: the M8 test suite proves the dashboard, watcher, scoped runs and the
device-setting plumbing on fakes (offline, CPU). What it cannot prove on a
CPU-only dev box is that the **GPU/CPU device toggle actually moves the real
models** and that a watcher-triggered scoped reindex embeds on the selected
device. This is the same class of check that produced `m4-reranker-benchmarks.md`.
Run it once on a Colab T4; paste the printed block back into the PR.

## Setup (one cell)

```bash
# Runtime → Change runtime type → T4 GPU, then:
!git clone https://github.com/behl1anmol/noesis && cd noesis && git checkout claude/m8-dashboard-watcher
!pip install uv -q
!cd noesis && uv sync --frozen
# Qdrant server (dashboard/scoped-run test uses in-process :memory:, no Docker needed)
```

## Verification script (one cell)

```python
%cd /content/noesis
import asyncio, time, pathlib, tempfile
from qdrant_client import QdrantClient

from noesis.app import AppContext
from noesis.core import dashboard, indexer, state
from noesis.core.compute import available_devices
from noesis.core.embedder import LocalSTEmbedder
from noesis.core.vectorstore import VectorStore

print("available devices:", available_devices())          # expect ('cuda', 'cpu')

tmp = pathlib.Path(tempfile.mkdtemp())
repo = tmp / "repo"; repo.mkdir()
for i in range(20):
    (repo / f"mod_{i}.py").write_text(f"def fn_{i}(x):\n    return x * {i}\n")

conn = state.connect(tmp / "state.sqlite"); state.init_db(conn)
embedder = LocalSTEmbedder()                               # real CodeRankEmbed
store = VectorStore(QdrantClient(":memory:")); store.ensure_collection(embedder)
ctx = AppContext(conn=conn, store=store, embedder=embedder)

async def run():
    # 1. auto device on a T4 must resolve to cuda
    t0 = time.perf_counter()
    r1 = await indexer.index_project(conn, store, embedder, str(repo))
    print(f"full index: {r1.chunks_written} chunks in {time.perf_counter()-t0:.1f}s "
          f"on device={embedder.resolved_device}")          # expect cuda

    # 2. dashboard device setting → cpu: generation-bump reload must move the model
    dashboard.set_compute_device(ctx, "cpu")
    (repo / "mod_0.py").write_text("def fn_0(x):\n    return -x\n")
    t0 = time.perf_counter()
    pid, rid = indexer.prepare_run(conn, embedder, str(repo))
    r2 = await indexer.execute_run(conn, store, embedder, str(repo), pid, rid,
                                   paths=["mod_0.py"])      # watcher-style scoped run
    print(f"scoped reindex (1 file): {time.perf_counter()-t0:.1f}s "
          f"on device={embedder.resolved_device}")          # expect cpu
    assert r2.files_indexed == 1

    # 3. back to cuda via the setting
    dashboard.set_compute_device(ctx, "cuda")
    (repo / "mod_1.py").write_text("def fn_1(x):\n    return -2*x\n")
    pid, rid = indexer.prepare_run(conn, embedder, str(repo))
    t0 = time.perf_counter()
    await indexer.execute_run(conn, store, embedder, str(repo), pid, rid,
                              paths=["mod_1.py"])
    print(f"scoped reindex (1 file): {time.perf_counter()-t0:.1f}s "
          f"on device={embedder.resolved_device}")          # expect cuda

asyncio.run(run())
```

## Pass criteria

| Check | Expectation |
|---|---|
| `available_devices()` | contains `cuda` on the T4 runtime |
| Full index device | `resolved_device == "cuda"` with no config/setting (auto) |
| After `set_compute_device(ctx, "cpu")` | next run reloads and reports `cpu` (reload happens on the worker's next job — first scoped run pays the reload) |
| After `set_compute_device(ctx, "cuda")` | next run reports `cuda` again |
| Scoped run | `files_indexed == 1` — one file hashed/embedded, not 20 |

## Results (paste from Colab)

_Pending a Colab run — the stakeholder executes this notebook; results and
the device-transition timings get recorded here (measure, don't attribute:
lesson 5)._
