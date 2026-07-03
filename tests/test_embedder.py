"""M1 tests for the Embedder Protocol boundary and FakeEmbedder."""

from code_index.core.embedder import Embedder, FakeEmbedder


async def takes(e: Embedder) -> int:
    return e.dim


async def test_fake_embedder_satisfies_protocol():
    assert isinstance(FakeEmbedder(), Embedder)


async def test_protocol_typed_helper_accepts_fake():
    assert await takes(FakeEmbedder(dim=16)) == 16


async def test_determinism_same_text_same_vector():
    e = FakeEmbedder()
    first = await e.embed_documents(["def foo(): pass"])
    second = await e.embed_documents(["def foo(): pass"])
    assert first == second
    assert await e.embed_query("find foo") == await e.embed_query("find foo")


async def test_dim_respected_and_batch_order_aligned():
    e = FakeEmbedder(dim=12)
    texts = ["alpha", "beta", "gamma"]
    vectors = await e.embed_documents(texts)
    assert len(vectors) == len(texts)
    assert all(len(v) == 12 for v in vectors)
    # Distinct texts embed to distinct vectors, in input order.
    assert len({tuple(v) for v in vectors}) == 3
    assert vectors[0] == (await e.embed_documents(["alpha"]))[0]
    assert len(await e.embed_query("alpha")) == 12
    assert all(-1.0 <= x <= 1.0 for v in vectors for x in v)


async def test_query_prefix_seam_differs_from_document():
    e = FakeEmbedder()
    assert await e.embed_query("x") != (await e.embed_documents(["x"]))[0]


async def test_call_recording():
    e = FakeEmbedder()
    await e.embed_documents(["a", "b"])
    await e.embed_documents(["c"])
    await e.embed_query("q1")
    assert e.document_calls == [["a", "b"], ["c"]]
    assert e.query_calls == ["q1"]
