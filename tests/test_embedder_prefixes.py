"""Per-model prefix registry + symmetric wiring (768 embedder POC).

Covers :mod:`app.embedders.prefixes` and the two call sites that MUST stay
symmetric: the query side (``embed_text_sync(role="query")``, used by
``search._embed_query``) and the index side (``resolve_batch_embedder``'s
``_embed_via_backend``, ``role="document"``).

The gate is ``backend.name == "local"`` — prod backends keep raw-text
behaviour, which is what keeps the existing ``test_embed_driver`` symmetry
tests (they use a ``sagemaker``-named fake) green.
"""
from __future__ import annotations

import pytest

from app.embedders.prefixes import ModelPrefix, apply_prefix, for_model


class _FakeBackend:
    """EmbedderBackend stub recording the exact texts passed to ``embed()``."""

    def __init__(self, *, name: str, model: str) -> None:
        self.name = name
        self.model = model
        self.dim = 768
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(len(t))] * self.dim for t in texts]


# --- for_model -----------------------------------------------------------


def test_for_model_returns_e5_prefixes() -> None:
    assert for_model("intfloat/e5-base-v2") == ModelPrefix(
        query="query: ", document="passage: "
    )


def test_for_model_coderank_is_query_only() -> None:
    mp = for_model("nomic-ai/CodeRankEmbed")
    assert mp.query == "Represent this query for searching relevant code: "
    # Documents are embedded raw per the model card (NOT the stale engine
    # constant "Represent this code snippet: ").
    assert mp.document == ""


@pytest.mark.parametrize(
    "model",
    [
        "jinaai/jina-embeddings-v2-base-code",
        "Alibaba-NLP/gte-modernbert-base",
        "ibm-granite/granite-embedding-english-r2",
        "some/unregistered-model",
    ],
)
def test_for_model_symmetric_models_have_no_prefix(model: str) -> None:
    assert for_model(model) == ModelPrefix(query="", document="")


def test_for_model_strips_whitespace() -> None:
    assert for_model("  intfloat/e5-base-v2  ").query == "query: "


# --- apply_prefix: gating + roles ---------------------------------------


def test_apply_prefix_local_e5_query_and_document() -> None:
    be = _FakeBackend(name="local", model="intfloat/e5-base-v2")
    assert apply_prefix(be, ["x", "y"], role="query") == ["query: x", "query: y"]
    assert apply_prefix(be, ["x", "y"], role="document") == [
        "passage: x",
        "passage: y",
    ]


def test_apply_prefix_noop_for_nonlocal_backend_even_with_known_model() -> None:
    # Prod backends keep raw-text behaviour so already-built indexes stay
    # valid — the gate is backend.name == "local".
    be = _FakeBackend(name="sagemaker", model="intfloat/e5-base-v2")
    assert apply_prefix(be, ["x"], role="query") == ["x"]
    assert apply_prefix(be, ["x"], role="document") == ["x"]


def test_apply_prefix_noop_for_symmetric_model_on_local() -> None:
    be = _FakeBackend(name="local", model="Alibaba-NLP/gte-modernbert-base")
    assert apply_prefix(be, ["x"], role="query") == ["x"]
    assert apply_prefix(be, ["x"], role="document") == ["x"]


def test_apply_prefix_coderank_query_prefixed_document_raw() -> None:
    be = _FakeBackend(name="local", model="nomic-ai/CodeRankEmbed")
    assert apply_prefix(be, ["q"], role="query") == [
        "Represent this query for searching relevant code: q"
    ]
    assert apply_prefix(be, ["doc"], role="document") == ["doc"]


def test_apply_prefix_empty_list_is_noop() -> None:
    be = _FakeBackend(name="local", model="intfloat/e5-base-v2")
    assert apply_prefix(be, [], role="query") == []


# --- symmetry contract ---------------------------------------------------


@pytest.mark.parametrize(
    "model",
    [
        "intfloat/e5-base-v2",
        "nomic-ai/CodeRankEmbed",
        "Alibaba-NLP/gte-modernbert-base",
    ],
)
def test_index_and_query_sides_use_same_registry_entry(model: str) -> None:
    """The document prefix the index side applies and the query prefix the
    query side applies are the two halves of ONE registry entry, so they can
    never drift. Both real call sites go through ``apply_prefix``; asserting it
    here proves they receive matching halves."""
    be = _FakeBackend(name="local", model=model)
    mp = for_model(model)
    doc = apply_prefix(be, ["t"], role="document")[0]
    qry = apply_prefix(be, ["t"], role="query")[0]
    assert doc == (mp.document + "t" if mp.document else "t")
    assert qry == (mp.query + "t" if mp.query else "t")


# --- embed_text_sync role plumbing (query side) -------------------------


def test_embed_text_sync_query_role_prefixes_local_e5(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    be = _FakeBackend(name="local", model="intfloat/e5-base-v2")
    monkeypatch.setattr(
        "app.embedders.sync_bridge.get_embedder_or_none", lambda: be
    )
    from app.embedders.sync_bridge import embed_text_sync

    embed_text_sync("hello", role="query")
    assert be.calls == [["query: hello"]]


def test_embed_text_sync_default_role_is_raw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # role=None preserves the legacy raw-text contract (warmup, v2 embedder).
    be = _FakeBackend(name="local", model="intfloat/e5-base-v2")
    monkeypatch.setattr(
        "app.embedders.sync_bridge.get_embedder_or_none", lambda: be
    )
    from app.embedders.sync_bridge import embed_text_sync

    embed_text_sync("hello")
    assert be.calls == [["hello"]]


# --- index side: resolve_batch_embedder applies document prefix ---------


def test_index_pass_applies_document_prefix_for_local_e5(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.scripts.embed_driver import resolve_batch_embedder

    be = _FakeBackend(name="local", model="intfloat/e5-base-v2")
    monkeypatch.setattr(
        "app.embedders.sync_bridge.get_embedder_or_none", lambda: be
    )
    fn = resolve_batch_embedder()
    fn(["def foo(): ...", "class Bar: ..."])
    assert be.calls == [
        ["passage: def foo(): ...", "passage: class Bar: ..."]
    ]
