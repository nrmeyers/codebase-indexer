"""Unit tests for the LM Studio adapter (`app.services.lm_studio`).

The adapter is opt-in via ``LM_STUDIO_URL`` and best-effort: every public
function returns a sentinel (``None``, ``[]``, ``False``) on any error
instead of raising.  These tests stub the HTTP layer so we never touch
a real LM Studio server, and assert both happy-path and graceful-failure
behaviour for each surface area.
"""
from __future__ import annotations

import importlib

import pytest


@pytest.fixture(autouse=True)
def _reset_module(monkeypatch: pytest.MonkeyPatch):
    """Reload ``lm_studio`` per test so the module-level probe cache and
    ``_log_backend_once`` lru_cache start clean.
    """
    # Clear env first so the fresh module sees a known baseline.
    for k in (
        "LM_STUDIO_URL",
        "LM_STUDIO_EMBED_MODEL",
        "LM_STUDIO_RERANK_MODEL",
        "LM_STUDIO_TIMEOUT",
    ):
        monkeypatch.delenv(k, raising=False)
    from app.services import lm_studio  # noqa: WPS433
    importlib.reload(lm_studio)
    yield
    importlib.reload(lm_studio)


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


def test_base_url_empty_when_unset() -> None:
    from app.services import lm_studio
    assert lm_studio.base_url() == ""


def test_base_url_strips_trailing_slash(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.services import lm_studio
    monkeypatch.setenv("LM_STUDIO_URL", "http://localhost:1234/")
    assert lm_studio.base_url() == "http://localhost:1234"


def test_default_hints() -> None:
    from app.services import lm_studio
    assert lm_studio.embed_model_hint() == "CodeRankEmbed"
    assert lm_studio.rerank_model_hint() == "CodeRankLLM"


def test_request_timeout_clamps_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.services import lm_studio
    monkeypatch.setenv("LM_STUDIO_TIMEOUT", "not-a-number")
    assert lm_studio.request_timeout_s() == 30.0
    monkeypatch.setenv("LM_STUDIO_TIMEOUT", "0.0")
    # Floor is 1.0 so a transient socket error doesn't burn a 0-second wait.
    assert lm_studio.request_timeout_s() == 1.0


# ---------------------------------------------------------------------------
# Health probe + model resolution
# ---------------------------------------------------------------------------


def test_list_models_returns_empty_when_disabled() -> None:
    from app.services import lm_studio
    # No URL set → don't even probe the network.
    assert lm_studio.list_models() == []
    assert lm_studio.is_available() is False


def test_list_models_caches_result(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.services import lm_studio
    monkeypatch.setenv("LM_STUDIO_URL", "http://localhost:9999")

    calls = {"n": 0}

    def fake_get_json(url: str, timeout: float):
        calls["n"] += 1
        return {"data": [{"id": "nomic-ai/CodeRankEmbed-Q4"}, {"id": "CodeRankLLM-7B"}]}

    monkeypatch.setattr(lm_studio, "_get_json", fake_get_json)

    a = lm_studio.list_models()
    b = lm_studio.list_models()
    assert a == ["nomic-ai/CodeRankEmbed-Q4", "CodeRankLLM-7B"]
    assert b == a
    # Cache TTL keeps us at exactly one network call within the test window.
    assert calls["n"] == 1


def test_resolve_model_substring_match(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.services import lm_studio
    monkeypatch.setenv("LM_STUDIO_URL", "http://localhost:9999")
    monkeypatch.setattr(
        lm_studio,
        "_get_json",
        lambda *a, **k: {
            "data": [
                {"id": "nomic-ai/CodeRankEmbed-GGUF/CodeRankEmbed-Q4_K_M.gguf"},
                {"id": "Qwen2.5-Coder-7B-Instruct-CodeRankLLM"},
            ]
        },
    )
    assert lm_studio.resolve_model("CodeRankEmbed").endswith(".gguf")
    assert "CodeRankLLM" in lm_studio.resolve_model("coderankllm")  # case-insensitive
    assert lm_studio.resolve_model("does-not-exist") is None
    assert lm_studio.resolve_model("") is None


def test_list_models_returns_empty_on_http_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.services import lm_studio
    monkeypatch.setenv("LM_STUDIO_URL", "http://localhost:9999")

    def boom(*_a, **_k):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(lm_studio, "_get_json", boom)
    assert lm_studio.list_models() == []
    assert lm_studio.is_available() is False


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def test_embed_returns_none_when_disabled() -> None:
    from app.services import lm_studio
    assert lm_studio.embed("def foo(): pass") is None


def test_embed_passes_prefix_in_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.services import lm_studio
    monkeypatch.setenv("LM_STUDIO_URL", "http://localhost:9999")
    monkeypatch.setattr(
        lm_studio,
        "_get_json",
        lambda *a, **k: {"data": [{"id": "CodeRankEmbed"}]},
    )

    captured: dict = {}

    def fake_post(url: str, payload: dict, timeout: float):
        captured["url"] = url
        captured["payload"] = payload
        return {"data": [{"embedding": [0.1] * 768}]}

    monkeypatch.setattr(lm_studio, "_post_json", fake_post)

    vec = lm_studio.embed("def foo(): pass", prefix="Represent this code snippet: ")
    assert vec is not None
    assert len(vec) == 768
    assert captured["url"].endswith("/v1/embeddings")
    assert captured["payload"]["input"].startswith("Represent this code snippet: ")
    assert captured["payload"]["model"] == "CodeRankEmbed"


def test_embed_returns_none_on_post_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.services import lm_studio
    monkeypatch.setenv("LM_STUDIO_URL", "http://localhost:9999")
    monkeypatch.setattr(
        lm_studio,
        "_get_json",
        lambda *a, **k: {"data": [{"id": "CodeRankEmbed"}]},
    )

    def boom(*_a, **_k):
        raise RuntimeError("LM Studio crashed")

    monkeypatch.setattr(lm_studio, "_post_json", boom)
    assert lm_studio.embed("hello") is None


def test_embed_returns_none_when_model_not_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.services import lm_studio
    monkeypatch.setenv("LM_STUDIO_URL", "http://localhost:9999")
    # /v1/models returns a model that does NOT match the embed hint
    monkeypatch.setattr(
        lm_studio,
        "_get_json",
        lambda *a, **k: {"data": [{"id": "some-other-llm"}]},
    )
    assert lm_studio.embed("hello") is None


# ---------------------------------------------------------------------------
# Chat completion
# ---------------------------------------------------------------------------


def test_chat_complete_returns_none_when_disabled() -> None:
    from app.services import lm_studio
    assert lm_studio.chat_complete([{"role": "user", "content": "hi"}]) is None


def test_chat_complete_extracts_assistant_content(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.services import lm_studio
    monkeypatch.setenv("LM_STUDIO_URL", "http://localhost:9999")
    monkeypatch.setattr(
        lm_studio,
        "_get_json",
        lambda *a, **k: {"data": [{"id": "CodeRankLLM"}]},
    )
    monkeypatch.setattr(
        lm_studio,
        "_post_json",
        lambda *a, **k: {"choices": [{"message": {"content": "[3] > [1] > [2]"}}]},
    )
    out = lm_studio.chat_complete(
        [{"role": "user", "content": "rank these"}],
        max_tokens=64,
    )
    assert out == "[3] > [1] > [2]"


def test_chat_complete_returns_none_on_empty_choices(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.services import lm_studio
    monkeypatch.setenv("LM_STUDIO_URL", "http://localhost:9999")
    monkeypatch.setattr(
        lm_studio,
        "_get_json",
        lambda *a, **k: {"data": [{"id": "CodeRankLLM"}]},
    )
    monkeypatch.setattr(lm_studio, "_post_json", lambda *a, **k: {"choices": []})
    assert lm_studio.chat_complete([{"role": "user", "content": "x"}]) is None
