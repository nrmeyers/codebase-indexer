"""Per-model query/document prefix registry (768 embedder POC).

Instruction-tuned embedders need a *different* textual prefix on the query
side vs the document side; omitting them silently degrades recall — the #1
footgun called out in ``docs/embedder-poc-768.md``. The configured-backend
embed path is otherwise model-agnostic and sends raw text, so this module is
the single place that injects the correct prefix per model. Both the index
side (``app/scripts/embed_driver.py``) and the query sides
(``app/routers/search.py::_embed_query`` and ``POST /embed``) consult
:func:`apply_prefix`, so they can never drift out of symmetry.

Scope: prefixes are applied ONLY to the in-process ``local`` backend — the
POC swap path (``EMBEDDER_BACKEND=local`` + ``LOCAL_EMBED_MODEL``). Prod
backends (``sagemaker``/``tei``/``openai``) keep their existing raw-text
behaviour so already-built indexes stay valid.

Prefix values are verified against each model card — NOT the vendored
``codebase_rag.constants`` values, which are stale for CodeRankEmbed: those
apply ``search_query: `` / ``Represent this code snippet: `` (nomic-embed-text
conventions), whereas the CodeRankEmbed card requires the query-only
``Represent this query for searching relevant code: `` with raw documents.
"""
from __future__ import annotations

from typing import Literal, NamedTuple

from .base import EmbedderBackend

Role = Literal["query", "document"]


class ModelPrefix(NamedTuple):
    """Query- and document-side prefix for one model.

    An empty string means "embed raw" for that side. Asymmetric models set
    exactly one side (CodeRankEmbed is query-only); symmetric models are
    simply absent from :data:`PREFIXES` and default to empty/empty.
    """

    query: str
    document: str


#: HF model id -> prefixes. Only asymmetric / instruction-tuned models need an
#: entry; the symmetric roster members (``jinaai/jina-embeddings-v2-base-code``,
#: ``Alibaba-NLP/gte-modernbert-base``, ``ibm-granite/granite-embedding-english-r2``)
#: are intentionally omitted and default to no prefix, which is correct per
#: their cards.
PREFIXES: dict[str, ModelPrefix] = {
    # e5 family: trained with these exact prefixes. Running prefix-less
    # understates recall — this is the baseline bug the registry fixes.
    "intfloat/e5-base-v2": ModelPrefix(query="query: ", document="passage: "),
    # Nomic CodeRankEmbed: query-only instruction prefix, documents raw.
    # Verified against huggingface.co/nomic-ai/CodeRankEmbed (the vendored
    # codebase_rag.constants values are wrong — see module docstring).
    "nomic-ai/CodeRankEmbed": ModelPrefix(
        query="Represent this query for searching relevant code: ",
        document="",
    ),
}

_NO_PREFIX = ModelPrefix(query="", document="")


def for_model(model: str) -> ModelPrefix:
    """Return the :class:`ModelPrefix` for ``model``, or empty when unregistered."""
    return PREFIXES.get(model.strip(), _NO_PREFIX)


def apply_prefix(
    backend: EmbedderBackend,
    texts: list[str],
    *,
    role: Role,
) -> list[str]:
    """Prepend the model's ``role``-appropriate prefix to each text.

    No-op (returns ``texts`` unchanged) unless the backend is the in-process
    ``local`` backend AND its model has a non-empty prefix for this role.
    Gating here means callers don't need to know which models are asymmetric —
    they only declare whether they're embedding a query or a document, and
    index/query symmetry is guaranteed because both sides read this registry.

    Args:
        backend: The resolved embedder backend (``name`` + ``model`` are read).
        texts: Raw input strings (without any prefix).
        role: ``"query"`` or ``"document"``.

    Returns:
        A new prefixed list, or ``texts`` unchanged when no prefix applies.
    """
    if not texts or getattr(backend, "name", None) != "local":
        return texts
    mp = for_model(getattr(backend, "model", ""))
    prefix = mp.query if role == "query" else mp.document
    if not prefix:
        return texts
    return [prefix + t for t in texts]


__all__ = ["ModelPrefix", "PREFIXES", "Role", "apply_prefix", "for_model"]
