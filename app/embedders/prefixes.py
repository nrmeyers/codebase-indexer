"""Per-model query/document prefix registry (768 embedder POC).

Instruction-tuned embedders need a *different* textual prefix on the query
side vs the document side; omitting them silently degrades recall — the #1
footgun called out in ``docs/embedder-poc-768.md``. The configured-backend
embed path is otherwise model-agnostic and sends raw text, so this module is
the single place that injects the correct prefix per model. Both the index
side (``app/scripts/embed_driver.py``) and the query sides
(``app/routers/search.py::_embed_query`` and ``POST /embed``) consult
:func:`apply_prefix`, so they can never drift out of symmetry.

Scope: prefixes are applied to the POC swap-path backends — the in-process
``local`` backend (``EMBEDDER_BACKEND=local`` + ``LOCAL_EMBED_MODEL``), the
``llama_server`` backend (``EMBEDDER_BACKEND=llama_server`` against a llama.cpp
embedding endpoint), and the ``sagemaker`` backend once the endpoint advertises
a registered HF model id via ``SAGEMAKER_PREFIX_MODEL``. The remaining prod
backends (``tei``/``openai``) keep their existing raw-text behaviour so
already-built indexes stay valid. The ``llama_server`` and ``sagemaker``
backends expose the HF tokenizer / model id via ``prefix_model``; when
present, the registry keys off that instead of the friendlier ``model`` label
so a GGUF filename or a SageMaker deployment alias still maps to its HF model
card. SageMaker with ``prefix_model`` unset is the back-compat path: the gate
finds no registry hit and returns texts unchanged.

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
    # Nomic general text embedder: symmetric task-instruction prefixes
    # (search_query / search_document) per huggingface.co/nomic-ai/nomic-embed-text-v1.5.
    "nomic-ai/nomic-embed-text-v1.5": ModelPrefix(
        query="search_query: ",
        document="search_document: ",
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

    No-op (returns ``texts`` unchanged) unless the backend is one of the
    swap-path backends (``local``, ``llama_server``, or ``sagemaker``) AND
    its resolved model has a non-empty prefix for this role. The resolved
    model is ``backend.prefix_model`` when present (so a ``llama_server``
    GGUF or a ``sagemaker`` endpoint can advertise its HF id), otherwise
    ``backend.model``. Gating here means callers don't need to know which
    models are asymmetric — they only declare whether they're embedding a
    query or a document, and index/query symmetry is guaranteed because
    both sides read this registry.

    Args:
        backend: The resolved embedder backend (``name`` + ``model`` are read).
        texts: Raw input strings (without any prefix).
        role: ``"query"`` or ``"document"``.

    Returns:
        A new prefixed list, or ``texts`` unchanged when no prefix applies.
    """
    if not texts or getattr(backend, "name", None) not in (
        "local",
        "llama_server",
        "sagemaker",
    ):
        return texts
    # ``prefix_model`` lets a backend (e.g. ``llama_server``) advertise an
    # HF model id distinct from its serving name (e.g. a ``.gguf`` filename).
    model_id = getattr(backend, "prefix_model", None) or getattr(backend, "model", "")
    mp = for_model(model_id)
    prefix = mp.query if role == "query" else mp.document
    if not prefix:
        return texts
    return [prefix + t for t in texts]


__all__ = ["ModelPrefix", "PREFIXES", "Role", "apply_prefix", "for_model"]
