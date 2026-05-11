"""Embedder protocol + shared constants (BUC-1605).

Every backend in this package implements :class:`EmbedderBackend`. The
protocol is intentionally minimal: one async ``embed`` method that takes a
batch of texts and returns one 768-dim vector per text, in input order.

Design notes
------------
* **Async** — the orchestrator path is FastAPI / asyncio; making the
  protocol async avoids the in-thread blocking dance for HTTP backends
  (TEI, SageMaker via boto3 → ``asyncio.to_thread``) while staying
  cheap for the in-process ``local`` backend (which wraps the
  synchronous ``sentence-transformers`` call in ``to_thread``).
* **Batch-only** — single-text embedding is just ``embed([text])[0]``.
  Forcing the batched shape on every caller keeps GPU/SageMaker utilisation
  high and eliminates a footgun where someone introduces an N+1 loop.
* **Fail loud** — backends raise :class:`EmbedderError` on hard
  configuration / network failures rather than returning ``None``. The old
  "return None and fall through" pattern hid a lot of bugs; with explicit
  backend selection the operator knows exactly which backend should work.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

#: e5-base-v2 native output dimension. Every backend in this package MUST
#: produce vectors of this length so the existing DuckDB ``FLOAT[768]``
#: schema stays compatible — zero migration when switching backends.
EMBEDDING_DIM = 768


class EmbedderError(RuntimeError):
    """Raised when an embedder backend cannot satisfy a request.

    Common causes:
        * Missing optional dependency (``sentence-transformers`` not
          installed for the ``local`` backend).
        * Missing configuration (no SageMaker endpoint set for the
          ``sagemaker`` backend).
        * Upstream service unreachable (TEI sidecar down, SageMaker
          endpoint cold-start timeout).
        * Protocol mismatch (backend returned a non-768-dim vector).
    """


@runtime_checkable
class EmbedderBackend(Protocol):
    """Common interface for all embedder backends.

    Implementations are stateful (they may cache HTTP clients, loaded
    torch models, etc.) but ``embed`` MUST be safe to call concurrently
    from multiple asyncio tasks.

    Attributes:
        name: Stable identifier of the backend (``"local"``, ``"sagemaker"``,
            or ``"tei"``). Surfaced in /health responses and logs.
        model: Name of the underlying model. All three default backends
            serve ``intfloat/e5-base-v2`` so this is informational.
    """

    name: str
    model: str

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts; return one 768-dim vector per text.

        Args:
            texts: Input strings to embed. Empty list yields ``[]`` without
                a network round-trip. Length is unbounded at the protocol
                level — backends batch internally as needed.

        Returns:
            list[list[float]]: ``len(result) == len(texts)``, each inner
            list has ``EMBEDDING_DIM`` elements.

        Raises:
            EmbedderError: On configuration / network / protocol failure.
        """
        ...
