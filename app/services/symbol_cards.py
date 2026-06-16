"""Symbol-card marker + parent-qname fold (methodology §5).

A ``{qn}::Symbol::card`` doc is a never-emitted, task-vocabulary retrieval
proxy: when a card hit lands in semantic results, callers fold its qname
back to the parent ``{qn}`` and de-duplicate so the parent symbol surfaces
with the card's score and the card qname never reaches consumers.

Centralising the marker + fold here keeps search.py, context_bundle.py,
and embed_driver.py from drifting on the same invariant.
"""
from __future__ import annotations

SYMBOL_CARD_MARKER = "::Symbol::card"


def fold_card_qname(qn: str) -> str:
    """Map a ``{qn}::Symbol::card`` proxy to its parent symbol; else identity."""
    if qn.endswith(SYMBOL_CARD_MARKER):
        return qn[: -len(SYMBOL_CARD_MARKER)]
    return qn
