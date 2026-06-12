from __future__ import annotations

import importlib.util
from collections.abc import Sequence

from codebase_rag.constants import (
    MODULE_TORCH,
    MODULE_TRANSFORMERS,
)

_dependency_cache: dict[str, bool] = {}


def _check_dependency(module_name: str) -> bool:
    if module_name not in _dependency_cache:
        _dependency_cache[module_name] = (
            importlib.util.find_spec(module_name) is not None
        )
    return _dependency_cache[module_name]


def has_torch() -> bool:
    return _check_dependency(MODULE_TORCH)


def has_transformers() -> bool:
    return _check_dependency(MODULE_TRANSFORMERS)


def has_semantic_dependencies() -> bool:
    """Return True when CodeRankEmbed embedding dependencies are available.

    Requires ``torch`` and ``transformers`` (loaded via the in-process
    embedder fallback path).  When the LM Studio adapter is in use the
    service makes HTTP calls to ``$LM_STUDIO_URL`` and these deps are
    unnecessary; this check still returns True if torch/transformers are
    present so the fallback path remains available.  Qdrant was removed
    in the LadybugDB migration and is no longer a dependency.
    """
    return has_torch() and has_transformers()


def check_dependencies(required_modules: Sequence[str]) -> bool:
    return all(_check_dependency(module) for module in required_modules)


def get_missing_dependencies(required_modules: Sequence[str]) -> list[str]:
    return [module for module in required_modules if not _check_dependency(module)]
