"""Tests for the Python __init__.py module-metadata extractor
(Phase 1.2b)."""
from __future__ import annotations

from app.services.chunk_strategies import (
    ModuleMetadata,
    build_module_chunk_input,
    extract_module_metadata,
)


def test_should_extract_all_and_docstring_when_init_py_declares_dunder_all() -> None:
    """A canonical __init__.py with a docstring + ``__all__`` round-trips
    cleanly through extract → build_module_chunk_input."""
    src = '''"""Service registry — composes adapters."""
from .foo import Foo
from .bar import bar

__all__ = ["Foo", "bar"]
'''
    meta = extract_module_metadata("src/services/__init__.py", src)
    assert meta == ModuleMetadata(
        docstring="Service registry — composes adapters.",
        public_symbols=["Foo", "bar"],
    )

    out = build_module_chunk_input(
        module_qname="forge.services",
        module_path="src/services/__init__.py",
        docstring=meta.docstring,
        public_symbols=meta.public_symbols,
    )
    assert "# Public: Foo, bar" in out
    assert "Service registry — composes adapters." in out


def test_should_return_empty_metadata_when_init_py_is_empty() -> None:
    """An empty __init__.py — common in Python packages — produces a
    valid :class:`ModuleMetadata` with empty docstring + empty
    public_symbols.  The caller decides whether to emit a chunk; we
    return non-None so the caller knows the file parsed successfully."""
    meta = extract_module_metadata("pkg/__init__.py", "")
    assert meta is not None
    assert meta.docstring == ""
    assert meta.public_symbols == []
