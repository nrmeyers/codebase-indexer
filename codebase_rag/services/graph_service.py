"""graph_service.py — LadybugDB drop-in shim.

MemgraphIngestor is re-exported from ladybug_ingestor so all existing
imports (main.py, mcp/server.py, mcp/tools.py, semantic_search.py, tests)
continue to work without modification.
"""
from __future__ import annotations

# Re-export LadybugIngestor under the original name for backward compatibility
from codebase_rag.services.ladybug_ingestor import LadybugIngestor as MemgraphIngestor

__all__ = ["MemgraphIngestor"]

