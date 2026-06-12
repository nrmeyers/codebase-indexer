"""Spike-only SCIP parser — BUC-1615.

Reads a `.scip` protobuf file produced by `scip-typescript index --output ...`
and prints a human-readable summary: symbols, occurrences, relationships,
external symbols, and coarse statistics.

This is NOT wired into the production parser. It exists only to validate
that scip-typescript output is ingestible into LadybugDB alongside our
tree-sitter parse pipeline.

Prerequisites:
  protoc --python_out=. scip.proto   # produces scip_pb2.py

Usage:
  python parse_scip.py /tmp/scip-theforge.scip
  python parse_scip.py /tmp/scip-theforge.scip --json   # full JSON dump
  python parse_scip.py /tmp/scip-theforge.scip --sample 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

# scip_pb2 is generated from scip.proto via `protoc --python_out=. scip.proto`.
# For the spike we expect it to live next to this file or on PYTHONPATH.
THIS_DIR = Path(__file__).parent
for candidate in (THIS_DIR, Path("/tmp")):
    if (candidate / "scip_pb2.py").exists():
        sys.path.insert(0, str(candidate))
        break

try:
    import scip_pb2  # type: ignore
except ImportError:
    sys.stderr.write(
        "ERROR: scip_pb2 not found. Generate it with:\n"
        "  curl -fsSL https://raw.githubusercontent.com/sourcegraph/scip/main/scip.proto -o scip.proto\n"
        "  protoc --python_out=. scip.proto\n"
    )
    sys.exit(2)


# SymbolRole bitfield values from scip.proto SymbolRole enum.
ROLE_DEFINITION = 0x1
ROLE_IMPORT = 0x2
ROLE_WRITE_ACCESS = 0x4
ROLE_READ_ACCESS = 0x8
ROLE_GENERATED = 0x10
ROLE_TEST = 0x20


def role_names(role: int) -> list[str]:
    names = []
    if role & ROLE_DEFINITION:
        names.append("definition")
    if role & ROLE_IMPORT:
        names.append("import")
    if role & ROLE_WRITE_ACCESS:
        names.append("write")
    if role & ROLE_READ_ACCESS:
        names.append("read")
    if role & ROLE_GENERATED:
        names.append("generated")
    if role & ROLE_TEST:
        names.append("test")
    return names or ["reference"]


def parse_symbol(symbol: str) -> dict:
    """Crack a SCIP symbol string into scheme, package, descriptors.

    Example:
      'scip-typescript npm theforge 0.0.0 src/`App.tsx`/App().'
        -> scheme='scip-typescript', manager='npm', pkg='theforge',
           version='0.0.0', descriptors='src/`App.tsx`/App().'
    """
    # SCIP symbol format: <scheme> ' ' <manager> ' ' <name> ' ' <version> ' ' <descriptors>
    # Local symbols look like: 'local 0', 'local 1' ...
    if symbol.startswith("local "):
        return {"scheme": "local", "local_id": symbol[len("local "):], "descriptors": ""}
    parts = symbol.split(" ", 4)
    if len(parts) < 5:
        return {"scheme": parts[0] if parts else "", "descriptors": symbol}
    return {
        "scheme": parts[0],
        "manager": parts[1],
        "package": parts[2],
        "version": parts[3],
        "descriptors": parts[4],
    }


def load_index(path: str) -> "scip_pb2.Index":
    idx = scip_pb2.Index()
    with open(path, "rb") as fh:
        idx.ParseFromString(fh.read())
    return idx


def summarize(idx: "scip_pb2.Index") -> dict:
    docs = list(idx.documents)
    n_occurrences = 0
    n_definitions = 0
    n_references = 0
    n_imports = 0
    occurrence_symbols: Counter[str] = Counter()
    languages: Counter[str] = Counter()
    external_pkgs: Counter[str] = Counter()
    docs_with_symbols = 0
    local_def_count = 0
    global_def_count = 0

    for doc in docs:
        languages[doc.language] += 1
        if doc.symbols:
            docs_with_symbols += 1
        for sym in doc.symbols:
            if sym.symbol.startswith("local "):
                local_def_count += 1
            else:
                global_def_count += 1
        for occ in doc.occurrences:
            n_occurrences += 1
            roles = occ.symbol_roles
            if roles & ROLE_DEFINITION:
                n_definitions += 1
            else:
                n_references += 1
            if roles & ROLE_IMPORT:
                n_imports += 1
            occurrence_symbols[occ.symbol] += 1

    for ext in idx.external_symbols:
        parsed = parse_symbol(ext.symbol)
        pkg = parsed.get("package", "?")
        external_pkgs[pkg] += 1

    return {
        "metadata": {
            "version": int(idx.metadata.version),
            "tool_name": idx.metadata.tool_info.name,
            "tool_version": idx.metadata.tool_info.version,
            "project_root": idx.metadata.project_root,
            "text_encoding": int(idx.metadata.text_document_encoding),
        },
        "doc_count": len(docs),
        "docs_with_symbols": docs_with_symbols,
        "languages": dict(languages),
        "symbols": {
            "local_definitions": local_def_count,
            "global_definitions": global_def_count,
            "total_in_documents": local_def_count + global_def_count,
        },
        "occurrences": {
            "total": n_occurrences,
            "definitions": n_definitions,
            "references": n_references,
            "imports": n_imports,
            "unique_symbol_references": len(occurrence_symbols),
        },
        "external_symbols_count": len(idx.external_symbols),
        "external_packages_top_20": external_pkgs.most_common(20),
        "most_referenced_symbols_top_20": occurrence_symbols.most_common(20),
    }


def sample_doc(idx: "scip_pb2.Index", n: int = 10) -> list[dict]:
    """Return a sample of documents with a few occurrences each."""
    out = []
    docs = sorted(idx.documents, key=lambda d: -len(d.occurrences))[:n]
    for doc in docs:
        sample_occs = []
        for occ in list(doc.occurrences)[:5]:
            sample_occs.append({
                "symbol": occ.symbol,
                "roles": role_names(occ.symbol_roles),
                "range": list(occ.range),
                "syntax_kind": int(occ.syntax_kind) if occ.syntax_kind else 0,
            })
        sample_syms = []
        for sym in list(doc.symbols)[:5]:
            sample_syms.append({
                "symbol": sym.symbol,
                "kind": int(sym.kind) if sym.kind else 0,
                "display_name": sym.display_name,
                "n_relationships": len(sym.relationships),
            })
        out.append({
            "path": doc.relative_path,
            "language": doc.language,
            "occurrence_count": len(doc.occurrences),
            "symbol_count": len(doc.symbols),
            "sample_occurrences": sample_occs,
            "sample_symbols": sample_syms,
        })
    return out


def find_calls(idx: "scip_pb2.Index", limit: int = 20) -> list[dict]:
    """Find occurrences with syntax_kind == IdentifierFunction or callable references.

    SCIP doesn't have an explicit CALLS edge — calls are represented as
    references to a function/method symbol. The synthesizable graph edge
    is `(caller_def, ref_occurrence_symbol)` joined by enclosing range.
    """
    # SyntaxKind enum values (subset from scip.proto):
    #   IdentifierFunction = 50, IdentifierFunctionDefinition = 51, IdentifierBuiltin = 13
    SK_IDENT_FUNCTION = 50
    SK_IDENT_FN_DEF = 51

    samples = []
    for doc in idx.documents:
        for occ in doc.occurrences:
            if occ.syntax_kind in (SK_IDENT_FUNCTION, SK_IDENT_FN_DEF):
                samples.append({
                    "path": doc.relative_path,
                    "symbol": occ.symbol,
                    "is_definition": bool(occ.symbol_roles & ROLE_DEFINITION),
                    "range": list(occ.range),
                })
                if len(samples) >= limit:
                    return samples
    return samples


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("scip_file")
    ap.add_argument("--json", action="store_true", help="Emit full JSON summary")
    ap.add_argument("--sample", type=int, default=5, help="Sample N docs in detail")
    ap.add_argument("--calls", action="store_true", help="Show sample call-site occurrences")
    args = ap.parse_args()

    if not os.path.exists(args.scip_file):
        sys.stderr.write(f"ERROR: file not found: {args.scip_file}\n")
        return 1

    size_mb = os.path.getsize(args.scip_file) / (1024 * 1024)
    sys.stderr.write(f"Loading {args.scip_file} ({size_mb:.1f} MiB)...\n")
    idx = load_index(args.scip_file)

    summary = summarize(idx)
    summary["sample_documents"] = sample_doc(idx, args.sample)
    if args.calls:
        summary["sample_call_occurrences"] = find_calls(idx)

    if args.json:
        print(json.dumps(summary, indent=2, default=str))
    else:
        print(f"=== SCIP Index Summary ===")
        print(f"Tool: {summary['metadata']['tool_name']} v{summary['metadata']['tool_version']}")
        print(f"Project root: {summary['metadata']['project_root']}")
        print(f"Documents: {summary['doc_count']} ({summary['docs_with_symbols']} with symbols)")
        print(f"Languages: {summary['languages']}")
        print(f"\n=== Symbols (definitions emitted in Document.symbols) ===")
        print(f"Local: {summary['symbols']['local_definitions']}")
        print(f"Global: {summary['symbols']['global_definitions']}")
        print(f"Total: {summary['symbols']['total_in_documents']}")
        print(f"\n=== Occurrences ===")
        print(f"Total: {summary['occurrences']['total']}")
        print(f"Definitions: {summary['occurrences']['definitions']}")
        print(f"References: {summary['occurrences']['references']}")
        print(f"Imports: {summary['occurrences']['imports']}")
        print(f"Unique symbols referenced: {summary['occurrences']['unique_symbol_references']}")
        print(f"\n=== External symbols ===")
        print(f"Count: {summary['external_symbols_count']}")
        print("Top 20 external packages:")
        for pkg, n in summary["external_packages_top_20"]:
            print(f"  {n:>6}  {pkg}")
        print(f"\n=== Most-referenced symbols (top 20) ===")
        for sym, n in summary["most_referenced_symbols_top_20"]:
            print(f"  {n:>6}  {sym[:120]}")
        if args.calls:
            print(f"\n=== Sample call-site occurrences (first 20) ===")
            for c in summary["sample_call_occurrences"]:
                marker = "DEF " if c["is_definition"] else "CALL"
                print(f"  {marker}  {c['path']}  {c['symbol'][:100]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
