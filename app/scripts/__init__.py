"""Subprocess entry-point scripts for the Code Indexer Service.

Modules here are designed to be invoked via ``python -m app.scripts.<name>``
from a parent FastAPI worker so that heavy / OOM-prone work runs in an
isolated process. They MUST stay importable so their helpers can be unit
tested without spawning a subprocess.
"""
