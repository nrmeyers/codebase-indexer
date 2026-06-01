"""Regression test — LocalEmbedder cross-loop asyncio.Lock deadlock.

Root cause (confirmed 2026-06-01):
    ``LocalEmbedder.__init__`` originally created ``self._lock = asyncio.Lock()``.
    ``embed_driver.py`` runs the local backend from a ``ThreadPoolExecutor``
    via ``asyncio.run(backend.embed(texts))``.  When ``_CONCURRENCY >= 2``
    (or when a first-call model-load overlaps with a second call), two
    ``asyncio.run()`` invocations in different threads both try to acquire the
    same ``asyncio.Lock``:

    * Thread 0 creates loop L0, acquires the lock, starts loading the model.
    * Thread 1 creates loop L1, tries to acquire the lock, and adds a waiter
      ``Future`` tied to L1 to the lock's internal ``_waiters`` deque.
    * Thread 0 finishes, calls ``lock.release()`` — which calls
      ``_wake_up_first()`` → ``fut.set_result(True)`` on Thread 1's waiter.
      But ``set_result`` is called from L0 (or no loop), not from L1.  The
      call is a no-op / raises because the Future belongs to a different loop.
    * Thread 1's waiter never fires → permanent hang at whatever
      ``embedding_count`` was reached before the concurrent overlap.

    Confirmed behaviour: exactly one thread completes, the second hangs
    indefinitely at the ``async with self._lock:`` line.  With a 18 707-node
    repo the first model-load flush covers ~3 714 symbols — the observed
    plateau.

Fix:
    ``asyncio.Lock`` replaced with ``threading.Lock`` in ``_ensure_model_loaded``.
    A plain OS threading lock is loop-agnostic and safe across concurrent
    ``asyncio.run()`` invocations.

This test file reproduces the hang at the unit level (no real model, no
subprocess) and asserts it does NOT hang with the fix in place.  The 4-second
``concurrent.futures.wait`` timeout is the hung-detection gate: if any future
is still pending after the timeout the fix is broken.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest.mock import MagicMock

import numpy as np

from app.embedders.local import LocalEmbedder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_embedder_with_slow_load(load_delay: float = 0.2) -> LocalEmbedder:
    """Return a fresh LocalEmbedder whose _load_model sleeps for ``load_delay``s.

    The delay ensures that a concurrent second ``asyncio.run(embed(...))``
    call starts WHILE the first is still inside the model-load critical
    section, which is the exact race that triggers the deadlock.
    """
    backend = LocalEmbedder()

    def _slow_load() -> Any:
        time.sleep(load_delay)
        model_mock = MagicMock()
        model_mock.encode.side_effect = lambda texts, **kw: np.zeros(
            (len(texts), 768), dtype="float32"
        )
        return model_mock

    backend._load_model = _slow_load  # type: ignore[method-assign]
    return backend


# ---------------------------------------------------------------------------
# 1. Cross-loop asyncio.Lock deadlock regression
# ---------------------------------------------------------------------------


def test_should_not_deadlock_when_concurrent_asyncio_run_calls_share_embedder() -> None:
    """Two concurrent asyncio.run(embed(...)) calls on one LocalEmbedder must both complete.

    This is the exact pattern used by embed_driver.py with _CONCURRENCY=2:
    the shared LocalEmbedder singleton (lru_cache) is called from multiple
    ThreadPoolExecutor threads, each running asyncio.run().  The original
    asyncio.Lock caused the second thread's waiter to be attached to a
    different event loop than the one that called release(), so the wakeup
    never fired and the thread hung forever.

    Failure mode (pre-fix): after ~4 seconds, done=1, pending=1 (one thread
    still hung at ``async with self._lock``).

    Pass condition (post-fix): both futures complete within 4 seconds.
    """
    backend = _make_embedder_with_slow_load(load_delay=0.3)

    def run_embed(texts: list[str]) -> list[list[float]]:
        return asyncio.run(backend.embed(texts))

    pool = ThreadPoolExecutor(max_workers=2)
    try:
        # Submit two batches concurrently — both start before model is loaded.
        futs = [pool.submit(run_embed, ["def fn(): pass"] * 10) for _ in range(2)]
        done, pending = concurrent.futures.wait(futs, timeout=8)
    finally:
        pool.shutdown(wait=False)

    assert len(pending) == 0, (
        f"asyncio.Lock deadlock regression: {len(pending)} future(s) still hung "
        f"after 8s — the threading.Lock fix did not take effect.  "
        f"Completed: {len(done)}.  "
        "This means LocalEmbedder._load_lock is still an asyncio.Lock."
    )
    assert len(done) == 2, f"expected 2 completed futures, got {len(done)}"

    for fut in done:
        result = fut.result()
        assert len(result) == 10, f"expected 10 vectors, got {len(result)}"
        for vec in result:
            assert len(vec) == 768


def test_should_not_deadlock_with_many_concurrent_batches() -> None:
    """Eight concurrent asyncio.run(embed(...)) calls must all complete.

    A higher-concurrency stress case that would expose even occasional races
    in the model-load guard.  Uses a short delay so the test stays fast but
    long enough that all 8 threads are in flight simultaneously.
    """
    backend = _make_embedder_with_slow_load(load_delay=0.1)

    def run_embed(texts: list[str]) -> list[list[float]]:
        return asyncio.run(backend.embed(texts))

    n = 8
    pool = ThreadPoolExecutor(max_workers=n)
    try:
        futs = [pool.submit(run_embed, [f"def fn_{i}(): pass"]) for i in range(n)]
        done, pending = concurrent.futures.wait(futs, timeout=10)
    finally:
        pool.shutdown(wait=False)

    assert len(pending) == 0, (
        f"asyncio.Lock deadlock regression: {len(pending)} future(s) still hung "
        f"after 10s — fix is broken for {n}-way concurrency."
    )
    assert len(done) == n


# ---------------------------------------------------------------------------
# 2. _ensure_model_loaded uses threading.Lock (not asyncio.Lock)
# ---------------------------------------------------------------------------


def test_should_use_threading_lock_for_model_loading() -> None:
    """LocalEmbedder._load_lock must be a threading.Lock, not asyncio.Lock.

    This is the structural fix: asyncio.Lock is unsafe across different
    event loops (concurrent asyncio.run() calls in different threads).
    threading.Lock has no such restriction.
    """
    backend = LocalEmbedder()
    assert isinstance(backend._load_lock, type(threading.Lock())), (
        "LocalEmbedder._load_lock must be a threading.Lock (not asyncio.Lock).  "
        "The asyncio.Lock caused a deterministic hang when embed_driver.py called "
        "asyncio.run(embed(...)) concurrently from a ThreadPoolExecutor."
    )


def test_should_not_have_asyncio_lock_attribute() -> None:
    """The old ``_lock`` asyncio.Lock attribute must not exist on LocalEmbedder.

    If both ``_lock`` (asyncio.Lock) and ``_load_lock`` (threading.Lock) are
    present, the fix is incomplete — something might still use the asyncio one.
    """
    backend = LocalEmbedder()
    assert not hasattr(backend, "_lock"), (
        "LocalEmbedder still has a ``_lock`` attribute — this was the buggy "
        "asyncio.Lock.  Remove it and use ``_load_lock`` (threading.Lock) everywhere."
    )


# ---------------------------------------------------------------------------
# 3. Model is loaded exactly once under concurrent pressure
# ---------------------------------------------------------------------------


def test_should_load_model_exactly_once_under_concurrent_load() -> None:
    """Threading.Lock guarantees the model is loaded at most once.

    Without a load guard, two threads could both pass the ``if _model is None``
    check and both call ``_load_model()``, which is expensive and potentially
    inconsistent (the second write clobbers the first).  The threading.Lock
    double-check idiom prevents this.
    """
    backend = LocalEmbedder()
    load_call_count = [0]

    def _counting_load() -> Any:
        load_call_count[0] += 1
        # Brief sleep to maximise the chance two threads race.
        time.sleep(0.05)
        model_mock = MagicMock()
        model_mock.encode.side_effect = lambda texts, **kw: np.zeros(
            (len(texts), 768), dtype="float32"
        )
        return model_mock

    backend._load_model = _counting_load  # type: ignore[method-assign]

    def run_embed() -> None:
        asyncio.run(backend.embed(["test"]))

    threads = [threading.Thread(target=run_embed) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert load_call_count[0] == 1, (
        f"_load_model was called {load_call_count[0]} times; expected exactly 1.  "
        "The double-check lock idiom in _ensure_model_loaded is broken."
    )


# ---------------------------------------------------------------------------
# 4. Fast path: no lock acquired when model already loaded
# ---------------------------------------------------------------------------


def test_should_skip_lock_acquisition_when_model_is_already_loaded() -> None:
    """_ensure_model_loaded is a no-op when _model is not None.

    The fast-path check (``if self._model is not None: return``) avoids
    touching the lock on every embed call after the model is warm.  This
    test verifies the lock is NOT acquired on the fast path by replacing
    ``_load_lock`` with a mock that asserts it is never entered.
    """
    import numpy as np

    model_mock = MagicMock()
    model_mock.encode.side_effect = lambda texts, **kw: np.zeros(
        (len(texts), 768), dtype="float32"
    )

    backend = LocalEmbedder()
    backend._model = model_mock  # model already loaded

    # Wrap _load_lock with a sentinel that tracks acquisitions.
    lock_acquired = [False]

    class _SentinelLock:
        def __enter__(self) -> "_SentinelLock":
            lock_acquired[0] = True
            return self

        def __exit__(self, *_: object) -> None:
            pass

    backend._load_lock = _SentinelLock()  # type: ignore[assignment]

    # embed() with model already loaded must NOT enter _load_lock.
    asyncio.run(backend.embed(["def fn(): pass"]))

    assert not lock_acquired[0], (
        "_load_lock was acquired even though _model was already set.  "
        "The fast-path ``if self._model is not None: return`` in "
        "_ensure_model_loaded is missing or broken."
    )
