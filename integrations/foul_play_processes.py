# SPDX-License-Identifier: GPL-3.0-only
"""Reuse a spawn-based search pool; never fork a Rust runtime from an asyncio thread."""

from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import hashlib
import multiprocessing
from pathlib import Path


class SearchPool:
    def __init__(self):
        self.executor = None
        self.workers = None

    @contextmanager
    def context(self, *, max_workers):
        if self.executor is None:
            self.workers = max_workers
            self.executor = ProcessPoolExecutor(
                max_workers=max_workers, mp_context=multiprocessing.get_context("spawn")
            )
        if max_workers != self.workers:
            raise ValueError("Search parallelism cannot change within a frozen worker")
        yield self.executor

    def close(self):
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None


def install_spawn_search():
    import fp.search.main as upstream

    expected = "74fd40c4f93bb60acd4436737fcae4737775a8575a0bfe365255599872db03f9"
    # Git checkouts can use CRLF; bind normalized text consistently across hosts.
    digest = hashlib.sha256(Path(upstream.__file__).read_text().encode()).hexdigest()
    if digest != expected:
        raise ValueError("Foul Play search source changed; audit the process adapter")
    pool = SearchPool()
    upstream.ProcessPoolExecutor = pool.context
    return pool
