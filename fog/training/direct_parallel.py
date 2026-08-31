"""Concurrency helpers for authoritative direct-Showdown rollout collection."""

from __future__ import annotations

import multiprocessing as mp
import time
import traceback
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from multiprocessing.connection import Connection, wait
from typing import Any, Self, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class WorkerGame:
    """A deterministically assigned game; tags remain unique across all workers/chunks."""

    worker_id: int
    game_number: int
    seed: int
    p1_index: int
    p2_index: int

    @property
    def battle_tag(self) -> str:
        return f"battle-direct-w{self.worker_id}-g{self.game_number}"


def partition_games(games: list[WorkerGame], worker_count: int) -> list[list[WorkerGame]]:
    """Stable round-robin assignment avoids a fast worker changing future RNG assignments."""
    if worker_count < 1:
        raise ValueError("worker_count must be >= 1")
    parts = [[] for _ in range(worker_count)]
    for index, game in enumerate(games):
        worker_id = index % worker_count
        parts[worker_id].append(replace(game, worker_id=worker_id))
    return parts


def run_workers(
    assignments: list[list[WorkerGame]],
    fn: Callable[[int, list[WorkerGame]], T],
) -> list[T]:
    """Run isolated workers, propagate the first exception, and always shut down cleanly."""
    if not assignments:
        return []
    executor = ThreadPoolExecutor(
        max_workers=len(assignments), thread_name_prefix="fog-direct-rollout"
    )
    futures: list[Future[T]] = []
    try:
        futures = [
            executor.submit(fn, worker_id, games)
            for worker_id, games in enumerate(assignments)
        ]
        # Consume in worker-id order so merging and metrics are deterministic.
        return [future.result() for future in futures]
    except BaseException:
        for future in futures:
            future.cancel()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


class WorkerProcessError(RuntimeError):
    """A persistent rollout process failed or exited without returning a result."""


def _process_worker_main(
    connection: Connection,
    worker_id: int,
    worker_factory: Callable[..., Any],
    factory_args: tuple[Any, ...],
) -> None:
    """Own one long-lived worker object and expose its ``run`` method over a pipe."""
    worker = None
    try:
        worker = worker_factory(worker_id, *factory_args)
        connection.send(("ready", None))
        while True:
            command, payload = connection.recv()
            if command == "run":
                try:
                    connection.send(("ok", worker.run(payload)))
                except BaseException as exc:  # noqa: BLE001 - child-process exception boundary
                    connection.send(
                        (
                            "error",
                            type(exc).__name__,
                            str(exc),
                            traceback.format_exc(),
                        )
                    )
            elif command == "stop":
                connection.send(("stopped", None))
                return
            else:
                raise ValueError(f"unknown persistent-worker command: {command!r}")
    except EOFError:
        return
    except BaseException as exc:  # noqa: BLE001 - report startup/system exits to the parent
        try:
            connection.send(
                (
                    "error",
                    type(exc).__name__,
                    str(exc),
                    traceback.format_exc(),
                )
            )
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if worker is not None:
            close = getattr(worker, "close", None)
            if close is not None:
                try:
                    close()
                except BaseException:  # noqa: BLE001,S110 - best-effort child cleanup
                    pass
        connection.close()


class PersistentProcessPool:
    """One addressable, persistent spawned process per rollout worker.

    Unlike ``multiprocessing.Pool``, worker identity is stable: assignment ``i`` always goes to
    process ``i``.  That lets each worker retain its Showdown subprocess, actor wrapper, opponent
    cache, and deterministic tag/seed stream across collection chunks.
    """

    def __init__(
        self,
        worker_count: int,
        worker_factory: Callable[..., Any],
        *factory_args: Any,
        start_method: str = "spawn",
        timeout: float = 900.0,
    ) -> None:
        if worker_count < 1:
            raise ValueError("worker_count must be >= 1")
        if timeout <= 0:
            raise ValueError("worker timeout must be positive")
        self.worker_count = int(worker_count)
        self.timeout = float(timeout)
        self._closed = False
        ctx = mp.get_context(start_method)
        self._connections: list[Connection] = []
        self._processes: list[mp.Process] = []
        for worker_id in range(self.worker_count):
            parent, child = ctx.Pipe(duplex=True)
            process = ctx.Process(
                target=_process_worker_main,
                args=(child, worker_id, worker_factory, tuple(factory_args)),
                name=f"fog-direct-rollout-{worker_id}",
            )
            process.start()
            child.close()
            self._connections.append(parent)
            self._processes.append(process)
        try:
            self._receive_all(expected="ready", timeout=self.timeout)
        except BaseException:
            self.close(force=True)
            raise

    @property
    def pids(self) -> tuple[int | None, ...]:
        return tuple(process.pid for process in self._processes)

    @property
    def alive(self) -> tuple[bool, ...]:
        return tuple(process.is_alive() for process in self._processes)

    def map(self, payloads: list[Any] | tuple[Any, ...]) -> list[Any]:
        """Run one payload on each stable worker and return results in worker-id order."""
        if self._closed:
            raise RuntimeError("persistent worker pool is closed")
        if len(payloads) != self.worker_count:
            raise ValueError(
                f"expected {self.worker_count} worker payloads, got {len(payloads)}"
            )
        try:
            for connection, payload in zip(self._connections, payloads, strict=True):
                connection.send(("run", payload))
            return self._receive_all(expected="ok", timeout=self.timeout)
        except BaseException:
            self.close(force=True)
            raise

    def _receive_all(self, *, expected: str, timeout: float) -> list[Any]:
        pending = set(range(self.worker_count))
        results: list[Any] = [None] * self.worker_count
        by_connection = {
            connection: worker_id
            for worker_id, connection in enumerate(self._connections)
        }
        deadline = time.monotonic() + timeout
        while pending:
            for worker_id in tuple(pending):
                process = self._processes[worker_id]
                if not process.is_alive() and not self._connections[worker_id].poll():
                    raise WorkerProcessError(
                        f"rollout worker {worker_id} exited with code {process.exitcode}"
                    )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise WorkerProcessError(
                    f"timed out after {timeout:.1f}s waiting for rollout workers "
                    f"{sorted(pending)}"
                )
            ready = wait(
                [self._connections[worker_id] for worker_id in pending],
                timeout=min(0.25, remaining),
            )
            for connection in ready:
                worker_id = by_connection[connection]
                message = connection.recv()
                status, *payload = message
                if status == "error":
                    exc_type, detail, remote_traceback = payload
                    raise WorkerProcessError(
                        f"rollout worker {worker_id} raised {exc_type}: {detail}\n"
                        f"{remote_traceback}"
                    )
                if status != expected:
                    raise WorkerProcessError(
                        f"rollout worker {worker_id} returned {status!r}; expected {expected!r}"
                    )
                results[worker_id] = payload[0] if payload else None
                pending.remove(worker_id)
        return results

    def close(self, *, force: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        if not force:
            for process, connection in zip(
                self._processes, self._connections, strict=True
            ):
                if process.is_alive():
                    try:
                        connection.send(("stop", None))
                    except (BrokenPipeError, EOFError, OSError):
                        pass
        deadline = time.monotonic() + min(10.0, self.timeout)
        for process in self._processes:
            process.join(timeout=max(0.0, deadline - time.monotonic()))
        for process in self._processes:
            if process.is_alive():
                process.terminate()
        for process in self._processes:
            process.join(timeout=5.0)
        for connection in self._connections:
            connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback_) -> None:
        self.close(force=exc_type is not None)
