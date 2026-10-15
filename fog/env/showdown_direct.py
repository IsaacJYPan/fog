"""Persistent local Pokemon Showdown client with exact chance enumeration.

The Node worker imports the pinned simulator directly. Communication uses local stdin/stdout pipes;
there is no Showdown server, account, HTTP request, or WebSocket. Exact enumeration is controlled
replay, not seed discovery: every finite random request is expanded and carries a rational weight.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SHOWDOWN_ROOT = PROJECT_ROOT / "server" / "pokemon-showdown"
WORKER_PATH = Path(__file__).with_name("showdown_worker.js")


class ShowdownWorkerError(RuntimeError):
    """The local worker rejected a request or terminated unexpectedly."""


@dataclass(frozen=True)
class ExactProbability:
    """One exact rational probability plus its convenience floating-point form."""

    numerator: int
    denominator: int
    decimal: float

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> ExactProbability:
        return cls(
            numerator=int(value["numerator"]),
            denominator=int(value["denominator"]),
            decimal=float(value["decimal"]),
        )


@dataclass(frozen=True)
class ShowdownChanceBranch:
    """One completed Showdown leaf and its exact chance weight."""

    probability: ExactProbability
    raw_probability: ExactProbability
    state: dict[str, Any] | None
    ended: bool
    winner: str | None
    request_state: str
    turn: int
    public_log: tuple[str, ...]
    player_logs: dict[str, tuple[str, ...]]
    requests: dict[str, dict[str, Any] | None]


@dataclass(frozen=True)
class ShowdownEnumeration:
    """Exact leaves from one committed action pair."""

    branches: tuple[ShowdownChanceBranch, ...]
    kept_mass: ExactProbability
    pruned_mass: ExactProbability
    replay_runs: int
    branch_points: int
    merged_prefixes: int
    chance_depth: int
    max_pending: int
    branch_key_counts: dict[str, int]
    exact: bool
    sampled: bool


@dataclass(frozen=True)
class ShowdownSessionUpdate:
    """New per-player protocol messages and requests from one in-memory battle."""

    messages: dict[str, tuple[str, ...]]
    public_messages: tuple[str, ...]
    requests: dict[str, dict[str, Any] | None]
    state: dict[str, Any] | None
    request_state: str
    turn: int
    ended: bool
    winner: str | None
    choice_errors: dict[str, str]


class ShowdownDirectClient:
    """Synchronous client for one long-running local Node worker.

    One client owns one process and serializes its requests with a lock. Data-generation workers
    should each own a client, which keeps battle simulation parallel without interleaving protocol
    lines from different Python threads.
    """

    def __init__(
        self,
        *,
        showdown_root: str | Path = DEFAULT_SHOWDOWN_ROOT,
        node_binary: str = "node",
    ) -> None:
        self.showdown_root = Path(showdown_root).resolve()
        self.node_binary = (
            os.environ.get("FOG_NODE_BINARY", node_binary)
            if node_binary == "node"
            else node_binary
        )
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()
        self._next_id = 1

    def start(self) -> ShowdownDirectClient:
        if self._process is not None:
            return self
        if not (self.showdown_root / "dist" / "sim" / "index.js").is_file():
            raise FileNotFoundError(
                f"built Pokemon Showdown simulator not found under {self.showdown_root}"
            )
        try:
            self._process = subprocess.Popen(
                [
                    self.node_binary,
                    str(WORKER_PATH),
                    "--showdown-root",
                    str(self.showdown_root),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,
                cwd=PROJECT_ROOT,
            )
            self.ping()
        except BaseException:
            self.close()
            raise
        return self

    def close(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        if process.stdin is not None:
            process.stdin.close()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _request(self, op: str, **payload: Any) -> dict[str, Any]:
        if self._process is None:
            self.start()
        process = self._process
        assert process is not None and process.stdin is not None and process.stdout is not None
        with self._lock:
            request_id = self._next_id
            self._next_id += 1
            message = {"id": request_id, "op": op, **payload}
            try:
                process.stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
                process.stdin.flush()
                line = process.stdout.readline()
            except (BrokenPipeError, OSError) as exc:
                raise ShowdownWorkerError(self._terminated_message(process)) from exc
            if not line:
                raise ShowdownWorkerError(self._terminated_message(process))
            response = json.loads(line)
            if response.get("id") != request_id:
                raise ShowdownWorkerError(
                    f"worker response id {response.get('id')} did not match request {request_id}"
                )
            if not response.get("ok"):
                detail = response.get("stack") or response.get("error") or "unknown worker error"
                raise ShowdownWorkerError(detail)
            return response["result"]

    @staticmethod
    def _terminated_message(process: subprocess.Popen[str]) -> str:
        stderr = ""
        if process.poll() is not None and process.stderr is not None:
            stderr = process.stderr.read().strip()
        suffix = f": {stderr}" if stderr else ""
        return f"Showdown worker terminated with exit code {process.poll()}{suffix}"

    def ping(self) -> dict[str, Any]:
        return self._request("ping")

    def create_state(
        self,
        options: dict[str, Any],
        *,
        choices: list[list[str | None]] | None = None,
    ) -> dict[str, Any]:
        """Create a battle and optionally advance complete choice rounds to a forkable root."""
        return self._request("create", options=options, choices=choices or [])

    def enumerate(
        self,
        state: dict[str, Any] | str,
        choices: tuple[str | None, str | None] | list[str | None],
        *,
        min_probability: float = 0.0,
        max_paths: int = 100_000,
        max_branches: int | None = None,
        include_state: bool = True,
    ) -> ShowdownEnumeration:
        """Resolve one action pair by exact finite branching.

        ``min_probability`` and ``max_branches`` are explicit pruning controls. Defaults perform no
        probability pruning. Hitting ``max_paths`` raises; it never falls back to random samples.
        """
        result = self._request(
            "enumerate",
            state=state,
            choices=list(choices),
            min_probability=min_probability,
            max_paths=max_paths,
            max_branches=max_branches,
            include_state=include_state,
        )
        branches = tuple(
            ShowdownChanceBranch(
                probability=ExactProbability.from_json(branch["probability"]),
                raw_probability=ExactProbability.from_json(branch["raw_probability"]),
                state=branch.get("state"),
                ended=bool(branch["ended"]),
                winner=branch.get("winner"),
                request_state=str(branch["request_state"]),
                turn=int(branch["turn"]),
                public_log=tuple(branch["public_log"]),
                player_logs={
                    role: tuple(lines) for role, lines in branch["player_logs"].items()
                },
                requests=branch["requests"],
            )
            for branch in result["branches"]
        )
        return ShowdownEnumeration(
            branches=branches,
            kept_mass=ExactProbability.from_json(result["kept_mass"]),
            pruned_mass=ExactProbability.from_json(result["pruned_mass"]),
            replay_runs=int(result["replay_runs"]),
            branch_points=int(result["branch_points"]),
            merged_prefixes=int(result.get("merged_prefixes", 0)),
            chance_depth=int(result.get("chance_depth", 0)),
            max_pending=int(result.get("max_pending", 0)),
            branch_key_counts={
                str(key): int(value)
                for key, value in result.get("branch_key_counts", {}).items()
            },
            exact=bool(result["exact"]),
            sampled=bool(result["sampled"]),
        )

    def sample(
        self,
        state: dict[str, Any] | str,
        choices: tuple[str | None, str | None] | list[str | None],
        *,
        seed: str,
        include_state: bool = True,
    ) -> dict[str, Any]:
        """Resolve one ordinary seeded Showdown path for equivalence/frequency validation."""
        return self._request(
            "sample",
            state=state,
            choices=list(choices),
            seed=seed,
            include_state=include_state,
        )

    @staticmethod
    def _session_update(value: dict[str, Any]) -> ShowdownSessionUpdate:
        return ShowdownSessionUpdate(
            messages={role: tuple(lines) for role, lines in value["messages"].items()},
            public_messages=tuple(value["public_messages"]),
            requests=value["requests"],
            state=value.get("state"),
            request_state=str(value["request_state"]),
            turn=int(value["turn"]),
            ended=bool(value["ended"]),
            winner=value.get("winner"),
            choice_errors={
                str(role): str(error)
                for role, error in value.get("choice_errors", {}).items()
            },
        )

    def create_session(
        self,
        options: dict[str, Any],
        *,
        session_id: str | None = None,
        include_state: bool = False,
    ) -> tuple[str, ShowdownSessionUpdate]:
        """Create a persistent in-memory battle; no HTTP server or socket is involved."""
        result = self._request(
            "session_create",
            options=options,
            session_id=session_id,
            include_state=include_state,
        )
        return str(result["session_id"]), self._session_update(result)

    def step_session(
        self,
        session_id: str,
        choices: tuple[str | None, str | None] | list[str | None],
        *,
        include_state: bool = False,
    ) -> ShowdownSessionUpdate:
        """Commit the currently owed choices for both sides and return only new messages."""
        result = self._request(
            "session_step",
            session_id=session_id,
            choices=list(choices),
            include_state=include_state,
        )
        return self._session_update(result)

    def get_session(
        self, session_id: str, *, include_state: bool = True
    ) -> ShowdownSessionUpdate:
        """Read a live session without advancing it."""
        result = self._request(
            "session_get", session_id=session_id, include_state=include_state
        )
        return self._session_update(result)

    def close_session(self, session_id: str) -> bool:
        return bool(self._request("session_close", session_id=session_id)["closed"])

    def reseed_session(self, session_id: str, *, seed: str) -> None:
        """Preserve a training prefix and requests, drawing fresh chance thereafter.

        This coordinator operation supplies no observations and consumes no log messages.
        It must never use the historical future seed as a model feature or training target.
        """
        result = self._request("session_reseed", session_id=session_id, seed=seed)
        if result != {"reseeded": True}:
            raise ShowdownWorkerError("Showdown did not confirm the training reseed")

    def reconstruct_session_strict(
        self,
        session_id: str,
        *,
        p1_team: str | list[dict[str, Any]],
        p2_team: str | list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Diagnostic: replay candidate teams with the identical semantic chance transcript.

        This is deliberately stricter than public compatibility. It can reject a genuinely possible
        candidate when a hidden speed/item difference adds or removes a publicly irrelevant random
        ordering call. Therefore its failures MUST NOT eliminate belief support, and it is not yet
        the PB-grid substitution bridge. Its purpose is differential validation of candidates whose
        random-call structure is identical. No serialized-state field surgery is used.
        """
        return self._request(
            "session_reconstruct_strict",
            session_id=session_id,
            p1_team=p1_team,
            p2_team=p2_team,
        )
