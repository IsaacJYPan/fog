"""Bounded subprocess decisions with explicit failures, fallbacks and public-only payloads."""

from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import threading
import time


class DecisionProcess:
    def __init__(self, command: list[str], *, cwd: Path, log: Path, env=None, deadline=5.0):
        self.command, self.cwd, self.log, self.env = command, cwd, log, env
        self.deadline = deadline
        self.process = None
        self.responses = queue.Queue()
        self.sequence = 0

    def start(self):
        self.responses = queue.Queue()
        self.log.parent.mkdir(parents=True, exist_ok=True)
        self.stderr = self.log.open("a", encoding="utf-8")
        self.process = subprocess.Popen(
            self.command,
            cwd=self.cwd,
            env=self.env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr,
            text=True,
            encoding="utf-8",
            bufsize=1,
            start_new_session=os.name != "nt",
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )

        process, responses, stderr = self.process, self.responses, self.stderr

        def read():
            for line in process.stdout:
                try:
                    responses.put(json.loads(line))
                except json.JSONDecodeError:
                    stderr.write(line)
                    stderr.flush()
            responses.put({"error": "decision worker exited"})

        self.reader = threading.Thread(target=read, daemon=True)
        self.reader.start()
        try:
            ready = self.responses.get(timeout=120)
        except queue.Empty as error:
            self.close(force=True)
            raise TimeoutError("Decision worker initialization timed out") from error
        if ready.get("status") != "ready":
            self.close(force=True)
            raise RuntimeError(f"Decision worker failed initialization: {ready}")
        return ready

    def request(self, message: dict, *, timeout=None) -> dict:
        forbidden = {"opponent_team", "truth", "labels", "simulator_state", "submitted_teams"}

        def has_private_label(value):
            if isinstance(value, dict):
                return bool(forbidden.intersection(value)) or any(
                    has_private_label(item) for item in value.values()
                )
            if isinstance(value, (list, tuple)):
                return any(has_private_label(item) for item in value)
            return False

        if has_private_label(message):
            raise ValueError("Complete-team labels cannot enter a player decision payload")
        if self.process is None:
            self.start()
        self.sequence += 1
        request = dict(message, id=self.sequence)
        started = time.perf_counter()
        self.process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        try:
            budget = self.deadline if timeout is None else timeout
            remaining = max(0.0, budget - (time.perf_counter() - started))
            response = self.responses.get(timeout=remaining)
        except queue.Empty as error:
            self.close()
            raise TimeoutError("Decision exceeded the five-second deadline") from error
        response["elapsed_seconds"] = time.perf_counter() - started
        if response.get("id") != self.sequence or response.get("error"):
            raise RuntimeError(f"Invalid decision response: {response}")
        return response

    def close(self, *, force=False):
        if self.process is not None:
            if not force:
                if self.process.stdin:
                    self.process.stdin.close()
                try:
                    self.process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    force = True
            if force and self.process.poll() is None:
                # Foul Play creates search children. A timed-out parent alone is
                # insufficient: kill its process group/tree before restarting.
                if os.name == "nt":
                    subprocess.run(
                        ["taskkill", "/PID", str(self.process.pid), "/T", "/F"],
                        capture_output=True,
                        timeout=5,
                        check=False,
                    )
                else:
                    os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=2)
            if self.process.stdout:
                self.process.stdout.close()
            self.process = None
            self.stderr.close()
