"""Poke-env-compatible views over a direct in-memory Pokemon Showdown battle.

Pokemon Showdown remains the rule authority, but no server, login, HTTP request, or WebSocket is
started. The Node worker returns the same player-censored protocol lines and private choice request
that a server would send. Feeding those into poke-env's battle parser preserves the project's one
observation/encoding path instead of introducing a second interpretation of battle state.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Self

from fog.env.showdown_direct import (
    ShowdownDirectClient,
    ShowdownSessionUpdate,
)

# Transport-level messages poke-env's Player normally removes before calling Battle.parse_message.
_TRANSPORT_MESSAGES = {"t:", "expire", "uhtmlchange"}


def _parse_protocol_line(line: str) -> list[str]:
    if not line.startswith("|"):
        return [line]
    return line.split("|")


@dataclass
class LocalShowdownBattle:
    """One in-memory Showdown battle plus the two censored poke-env views."""

    client: ShowdownDirectClient
    session_id: str
    views: dict[str, Any]
    update: ShowdownSessionUpdate

    @classmethod
    def create(
        cls,
        client: ShowdownDirectClient,
        options: dict[str, Any],
        *,
        battle_tag: str = "battle-local-1",
    ) -> LocalShowdownBattle:
        """Create a battle and initialize both player-censored views from its protocol output."""
        try:
            from poke_env.battle import Battle
        except ImportError as exc:  # pragma: no cover - project dependency, import guard for tools
            raise RuntimeError("poke-env is required to construct censored Showdown views") from exc

        names = {
            "p1": str((options.get("p1") or {}).get("name") or "Player 1"),
            "p2": str((options.get("p2") or {}).get("name") or "Player 2"),
        }
        logger = logging.getLogger(__name__)
        views = {
            role: Battle(battle_tag, names[role], logger, gen=9)
            for role in ("p1", "p2")
        }
        session_id, update = client.create_session(options, include_state=True)
        local = cls(client=client, session_id=session_id, views=views, update=update)
        try:
            local._apply(update)
        except BaseException:
            client.close_session(session_id)
            raise
        return local

    @property
    def p1(self):
        return self.views["p1"]

    @property
    def p2(self):
        return self.views["p2"]

    @property
    def ended(self) -> bool:
        return self.update.ended

    @property
    def winner(self) -> str | None:
        return self.update.winner

    @property
    def state(self) -> dict[str, Any]:
        if self.update.state is None:
            refreshed = self.client.get_session(self.session_id, include_state=True)
            self.update = refreshed
        assert self.update.state is not None
        return self.update.state

    def _apply(self, update: ShowdownSessionUpdate) -> None:
        for role, view in self.views.items():
            self._apply_view_update(view, update.messages[role], update.requests.get(role))
            # Direct-search adapters need the authoritative forkable session, but ordinary
            # Observation construction must remain censored to this role's poke-env view.  The
            # back-reference is transport context, not a model input.
            view._fog_local_battle = self
            view._fog_local_role = role
        self.update = update

    @staticmethod
    def _apply_view_update(
        view: Any,
        lines: tuple[str, ...],
        request: dict[str, Any] | None,
    ) -> None:
        """Apply one censored protocol delta to a poke-env battle view."""
        for line in lines:
            split = _parse_protocol_line(line)
            event = split[1] if len(split) > 1 else ""
            if event in _TRANSPORT_MESSAGES:
                continue
            if event == "win":
                view.won_by(split[2])
            elif event == "tie":
                view.tied()
            else:
                view.parse_message(split)
        if request is not None:
            view.parse_request(request)

    def branch_view(self, role: str, branch: Any) -> Any:
        """Return a detached censored view advanced through one exact chance branch.

        The live session and its views are never mutated.  This is the direct-Showdown analogue of
        an engine search fork and is the only supported way for a value function to inspect an
        enumerated branch.
        """
        if role not in self.views:
            raise ValueError(f"unknown battle role: {role!r}")
        # The view carries a transport back-reference to ``self``; pre-seed deepcopy's memo so it
        # does not try to copy the client's thread lock or recursively copy both live views.
        view = deepcopy(self.views[role], {id(self): None})
        self._apply_view_update(view, branch.player_logs[role], branch.requests.get(role))
        # A detached branch must never recursively expose the live authoritative session.
        view._fog_local_battle = None
        view._fog_local_role = role
        return view

    def step(
        self,
        p1_choice: str | None,
        p2_choice: str | None,
        *,
        include_state: bool = True,
    ) -> ShowdownSessionUpdate:
        """Commit both currently owed choices and update both censored views."""
        update = self.client.step_session(
            self.session_id,
            (p1_choice, p2_choice),
            include_state=include_state,
        )
        self._apply(update)
        return update

    def close(self) -> None:
        self.client.close_session(self.session_id)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
