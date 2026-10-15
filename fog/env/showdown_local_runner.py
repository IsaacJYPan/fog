"""Synchronous games on the direct, in-memory Pokemon Showdown backend."""

from __future__ import annotations

import random
import threading
from dataclasses import dataclass
from typing import Any

from fog.env.showdown_direct import ShowdownDirectClient
from fog.env.showdown_local import LocalShowdownBattle

_SCRIPTED_RNG_LOCK = threading.Lock()


class LocalOrderFactory:
    """The two order-formatting methods FogPlayer needs, without constructing a network client."""

    @staticmethod
    def create_order(order, **kwargs):
        from poke_env.player import Player

        return Player.create_order(order, **kwargs)

    @staticmethod
    def choose_default_move():
        from poke_env.player import Player

        return Player.choose_default_move()


@dataclass
class LocalBattleResult:
    winner: str | None
    turns: int
    p1_won: bool | None
    truncated: bool = False


def bind_fog_player(adapter: Any) -> Any:
    """Bind FogPlayer's action mapper to local order formatting instead of a poke-env Player."""
    adapter._player_ref = LocalOrderFactory()
    return adapter


def _teampreview(controller: Any, battle: Any, rng: random.Random) -> str:
    if hasattr(controller, "choose_teampreview"):
        return str(controller.choose_teampreview(battle))
    members = list(range(1, len(battle.team) + 1))
    rng.shuffle(members)
    return "/team " + "".join(str(member) for member in members)


def _scripted_order(name: str, battle: Any, rng: random.Random):
    from poke_env.player import MaxBasePowerPlayer, Player, SimpleHeuristicsPlayer

    if name == "default":
        return Player.choose_default_move()
    if name == "random":
        orders = battle.valid_orders
        return orders[rng.randrange(len(orders))] if orders else Player.choose_default_move()
    # The poke-env baselines use Python's module-global RNG in fallback cases. Isolate that call
    # behind this runner's seeded stream so direct-backend runs remain reproducible and do not
    # perturb unrelated Python randomness.
    # These poke-env helpers use Python's module-global RNG in fallback cases.  Direct rollout
    # workers are threads, so save/seed/call/restore must be one atomic region.
    with _SCRIPTED_RNG_LOCK:
        global_state = random.getstate()
        random.seed(rng.getrandbits(128))
        try:
            if name == "maxdamage":
                return MaxBasePowerPlayer.choose_singles_move(battle)
            if name == "simpleheuristics":
                return SimpleHeuristicsPlayer.choose_singles_move(battle)[0]
        finally:
            random.setstate(global_state)
    raise ValueError(f"unknown local scripted controller: {name!r}")


def _choice(controller: Any, battle: Any, request: dict[str, Any] | None, rng: random.Random):
    if request is None or request.get("wait"):
        return None
    if request.get("teamPreview"):
        return _teampreview(controller, battle, rng)
    if hasattr(controller, "act"):
        order = controller.act(battle)
    elif isinstance(controller, str):
        order = _scripted_order(controller.lower(), battle, rng)
    elif callable(controller):
        order = controller(battle)
    else:
        raise TypeError("local controller must be FogPlayer-like, a scripted name, or callable")

    # poke-env's scripted baselines and frozen Player-like controllers bypass FogPlayer's action
    # mapper. Normalize their Pokemon switch orders here too: numeric-leading nicknames otherwise
    # reach Showdown as enormous party-slot numbers (for example nickname "6666" -> slot 6666).
    order_target = getattr(order, "order", None)
    if order_target is not None:
        from fog.env.actions import _request_switch_position, same_pokemon

        if any(
            same_pokemon(order_target, offered)
            for offered in (getattr(battle, "available_switches", None) or [])
        ):
            request_position = _request_switch_position(battle, order_target)
            if request_position is None:
                raise AssertionError(
                    "controller switch target could not be mapped to an authoritative request slot"
                )
            return f"/choose switch {request_position}"
    return getattr(order, "message", str(order))


def play_local_battle(
    client: ShowdownDirectClient,
    options: dict[str, Any],
    p1_controller: Any,
    p2_controller: Any,
    *,
    seed: int = 0,
    battle_tag: str = "battle-local-1",
    max_choice_rounds: int = 1_000,
    max_turns: int | None = None,
    initial_battle: LocalShowdownBattle | None = None,
) -> LocalBattleResult:
    """Play to completion, optionally continuing a caller-prepared training prefix.

    A supplied battle must already have both censored views and controller
    histories initialized. The runner owns closing it, including on failure.
    """
    rng = random.Random(seed)
    if initial_battle is not None and initial_battle.client is not client:
        raise ValueError("A resumed battle must belong to the supplied Showdown client")
    local = initial_battle or LocalShowdownBattle.create(client, options, battle_tag=battle_tag)
    try:
        truncated = False
        for _ in range(max_choice_rounds):
            if local.ended:
                break
            if max_turns is not None and int(local.p1.turn) >= int(max_turns):
                truncated = True
                break
            controllers = (p1_controller, p2_controller)
            views = (local.p1, local.p2)
            choices = [
                _choice(p1_controller, local.p1, local.update.requests["p1"], rng),
                _choice(p2_controller, local.p2, local.update.requests["p2"], rng),
            ]
            if choices == [None, None]:
                raise RuntimeError("direct Showdown battle is live but neither side owes a choice")
            for retry in range(4):
                update = local.step(choices[0], choices[1], include_state=False)
                if not update.choice_errors:
                    break
                for index, role in enumerate(("p1", "p2")):
                    if role not in update.choice_errors:
                        continue
                    discard = getattr(
                        controllers[index], "discard_last_uncommitted_decision", None
                    )
                    if discard is not None:
                        discard(views[index])
                    choices[index] = _choice(
                        controllers[index],
                        views[index],
                        update.requests[role],
                        rng,
                    )
                    if choices[index] is None:
                        raise RuntimeError(
                            f"{role} received an unavailable-choice retry without a new request"
                        )
            else:
                raise RuntimeError(
                    "direct Showdown battle exceeded 4 unavailable-choice retries in one round"
                )
        else:
            raise RuntimeError(
                f"direct Showdown battle exceeded max_choice_rounds={max_choice_rounds}"
            )

        for controller, view in ((p1_controller, local.p1), (p2_controller, local.p2)):
            finished = getattr(controller, "_battle_finished", None)
            if finished is not None:
                finished(view)
        return LocalBattleResult(
            winner=None if truncated else local.winner,
            turns=int(local.p1.turn),
            p1_won=None if truncated else local.p1.won,
            truncated=truncated,
        )
    finally:
        local.close()
