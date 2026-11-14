"""Many concurrent direct-Showdown battles in one process with one batched forward per wave.

`play_local_battle` plays one battle at a time and runs one batch-1 forward per decision. This
runner keeps up to `batch_size` battles open on one Node worker (the worker already hosts any
number of sessions, keyed by session id) and advances them in waves:

1. every battle that owes choices prepares its agent decisions (`FogPlayer.prepare_decision`:
   per-battle tracker and ledger updates, observation encoding; no model, no randomness);
2. the prepared observations are collated per model and forwarded once;
3. each battle, seat p1 then p2, samples and records (`FogPlayer.complete_decision`) or, for
   scripted controllers, team preview and controllers with one-row hooks, chooses as before;
4. each battle commits its choices to Showdown; rejected choices are retried in the next wave.

Reproducibility does not depend on the batch size or on which battles share a wave. Each battle
owns the torch and numpy global random states that the sequential rollout worker would have
seeded for it (`torch.manual_seed(seed)`, `np.random.seed(seed)`), and those are swapped in
whenever the runner works on that battle. Showdown itself is seeded per session. With batch
size 1 this reproduces the sequential path exactly; with more battles per forward only
floating-point reduction order in the forward can differ.
"""

from __future__ import annotations

import random
import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from fog.env.showdown_direct import ShowdownDirectClient
from fog.env.showdown_local import LocalShowdownBattle
from fog.env.showdown_local_runner import LocalBattleResult, _choice, order_message

_ROLES = ("p1", "p2")
# Hooks that receive the whole agent output or search with a one-row batch. Controllers that set
# any of them keep the sequential `act` path for their decisions.
_ONE_ROW_HOOKS = ("search", "action_override", "supervised_label_provider", "decision_observer")


@dataclass
class BattleSpec:
    """One battle to play: Showdown options, a unique tag, and the game seed.

    `seed` is used exactly as the sequential rollout worker uses `WorkerGame.seed`: torch is
    seeded with `seed % (2**63 - 1)`, numpy with `seed % 2**32` and the local `random.Random`
    with `seed & (2**31 - 1)`. `context` is passed to the runner's `enter` hook.
    """

    options: dict[str, Any]
    battle_tag: str
    seed: int
    context: Any = None


@dataclass
class _Live:
    index: int
    spec: BattleSpec
    local: LocalShowdownBattle
    rng: random.Random
    torch_state: Any
    np_state: Any
    choices: list[str | None] = field(default_factory=lambda: [None, None])
    owed: list[bool] = field(default_factory=lambda: [False, False])
    pending: list[Any] = field(default_factory=lambda: [None, None])
    new_round: bool = True
    rounds: int = 0
    attempts: int = 0


def _batched_agent(controller: Any) -> Any | None:
    """The controller's model if its decisions can join a batched forward, else None."""
    if not hasattr(controller, "prepare_decision") or getattr(controller, "agent", None) is None:
        return None
    if any(getattr(controller, name, None) is not None for name in _ONE_ROW_HOOKS):
        return None
    return controller.agent


class BatchedShowdownRunner:
    """Play battles between two controllers, up to `batch_size` at a time.

    Controllers are what `play_local_battle` accepts. Two seats driven by the same model must use
    two `FogPlayer` objects that share the agent (per-battle state is keyed by battle tag, which
    both seats of one battle share); their decisions are forwarded together.
    """

    def __init__(
        self,
        client: ShowdownDirectClient,
        p1_controller: Any,
        p2_controller: Any,
        *,
        batch_size: int,
        max_choice_rounds: int = 1_000,
        max_turns: int | None = None,
        enter: Callable[[BattleSpec], None] | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if p1_controller is p2_controller and hasattr(p1_controller, "prepare_decision"):
            raise ValueError("use two FogPlayer objects sharing one agent for both seats")
        self.client = client
        self.controllers = (p1_controller, p2_controller)
        self.batch_size = int(batch_size)
        self.max_choice_rounds = int(max_choice_rounds)
        self.max_turns = max_turns
        # Called with the battle's spec before any controller work on that battle, so providers
        # that read per-game state (teams, truth) see the right game.
        self.enter = enter
        self.stats: dict[str, Any] = {}

    # ------------------------------------------------------------------------------ random state
    @contextmanager
    def _battle(self, live: _Live, *, random_state: bool = True) -> Iterator[None]:
        """Work on one battle: its providers see it, and it draws from its own random states.

        `random_state=False` skips swapping the global torch and numpy generators; it is for
        phases that draw no random numbers (decision preparation).
        """
        import torch

        if not random_state:
            if self.enter is not None:
                self.enter(live.spec)
            yield
            return
        torch.set_rng_state(live.torch_state)
        np.random.set_state(live.np_state)
        if self.enter is not None:
            self.enter(live.spec)
        try:
            yield
        finally:
            live.torch_state = torch.get_rng_state()
            live.np_state = np.random.get_state()

    def _start(self, index: int, spec: BattleSpec) -> _Live:
        import torch

        generator = torch.Generator()
        generator.manual_seed(spec.seed % (2**63 - 1))
        live = _Live(
            index=index,
            spec=spec,
            local=LocalShowdownBattle.create(self.client, spec.options, battle_tag=spec.battle_tag),
            rng=random.Random(spec.seed & (2**31 - 1)),
            torch_state=generator.get_state(),
            np_state=np.random.RandomState(spec.seed % (2**32)).get_state(),
        )
        # Reserve rollout game ids at battle start so ids follow spec order whatever the
        # interleaving; the sequential path creates them in the same order at each battle's
        # first recorded preview or decision.
        with self._battle(live):
            for controller in self.controllers:
                if (
                    getattr(controller, "_buffer", None) is not None
                    and getattr(controller, "agent", None) is not None
                ):
                    controller._game_id(spec.battle_tag)
        return live

    # ------------------------------------------------------------------------------------- play
    def play(self, specs: Sequence[BattleSpec]) -> list[LocalBattleResult]:
        """Play every spec; results are returned in spec order."""
        import torch

        tags = [spec.battle_tag for spec in specs]
        if len(set(tags)) != len(tags):
            raise ValueError("battle tags must be unique")
        self.stats = {
            "waves": 0,
            "forwards": 0,
            "forward_rows": 0,
            "move_choices": [0, 0],
            "seconds": {"start": 0.0, "prepare": 0.0, "forward": 0.0, "choose": 0.0,
                        "step": 0.0, "finish": 0.0},
        }
        seconds = self.stats["seconds"]
        results: list[LocalBattleResult | None] = [None] * len(specs)
        queue = deque(enumerate(specs))
        live: list[_Live] = []
        saved_torch, saved_np = torch.get_rng_state(), np.random.get_state()
        try:
            with torch.inference_mode():
                while queue or live:
                    t0 = time.perf_counter()
                    while queue and len(live) < self.batch_size:
                        live.append(self._start(*queue.popleft()))
                    seconds["start"] += time.perf_counter() - t0
                    still: list[_Live] = []
                    for battle in live:
                        result = self._begin_round(battle)
                        if result is None:
                            still.append(battle)
                        else:
                            results[battle.index] = result
                    live = still
                    if live:
                        self._wave(live)
        except BaseException:
            for battle in live:
                try:
                    battle.local.close()
                except Exception:  # noqa: BLE001 - keep the original error
                    pass
            raise
        finally:
            torch.set_rng_state(saved_torch)
            np.random.set_state(saved_np)
        return [result for result in results if result is not None]

    def _begin_round(self, live: _Live) -> LocalBattleResult | None:
        """Finish an ended or truncated battle, or mark the seats that owe a fresh choice."""
        if not live.new_round:
            return None
        local = live.local
        truncated = (
            not local.ended
            and self.max_turns is not None
            and int(local.p1.turn) >= int(self.max_turns)
        )
        if local.ended or truncated:
            t0 = time.perf_counter()
            with self._battle(live):
                for controller, view in zip(self.controllers, (local.p1, local.p2), strict=True):
                    finished = getattr(controller, "_battle_finished", None)
                    if finished is not None:
                        finished(view)
            local.close()
            self.stats["seconds"]["finish"] += time.perf_counter() - t0
            return LocalBattleResult(
                winner=None if truncated else local.winner,
                turns=int(local.p1.turn),
                p1_won=None if truncated else local.p1.won,
                truncated=truncated,
            )
        if live.rounds >= self.max_choice_rounds:
            raise RuntimeError(
                f"direct Showdown battle exceeded max_choice_rounds={self.max_choice_rounds}"
            )
        live.rounds += 1
        live.attempts = 0
        live.new_round = False
        live.choices = [None, None]
        live.owed = [True, True]
        return None

    def _wave(self, live: list[_Live]) -> None:
        import torch

        from fog.encoding.batch import collate_observations
        from fog.env.player import _agent_device

        seconds = self.stats["seconds"]
        self.stats["waves"] += 1
        t0 = time.perf_counter()
        groups: dict[int, list[tuple[_Live, int]]] = {}
        for battle in live:
            battle.pending = [None, None]
            with self._battle(battle, random_state=False):
                for seat, role in enumerate(_ROLES):
                    if not battle.owed[seat]:
                        continue
                    request = battle.local.update.requests.get(role)
                    if request is None or request.get("wait") or request.get("teamPreview"):
                        continue
                    agent = _batched_agent(self.controllers[seat])
                    if agent is None:
                        continue
                    view = battle.local.views[role]
                    battle.pending[seat] = self.controllers[seat].prepare_decision(view)
                    groups.setdefault(id(agent), []).append((battle, seat))
        t1 = time.perf_counter()
        seconds["prepare"] += t1 - t0

        outputs: dict[tuple[int, int], tuple[Any, int]] = {}
        for items in groups.values():
            adapters = [self.controllers[seat] for _, seat in items]
            agent = adapters[0].agent
            device = _agent_device(agent)
            pendings = [battle.pending[seat] for battle, seat in items]
            batch = collate_observations([pending.obs for pending in pendings], device)
            truths = [pending.oracle_truth for pending in pendings]
            if any(truth is not None for truth in truths):
                if any(truth is None for truth in truths):
                    raise ValueError("oracle truth must be present for every row of a batch")
                batch["opp_truth_tokens"] = torch.as_tensor(
                    np.stack(truths), dtype=torch.long, device=device
                )
            with torch.no_grad():
                if any(adapter._act_needs_belief() for adapter in adapters):
                    out = agent(batch)
                else:
                    out = agent(batch, need_belief=False)
            if device.type != "cpu" and out.belief is None and out.belief_chain is None:
                # Sampling and recording read one row at a time. Bring the two tensors they read
                # to the CPU in one transfer, and sample with the per-battle CPU generators.
                from fog.models.agent import AgentOutput

                out = AgentOutput(
                    log_probs=out.log_probs.float().cpu(), value=out.value.float().cpu()
                )
            self.stats["forwards"] += 1
            self.stats["forward_rows"] += len(items)
            for row, (battle, seat) in enumerate(items):
                outputs[(id(battle), seat)] = (out, row)
        t2 = time.perf_counter()
        seconds["forward"] += t2 - t1

        for battle in live:
            with self._battle(battle):
                for seat, role in enumerate(_ROLES):
                    if not battle.owed[seat]:
                        continue
                    controller = self.controllers[seat]
                    view = battle.local.views[role]
                    request = battle.local.update.requests.get(role)
                    pending = battle.pending[seat]
                    if pending is not None:
                        out, row = outputs[(id(battle), seat)]
                        order = controller.complete_decision(pending, out, row)
                        choice = order_message(order, view)
                    else:
                        choice = _choice(controller, view, request, battle.rng)
                    if choice is not None and not (request or {}).get("teamPreview"):
                        self.stats["move_choices"][seat] += 1
                    if choice is None and battle.attempts > 0:
                        raise RuntimeError(
                            f"{role} received an unavailable-choice retry without a new request"
                        )
                    battle.choices[seat] = choice
                    battle.owed[seat] = False
                battle.pending = [None, None]
        t3 = time.perf_counter()
        seconds["choose"] += t3 - t2

        for battle in live:
            if battle.choices == [None, None]:
                raise RuntimeError("direct Showdown battle is live but neither side owes a choice")
            update = battle.local.step(battle.choices[0], battle.choices[1], include_state=False)
            battle.attempts += 1
            if not update.choice_errors:
                battle.new_round = True
                continue
            if battle.attempts >= 4:
                raise RuntimeError(
                    "direct Showdown battle exceeded 4 unavailable-choice retries in one round"
                )
            with self._battle(battle):
                for seat, role in enumerate(_ROLES):
                    if role not in update.choice_errors:
                        continue
                    discard = getattr(
                        self.controllers[seat], "discard_last_uncommitted_decision", None
                    )
                    if discard is not None:
                        discard(battle.local.views[role])
                    battle.owed[seat] = True
        seconds["step"] += time.perf_counter() - t3


def play_local_battles(
    client: ShowdownDirectClient,
    specs: Sequence[BattleSpec],
    p1_controller: Any,
    p2_controller: Any,
    *,
    batch_size: int,
    max_choice_rounds: int = 1_000,
    max_turns: int | None = None,
    enter: Callable[[BattleSpec], None] | None = None,
) -> list[LocalBattleResult]:
    """Play `specs` with up to `batch_size` concurrent battles; results in spec order."""
    runner = BatchedShowdownRunner(
        client,
        p1_controller,
        p2_controller,
        batch_size=batch_size,
        max_choice_rounds=max_choice_rounds,
        max_turns=max_turns,
        enter=enter,
    )
    return runner.play(specs)
