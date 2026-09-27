"""poke-engine implementations of the determinized-search `step_fn` contract.

Two steppers, one contract (`(action_idx, world_tokens) -> Observation | float | None`), one shared
leaf-play core (`_play_one_ply`):

  * `EngineWorldStepper`   — engine self-play: forks a live `EngineBattle` (opponent truth exists;
    `fork_for_search` swaps its hidden attributes for the sampled world).
  * `ShowdownWorldStepper` — Showdown play: reconstructs an engine position from a poke-env `Battle`
    (searcher's observable view) + the sampled world (`state_recon.reconstruct_battle`), then
    clone-steps it. This is what makes search usable in the gauntlet.

Both: play the searcher's action against an opponent reply, resolve opponent-only forced
replacements, and return the searcher's CENSORED leaf `Observation` (fix-1 semantics — the leaf
belief input is root reveals + reveals the simulated ply produced, never the sampled truth), or a
terminal outcome float in {-1, 0, 1}, or None when the simulation failed.

The default opponent reply is `EngineMaxDamagePolicy` — it reads only its own side, so no engine
truth leaks through the reply. Anything acting on the raw state sees the searcher's TRUE side, which
models an opponent with perfect scouting: acceptable pessimism at 1 ply, worth remembering for
depth>1.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from fog.search.determinized import determinized_action


class EngineAgentPolicy:
    """A frozen FogAgent reply policy over an engine fork's properly censored view."""

    def __init__(self, agent: Any, *, temperature: float = 0.0) -> None:
        self.agent = agent
        self.temperature = float(temperature)
        self.agent.eval()

    def choose(self, battle: Any, role: str) -> int:
        import torch

        from fog.env.player import _sample_action

        with torch.no_grad():
            output = self.agent(engine_root_batch(self.agent, battle, role))
        return _sample_action(output.log_probs[0], temperature=self.temperature)


def _play_one_ply(fork, role: str, action_idx: int, opponent_policy, *, max_forced_steps: int):
    """Step `fork` one searcher-ply and return the leaf (Observation) / terminal float. Assumes the
    searcher is pending on `fork`; mutates only `fork` (a disposable clone)."""
    from fog.encoding.observation import Observation

    opp_role = "p2" if role == "p1" else "p1"
    pending = fork.pending_sides()
    if role not in pending:
        raise RuntimeError(f"searcher {role} is not pending (pending={pending})")
    moves = {role: fork.engine_action(role, action_idx)}
    if opp_role in pending:
        moves[opp_role] = fork.engine_action(opp_role, opponent_policy.choose(fork, opp_role))
    fork.step(moves)

    # opponent-only forced replacements (faints, pivots) resolve inside this ply; the leaf is the
    # first state where the SEARCHER owes a decision again.
    guard = 0
    while not fork.finished and role not in fork.pending_sides():
        guard += 1
        if guard > max_forced_steps:
            raise RuntimeError("forced-replacement loop exceeded max_forced_steps")
        idx = opponent_policy.choose(fork, opp_role)
        fork.step({opp_role: fork.engine_action(opp_role, idx)})

    if fork.finished:
        if fork.winner == role:
            return 1.0
        return -1.0 if fork.winner is not None else 0.0
    return Observation.from_battle(fork.view(role), tracker=fork.trackers[role])


class _BaseStepper:
    def __init__(self, role: str, opponent_policy: Any | None, seed: int, max_forced_steps: int,
                 tables) -> None:
        import random

        if opponent_policy is None:
            from fog.env.engine_scripted import EngineMaxDamagePolicy

            opponent_policy = EngineMaxDamagePolicy(tables=tables, seed=seed)
        self.role = role
        self.opp_role = "p2" if role == "p1" else "p1"
        self.opponent_policy = opponent_policy
        self.rng = random.Random(seed)
        self.max_forced_steps = int(max_forced_steps)
        self.failures = 0  # a nonzero trickle is tolerable; a flood signals a reconstruction bug
        self.failure_reasons: dict[str, int] = {}

    def __call__(self, action_idx: int, world_tokens: np.ndarray):
        try:
            fork = self._fork(np.asarray(world_tokens, dtype=np.int64))
            return _play_one_ply(fork, self.role, int(action_idx), self.opponent_policy,
                                 max_forced_steps=self.max_forced_steps)
        except Exception as exc:  # noqa: BLE001 - failed worlds are explicit search observations
            self.failures += 1
            reason = f"{type(exc).__name__}: {exc}"[:240]
            self.failure_reasons[reason] = self.failure_reasons.get(reason, 0) + 1
            return None

    def _fork(self, world_tokens: np.ndarray):  # pragma: no cover - overridden
        raise NotImplementedError


class EngineWorldStepper(_BaseStepper):
    """step_fn over a live EngineBattle (engine self-play). See module docstring."""

    def __init__(self, battle: Any, role: str = "p1", *, opponent_policy: Any | None = None,
                 seed: int = 0, max_forced_steps: int = 8) -> None:
        super().__init__(role, opponent_policy, seed, max_forced_steps, tables=battle.tables)
        self.battle = battle

    def _fork(self, world_tokens: np.ndarray):
        return self.battle.fork_for_search(self.role, world_tokens, seed=self.rng.randrange(2**31))


class ShowdownWorldStepper(_BaseStepper):
    """step_fn over a poke-env Battle (Showdown play): reconstruct the position with the sampled
    world, then clone-step. Reconstructed base battles are cached per distinct world so re-scoring
    each legal action against the same world reconstructs once, not once per action."""

    def __init__(self, battle: Any, tracker: Any, role: str = "p1", *,
                 tables=None, opponent_policy: Any | None = None, seed: int = 0,
                 max_forced_steps: int = 8) -> None:
        if tables is None:
            from fog.constants.tables import TABLES

            tables = TABLES
        super().__init__(role, opponent_policy, seed, max_forced_steps, tables=tables)
        self.battle = battle
        self.tracker = tracker
        self.tables = tables
        self._bases: dict[bytes, Any] = {}

    def _fork(self, world_tokens: np.ndarray):
        from fog.env.state_recon import reconstruct_battle

        key = world_tokens.tobytes()
        base = self._bases.get(key)
        if base is None:
            base = self._bases[key] = reconstruct_battle(
                self.battle, world_tokens, self.tables, self.tracker, role=self.role,
                seed=self.rng.randrange(2**31),
            )
        return base.clone_for_step(seed=self.rng.randrange(2**31))


# --------------------------------------------------------------------------- engine convenience wiring
def engine_root_batch(agent: Any, battle: Any, role: str = "p1") -> dict:
    """The searcher's B=1 fog input dict for `determinized_action` (same build as EngineRunner)."""
    from fog.encoding.observation import Observation
    from fog.env.engine_runner import _device, _stack_obs

    obs = Observation.from_battle(battle.view(role), tracker=battle.trackers[role])
    return _stack_obs([obs], _device(agent))


def engine_determinized_action(
    agent: Any, battle: Any, role: str = "p1", *, k: int = 16, prior_weight: float = 0.5,
    opponent_policy: Any | None = None, seed: int = 0,
) -> int:
    """End-to-end determinized decision on a live EngineBattle: build the root fog batch, sample
    k worlds, simulate each legal action in each world, blend with the policy prior, argmax."""
    stepper = EngineWorldStepper(battle, role, opponent_policy=opponent_policy, seed=seed)
    batch = engine_root_batch(agent, battle, role)
    return determinized_action(agent, batch, step_fn=stepper, k=k, prior_weight=prior_weight)
