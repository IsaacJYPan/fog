"""Shared match-running harness for evaluation (spec §9).

Runs games between FOG agents (checkpoints) and/or poke-env scripted baselines on a team pool, and
returns win counts. Everything else in eval/ (gauntlet, elo) builds on this.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

SCRIPTED = {"Random", "MaxDamage", "SimpleHeuristics"}


def _scripted(name: str, fmt: str, team):
    from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer

    from fog.env.player import unique_account

    cls = {"Random": RandomPlayer, "MaxDamage": MaxBasePowerPlayer,
           "SimpleHeuristics": SimpleHeuristicsPlayer}[name]
    return cls(battle_format=fmt, account_configuration=unique_account("fogb"), team=team)


def _fog_from_ckpt(ckpt: str, fmt: str, team, cfg: dict, device, oracle_pool=None):
    import copy

    import torch

    from fog.env.player import FogPlayer, unique_account
    from fog.models.agent import FogAgent

    # "oracle-set:<ckpt>" — the same checkpoint played as a READING opponent (§9: a gauntlet entry
    # where concealment demonstrably matters): belief net fed the pool-matched opponent truth.
    if ckpt.startswith("oracle-set:"):
        ckpt = ckpt[len("oracle-set:"):]
        cfg = copy.deepcopy(cfg)
        cfg.setdefault("model", {})["oracle"] = "set"

    agent = FogAgent.from_tables(cfg).to(device)
    agent.load_state_dict(torch.load(ckpt, map_location=device))
    agent.eval()
    adapter = FogPlayer(agent, temperature=cfg.get("eval", {}).get("temperature", 0.0))
    # set-oracle ceiling: feed the belief net the opponent's true team, matched from the pool by
    # revealed species (progressively perfect — full truth once the multiset resolves uniquely).
    if agent.oracle in ("set", "action") and oracle_pool:
        from fog.constants.tables import TABLES
        from fog.env.team_truth import PoolTruth, revealed_opponent_species

        pt = PoolTruth(oracle_pool, TABLES)
        adapter.oracle_truth_provider = lambda b: pt.match(revealed_opponent_species(b, TABLES))
    return adapter.build(
        battle_format=fmt, max_concurrent_battles=1,
        account_configuration=unique_account("foge"), team=team,
    )


def build_player(spec, fmt: str, team, cfg: dict, device, oracle_pool=None):
    """spec: a scripted name or a checkpoint path. Returns a poke-env Player."""
    if str(spec) in SCRIPTED:
        return _scripted(str(spec), fmt, team)
    return _fog_from_ckpt(str(spec), fmt, team, cfg, device, oracle_pool=oracle_pool)


async def _play(p1, p2, n: int) -> tuple[int, int]:
    p1.reset_battles()
    p2.reset_battles()
    await p1.battle_against(p2, n_battles=n)
    return p1.n_won_battles, p1.n_finished_battles


def run_match(spec_a, spec_b, n_games: int, fmt: str, team_pool=None, cfg=None, device=None) -> tuple[int, int]:
    """Play n_games between two specs. Returns (a_wins, finished). team_pool = list of pastes."""
    import torch

    from fog.env.teams import make_pool_teambuilder

    cfg = cfg or {}
    device = device or torch.device("cpu")
    team_a = make_pool_teambuilder(team_pool, seed=1) if team_pool else None
    team_b = make_pool_teambuilder(team_pool, seed=2) if team_pool else None
    # set-oracle agents on either side read the other's team truth from the shared pool
    a = build_player(spec_a, fmt, team_a, cfg, device, oracle_pool=team_pool)
    b = build_player(spec_b, fmt, team_b, cfg, device, oracle_pool=team_pool)
    return asyncio.run(_play(a, b, n_games))
