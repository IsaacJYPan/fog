"""Shared match-running harness for evaluation.

Runs games between FOG agents (checkpoints) and/or poke-env scripted baselines on a team pool, and
returns win counts. Everything else in eval/ (gauntlet, elo) builds on this.
"""
from __future__ import annotations

import asyncio

SCRIPTED = {"Random", "MaxDamage", "SimpleHeuristics"}


def _scripted(name: str, fmt: str, team):
    from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer

    from fog.env.player import unique_account

    cls = {"Random": RandomPlayer, "MaxDamage": MaxBasePowerPlayer,
           "SimpleHeuristics": SimpleHeuristicsPlayer}[name]
    return cls(battle_format=fmt, account_configuration=unique_account("fogb"), team=team)


def _fog_from_ckpt(ckpt: str, fmt: str, team, cfg: dict, device, oracle_pool=None, search=None):
    import copy

    import torch

    from fog.env.player import FogPlayer, unique_account
    from fog.models.agent import FogAgent

    # "oracle-set:<ckpt>" — the same checkpoint played with truth injected. NOTE what this measures:
    # the VALUE OF THE CONCEALED INFORMATION (an upper bound on what any reader could gain), not the
    # payoff of concealment — the oracle is handed the truth, so nothing the other side does
    # behaviorally can deny it. Concealment's payoff channel is a learned reader.
    if ckpt.startswith("oracle-set:"):
        ckpt = ckpt[len("oracle-set:"):]
        cfg = copy.deepcopy(cfg)
        cfg.setdefault("model", {})["oracle"] = "set"

    agent = FogAgent.from_tables(cfg).to(device)
    agent.load_state_dict(torch.load(ckpt, map_location=device))
    agent.eval()
    adapter = FogPlayer(agent, temperature=cfg.get("eval", {}).get("temperature", 0.0))
    # Determinized search at action time (opt-in; reconstructs engine positions from the live
    # battle + belief-sampled worlds). `search` is a dict {"k","prior_weight","seed"} or None.
    if search:
        adapter.search = dict(search)
    # set-oracle ceiling: truth from the pool via attribute-consistency matching (progressive; also
    # disambiguates shared-species pools like POOL-A once any set-distinguishing attr is revealed —
    # the old species-only matcher was permanently ambiguous there and the oracle never fired).
    if agent.oracle in ("set", "action") and oracle_pool:
        from fog.constants.tables import TABLES
        from fog.env.team_truth import PoolTruth

        pt = PoolTruth(oracle_pool, TABLES)
        adapter.oracle_truth_provider = pt.match_battle
    return adapter.build(
        battle_format=fmt, max_concurrent_battles=1,
        account_configuration=unique_account("foge"), team=team,
    )


def build_player(spec, fmt: str, team, cfg: dict, device, oracle_pool=None, search=None):
    """spec: a scripted name or a checkpoint path. Returns a poke-env Player. `search` (dict) turns
    on determinized search for a checkpoint player (ignored for scripted baselines)."""
    if str(spec) in SCRIPTED:
        return _scripted(str(spec), fmt, team)
    return _fog_from_ckpt(str(spec), fmt, team, cfg, device, oracle_pool=oracle_pool, search=search)


async def _play(p1, p2, n: int) -> tuple[int, int]:
    p1.reset_battles()
    p2.reset_battles()
    await p1.battle_against(p2, n_battles=n)
    return p1.n_won_battles, p1.n_finished_battles


def run_match(spec_a, spec_b, n_games: int, fmt: str, team_pool=None, cfg=None, device=None,
              paired: bool = False, search=None, cfg_b=None) -> tuple[int, int]:
    """Play n_games between two specs. Returns (a_wins, finished). team_pool = list of pastes.

    `paired=True` plays each sampled team matchup twice with sides swapped (shared-schedule
    teambuilders) — the variance-reduction design for head-to-head comparisons of leakage-trained and baseline agents,
    where on a small pool the team matchup, not the policy, dominates single-game outcomes.

    `search` (dict) enables determinized search for spec_a (the checkpoint under test).

    `cfg_b` builds spec_b's agent when the two checkpoints have different architectures (e.g. the
    M1.5-vs-M1 b_vec gate: use_belief_vec true vs false changes the head width, so one shared cfg
    cannot load both). Defaults to `cfg`.
    """
    import torch

    from fog.env.teams import make_paired_teambuilders, make_pool_teambuilder

    cfg = cfg or {}
    cfg_b = cfg if cfg_b is None else cfg_b
    device = device or torch.device("cpu")
    if team_pool and paired:
        team_a, team_b = make_paired_teambuilders(team_pool, n_games, seed=1)
    else:
        team_a = make_pool_teambuilder(team_pool, seed=1) if team_pool else None
        team_b = make_pool_teambuilder(team_pool, seed=2) if team_pool else None
    # set-oracle agents on either side read the other's team truth from the shared pool
    a = build_player(spec_a, fmt, team_a, cfg, device, oracle_pool=team_pool, search=search)
    b = build_player(spec_b, fmt, team_b, cfg_b, device, oracle_pool=team_pool)
    return asyncio.run(_play(a, b, n_games))
