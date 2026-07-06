"""Oracle-gap headline metric (spec §9).

    closure = (N_agent − blind_reactive) / (set_oracle − blind_reactive)

where blind_reactive = the SAME model with history length forced to 0 and epistemic inputs zeroed.
Report closure for belief-conditioned vs not — the headline number for the belief hypothesis.
"""
from __future__ import annotations


def closure(n_agent: float, blind_reactive: float, set_oracle: float) -> float:
    """Fraction of the set-oracle headroom that belief-conditioning closes.

    All three args are win-rates (or Elo-implied win-rates) vs a common opponent pool.
    """
    denom = set_oracle - blind_reactive
    if denom == 0:
        return float("nan")
    return (n_agent - blind_reactive) / denom


def run_closure_report(ckpt, opponent, games: int, fmt: str, team_pool, cfg: dict) -> dict:
    """Play the same checkpoint in three modes vs a fixed opponent and compute the closure headline
    (§9). The three modes are one-line config overrides on the SAME weights:
      - N-agent:        model.blind_reactive=False, oracle=none
      - blind_reactive: model.blind_reactive=True  (history=0, belief conditioning off)
      - set_oracle:     model.oracle=set           (opponent truth fed to the belief net)
    Returns win-rates + closure.
    """
    import copy

    from fog.eval.harness import run_match

    def winrate(overrides: dict) -> float:
        c = copy.deepcopy(cfg)
        c.setdefault("model", {}).update(overrides)
        wins, finished = run_match(ckpt, opponent, games, fmt, team_pool=team_pool, cfg=c)
        return wins / max(1, finished)

    # All three modes share the SAME architecture (so one checkpoint loads for all) — they differ
    # only in inputs/conditioning. blind_reactive ablates history + epistemic inputs at forward time.
    n = winrate({"blind_reactive": False, "oracle": "none"})
    blind = winrate({"blind_reactive": True, "oracle": "none"})
    oracle = winrate({"blind_reactive": False, "oracle": "set"})
    return {
        "n_agent": n, "blind_reactive": blind, "set_oracle": oracle,
        "closure": closure(n, blind, oracle),
    }
