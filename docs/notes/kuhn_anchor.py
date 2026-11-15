"""Kuhn poker, tabular, exact action values: does a magnet make self-play converge?

Update per information state (behavioural form), both players at once:
    pi_new  ∝  [ pi * rho^(alpha*eta) * exp(eta * Q) ]^(1 / (1 + alpha*eta))
alpha = 0 is plain mirror descent (the "pi * exp(Q/tau)" improvement step, iterated).
Exploitability = (best-response value of P1 + best-response value of P2) / 2; 0 at a Nash equilibrium.
"""
import itertools
import sys

import numpy as np

CARDS = (0, 1, 2)
DEALS = [(a, b) for a in CARDS for b in CARDS if a != b]
INFOSTATES = [(p, c, h) for p, hs in ((0, ("", "pb")), (1, ("p", "b"))) for c in CARDS for h in hs]


def terminal_value(cards, h):
    """Payoff to player 0, or None if h is not terminal."""
    win = 1.0 if cards[0] > cards[1] else -1.0
    return {"pp": win, "pbp": -1.0, "pbb": 2 * win, "bp": 1.0, "bb": 2 * win}.get(h)


def value(cards, h, pol):
    u = terminal_value(cards, h)
    if u is not None:
        return u
    p = len(h) % 2
    probs = pol[(p, cards[p], h)]
    return sum(probs[a] * value(cards, h + "pb"[a], pol) for a in (0, 1))


def action_values(pol):
    """Q(I, a) for the acting player: posterior over the opponent's card comes from pol itself."""
    num = {i: np.zeros(2) for i in INFOSTATES}
    den = {i: 0.0 for i in INFOSTATES}

    def walk(cards, h, reach):
        if terminal_value(cards, h) is not None:
            return
        p = len(h) % 2
        key = (p, cards[p], h)
        den[key] += reach[1 - p]
        for a in (0, 1):
            u0 = value(cards, h + "pb"[a], pol)
            num[key][a] += reach[1 - p] * (u0 if p == 0 else -u0)
            nxt = list(reach)
            nxt[p] *= pol[key][a]
            walk(cards, h + "pb"[a], nxt)

    for cards in DEALS:
        walk(cards, "", [1.0, 1.0])
    return {i: num[i] / den[i] for i in INFOSTATES}


def game_value(pol):
    return float(np.mean([value(c, "", pol) for c in DEALS]))


def exploitability(pol):
    total = 0.0
    for p in (0, 1):
        mine = [i for i in INFOSTATES if i[0] == p]
        best = -np.inf
        for choice in itertools.product((0, 1), repeat=len(mine)):
            trial = dict(pol)
            for key, a in zip(mine, choice):
                trial[key] = np.eye(2)[a]
            v = game_value(trial)
            best = max(best, v if p == 0 else -v)
        total += best
    return total / 2


def run(alpha, eta, steps, magnet_every=0, noise=0.0, hard=False, seed=0, report=()):
    rng = np.random.default_rng(seed)
    # A deliberately lopsided start, so that nothing converges by symmetry.
    pol = {i: np.array([0.8, 0.2]) if k % 2 else np.array([0.3, 0.7]) for k, i in enumerate(INFOSTATES)}
    rho = {i: np.full(2, 0.5) for i in INFOSTATES}
    out = {}
    for t in range(1, steps + 1):
        q = action_values(pol)
        new = {}
        for i in INFOSTATES:
            qi = q[i] + noise * rng.standard_normal(2)
            if hard:
                new[i] = np.eye(2)[int(np.argmax(qi))] * 0.98 + 0.01
                continue
            logit = (np.log(pol[i]) + alpha * eta * np.log(rho[i]) + eta * qi) / (1 + alpha * eta)
            logit -= logit.max()
            new[i] = np.exp(logit) / np.exp(logit).sum()
        pol = new
        if magnet_every and t % magnet_every == 0:
            rho = {i: pol[i].copy() for i in INFOSTATES}
        if t in report:
            out[t] = exploitability(pol)
    return out, pol


if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    report = tuple(sorted({100, 1000, steps // 4, steps // 2, steps}))
    rows = [
        ("hard best response to the last policy, repeated", dict(alpha=0.0, eta=0.0, hard=True)),
        ("mirror descent, no magnet (alpha=0)", dict(alpha=0.0, eta=0.2)),
        ("fixed uniform magnet (alpha=0.2)", dict(alpha=0.2, eta=0.2)),
        ("fixed uniform magnet (alpha=0.05)", dict(alpha=0.05, eta=0.2)),
        ("moving magnet (alpha=0.2, reset every 500)", dict(alpha=0.2, eta=0.2, magnet_every=500)),
        ("moving magnet, noisy Q (sd 0.3)", dict(alpha=0.2, eta=0.05, magnet_every=2000, noise=0.3)),
        ("no magnet, noisy Q (sd 0.3)", dict(alpha=0.0, eta=0.05, noise=0.3)),
    ]
    print("steps:", report)
    for name, kw in rows:
        res, pol = run(steps=steps, report=report, **kw)
        print(f"{name:48s}", "  ".join(f"{res[t]:.5f}" for t in report), f" value={game_value(pol):+.4f}")
    print("Nash value for player 0 is -1/18 =", round(-1 / 18, 4))
