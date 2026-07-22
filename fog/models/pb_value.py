"""Public-belief value network: the strategy-dependent leaf evaluator for two-sided search
(PROJECT_DESIGN_PROPOSAL.md §9, PROPOSAL_IMPL_PLAN.md Phase G — OWNER DECISION 2026-07-16).

WHAT THIS REPLACES. The interim leaf evaluator (fog/search/pb_values.py) reused the policy's small
value head with four scalar posterior features. Four numbers cannot say WHICH candidate teams retain
probability, and the head must re-run every solver iteration because its scalar output moves with the
posterior. This module is the real thing the proposal §9 asks for: it reads the full K x J belief and
each candidate team's identity, and — crucially — emits a value for EVERY candidate pair at once so
the solver can reuse a cached table between periodic belief-conditioned refreshes.

THE OUTPUT SHAPE IS THE COST DECISION (§9.2, settled). During one real decision the solver runs
hundreds of regret iterations; each iteration moves both strategies, hence the action-conditioned
posterior at every cached leaf (§5.1: same leaf observation, different iteration, different belief).
A single-scalar value would demand a full encoder pass per leaf per iteration — the dominant cost of
a decision. Instead this network runs ONCE per leaf and returns the per-pair table

    V[leaf, k, j] = our expected continuation payoff from this leaf IF the true teams are (k, j),
                    under the belief the network was conditioned on.

The simulated cell for private pair `(k, j)` receives `V[leaf, k, j]`. The two-sided solver later
performs the single required average over the other player's possible private types. Averaging the
table inside the bridge as well would use the belief twice and destroy type-specific decisions.

Between recomputes `V` is held fixed. Because a pair's true continuation
value ALSO depends on the belief carried into the leaf (continuation play is belief-dependent — the
whole §5.1 thesis), a table conditioned once slowly drifts stale as the strategies move, so the
bridge recomputes it every N iterations. Refreshing every iteration stays the correctness reference
on the exact small-pool gate (§15.3); the drift error is measured against it.

THE SEVEN REQUIRED PROPERTIES (§9.1), and where each lives here:

  1. Reads the shared public-state representation and explicit current state.
        -> `s_board` and `c_dyn` per leaf, projected into the per-leaf context vector `ctx`. These
           are exactly the backbone outputs the policy head consumes (agent.head_inputs), so the
           expensive trunk pass over the leaf set is shared, not duplicated (invariant 19.13).
  2. Reads individual embeddings for the K own-state and J opponent-state candidates.
        -> `TeamEncoder` embeds each candidate team's 6x9 attribute grid into one token. Own and
           opponent candidates are encoded ONCE per decision (shared across all leaves).
  3. Reads the complete K x J belief weights.
        -> `rho[L, K, J]` drives the bipartite messages (own<->opp) AND is available to the pair
           head. This is the coupling channel: change rho, and V changes.
  4. Represents different continuation values for different private types where required.
        -> the head emits an independent scalar per (k, j); nothing averages types away inside.
  5. Permutation-invariant to particle storage order.
        -> teams carry identity through their token CONTENT, never position (mirrors belief.py's
           "identity is explicit, not positional"); the message passing is order-equivariant; the
           head is applied per pair. Permuting the K (or J) particles permutes V's k- (or j-) axis
           identically. Pinned by tests/test_pb_value.py.
  6. Does NOT independently solve each private-state pair as a perfect-information game.
        -> a bare per-pair MLP would; the coupling comes from the message-passing rounds, where each
           own token mixes in the opponent set weighted by rho (and vice versa) before the head sees
           it. The pair value therefore depends on the whole candidate field and the joint belief,
           not just its own two teams.
  7. Batchable over the large leaf set search creates.
        -> teams encode once [K, d] / [J, d]; leaves broadcast over the L axis; one forward scores
           all leaves x all pairs.

VALUE CONVENTION. Output is OUR-perspective continuation payoff, `tanh`-bounded to (-1, 1) to match
the zero-sum terminal-payoff convention the grid stores in `terminal_value` (win = +1, loss = -1).
The opponent's value is the negation; the solver's `values_fn` uses this network's output directly as
`u`'s continuation entries.

This module is engine-free and self-contained: it takes tensors and returns a tensor, so every
property above is testable on hand-built inputs without a battle (tests/test_pb_value.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Attribute layout of one team, shared with fog/models/belief.py (6 slots x 9 attributes = 54).
SLOT_ATTRS = ("species", "item", "ability", "move0", "move1", "move2", "move3", "spread", "tera")
N_SLOTS = 6
N_ATTRS = N_SLOTS * len(SLOT_ATTRS)  # 54
# attribute-within-slot -> output/embedding family (the 4 move slots share one family)
_ATTR_FAMILY = ("species", "item", "ability", "move", "move", "move", "move", "spread", "tera")
_POS_FAMILY = tuple(_ATTR_FAMILY[p % len(SLOT_ATTRS)] for p in range(N_ATTRS))


class TeamEncoder(nn.Module):
    """A 6x9 candidate-team grid -> one team embedding without attribute-assignment collisions.

    Each field is transformed before pooling. Fields are first combined within one Pokémon, so an
    item, ability, move set, spread, and Tera type remain bound to that Pokémon's species. The six
    complete Pokémon representations are then pooled as a set. A prior implementation averaged all
    54 raw embeddings before its first nonlinear transform; swapping two items between species was
    therefore an exact collision. The staged pooling below makes that regression testable.
    """

    def __init__(self, d: int, sizes: dict[str, int]) -> None:
        super().__init__()
        self.d = d
        self.sizes = dict(sizes)
        self.value_embeds = nn.ModuleDict(
            {name: nn.Embedding(n + 1, d) for name, n in self.sizes.items()}  # +1 == <UNK>
        )
        # species, item, ability, move (shared by all four move slots), spread, tera
        self.field_id = nn.Embedding(6, d)
        self.attr_proj = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, d)
        )
        self.mon_proj = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d)
        )
        self.team_proj = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, d)
        )
        self.register_buffer(
            "field_kind", torch.tensor([0, 1, 2, 3, 3, 3, 3, 4, 5], dtype=torch.long),
            persistent=False,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """`tokens` [N, 6, 9] long value indices (-1 or >= family size -> <UNK>). Returns [N, d]."""
        if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (N_SLOTS, len(SLOT_ATTRS)):
            raise ValueError(f"team tokens must be [N,6,9], got {tuple(tokens.shape)}")
        fields = []
        for column, fam in enumerate(_ATTR_FAMILY):
            size = self.sizes[fam]
            idx = tokens[:, :, column].clone()
            idx[(idx < 0) | (idx >= size)] = size                         # hidden / OOB -> <UNK>
            fields.append(self.value_embeds[fam](idx))                    # [N, 6, d]
        attr = torch.stack(fields, dim=2)                                 # [N, 6, 9, d]
        attr = attr + self.field_id(self.field_kind).view(1, 1, 9, self.d)
        attr = self.attr_proj(attr)
        pokemon = self.mon_proj(attr.mean(dim=2))                         # [N, 6, d]
        return self.team_proj(pokemon.mean(dim=1))                        # [N, d]


def _weighted_messages(src: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Belief-weighted mean of `src` particles for each destination particle (§9.2 bipartite step).

    src    [L, S, d]   — the source set's tokens (e.g. the J opponent tokens).
    weight [L, D, S]   — nonnegative message weights dest<-src (a slice of rho or its transpose).
    Returns [L, D, d]: for each destination d0, sum_s w[d0, s] src[s] / sum_s w[d0, s]. Destinations
    whose weights are all zero (an eliminated candidate carries no belief mass) get a zero message,
    never NaN.
    """
    num = torch.einsum("lds,lse->lde", weight, src)                       # [L, D, d]
    den = weight.sum(dim=-1, keepdim=True).clamp_min(1e-12)               # [L, D, 1]
    return num / den


class PBValueNet(nn.Module):
    """Public-belief value network (§9). `forward` returns V[L, K, J] (see module docstring)."""

    @classmethod
    def from_tables(cls, tables=None, *, state_dim: int = 256, dyn_dim: int = 256,
                    d: int = 128, n_rounds: int = 2) -> "PBValueNet":
        """Real token-family cardinalities from the generated constants — use this (not the bare
        constructor) for anything consuming real candidate token grids: the bare defaults are
        test-scale and silently map real indices (e.g. species 912 > 320) to <UNK>. Tera is
        TYPE_INDEX plus the truth-tokenizer's out-of-table fallback row (18)."""
        from fog.constants.schema import MAX_SPREADS_PER_SPECIES, TYPE_INDEX

        if tables is None:
            from fog.constants.tables import TABLES as tables
        sizes = {"species": tables.n_species, "item": tables.n_items,
                 "ability": tables.n_abilities, "move": tables.n_moves,
                 "spread": MAX_SPREADS_PER_SPECIES + 1, "tera": len(TYPE_INDEX) + 1}
        return cls(state_dim=state_dim, dyn_dim=dyn_dim, d=d, n_rounds=n_rounds, sizes=sizes)

    def __init__(
        self,
        state_dim: int = 256,
        dyn_dim: int = 256,
        d: int = 128,
        n_rounds: int = 2,
        sizes: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.d = d
        self.n_rounds = n_rounds
        sizes = sizes or {"species": 320, "item": 64, "ability": 160, "move": 720,
                          "spread": 17, "tera": 19}
        self.team = TeamEncoder(d, sizes)
        # Leaf public state -> per-leaf context added to every particle token.
        self.ctx = nn.Sequential(nn.Linear(state_dim + dyn_dim, d), nn.GELU(), nn.Linear(d, d))
        # One message-passing round per side: mix the belief-weighted other-set message into a token.
        self.own_upd = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, d), nn.GELU()) for _ in range(n_rounds)
        )
        self.opp_upd = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, d), nn.GELU()) for _ in range(n_rounds)
        )
        # Pair head: (own_k, opp_j, rho_kj) -> scalar continuation value.
        self.head = nn.Sequential(
            nn.LayerNorm(2 * d + 1), nn.Linear(2 * d + 1, d), nn.GELU(), nn.Linear(d, 1)
        )

    def forward(
        self,
        s_board: torch.Tensor,     # [L, state_dim]  per-leaf board summary (backbone)
        c_dyn: torch.Tensor,       # [L, dyn_dim]    per-leaf dynamics context (backbone)
        own_tokens: torch.Tensor,  # [K, 6, 9]       our K candidate teams (shared across leaves)
        opp_tokens: torch.Tensor,  # [J, 6, 9]       opponent's J candidate teams (shared)
        rho: torch.Tensor,         # [L, K, J]       each leaf's conditioning belief
    ) -> torch.Tensor:
        """Returns V[L, K, J], tanh-bounded OUR-perspective continuation payoff per candidate pair."""
        L = s_board.shape[0]
        K, J = own_tokens.shape[0], opp_tokens.shape[0]
        ctx = self.ctx(torch.cat([s_board, c_dyn], dim=-1))               # [L, d]

        own = self.team(own_tokens)                                       # [K, d], once per decision
        opp = self.team(opp_tokens)                                       # [J, d]
        # Broadcast particles across leaves and inject the leaf context (property 1).
        own = own.unsqueeze(0).expand(L, K, self.d) + ctx.unsqueeze(1)    # [L, K, d]
        opp = opp.unsqueeze(0).expand(L, J, self.d) + ctx.unsqueeze(1)    # [L, J, d]

        rho = rho.to(own.dtype)
        rho_t = rho.transpose(1, 2).contiguous()                          # [L, J, K]
        for own_upd, opp_upd in zip(self.own_upd, self.opp_upd):
            # each own token mixes in the opponent set weighted by P(j | this k); symmetric for opp.
            own_msg = _weighted_messages(opp, rho)                        # [L, K, d]
            opp_msg = _weighted_messages(own, rho_t)                      # [L, J, d]
            own = own + own_upd(torch.cat([own, own_msg], dim=-1))
            opp = opp + opp_upd(torch.cat([opp, opp_msg], dim=-1))

        # Pair head over the full K x J grid (property 4/6): broadcast own over j and opp over k.
        own_g = own.unsqueeze(2).expand(L, K, J, self.d)                  # [L, K, J, d]
        opp_g = opp.unsqueeze(1).expand(L, K, J, self.d)                  # [L, K, J, d]
        pair = torch.cat([own_g, opp_g, rho.unsqueeze(-1)], dim=-1)       # [L, K, J, 2d+1]
        return torch.tanh(self.head(pair).squeeze(-1))                    # [L, K, J]

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
