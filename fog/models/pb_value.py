"""Public-belief value network for two-sided search.

Returns V[leaf, k, j], the expected continuation payoff for each candidate-team
pair under the supplied joint belief rho[leaf, k, j]. Values use our perspective
and are tanh-bounded to (-1, 1); the opponent's payoff is their negation.

Public-leaf and candidate encoders run once per decision. Belief-weighted
bipartite messages couple the candidate teams before the pair head scores them.
`forward_cached` refreshes these messages and scores when the solver's belief
changes, reusing the encoded leaves and teams. Holding scores fixed across
iterations is an approximation because continuation values depend on belief.

The solver averages over private types. The value bridge must retain the full
pair table to avoid applying belief weights twice. Candidate permutations
permute the corresponding output axes; candidate positions carry no identity.
See tests/test_pb_value.py for tensor-only checks of these contracts."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from fog.constants.schema import N_EV_UNITS, N_NATURES, SET_SLOT_ATTRS

# Attribute layout of one team, shared with fog/models/belief.py.
SLOT_ATTRS = SET_SLOT_ATTRS
N_SLOTS = 6
N_ATTRS = N_SLOTS * len(SLOT_ATTRS)
# attribute-within-slot -> output/embedding family (the 4 move slots share one family)
_ATTR_FAMILY = (
    "species", "item", "ability", "move", "move", "move", "move",
    "nature", "ev", "ev", "ev", "ev", "ev", "ev", "tera",
)
_POS_FAMILY = tuple(_ATTR_FAMILY[p % len(SLOT_ATTRS)] for p in range(N_ATTRS))


class TeamEncoder(nn.Module):
    """A 6x15 candidate-team grid -> one team embedding without attribute-assignment collisions.

    Each field is transformed before pooling. Fields are first combined within one Pokémon, so an
    item, ability, move set, nature, six EV values, and Tera type remain bound to that Pokémon's
    species. The six
    complete Pokémon representations are then pooled as a set. A prior implementation averaged all
    raw embeddings before its first nonlinear transform; swapping two items between species was
    therefore an exact collision. The staged pooling below makes that regression testable.
    """

    def __init__(self, d: int, sizes: dict[str, int]) -> None:
        super().__init__()
        self.d = d
        self.sizes = dict(sizes)
        self.value_embeds = nn.ModuleDict(
            {name: nn.Embedding(n + 1, d) for name, n in self.sizes.items()}  # +1 == <UNK>
        )
        # species, item, ability, move, nature, EV (shared by six stats), tera
        self.field_id = nn.Embedding(7, d)
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
            "field_kind",
            torch.tensor([0, 1, 2, 3, 3, 3, 3, 4, 5, 5, 5, 5, 5, 5, 6], dtype=torch.long),
            persistent=False,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """`tokens` [N,6,15] value indices (-1 or OOB -> <UNK>). Returns [N,d]."""
        if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (N_SLOTS, len(SLOT_ATTRS)):
            raise ValueError(
                f"team tokens must be [N,6,{len(SLOT_ATTRS)}], got {tuple(tokens.shape)}"
            )
        fields = []
        for column, fam in enumerate(_ATTR_FAMILY):
            size = self.sizes[fam]
            idx = tokens[:, :, column].clone()
            idx[(idx < 0) | (idx >= size)] = size                         # hidden / OOB -> <UNK>
            fields.append(self.value_embeds[fam](idx))                    # [N, 6, d]
        attr = torch.stack(fields, dim=2)                                 # [N, 6, 15, d]
        attr = attr + self.field_id(self.field_kind).view(1, 1, len(SLOT_ATTRS), self.d)
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


@dataclass
class PBValueCache:
    """Belief-independent representations reusable across every solver iteration."""

    ctx: torch.Tensor          # [L, d] public leaf representation
    own: torch.Tensor          # [K, d] candidate-team representations
    opp: torch.Tensor          # [J, d]


class PBValueNet(nn.Module):
    """Public-belief value network (§9). `forward` returns V[L, K, J] (see module docstring)."""

    @classmethod
    def from_tables(cls, tables=None, *, state_dim: int = 256, dyn_dim: int = 256,
                    d: int = 128, n_rounds: int = 2) -> PBValueNet:
        """Real token-family cardinalities from the generated constants — use this (not the bare
        constructor) for anything consuming real candidate token grids: the bare defaults are
        test-scale and silently map real indices (e.g. species 912 > 320) to <UNK>. Tera is
        TYPE_INDEX plus the truth-tokenizer's out-of-table fallback row (18)."""
        from fog.constants.schema import TYPE_INDEX

        if tables is None:
            from fog.constants.tables import TABLES as tables
        sizes = {"species": tables.n_species, "item": tables.n_items,
                 "ability": tables.n_abilities, "move": tables.n_moves,
                 "nature": N_NATURES, "ev": N_EV_UNITS, "tera": len(TYPE_INDEX) + 1}
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
        sizes = sizes or {
            "species": 320, "item": 64, "ability": 160, "move": 720,
            "nature": N_NATURES, "ev": N_EV_UNITS, "tera": 19,
        }
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

    def prepare_candidates(
        self, own_tokens: torch.Tensor, opp_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode both candidate sets once per decision, before regret iterations begin."""
        return self.team(own_tokens), self.team(opp_tokens)

    def prepare_leaves(
        self, private_state: torch.Tensor, public_context: torch.Tensor
    ) -> torch.Tensor:
        """Encode public leaf state once; no belief or strategy enters this computation."""
        return self.ctx(torch.cat([private_state, public_context], dim=-1))

    def prepare(
        self,
        private_state: torch.Tensor,  # [L, state_dim] per-leaf private-state context
        public_context: torch.Tensor, # [L, dyn_dim] fused public-state + history context
        own_tokens: torch.Tensor,  # [K, 6, 15]      our K candidate teams (shared across leaves)
        opp_tokens: torch.Tensor,  # [J, 6, 15]      opponent's J candidate teams (shared)
    ) -> PBValueCache:
        """All expensive belief-independent work, cached once per decision."""
        own, opp = self.prepare_candidates(own_tokens, opp_tokens)
        return PBValueCache(
            ctx=self.prepare_leaves(private_state, public_context), own=own, opp=opp
        )

    def forward_cached(
        self,
        cache: PBValueCache,
        rho: torch.Tensor,         # [L, K, J] each leaf's conditioning belief
    ) -> torch.Tensor:
        """Run only belief-conditioned messages and the small pair head."""
        L, K, J = cache.ctx.shape[0], cache.own.shape[0], cache.opp.shape[0]
        # Broadcast particles across leaves and inject the leaf context (property 1).
        own = cache.own.unsqueeze(0).expand(L, K, self.d) + cache.ctx.unsqueeze(1)
        opp = cache.opp.unsqueeze(0).expand(L, J, self.d) + cache.ctx.unsqueeze(1)

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

    def forward(
        self,
        private_state: torch.Tensor,
        public_context: torch.Tensor,
        own_tokens: torch.Tensor,
        opp_tokens: torch.Tensor,
        rho: torch.Tensor,
    ) -> torch.Tensor:
        """Reference convenience path; search uses ``prepare`` + ``forward_cached``."""
        return self.forward_cached(
            self.prepare(private_state, public_context, own_tokens, opp_tokens), rho
        )

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
