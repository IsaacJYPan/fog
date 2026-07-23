"""Permutation-equivariant set-autoregressive belief model.

The opponent's visible facts are not a sequence.  They are a set: a Tera type revealed on turn 20
must be available when predicting an item just as immediately as a species shown at team preview.
Generated hidden facts have the same status.  Consequently this model never encodes the physical
order in which facts were supplied.

The model is nevertheless autoregressive: it completes a team one fact at a time.  At every step:

1. all public reveals and all facts generated so far are encoded as an unordered set;
2. a value-independent scheduler chooses uniformly among the fields that are currently eligible;
3. the model samples that field's value and adds the resulting fact to the set.

Only mechanical dependencies constrain the scheduler.  Species precedes species-dependent legality;
spread follows species, item, and ability so a damage-consistency mask can be applied.  Everything
else is free to occur in either direction.  The four moves of one Pokemon are one exchangeable set:
they use the same query and value head, and generated values are stored in the next free move column
only as an external tensor convention.

Training matches sampling.  A dependency-valid random route and a random prefix length are drawn for
each example.  The input is the real reveal set plus that generated prefix, and the loss predicts the
route's next fact.  Thus every supervised conditional has exactly the same kind of partially completed
set as ancestral sampling.  This replaces the old mismatch in which random-order teacher forcing was
trained while the sampler walked one canonical order.

There are two probability objects and they must not be confused:

* a route probability, determined entirely by the uniform scheduler; and
* a value probability, the product of the network's conditional probabilities along that route.

Together they form a normalized distribution over (route, completed team).  A completed unordered
team marginalizes over routes.  Exact marginalization is factorial, so candidate scoring averages a
small, explicitly reported collection of random routes.  Sampling itself is exact for the normalized
extended distribution and returns both log-probabilities when ``return_trace=True``.

Set structure is hierarchical.  Facts belonging to one Pokemon are pooled only after a nonlinear
field-aware transform; the six Pokemon are then processed by a Transformer with no positional
encoding.  Therefore move order and input fact order are invariant, while permuting Pokemon slots
permutes the corresponding outputs instead of changing the represented team.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

SLOT_ATTRS = ("species", "item", "ability", "move0", "move1", "move2", "move3", "spread", "tera")
N_SLOTS = 6
N_ATTRS = N_SLOTS * len(SLOT_ATTRS)
SEQ_LEN = N_ATTRS + 1
_START = N_ATTRS

_ATTR_FAMILY = ("species", "item", "ability", "move", "move", "move", "move", "spread", "tera")
_FIELD_KIND = (0, 1, 2, 3, 3, 3, 3, 4, 5)
_MOVE_SUBSLOT = (None, None, None, 0, 1, 2, 3, None, None)
_MOVE_COLS = (3, 4, 5, 6)
_ITEM_COL, _ABILITY_COL, _SPREAD_COL = 1, 2, 7

_POS_SLOT = tuple(p // len(SLOT_ATTRS) for p in range(N_ATTRS))
_POS_FAMILY = tuple(_ATTR_FAMILY[p % len(SLOT_ATTRS)] for p in range(N_ATTRS))
_POS_SUB = tuple(_MOVE_SUBSLOT[p % len(SLOT_ATTRS)] for p in range(N_ATTRS))
_SLOT_MOVE_FLATS = tuple(
    tuple(j * len(SLOT_ATTRS) + c for c in _MOVE_COLS) for j in range(N_SLOTS)
)
_SLOT_ITEM_FLATS = tuple(j * len(SLOT_ATTRS) + _ITEM_COL for j in range(N_SLOTS))
_SLOT_ABILITY_FLATS = tuple(j * len(SLOT_ATTRS) + _ABILITY_COL for j in range(N_SLOTS))
_SLOT_SPREAD_FLATS = tuple(j * len(SLOT_ATTRS) + _SPREAD_COL for j in range(N_SLOTS))

_NEG_INF = float("-inf")


@dataclass
class BeliefOutput:
    logits: dict[str, torch.Tensor]
    # The set-AR training forward predicts one route step per row.  These optional dictionaries
    # identify that step and its possibly move-permuted target.  Marginal forwards leave them None.
    loss_valid: dict[str, torch.Tensor] | None = None
    loss_targets: dict[str, torch.Tensor] | None = None
    order: torch.Tensor | None = None


@dataclass
class BeliefSamples:
    teams: torch.Tensor              # [B, k, 6, 9]
    log_value_prob: torch.Tensor     # [B, k]
    log_route_prob: torch.Tensor     # [B, k]
    order: torch.Tensor              # [B, k, 54], -1 after completion

    @property
    def log_joint_path_prob(self) -> torch.Tensor:
        return self.log_value_prob + self.log_route_prob


class BeliefModel(nn.Module):
    """An unordered fact-set encoder used inside a matched autoregressive generator."""

    def __init__(
        self,
        state_dim: int = 256,
        d: int = 128,
        n_species: int = 320,
        n_items: int = 64,
        n_abilities: int = 160,
        n_moves: int = 720,
        n_spreads: int = 17,
        n_tera: int = 19,
    ) -> None:
        super().__init__()
        self.d = d
        self.nhead = 4
        self.sizes = {
            "species": n_species,
            "item": n_items,
            "ability": n_abilities,
            "move": n_moves,
            "spread": n_spreads,
            "tera": n_tera,
        }

        # These names intentionally survive from the preceding architecture.  A warm-start loader
        # can retain learned value embeddings, family heads, and the dynamics projection while the
        # obsolete ordered decoder is discarded.
        self.s_proj = nn.Linear(state_dim, d)
        self.heads = nn.ModuleDict({name: nn.Linear(d, n) for name, n in self.sizes.items()})
        self.value_embeds = nn.ModuleDict(
            {name: nn.Embedding(n + 1, d) for name, n in self.sizes.items()}
        )

        # Six semantic field identities; all four move columns deliberately share identity 3.
        self.field_id = nn.Embedding(6, d)
        self.count_id = nn.Embedding(len(SLOT_ATTRS) + 1, d)
        self.fact_proj = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, d)
        )
        self.mon_proj = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d)
        )
        team_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=self.nhead, dim_feedforward=4 * d, batch_first=True,
            norm_first=True,
        )
        self.team_encoder = nn.TransformerEncoder(team_layer, num_layers=2)
        self.team_token = nn.Parameter(torch.empty(1, 1, d))
        nn.init.normal_(self.team_token, std=0.02)

        self.query_id = nn.Embedding(6, d)
        self.query_proj = nn.Sequential(
            nn.LayerNorm(3 * d), nn.Linear(3 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, d)
        )

        pos_n = torch.tensor([self.sizes[_POS_FAMILY[p]] for p in range(N_ATTRS)])
        self.register_buffer("pos_n", pos_n.to(torch.long), persistent=False)
        self.register_buffer(
            "field_kind", torch.tensor(_FIELD_KIND, dtype=torch.long), persistent=False
        )

    # -------------------------------------------------------------------------- set/route mechanics
    def _revealed_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        flat = tokens.reshape(tokens.shape[0], N_ATTRS)
        n = self.pos_n.to(flat.device)
        return (flat >= 0) & (flat < n)

    def _fed_values(self, tokens: torch.Tensor, teacher_values: torch.Tensor | None) -> torch.Tensor:
        """Compatibility helper: reveals plus all supplied teacher values.

        The training forward below uses only a sampled teacher prefix; this helper remains useful to
        tests and callers constructing a fully known context explicitly.
        """
        flat = tokens.reshape(tokens.shape[0], N_ATTRS)
        revealed = self._revealed_mask(tokens)
        if teacher_values is None:
            return torch.where(revealed, flat, torch.full_like(flat, -1))
        truth = teacher_values.reshape(tokens.shape[0], N_ATTRS)
        return torch.where(revealed, flat, truth)

    @staticmethod
    def _spread_after_item_ability(r: torch.Tensor, revealed: torch.Tensor) -> torch.Tensor:
        """Compatibility utility used by the ordering tests.

        The real scheduler enforces dependencies through `_eligible`; this stable-sort key helper
        expresses the same spread-after-item/ability constraint for diagnostic random keys.
        """
        r = r.clone()
        it = torch.where(
            revealed[:, _SLOT_ITEM_FLATS], r.new_full((), float("-inf")),
            r[:, _SLOT_ITEM_FLATS],
        )
        ab = torch.where(
            revealed[:, _SLOT_ABILITY_FLATS], r.new_full((), float("-inf")),
            r[:, _SLOT_ABILITY_FLATS],
        )
        r[:, _SLOT_SPREAD_FLATS] = torch.maximum(
            torch.maximum(it, ab), r[:, _SLOT_SPREAD_FLATS]
        ) + 1e-3
        return r

    def _eligible(self, known: torch.Tensor) -> torch.Tensor:
        """Fields that may be generated next, given a set of known flat attributes.

        Exactly one move query per Pokemon is eligible: the first free storage column represents
        "draw another member of this unordered move set", not a semantically distinct move slot.
        """
        b = known.shape[0]
        out = torch.zeros_like(known)
        for slot in range(N_SLOTS):
            base = slot * len(SLOT_ATTRS)
            species = known[:, base]
            item = known[:, base + _ITEM_COL]
            ability = known[:, base + _ABILITY_COL]

            out[:, base] = ~species
            out[:, base + _ITEM_COL] = ~item
            out[:, base + 8] = ~known[:, base + 8]  # Tera has no prerequisite.

            species_known = species
            out[:, base + _ABILITY_COL] = species_known & ~ability

            move_known = known[:, [base + c for c in _MOVE_COLS]]
            for sub, col in enumerate(_MOVE_COLS):
                # Only the first unresolved move storage cell represents the next set insertion.
                earlier_full = move_known[:, :sub].all(dim=1) if sub else torch.ones(
                    b, dtype=torch.bool, device=known.device
                )
                out[:, base + col] = species_known & ~move_known[:, sub] & earlier_full

            out[:, base + _SPREAD_COL] = (
                species_known & item & ability & ~known[:, base + _SPREAD_COL]
            )
        return out

    def _choose_next(
        self,
        known: torch.Tensor,
        mode: str,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (flat field id, number eligible, has_choice) for each row."""
        eligible = self._eligible(known)
        count = eligible.sum(dim=1)
        active = count > 0
        if mode == "canonical":
            # A diagnostic factorization only.  The deployed/training default is random.
            score = torch.arange(N_ATTRS, device=known.device).expand(len(known), -1)
            score = score.masked_fill(~eligible, N_ATTRS + 1)
            chosen = score.argmin(dim=1)
        elif mode == "random":
            score = torch.rand(
                len(known), N_ATTRS, device=known.device, generator=generator
            ).masked_fill(~eligible, 2.0)
            chosen = score.argmin(dim=1)
        else:
            raise ValueError(f"order mode must be 'random' or 'canonical', got {mode!r}")
        chosen = torch.where(active, chosen, torch.full_like(chosen, -1))
        return chosen, count, active

    def _route(
        self,
        revealed: torch.Tensor,
        mode: str,
        generator: torch.Generator | None,
        skip: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """A complete dependency-valid route over hidden fields.

        Returns order [B,54] padded with -1, route lengths [B], and the scheduler log-probability.
        """
        known = revealed.clone() if skip is None else (revealed | skip)
        order = torch.full(
            (len(known), N_ATTRS), -1, dtype=torch.long, device=known.device
        )
        log_q = torch.zeros(len(known), dtype=torch.float32, device=known.device)
        length = torch.zeros(len(known), dtype=torch.long, device=known.device)
        rows = torch.arange(len(known), device=known.device)
        for step in range(N_ATTRS):
            chosen, count, active = self._choose_next(known, mode, generator)
            if not bool(active.any()):
                break
            ar = rows[active]
            cp = chosen[active]
            order[ar, step] = cp
            known[ar, cp] = True
            length[active] += 1
            if mode == "random":
                log_q[active] -= count[active].to(torch.float32).log()
        return order, length, log_q

    def _order(
        self,
        revealed: torch.Tensor,
        tail_order: str,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """Compatibility layout: revealed facts | START | a dependency-valid hidden route."""
        route, length, _ = self._route(revealed, tail_order, generator)
        result = torch.full(
            (len(revealed), SEQ_LEN), _START, dtype=torch.long, device=revealed.device
        )
        for row in range(len(revealed)):
            prefix = torch.nonzero(revealed[row], as_tuple=True)[0]
            n_r = len(prefix)
            result[row, :n_r] = prefix
            result[row, n_r] = _START
            n_h = int(length[row])
            result[row, n_r + 1:n_r + 1 + n_h] = route[row, :n_h]
        return result

    def _permute_hidden_moves(
        self,
        truth: torch.Tensor,
        revealed: torch.Tensor,
        mode: str,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """Randomize only unrevealed move values; their storage columns have no semantics."""
        out = truth.clone()
        if mode != "random":
            return out
        for row in range(len(out)):
            for slot in range(N_SLOTS):
                flats = list(_SLOT_MOVE_FLATS[slot])
                hidden = [p for p in flats if not bool(revealed[row, p])]
                valid = [p for p in hidden if 0 <= int(out[row, p]) < self.sizes["move"]]
                if len(valid) <= 1:
                    continue
                perm = torch.randperm(len(valid), generator=generator, device=out.device)
                vals = out[row, valid].clone()
                out[row, valid] = vals[perm]
        return out

    # ------------------------------------------------------------------------------- set encoding
    def _fact_embeddings(self, fed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return transformed facts [B,6,9,d] and a known-value mask [B,6,9]."""
        fields = []
        known_cols = []
        for col, fam in enumerate(_ATTR_FAMILY):
            size = self.sizes[fam]
            idx = fed[:, :, col].clone()
            known = (idx >= 0) & (idx < size)
            idx = torch.where(known, idx, torch.full_like(idx, size))
            fields.append(self.value_embeds[fam](idx))
            known_cols.append(known)
        fact = torch.stack(fields, dim=2)
        known = torch.stack(known_cols, dim=2)
        fact = fact + self.field_id(self.field_kind).view(1, 1, 9, self.d)
        return self.fact_proj(fact), known

    def _set_context(
        self, c_dyn: torch.Tensor, fed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode an unordered partial team -> (global [B,d], Pokemon [B,6,d])."""
        b = fed.shape[0]
        grid = fed.reshape(b, N_SLOTS, len(SLOT_ATTRS))
        fact, known = self._fact_embeddings(grid)
        count = known.sum(dim=2)
        pooled = (fact * known.unsqueeze(-1)).sum(dim=2)
        pooled = pooled / count.clamp_min(1).unsqueeze(-1)
        pokemon = self.mon_proj(pooled + self.count_id(count))

        dyn = self.s_proj(c_dyn)
        team = torch.cat(
            [self.team_token.expand(b, -1, -1) + dyn.unsqueeze(1), pokemon], dim=1
        )
        team = self.team_encoder(team)  # no positional encoding: Pokemon order is equivariant
        global_ctx = team[:, 0]
        pokemon_ctx = team[:, 1:]
        return global_ctx, pokemon_ctx

    def _context_hidden(self, c_dyn: torch.Tensor, fed: torch.Tensor) -> torch.Tensor:
        """Encode an unordered partial team and return one query hidden per flat field [B,54,d]."""
        b = fed.shape[0]
        global_ctx, pokemon_ctx = self._set_context(c_dyn, fed)

        kind = self.field_kind.repeat(N_SLOTS)
        slot = torch.arange(N_SLOTS, device=fed.device).repeat_interleave(len(SLOT_ATTRS))
        q = self.query_id(kind).unsqueeze(0).expand(b, -1, -1)
        mon = pokemon_ctx[:, slot]
        glob = global_ctx.unsqueeze(1).expand(-1, N_ATTRS, -1)
        return self.query_proj(torch.cat([mon, glob, q], dim=-1))

    def _selected_context_hidden(
        self, c_dyn: torch.Tensor, fed: torch.Tensor, position: torch.Tensor
    ) -> torch.Tensor:
        """One query per row, avoiding 53 unused query tensors during sampling/scoring."""
        global_ctx, pokemon_ctx = self._set_context(c_dyn, fed)
        slot = torch.div(position, len(SLOT_ATTRS), rounding_mode="floor")
        column = position % len(SLOT_ATTRS)
        kind = self.field_kind[column]
        rows = torch.arange(len(fed), device=fed.device)
        query = self.query_id(kind)
        return self.query_proj(
            torch.cat([pokemon_ctx[rows, slot], global_ctx, query], dim=-1)
        )

    def _apply_mask(self, logits: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return logits
        ok = mask.any(dim=-1, keepdim=True)
        return logits.masked_fill(~mask & ok, _NEG_INF)

    def _move_set_mask(
        self,
        fed: torch.Tensor,
        slot: int,
        base: torch.Tensor | None,
    ) -> torch.Tensor:
        b = fed.shape[0]
        mask = (
            base.clone()
            if base is not None
            else torch.ones(
                b, self.sizes["move"], dtype=torch.bool, device=fed.device
            )
        )
        for p in _SLOT_MOVE_FLATS[slot]:
            val = fed[:, p]
            valid = (val >= 0) & (val < self.sizes["move"])
            if bool(valid.any()):
                rows = torch.nonzero(valid, as_tuple=True)[0]
                mask[rows, val[valid]] = False
        return mask

    def _delta_fill(
        self,
        logits: torch.Tensor,
        clamp: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        delta = torch.full_like(logits, _NEG_INF)
        delta.scatter_(1, value.clamp(min=0, max=logits.shape[1] - 1).unsqueeze(1), 0.0)
        return torch.where(clamp.unsqueeze(1), delta, logits)

    def _heads_from_context(
        self,
        hidden: torch.Tensor,
        tokens_flat: torch.Tensor,
        fed: torch.Tensor,
        clamp: torch.Tensor,
        masks: dict,
    ) -> dict[str, torch.Tensor]:
        b = hidden.shape[0]
        per_family: dict[str, list[torch.Tensor]] = {f: [] for f in self.sizes}
        for p in range(N_ATTRS):
            slot, fam = _POS_SLOT[p], _POS_FAMILY[p]
            logits = self.heads[fam](hidden[:, p])
            family_mask = masks.get(fam)
            legal = family_mask[:, slot] if family_mask is not None else None
            if fam == "move":
                legal = self._move_set_mask(fed, slot, legal)
            logits = self._apply_mask(logits, legal)
            logits = self._delta_fill(logits, clamp[:, p], tokens_flat[:, p])
            per_family[fam].append(logits)

        out: dict[str, torch.Tensor] = {}
        for fam, values in per_family.items():
            stacked = torch.stack(values, dim=1)
            if fam == "move":
                stacked = stacked.reshape(b, N_SLOTS, 4, self.sizes["move"])
            out[fam] = stacked
        return out

    @staticmethod
    def _split_flat(flat: torch.Tensor) -> dict[str, torch.Tensor]:
        grid = flat.reshape(len(flat), N_SLOTS, len(SLOT_ATTRS))
        return {
            "species": grid[:, :, 0],
            "item": grid[:, :, 1],
            "ability": grid[:, :, 2],
            "move": grid[:, :, 3:7],
            "spread": grid[:, :, 7],
            "tera": grid[:, :, 8],
        }

    # ------------------------------------------------------------------------------------- forward
    def forward(
        self,
        c_dyn: torch.Tensor,
        opp_revealed: dict,
        *,
        teacher_values: torch.Tensor | None = None,
        tail_order: str | None = None,
        generator: torch.Generator | None = None,
    ) -> BeliefOutput:
        """Marginals from the reveal set, or one matched set-AR training conditional per row.

        With ``teacher_values=None`` all hidden fields are predicted from all public reveals.
        With teacher values, a random generation route and prefix cut are sampled and only the
        next route field is marked in ``loss_valid``.  Callers must intersect that mask with their
        ordinary label/censoring mask; ``loss_targets`` carries the exchangeable move permutation.
        """
        tokens: torch.Tensor = opp_revealed["tokens"]
        masks: dict = opp_revealed.get("masks") or {}
        b = tokens.shape[0]
        tokens_flat = tokens.reshape(b, N_ATTRS)
        revealed = self._revealed_mask(tokens)
        fed = torch.where(revealed, tokens_flat, torch.full_like(tokens_flat, -1))

        if teacher_values is None:
            hidden = self._context_hidden(c_dyn, fed)
            logits = self._heads_from_context(
                hidden, tokens_flat, fed, revealed, masks
            )
            return BeliefOutput(logits=logits)

        mode = tail_order or "random"
        truth = teacher_values.reshape(b, N_ATTRS)
        truth = self._permute_hidden_moves(truth, revealed, mode, generator)
        target_valid = (truth >= 0) & (truth < self.pos_n)
        order, length, _ = self._route(revealed, mode, generator, skip=~target_valid)

        # Uniform cut over the actual hidden route.  Rows without any hidden fields have no chain
        # target and remain all-reveal delta rows.
        rand = torch.rand(b, device=tokens.device, generator=generator)
        cut = torch.floor(rand * length.clamp_min(1).to(rand.dtype)).to(torch.long)
        selected = torch.full((b,), -1, dtype=torch.long, device=tokens.device)
        rows = torch.arange(b, device=tokens.device)
        has_hidden = length > 0
        selected[has_hidden] = order[rows[has_hidden], cut[has_hidden]]

        known = revealed.clone()
        for step in range(N_ATTRS):
            add = has_hidden & (cut > step)
            if not bool(add.any()):
                continue
            ar = rows[add]
            pos = order[ar, step]
            valid_value = (truth[ar, pos] >= 0) & (truth[ar, pos] < self.pos_n[pos])
            if bool(valid_value.any()):
                vr = ar[valid_value]
                vp = pos[valid_value]
                fed[vr, vp] = truth[vr, vp]
                known[vr, vp] = True

        hidden = self._context_hidden(c_dyn, fed)
        # Clamp facts already in the context.  The next selected field remains a genuine prediction.
        logits = self._heads_from_context(hidden, fed, fed, known, masks)

        valid_flat = torch.zeros_like(known)
        sel_rows = rows[has_hidden]
        valid_flat[sel_rows, selected[has_hidden]] = True
        return BeliefOutput(
            logits=logits,
            loss_valid=self._split_flat(valid_flat),
            loss_targets=self._split_flat(truth),
            order=order,
        )

    # --------------------------------------------------------------------------- path probabilities
    @torch.no_grad()
    def path_logprob(
        self,
        c_dyn: torch.Tensor,
        opp_revealed: dict,
        team_values: torch.Tensor,
        *,
        order_mode: str = "random",
        generator: torch.Generator | None = None,
        include_route: bool = False,
        return_steps: bool = False,
    ):
        """Log probability of complete candidates along one sampled dependency-valid route.

        This is exact for that field route and one uniformly sampled ordering of each hidden move
        set.  For random scoring, the returned value includes the factorial importance correction
        that turns an average over move permutations into their sum.  Average ``exp(result)`` over
        independently sampled field routes to approximate the unordered-team marginal.  The
        scheduler term is constant across candidates sharing a reveal pattern, so candidate ranking
        normally leaves ``include_route=False``.
        """
        tokens: torch.Tensor = opp_revealed["tokens"]
        masks: dict = opp_revealed.get("masks") or {}
        b = tokens.shape[0]
        revealed = self._revealed_mask(tokens)
        truth = team_values.reshape(b, N_ATTRS)
        truth = self._permute_hidden_moves(truth, revealed, order_mode, generator)
        target_valid = (truth >= 0) & (truth < self.pos_n)
        # Move identities form an unordered set. `_permute_hidden_moves` samples one of m! possible
        # sequences uniformly; multiplying its probability by m! makes that single-sequence value
        # an unbiased estimate of the sum over sequences. The factor is constant for fully labeled
        # candidates under one reveal pattern, but retaining it keeps partial-label scores honest.
        move_permutation_log_factor = torch.zeros(
            b, dtype=torch.float32, device=tokens.device
        )
        if order_mode == "random":
            for slot in range(N_SLOTS):
                positions = list(_SLOT_MOVE_FLATS[slot])
                hidden_valid = target_valid[:, positions] & ~revealed[:, positions]
                count = hidden_valid.sum(dim=1).to(torch.float32)
                move_permutation_log_factor += torch.lgamma(count + 1.0)
        order, length, log_route = self._route(
            revealed, order_mode, generator, skip=~target_valid
        )
        initial = torch.where(
            revealed, tokens.reshape(b, N_ATTRS), torch.full_like(truth, -1)
        )
        contexts = torch.full(
            (b, N_ATTRS, N_ATTRS), -1, dtype=torch.long, device=tokens.device
        )
        fed = initial.clone()
        for step in range(N_ATTRS):
            active = step < length
            if not bool(active.any()):
                break
            contexts[active, step] = fed[active]
            pos = order[active, step]
            target = truth[active, pos]
            valid = (target >= 0) & (target < self.pos_n[pos])
            active_rows = torch.nonzero(active, as_tuple=True)[0]
            if bool(valid.any()):
                fed[active_rows[valid], pos[valid]] = target[valid]

        step_id = torch.arange(N_ATTRS, device=tokens.device).view(1, -1)
        active_ctx = step_id < length.view(-1, 1)
        route_row, route_step = torch.nonzero(active_ctx, as_tuple=True)
        context_flat = contexts[route_row, route_step]
        selected = order[route_row, route_step]
        target = truth[route_row, selected]

        log_step = torch.full(
            (len(route_row),), _NEG_INF, dtype=torch.float32, device=tokens.device
        )
        if len(route_row) == 0:
            empty = torch.zeros(b, N_ATTRS, device=tokens.device)
            total = move_permutation_log_factor
            if include_route:
                total = total + log_route
            return (total, empty, order, length) if return_steps else total

        was_training = self.training
        self.eval()
        try:
            selected_hidden = self._selected_context_hidden(
                c_dyn[route_row], context_flat, selected
            )
            for p in torch.unique(selected).tolist():
                use = selected == int(p)
                fam, slot = _POS_FAMILY[p], _POS_SLOT[p]
                logits = self.heads[fam](selected_hidden[use])
                family_mask = masks.get(fam)
                legal = (
                    family_mask[route_row[use], slot]
                    if family_mask is not None else None
                )
                if fam == "move":
                    legal = self._move_set_mask(context_flat[use], slot, legal)
                logits = self._apply_mask(logits, legal)
                target_p = target[use]
                target_ok = (target_p >= 0) & (target_p < logits.shape[-1])
                lp = torch.log_softmax(logits, dim=-1)
                got = lp.gather(
                    1, target_p.clamp(min=0, max=logits.shape[-1] - 1).unsqueeze(1)
                ).squeeze(1)
                finite = target_ok & torch.isfinite(got)
                log_step[use] = torch.where(
                    finite, got, torch.full_like(got, _NEG_INF)
                )
            log_value = torch.zeros(b, dtype=torch.float32, device=tokens.device)
            log_value.index_add_(0, route_row, log_step)
            per_field = torch.zeros(
                b, N_ATTRS, dtype=torch.float32, device=tokens.device
            )
            per_field[route_row, selected] = log_step
        finally:
            self.train(was_training)
        total = log_value + move_permutation_log_factor
        if include_route:
            total = total + log_route
        return (total, per_field, order, length) if return_steps else total

    # ------------------------------------------------------------------------------------ sampling
    @torch.no_grad()
    def sample_team(
        self,
        c_dyn: torch.Tensor,
        opp_revealed: dict,
        k: int = 16,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
        *,
        return_trace: bool = False,
    ) -> torch.Tensor | BeliefSamples:
        """Sample full teams by repeatedly inserting a fact into an unordered context set."""
        tokens0: torch.Tensor = opp_revealed["tokens"]
        masks: dict = opp_revealed.get("masks") or {}
        b = tokens0.shape[0]
        was_training = self.training
        self.eval()

        def rep(x: torch.Tensor) -> torch.Tensor:
            return x.repeat_interleave(k, dim=0)

        c_k = rep(c_dyn)
        tokens_k = rep(tokens0)
        masks_k = {name: rep(mask) for name, mask in masks.items()}
        revealed = self._revealed_mask(tokens_k)
        fed = torch.where(
            revealed, tokens_k.reshape(b * k, N_ATTRS), torch.full_like(
                tokens_k.reshape(b * k, N_ATTRS), -1
            )
        )
        known = revealed.clone()
        log_value = torch.zeros(b * k, device=fed.device)
        log_route = torch.zeros(b * k, device=fed.device)
        order = torch.full(
            (b * k, N_ATTRS), -1, dtype=torch.long, device=fed.device
        )
        rows = torch.arange(b * k, device=fed.device)

        try:
            for step in range(N_ATTRS):
                chosen, count, active = self._choose_next(known, "random", generator)
                if not bool(active.any()):
                    break
                order[active, step] = chosen[active]
                log_route[active] -= count[active].to(torch.float32).log()
                active_rows = rows[active]
                selected_hidden = self._selected_context_hidden(
                    c_k[active], fed[active], chosen[active]
                )
                for p in torch.unique(chosen[active]).tolist():
                    local_use = chosen[active] == int(p)
                    ur = active_rows[local_use]
                    fam, slot = _POS_FAMILY[p], _POS_SLOT[p]
                    logits = self.heads[fam](selected_hidden[local_use])
                    family_mask = masks_k.get(fam)
                    legal = family_mask[ur, slot] if family_mask is not None else None
                    if fam == "move":
                        legal = self._move_set_mask(fed[ur], slot, legal)
                    logits = self._apply_mask(logits, legal)
                    probs = torch.softmax(logits / max(float(temperature), 1e-6), dim=-1)
                    draw = torch.multinomial(probs, 1, generator=generator).squeeze(1)
                    fed[ur, p] = draw
                    log_value[ur] += torch.log(
                        probs.gather(1, draw.unsqueeze(1)).squeeze(1).clamp_min(1e-30)
                    )
                    known[ur, p] = True
        finally:
            self.train(was_training)

        teams = fed.reshape(b, k, N_SLOTS, len(SLOT_ATTRS))
        if not return_trace:
            return teams
        return BeliefSamples(
            teams=teams,
            log_value_prob=log_value.reshape(b, k),
            log_route_prob=log_route.reshape(b, k),
            order=order.reshape(b, k, N_ATTRS),
        )

    @staticmethod
    def _flat(pos: int) -> tuple[int, str, int | None]:
        return _POS_SLOT[pos], _POS_FAMILY[pos], _POS_SUB[pos]
