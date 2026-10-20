"""Permutation-equivariant set-autoregressive belief model.

Public reveals and generated assumptions are separately tagged unordered sets.
Facts are transformed and pooled within each Pokemon before a Transformer mixes
the six Pokemon without positional encodings. Reordering moves or facts leaves
the distribution unchanged; reordering Pokemon permutes its output slots.

Generation adds one fact at a time. A value-independent scheduler samples
uniformly among eligible fields: species precedes species-dependent legality
and nature/EV fields, and the EV mask caps the total at 127 units (508 EVs).
The four move fields share a query and value head; storage uses the next free
move column.

Training samples a route and prefix over recoverable replay fields, then
predicts the next fact. Missing replay labels remain absent. `query_field`
can address any labeled slot without treating censored data as a full team.

Scheduler and conditional-value probabilities define a normalized distribution
over (route, completed team). Team probabilities marginalize over routes;
candidate scoring approximates this with sampled routes. Sampling returns
both log-probabilities when `return_trace=True`."""
from __future__ import annotations

import torch
from torch import nn

from fog.constants.schema import (
    MAX_TOTAL_EV_UNITS,
    N_EV_UNITS,
    N_NATURES,
    SET_EV_COLS,
    SET_MOVE_COLS,
    SET_NATURE_COL,
    SET_SLOT_ATTRS,
    SET_TERA_COL,
)
from fog.models.belief_types import BeliefOutput, BeliefSamples

SLOT_ATTRS = SET_SLOT_ATTRS
N_SLOTS = 6
N_ATTRS = N_SLOTS * len(SLOT_ATTRS)
SEQ_LEN = N_ATTRS + 1
_START = N_ATTRS

_ATTR_FAMILY = (
    "species", "item", "ability", "move", "move", "move", "move",
    "nature", "ev", "ev", "ev", "ev", "ev", "ev", "tera",
)
_FIELD_KIND = (0, 1, 2, 3, 3, 3, 3, 4, 5, 5, 5, 5, 5, 5, 6)
_MOVE_SUBSLOT = (None, None, None, 0, 1, 2, 3, None, None, None, None, None, None, None, None)
_MOVE_COLS = SET_MOVE_COLS
_EV_COLS = SET_EV_COLS
_EV_SUBINDEX = tuple(
    _EV_COLS.index(column) if column in _EV_COLS else -1
    for column in range(len(SLOT_ATTRS))
)
_ITEM_COL, _ABILITY_COL, _NATURE_COL, _TERA_COL = 1, 2, SET_NATURE_COL, SET_TERA_COL

_POS_SLOT = tuple(p // len(SLOT_ATTRS) for p in range(N_ATTRS))
_POS_FAMILY = tuple(_ATTR_FAMILY[p % len(SLOT_ATTRS)] for p in range(N_ATTRS))
_POS_SUB = tuple(_MOVE_SUBSLOT[p % len(SLOT_ATTRS)] for p in range(N_ATTRS))
_SLOT_MOVE_FLATS = tuple(
    tuple(j * len(SLOT_ATTRS) + c for c in _MOVE_COLS) for j in range(N_SLOTS)
)
_SLOT_ITEM_FLATS = tuple(j * len(SLOT_ATTRS) + _ITEM_COL for j in range(N_SLOTS))
_SLOT_ABILITY_FLATS = tuple(j * len(SLOT_ATTRS) + _ABILITY_COL for j in range(N_SLOTS))
_SLOT_NATURE_FLATS = tuple(j * len(SLOT_ATTRS) + _NATURE_COL for j in range(N_SLOTS))
_SLOT_EV_FLATS = tuple(
    tuple(j * len(SLOT_ATTRS) + c for c in _EV_COLS) for j in range(N_SLOTS)
)

_NEG_INF = float("-inf")


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
        n_natures: int = N_NATURES,
        n_ev_units: int = N_EV_UNITS,
        n_tera: int = 19,
        last_context_probability: float = 0.0,
    ) -> None:
        super().__init__()
        self.d = d
        self.nhead = 4
        self.sizes = {
            "species": n_species,
            "item": n_items,
            "ability": n_abilities,
            "move": n_moves,
            "nature": n_natures,
            "ev": n_ev_units,
            "tera": n_tera,
        }
        # Item labels describe the submitted set. <NO_ITEM> is a public CURRENT-state fact after
        # consumption/Knock Off, not the Pokemon's original set item. Keep it in the board/entity
        # encoding, but do not clamp the set-AR item field to that sentinel. The exact removed item
        # remains available through categorical battle history when Showdown revealed it.
        self.no_item_index = n_items - 2
        self.last_context_probability = float(last_context_probability)
        if not 0.0 <= self.last_context_probability <= 1.0:
            raise ValueError("last_context_probability must be in [0,1]")

        # These names intentionally survive from the preceding architecture.  A warm-start loader
        # can retain learned value embeddings, family heads, and the dynamics projection while the
        # obsolete ordered decoder is discarded.
        self.s_proj = nn.Linear(state_dim, d)
        self.heads = nn.ModuleDict({name: nn.Linear(d, n) for name, n in self.sizes.items()})
        # Stat-specific residuals let the six very different EV marginals specialize directly.
        # All-zero initialization keeps compatible checkpoint outputs exactly unchanged.
        self.ev_stat_heads = nn.ModuleList(
            nn.Linear(d, n_ev_units) for _ in range(len(_EV_COLS))
        )
        for head in self.ev_stat_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.value_embeds = nn.ModuleDict(
            {name: nn.Embedding(n + 1, d) for name, n in self.sizes.items()}
        )

        # Seven semantic field identities; move columns and EV-stat columns share identities.
        self.field_id = nn.Embedding(7, d)
        # EV amount is not exchangeable across stats: 252 Atk and 252 Spe mean different things.
        # Zero initialization preserves old checkpoint outputs under compatible loading while
        # continuation training learns the six stat roles.
        self.ev_stat_id = nn.Embedding(len(_EV_COLS), d)
        nn.init.zeros_(self.ev_stat_id.weight)
        # Public observations and hypothetical chain assignments are both facts, but they are not
        # the same evidence source.  Keeping this distinction lets the model learn how much to trust
        # an ancestral assumption without ever hiding or weakening a real public reveal.
        self.source_id = nn.Embedding(2, d)  # 0 = public reveal, 1 = generated/teacher assumption
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
        self.team_encoder = nn.TransformerEncoder(
            team_layer,
            num_layers=2,
            # ``norm_first=True`` is intentional and cannot use PyTorch's
            # nested-tensor fast path.
            enable_nested_tensor=False,
        )
        self.team_token = nn.Parameter(torch.empty(1, 1, d))
        nn.init.normal_(self.team_token, std=0.02)

        self.query_id = nn.Embedding(7, d)
        self.query_proj = nn.Sequential(
            nn.LayerNorm(3 * d), nn.Linear(3 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, d)
        )

        pos_n = torch.tensor([self.sizes[_POS_FAMILY[p]] for p in range(N_ATTRS)])
        self.register_buffer("pos_n", pos_n.to(torch.long), persistent=False)
        self.register_buffer(
            "field_kind", torch.tensor(_FIELD_KIND, dtype=torch.long), persistent=False
        )

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs,
    ):
        # Early typed checkpoints have the shared EV head but no stat residuals. Only migrate
        # that complete old layout; a partially missing residual group is still an error.
        residuals = {
            name: value for name, value in self.state_dict().items()
            if name.startswith(("ev_stat_heads.", "ev_stat_id."))
        }
        if prefix + "heads.ev.weight" in state_dict and not any(
            prefix + name in state_dict for name in residuals
        ):
            for name, value in residuals.items():
                state_dict[prefix + name] = torch.zeros_like(value)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs,
        )

    # -------------------------------------------------------------------------- set/route mechanics
    def _revealed_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        flat = tokens.reshape(tokens.shape[0], N_ATTRS)
        n = self.pos_n.to(flat.device)
        revealed = (flat >= 0) & (flat < n)
        for position in _SLOT_ITEM_FLATS:
            revealed[:, position] &= flat[:, position] != self.no_item_index
        return revealed

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
            out[:, base] = ~species
            out[:, base + _ITEM_COL] = ~known[:, base + _ITEM_COL]
            out[:, base + _TERA_COL] = ~known[:, base + _TERA_COL]  # no prerequisite

            species_known = species
            out[:, base + _ABILITY_COL] = species_known & ~known[:, base + _ABILITY_COL]

            move_known = known[:, [base + c for c in _MOVE_COLS]]
            for sub, col in enumerate(_MOVE_COLS):
                # Only the first unresolved move storage cell represents the next set insertion.
                earlier_full = move_known[:, :sub].all(dim=1) if sub else torch.ones(
                    b, dtype=torch.bool, device=known.device
                )
                out[:, base + col] = species_known & ~move_known[:, sub] & earlier_full

            out[:, base + _NATURE_COL] = species_known & ~known[:, base + _NATURE_COL]
            for col in _EV_COLS:
                out[:, base + col] = species_known & ~known[:, base + col]
        return out

    def _training_route(
        self,
        revealed: torch.Tensor,
        target_valid: torch.Tensor,
        mode: str,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Arbitrary replay-query route over the labeled, currently hidden fields.

        This deliberately does *not* use the ancestral scheduler.  A replay may reveal any
        slot/field next, so training samples a random permutation of every recoverable hidden fact
        and a random prefix of that permutation.  The prefix is hypothetical chain context and the
        next entry is the addressed query.  Missing replay labels never become values or negatives.
        """
        available = target_valid & ~revealed
        length = available.sum(dim=1)
        if mode == "canonical":
            score = torch.arange(N_ATTRS, device=revealed.device).expand(len(revealed), -1)
            score = score.masked_fill(~available, N_ATTRS + 1)
        elif mode == "random":
            score = torch.rand(
                len(revealed), N_ATTRS, device=revealed.device, generator=generator
            ).masked_fill(~available, 2.0)
        else:
            raise ValueError(f"order mode must be 'random' or 'canonical', got {mode!r}")
        raw = score.argsort(dim=1)
        steps = torch.arange(N_ATTRS, device=revealed.device).view(1, -1)
        return raw.masked_fill(steps >= length.view(-1, 1), -1), length

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

        Returns order [B,N_ATTRS] padded with -1, route lengths, and scheduler log-probability.
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
    def _fact_embeddings(
        self, fed: torch.Tensor, source: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return transformed facts [B,6,15,d] and a known-value mask [B,6,15]."""
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
        fact = fact + self.field_id(self.field_kind).view(
            1, 1, len(SLOT_ATTRS), self.d
        )
        ev_subindex = torch.tensor(_EV_SUBINDEX, device=fed.device)
        ev_columns = ev_subindex >= 0
        ev_identity = torch.zeros(
            len(SLOT_ATTRS), self.d, device=fed.device, dtype=fact.dtype
        )
        ev_identity[ev_columns] = self.ev_stat_id(ev_subindex[ev_columns])
        fact = fact + ev_identity.view(1, 1, len(SLOT_ATTRS), self.d)
        if source is None:
            source = torch.zeros_like(fed)
        fact = fact + self.source_id(source.clamp(min=0, max=1))
        return self.fact_proj(fact), known

    def _set_context(
        self,
        belief_context: torch.Tensor,
        fed: torch.Tensor,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode an unordered partial team -> (global [B,d], Pokemon [B,6,d])."""
        b = fed.shape[0]
        grid = fed.reshape(b, N_SLOTS, len(SLOT_ATTRS))
        source_grid = None if source is None else source.reshape_as(grid)
        fact, known = self._fact_embeddings(grid, source_grid)
        count = known.sum(dim=2)
        pooled = (fact * known.unsqueeze(-1)).sum(dim=2)
        pooled = pooled / count.clamp_min(1).unsqueeze(-1)
        pokemon = self.mon_proj(pooled + self.count_id(count))

        dyn = self.s_proj(belief_context)
        if dyn.ndim == 3:
            if tuple(dyn.shape[1:2]) != (N_SLOTS,):
                raise ValueError("slot-aware belief context must have shape [batch,6,state_dim]")
            global_dyn = dyn.mean(dim=1)
            pokemon = pokemon + dyn
        elif dyn.ndim == 2:
            global_dyn = dyn
        else:
            raise ValueError("belief context must have shape [batch,state_dim] or [batch,6,state_dim]")
        team = torch.cat(
            [self.team_token.expand(b, -1, -1) + global_dyn.unsqueeze(1), pokemon], dim=1
        )
        team = self.team_encoder(team)  # no positional encoding: Pokemon order is equivariant
        global_ctx = team[:, 0]
        pokemon_ctx = team[:, 1:]
        return global_ctx, pokemon_ctx

    def _context_hidden(
        self,
        belief_context: torch.Tensor,
        fed: torch.Tensor,
        source: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode an unordered partial team and return one query hidden per flat field."""
        b = fed.shape[0]
        global_ctx, pokemon_ctx = self._set_context(belief_context, fed, source)

        kind = self.field_kind.repeat(N_SLOTS)
        column = torch.arange(len(SLOT_ATTRS), device=fed.device).repeat(N_SLOTS)
        slot = torch.arange(N_SLOTS, device=fed.device).repeat_interleave(len(SLOT_ATTRS))
        q = self.query_id(kind).unsqueeze(0).expand(b, -1, -1)
        ev_subindex = torch.tensor(_EV_SUBINDEX, device=fed.device)[column]
        ev_columns = ev_subindex >= 0
        if bool(ev_columns.any()):
            q = q.clone()
            q[:, ev_columns] += self.ev_stat_id(ev_subindex[ev_columns])
        mon = pokemon_ctx[:, slot]
        glob = global_ctx.unsqueeze(1).expand(-1, N_ATTRS, -1)
        return self.query_proj(torch.cat([mon, glob, q], dim=-1))

    def _selected_context_hidden(
        self,
        belief_context: torch.Tensor,
        fed: torch.Tensor,
        position: torch.Tensor,
        source: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One query per row, avoiding 53 unused query tensors during sampling/scoring."""
        global_ctx, pokemon_ctx = self._set_context(belief_context, fed, source)
        slot = torch.div(position, len(SLOT_ATTRS), rounding_mode="floor")
        column = position % len(SLOT_ATTRS)
        kind = self.field_kind[column]
        rows = torch.arange(len(fed), device=fed.device)
        query = self.query_id(kind)
        ev_subindex = torch.tensor(_EV_SUBINDEX, device=fed.device)[column]
        ev_columns = ev_subindex >= 0
        if bool(ev_columns.any()):
            query = query.clone()
            query[ev_columns] += self.ev_stat_id(ev_subindex[ev_columns])
        return self.query_proj(
            torch.cat([pokemon_ctx[rows, slot], global_ctx, query], dim=-1)
        )

    def _apply_mask(self, logits: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return logits
        ok = mask.any(dim=-1, keepdim=True)
        return logits.masked_fill(~mask & ok, _NEG_INF)

    def _value_logits(
        self,
        family: str,
        hidden: torch.Tensor,
        *,
        ev_subslot: int | None = None,
    ) -> torch.Tensor:
        logits = self.heads[family](hidden)
        if family == "ev":
            if ev_subslot is None:
                raise ValueError("EV logits require an explicit stat subslot")
            logits = logits + self.ev_stat_heads[int(ev_subslot)](hidden)
        return logits

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

    def _ev_budget_mask(
        self,
        fed: torch.Tensor,
        slot: int,
        position: int,
        base: torch.Tensor | None,
    ) -> torch.Tensor:
        """Legal EV-unit values for one stat after accounting for the other five stats."""
        b = fed.shape[0]
        mask = (
            base.clone()
            if base is not None
            else torch.ones(b, self.sizes["ev"], dtype=torch.bool, device=fed.device)
        )
        other = [p for p in _SLOT_EV_FLATS[slot] if p != position]
        values = fed[:, other]
        valid = (values >= 0) & (values < self.sizes["ev"])
        used = torch.where(valid, values, torch.zeros_like(values)).sum(dim=1)
        remaining = (MAX_TOTAL_EV_UNITS - used).clamp(
            min=0, max=self.sizes["ev"] - 1
        )
        candidates = torch.arange(self.sizes["ev"], device=fed.device).unsqueeze(0)
        return mask & (candidates <= remaining.unsqueeze(1))

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
        prior=None,
    ) -> dict[str, torch.Tensor]:
        b = hidden.shape[0]
        per_family: dict[str, list[torch.Tensor]] = {f: [] for f in self.sizes}
        for p in range(N_ATTRS):
            slot, fam = _POS_SLOT[p], _POS_FAMILY[p]
            ev_subslot = _EV_COLS.index(p % len(SLOT_ATTRS)) if fam == "ev" else None
            logits = self._value_logits(fam, hidden[:, p], ev_subslot=ev_subslot)
            family_mask = masks.get(fam)
            legal = family_mask[:, slot] if family_mask is not None else None
            if fam == "move":
                legal = self._move_set_mask(fed, slot, legal)
            elif fam == "ev":
                legal = self._ev_budget_mask(fed, slot, p, legal)
            logits = self._apply_mask(logits, legal)
            if prior is not None:
                logits = prior.condition(logits, fed, p)
            logits = self._delta_fill(logits, clamp[:, p], tokens_flat[:, p])
            per_family[fam].append(logits)

        out: dict[str, torch.Tensor] = {}
        for fam, values in per_family.items():
            stacked = torch.stack(values, dim=1)
            if fam == "move":
                stacked = stacked.reshape(b, N_SLOTS, 4, self.sizes["move"])
            elif fam == "ev":
                stacked = stacked.reshape(b, N_SLOTS, 6, self.sizes["ev"])
            out[fam] = stacked
        return out

    def _selected_heads(self, hidden, fed, positions, active, masks, prior):
        """Only addressed queries; other entries are zero and explicitly invalid."""
        b = len(fed)
        counts = {family: _POS_FAMILY.count(family) for family in self.sizes}
        out = {family: hidden.new_zeros(b, count, self.sizes[family]) for family, count in counts.items()}
        for position in torch.unique(positions[active]).tolist():
            p = int(position)
            use = active & (positions == p)
            family, slot = _POS_FAMILY[p], _POS_SLOT[p]
            column = p % len(SLOT_ATTRS)
            ev_subslot = _EV_COLS.index(column) if family == "ev" else None
            logits = self._value_logits(family, hidden[use], ev_subslot=ev_subslot)
            base = masks.get(family)
            legal = base[use, slot] if base is not None else None
            if family == "move":
                legal = self._move_set_mask(fed[use], slot, legal)
            elif family == "ev":
                legal = self._ev_budget_mask(fed[use], slot, p, legal)
            logits = self._apply_mask(logits, legal)
            if prior is not None:
                logits = prior.index_select(use).condition(logits, fed[use], p)
            sub = column - 3 if family == "move" else ev_subslot
            index = slot if sub is None else slot * (4 if family == "move" else 6) + sub
            out[family][use, index] = logits
        out["move"] = out["move"].reshape(b, 6, 4, self.sizes["move"])
        out["ev"] = out["ev"].reshape(b, 6, 6, self.sizes["ev"])
        return out

    @staticmethod
    def _split_flat(flat: torch.Tensor) -> dict[str, torch.Tensor]:
        grid = flat.reshape(len(flat), N_SLOTS, len(SLOT_ATTRS))
        return {
            "species": grid[:, :, 0],
            "item": grid[:, :, 1],
            "ability": grid[:, :, 2],
            "move": grid[:, :, 3:7],
            "nature": grid[:, :, _NATURE_COL],
            "ev": grid[:, :, list(_EV_COLS)],
            "tera": grid[:, :, _TERA_COL],
        }

    # ------------------------------------------------------------------------------------- forward
    def forward(
        self,
        belief_context: torch.Tensor,
        opp_revealed: dict,
        *,
        teacher_values: torch.Tensor | None = None,
        tail_order: str | None = None,
        generator: torch.Generator | None = None,
        selected_only: bool = False,
    ) -> BeliefOutput:
        """Marginals from the reveal set, or one matched set-AR training conditional per row.

        With ``teacher_values=None`` all hidden fields are predicted from all public reveals.
        With teacher values, an arbitrary random ordering of the replay-labeled hidden fields and
        a prefix cut are sampled.  The prefix is tagged as generated assumptions and only the next
        addressed field is marked in ``loss_valid``.  Callers must intersect that mask with their
        ordinary label/censoring mask; ``loss_targets`` carries the exchangeable move permutation.
        """
        tokens: torch.Tensor = opp_revealed["tokens"]
        masks: dict = opp_revealed.get("masks") or {}
        prior = opp_revealed.get("prior")
        b = tokens.shape[0]
        tokens_flat = tokens.reshape(b, N_ATTRS)
        revealed = self._revealed_mask(tokens)
        fed = torch.where(revealed, tokens_flat, torch.full_like(tokens_flat, -1))
        source = torch.zeros_like(fed)

        if teacher_values is None:
            hidden = self._context_hidden(belief_context, fed, source)
            logits = self._heads_from_context(
                hidden, tokens_flat, fed, revealed, masks, prior
            )
            return BeliefOutput(logits=logits)

        mode = tail_order or "random"
        truth = teacher_values.reshape(b, N_ATTRS)
        truth = self._permute_hidden_moves(truth, revealed, mode, generator)
        target_valid = (truth >= 0) & (truth < self.pos_n)
        order, length = self._training_route(revealed, target_valid, mode, generator)

        # Uniform cut over the actual hidden route.  Rows without any hidden fields have no chain
        # target and remain all-reveal delta rows.
        rand = torch.rand(b, device=tokens.device, generator=generator)
        cut = torch.floor(rand * length.clamp_min(1).to(rand.dtype)).to(torch.long)
        if self.last_context_probability > 0:
            choose_last = torch.rand(
                b, device=tokens.device, generator=generator
            ) < self.last_context_probability
            cut = torch.where(choose_last & (length > 0), length - 1, cut)
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
                source[vr, vp] = 1

        if selected_only:
            hidden = self._selected_context_hidden(belief_context, fed, selected.clamp_min(0), source)
            logits = self._selected_heads(hidden, fed, selected, has_hidden, masks, prior)
        else:
            hidden = self._context_hidden(belief_context, fed, source)
            # Clamp facts already in the context. The selected field remains a genuine prediction.
            logits = self._heads_from_context(hidden, fed, fed, known, masks, prior)

        valid_flat = torch.zeros_like(known)
        sel_rows = rows[has_hidden]
        valid_flat[sel_rows, selected[has_hidden]] = True
        return BeliefOutput(
            logits=logits,
            loss_valid=self._split_flat(valid_flat),
            loss_targets=self._split_flat(truth),
            order=order,
            context_count=cut,
            route_length=length,
        )

    def query_field(
        self,
        belief_context: torch.Tensor,
        opp_revealed: dict,
        *,
        slot: int,
        field: str,
        context_values: torch.Tensor | None = None,
        subslot: int | None = None,
    ) -> torch.Tensor:
        """Query one named fact for one Pokemon under an arbitrary partial fact set.

        This is the explicit censored-replay interface. ``context_values`` is an optional
        ``[B,6,15]`` grid with ``-1`` for unavailable facts. Public reveals always win, and the
        requested hidden field is removed from teacher context so its answer cannot leak into its
        own logits. Unlike the ancestral scheduler this query is allowed for any slot/field; it is
        a conditional auxiliary objective, not a claim that censored rows define a normalized
        complete-team likelihood.

        Moves are a set, so ``field="move"`` means "another move for this Pokemon", not one
        semantically ordered moveset column.
        """
        if not 0 <= int(slot) < N_SLOTS:
            raise ValueError(f"slot must be in [0,{N_SLOTS}), got {slot!r}")
        if field not in self.sizes:
            raise ValueError(f"unknown belief field {field!r}; expected one of {tuple(self.sizes)}")

        tokens: torch.Tensor = opp_revealed["tokens"]
        masks: dict = opp_revealed.get("masks") or {}
        b = tokens.shape[0]
        public = tokens.reshape(b, N_ATTRS)
        revealed = self._revealed_mask(tokens)
        fed = torch.where(revealed, public, torch.full_like(public, -1))
        source = torch.zeros_like(fed)

        if context_values is not None:
            supplied = context_values.reshape(b, N_ATTRS).to(
                device=tokens.device, dtype=tokens.dtype
            )
            valid = (supplied >= 0) & (supplied < self.pos_n)
            fed = torch.where(~revealed & valid, supplied, fed)
            source = torch.where(~revealed & valid, torch.ones_like(source), source)

        if field == "move":
            # External move columns have no identity. Query the first currently unresolved storage
            # cell, while all known move values remain available as a set in the context.
            base = int(slot) * len(SLOT_ATTRS)
            move_known = (fed[:, [base + c for c in _MOVE_COLS]] >= 0)
            counts = move_known.sum(dim=1).clamp(max=len(_MOVE_COLS) - 1)
            positions = counts + base + _MOVE_COLS[0]
        elif field == "ev":
            if subslot is None:
                base = int(slot) * len(SLOT_ATTRS)
                ev_known = fed[:, [base + c for c in _EV_COLS]] >= 0
                counts = ev_known.sum(dim=1).clamp(max=len(_EV_COLS) - 1)
                positions = counts + base + _EV_COLS[0]
            else:
                if not 0 <= int(subslot) < len(_EV_COLS):
                    raise ValueError(
                        f"EV subslot must be in [0,{len(_EV_COLS)}), got {subslot!r}"
                    )
                positions = torch.full(
                    (b,), int(slot) * len(SLOT_ATTRS) + _EV_COLS[int(subslot)],
                    dtype=torch.long, device=tokens.device,
                )
        else:
            column = _ATTR_FAMILY.index(field)
            positions = torch.full(
                (b,), int(slot) * len(SLOT_ATTRS) + column,
                dtype=torch.long, device=tokens.device,
            )

        rows = torch.arange(b, device=tokens.device)
        public_target = revealed[rows, positions]
        # A hidden teacher value for the requested target must never enter its own context.
        fed[rows[~public_target], positions[~public_target]] = -1
        hidden = self._selected_context_hidden(belief_context, fed, positions, source)
        if field == "ev":
            ev_subslots = torch.tensor(_EV_SUBINDEX, device=tokens.device)[
                positions % len(SLOT_ATTRS)
            ]
            logits = torch.empty(
                b, self.sizes["ev"], device=hidden.device, dtype=hidden.dtype
            )
            for ev_subslot in torch.unique(ev_subslots).tolist():
                use = ev_subslots == int(ev_subslot)
                logits[use] = self._value_logits(
                    "ev", hidden[use], ev_subslot=int(ev_subslot)
                )
        else:
            logits = self._value_logits(field, hidden)

        family_mask = masks.get(field)
        legal = family_mask[:, int(slot)] if family_mask is not None else None
        if field == "move":
            legal = self._move_set_mask(fed, int(slot), legal)
        elif field == "ev":
            budget = torch.ones_like(logits, dtype=torch.bool)
            for p in torch.unique(positions).tolist():
                use = positions == int(p)
                budget[use] = self._ev_budget_mask(fed[use], int(slot), int(p), None)
            legal = budget if legal is None else (legal & budget)
        logits = self._apply_mask(logits, legal)
        prior = opp_revealed.get("prior")
        if prior is not None:
            logits = logits.clone()
            for position in torch.unique(positions).tolist():
                use = positions == int(position)
                logits[use] = prior.index_select(use).condition(logits[use], fed[use], int(position))
        if bool(public_target.any()):
            logits = self._delta_fill(logits, public_target, public[rows, positions])
        return logits

    # --------------------------------------------------------------------------- path probabilities
    @torch.no_grad()
    def path_logprob(
        self,
        belief_context: torch.Tensor,
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
        initial_source = torch.zeros_like(initial)
        contexts = torch.full(
            (b, N_ATTRS, N_ATTRS), -1, dtype=torch.long, device=tokens.device
        )
        context_sources = torch.zeros_like(contexts)
        fed = initial.clone()
        source = initial_source.clone()
        for step in range(N_ATTRS):
            active = step < length
            if not bool(active.any()):
                break
            contexts[active, step] = fed[active]
            context_sources[active, step] = source[active]
            pos = order[active, step]
            target = truth[active, pos]
            valid = (target >= 0) & (target < self.pos_n[pos])
            active_rows = torch.nonzero(active, as_tuple=True)[0]
            if bool(valid.any()):
                fed[active_rows[valid], pos[valid]] = target[valid]
                source[active_rows[valid], pos[valid]] = 1

        step_id = torch.arange(N_ATTRS, device=tokens.device).view(1, -1)
        active_ctx = step_id < length.view(-1, 1)
        route_row, route_step = torch.nonzero(active_ctx, as_tuple=True)
        context_flat = contexts[route_row, route_step]
        source_flat = context_sources[route_row, route_step]
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
                belief_context[route_row], context_flat, selected, source_flat
            )
            for p in torch.unique(selected).tolist():
                use = selected == int(p)
                fam, slot = _POS_FAMILY[p], _POS_SLOT[p]
                ev_subslot = _EV_COLS.index(p % len(SLOT_ATTRS)) if fam == "ev" else None
                logits = self._value_logits(
                    fam, selected_hidden[use], ev_subslot=ev_subslot
                )
                family_mask = masks.get(fam)
                legal = (
                    family_mask[route_row[use], slot]
                    if family_mask is not None else None
                )
                if fam == "move":
                    legal = self._move_set_mask(context_flat[use], slot, legal)
                elif fam == "ev":
                    legal = self._ev_budget_mask(context_flat[use], slot, int(p), legal)
                logits = self._apply_mask(logits, legal)
                prior = opp_revealed.get("prior")
                if prior is not None:
                    logits = prior.index_select(route_row[use]).condition(logits, context_flat[use], int(p))
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
        belief_context: torch.Tensor,
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

        c_k = rep(belief_context)
        tokens_k = rep(tokens0)
        masks_k = {name: rep(mask) for name, mask in masks.items()}
        prior = opp_revealed.get("prior")
        prior_k = prior.repeat_interleave(k) if prior is not None else None
        revealed = self._revealed_mask(tokens_k)
        fed = torch.where(
            revealed, tokens_k.reshape(b * k, N_ATTRS), torch.full_like(
                tokens_k.reshape(b * k, N_ATTRS), -1
            )
        )
        source = torch.zeros_like(fed)
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
                    c_k[active], fed[active], chosen[active], source[active]
                )
                for p in torch.unique(chosen[active]).tolist():
                    local_use = chosen[active] == int(p)
                    ur = active_rows[local_use]
                    fam, slot = _POS_FAMILY[p], _POS_SLOT[p]
                    ev_subslot = _EV_COLS.index(p % len(SLOT_ATTRS)) if fam == "ev" else None
                    logits = self._value_logits(
                        fam, selected_hidden[local_use], ev_subslot=ev_subslot
                    )
                    family_mask = masks_k.get(fam)
                    legal = family_mask[ur, slot] if family_mask is not None else None
                    if fam == "move":
                        legal = self._move_set_mask(fed[ur], slot, legal)
                    elif fam == "ev":
                        legal = self._ev_budget_mask(fed[ur], slot, int(p), legal)
                    logits = self._apply_mask(logits, legal)
                    if prior_k is not None:
                        logits = prior_k.index_select(ur).condition(logits, fed[ur], int(p))
                    probs = torch.softmax(logits / max(float(temperature), 1e-6), dim=-1)
                    draw = torch.multinomial(probs, 1, generator=generator).squeeze(1)
                    fed[ur, p] = draw
                    source[ur, p] = 1
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
