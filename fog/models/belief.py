"""Battle-conditioned opponent-team belief model: prefix-LM over the revealed-attribute SET plus a
causal chain over the hidden attributes (BELIEF_POLICY_PLAN §3.3; the chain's tail is canonical-order by
default per BELIEF_POLICY_PLAN §3.7 option (c) -- see TAIL ORDER below).

WHY THIS SHAPE. The 54 opponent attributes (6 slots x 9 = SLOT_ATTRS) split, per example, into the
ones already revealed this battle (a set R) and the ones still hidden (H). Every hidden-attribute
prediction must condition on ALL of R regardless of where it sits in the generation order (the owner
requirement) -- a fixed left-to-right chain cannot do that, because reveals that live late in the
chain (a revealed 4th move, a revealed tera) can never inform an earlier prediction (item, ability).
So R is laid out as an order-free prefix with bidirectional attention among its members, and H is a
causal tail AFTER the whole prefix. Each hidden conditional then sees every reveal exactly.

SEQUENCE LAYOUT (fixed length 55, no ragged shapes). Per example with |R| revealed attrs:

    physical:  [ r_1 ... r_|R| ] [START] [ h_1 ... h_|H| ]        |R| + 1 + |H| = 55 always
    attention: prefix + START   -> attend {all prefix, START}      (one bidirectional block)
               tail rank k       -> attend {prefix, START, tail ranks <= k}   (causal tail)

Because the tail sits physically after the prefix+START block, "attend j <= i" (ordinary causal) for a
tail query already includes the entire prefix; the only non-causal part is the prefix+START block,
which is fully bidirectional internally. See `_attn_mask`.

IDENTITY IS EXPLICIT, NOT POSITIONAL (owner revision 4). A token's input embedding is
`value_emb_fam(value)` (or the family's learned <UNK> row when a tail token carries no value) PLUS
`attr_id(slot*9 + attr)` -- the existing 54-way `attr_id` table, reused as an attribute-identity
embedding. Every token, including every order-free reveal token, therefore carries both its team slot
and its attribute type. Nothing encodes physical position; tail order is a property of the attention
mask + read shift, not of any positional embedding.

READ SHIFT (GPT teacher forcing). The conditional for tail rank k is read at the PREVIOUS physical
position -- START for k=1, tail token k-1 otherwise -- so a token never sees its own value. Prefix
attrs are never read; they are delta-filled (0 at the revealed value, -inf elsewhere), which makes
every downstream consumer correct for free (leakage masks them out; belief-loss on a revealed attr is
0/dropped; pool_id gets a hard consistency filter).

CONDITIONING MEMORY (BELIEF_POLICY_PLAN §3.1). The decoder cross-attends to a single vector `c_dyn`
-- the dynamics/behavioral context from the backbone (history + current field), NOT the old full
board summary `s`. The reveals reach the chain as prefix TOKENS; c_dyn carries only the temporal
evidence the tokens cannot ("he switched out, so he probably lacks this move"). c_dyn is the same
tensor the policy head consumes, so the belief aux loss shapes it for both consumers.

TAIL ORDER (BELIEF_POLICY_PLAN §3.7, option (c)). The tail defaults to the CANONICAL order in both
modes. The chain once defaulted to a fresh per-example RANDOM order (the "any-order" tail, so a hidden
attribute could condition on a hidden source drawn earlier in a random permutation). The gate proved
that costs sharpness: `sample_team` walks the CANONICAL tail, so the random-order chain trained a
regime the sampler never visits and ancestral samples hedged (rule fidelity 0.30). A canonical chain
trains exactly the conditionals the sampler traverses (fidelity -> 1.00). What is given up is
conditioning a hidden attr on a hidden attr later in canonical order; reveal-conditioning is untouched
because it lives in the bidirectional PREFIX, before every tail token, whatever the tail order. The
random tail stays selectable (`tail_order="random"`) for the pool_id §4.5.3 arm and the gate ablation.

TWO FORWARD MODES, both through this one module:
  * Marginal (no teacher_values) -- deterministic canonical tail, every tail value <UNK>. Produces the
    marginal conditionals p(h | R). No longer on the policy's critical path (BELIEF_POLICY_PLAN §3.6
    removes the mean-pooled b_vec); kept as an on-demand diagnostic for pool_id's fast path, the
    belief dashboards, and the leakage probe.
  * Chain (teacher_values set) -- teacher-forced training on the canonical tail (default). Tail = H in
    canonical order; FULL teacher forcing (every tail token whose label exists carries its TRUE value,
    later tokens causally masked from each read). Each tail rank's loss is the exact chain-rule factor
    p(h_k = truth | R, h_1..h_{k-1} = truth) -- byte-for-byte the input pattern that ancestral sampling
    and joint candidate scoring later query. This is the belief auxiliary loss and the search particle
    proposal.

Batch contract for `opp_revealed` (unchanged from the old model):
  {
    "tokens": LongTensor [B, 6, 9]  -- revealed value index per attribute (order = SLOT_ATTRS),
                                       or -1 where hidden/unknown.
    "masks": {                       -- optional per-attribute legality masks (True = legal)
       "species"/"item"/"ability"/"move"/"spread": Bool [B, 6, n_family]
    }                                 -- tera is never masked; missing keys = no mask
  }

Outputs:
  * logits             -- per-family masked logits; revealed attrs are delta rows (see above).
  * sample_team(k)     -- k concrete reveal-consistent opponent teams (search + diagnostics).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

# Attribute token order within one opponent slot (9 tokens/slot).
SLOT_ATTRS = ("species", "item", "ability", "move0", "move1", "move2", "move3", "spread", "tera")
N_SLOTS = 6
N_ATTRS = len(SLOT_ATTRS) * N_SLOTS  # 54
SEQ_LEN = N_ATTRS + 1                # 55: the 54 attributes + one START token
_START = N_ATTRS                     # content id of the START token in the [0..54] content space

# attr position within a slot -> (embedding/head family, move-slot-or-None)
_ATTR_FAMILY = ("species", "item", "ability", "move", "move", "move", "move", "spread", "tera")
_MOVE_SUBSLOT = (None, None, None, 0, 1, 2, 3, None, None)
_MOVE_COLS = (3, 4, 5, 6)  # attr columns holding the 4 move subslots within a slot
_ITEM_COL, _ABILITY_COL, _SPREAD_COL = 1, 2, 7  # SLOT_ATTRS column indices used by the EV ordering

# flat-index (0..53) lookups, precomputed once
_POS_SLOT = tuple(p // len(SLOT_ATTRS) for p in range(N_ATTRS))
_POS_FAMILY = tuple(_ATTR_FAMILY[p % len(SLOT_ATTRS)] for p in range(N_ATTRS))
_POS_SUB = tuple(_MOVE_SUBSLOT[p % len(SLOT_ATTRS)] for p in range(N_ATTRS))
# flat indices of a slot's 4 move columns, keyed by slot
_SLOT_MOVE_FLATS = tuple(
    tuple(j * len(SLOT_ATTRS) + c for c in _MOVE_COLS) for j in range(N_SLOTS)
)
# flat indices of each slot's item / ability / spread column (EV-ordering constraint, §4.2)
_SLOT_ITEM_FLATS = tuple(j * len(SLOT_ATTRS) + _ITEM_COL for j in range(N_SLOTS))
_SLOT_ABILITY_FLATS = tuple(j * len(SLOT_ATTRS) + _ABILITY_COL for j in range(N_SLOTS))
_SLOT_SPREAD_FLATS = tuple(j * len(SLOT_ATTRS) + _SPREAD_COL for j in range(N_SLOTS))

_NEG_INF = float("-inf")


@dataclass
class BeliefOutput:
    logits: dict[str, torch.Tensor]  # per-attribute masked logits (see forward)


class BeliefModel(nn.Module):
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
        self.s_proj = nn.Linear(state_dim, d)  # projects the c_dyn conditioning memory to width d
        dec = nn.TransformerDecoderLayer(
            d_model=d, nhead=self.nhead, dim_feedforward=4 * d, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec, num_layers=2)

        # per-attribute output heads (move head shared across the 4 move tokens)
        self.heads = nn.ModuleDict({name: nn.Linear(d, n) for name, n in self.sizes.items()})
        # per-attribute input value embeddings; index n (== sizes[name]) is the <UNK>/hidden token
        self.value_embeds = nn.ModuleDict(
            {name: nn.Embedding(n + 1, d) for name, n in self.sizes.items()}
        )
        # 54-way attribute-IDENTITY table (owner revision 4): index = slot*9 + attr. Reused (name
        # kept as attr_pos for state_dict continuity within this arch) as identity, gathered by
        # attribute id, NOT indexed by physical position.
        self.attr_pos = nn.Embedding(N_ATTRS, d)
        self.start_token = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.start_token, std=0.02)

        # per-flat-position class count, for the revealed/UNK test (species is big, tera small, ...)
        pos_n = torch.tensor([self.sizes[_POS_FAMILY[p]] for p in range(N_ATTRS)], dtype=torch.long)
        self.register_buffer("pos_n", pos_n, persistent=False)

    # ------------------------------------------------------------------------------------ layout util
    def _revealed_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        """[B, 6, 9] tokens -> [B, 54] bool: True where the attribute carries a valid revealed value."""
        flat = tokens.reshape(tokens.shape[0], N_ATTRS)
        n = self.pos_n.to(flat.device)
        return (flat >= 0) & (flat < n)

    def _fed_values(self, tokens: torch.Tensor, teacher_values: torch.Tensor | None) -> torch.Tensor:
        """[B, 54] long of the value fed at each attribute: the revealed value where revealed, else
        the teacher value (pass B) or -1 (UNK; pass A). -1 marks "no value" for both embedding
        (-> <UNK> row) and move-distinctness (contributes nothing)."""
        flat = tokens.reshape(tokens.shape[0], N_ATTRS).clone()
        revealed = self._revealed_mask(tokens)
        if teacher_values is None:
            return torch.where(revealed, flat, torch.full_like(flat, -1))
        tv = teacher_values.reshape(tokens.shape[0], N_ATTRS)
        return torch.where(revealed, flat, tv)

    @staticmethod
    def _spread_after_item_ability(r: torch.Tensor, revealed: torch.Tensor) -> torch.Tensor:
        """EV-ordering constraint (BELIEF_POLICY_PLAN §4.2): within each slot, force a HIDDEN spread's
        random tail key above its slot's item and ability keys, so the spread is generated with
        item/ability already pinned — the precondition for the damage-consistency mask (§3.4). Only
        HIDDEN item/ability constrain (a revealed one is in the prefix, already before every tail
        step). Canonical order already satisfies this (spread col 7 follows item col 1, ability col 2),
        so this touches only the random training/probe order. Everything else stays free (the +ε lands
        the spread just after max(item, ability), never globally last)."""
        r = r.clone()
        it = torch.where(revealed[:, _SLOT_ITEM_FLATS], r.new_full((), float("-inf")),
                         r[:, _SLOT_ITEM_FLATS])                          # [B, N_SLOTS]
        ab = torch.where(revealed[:, _SLOT_ABILITY_FLATS], r.new_full((), float("-inf")),
                         r[:, _SLOT_ABILITY_FLATS])
        sp = r[:, _SLOT_SPREAD_FLATS]
        r[:, _SLOT_SPREAD_FLATS] = torch.maximum(torch.maximum(it, ab), sp) + 1e-3
        return r

    def _order(
        self, revealed: torch.Tensor, tail_order: str, generator: torch.Generator | None
    ) -> torch.Tensor:
        """Per-example content order [B, 55] (physical position -> content id in 0..54, 54 = START).
        Layout: revealed attrs (canonical order) | START | hidden attrs (canonical or random order).

        Sort keys keep the three groups separated: revealed = its canonical index (0..53); START =
        100; hidden = 200 + (canonical index for the marginal, or a per-example random float for the
        chain). The random tail additionally enforces spread-after-item/ability per slot (§4.2).
        """
        b = revealed.shape[0]
        dev = revealed.device
        canon = torch.arange(N_ATTRS, device=dev, dtype=torch.float32).expand(b, -1)
        if tail_order == "canonical":
            tail_key = 200.0 + canon
        elif tail_order == "random":
            r = torch.rand(b, N_ATTRS, generator=generator, device=dev)
            r = self._spread_after_item_ability(r, revealed)
            tail_key = 200.0 + r
        else:
            raise ValueError(f"tail_order must be 'canonical' or 'random', got {tail_order!r}")
        attr_key = torch.where(revealed, canon, tail_key)                 # [B, 54]
        start_key = torch.full((b, 1), 100.0, device=dev)
        key = torch.cat([attr_key, start_key], dim=1)                     # [B, 55]
        return key.argsort(dim=1)

    def _attn_mask(self, revealed_count: torch.Tensor, device) -> torch.Tensor:
        """Additive float self-attention mask [B, 55, 55]: 0 where allowed, -inf where forbidden.

        Prefix+START query (physical i <= |R|): attends every j <= |R| (bidirectional block).
        Tail query (i > |R|): attends every j <= i (ordinary causal, which already covers the whole
        prefix+START block since |R| < i). The diagonal is always allowed, so no row is fully masked.
        """
        rc = revealed_count.to(device).view(-1, 1, 1)                     # [B, 1, 1]
        ii = torch.arange(SEQ_LEN, device=device).view(1, SEQ_LEN, 1)     # [1, 55, 1]
        jj = torch.arange(SEQ_LEN, device=device).view(1, 1, SEQ_LEN)     # [1, 1, 55]
        prefix_q = (ii <= rc) & (jj <= rc)
        tail_q = (ii > rc) & (jj <= ii)
        allowed = prefix_q | tail_q                                       # [B, 55, 55]
        return torch.where(allowed, 0.0, _NEG_INF)

    # ------------------------------------------------------------------------------------- core decode
    def _content_embeddings(self, fed: torch.Tensor) -> torch.Tensor:
        """[B, 54] fed values -> [B, 55, d] content embeddings (attrs 0..53 then START at index 54).
        Attr embedding = value_emb_family(value or <UNK>) + attr_id(flat index)."""
        b = fed.shape[0]
        dev = fed.device
        embs = []
        for p in range(N_ATTRS):
            fam = _POS_FAMILY[p]
            n = self.sizes[fam]
            idx = fed[:, p].clone()
            idx[(idx < 0) | (idx >= n)] = n                               # hidden / OOB -> <UNK>
            embs.append(self.value_embeds[fam](idx))
        attr_emb = torch.stack(embs, dim=1)                               # [B, 54, d]
        attr_emb = attr_emb + self.attr_pos(torch.arange(N_ATTRS, device=dev)).unsqueeze(0)
        start = self.start_token.expand(b, -1, -1)                        # [B, 1, d]
        return torch.cat([attr_emb, start], dim=1)                        # [B, 55, d]

    def _decode(
        self, c_dyn: torch.Tensor, fed: torch.Tensor, order: torch.Tensor,
        revealed_count: torch.Tensor,
    ) -> torch.Tensor:
        """Run the decoder for the given per-example layout. Returns physical-position hiddens
        [B, 55, d]. `c_dyn` is the dynamics conditioning memory (§3.1)."""
        content = self._content_embeddings(fed)                          # [B, 55, d]
        idx = order.unsqueeze(-1).expand(-1, -1, self.d)                 # [B, 55, d]
        phys = content.gather(1, idx)                                     # content at physical pos
        memory = self.s_proj(c_dyn).unsqueeze(1)                          # [B, 1, d]
        mask = self._attn_mask(revealed_count, c_dyn.device)             # [B, 55, 55]
        mask = mask.repeat_interleave(self.nhead, dim=0)                  # [B*nhead, 55, 55]
        return self.decoder(phys, memory, tgt_mask=mask)

    @staticmethod
    def _read_positions(order: torch.Tensor) -> torch.Tensor:
        """[B, 55] content order -> [B, 54] read hidden position per attribute (the physical position
        just BEFORE the attribute's own token; clamped >= 0). Also returns inverse for distinctness."""
        inv = order.argsort(dim=1)                                        # content id -> physical pos
        return inv

    def _apply_mask(self, logits: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """-inf illegal entries; rows with no legal entry are left unmasked (defensive)."""
        if mask is None:
            return logits
        ok = mask.any(dim=-1, keepdim=True)
        return logits.masked_fill(~mask & ok, _NEG_INF)

    def _move_distinct_mask(
        self, fed: torch.Tensor, seq_pos: torch.Tensor, slot: int, p: int, base: torch.Tensor | None
    ) -> torch.Tensor:
        """Learnset mask minus same-slot move values already in THIS token's conditioning set: every
        other slot-`slot` move column q whose fed value is set and whose physical position is earlier
        (`seq_pos[:, q] < seq_pos[:, p]`). In pass A (UNK tail) only revealed moves qualify -> reduces
        to the old revealed-move exclusion; in pass B / sampling, earlier fed/drawn moves also drop.

        fed [B, 54]; seq_pos [B, 54] (attribute -> physical position). Returns [B, n_moves] bool.
        """
        n_moves = self.sizes["move"]
        b = fed.shape[0]
        mask = (
            base.clone()
            if base is not None
            else torch.ones(b, n_moves, dtype=torch.bool, device=fed.device)
        )
        for q in _SLOT_MOVE_FLATS[slot]:
            if q == p:
                continue
            val = fed[:, q]
            earlier = (val >= 0) & (val < n_moves) & (seq_pos[:, q] < seq_pos[:, p])
            if earlier.any():
                rows = torch.nonzero(earlier, as_tuple=True)[0]
                mask[rows, val[earlier]] = False
        return mask

    def _delta_fill(self, logits: torch.Tensor, revealed: torch.Tensor, value: torch.Tensor):
        """Overwrite revealed rows with a delta distribution: 0 at the revealed value, -inf elsewhere.
        logits [B, n]; revealed [B] bool; value [B] long (the revealed index, ignored where ~revealed).
        """
        delta = torch.full_like(logits, _NEG_INF)
        delta.scatter_(1, value.clamp(min=0).unsqueeze(1), 0.0)
        return torch.where(revealed.unsqueeze(1), delta, logits)

    # --------------------------------------------------------------------------------------- forward
    def forward(
        self,
        c_dyn: torch.Tensor,
        opp_revealed: dict,
        *,
        teacher_values: torch.Tensor | None = None,
        tail_order: str | None = None,
        generator: torch.Generator | None = None,
    ) -> BeliefOutput:
        """Marginal (default) or chain (with `teacher_values`). Returns masked per-family logits:

          species/item/ability/spread/tera : [B, 6, n_family]
          move                             : [B, 6, 4, n_moves]

        `c_dyn` is the dynamics conditioning memory (§3.1). Marginal: `teacher_values=None`, canonical
        tail, every hidden token <UNK> -> marginals p(h|R). Chain: `teacher_values` a [B, 6, 9]
        true-value grid (aligned) -> the chain-rule conditionals p(h_k | R, h_1..h_{k-1}). BOTH modes
        default to the CANONICAL tail (§3.7 option (c), so the chain trains the exact order ancestral
        sampling walks); pass `tail_order="random"` for the any-order tail. Revealed attrs are delta
        rows in both.
        """
        tokens: torch.Tensor = opp_revealed["tokens"]
        masks: dict = opp_revealed.get("masks") or {}
        b = tokens.shape[0]
        if tail_order is None:
            # Both passes default to the canonical tail (BELIEF_POLICY_PLAN §3.7, option (c)). The
            # chain USED to default to a fresh random order per example, but that left ancestral
            # sampling — which walks the CANONICAL tail — in a regime no loss term trained, so it
            # hedged (gate: sample fidelity 0.30). A canonical chain trains exactly the conditionals
            # sample_team traverses. Reveal-conditioning is unaffected: it lives in the bidirectional
            # prefix, before every tail token, whatever the tail order. Pass `tail_order="random"`
            # explicitly for the pool_id §4.5.3 arm / the conditioning-gate ablation.
            tail_order = "canonical"

        revealed = self._revealed_mask(tokens)                            # [B, 54]
        fed = self._fed_values(tokens, teacher_values)                    # [B, 54]
        order = self._order(revealed, tail_order, generator)              # [B, 55]
        hidden = self._decode(c_dyn, fed, order, revealed.sum(dim=1))     # [B, 55, d]

        seq_pos = self._read_positions(order)                            # [B, 55] content -> phys
        read_pos = (seq_pos[:, :N_ATTRS] - 1).clamp(min=0)               # [B, 54] read hidden pos
        read_hidden = hidden.gather(
            1, read_pos.unsqueeze(-1).expand(-1, -1, self.d)
        )                                                                # [B, 54, d]

        tokens_flat = tokens.reshape(b, N_ATTRS)
        logits_out = self._heads(read_hidden, tokens_flat, revealed, fed, seq_pos, masks)
        return BeliefOutput(logits=logits_out)

    def _heads(
        self,
        read_hidden: torch.Tensor,
        tokens_flat: torch.Tensor,
        revealed: torch.Tensor,
        fed: torch.Tensor,
        seq_pos: torch.Tensor,
        masks: dict,
    ) -> dict[str, torch.Tensor]:
        """Per-canonical-position head/mask/delta-fill loop (survives from the old design, now fed the
        inverse-scattered read hiddens). Hidden attrs get head logits + legality + distinctness;
        revealed attrs get delta-filled."""
        b = read_hidden.shape[0]
        per_family: dict[str, list] = {f: [] for f in self.sizes}
        for p in range(N_ATTRS):
            slot, fam = _POS_SLOT[p], _POS_FAMILY[p]
            logits = self.heads[fam](read_hidden[:, p])                   # [B, n_fam]
            fam_mask = masks.get(fam)
            m = fam_mask[:, slot] if fam_mask is not None else None
            if fam == "move":
                m = self._move_distinct_mask(fed, seq_pos, slot, p, m)
            logits = self._apply_mask(logits, m)
            logits = self._delta_fill(logits, revealed[:, p], tokens_flat[:, p])
            per_family[fam].append(logits)

        out: dict[str, torch.Tensor] = {}
        for fam, lst in per_family.items():
            stacked = torch.stack(lst, dim=1)                             # [B, 6 or 24, n]
            if fam == "move":
                stacked = stacked.reshape(b, N_SLOTS, 4, self.sizes["move"])
            out[fam] = stacked
        return out

    # -------------------------------------------------------------------------------------- sampling
    @torch.no_grad()
    def sample_team(
        self,
        c_dyn: torch.Tensor,
        opp_revealed: dict,
        k: int = 16,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Ancestral sampling of k full reveal-consistent opponent teams (search §8, diagnostics).

        Walks the hidden attributes in CANONICAL order; at each step re-decodes with the draws so far
        fed back, reads the step's conditional at its shifted position, and samples. Because the chain
        loss now trains the canonical tail too (BELIEF_POLICY_PLAN §3.7 option (c)), sampling meets
        exactly the conditionals training shaped -- each read attends the prefix reveals plus the
        earlier-canonical draws, and later tail tokens are causally masked (truth in training, <UNK>
        here -- masked either way, so the two regimes coincide). Revealed attributes are clamped.
        `c_dyn` is the dynamics conditioning memory (§3.1). Returns LongTensor [B, k, 6, 9].
        """
        tokens0: torch.Tensor = opp_revealed["tokens"]
        masks: dict = opp_revealed.get("masks") or {}
        b = tokens0.shape[0]
        was_training = self.training
        self.eval()  # dropout off: sampling is inference and must be seed-deterministic

        def rep(x: torch.Tensor) -> torch.Tensor:
            return x.repeat_interleave(k, dim=0)

        s_k = rep(c_dyn)
        tokens_k = rep(tokens0).clone()                                   # [B*k, 6, 9]
        masks_k = {name: rep(m) for name, m in masks.items()}
        revealed = self._revealed_mask(tokens_k)                          # [B*k, 54]
        rcount = revealed.sum(dim=1)
        fed = self._fed_values(tokens_k, None)                            # [B*k, 54], -1 where hidden
        order = self._order(revealed, "canonical", None)                  # fixed given the reveals
        seq_pos = self._read_positions(order)                             # [B*k, 55]
        read_pos = (seq_pos[:, :N_ATTRS] - 1).clamp(min=0)
        tokens_flat = tokens_k.reshape(-1, N_ATTRS)

        for p in range(N_ATTRS):
            slot, fam = _POS_SLOT[p], _POS_FAMILY[p]
            if bool(revealed[:, p].all()):
                continue  # clamped everywhere in the batch -> nothing to sample at this attr
            hidden = self._decode(s_k, fed, order, rcount)                # [B*k, 55, d]
            read = hidden.gather(
                1, read_pos[:, p].view(-1, 1, 1).expand(-1, 1, self.d)
            ).squeeze(1)                                                  # [B*k, d]
            logits = self.heads[fam](read) / max(temperature, 1e-6)
            fam_mask = masks_k.get(fam)
            m = fam_mask[:, slot] if fam_mask is not None else None
            if fam == "move":
                m = self._move_distinct_mask(fed, seq_pos, slot, p, m)
            logits = self._apply_mask(logits, m)
            probs = torch.softmax(logits, dim=-1)
            drawn = torch.multinomial(probs, 1, generator=generator).squeeze(-1)   # [B*k]
            known = revealed[:, p]
            fed[:, p] = torch.where(known, tokens_flat[:, p], drawn)

        if was_training:
            self.train()
        return fed.reshape(b, k, N_SLOTS, len(SLOT_ATTRS))

    # ------------------------------------------------------------------------------------------ util
    @staticmethod
    def _flat(pos: int) -> tuple[int, str, int | None]:
        """flat attr index (0..53) -> (slot j, family name, move subslot)."""
        return _POS_SLOT[pos], _POS_FAMILY[pos], _POS_SUB[pos]
