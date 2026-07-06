"""Battle-conditioned autoregressive opponent-team belief model (spec §3.2).

Input: `s` (so beliefs update as the battle reveals evidence) + the partially-revealed opponent records.

Generation order per opponent slot j = 1..6 (skipping/clamping anything already revealed):
  1. species_j  -- masked to preview species (always clamped in standard OU; head still exists for
                   no-preview formats and the M1.5 hidden-pool variant)
  2. item_j     -- masked to items with nonzero usage on species_j
  3. ability_j  -- masked to the species' legal 1-3
  4. moves_j    -- 4 sequential softmaxes over MOVES masked to learnset minus already-picked;
                   revealed moves clamped into the earliest slots
  5. spread_j   -- over SPREADS[species_j]
  6. tera_j     -- 19-way

Architecture (decoder-only conditioning on `s` via cross-attention):
  * memory  = s_proj [B, 1, d]
  * tgt     = [start-token, emb(attr_0), ..., emb(attr_52)]  (54 positions; 6 slots x 9 attrs = 54
              attributes total). The hidden at position p predicts attribute p, and by the causal mask
              only sees the start token + attributes 0..p-1. Teacher forcing feeds the revealed/true
              value embedding where known and a learned <UNK> embedding where hidden — which is exactly
              the information state at inference, so train/inference inputs agree on hidden attrs.
  * per-attribute linear heads (move head shared across the 4 move tokens), legality masks applied as
    -inf before returning logits.

Each conditional is over <=~120 legal options — the sparsity answer: the astronomically large joint is
factored into a chain of small dense classifications, with correlations captured by conditioning.

Batch contract for `opp_revealed` (produced by encoding/observation + the env/parser):
  {
    "tokens": LongTensor [B, 6, 9]  -- revealed (or, in self-play teacher forcing, true) value index
                                        per attribute, or -1 where hidden/unknown.
                                        Attribute order = SLOT_ATTRS.
    "masks": {                       -- optional per-attribute legality masks (True = legal)
       "species": Bool [B, 6, n_species],
       "item":    Bool [B, 6, n_items],
       "ability": Bool [B, 6, n_abilities],
       "move":    Bool [B, 6, n_moves],     # learnset mask; minus-already-picked handled internally
       "spread":  Bool [B, 6, n_spreads],
    }                                 -- tera is never masked (19 always legal); missing keys = no mask
  }

Two outputs consumed downstream:
  * b_vec in R^128       -- mean-pooled decoder hidden, concatenated into policy/value (cheap path)
  * sample_team(k)       -- k concrete opponent teams consistent with all reveals (search §8 + diag)

Losses: sum of masked cross-entropies per attribute (see training/losses.belief_loss). Labels are
CENSORED in replay training (only attributes eventually revealed in that replay contribute); full
ground truth in self-play.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

# Attribute token order within one opponent slot (9 tokens/slot).
SLOT_ATTRS = ("species", "item", "ability", "move0", "move1", "move2", "move3", "spread", "tera")
N_SLOTS = 6
N_ATTRS = len(SLOT_ATTRS) * N_SLOTS  # 54

# attr position within a slot -> (embedding/head family, move-slot-or-None)
_ATTR_FAMILY = ("species", "item", "ability", "move", "move", "move", "move", "spread", "tera")
_MOVE_SUBSLOT = (None, None, None, 0, 1, 2, 3, None, None)

_NEG_INF = float("-inf")


@dataclass
class BeliefOutput:
    b_vec: torch.Tensor                       # [B, belief_dim] pooled hidden (cheap conditioning path)
    logits: dict[str, torch.Tensor]           # per-attribute logits for the loss (masked; see forward)


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
        self.sizes = {
            "species": n_species,
            "item": n_items,
            "ability": n_abilities,
            "move": n_moves,
            "spread": n_spreads,
            "tera": n_tera,
        }
        self.s_proj = nn.Linear(state_dim, d)
        dec = nn.TransformerDecoderLayer(d_model=d, nhead=4, dim_feedforward=4 * d, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec, num_layers=2)

        # per-attribute output heads (move head shared across the 4 move tokens)
        self.heads = nn.ModuleDict(
            {name: nn.Linear(d, n) for name, n in self.sizes.items()}
        )
        # per-attribute input value embeddings; index n (== sizes[name]) is the <UNK>/hidden token
        self.value_embeds = nn.ModuleDict(
            {name: nn.Embedding(n + 1, d) for name, n in self.sizes.items()}
        )
        # learned token embeddings marking attribute position within the 54-token chain
        self.attr_pos = nn.Embedding(N_ATTRS, d)
        self.start_token = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.start_token, std=0.02)

        causal = nn.Transformer.generate_square_subsequent_mask(N_ATTRS)
        self.register_buffer("causal_mask", causal, persistent=False)

    # ------------------------------------------------------------------------------------------ util
    @staticmethod
    def _flat(pos: int) -> tuple[int, str, int | None]:
        """flat attr index (0..53) -> (slot j, family name, move subslot)."""
        j, a = divmod(pos, len(SLOT_ATTRS))
        return j, _ATTR_FAMILY[a], _MOVE_SUBSLOT[a]

    def _embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens [B, 6, 9] (long, -1 = hidden) -> tgt [B, 54, d] with position embeddings added.

        Position p of the returned sequence is the *input* for predicting attribute p, i.e. the value
        embedding of attribute p-1 (start token at p=0), following the shift in the module docstring.
        """
        b = tokens.shape[0]
        flat = tokens.reshape(b, N_ATTRS)  # attribute order == flat token order
        embs = []
        for p in range(N_ATTRS - 1):  # attrs 0..52 feed positions 1..53
            _, fam, _ = self._flat(p)
            n = self.sizes[fam]
            idx = flat[:, p].clone()
            idx[(idx < 0) | (idx >= n)] = n  # hidden / out-of-range -> <UNK>
            embs.append(self.value_embeds[fam](idx))
        tgt = torch.stack(embs, dim=1)                                   # [B, 53, d]
        start = self.start_token.expand(b, -1, -1)                       # [B, 1, d]
        tgt = torch.cat([start, tgt], dim=1)                             # [B, 54, d]
        pos = self.attr_pos(torch.arange(N_ATTRS, device=tokens.device)) # [54, d]
        return tgt + pos.unsqueeze(0)

    def _decode(self, s: torch.Tensor, tokens: torch.Tensor, upto: int = N_ATTRS) -> torch.Tensor:
        """Run the causal decoder over the first `upto` positions. Returns hiddens [B, upto, d]."""
        memory = self.s_proj(s).unsqueeze(1)                             # [B, 1, d]
        tgt = self._embed_tokens(tokens)[:, :upto]
        mask = self.causal_mask[:upto, :upto]
        return self.decoder(tgt, memory, tgt_mask=mask)

    def _apply_mask(self, logits: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """-inf illegal entries; rows with no legal entry are left unmasked (defensive)."""
        if mask is None:
            return logits
        ok = mask.any(dim=-1, keepdim=True)
        return logits.masked_fill(~mask & ok, _NEG_INF)

    def _move_distinct_mask(
        self, tokens: torch.Tensor, slot: int, subslot: int, base: torch.Tensor | None
    ) -> torch.Tensor | None:
        """Learnset mask minus moves already present in earlier move subslots of this slot.

        tokens [B, 6, 9]; returns [B, n_moves] or None.
        """
        n_moves = self.sizes["move"]
        b = tokens.shape[0]
        mask = (
            base.clone()
            if base is not None
            else torch.ones(b, n_moves, dtype=torch.bool, device=tokens.device)
        )
        for prev in range(subslot):
            prev_tok = tokens[:, slot, 3 + prev]  # move subslots live at attr positions 3..6
            known = (prev_tok >= 0) & (prev_tok < n_moves)
            if known.any():
                rows = torch.nonzero(known, as_tuple=True)[0]
                mask[rows, prev_tok[known]] = False
        return mask

    # --------------------------------------------------------------------------------------- forward
    def forward(self, s: torch.Tensor, opp_revealed: dict) -> BeliefOutput:
        """Teacher-forced pass. Returns masked logits per attribute family:

          species/item/ability/spread/tera : [B, 6, n_family]
          move                             : [B, 6, 4, n_moves]

        Legality masks from `opp_revealed["masks"]` are applied as -inf; the move logits additionally
        exclude moves already revealed/teacher-forced in earlier subslots of the same slot.
        """
        tokens: torch.Tensor = opp_revealed["tokens"]
        masks: dict = opp_revealed.get("masks") or {}
        b = tokens.shape[0]

        h = self._decode(s, tokens)                                      # [B, 54, d]

        per_family: dict[str, list] = {f: [] for f in self.sizes}
        for p in range(N_ATTRS):
            slot, fam, sub = self._flat(p)
            logits = self.heads[fam](h[:, p])                            # [B, n_fam]
            fam_mask = masks.get(fam)
            m = fam_mask[:, slot] if fam_mask is not None else None      # [B, n_fam] or None
            if fam == "move":
                m = self._move_distinct_mask(tokens, slot, sub, m)
            per_family[fam].append(self._apply_mask(logits, m))

        logits_out: dict[str, torch.Tensor] = {}
        for fam, lst in per_family.items():
            stacked = torch.stack(lst, dim=1)                            # [B, 6 or 24, n]
            if fam == "move":
                stacked = stacked.reshape(b, N_SLOTS, 4, self.sizes["move"])
            logits_out[fam] = stacked

        b_vec = h.mean(dim=1)                                            # [B, d]
        return BeliefOutput(b_vec=b_vec, logits=logits_out)

    # -------------------------------------------------------------------------------------- sampling
    @torch.no_grad()
    def sample_team(
        self,
        s: torch.Tensor,
        opp_revealed: dict,
        k: int = 16,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Ancestral sampling of k full opponent teams consistent with all reveals (§8, diagnostics).

        Returns LongTensor [B, k, 6, 9] of attribute indices (order = SLOT_ATTRS). Revealed attributes
        (tokens >= 0) are clamped, hidden ones sampled from the masked conditional. Note: the per-slot
        masks in `opp_revealed["masks"]` are correct whenever species is clamped (standard OU preview);
        for hidden-species formats the caller must pass masks valid for every species the species-mask
        allows, or post-filter samples.
        """
        tokens0: torch.Tensor = opp_revealed["tokens"]
        masks: dict = opp_revealed.get("masks") or {}
        b = tokens0.shape[0]
        was_training = self.training
        self.eval()  # dropout must be off: sampling is inference and must be seed-deterministic

        # expand batch B -> B*k
        def rep(x: torch.Tensor) -> torch.Tensor:
            return x.repeat_interleave(k, dim=0)

        s_k = rep(s)
        cur = rep(tokens0).clone()                                        # [B*k, 6, 9]
        masks_k = {name: rep(m) for name, m in masks.items()}

        flat_revealed = tokens0.reshape(b, N_ATTRS)
        for p in range(N_ATTRS):
            slot, fam, sub = self._flat(p)
            n = self.sizes[fam]
            revealed = flat_revealed[:, p]
            if bool(((revealed >= 0) & (revealed < n)).all()):
                continue  # fully clamped everywhere in the batch — nothing to sample
            h = self._decode(s_k, cur, upto=p + 1)                        # [B*k, p+1, d]
            logits = self.heads[fam](h[:, p]) / max(temperature, 1e-6)
            fam_mask = masks_k.get(fam)
            m = fam_mask[:, slot] if fam_mask is not None else None
            if fam == "move":
                m = self._move_distinct_mask(cur, slot, sub, m)
            logits = self._apply_mask(logits, m)
            probs = torch.softmax(logits, dim=-1)
            drawn = torch.multinomial(probs, 1, generator=generator).squeeze(-1)  # [B*k]
            # clamp where revealed
            cur_flat = cur.reshape(-1, N_ATTRS)
            known = (cur_flat[:, p] >= 0) & (cur_flat[:, p] < n)
            cur_flat[:, p] = torch.where(known, cur_flat[:, p], drawn)

        if was_training:
            self.train()
        return cur.reshape(b, k, N_SLOTS, len(SLOT_ATTRS))

    def belief_vec(self, s: torch.Tensor, opp_revealed: dict) -> torch.Tensor:
        """Cheap path: just the pooled hidden b_vec, for policy/value conditioning."""
        return self.forward(s, opp_revealed).b_vec
