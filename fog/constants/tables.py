"""Runtime loader for generated constant tables.

Loads the versioned JSON emitted by `build.py` into frozen numpy arrays + id<->index dicts, and
exposes the masks the belief model needs (item-usable-per-species, legal abilities, learnset).

Everything here is lazy: tables load on first access so that modules importing `constants` don't hard-
fail before `build_constants` has been run. Access raises a clear error if the generated files are
missing. Index conventions:
  * species: unknown ids fall back to the <RARE_SPECIES> index (last).
  * items:   unknown ids fall back to <OTHER_ITEM>; <NO_ITEM>/<UNKNOWN> are distinct indices (§2.1).
  * moves/abilities: no sentinel — unknown ids raise KeyError (callers quarantine, §4.3), or pass
    `default=` to get a fallback.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np

from fog.constants.schema import OTHER_ITEM, RARE_SPECIES

GENERATED_DIR = Path(__file__).resolve().parent / "generated"

_MISSING = object()


class TablesNotBuilt(RuntimeError):
    pass


@dataclass
class Tables:
    """Lazy container over the generated tables. One instance is exposed as `constants.tables.TABLES`.

    `gen_dir` is injectable so tests can point at fixture tables.
    """

    gen_dir: Path = field(default_factory=lambda: GENERATED_DIR)

    def _load_json(self, name: str) -> dict:
        path = self.gen_dir / f"{name}.json"
        if not path.exists():
            raise TablesNotBuilt(
                f"{path} missing. Run `python -m fog.scripts.build_constants` first (see §1)."
            )
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    @cached_property
    def species(self) -> dict:
        return self._load_json("species")

    @cached_property
    def moves(self) -> dict:
        return self._load_json("moves")

    @cached_property
    def items(self) -> dict:
        return self._load_json("items")

    @cached_property
    def abilities(self) -> dict:
        return self._load_json("abilities")

    @cached_property
    def spreads(self) -> dict:
        return self._load_json("spreads")

    # --- cardinalities (feed these into EntityEmbeddings / BeliefModel at build time) ---------------
    @property
    def n_species(self) -> int:
        return len(self.species["ids"])

    @property
    def n_moves(self) -> int:
        return len(self.moves["ids"])

    @property
    def n_items(self) -> int:
        return len(self.items["ids"])

    @property
    def n_abilities(self) -> int:
        return len(self.abilities["ids"])

    # --- id <-> index -------------------------------------------------------------------------------
    @cached_property
    def _species_idx(self) -> dict[str, int]:
        return {s: i for i, s in enumerate(self.species["ids"])}

    @cached_property
    def _move_idx(self) -> dict[str, int]:
        return {m: i for i, m in enumerate(self.moves["ids"])}

    @cached_property
    def _item_idx(self) -> dict[str, int]:
        return {it: i for i, it in enumerate(self.items["ids"])}

    @cached_property
    def _ability_idx(self) -> dict[str, int]:
        return {a: i for i, a in enumerate(self.abilities["ids"])}

    def species_index(self, species_id: str) -> int:
        """Unknown species fall back to <RARE_SPECIES> (the expected rare-tail path, §1.1)."""
        return self._species_idx.get(species_id, self._species_idx[RARE_SPECIES])

    def move_index(self, move_id: str, default: int | object = _MISSING) -> int:
        idx = self._move_idx.get(move_id)
        if idx is None:
            if default is _MISSING:
                raise KeyError(f"move {move_id!r} not in the generated MOVES table")
            return default  # type: ignore[return-value]
        return idx

    def item_index(self, item_id: str) -> int:
        """Unknown items fall back to <OTHER_ITEM> (§1.4)."""
        return self._item_idx.get(item_id, self._item_idx[OTHER_ITEM])

    def ability_index(self, ability_id: str, default: int | object = _MISSING) -> int:
        idx = self._ability_idx.get(ability_id)
        if idx is None:
            if default is _MISSING:
                raise KeyError(f"ability {ability_id!r} not in the generated ABILITIES table")
            return default  # type: ignore[return-value]
        return idx

    # --- masks consumed by the belief model (§3.2) --------------------------------------------------
    # Cached as full [n_species, n_x] bool matrices; row lookup is a view. The <RARE_SPECIES> row is
    # all-True everywhere (uninformative prior — we know nothing about a rare species' options).

    @cached_property
    def _learnset_matrix(self) -> np.ndarray:
        m = np.zeros((self.n_species, self.n_moves), dtype=bool)
        for sid, feats in self.species["features"].items():
            row = self._species_idx[sid]
            for mv in feats["learnset"]:
                idx = self._move_idx.get(mv)
                if idx is not None:
                    m[row, idx] = True
        # Showdown's ordinary Smeargle learnset contains only Sketch, while a submitted Smeargle
        # can carry every sketchable move.  Treating the raw learnset as a hard mask therefore
        # makes legitimate replay labels (for example Jaw Lock) impossible.  The generated move
        # flags retain Showdown's `nosketch` marker, so this is still a legality mask rather than
        # an unrestricted usage prior.
        smeargle = self._species_idx.get("smeargle")
        if smeargle is not None:
            for move_id, features in self.moves["features"].items():
                idx = self._move_idx.get(move_id)
                if idx is not None and "nosketch" not in features.get("flags", ()):
                    m[smeargle, idx] = True
        m[self._species_idx[RARE_SPECIES]] = True
        return m

    @cached_property
    def _abilities_matrix(self) -> np.ndarray:
        m = np.zeros((self.n_species, self.n_abilities), dtype=bool)
        for sid, feats in self.species["features"].items():
            row = self._species_idx[sid]
            for ab in feats["abilities"]:
                idx = self._ability_idx.get(ab)
                if idx is not None:
                    m[row, idx] = True
        # Ogerpon's public species changes to an `...tera` battle-only form whose live ability is
        # Embody Aspect, while submitted-set truth correctly remains the mask form's original
        # ability (for example Water Absorb). Belief supervision predicts submitted private state,
        # so both abilities must remain legal under that public form. The generated Gen 9 species
        # table contains exactly these four `...tera` forms, each paired with its non-Tera id.
        for sid, row in self._species_idx.items():
            if not sid.startswith("ogerpon") or not sid.endswith("tera"):
                continue
            submitted_id = sid[: -len("tera")]
            # Showdown names the battle-only Teal form `Ogerpon-Teal-Tera`, but the submitted
            # no-mask species is simply `Ogerpon` (there is no `ogerponteal` table row).
            if submitted_id == "ogerponteal":
                submitted_id = "ogerpon"
            submitted_form = self._species_idx.get(submitted_id)
            if submitted_form is not None:
                m[row] |= m[submitted_form]
        m[self._species_idx[RARE_SPECIES]] = True
        return m

    @cached_property
    def _items_matrix(self) -> np.ndarray:
        # default all-True: species without usage rows keep every item open (incl. sentinels)
        m = np.ones((self.n_species, self.n_items), dtype=bool)
        n_sentinels = 3  # <OTHER_ITEM>, <NO_ITEM>, <UNKNOWN> stay legal for everyone
        for sid, item_ids in self.items.get("usable", {}).items():
            row = self._species_idx.get(sid)
            if row is None:
                continue
            m[row, : self.n_items - n_sentinels] = False
            for iid in item_ids:
                idx = self._item_idx.get(iid)
                if idx is not None:
                    m[row, idx] = True
        return m

    def learnset_mask(self, species_idx: int) -> np.ndarray:
        """Binary mask over MOVES: legal moves for this species."""
        return self._learnset_matrix[species_idx]

    def legal_abilities_mask(self, species_idx: int) -> np.ndarray:
        """Binary mask over ABILITIES: the species' 1-3 legal abilities."""
        return self._abilities_matrix[species_idx]

    def usable_items_mask(self, species_idx: int) -> np.ndarray:
        """Binary mask over ITEMS: items with nonzero usage on this species (+ sentinels)."""
        return self._items_matrix[species_idx]


TABLES = Tables()
