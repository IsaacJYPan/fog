# SPDX-License-Identifier: GPL-3.0-only
"""Finite scheduling for Foul Play's independent Bernoulli move proposals.

Derived loop: pmariglia/foul-play, revision 8f541f80f91f960603673796559f796b10ccf0fb.
Failed full sweeps can be skipped without changing the first-success distribution.
All-zero remaining rates terminate with the available partial moveset.
"""

from __future__ import annotations

import hashlib
import inspect
import math
from pathlib import Path


def first_success(chances, start, rng):
    """Choose the first success conditional on a successful cyclic sweep.

    Weight i is p_i times the preceding failure probabilities. Normalization
    marginalizes any number of failed sweeps; no rejection iterations are needed.
    """
    positions, weights, survival = [], [], 1.0
    for offset in range(len(chances)):
        index = (start + offset) % len(chances)
        chance = float(chances[index])
        if not math.isfinite(chance) or not 0 <= chance <= 1:
            raise ValueError("Foul Play move proposal rates must be finite probabilities")
        positions.append(index)
        weights.append(survival * chance)
        survival *= 1 - chance
    return rng.choices(positions, weights=weights)[0] if sum(weights) > 0 else None


def install_bounded_sampling():
    import fp.search.standard_battles as upstream

    # Only the audited rejection loop is replaced. Full-set donors, consistency
    # checks, trait inference, engine searches and move selection remain upstream.
    source_path = Path(upstream.__file__)
    digest = hashlib.sha256(source_path.read_text().encode()).hexdigest()
    expected = "c64677c5b3a51a1691a6f9fa80ab1e184764ee93fd0b0b0eb2576e49bc877062"
    if digest != expected:
        raise ValueError("Foul Play sampler source changed; audit this adapter before use")
    function = upstream.sample_pokemon_moveset_with_known_pkmn_set
    source = inspect.getsource(function)
    begin = source.index("    index = 0\n    while True:")
    end = source.index("\n    return pkmn_known_moves", begin)
    replacement = """    index = 0
    while len(pkmn_known_moves) < 4 and moves_adjusted_probabilities:
        selected = _first_success([chance for _, chance in moves_adjusted_probabilities], index, random)
        if selected is None:
            break
        index = selected
        mv, chance = moves_adjusted_probabilities[index]
        pkmn_known_moves.append(mv)
        if not PredictedPokemonSet(
            pkmn_set=pkmn_set,
            pkmn_moveset=PokemonMoveset(moves=pkmn_known_moves),
        ).set_makes_logical_sense():
            pkmn_known_moves.pop()
        moves_adjusted_probabilities.pop(index)
"""
    namespace = dict(vars(upstream), _first_success=first_success)
    exec(
        compile(source[:begin] + replacement + source[end:], str(source_path) + ":bounded", "exec"),
        namespace,
    )
    upstream.sample_pokemon_moveset_with_known_pkmn_set = namespace[function.__name__]
    return "bounded-bernoulli-move-proposals-v1"
