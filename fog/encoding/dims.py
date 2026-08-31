"""Tensor dimension bookkeeping for the encoders (spec §2).

Kept in one place so the encoder network (§2.4) and the tests agree on every width. Widths that depend
on generated tables (e.g. embedding input cardinalities) are looked up at model-build time; the widths
here are the *layout* widths of the raw feature blocks before embedding.
"""
from __future__ import annotations

from dataclasses import dataclass

from fog.constants.schema import (
    N_ACTIONS,
    N_STATUS,
    N_TERA_TYPES,
    N_TERRAIN,
    N_TYPES,
    N_VOLATILE_COUNTERS,
    N_VOLATILES,
    N_WEATHER,
    PARADOX_BOOST_STATS,
)

N_ENTITIES = 12          # 6 mine + 6 opponent
N_MY_SLOTS = 6
N_MOVE_SLOTS = 4
# N_VOLATILES (32) and N_VOLATILE_COUNTERS (5) come from schema.VOLATILES / VOLATILE_COUNTERS —
# the canonical, frozen orderings from the gen-9 encoding audit.
N_BOOST_STATS = 7        # atk, def, spa, spd, spe, acc, eva


@dataclass(frozen=True)
class Dims:
    # --- §2.1 entity record raw blocks (pre-embedding floats/bits) ---------------------------------
    species_embed: int = 64          # index -> embedding
    base_stats: int = 6              # /255
    typing: int = N_TYPES            # multi-hot (one or two)
    weight_bucket: int = 8           # one-hot weight class
    hp_fraction: int = 1
    fainted_flag: int = 1
    status: int = N_STATUS           # one-hot(7)
    sleep_tox_counters: int = 2      # /8
    boosts: int = N_BOOST_STATS      # (+6..-6)/6
    volatiles: int = N_VOLATILES     # multi-hot (schema.VOLATILES order, 32)
    volatile_counters: int = N_VOLATILE_COUNTERS  # turns-remaining /8 (schema.VOLATILE_COUNTERS order)
    paradox_boost: int = len(PARADOX_BOOST_STATS) # Proto/Quark boosted-stat one-hot (all-zero = none)
    booster_consumed: int = 1        # Booster Energy already used flag (item inference signal)
    fallen_count: int = 1            # Supreme Overlord fallen count at switch-in /5 (Kingambit)
    # revealed set (embeddings sized at model build; these are the *counts* of embedded slots)
    n_item_slots: int = 1
    n_ability_slots: int = 1
    n_move_slots: int = N_MOVE_SLOTS
    move_pp: int = N_MOVE_SLOTS      # PP fractions where known
    tera_type: int = N_TERA_TYPES    # one-hot(19)
    tera_used_flag: int = 1
    n_nature_slots: int = 1
    n_ev_slots: int = 6
    # known/unknown per hidden field: item, ability, 4 moves, nature, 6 EV stats, tera
    epistemic_flags: int = 14
    # positional
    is_mine: int = 1
    is_active: int = 1
    party_slot: int = 6              # one-hot

    # --- §2.2 field vector -------------------------------------------------------------------------
    field_weather: int = N_WEATHER + 1       # one-hot + turns-left/8
    field_terrain: int = N_TERRAIN + 1
    field_hazards: int = 2 * 6               # per-side: SR, spikes, tspikes, web, (+counters)
    field_screens: int = 2 * 5               # per-side: reflect, lightscreen, auroraveil, tailwind (+counters)
    field_trickroom: int = 2                 # flag + counter
    field_turn: int = 1                      # /50
    field_forced_switch: int = 2             # my-side / opp-side last-switch-was-forced
    # delayed effects (encoding audit 2026-07): per-side (Future Sight pending, FS turns-left /3,
    # Wish pending, Wish turns-left /2, Wish heal amount as hp-frac of the incoming target).
    # Slowking-Galar Future Sight pivoting and Wish passing are decision-relevant in current OU.
    field_delayed: int = 2 * 5
    field_gravity: int = 2                   # flag + counter (rare but cheap)

    # --- §2.3 per-turn history record --------------------------------------------------------------
    hist_snapshot: int = 30          # both actives' species/hp/status/boost sums + field hash features
    hist_my_action: int = N_ACTIONS  # 14-way one-hot
    hist_opp_action: int = 40        # observed category: move idx / switch species / tera flag / <UNSEEN>
    hist_resolution: int = 24        # dmg dealt/taken, crit, effectiveness, miss, reveals, faints, order
    # Checkpoint-compatible sidecars.  The legacy 108-float record remains unchanged; categorical
    # identities and the two actives' individual boost stages are embedded/projected alongside it.
    hist_id_fields: int = 17
    hist_state: int = 2 * N_BOOST_STATS

    history_cap: int = 60            # T; pad/truncate from the front

    @property
    def field_dim(self) -> int:
        return (
            self.field_weather
            + self.field_terrain
            + self.field_hazards
            + self.field_screens
            + self.field_trickroom
            + self.field_turn
            + self.field_forced_switch
            + self.field_delayed
            + self.field_gravity
        )

    @property
    def turn_dim(self) -> int:
        return self.hist_snapshot + self.hist_my_action + self.hist_opp_action + self.hist_resolution


DIMS = Dims()
