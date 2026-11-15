"""Separate prefix-only preview observations and public lead labels."""
from __future__ import annotations

import numpy as np

from fog.encoding.entities import L
from fog.encoding.observation import Observation
from fog.scripts.parse_replays import _PerspectiveState, replay_player_names
from fog.training.replay_supervision import _preview_species_id


def _base_species(species: str) -> str:
    from poke_env.data import GenData

    from fog.scripts.parse_replays import _to_id

    data = GenData.from_gen(9).pokedex.get(species, {})
    return _to_id(data.get('baseSpecies', species))


def public_lead_labels(replay: dict) -> tuple[dict, dict]:
    """Resolve the first physical send-out, including a publicly unmasked Illusion.

    A disguised first stint that never unmasks is not a known physical identity.
    Later protocol is used only for the target, never for a preview input.
    """
    teams = {role: [] for role in ('p1', 'p2')}
    leads = {}
    ended = set()
    unmasked = set()
    for line in replay['log'].splitlines():
        parts = line.split('|')
        if len(parts) < 3:
            continue
        tag = parts[1]
        role = parts[2][:2]
        if role not in teams:
            continue
        if tag == 'poke' and len(parts) >= 4:
            teams[role].append(_preview_species_id(parts[3]))
        elif tag == 'switch' and len(parts) >= 4:
            if role in leads:
                ended.add(role)
            else:
                leads[role] = _preview_species_id(parts[3])
        elif tag == 'replace' and len(parts) >= 4 and role in leads and role not in ended:
            leads[role] = _preview_species_id(parts[3])
            unmasked.add(role)
        elif tag == 'faint' and role in leads:
            ended.add(role)
    labels = {}
    reasons = {}
    for role, team in teams.items():
        lead = leads.get(role)
        illusion = any(_base_species(s) in {'zoroark', 'zorua'} for s in team)
        if illusion and role not in unmasked and lead is not None and _base_species(lead) not in {'zoroark', 'zorua'}:
            labels[role] = -1
            reasons[role] = 'unresolved_illusion'
            continue
        matches = [i for i, s in enumerate(team) if s == lead]
        if not matches and lead is not None:
            matches = [i for i, s in enumerate(team) if _base_species(s) == _base_species(lead)]
        labels[role] = matches[0] if len(matches) == 1 else -1
        reasons[role] = 'public_lead' if labels[role] >= 0 else 'unmatched_lead'
    return labels, reasons


def preview_arrays(replay: dict, private_sets: dict) -> dict[str, np.ndarray]:
    """Exactly two rows from fresh battles, stopping at preview before any send-out."""
    names = replay_player_names(replay)
    states = [_PerspectiveState(role, names[role], private_sets[role]) for role in ('p1', 'p2')]
    for line in replay['log'].splitlines():
        if not line.startswith('|'):
            continue
        parts = line.split('|')
        tag = parts[1]
        if tag in {'start', 'switch', 'turn', 'move'}:
            break
        if tag == 'poke' and len(parts) >= 4:
            for st in states:
                st.register_own_preview(parts[2], parts[3])
        for st in states:
            st.battle.parse_message(parts)
        if tag == 'teampreview':
            break
    observations = []
    for st in states:
        # This tracker is independent of the ordinary replay pass. Opponent sets are never
        # installed; their only input is the public preview protocol above.
        obs = Observation.from_battle(st.battle, perspective=st.role, tracker=st.tracker)
        obs.entities[:, L.is_active] = 0
        for side, count in ((0, len(st._preview_species)),
                            (6, len(states[1 if st.role == 'p1' else 0]._preview_species))):
            obs.entities[side:side+count, L.hp] = 1
        observations.append(obs)
    labels, reasons = public_lead_labels(replay)
    complete = all(len(st._preview_species) == 6 for st in states)
    return {
        'preview_entities': np.stack([obs.entities for obs in observations]),
        'preview_entity_ids': np.stack([obs.entity_ids for obs in observations]),
        'preview_field': np.stack([obs.field for obs in observations]),
        'preview_label': np.array([labels[st.role] for st in states], dtype=np.int64),
        'preview_valid': np.array([complete and labels[st.role] >= 0 for st in states]),
        'preview_species': np.array([st._preview_species+['']*(6-len(st._preview_species)) for st in states], dtype='U64'),
        'preview_role': np.array(['p1', 'p2']),
        'preview_imputation_id': np.zeros(2, dtype=np.int16),
        'preview_label_reason': np.array([reasons[st.role] if complete else 'incomplete_preview' for st in states]),
    }
