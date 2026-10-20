import copy
from types import SimpleNamespace

import numpy as np
import torch

from fog.constants.tables import TABLES
from fog.models.belief import BeliefModel
from fog.scripts import pretrained_prior_audit as audit


def test_cpu_prior_known_fields_match_the_actual_decoder_including_consumed_items():
    model = BeliefModel(
        d=16,
        n_species=TABLES.n_species,
        n_items=TABLES.n_items,
        n_moves=TABLES.n_moves,
        n_abilities=TABLES.n_abilities,
    )
    sizes = np.array(
        [
            TABLES.n_species,
            TABLES.n_items,
            TABLES.n_abilities,
            *([TABLES.n_moves] * 4),
            25,
            *([64] * 6),
            19,
        ]
    )
    tokens = np.random.default_rng(1).integers(-1, sizes + 1, size=(6, 15))
    tokens[0, 1] = TABLES.item_index("<NO_ITEM>")
    expected = model._revealed_mask(torch.from_numpy(tokens[None])).reshape(6, 15).numpy()
    assert np.array_equal(expected, audit.known_fields(tokens))


def test_prior_proposals_do_not_change_when_private_truth_changes(monkeypatch):
    public = np.full((6, 15), -1, dtype=np.int64)
    public[:, 0] = TABLES.species_index("greattusk")
    target = np.zeros((6, 15), dtype=np.int64)
    target[:, 0] = public[:, 0]
    target[:, 1] = TABLES.item_index("leftovers")
    rows = [
        dict(
            game="game",
            source_sha256="0" * 64,
            team_rosters=["a", "b"],
            reveals=public.tolist(),
            truth=target.tolist(),
            verified=np.ones((6, 15), dtype=bool).tolist(),
        )
    ]

    def propose(facts, **kwargs):
        assert np.array_equal(facts, public)
        return target.copy(), {}

    monkeypatch.setattr(audit, "complete_private_team", propose)
    monkeypatch.setattr(audit, "replay_set_priors", lambda: {})
    before = audit.compare(rows, {"p": SimpleNamespace(sha256="1" * 64)}, draws=2)
    changed = copy.deepcopy(rows)
    for mon in changed[0]["truth"]:
        mon[1] = TABLES.item_index("choicescarf")
    after = audit.compare(changed, {"p": SimpleNamespace(sha256="1" * 64)}, draws=2)
    assert before["methods"]["p"]["sample_sha256"] == after["methods"]["p"]["sample_sha256"]
    assert before["methods"]["p"]["metrics"]["families"]["item"]["empirical_top1"] == 1
    assert after["methods"]["p"]["metrics"]["families"]["item"]["empirical_top1"] == 0
