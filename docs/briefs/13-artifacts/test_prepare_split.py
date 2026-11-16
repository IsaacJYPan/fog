"""Exercise the exclusion boundary independently of any trained checkpoint."""
import importlib.util
from pathlib import Path

import numpy as np

from fog.constants.tables import TABLES

spec = importlib.util.spec_from_file_location("prepare_split", Path(__file__).with_name("prepare_split.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_only_acting_composition_is_excluded_and_all_columns_stay_aligned():
    teams = [["amoonguss", "corviknight", "hatterene", "talonflame", "zamazenta", "alomomola"],
             ["greattusk", "dragapult", "kingambit", "gholdengo", "dragonite", "gliscor"]]
    ids = np.array([[TABLES.species_index(s) for s in team] for team in teams])
    own = np.zeros((3, 6, 15), dtype=np.int64)
    own[:, :, 0] = ids[[0, 1, 0]]
    own[2] = own[2, ::-1]  # Same multiset, including reordered/fainted members.
    arrays = {"self_belief_targets": own, "action": np.array([4, 8, 2]),
              "terminal_valid": np.ones(3, dtype=bool),
              "preview_species": np.array(teams), "preview_valid": np.ones(2, dtype=bool)}
    forbidden = {tuple(sorted(ids[0]))}
    filtered, removed = module.filter_arrays(arrays, forbidden)
    assert removed == 2
    assert filtered["action"].tolist() == [8]
    np.testing.assert_array_equal(filtered["self_belief_targets"], own[1:2])
    np.testing.assert_array_equal(filtered["terminal_valid"], [True])
    assert filtered["preview_valid"].tolist() == [False, True]
    assert len(filtered["preview_species"]) == 2
    np.testing.assert_array_equal(arrays["preview_valid"], [True, True])


def test_no_exclusions_preserve_every_array():
    arrays = {"self_belief_targets": np.zeros((3, 6, 15), dtype=np.int64),
              "action": np.array([1, 2, 3]), "preview_species": np.full((2, 6), "greattusk"),
              "preview_valid": np.array([True, False])}
    filtered, removed = module.filter_arrays(arrays, set())
    assert removed == 0
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], filtered[key])
