"""Controlled comparison with Metamon's actual Gen9 sampling/merge functions.

Load only named AST definitions from a locally downloaded, revision-pinned source checkout.
Supply FOG's *same* usage tables to isolate algorithm differences. This is NOT a benchmark of
Metamon's complete parser, historical usage dataset, or published reconstructed trajectories.
No dependency replacement, package installation, or automatic network access is performed.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import numpy as np

from fog.constants.schema import NATURE_INDEX, SET_EV_COLS, TYPE_INDEX
from fog.constants.tables import TABLES
from fog.training.replay_policy_data import complete_private_team, replay_set_priors

PINNED_REVISION = "0a00a759c9a4382a2877088d828302ec294a05a5"
SOURCE_FILES = {
    "metamon/backend/team_prediction/usage_stats/legacy_team_builder.py": (
        "legacy_team_builder.py", "d4a245f0d2e5119ee01c7ad6ad114323c2885a538df97b2f686417c3421f1f29"),
    "metamon/backend/team_prediction/team.py": (
        "metamon/backend/team_prediction/team.py",
        "27175630b987c5dc6d24ff7037129bc76085ab87237f39f3ddf2df9c00d5eacb"),
    "metamon/backend/team_prediction/usage_stats/constants.py": (
        "metamon/backend/team_prediction/usage_stats/constants.py",
        "5b64478e08281c994643d7920e95b94d4053385d0ff846b66677587a6207df58"),
}


def download_baseline(out: Path) -> None:
    """Explicit, bounded download of three public source files; no package installation."""
    from fog.scripts.import_metamon_replays import _request

    out.mkdir(parents=True, exist_ok=True)
    for remote, (relative, expected) in SOURCE_FILES.items():
        path = out / relative
        if path.exists():
            data = path.read_bytes()
        else:
            url = f"https://raw.githubusercontent.com/UT-Austin-RPL/metamon/{PINNED_REVISION}/{remote}"
            with _request(url) as response:
                data = response.read(1024 * 1024)
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"upstream source hash mismatch: {relative}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (out / "revision.txt").write_text(PINNED_REVISION, encoding="utf-8")


def _definitions(path: Path, names: set[str]) -> list:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [node for node in tree.body if getattr(node, "name", None) in names]


class MetamonSharedPriorBaseline:
    """Use upstream move sampling, scalar sampling and partial-set merging unmodified."""

    def __init__(self, source: Path):
        source = Path(source)
        if (source / "revision.txt").read_text().strip() != PINNED_REVISION:
            raise ValueError("Metamon baseline revision does not match the audited source")
        for relative, expected in SOURCE_FILES.values():
            if hashlib.sha256((source / relative).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Metamon baseline source changed: {relative}")
        builder_path = source / "legacy_team_builder.py"
        team_path = source / "metamon/backend/team_prediction/team.py"
        constants_path = source / "metamon/backend/team_prediction/usage_stats/constants.py"
        namespace = {"np": np}
        # The constants module contains literal definitions only. Do not execute imports/top-level
        # effects from an external package in our training process.
        constants = ast.parse(constants_path.read_text(encoding="utf-8"))
        assignments = [node for node in constants.body if isinstance(node, ast.Assign)]
        exec(compile(ast.Module(body=assignments, type_ignores=[]), str(constants_path), "exec"),
             namespace)
        nodes = _definitions(builder_path, {"weighted_random_choice", "TeamBuilder"})
        builder = next(node for node in nodes if isinstance(node, ast.ClassDef))
        builder.body = [node for node in builder.body if isinstance(node, ast.FunctionDef)
                        and node.name in {"check_valid_moves", "get_valid_moves", "generate_moveset"}]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(builder_path), "exec"), namespace)
        pokemon = _definitions(team_path, {"PokemonSet"})[0]
        pokemon.decorator_list = []
        pokemon.body = [node for node in pokemon.body if isinstance(node, ast.Assign)
                        or (isinstance(node, ast.FunctionDef) and node.name == "fill_from_PokemonSet")]
        namespace["pokemon_name"] = lambda value: str(value).lower().replace("-", "")
        exec(compile(ast.Module(body=[pokemon], type_ignores=[]), str(team_path), "exec"), namespace)
        self.PokemonSet = namespace["PokemonSet"]
        self.builder = namespace["TeamBuilder"]()
        self.builder.gen = 9
        self.builder.format = "gen9ou"
        self.builder.verbose = False
        self.builder.isgen2 = self.builder.isgen7 = False
        self.builder.stat = {}
        priors = replay_set_priors()
        for sid in TABLES.species["ids"]:
            spreads = TABLES.spreads.get("species", {}).get(sid, [])
            self.builder.stat[sid] = {
                "moves": dict(priors.get("move_priors", {}).get(sid, {})),
                "items": dict(priors.get("item_priors", {}).get(sid, {})),
                "abilities": dict(priors.get("ability_priors", {}).get(sid, {})),
                "tera_types": dict(priors.get("tera_priors", {}).get(sid, {})),
                "spreads": {str(s["nature"]) + ":" + "/".join(map(str, s["evs"])):
                            float(s.get("weight", 0)) for s in spreads},
            }
        self.metadata = {
            "name": "metamon_gen9_core_shared_priors",
            "revision": (source / "revision.txt").read_text().strip(),
            "files_sha256": {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in (builder_path, team_path, constants_path)},
            "scope": "upstream sampling and merge code; FOG usage data and fallback; not full Metamon",
        }
        self.coverage = {"sampled_slots": 0, "fallback_slots": 0}

    def complete(self, revealed: np.ndarray, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
        # Upstream uses numpy's global RNG; restore it so this comparator cannot affect other arms.
        state = np.random.get_state()
        np.random.seed(seed % 2**32)
        result = revealed.copy()
        try:
            for slot, row in enumerate(revealed):
                if row[0] < 0:
                    continue
                sid = TABLES.species["ids"][int(row[0])]
                stat = self.builder.stat[sid]
                if any(not value or sum(value.values()) <= 0 for value in stat.values()):
                    self.coverage["fallback_slots"] += 1
                    continue  # common FOG fallback, explicitly outside upstream comparison coverage
                self.coverage["sampled_slots"] += 1
                generated = self.builder.generate_moveset(sid)
                other = self.PokemonSet()
                current = self.PokemonSet()
                for obj in (current, other):
                    obj.name = obj.base_species = sid
                    obj.ivs = [31] * 6
                nature, evs = generated["spread"].split(":")
                other.moves = generated["moves"]
                other.ability, other.item = generated["ability"], generated["item"]
                other.nature, other.tera_type = nature, generated["tera_type"]
                other.evs = list(map(int, evs.split("/")))
                current.moves = [TABLES.moves["ids"][int(m)] if m >= 0 else current.MISSING_MOVE
                                 for m in row[3:7]]
                current.ability = (TABLES.abilities["ids"][int(row[2])] if row[2] >= 0
                                   else current.MISSING_ABILITY)
                current.item = TABLES.items["ids"][int(row[1])] if row[1] >= 0 else current.MISSING_ITEM
                current.nature = current.MISSING_NATURE
                current.tera_type = current.MISSING_TERA_TYPE
                current.evs = [current.MISSING_EV] * 6
                current.fill_from_PokemonSet(other)
                proposed = row.copy()
                proposed[1] = TABLES.item_index(current.item)
                proposed[2] = TABLES.ability_index(current.ability, default=-1)
                proposed[3:7] = [TABLES.move_index(m, default=-1) for m in current.moves]
                proposed[7] = NATURE_INDEX.get(current.nature.lower(), -1)
                proposed[list(SET_EV_COLS)] = [ev // 4 for ev in current.evs]
                proposed[14] = {**TYPE_INDEX, "stellar": 18}.get(current.tera_type.lower(), -1)
                hidden = row < 0
                result[slot, hidden] = proposed[hidden]
        finally:
            np.random.set_state(state)
        completed, provenance = complete_private_team(result, seed=seed)
        provenance[(revealed < 0) & (completed >= 0)] = 1
        return completed, provenance


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch checksum-pinned upstream benchmark functions")
    parser.add_argument("--out", type=Path, required=True)
    download_baseline(parser.parse_args().out)
