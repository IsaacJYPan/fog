"""Create the synthetic source used for the runner interruption check.

Run from the repository root with the project environment. These are generated
protocol records, not games retrieved from the pinned replay corpus.
"""
import ast
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

root = Path("runs/blueprint_runner_fixture_source")
root.mkdir(parents=True, exist_ok=True)
module = ast.parse(Path("tests/test_replay_parser.py").read_text())
fixture = next(node for node in module.body if isinstance(node, ast.FunctionDef)
               and node.name == "test_replay_parser_builds_opposite_public_perspectives")
log = next(node.value.value for node in fixture.body if isinstance(node, ast.Assign)
           and any(isinstance(target, ast.Name) and target.id == "log" for target in node.targets))
rows = [
    {"id": f"gen9ou-fixture-{index:04d}", "formatid": "gen9ou", "format": "[Gen 9] OU",
     "rating": "1800", "uploadtime": 1707000000 + index,
     "players": ["Alice", "Bob"], "log": log}
    for index in range(600)
]
pq.write_table(pa.Table.from_pylist(rows), root / "synthetic-public-replays.parquet")
(root / "README.txt").write_text(
    "Synthetic replay records for runner interruption integration test; copied two-turn "
    "protocol fixture from tests/test_replay_parser.py; not observations from the pinned "
    "corpus and not a learning curve.\n"
)
print(root / "synthetic-public-replays.parquet")
