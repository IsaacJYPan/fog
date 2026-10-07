"""Apply current Showdown rules to a historical human freeze without modifying it."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import random
import shutil
import subprocess

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("METAMON_CACHE_DIR", str(Path("runs/pretrained/cache").resolve()))

from fog.pretrained.kakuna import sha256
from fog.pretrained.loading import read_record, SequenceLoader


def own_paste(record):
    state = record["states"][0]
    mons = [state["player_active_pokemon"], *state["available_switches"]]
    pastes = []
    for mon in mons:
        item = mon["item"]
        heading = mon["name"] + (" @ " + item if item not in ("noitem", "unknown", "") else "")
        # A single unused EV leaves stats unchanged and tells Showdown that the
        # neutral validation spread is intentional, bypassing its zero-EV warning.
        rows = [heading, "Ability: " + mon["ability"], "Level: " + str(mon["lvl"]), "EVs: 1 HP"]
        if mon["tera_type"] not in ("unknown", "notype", ""):
            rows.append("Tera Type: " + mon["tera_type"])
        rows += ["- " + move["name"] for move in mon["moves"]]
        pastes.append("\n".join(rows))
    return "\n\n".join(pastes)


def screen(source: Path, out: Path, showdown: Path, checkpoint: Path, *, samples=1000):
    if out.exists():
        raise FileExistsError("Choose a new screened data identity")
    parent = json.loads((source / "manifest.json").read_text())
    groups = defaultdict(list)
    for record in parent["records"]:
        groups[record["match_id"]].append(record)
    worker = Path(__file__).resolve().parents[1] / "env/team_validation_worker.js"
    rejected, reasons = set(), Counter()
    entries = parent["records"]
    from metamon.interface import UniversalState, UniversalAction, DefaultActionSpace

    action_space = DefaultActionSpace()
    native_reasons = Counter()
    for offset in range(0, len(entries), 300):
        chunk = entries[offset : offset + 300]
        payload = []
        for row in chunk:
            path = source / row["path"]
            if sha256(path) != row["sha256"]:
                raise ValueError("Historical frozen trajectory changed")
            data = read_record(path)
            payload.append(json.dumps(own_paste(data)) + "\n")
            # Audit every recorded action and nonterminal legal set, including
            # states outside any sampled training crop. Reject the whole pair.
            try:
                for raw, action in zip(data["states"][:-1], data["actions"][:-1]):
                    state = UniversalState.from_dict(raw)
                    legal = {
                        action_space.action_to_agent_output(state, candidate)
                        for candidate in UniversalAction.maybe_valid_actions(state)
                    }
                    if not legal or (action >= 0 and action not in legal):
                        raise ValueError(
                            "nonterminal native action mask is empty or excludes the recorded choice"
                        )
            except Exception as error:
                rejected.add(row["match_id"])
                native_reasons[type(error).__name__ + ": " + str(error)] += 1
        result = subprocess.run(
            ["node", str(worker), str(showdown.resolve()), "gen9ou"],
            input="".join(payload),
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=True,
            timeout=120,
        )
        answers = [json.loads(line) for line in result.stdout.splitlines()]
        if len(answers) != len(chunk):
            raise ValueError("Current rules validator omitted a perspective")
        for row, answer in zip(chunk, answers):
            if not answer["valid"]:
                rejected.add(row["match_id"])
                reasons.update(answer["problems"])
        if offset % 3000 == 0:
            print(
                json.dumps(
                    {
                        "screened_perspectives": offset + len(chunk),
                        "rejected_matches": len(rejected),
                    }
                ),
                flush=True,
            )
    # The live importer must be able to reproduce inputs and action masks. This
    # sample is declared before training; a failure stops packaging rather than
    # being dropped in the training loop.
    import torch
    from fog.pretrained.kakuna import load_kakuna

    torch.set_num_threads(1)
    runtime = load_kakuna(checkpoint)
    loader = SequenceLoader(runtime)
    retained = [row for row in entries if row["match_id"] not in rejected]
    if len(retained) < 1000 or samples < 500:
        raise ValueError(
            "Require at least 1000 retained perspectives and a 500-perspective native input sample"
        )
    selected = random.Random(20261007).sample(retained, min(samples, len(retained)))
    failures = []
    for index, row in enumerate(selected):
        try:
            loader.load(("published", str(source / row["path"]), index))
        except Exception as error:
            failures.append(
                {"path": row["path"], "error": type(error).__name__ + ": " + str(error)}
            )
    if failures:
        audit = source.parent / (out.name + "-failed-audit.json")
        audit.write_text(
            json.dumps({"passed": False, "sampled": len(selected), "failures": failures}, indent=2)
            + "\n"
        )
        raise ValueError("Native replay sample failed; inspect " + str(audit))
    out.mkdir(parents=True)
    records = []
    for match_id, pair in sorted(groups.items()):
        if match_id in rejected:
            continue
        if len(pair) != 2:
            raise ValueError("Screening input lost a complete match pair")
        for row in pair:
            target = out / row["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / row["path"], target)
            records.append(row)
    manifest = dict(
        parent,
        matches=len(records) // 2,
        perspectives=len(records),
        records=records,
        bytes=sum(row["bytes"] for row in records),
    )
    manifest["parent_manifest_sha256"] = sha256(source / "manifest.json")
    manifest["current_rules_screen"] = {
        "showdown_revision": subprocess.check_output(
            ["git", "-C", str(showdown), "rev-parse", "HEAD"], text=True
        ).strip(),
        "rejected_matches": len(rejected),
        "reasons": dict(reasons),
        "native_rejections": dict(native_reasons),
        "all_actions_screened": True,
        "native_sample": {"passed": True, "perspectives": len(selected), "seed": 20261007},
        "limitation": "Validates reconstructed own sets with neutral EV/IV defaults (one stat-neutral EV); submitted human teams are unverified.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "matches": manifest["matches"],
                "bytes": manifest["bytes"],
                "rejected_matches": len(rejected),
                "native_sample": len(selected),
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--showdown", type=Path, default=Path("server/pokemon-showdown"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1000)
    args = parser.parse_args()
    screen(args.source, args.out, args.showdown, args.checkpoint, samples=args.samples)


if __name__ == "__main__":
    main()
