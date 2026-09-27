"""Stable split groups independent of nicknames, move ordering and roster presentation.

Composition groups are deliberately broader than battle equivalence: party order can affect
Illusion. Keep original pastes and packed hashes for execution; never split these variants apart.
"""
from __future__ import annotations

import hashlib
import json
import re

from fog.env.teams import to_packed


def _id(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def team_group_key(paste: str, *, mode: str = "composition") -> str:
    if mode not in {"composition", "roster"}:
        raise ValueError("team grouping must be composition or roster")
    members = []
    for packed in to_packed(paste).split("]"):
        fields = packed.split("|")
        fields += [""] * max(0, 12 - len(fields))
        species = _id(fields[1] or fields[0])
        if not species:
            raise ValueError("cannot group a team with an empty species")
        if mode == "roster":
            members.append(species)
            continue
        fields[0], fields[1] = "", species
        fields[2], fields[3] = _id(fields[2]), _id(fields[3])
        fields[4] = ",".join(sorted(_id(move) for move in fields[4].split(",") if move))
        fields[5] = _id(fields[5] or "Serious")
        for index, default in ((6, "0"), (8, "31")):
            values = fields[index].split(",")
            values += [""] * (6 - len(values))
            fields[index] = ",".join(value or default for value in values)
        fields[10] = fields[10] or "100"
        misc = fields[11].split(",")
        misc[0] = misc[0] or "255"
        fields[11] = ",".join(misc).rstrip(",")
        members.append("|".join(fields))
    encoded = json.dumps(sorted(members), separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
