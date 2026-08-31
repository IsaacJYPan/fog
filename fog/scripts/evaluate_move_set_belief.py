"""Read-only permutation-invariant evaluation of the opponent move-set belief.

The belief model intentionally treats a Pokemon's four moves as an unordered set.  Reporting
storage-column accuracy therefore penalizes harmless permutations.  This evaluator instead pools
the unresolved move queries for each Pokemon and scores the hidden set as a set.  It also reports
an assignment-marginalized NLL: the probability of the target set is the sum of the probabilities
of every one-to-one assignment of hidden targets to unresolved query rows (at most 4! terms).
"""
from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from fog.config import load_config_with_args
from fog.models.agent import FogAgent
from fog.training.dataset import StreamingReplayDataset


@dataclass
class MoveSetAccumulator:
    pokemon: int = 0
    targets: int = 0
    hits_at_cardinality: int = 0
    hits_at_4: int = 0
    hits_at_8: int = 0
    exact_at_cardinality: int = 0
    assignment_nll_sum: float = 0.0
    assignment_finite_pokemon: int = 0
    assignment_impossible_pokemon: int = 0
    assignment_finite_targets: int = 0

    def update(
        self, logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor
    ) -> None:
        """Consume ``[B,6,4,V]`` logits/targets/valid masks."""
        flat_logits = logits.detach().reshape(-1, 4, logits.shape[-1])
        flat_targets = targets.detach().reshape(-1, 4)
        flat_valid = valid.detach().reshape(-1, 4) & (flat_targets >= 0)
        counts = flat_valid.sum(dim=-1)
        for h in range(1, 5):
            rows = counts == h
            if not bool(rows.any()):
                continue
            row_logits = flat_logits[rows]
            row_targets = flat_targets[rows]
            row_valid = flat_valid[rows]
            query_logits = row_logits[row_valid].reshape(-1, h, row_logits.shape[-1])
            target = row_targets[row_valid].reshape(-1, h)
            # Malformed duplicate moves are not valid sets. They are absent in authoritative data,
            # but excluding them makes the metric robust to legacy scratch corpora.
            unique = torch.ones(len(target), dtype=torch.bool, device=target.device)
            for left in range(h):
                for right in range(left + 1, h):
                    unique &= target[:, left] != target[:, right]
            query_logits = query_logits[unique]
            target = target[unique]
            if not len(target):
                continue
            log_probs = torch.log_softmax(query_logits, dim=-1)
            pooled = log_probs.max(dim=1).values

            predicted = pooled.topk(h, dim=-1).indices
            hits = predicted.unsqueeze(-1).eq(target.unsqueeze(1)).any(dim=-1).sum(dim=-1)
            n = len(target)
            self.pokemon += n
            self.targets += n * h
            self.hits_at_cardinality += int(hits.sum())
            self.exact_at_cardinality += int((hits == h).sum())
            for k, attr in ((4, "hits_at_4"), (8, "hits_at_8")):
                top = pooled.topk(min(k, pooled.shape[-1]), dim=-1).indices
                top_hits = top.unsqueeze(-1).eq(target.unsqueeze(1)).any(dim=-1).sum()
                setattr(self, attr, getattr(self, attr) + int(top_hits))

            assignment_terms = []
            for permutation in itertools.permutations(range(h)):
                permuted_target = target[:, permutation]
                assignment_terms.append(
                    log_probs.gather(2, permuted_target.unsqueeze(-1)).squeeze(-1).sum(dim=-1)
                )
            set_log_probability = torch.logsumexp(torch.stack(assignment_terms), dim=0)
            finite = torch.isfinite(set_log_probability)
            self.assignment_nll_sum += float((-set_log_probability[finite]).sum())
            self.assignment_finite_pokemon += int(finite.sum())
            self.assignment_impossible_pokemon += int((~finite).sum())
            self.assignment_finite_targets += int(finite.sum()) * h

    def report(self) -> dict:
        return {
            "pokemon_with_hidden_moves": self.pokemon,
            "hidden_move_targets": self.targets,
            "set_recall_at_cardinality": (
                self.hits_at_cardinality / self.targets if self.targets else None
            ),
            "set_precision_at_cardinality": (
                self.hits_at_cardinality / self.targets if self.targets else None
            ),
            "exact_hidden_set_at_cardinality": (
                self.exact_at_cardinality / self.pokemon if self.pokemon else None
            ),
            "target_recall_at_4": self.hits_at_4 / self.targets if self.targets else None,
            "target_recall_at_8": self.hits_at_8 / self.targets if self.targets else None,
            "assignment_marginalized_set_nll_per_pokemon": (
                self.assignment_nll_sum / self.assignment_finite_pokemon
                if self.assignment_finite_pokemon else None
            ),
            "assignment_marginalized_set_nll_per_hidden_move": (
                self.assignment_nll_sum / self.assignment_finite_targets
                if self.assignment_finite_targets else None
            ),
            "assignment_finite_pokemon": self.assignment_finite_pokemon,
            "assignment_impossible_pokemon": self.assignment_impossible_pokemon,
        }


def evaluate(agent, dataset, *, batch_size: int, device: torch.device) -> dict:
    result = MoveSetAccumulator()
    agent.eval()
    with torch.no_grad():
        for batch in dataset.batches(batch_size, shuffle=False, seed=0, device=device):
            output = agent(batch).belief
            if output is None:
                raise RuntimeError("checkpoint has no belief model")
            result.update(
                output.logits["move"],
                batch["belief_targets"]["move"],
                batch["belief_valid"]["move"],
            )
    return result.report()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", required=True, help="LABEL=PATH")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--shard-buffer", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_config_with_args(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = StreamingReplayDataset(args.manifest, shard_buffer=args.shard_buffer)
    report = {
        "status": "complete",
        "read_only": True,
        "permutation_invariant": True,
        "manifest": str(args.manifest),
        "rows": len(dataset),
        "checkpoints": {},
    }
    for specification in args.checkpoint:
        label, path = specification.split("=", 1)
        agent = FogAgent.from_tables(cfg).to(device)
        state = torch.load(path, map_location=device, weights_only=False)
        agent.load_state_dict(state)
        report["checkpoints"][label] = {
            "path": path,
            **evaluate(agent, dataset, batch_size=args.batch_size, device=device),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
