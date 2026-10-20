"""Matched frozen-feature belief diagnostics; private labels are scoring-only."""

from __future__ import annotations

import torch

from fog.pretrained.belief_diagnostics import DistributionMetrics
from fog.pretrained.belief_metrics import SampleMetrics
from fog.pretrained.frozen_belief import public_payload


def evaluation_queries(data, *, split="development", maximum_games=None):
    games = sorted({r["game"] for r in data.rows[split]})
    if maximum_games is not None:
        games = games[:maximum_games]
    selected = []
    for row in data.rows[split]:
        if row["game"] not in games:
            continue
        for index in sorted({0, row["positions"] // 2, row["positions"] - 1}):
            selected.append((row["path"], index))
    return selected


@torch.no_grad()
def score_distributions(
    model,
    data,
    prior,
    specifications,
    *,
    mode,
    device="cpu",
    seed=0,
    chain_queries=8,
    use_prior=True,
    permute_latent=False,
):
    model.eval()
    marginal, conditional = DistributionMetrics(), DistributionMetrics()
    generator = torch.Generator(device=device).manual_seed(seed)
    outcome_errors = []
    metadata = {r["path"]: r for rows in data.rows.values() for r in rows}
    by_game = {}
    for start in range(0, len(specifications), 16):
        batch = data.batch(specifications[start : start + 16], device=device)
        labels, latent = batch["labels"], batch["latent"]
        payload = public_payload(model, labels["reveals"], prior)
        if not use_prior:
            payload.pop("prior")
        if permute_latent:
            if len(specifications) % 6:
                raise ValueError("Phase-matched permutation requires six ordered contexts per game")
            alternate = [
                specifications[(i + 6) % len(specifications)]
                for i in range(start, min(start + 16, len(specifications)))
            ]
            latent = data.batch(alternate, device=device)["latent"]
        if mode == "facts":
            latent = torch.zeros_like(latent)
        elif mode != "native":
            raise ValueError("Unknown belief-context mode")
        context = model.context_projection(latent)
        output = model.belief(context, payload)
        truth = torch.where(labels["verified"], labels["truth"], -1)
        hidden = labels["verified"] & ~model.belief._revealed_mask(labels["reveals"]).reshape_as(
            truth
        )
        marginal.update(
            output.logits, model.belief._split_flat(truth), model.belief._split_flat(hidden)
        )
        repeated = {
            "tokens": payload["tokens"].repeat_interleave(chain_queries, 0),
            "masks": {
                k: v.repeat_interleave(chain_queries, 0) for k, v in payload["masks"].items()
            },
        }
        if use_prior:
            repeated["prior"] = payload["prior"].repeat_interleave(chain_queries)
        chain = model.belief(
            context.repeat_interleave(chain_queries, 0),
            repeated,
            teacher_values=truth.repeat_interleave(chain_queries, 0),
            generator=generator,
            selected_only=True,
        )
        valid = {
            key: value & (chain.loss_targets[key] >= 0) for key, value in chain.loss_valid.items()
        }
        conditional.update(chain.logits, chain.loss_targets, valid)
        # Real frozen latent for the outcome head in every belief-context ablation.
        outcome = model.outcome(batch["latent"]).squeeze(-1).tanh()
        valid_outcome = labels["outcome_valid"].bool()
        outcome_errors.extend(
            (outcome[valid_outcome] - labels["outcomes"][valid_outcome]).square().cpu().tolist()
        )
        for row, (name, index) in enumerate(specifications[start : start + 16]):
            item = metadata[name]
            game = by_game.setdefault(
                item["game"],
                {
                    "marginal": DistributionMetrics(),
                    "conditional": DistributionMetrics(),
                    "team_rosters": item["roster_groups"],
                    "outcome_sum": 0.0,
                    "outcome_count": 0,
                },
            )
            sl = slice(row, row + 1)
            game["marginal"].update(
                {k: v[sl] for k, v in output.logits.items()},
                model.belief._split_flat(truth[sl]),
                model.belief._split_flat(hidden[sl]),
            )
            chain_slice = slice(row * chain_queries, (row + 1) * chain_queries)
            game["conditional"].update(
                {k: v[chain_slice] for k, v in chain.logits.items()},
                {k: v[chain_slice] for k, v in chain.loss_targets.items()},
                {k: v[chain_slice] for k, v in valid.items()},
            )
            if bool(valid_outcome[row]):
                game["outcome_sum"] += float((outcome[row] - labels["outcomes"][row]).square())
                game["outcome_count"] += 1
    grouped = {
        name: {
            "team_rosters": value["team_rosters"],
            "marginal": value["marginal"].families,
            "conditional": value["conditional"].families,
            "outcome_sum": value["outcome_sum"],
            "outcome_count": value["outcome_count"],
        }
        for name, value in by_game.items()
    }
    return {
        "contexts": len(specifications),
        "AR_queries_per_context": chain_queries,
        "public_marginal": marginal.report(),
        "correct_prefix": conditional.report(),
        "outcome_mse": sum(outcome_errors) / len(outcome_errors),
        "context_mode": mode,
        "ancestral_sampling_evaluated": False,
        "prior_used": use_prior,
        "latent_permuted": permute_latent,
        "by_game": grouped,
    }


@torch.no_grad()
def score_samples(
    model,
    data,
    prior,
    specifications,
    *,
    mode,
    device="cpu",
    draws=8,
    seed=0,
    use_prior=True,
    progress=None,
):
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    rows = {r["path"]: r for split in data.rows.values() for r in split}
    total, by_game = SampleMetrics(), {}
    for number, (name, index) in enumerate(specifications):
        record = rows[name]
        batch = data.batch([(name, index)], device=device)
        latent, labels = batch["latent"], batch["labels"]
        payload = public_payload(model, labels["reveals"], prior)
        if not use_prior:
            payload.pop("prior")
        if mode == "facts":
            latent = torch.zeros_like(latent)
        elif mode != "native":
            raise ValueError("Unknown belief-context mode")
        sample = model.sample_team(latent, payload, k=draws, generator=generator, return_trace=True)
        if not (
            torch.isfinite(sample.log_value_prob).all()
            and torch.isfinite(sample.log_route_prob).all()
        ):
            raise FloatingPointError("Nonfinite autoregressive probability trace")
        known = model.belief._revealed_mask(labels["reveals"]).reshape_as(labels["reveals"])
        # Verify public clamps, distinct moves and total EV budget for every sample.
        if not torch.equal(
            sample.teams[0][:, known[0]], labels["reveals"][0][known[0]].expand(draws, -1)
        ):
            raise ValueError("An AR sample changed a public submitted-set fact")
        if bool((sample.teams[..., 8:14].sum(-1) > 127).any()):
            raise ValueError("An AR sample exceeded the encoded EV budget")
        moves = sample.teams[..., 3:7].sort(-1).values
        if bool((moves[..., 1:] == moves[..., :-1]).any()):
            raise ValueError("An AR sample repeated a move")
        for family, columns in [("ability", [2]), ("move", [3, 4, 5, 6])]:
            mask = payload["masks"][family][0]
            for slot in range(6):
                for column in columns:
                    if not bool(known[0, slot, column]) and not bool(
                        mask[slot][sample.teams[0, :, slot, column]].all()
                    ):
                        raise ValueError("An AR sample violated its public species legality mask")
        game = by_game.setdefault(
            record["game"],
            {"metric": SampleMetrics(), "team_rosters": record["roster_groups"], "contexts": 0},
        )
        values = (
            sample.teams[0].cpu().numpy(),
            labels["truth"][0].cpu().numpy(),
            labels["verified"][0].cpu().numpy(),
            known[0].cpu().numpy(),
        )
        total.update(*values)
        game["metric"].update(*values)
        game["contexts"] += 1
        if progress is not None and (number + 1) % 24 == 0:
            progress(number + 1)
    reports = {}
    for name, row in by_game.items():
        metric = row.pop("metric")
        reports[name] = {
            **row,
            "metrics": metric.report(),
            "resampling_data": {
                "fields": metric.fields,
                "joint_count": metric.joint_count,
                "joint_correct": metric.joint_correct,
                "recall_count": metric.recall_count,
                "recall_correct": metric.recall_correct,
            },
        }
    return {
        "contexts": len(specifications),
        "draws": draws,
        "seed": seed,
        "context_mode": mode,
        "metrics": total.report(),
        "by_game": reports,
        "probability_traces_finite": True,
        "public_clamps_and_constraints_passed": True,
        "native_policy_changed": False,
        "search_connected": False,
        "prior_used": use_prior,
    }
