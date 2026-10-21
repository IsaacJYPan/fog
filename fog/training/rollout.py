"""Rollout collection: per-game decision storage -> GAE -> one flat C4 PPO batch.

This is deliberately separated from the env loop because it is where the classic silent PPO bugs
live (reward on the wrong step, bootstrap value at episode end, cross-game leakage, nested-dict
stacking mistakes). The env side only has to do:

    buf = RolloutBuffer(gamma=1.0, lam=0.95, beta=cfg_beta)
    g = buf.start_game()
    ... per decision point:
        buf.add_decision(g, frag, action=a, log_prob=lp, value=v, leak_delta=dI)  # dI from the frozen probe
    buf.end_game(g, outcome=+1.0 or -1.0, faints_bonus=0.0)
    ... when buf.n_decisions >= cfg batch size:
        batch = buf.to_batch(device); ppo_update(agent, opt, batch, cfg); buf.clear()

`frag` is the per-decision observation fragment: an (arbitrarily nested) dict of numpy arrays /
torch tensors / python scalars holding everything the agent's forward needs, UNBATCHED (no leading
batch dim) — for FogAgent that is the C3 keys (entities [12,D], ..., legality_mask [14],
opp_revealed{tokens[6,15], masks{...}}, optional opp_view{...}) plus optional label dicts
(belief_targets/belief_valid/...). `to_batch` stacks fragments recursively into [N, ...] tensors;
the special key "history" ([T_i, D_turn], variable T_i) is front-padded to the longest T in the
batch, with the most recent turn always last.

Reward layout: every step gets the leakage shaping −β·max(0, ΔI_t) (zero when β=0 or
no leak); the LAST step additionally gets the terminal ±1 outcome and the optional per-opposing-
faint bonus. γ=1, GAE bootstrap value at episode end is 0 (episodes are finite and terminal).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from fog.constants.schema import SET_N_ATTRS
from fog.training.gae import compute_gae


@dataclass
class _Game:
    frags: list[dict] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    leak_deltas: list[float] = field(default_factory=list)
    # frozen-Φ̂ self-readability CE per decision. When present, per-step ΔI (how much more
    # readable I became this turn) is derived as ce[t]-ce[t+1] at batch time; `final_leak_ce`
    # (measured after the battle ends) closes the last step so end-of-game reveals count too.
    leak_ce: list[float] = field(default_factory=list)
    progress_scores: list[float | None] = field(default_factory=list)
    final_progress_score: float | None = None
    final_leak_ce: float | None = None
    outcome: float | None = None
    bonus: float = 0.0
    # belief targets known only at game end (self-play full truth). Broadcast to every decision;
    # the belief net conditions on the running reveal state so the SAME final target is a valid
    # (censored, in the replay sense) label at every turn.
    belief_targets: np.ndarray | None = None  # [6,15] opponent truth tokens, -1 = no label
    # team-preview pick (trained as a bandit on the game outcome)
    preview: dict | None = None               # {"frag": {...}, "slot": int, "log_prob": float}


class RolloutBuffer:
    def __init__(
        self,
        gamma: float = 1.0,
        lam: float = 0.95,
        beta: float = 0.0,
        clip_deltas: bool = True,
        progress_coef: float = 0.0,
    ) -> None:
        """`clip_deltas`: True = penalize max(0, ΔI) per step; False = signed ΔI,
        whose per-game sum telescopes to CE_first − CE_final (estimator ablation — see
        training.leakage.shaped_reward for why the clipped form is biased by probe-CE noise)."""
        self.gamma = gamma
        self.lam = lam
        self.beta = beta
        self.clip_deltas = clip_deltas
        if not 0.0 <= progress_coef <= 0.5:
            raise ValueError(
                "progress_coef must be between 0 and 0.5 so win/loss remains dominant"
            )
        self.progress_coef = float(progress_coef)
        self._games: dict[int, _Game] = {}
        self._next_id = 0

    # ------------------------------------------------------------------------------- collection API
    def start_game(self) -> int:
        gid = self._next_id
        self._next_id += 1
        self._games[gid] = _Game()
        return gid

    def add_decision(
        self,
        game: int,
        frag: dict,
        *,
        action: int,
        log_prob: float,
        value: float,
        leak_delta: float = 0.0,
        leak_ce: float | None = None,
        progress_score: float | None = None,
    ) -> None:
        g = self._games[game]
        if g.outcome is not None:
            raise RuntimeError(f"game {game} already ended — decisions would leak across games")
        g.frags.append(frag)
        g.actions.append(int(action))
        g.log_probs.append(float(log_prob))
        g.values.append(float(value))
        g.leak_deltas.append(float(leak_delta))
        g.leak_ce.append(float(leak_ce) if leak_ce is not None else None)
        g.progress_scores.append(
            float(progress_score) if progress_score is not None else None
        )

    def add_preview(self, game: int, frag: dict, *, slot: int, log_prob: float) -> None:
        """Record the team-preview pick for `game` (one per game, trained as a bandit)."""
        g = self._games[game]
        if g.outcome is not None:
            raise RuntimeError(f"game {game} already ended — preview would leak across games")
        g.preview = {"frag": frag, "slot": int(slot), "log_prob": float(log_prob)}

    def discard_last_decision(self, game: int) -> None:
        """Discard an uncommitted choice rejected by authoritative Showdown.

        Hidden trapping/disabling information can make a request-advertised action unavailable.
        Showdown then updates the private request and asks that player to choose again without
        advancing the battle. Such an attempt is not an environment transition and must not become
        a PPO row.
        """
        g = self._games[game]
        if g.outcome is not None:
            raise RuntimeError(f"game {game} already ended — cannot discard a decision")
        if not g.frags:
            raise RuntimeError(f"game {game} has no decision to discard")
        for values in (
            g.frags,
            g.actions,
            g.log_probs,
            g.values,
            g.leak_deltas,
            g.leak_ce,
            g.progress_scores,
        ):
            values.pop()

    def end_game(
        self,
        game: int,
        outcome: float,
        faints_bonus: float = 0.0,
        belief_targets: np.ndarray | None = None,
        final_leak_ce: float | None = None,
        final_progress_score: float | None = None,
    ) -> None:
        g = self._games[game]
        if g.outcome is not None:
            raise RuntimeError(f"game {game} ended twice")
        if not g.frags:
            del self._games[game]  # zero-decision game (e.g. forfeit at preview): drop silently
            return
        g.outcome = float(outcome)
        g.bonus = float(faints_bonus)
        g.final_leak_ce = float(final_leak_ce) if final_leak_ce is not None else None
        g.final_progress_score = (
            float(final_progress_score) if final_progress_score is not None else None
        )
        if belief_targets is not None:
            g.belief_targets = np.asarray(belief_targets, dtype=np.int64)

    @property
    def n_decisions(self) -> int:
        return sum(len(g.frags) for g in self._games.values() if g.outcome is not None)

    @property
    def n_games(self) -> int:
        """Finished games currently buffered (dashboard: correct avg game length)."""
        return sum(1 for g in self._games.values() if g.outcome is not None)

    def clear(self) -> None:
        self._games.clear()

    def extend(self, other: RolloutBuffer) -> None:
        """Move finished games from ``other`` into this buffer without re-flattening them.

        Parallel direct-Showdown workers collect into isolated buffers.  Renumbering the games
        here preserves every fragment/label/preview object exactly while preventing worker-local
        game ids from colliding in the centralized learner buffer.
        """
        unfinished = [gid for gid, game in other._games.items() if game.outcome is None]
        if unfinished:
            raise RuntimeError(f"cannot merge unfinished rollout games: {unfinished}")
        for game in other._games.values():
            gid = self._next_id
            self._next_id += 1
            self._games[gid] = game
        other._games.clear()

    # --------------------------------------------------------------------------------- batch build
    def to_batch(self, device=None) -> dict:
        """Flatten all FINISHED games into one C4 batch. Unfinished games raise — dropping them
        silently would bias the data toward short games."""
        unfinished = [gid for gid, g in self._games.items() if g.outcome is None]
        if unfinished:
            raise RuntimeError(f"games not ended before to_batch: {unfinished}")
        games = [g for g in self._games.values()]
        if not games:
            raise RuntimeError("empty rollout buffer")

        frags: list[dict] = []
        actions: list[int] = []
        log_probs: list[float] = []
        advs: list[np.ndarray] = []
        rets: list[np.ndarray] = []
        step_rewards: list[np.ndarray] = []
        game_outcomes: list[np.ndarray] = []
        game_ids: list[int] = []
        belief_tok: list[np.ndarray] = []
        any_belief = False

        for game_id, g in self._games.items():
            t = len(g.frags)
            game_ids.extend([game_id] * t)
            # ΔI_t = ce[t]-ce[t+1] (readability increase) when Φ̂ CE was recorded; else stored deltas.
            # Moving-mask estimator (see training.leakage.delta_information): each
            # CE sums over the attrs still hidden AT THAT measurement, so a hard reveal between t
            # and t+1 contributes its full surprisal under Φ̂ at time t. The post-game measurement
            # (final_leak_ce) closes the last step; without it the final decision's leak is 0.
            if any(c is not None for c in g.leak_ce):
                # None = measurement skipped that decision (e.g. mirror failure). Treating it as 0
                # fabricates a full-CE ΔI spike; forward/back-fill instead, so skipped steps get
                # ΔI=0 and the accumulated change lands when measurement resumes.
                ce = _fill_gaps(g.leak_ce)
                deltas = np.zeros(t, dtype=np.float32)
                deltas[:-1] = ce[:-1] - ce[1:]
                if g.final_leak_ce is not None:
                    deltas[-1] = ce[-1] - g.final_leak_ce
            else:
                deltas = np.asarray(g.leak_deltas, dtype=np.float32)
            penalties = np.clip(deltas, 0.0, None) if self.clip_deltas else deltas
            rewards = -self.beta * penalties
            if self.progress_coef:
                if any(score is None for score in g.progress_scores):
                    raise RuntimeError(
                        "progress reward is enabled but a decision has no progress score"
                    )
                if g.final_progress_score is None:
                    raise RuntimeError(
                        "progress reward is enabled but the finished game has no final score"
                    )
                scores = np.asarray(g.progress_scores, dtype=np.float32)
                next_scores = np.concatenate(
                    [scores[1:], np.asarray([g.final_progress_score], dtype=np.float32)]
                )
                # Potential difference: the sum over a battle is final - initial.  Merely making
                # the battle longer produces exactly zero extra reward.
                rewards = rewards + self.progress_coef * (next_scores - scores)
            rewards[-1] += g.outcome + g.bonus
            values = np.asarray(g.values + [0.0], dtype=np.float32)  # terminal bootstrap = 0
            dones = np.zeros(t, dtype=np.float32)
            dones[-1] = 1.0
            adv, ret = compute_gae(rewards, values, dones, gamma=self.gamma, lam=self.lam)
            frags.extend(g.frags)
            actions.extend(g.actions)
            log_probs.extend(g.log_probs)
            advs.append(adv)
            rets.append(ret)
            step_rewards.append(rewards)
            game_outcomes.append(np.full(t, g.outcome, dtype=np.float32))
            if g.belief_targets is not None:
                any_belief = True
                belief_tok.extend([g.belief_targets] * t)
            else:
                belief_tok.extend([np.full((6, SET_N_ATTRS), -1, dtype=np.int64)] * t)

        batch = _stack_frags(frags, device=device)
        batch["actions"] = torch.as_tensor(actions, dtype=torch.long, device=device)
        batch["game_id"] = torch.as_tensor(game_ids, dtype=torch.long, device=device)
        batch["old_log_probs"] = torch.as_tensor(log_probs, dtype=torch.float32, device=device)
        batch["advantages"] = torch.as_tensor(
            np.concatenate(advs), dtype=torch.float32, device=device
        )
        batch["returns"] = torch.as_tensor(np.concatenate(rets), dtype=torch.float32, device=device)
        # Retain the actual per-decision learning signal for run-time diagnostics.  PPO still
        # optimizes advantages/returns; this tensor makes it possible to prove that a run is seeing
        # both improvements and setbacks instead of silently learning from a constant reward.
        batch["step_rewards"] = torch.as_tensor(
            np.concatenate(step_rewards), dtype=torch.float32, device=device
        )
        # Passive per-decision copy of the terminal result.  This is diagnostics only; PPO still
        # learns from advantages/returns.  It lets exploration reports answer whether a newly
        # sampled action family occurred in wins or losses instead of merely counting selections.
        batch["game_outcomes"] = torch.as_tensor(
            np.concatenate(game_outcomes), dtype=torch.float32, device=device
        )

        if any_belief:
            from fog.constants.tables import TABLES
            from fog.encoding.reveals import (
                FAMILY_COLS,
                align_training_truth_to_views,
                labels_from_truth,
                original_field_hidden,
            )

            toks = np.stack(belief_tok)  # [N,6,15]
            # Targets must follow the INPUT packing convention (revealed moves clamped into the
            # earliest subslots), not paste order — the truth is broadcast per game but the
            # reveal state differs per decision, so alignment is per row against that row's own
            # observation tokens. Frags without opp_revealed (unit-scale tests) align against an
            # all-hidden view, which is the identity.
            views = np.stack([
                np.asarray(f["opp_revealed"]["tokens"])
                if isinstance(f.get("opp_revealed"), dict) and "tokens" in f["opp_revealed"]
                else np.full((6, SET_N_ATTRS), -1, dtype=np.int64)
                for f in frags
            ])
            toks, supervision_views, ok = align_training_truth_to_views(toks, views)
            n_bad = int((~ok).sum())
            if n_bad:
                print(
                    f"[rollout] INFO: {n_bad} slot(s) had an Illusion-like off-sheet public move; "
                    f"public evidence retained and submitted-set truth supervision preserved.",
                    flush=True,
                )
            labels = labels_from_truth(toks)
            # Valid means label >= 0 and the attribute was still hidden at decision time. A
            # revealed attr is a delta row under the new arch — 0/no-grad and meaningless to train —
            # so it must not enter the loss. (Belief-loss SCALE therefore shifts vs the old arch;
            # dashboards are not comparable across the change.)
            hidden = original_field_hidden(supervision_views, TABLES)  # [N, 6, 15]
            consistent_slots = torch.as_tensor(ok, dtype=torch.bool, device=device)
            targets, valid = {}, {}
            for fam, arr in labels.items():
                cols = list(FAMILY_COLS[fam])
                h = hidden[..., cols]
                if fam not in ("move", "ev"):
                    h = h.squeeze(-1)
                    eligible = consistent_slots
                else:
                    eligible = consistent_slots.unsqueeze(-1)
                targets[fam] = torch.as_tensor(arr, dtype=torch.long, device=device)
                valid[fam] = (
                    torch.as_tensor((arr >= 0) & h, dtype=torch.bool, device=device)
                    & eligible
                )
            batch["belief_targets"] = targets
            batch["belief_valid"] = valid

        # opp_view (mirror) plumbing for self-belief training: the mirror frag omits the
        # public turn history (identical to the primary observation) — alias it here instead of
        # storing it twice; then surface the my_view labels as the C4 self-belief targets so
        # ppo_update's self-belief branch fires.
        opp_view = batch.get("opp_view")
        if isinstance(opp_view, dict):
            opp_view.setdefault("history", batch["history"])
            opp_view.setdefault("history_ids", batch.get("history_ids"))
            opp_view.setdefault("history_state", batch.get("history_state"))
            opp_view.setdefault("history_len", batch.get("history_len"))
            labels = (opp_view.get("my_view") or {}).get("labels")
            if labels:
                from fog.models.self_belief import still_hidden_from_tokens

                # Only my attributes the opponent still can't see are trainable leakage
                # targets; the ones they've already seen are delta rows.
                hidden = still_hidden_from_tokens(
                    opp_view["my_view"].get(
                        "supervision_tokens", opp_view["my_view"]["tokens"]
                    )
                )
                consistent_slots = opp_view["my_view"].get("supervision_consistent")
                batch["self_belief_targets"] = labels
                valid = {}
                for family, values in labels.items():
                    family_valid = (values >= 0) & hidden[family]
                    if consistent_slots is not None:
                        eligible = (
                            consistent_slots.unsqueeze(-1)
                            if family in ("move", "ev")
                            else consistent_slots
                        )
                        family_valid = family_valid & eligible
                    valid[family] = family_valid
                batch["self_belief_valid"] = valid

        # preview sub-batch: one row per game that recorded a preview pick. NOT per-decision —
        # ppo_update handles it outside the minibatch loop.
        previews = [g for g in games if g.preview is not None]
        if previews:
            pv_frags = _stack_frags([g.preview["frag"] for g in previews], device=device)
            batch["preview"] = {
                **pv_frags,
                "slot": torch.as_tensor([g.preview["slot"] for g in previews],
                                        dtype=torch.long, device=device),
                "old_log_prob": torch.as_tensor([g.preview["log_prob"] for g in previews],
                                                dtype=torch.float32, device=device),
                "outcome": torch.as_tensor([g.outcome for g in previews],
                                           dtype=torch.float32, device=device),
            }
        return batch


def _fill_gaps(values: list[float | None]) -> np.ndarray:
    """Forward-fill None entries from the last real measurement (back-fill leading Nones)."""
    out = np.empty(len(values), dtype=np.float32)
    last: float | None = None
    for i, v in enumerate(values):
        if v is not None:
            last = float(v)
        out[i] = last if last is not None else np.nan
    first = next(float(v) for v in values if v is not None)
    return np.where(np.isnan(out), first, out).astype(np.float32)


# ------------------------------------------------------------------------------------ frag stacking
def _stack_frags(frags: list[dict], device=None) -> dict:
    """Recursively stack a list of structurally-identical nested dicts into batched tensors."""
    keys = frags[0].keys()
    for f in frags[1:]:
        if f.keys() != keys:
            raise ValueError(f"fragment structure mismatch: {sorted(keys)} vs {sorted(f.keys())}")
    out: dict = {}
    for k in keys:
        vals = [f[k] for f in frags]
        if vals[0] is None:
            if any(v is not None for v in vals):
                raise ValueError(f"fragment key {k!r} is None in some fragments but not others")
            out[k] = None
        elif isinstance(vals[0], dict):
            out[k] = _stack_frags(vals, device=device)
        elif k in ("history", "history_ids", "history_state"):
            dtype = torch.long if k == "history_ids" else torch.float32
            out[k] = _stack_histories(vals, device=device, dtype=dtype)
        else:
            tensors = [torch.as_tensor(v) for v in vals]
            out[k] = torch.stack(tensors).to(device) if device is not None else torch.stack(tensors)
            if out[k].dtype == torch.float64:
                out[k] = out[k].float()
            if out[k].dtype in (torch.int32, torch.int16, torch.uint8) and k != "legality_mask":
                out[k] = out[k].long()
    return out


def _stack_histories(histories: list, device=None, dtype=torch.float32) -> torch.Tensor:
    """[T_i, D] with varying T_i -> [N, T_max, D], front-padded with zeros (recent turn last)."""
    tensors = [torch.as_tensor(h, dtype=dtype) for h in histories]
    t_max = max(1, max(h.shape[0] for h in tensors))
    d = tensors[0].shape[1]
    out = torch.zeros(len(tensors), t_max, d, dtype=dtype)
    for i, h in enumerate(tensors):
        if h.shape[0]:
            out[i, t_max - h.shape[0] :] = h
    return out.to(device) if device is not None else out
