"""Rung-1/2 conditioning gate for the belief chain: prove the chain
CONDITIONS on reveals — on controlled distributions with KNOWN joint structure — before any
usage-data training is trusted.

WHY THIS EXISTS. The autoregressive chain's entire advantage over independent per-attribute heads
is modeling the JOINT structure of hidden attributes. Usage data cannot demonstrate that: Smogon
chaos files carry marginals only and the generative pool draws attributes independently, so on that
data the true joint IS the product of marginals — nothing to condition on (KNOWN_ISSUES). Worse,
the marginal is the DEFAULT OPTIMIZATION BASIN: the 2026-07-13 scratch run watched a small chain
fit a confident marginal for ~300 steps before discovering the designed dependency at all. So the
gate trains on synthetic distributions where the joint is designed and conditioning is checkable
exactly, and only a pass here licenses training on data whose joint structure we cannot inspect.

RUNG 2 (the real gate) — a designed cross-slot rule, preview-indistinguishable. Every example
shares the SAME 6 species (species always revealed = team preview carries zero information about
the hidden tail). All hidden attributes are iid uniform EXCEPT one arbitrary designed rule:

    tera[slot 0]  =  rule_map[ item[slot 3] ]        (rule_map a fixed injective random map)

The source (flat 28) sits AFTER the dependent (flat 8) in canonical order, so a fixed left-to-right
chain can never condition the dependent on a revealed source — exactly the property the reveal-set
prefix model exists to provide. n_tera (8) > n_items (4), which separates three
behaviors: a model beats uniform (ln 8 = 2.079) by learning the image support, reaches the true
marginal (ln 4 = 1.386) without any conditioning, and reaches 0 only by conditioning.

WHAT THE SCRATCH RUN GOT WRONG (both fixed here). Its eval (a) scored ONE random tail order and
(b) revealed only the source while training revealed ~30% of everything — a train/eval mismatch —
so its late-training "probe drifted back to a constant" could not be distinguished from an
OOD-eval artifact. Here: (a) every chain metric averages over --orders fresh random tail orders;
(b) the eval set's reveal distribution IS the training distribution — per-example reveal rate
~ U(0, p_max), which also puts sparse-reveal states (the strict probe's neighborhood, and real
turn-1 information states) in-support; (c) every metric is a TIME SERIES, so late-training
regression (the drift-back question) is measured rather than guessed.

THE GATE (checked at the FINAL eval step, i.e. conditioning must PERSIST, plus an explicit
no-drift-back check against the best step):
  (i)   conditions on a revealed source:      dep CE | source revealed          <= 0.20 nats;
        and on a teacher-forced tail source:  dep CE | source earlier in tail   <= 0.20 nats
        (the second is the input pattern ancestral sampling / joint scoring actually query);
        and held-out total chain CE beats the marginal-product baseline;
  (ii)  probe matches the designed rule:      argmax p(tera0 | item3=v) == rule_map[v] for all v,
        both with nothing else revealed (strict) and under training-matched extra reveals;
  (iii) no-reveal fallback:                   with the source hidden and not yet generated, the
        dependent's conditional is the MARGINAL (uniform over the image), not a constant:
        CE within 0.15 of ln 4, image mass >= 0.95, max single prob <= 0.40;
  plus  sampling consistency: ancestral samples with the source revealed obey the rule >= 0.95.
Reported but NOT gated: reverse consistency — with nothing revealed, canonical-order sampling draws
tera0 BEFORE item3, so rule-consistent (item3, tera0) pairs require the learned REVERSE conditional
p(item3 | tera0): the strongest joint-structure statistic here.

RUNG 1 (memorization sanity, POOL-A style) — 8 fixed teams SHARING the 6 species, uniform team
draws, the same reveal process. Gate: model chain CE within 0.10 nats of the EXACT mixture floor
(the true 8-team posterior conditional, computed per tail order), sampled teams are members of the
set (>= 0.95), and a uniquely-identifying reveal (all 6 items) makes samples reproduce the full
90-attribute team exactly (>= 0.95).

Everything runs on the real BeliefModel (small dims) with belief_context = 0 — the synthetic world has no
dynamics, so conditioning can only come from the prefix/tail tokens, which is precisely what is
under test. Training mirrors ppo.py: BOTH the marginal (pass-A) and chain (pass-B) CE terms every
step (§4.5, decision 2). This is load-bearing for the eval — the probe, the fallback read, and
sample_team all query the pass-A canonical forward, so without the marginal term pass A is OOD and
reads a hedged conditional even when the chain has learned the rule (measured: chain-only leaves the
probe at p(rule)≈0.28 with 100% argmax). Data is sampled fresh every step (infinite data): the fixed
eval set is held out by construction. Per §3.7 sequencing, re-run this gate after the §3.4 EV-native
surgery to confirm the 109-token chain still conditions.

Run (defaults are the gate):
    python -m fog.eval.belief_conditioning --rung 2
    python -m fog.eval.belief_conditioning --rung 1
Writes runs/belief_conditioning/rung<k>_seed<s>.json (config + references + series + verdict);
exit code 0 iff the gate passed.

OPTION-(c) ABLATION (§3.7 owner decision). `--tail-order canonical` trains AND evaluates the chain
in a FIXED left-to-right tail instead of the any-order random tail, for BOTH rungs. Motivation: the
random tail is trained by neither the marginal term (canonical zero-fill) nor a regime matching
`sample_team`, which walks CANONICAL order with progressive fill — so ancestral sampling sits in an
untrained regime and hedges (measured sample fidelity 0.295 = exp(-1.214)). A canonical chain trains
exactly the conditionals `sample_team` traverses (each hidden attr conditions on the prefix reveals
plus earlier-canonical tail tokens; later tokens are causally masked in both training and sampling),
so this ablation tests whether sampling SHARPENS. It gives up the any-order property (a HIDDEN source
later in canonical order can no longer inform an earlier dependent through the tail — reveal-based
conditioning survives, it lives in the bidirectional prefix); the `conditions_on_forced_tail` check is
therefore N/A here (its bin is empty by construction). Writes ..._canonical.json so both arms persist.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import torch

from fog.models.belief import N_ATTRS, N_SLOTS, SLOT_ATTRS, BeliefModel
from fog.training.losses import belief_loss, belief_output_loss

# The designed rule: source = slot-3 item (flat 28, LATE in canonical order), dependent = slot-0
# tera (flat 8, EARLY). A fixed left-to-right chain cannot condition flat 8 on a revealed flat 28.
SRC_SLOT, SRC_COL = 3, 1
DEP_SLOT, DEP_COL = 0, 14
SRC_FLAT = SRC_SLOT * len(SLOT_ATTRS) + SRC_COL
DEP_FLAT = DEP_SLOT * len(SLOT_ATTRS) + DEP_COL

_SCALAR_COLS = {"species": 0, "item": 1, "ability": 2, "nature": 7, "tera": 14}
_FAM_COLS = {"species": (0,), "item": (1,), "ability": (2,), "move": (3, 4, 5, 6),
             "nature": (7,), "ev": (8, 9, 10, 11, 12, 13), "tera": (14,)}
_FAM_FLATS = {f: tuple(s * len(SLOT_ATTRS) + c for s in range(N_SLOTS) for c in cols)
              for f, cols in _FAM_COLS.items()}
_FAM_SIZE_KEY = {"species": "n_species", "item": "n_items", "ability": "n_abilities",
                  "move": "n_moves", "nature": "n_natures",
                  "ev": "n_ev_units", "tera": "n_tera"}


# ------------------------------------------------------------------- the designed distribution
@dataclass
class GateSpec:
    """Rung-2 distribution: fixed shared species, iid-uniform hidden attributes, one designed
    cross-slot rule tera[DEP_SLOT] = rule_map[item[SRC_SLOT]]. `p_reveal_max`: per-example reveal
    rate ~ U(0, p_reveal_max) (covers sparse AND dense information states, like a battle does)."""

    sizes: dict[str, int]
    species_team: torch.Tensor  # [6] fixed values, shared by every example (preview-uninformative)
    rule_map: torch.Tensor      # [n_items] -> tera value, injective
    p_reveal_max: float = 0.6


def make_spec(seed: int, p_reveal_max: float = 0.6) -> GateSpec:
    g = torch.Generator().manual_seed(seed)
    sizes = {
        "n_species": 8, "n_items": 4, "n_abilities": 6, "n_moves": 16,
        "n_natures": 5, "n_ev_units": 8, "n_tera": 8,
    }
    species_team = torch.randperm(sizes["n_species"], generator=g)[:N_SLOTS]
    rule_map = torch.randperm(sizes["n_tera"], generator=g)[: sizes["n_items"]]
    return GateSpec(sizes=sizes, species_team=species_team, rule_map=rule_map,
                    p_reveal_max=p_reveal_max)


def sample_truth(spec: GateSpec, n: int, g: torch.Generator) -> torch.Tensor:
    """Synthetic true teams: fixed species, uniform iid fields (moves distinct),
    then the designed rule overwrites the dependent."""
    sz = spec.sizes
    t = torch.zeros(n, N_SLOTS, len(SLOT_ATTRS), dtype=torch.long)
    t[:, :, 0] = spec.species_team
    t[:, :, 1] = torch.randint(0, sz["n_items"], (n, N_SLOTS), generator=g)
    t[:, :, 2] = torch.randint(0, sz["n_abilities"], (n, N_SLOTS), generator=g)
    t[:, :, 3:7] = torch.rand(n, N_SLOTS, sz["n_moves"], generator=g).argsort(-1)[..., :4]
    t[:, :, 7] = torch.randint(0, sz["n_natures"], (n, N_SLOTS), generator=g)
    t[:, :, 8:14] = torch.randint(
        0, sz["n_ev_units"], (n, N_SLOTS, 6), generator=g
    )
    t[:, :, 14] = torch.randint(0, sz["n_tera"], (n, N_SLOTS), generator=g)
    t[:, DEP_SLOT, DEP_COL] = spec.rule_map[t[:, SRC_SLOT, SRC_COL]]
    return t


def sample_view(spec: GateSpec, truth: torch.Tensor, g: torch.Generator,
                rate: float | None = None) -> torch.Tensor:
    """Observation tokens off `truth`: species always revealed (preview); everything else revealed
    iid at the example's rate (None -> per-example ~ U(0, p_reveal_max)). Moves follow the clamp
    convention: the FIRST k subslots of a slot are revealed (k ~ Bin(4, rate)) — revealed moves
    pack the earliest subslots and view order = truth order, as in engine games."""
    n = truth.shape[0]
    if rate is None:
        r = torch.rand(n, 1, 1, generator=g) * spec.p_reveal_max
    else:
        r = torch.full((n, 1, 1), float(rate))
    keep = torch.rand(n, N_SLOTS, len(SLOT_ATTRS), generator=g) < r
    keep[:, :, 0] = True
    keep[:, :, 3:7] = keep[:, :, 3:7].to(torch.uint8).sort(dim=-1, descending=True).values.bool()
    return torch.where(keep, truth, torch.full_like(truth, -1))


def targets_and_valid(truth: torch.Tensor, view: torch.Tensor) -> tuple[dict, dict]:
    """Per-family chain-CE targets + hidden-only valid masks (the training convention)."""
    targets = {f: truth[:, :, c] for f, c in _SCALAR_COLS.items()}
    targets["move"] = truth[:, :, 3:7]
    targets["ev"] = truth[:, :, 8:14]
    hidden = view < 0
    valid = {f: hidden[:, :, c] for f, c in _SCALAR_COLS.items()}
    valid["move"] = hidden[:, :, 3:7]
    valid["ev"] = hidden[:, :, 8:14]
    return targets, valid


# ------------------------------------------------------------------------- order capture + refs
class _CaptureOrder:
    """Record the per-example content order the forward ACTUALLY used (the real path runs
    unchanged) — needed to split the dependent's CE by where the source sat, and for the exact
    reference CEs, whose move/dependent terms are order-dependent."""

    def __init__(self, model: BeliefModel) -> None:
        self.model = model
        self.orders: list[torch.Tensor] = []

    def __enter__(self) -> Self:
        self._orig = self.model._training_route

        def wrapped(revealed, target_valid, tail_order, generator):
            route, length = self._orig(revealed, target_valid, tail_order, generator)
            # Preserve the old diagnostic representation: physical order contains every reveal,
            # START, then the set-AR route.  The model itself never consumes this physical layout.
            content = torch.full(
                (len(revealed), N_ATTRS + 1), N_ATTRS,
                dtype=torch.long, device=revealed.device,
            )
            for row in range(len(revealed)):
                prefix = torch.nonzero(revealed[row], as_tuple=True)[0]
                n_revealed = len(prefix)
                content[row, :n_revealed] = prefix
                n_hidden = int(length[row])
                content[row, n_revealed + 1:n_revealed + 1 + n_hidden] = (
                    route[row, :n_hidden]
                )
            self.orders.append(content)
            return route, length

        self.model._training_route = wrapped
        return self

    def __exit__(self, *exc) -> None:
        self.model._training_route = self._orig


def _seq_pos_from_route(
    revealed: torch.Tensor, route: torch.Tensor, length: torch.Tensor
) -> torch.Tensor:
    """Content-id -> physical rank for reveal-set prefix plus a generated route."""
    seq_pos = torch.full(
        (len(revealed), N_ATTRS + 1), N_ATTRS + 1,
        dtype=torch.long, device=revealed.device,
    )
    for row in range(len(revealed)):
        prefix = torch.nonzero(revealed[row], as_tuple=True)[0]
        seq_pos[row, prefix] = torch.arange(len(prefix), device=revealed.device)
        seq_pos[row, N_ATTRS] = len(prefix)
        n_hidden = int(length[row])
        if n_hidden:
            seq_pos[row, route[row, :n_hidden]] = (
                torch.arange(n_hidden, device=revealed.device) + len(prefix) + 1
            )
    return seq_pos


def _family_mean_sum(per_entry: torch.Tensor, valid: torch.Tensor) -> float:
    """Aggregate a [B, 90] per-entry quantity exactly like belief_loss: mean over each family's
    valid entries, summed across families (a family with no valid entry contributes 0)."""
    total = 0.0
    for flats in _FAM_FLATS.values():
        v = valid[:, list(flats)]
        if v.any():
            total += float(per_entry[:, list(flats)][v].mean())
    return total


def reference_ce(spec: GateSpec, seq_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact per-entry CE [B, 90] of (marginal-product baseline, true conditional) for the rung-2
    distribution under the order the forward used. Uniform families: marginal = conditional = ln N.
    Moves: conditional = ln(n_moves - #same-slot moves already in the conditioning set) — revealed
    moves sit in the prefix, physically before every tail token, so "earlier physical position"
    covers revealed and teacher-forced predecessors uniformly. Dependent: marginal = ln n_items
    (uniform over the rule image); conditional = 0 iff the source is known before it."""
    sz = spec.sizes
    b = seq_pos.shape[0]
    pos = seq_pos[:, :N_ATTRS].float()
    marg = torch.zeros(b, N_ATTRS, device=seq_pos.device)
    cond = torch.zeros(b, N_ATTRS, device=seq_pos.device)
    for fam in ("item", "ability", "nature", "ev", "tera"):
        val = math.log(sz[_FAM_SIZE_KEY[fam]])
        marg[:, list(_FAM_FLATS[fam])] = val
        cond[:, list(_FAM_FLATS[fam])] = val
    marg[:, DEP_FLAT] = math.log(sz["n_items"])
    src_before = pos[:, SRC_FLAT] < pos[:, DEP_FLAT]
    cond[:, DEP_FLAT] = torch.where(src_before, 0.0, math.log(sz["n_items"]))
    for s in range(N_SLOTS):
        fl = [s * len(SLOT_ATTRS) + c for c in (3, 4, 5, 6)]
        pm = pos[:, fl]                                            # [B, 4]
        count = (pm.unsqueeze(1) < pm.unsqueeze(2)).sum(-1)        # [B, 4] earlier same-slot moves
        cond[:, fl] = (sz["n_moves"] - count.float()).log()
        marg[:, fl] = math.log(sz["n_moves"])
    return marg, cond


# ------------------------------------------------------------------------------- rung-2 metrics
def _zero_belief_context(model: BeliefModel, b: int, device) -> torch.Tensor:
    return torch.zeros(b, model.s_proj.in_features, device=device)


@torch.no_grad()
def _probe_metrics(model: BeliefModel, spec: GateSpec, device, *, matched: bool, n_per_v: int,
                   seed: int) -> tuple[float, float]:
    """Reveal the source (+ training-matched extra reveals iff `matched`), keep the dependent
    hidden, read the dependent's MARGINAL conditional (pass A — the pool_id / dashboard read).
    Returns (argmax-follows-rule accuracy, mean prob on the rule value) over all source values."""
    g = torch.Generator().manual_seed(seed)
    n_src = spec.sizes["n_items"]
    v = torch.arange(n_src).repeat_interleave(n_per_v)
    truth = sample_truth(spec, n_src * n_per_v, g)
    truth[:, SRC_SLOT, SRC_COL] = v
    truth[:, DEP_SLOT, DEP_COL] = spec.rule_map[v]
    view = sample_view(spec, truth, g, rate=None if matched else 0.0)
    view[:, SRC_SLOT, SRC_COL] = v
    view[:, DEP_SLOT, DEP_COL] = -1
    out = model(_zero_belief_context(model, len(v), device), {"tokens": view.to(device), "masks": None})
    probs = out.logits["tera"][:, DEP_SLOT].softmax(-1).cpu()
    want = spec.rule_map[v]
    acc = float((probs.argmax(-1) == want).float().mean())
    p_rule = float(probs.gather(1, want.unsqueeze(1)).mean())
    return acc, p_rule


@torch.no_grad()
def _fallback_metrics(model: BeliefModel, spec: GateSpec, device) -> tuple[float, float, float]:
    """Nothing revealed but species: the dependent's marginal read must be ~uniform over the rule
    image (ln n_items), NOT a constant. Returns (CE over the image, image mass, max single prob);
    CE - ln(n_items) = KL(uniform-image || model)."""
    view = torch.full((1, N_SLOTS, len(SLOT_ATTRS)), -1, dtype=torch.long)
    view[:, :, 0] = spec.species_team
    out = model(_zero_belief_context(model, 1, device), {"tokens": view.to(device), "masks": None})
    p = out.logits["tera"][0, DEP_SLOT].softmax(-1).cpu()
    image = p[spec.rule_map]
    return float(-image.clamp_min(1e-12).log().mean()), float(image.sum()), float(p.max())


@torch.no_grad()
def _sample_metrics(model: BeliefModel, spec: GateSpec, device, *, b: int, k: int,
                    seed: int) -> tuple[float, float]:
    """Ancestral-sampling rule consistency. Forward: source revealed -> sampled dependent must be
    its rule image. Reverse: nothing revealed -> canonical order draws the dependent BEFORE the
    source, so consistency requires the learned reverse conditional p(source | dependent)."""
    g = torch.Generator().manual_seed(seed)
    truth = sample_truth(spec, b, g)
    rule = spec.rule_map.to(device)
    c0 = _zero_belief_context(model, b, device)

    def draw(view: torch.Tensor) -> torch.Tensor:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        return model.sample_team(c0, {"tokens": view.to(device), "masks": None}, k=k,
                                 generator=gen)

    view0 = sample_view(spec, truth, g, rate=0.0)                  # species only
    t0 = draw(view0)
    rev = float((t0[..., DEP_SLOT, DEP_COL] == rule[t0[..., SRC_SLOT, SRC_COL]]).float().mean())
    view1 = view0.clone()
    view1[:, SRC_SLOT, SRC_COL] = truth[:, SRC_SLOT, SRC_COL]      # + source revealed
    t1 = draw(view1)
    want = rule[truth[:, SRC_SLOT, SRC_COL].to(device)].view(b, 1)
    fwd = float((t1[..., DEP_SLOT, DEP_COL] == want).float().mean())
    return fwd, rev


@torch.no_grad()
def eval_rung2(model: BeliefModel, spec: GateSpec, truth: torch.Tensor, view: torch.Tensor,
               device, *, n_orders: int, eval_seed: int, sample_b: int, sample_k: int,
               probe_n_per_v: int = 256, tail_order: str = "random") -> dict:
    """The full fair eval: chain CE averaged over `n_orders` random tail orders on the fixed
    held-out set (reveal distribution = training distribution), the dependent's CE split by where
    the source sat, both probes, the fallback read, and sampling consistency.

    `tail_order` selects the chain's generation order to evaluate under, matching training: "random"
    (the any-order tail) or "canonical" (the option-(c) ablation — fixed left-to-right tail that
    matches what `sample_team` traverses). Under "canonical" the order is deterministic, so the
    per-order averaging collapses to a single pass and the `src_hidden_after` bin is empty (a hidden
    source is always later than the earlier dependent, so it can never inform it through the tail)."""
    was_training = model.training
    model.eval()
    if tail_order == "canonical":
        n_orders = 1
    truth_d, view_d = truth.to(device), view.to(device)
    b = truth.shape[0]
    belief_context = _zero_belief_context(model, b, device)
    hidden_flat = view_d.reshape(b, N_ATTRS) < 0
    dep_hidden = hidden_flat[:, DEP_FLAT]
    src_revealed = ~hidden_flat[:, SRC_FLAT]

    tot = {"model": 0.0, "marg": 0.0, "cond": 0.0}
    bins: dict[str, list] = {"src_revealed": [], "src_hidden_after": [], "src_hidden_before": []}
    for m in range(n_orders):
        gen = torch.Generator(device=device)
        gen.manual_seed(eval_seed * 1000 + m)
        _, per_field_logp, route, length = model.path_logprob(
            belief_context, {"tokens": view_d, "masks": None}, truth_d,
            order_mode=tail_order, generator=gen, return_steps=True,
        )
        seq_pos = _seq_pos_from_route(
            model._revealed_mask(view_d), route, length
        )
        per_field_ce = -per_field_logp
        tot["model"] += _family_mean_sum(per_field_ce, hidden_flat)
        marg, cond = reference_ce(spec, seq_pos)
        tot["marg"] += _family_mean_sum(marg, hidden_flat)
        tot["cond"] += _family_mean_sum(cond, hidden_flat)
        ce = per_field_ce[:, DEP_FLAT]
        before = seq_pos[:, DEP_FLAT] < seq_pos[:, SRC_FLAT]
        bins["src_revealed"].append(ce[dep_hidden & src_revealed])
        bins["src_hidden_after"].append(ce[dep_hidden & ~src_revealed & ~before])
        bins["src_hidden_before"].append(ce[dep_hidden & ~src_revealed & before])

    metrics = {
        "total_ce": tot["model"] / n_orders,
        "total_ce_marginal_ref": tot["marg"] / n_orders,
        "total_ce_conditional_ref": tot["cond"] / n_orders,
    }
    for name, lst in bins.items():
        cat = torch.cat(lst)
        metrics[f"dep_ce_{name}"] = float(cat.mean()) if cat.numel() else float("nan")
        metrics[f"dep_n_{name}"] = int(cat.numel())
    metrics["probe_acc_strict"], metrics["probe_prule_strict"] = _probe_metrics(
        model, spec, device, matched=False, n_per_v=1, seed=eval_seed + 11)
    metrics["probe_acc_matched"], metrics["probe_prule_matched"] = _probe_metrics(
        model, spec, device, matched=True, n_per_v=probe_n_per_v, seed=eval_seed + 12)
    fb_ce, fb_mass, fb_max = _fallback_metrics(model, spec, device)
    metrics.update(fallback_ce=fb_ce, fallback_image_mass=fb_mass, fallback_max_prob=fb_max)
    fwd, rev = _sample_metrics(model, spec, device, b=sample_b, k=sample_k, seed=eval_seed + 13)
    metrics.update(sample_fwd_consistency=fwd, sample_rev_consistency=rev)
    if was_training:
        model.train()
    return metrics


def _window_mean(series: list[dict], key: str, i: int, w: int) -> float:
    win = series[i : i + w]
    return sum(s[key] for s in win) / len(win)


def _drift_vs_best(series: list[dict], key: str, w: int = 3, worst: str = "max") -> float:
    """Trailing-window mean of `key` minus its best window mean over the run. De-noises the chain
    metrics' tail-order oscillation (a single-eval final-vs-best comparison flags that ±0.1 wobble as
    regression). Positive = a metric that should stay LOW drifted up (worst='max' picks the best =
    min window); negative-capable for accuracies (worst='min')."""
    n = len(series)
    ww = min(w, n)
    windows = [_window_mean(series, key, i, ww) for i in range(n - ww + 1)]
    tail = _window_mean(series, key, n - ww, ww)
    return tail - (min(windows) if worst == "max" else max(windows))


def verdict_rung2(series: list[dict], refs: dict, tail_order: str = "random") -> dict:
    final = series[-1]
    checks = {
        "beats_marginal_total": final["total_ce"] <= final["total_ce_marginal_ref"] - 0.05,
        "conditions_on_reveal": final["dep_ce_src_revealed"] <= 0.20,
        # The any-order tail lets a hidden source (teacher-forced earlier in the tail) inform the
        # dependent — the property `sample_team` queries under "random". Under the option-(c)
        # "canonical" ablation the tail is fixed left-to-right, so a hidden source is ALWAYS later
        # than the earlier dependent: this bin is empty by construction and the check is N/A (None,
        # excluded from `passed`), NOT a failure. Reveal-conditioning still lives in the prefix.
        "conditions_on_forced_tail": (
            None if tail_order == "canonical" else final["dep_ce_src_hidden_after"] <= 0.20
        ),
        "probe_strict_exact": final["probe_acc_strict"] >= 1.0,
        "probe_matched": final["probe_acc_matched"] >= 0.98,
        # "genuinely uses the reveal": with NOTHING revealed the pass-A marginal forward (the read
        # pool_id / dashboards use) must give tera0's image-marginal, not a constant. Keyed off the
        # pass-A fallback read, NOT the chain's dep_ce_src_hidden_before — that bin is perturbed by
        # the same irrelevant-tail-token distraction as the chain conditional (a hedge finding, not a
        # fallback failure), so folding it in here would conflate two different phenomena.
        "fallback_is_marginal": (
            abs(final["fallback_ce"] - refs["dep_marginal_ce"]) <= 0.10
            and final["fallback_image_mass"] >= 0.95
            and final["fallback_max_prob"] <= 0.40
        ),
        "sampling_follows_rule": final["sample_fwd_consistency"] >= 0.95,
    }
    # Regression check keys off the STABLE probe accuracy (the clean conditioning signal — the old
    # single-order eval's "probe drifted back to a constant" was the fear this gate exists to test);
    # the chain CE is checked too but on trailing-window means so its tail-order oscillation isn't
    # misread as regression.
    drift = {
        "probe_acc_matched_window": _drift_vs_best(series, "probe_acc_matched", worst="min"),
        "dep_ce_src_revealed_window": _drift_vs_best(series, "dep_ce_src_revealed", worst="max"),
    }
    checks["no_drift_back"] = (
        drift["probe_acc_matched_window"] >= -0.02
        and drift["dep_ce_src_revealed_window"] <= 0.10
    )
    return {"checks": checks, "drift": drift,
            "passed": all(v for v in checks.values() if v is not None)}


# ------------------------------------------------------------------------------- rung-1 metrics
def make_rung1_teams(spec: GateSpec, seed: int, n_teams: int = 8) -> torch.Tensor:
    """[n_teams, 6, 15] fixed teams sharing the species line-up, item vectors pairwise distinct
    (so revealing all 6 items identifies the team uniquely — the rung-1 identification reveal)."""
    g = torch.Generator().manual_seed(seed)
    for _ in range(64):
        teams = sample_truth(spec, n_teams, g)
        if len({tuple(t[:, 1].tolist()) for t in teams}) == n_teams:
            return teams
    raise RuntimeError("could not draw item-distinguishable teams; change the seed")


def rung1_floor(teams_flat: torch.Tensor, truth_flat: torch.Tensor, hidden: torch.Tensor,
                seq_pos: torch.Tensor) -> float:
    """Exact chain-CE floor for the uniform mixture over the fixed teams, under the order the
    forward used: each entry's true conditional is the team-frequency of its value among the teams
    consistent with everything known before it (reveals + teacher-forced earlier tail = agreement
    with the true team on every earlier attribute). Aggregated like belief_loss."""
    pos = seq_pos[:, :N_ATTRS]
    known = pos.unsqueeze(1) < pos.unsqueeze(2)                    # [B, p, f]: f earlier than p
    match = teams_flat.unsqueeze(1) == truth_flat.unsqueeze(0)     # [T, B, f]
    violate = (known.unsqueeze(0) & ~match.unsqueeze(2)).any(-1)   # [T, B, p]
    consistent = ~violate
    num = (consistent & match).sum(0)                              # [B, p]
    den = consistent.sum(0)
    ce = -(num.float() / den.float()).clamp_min(1e-12).log()
    return _family_mean_sum(ce, hidden)


@torch.no_grad()
def eval_rung1(model: BeliefModel, spec: GateSpec, teams: torch.Tensor, truth: torch.Tensor,
               view: torch.Tensor, device, *, n_orders: int, eval_seed: int, sample_b: int,
               sample_k: int, tail_order: str = "random") -> dict:
    was_training = model.training
    model.eval()
    if tail_order == "canonical":
        n_orders = 1
    truth_d, view_d = truth.to(device), view.to(device)
    b = truth.shape[0]
    belief_context = _zero_belief_context(model, b, device)
    hidden_flat = view_d.reshape(b, N_ATTRS) < 0
    teams_flat = teams.reshape(len(teams), N_ATTRS).to(device)
    truth_flat = truth_d.reshape(b, N_ATTRS)

    tot_model = tot_floor = 0.0
    for m in range(n_orders):
        gen = torch.Generator(device=device)
        gen.manual_seed(eval_seed * 1000 + m)
        _, per_field_logp, route, length = model.path_logprob(
            belief_context, {"tokens": view_d, "masks": None}, truth_d,
            order_mode=tail_order, generator=gen, return_steps=True,
        )
        seq_pos = _seq_pos_from_route(
            model._revealed_mask(view_d), route, length
        )
        tot_model += _family_mean_sum(-per_field_logp, hidden_flat)
        tot_floor += rung1_floor(teams_flat, truth_flat, hidden_flat, seq_pos)

    g = torch.Generator().manual_seed(eval_seed + 21)
    idx = torch.randint(0, len(teams), (sample_b,), generator=g)
    truth_s = teams[idx]
    c0 = _zero_belief_context(model, sample_b, device)

    def draw(view_s: torch.Tensor) -> torch.Tensor:
        gen = torch.Generator(device=device)
        gen.manual_seed(eval_seed + 22)
        return model.sample_team(c0, {"tokens": view_s.to(device), "masks": None}, k=sample_k,
                                 generator=gen)

    view0 = sample_view(spec, truth_s, g, rate=0.0)                # species only
    samp = draw(view0).reshape(sample_b, sample_k, N_ATTRS)        # [B, k, 90]
    member = (samp.unsqueeze(2) == teams_flat.view(1, 1, -1, N_ATTRS)).all(-1).any(-1)
    view1 = view0.clone()
    view1[:, :, 1] = truth_s[:, :, 1]                              # + all 6 items (unique id)
    samp1 = draw(view1).reshape(sample_b, sample_k, N_ATTRS)
    want = truth_s.reshape(sample_b, N_ATTRS).to(samp1.device)
    exact = (samp1 == want.unsqueeze(1)).all(-1)

    metrics = {
        "total_ce": tot_model / n_orders,
        "total_ce_floor": tot_floor / n_orders,
        "excess_ce": (tot_model - tot_floor) / n_orders,
        "sample_membership": float(member.float().mean()),
        "sample_ident_exact": float(exact.float().mean()),
    }
    if was_training:
        model.train()
    return metrics


def verdict_rung1(series: list[dict]) -> dict:
    final = series[-1]
    checks = {
        "ce_near_exact_floor": final["excess_ce"] <= 0.10,
        "samples_in_pool": final["sample_membership"] >= 0.95,
        "ident_reveal_reproduces_team": final["sample_ident_exact"] >= 0.95,
    }
    return {"checks": checks, "passed": all(checks.values())}


# ------------------------------------------------------------------------------------ run loops
def _device(arg: str) -> str:
    if arg != "auto":
        return arg
    return "cuda" if torch.cuda.is_available() else "cpu"


_HEADER2 = ("  step |  total (marg/cond)  | dep-ce: srcR tail> none | probe: str mtch |"
            "  fallback ce/mass/max | smp: fwd rev")
_HEADER1 = "  step |  total    floor   excess | member ident"


def _row2(m: dict) -> str:
    return (f"{m['step']:6d} | {m['total_ce']:6.3f} ({m['total_ce_marginal_ref']:.3f}/"
            f"{m['total_ce_conditional_ref']:.3f}) | {m['dep_ce_src_revealed']:5.3f} "
            f"{m['dep_ce_src_hidden_after']:5.3f} {m['dep_ce_src_hidden_before']:5.3f} | "
            f"{m['probe_acc_strict']:4.2f} {m['probe_acc_matched']:4.2f} | "
            f"{m['fallback_ce']:5.3f} {m['fallback_image_mass']:4.2f} "
            f"{m['fallback_max_prob']:4.2f} | {m['sample_fwd_consistency']:4.2f} "
            f"{m['sample_rev_consistency']:4.2f}")


def _row1(m: dict) -> str:
    return (f"{m['step']:6d} | {m['total_ce']:7.4f} {m['total_ce_floor']:7.4f} "
            f"{m['excess_ce']:7.4f} | {m['sample_membership']:5.2f} "
            f"{m['sample_ident_exact']:5.2f}")


def _train(args, spec: GateSpec, sample_batch, evaluate, header: str, row) -> list[dict]:
    """Shared loop: eval at step 0, train on fresh batches, eval every eval_every + at the end.

    Trains BOTH belief CE terms every step, exactly like ppo.py (marginal `belief_coef` + chain
    `belief_chain_coef`, both default 1.0, same targets): the chain term is the primary aux loss and
    the sampling/scoring input pattern; the marginal term keeps the pass-A canonical forward — the
    read pool_id, the dashboards, the leakage probe, and the probe/fallback metrics here all use —
    directly calibrated instead of only weight-shared with the chain. Omitting it (chain-only) leaves
    pass A OOD and its conditioning hedged; `--marginal-coef 0` reproduces that as an ablation."""
    device = _device(args.device)
    torch.manual_seed(args.seed)                     # weight init + training tail orders
    model = BeliefModel(state_dim=args.state_dim, d=args.d, **spec.sizes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    g_data = torch.Generator().manual_seed(args.seed + 1)
    series: list[dict] = []

    def do_eval(step: int) -> None:
        m = {"step": step, **evaluate(model, device)}
        series.append(m)
        print(row(m), flush=True)

    print(header, flush=True)
    do_eval(0)
    model.train()
    belief_context = torch.zeros(args.batch, args.state_dim, device=device)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        truth = sample_batch(args.batch, g_data)
        view = sample_view(spec, truth, g_data)
        truth, view = truth.to(device), view.to(device)
        targets, valid = targets_and_valid(truth, view)
        out_chain = model(belief_context, {"tokens": view, "masks": None}, teacher_values=truth,
                          tail_order=args.tail_order)
        loss = args.chain_coef * belief_output_loss(out_chain, targets, valid)
        if args.marginal_coef > 0:                       # pass-A term (ppo.py's belief_coef)
            out_marg = model(belief_context, {"tokens": view, "masks": None})
            loss = loss + args.marginal_coef * belief_loss(out_marg.logits, targets, valid)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % args.eval_every == 0 or step == args.steps:
            do_eval(step)
            model.train()
    print(f"[done] {args.steps} steps in {time.time() - t0:.0f}s", flush=True)
    return series


def run_rung2(args) -> dict:
    device = _device(args.device)
    spec = make_spec(args.seed, args.p_reveal_max)
    refs = {
        "dep_uniform_ce": math.log(spec.sizes["n_tera"]),
        "dep_marginal_ce": math.log(spec.sizes["n_items"]),
        "dep_conditional_ce": 0.0,
    }
    print(f"[rung2] device={device} tail_order={args.tail_order} "
          f"rule item[{SRC_SLOT}]->tera[{DEP_SLOT}] "
          f"map={spec.rule_map.tolist()} | dep refs: uniform={refs['dep_uniform_ce']:.3f} "
          f"marginal={refs['dep_marginal_ce']:.3f} conditional=0.000", flush=True)
    g_eval = torch.Generator().manual_seed(args.seed + 2)
    eval_truth = sample_truth(spec, args.eval_n, g_eval)
    eval_view = sample_view(spec, eval_truth, g_eval)

    def evaluate(model, dev):
        return eval_rung2(model, spec, eval_truth, eval_view, dev, n_orders=args.orders,
                          eval_seed=args.seed + 100, sample_b=args.sample_b,
                          sample_k=args.sample_k, tail_order=args.tail_order)

    series = _train(args, spec, lambda n, g: sample_truth(spec, n, g), evaluate, _HEADER2, _row2)
    verdict = verdict_rung2(series, refs, tail_order=args.tail_order)
    return {"rung": 2, "config": vars(args), "rule_map": spec.rule_map.tolist(),
            "references": refs, "series": series, "verdict": verdict}


def run_rung1(args) -> dict:
    device = _device(args.device)
    spec = make_spec(args.seed, args.p_reveal_max)
    teams = make_rung1_teams(spec, args.seed + 50)
    print(f"[rung1] device={device} tail_order={args.tail_order} teams={len(teams)} "
          f"(shared species, distinct item vectors)", flush=True)
    g_eval = torch.Generator().manual_seed(args.seed + 2)
    idx = torch.randint(0, len(teams), (args.eval_n,), generator=g_eval)
    eval_truth = teams[idx]
    eval_view = sample_view(spec, eval_truth, g_eval)

    def evaluate(model, dev):
        return eval_rung1(model, spec, teams, eval_truth, eval_view, dev, n_orders=args.orders,
                          eval_seed=args.seed + 100, sample_b=args.sample_b,
                          sample_k=args.sample_k, tail_order=args.tail_order)

    def sample_batch(n, g):
        return teams[torch.randint(0, len(teams), (n,), generator=g)]

    series = _train(args, spec, sample_batch, evaluate, _HEADER1, _row1)
    verdict = verdict_rung1(series)
    return {"rung": 1, "config": vars(args), "series": series, "verdict": verdict}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--rung", type=int, choices=(1, 2), default=2)
    p.add_argument("--steps", type=int, default=None, help="default: 6000 (rung 2) / 2000 (rung 1)")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--d", type=int, default=32)
    p.add_argument("--state-dim", type=int, default=32)
    p.add_argument("--marginal-coef", type=float, default=1.0,
                   help="pass-A marginal CE weight (ppo.py belief_coef); 0 = chain-only ablation")
    p.add_argument("--chain-coef", type=float, default=1.0,
                   help="pass-B chain CE weight (ppo.py belief_chain_coef)")
    p.add_argument("--tail-order", choices=("random", "canonical"), default="random",
                   help="chain generation order for BOTH training and eval. 'random' = the any-order "
                        "tail (default); 'canonical' = the option-(c) ablation (fixed left-to-right "
                        "tail matching what sample_team traverses)")
    p.add_argument("--p-reveal-max", type=float, default=0.6,
                   help="per-example reveal rate ~ U(0, this)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-n", type=int, default=2048)
    p.add_argument("--orders", type=int, default=8, help="random tail orders averaged per eval")
    p.add_argument("--sample-b", type=int, default=256)
    p.add_argument("--sample-k", type=int, default=8)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="runs/belief_conditioning")
    args = p.parse_args(argv)
    if args.steps is None:
        args.steps = 6000 if args.rung == 2 else 2000

    result = run_rung2(args) if args.rung == 2 else run_rung1(args)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.tail_order == "random" else f"_{args.tail_order}"
    out_path = out_dir / f"rung{args.rung}_seed{args.seed}{suffix}.json"
    out_path.write_text(json.dumps(result, indent=1))
    v = result["verdict"]
    print(f"\n[verdict] rung {args.rung}: {'PASS' if v['passed'] else 'FAIL'}")
    for name, ok in v["checks"].items():
        label = "N/A " if ok is None else ("PASS" if ok else "FAIL")
        print(f"  {label}  {name}")
    if "drift" in v:
        for name, val in v["drift"].items():
            print(f"  drift  {name} = {val:+.4f}")
    print(f"[saved] {out_path}")
    return 0 if v["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
