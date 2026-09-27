"""Policy / value / team-preview heads."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from fog.constants.schema import N_ACTIONS

# Posterior features (ARCHITECTURE.md invariant S2): the opponent's belief over MY possible private states,
# summarized per decision as [normalized entropy, mass on the state being evaluated, top-1 mass,
# effective-state fraction exp(H)/K]. The value net cannot price signaling without them — this is
# the consumer that makes joint search's belief coupling matter (see fog/search/joint.py).
N_POSTERIOR_FEATURES = 4


class PolicyValueHead(nn.Module):
    """Policy: (private ‖ public) -> 14 masked logits; value -> tanh expected outcome.

    ``private_state`` is the private-state encoding. ``public_context`` is the explicit fusion of the
    public-state and causal-history encodings. There is no pooled belief input.

    `posterior_dim > 0` (ARCHITECTURE.md invariant S2, config `model.use_posterior_features`) additionally
    concatenates posterior features describing the opponent's belief over my own hidden state; absent
    features fall back to zeros so training paths that don't compute them keep working unchanged.
    """

    def __init__(
        self,
        state_dim: int = 256,
        dyn_dim: int = 256,
        posterior_dim: int = 0,
        action_feature_dim: int = 0,
        action_context_dim: int = 0,
        structured_dim: int = 0,
        shared_state_dim: int = 0,
        shared_token_dim: int = 0,
        action_query_policy: bool = False,
        action_query_heads: int = 4,
        simple_rule_prior_strength: float = 0.0,
    ) -> None:
        super().__init__()
        self.posterior_dim = int(posterior_dim)
        self.action_feature_dim = int(action_feature_dim)
        self.action_context_dim = int(action_context_dim)
        self.structured_dim = int(structured_dim)
        self.shared_state_dim = int(shared_state_dim)
        self.shared_token_dim = int(shared_token_dim)
        self.action_query_policy = bool(action_query_policy)
        if self.shared_token_dim and (
            not self.shared_state_dim or self.shared_state_dim % self.shared_token_dim
        ):
            raise ValueError(
                "shared token mode requires a nonzero divisible shared_state_dim"
            )
        self.simple_rule_prior_strength = float(simple_rule_prior_strength)
        in_dim = (
            self.shared_token_dim
            if self.shared_token_dim
            else self.shared_state_dim
            if self.shared_state_dim
            else state_dim + dyn_dim
        ) + self.posterior_dim
        self.policy = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, N_ACTIONS))
        self.value = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, 1))
        self.action_scorer = None
        if self.action_feature_dim:
            self.action_scorer = nn.Sequential(
                nn.Linear(self.action_feature_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
            # Compatibility guarantee: adding action inputs does not perturb the old policy until
            # a run explicitly trains the scorer or enables the transparent rule prior.
            nn.init.zeros_(self.action_scorer[-1].weight)
            nn.init.zeros_(self.action_scorer[-1].bias)
        self.conditional_action_scorer = None
        if self.action_context_dim:
            conditional_in = (
                in_dim
                + self.structured_dim
                + self.action_context_dim
                + self.action_feature_dim
            )
            self.conditional_action_scorer = nn.Sequential(
                nn.Linear(conditional_in, 256), nn.GELU(), nn.Linear(256, 1)
            )
            # Adding this optional route is behavior-preserving until explicitly trained.
            nn.init.zeros_(self.conditional_action_scorer[-1].weight)
            nn.init.zeros_(self.conditional_action_scorer[-1].bias)
        self.action_query_input = None
        self.action_query_attention = None
        self.action_query_scorer = None
        if self.action_query_policy:
            if not self.shared_token_dim:
                raise ValueError("action-query policy requires contextual shared tokens")
            if not self.action_context_dim:
                raise ValueError("action-query policy requires candidate action context")
            if self.shared_token_dim % int(action_query_heads):
                raise ValueError("shared token width must be divisible by action_query_heads")
            candidate_dim = self.action_context_dim + self.action_feature_dim
            self.action_query_input = nn.Sequential(
                nn.Linear(candidate_dim, self.shared_token_dim),
                nn.GELU(),
                nn.Linear(self.shared_token_dim, self.shared_token_dim),
            )
            self.action_query_attention = nn.MultiheadAttention(
                self.shared_token_dim,
                int(action_query_heads),
                batch_first=True,
            )
            self.action_query_scorer = nn.Sequential(
                nn.LayerNorm(self.shared_token_dim),
                nn.Linear(self.shared_token_dim, 128),
                nn.GELU(),
                nn.Linear(128, 1),
            )
        self.structured_value = None
        if self.structured_dim:
            self.structured_value = nn.Sequential(
                nn.Linear(self.structured_dim, 128), nn.GELU(), nn.Linear(128, 1)
            )
            nn.init.zeros_(self.structured_value[-1].weight)
            nn.init.zeros_(self.structured_value[-1].bias)

    def forward(
        self,
        private_state: torch.Tensor,           # [B, state_dim] private context (+ oracle extra)
        legality_mask: torch.Tensor,           # [B, 14] bool
        public_context: torch.Tensor,          # [B, dyn_dim] fused public state + history
        posterior_features: torch.Tensor | None = None,  # [B, posterior_dim] or None -> zeros
        action_features: torch.Tensor | None = None,  # [B,14,F] aligned per-choice facts
        action_context: torch.Tensor | None = None,  # [B,14,C] candidate/opponent facts
        policy_structured_context: torch.Tensor | None = None,  # [B,S] policy memory read
        value_structured_context: torch.Tensor | None = None,  # [B,S] value memory read
        shared_state: torch.Tensor | None = None,  # [B,Z] common battle latent
        policy_shared_context: torch.Tensor | None = None,  # [B,T] policy read of all tokens
        value_shared_context: torch.Tensor | None = None,  # [B,T] value read of all tokens
        value_logit_bias: torch.Tensor | None = None,  # [B] optional matchup prior
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.shared_state_dim:
            if shared_state is None or tuple(shared_state.shape[1:]) != (self.shared_state_dim,):
                raise ValueError(
                    f"shared_state must have shape [batch,{self.shared_state_dim}]"
                )
            if self.shared_token_dim:
                tokens = shared_state.reshape(
                    shared_state.shape[0], -1, self.shared_token_dim
                )
                # The trunk's final token is the shared global summary after
                # contextual attention over every current-state token.
                default_shared_context = tokens[:, -1]
            else:
                default_shared_context = shared_state
        else:
            default_shared_context = torch.cat([private_state, public_context], dim=-1)
        policy_x = (
            policy_shared_context
            if policy_shared_context is not None
            else default_shared_context
        )
        value_x = (
            value_shared_context
            if value_shared_context is not None
            else default_shared_context
        )
        expected_core_dim = self.policy[0].in_features - self.posterior_dim
        for label, context in (
            ("policy_shared_context", policy_x),
            ("value_shared_context", value_x),
        ):
            if tuple(context.shape[1:]) != (expected_core_dim,):
                raise ValueError(
                    f"{label} must have shape [batch,{expected_core_dim}]"
                )
        if self.posterior_dim:
            if posterior_features is None:
                posterior_features = policy_x.new_zeros(
                    policy_x.shape[0], self.posterior_dim
                )
            policy_x = torch.cat([policy_x, posterior_features], dim=-1)
            value_x = torch.cat([value_x, posterior_features], dim=-1)
        if self.action_query_policy:
            if action_context is None:
                raise ValueError("action-query policy requires action_context")
            candidate_parts = [action_context]
            if self.action_feature_dim:
                if action_features is None:
                    action_features = action_context.new_zeros(
                        action_context.shape[0], N_ACTIONS, self.action_feature_dim
                    )
                candidate_parts.append(action_features)
            candidate_query = self.action_query_input(torch.cat(candidate_parts, dim=-1))
            state_tokens = shared_state.reshape(
                shared_state.shape[0], -1, self.shared_token_dim
            )
            attended, _ = self.action_query_attention(
                candidate_query, state_tokens, state_tokens, need_weights=False
            )
            logits = self.action_query_scorer(candidate_query + attended).squeeze(-1)
        else:
            logits = self.policy(policy_x)
        if self.structured_dim:
            if policy_structured_context is None:
                policy_structured_context = policy_x.new_zeros(
                    policy_x.shape[0], self.structured_dim
                )
            if value_structured_context is None:
                value_structured_context = value_x.new_zeros(
                    value_x.shape[0], self.structured_dim
                )
            for label, context in (
                ("policy_structured_context", policy_structured_context),
                ("value_structured_context", value_structured_context),
            ):
                if tuple(context.shape[1:]) != (self.structured_dim,):
                    raise ValueError(
                        f"{label} must have shape [batch,{self.structured_dim}]"
                    )
        if self.action_scorer is not None:
            if action_features is None:
                action_features = logits.new_zeros(
                    logits.shape[0], N_ACTIONS, self.action_feature_dim
                )
            if tuple(action_features.shape[1:]) != (N_ACTIONS, self.action_feature_dim):
                raise ValueError(
                    "action_features must have shape "
                    f"[batch,{N_ACTIONS},{self.action_feature_dim}]"
                )
            logits = logits + self.action_scorer(action_features).squeeze(-1)
            if self.simple_rule_prior_strength:
                # The final feature is the exact current-state choice of SimpleHeuristicsPlayer.
                # A large positive prior makes a fresh guided run start at rule-player strength;
                # the learned score remains a residual that can later improve on that rule.
                logits = logits + self.simple_rule_prior_strength * action_features[..., -1]
        if self.conditional_action_scorer is not None and not self.action_query_policy:
            if action_context is None:
                action_context = logits.new_zeros(
                    logits.shape[0], N_ACTIONS, self.action_context_dim
                )
            if tuple(action_context.shape[1:]) != (N_ACTIONS, self.action_context_dim):
                raise ValueError(
                    "action_context must have shape "
                    f"[batch,{N_ACTIONS},{self.action_context_dim}]"
                )
            if action_features is None:
                action_features = logits.new_zeros(
                    logits.shape[0], N_ACTIONS, self.action_feature_dim
                )
            shared_parts = [policy_x]
            if policy_structured_context is not None:
                shared_parts.append(policy_structured_context)
            shared = torch.cat(shared_parts, dim=-1).unsqueeze(1).expand(-1, N_ACTIONS, -1)
            conditional = torch.cat([shared, action_context, action_features], dim=-1)
            logits = logits + self.conditional_action_scorer(conditional).squeeze(-1)
        logits = logits.masked_fill(~legality_mask, float("-inf"))
        log_probs = F.log_softmax(logits, dim=-1)
        value_logit = self.value(value_x).squeeze(-1)
        if self.structured_value is not None:
            value_logit = value_logit + self.structured_value(
                value_structured_context
            ).squeeze(-1)
        if value_logit_bias is not None:
            if tuple(value_logit_bias.shape) != tuple(value_logit.shape):
                raise ValueError("value_logit_bias must have shape [batch]")
            value_logit = value_logit + value_logit_bias
        value = torch.tanh(value_logit)
        return log_probs, value


class TeamPreviewHead(nn.Module):
    """Team-preview lead selection (spec §3.1).

    Consumes the encoded preview tokens (my 6 mons in slots 0-5) and scores each as a lead. Played
    live by `FogPlayer.choose_teampreview` and trained as a bandit on game outcome in ppo_update.
    """

    def __init__(self, d: int = 128) -> None:
        super().__init__()
        self.lead = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, preview_tokens: torch.Tensor) -> torch.Tensor:
        """preview_tokens [B, 12, d] (0-5 = mine) -> lead logits [B, 6] over my own mons."""
        return self.lead(preview_tokens[:, :6]).squeeze(-1)
