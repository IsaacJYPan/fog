"""Versioned validation for experiment configuration mappings.

The runtime still consumes dictionaries for checkpoint compatibility, but every loaded YAML and
CLI-overridden config crosses this schema boundary first.  Unknown keys are errors: a misspelled
learning-rate or backend option must never silently select a default.
"""
from __future__ import annotations

from dataclasses import dataclass
from difflib import get_close_matches
from typing import Any


@dataclass(frozen=True)
class ConfigSchema:
    version: int
    sections: dict[str, frozenset[str]]
    scalar_roots: frozenset[str]


_SECTION_KEYS = {
    "pins": {"showdown_commit", "ou_banlist_date", "usage_month", "usage_rating_cut"},
    "env": {
        "actor_threads", "backend", "balanced_team_sampling", "battle_format",
        "belief_supervision", "concurrent_battles", "engine_batch", "games_per_chunk",
        "min_team_pool_size", "no_security", "node_binary", "num_workers",
        "opponent_cache_size", "rollout_worker_mode", "rollout_worker_timeout",
        "independent_opponent_rng",
        "rollout_workers", "server_url", "showdown_root", "team_pool",
    },
    "encoding": {
        "current_dim", "dyn_dim", "entity_hidden", "history_cap", "history_dim",
        "history_encoder", "species_embed_dim", "state_dim", "dropout",
    },
    "model": {
        "aux_into_trunk", "belief_decoder_layers", "belief_dim",
        "belief_last_context_probability", "blind_reactive", "exact_action_context",
        "history_gru_hidden", "history_heads", "history_transformer_layers", "history_ff_dim", "oracle",
        "public_action_features_only", "shared_entity_dim", "shared_field_dim",
        "shared_contextual_data_global", "shared_contextual_dynamic_residual",
        "shared_contextual_head_readers",
        "shared_contextual_ff_dim", "shared_contextual_heads", "shared_contextual_layers",
        "shared_contextual_dropout", "shared_typed_world_tokens",
        "matchup_value_prior", "matchup_value_prior_initial_only",
        "shared_recent_history_tokens",
        "shared_latent_contextual", "shared_latent_dim", "shared_latent_enabled",
        "shared_latent_flat_autoencoder", "shared_latent_learned_only",
        "shared_latent_preserve_scale", "shared_latent_trunk", "shared_residual_dim",
        "shared_flat_hidden_dim", "shared_flat_layers",
        "simple_rule_prior_strength", "simple_rule_prior_strength_final", "structured_memory",
        "structured_token_dim", "token_action_policy", "action_query_policy",
        "action_query_heads", "use_action_features",
        "use_posterior_features",
    },
    "loss": {
        "action_coef", "belief_chain_coef", "belief_coef", "preview_coef",
        "require_exact_chosen_action_features", "require_exact_chosen_legality",
        "self_belief_chain_coef", "self_belief_coef", "value_coef",
    },
    "log": {"checkpoint_every_decisions", "dir", "halt_on_regression_sigma", "tensorboard"},
    "ppo": {
        "target_kl",
        "batch_decisions", "bc_anchor_ckpt", "bc_anchor_rule_prior_strength", "clip",
        "entropy_coef", "entropy_coef_final", "epochs", "games_per_chunk", "gamma",
        "init_ckpt", "init_compatible", "kl_anchor_coef", "kl_anchor_decisions", "lam", "lr",
        "max_grad_norm", "minibatch_size", "resume_ckpt", "rollout_temperature",
        "rollout_tera_logit_bonus", "total_decisions", "trainable_prefixes",
        "rollout_tera_logit_bonus_initial", "rollout_tera_logit_bonus_final",
        "rollout_tera_logit_bonus_decay_decisions", "gradient_diagnostics",
        "trainable_lr_scales",
        "heuristic_teacher", "heuristic_coef", "heuristic_coef_final", "heuristic_decay_decisions",
    },
    "league": {
        "freeze_history",
        "latest_frac", "past_ckpts", "past_frac", "scripted", "scripted_frac",
        "snapshot_every_decisions", "warmup_decisions", "warmup_opponent",
    },
    "leakage": {
        "allow_unique_species_pool", "baseline_run", "beta", "clip_delta", "family_weights",
        "fresh_probe", "fresh_probe_steps", "refreeze_every_decisions",
        "watchdog_warmup_decisions",
    },
    "preflight": {
        "match_source", "min_games", "min_parsed_replay_dryruns", "min_parsed_replay_games",
        "min_parsed_replay_rows", "min_replay_dryruns", "min_replay_hidden_targets",
        "replay_report", "replay_schema_version", "report", "required", "schema_version",
    },
    "cross_sim": {"every_decisions", "games", "halt_gap"},
    "bc": {
        "auxiliary_eval_batches", "auxiliary_eval_weight", "auxiliary_every_primary_batches",
        "auxiliary_gradient_surgery", "auxiliary_joint_batch",
        "auxiliary_head_lr_scale", "auxiliary_loss_overrides",
        "auxiliary_holdout_manifest", "auxiliary_manifest",
        "auxiliary_selection_metrics",
        "force_auxiliary_policy_valid", "force_primary_policy_valid",
        "guard_eval_batches", "guard_holdout_manifest", "guard_metric",
        "guard_train_manifest", "guard_train_value_only",
        "batch_size", "early_stopping_min_delta", "early_stopping_patience",
        "encoder_lr_scale", "epochs", "eval_batches", "holdout_fraction", "holdout_manifest", "init_ckpt",
        "init_compatible", "limit_rows", "lr", "lr_final", "manifest",
        "max_primary_batches_per_epoch", "reader_lr_scale",
        "primary_selection_metrics",
        "reconstruction_coef", "reconstruction_decoder", "reconstruction_slot_dim", "scope",
        "shared_lr_scale", "shard_buffer", "streaming",
    },
    "eval": {
        "closure_games", "closure_opponent", "games_per", "include_reading_opponent",
        "lvn_games", "readability", "temperature",
    },
    "gate": {
        "action_top1", "belief_item_top1_prereveal", "min_battles_per_sec", "vs_max_damage",
        "vs_simple_heuristics",
    },
    "run": {"agent", "kind", "num_games", "worker_sweep"},
    "reward": {"progress_coef"},
    "replay_policy": {
        "batch_size", "belief_chain_coef", "belief_coef", "coef", "manifest",
        "shard_buffer", "steps_per_update", "value_coef",
    },
    # Automation-job sections share the loader in user tooling even though PPO does not consume
    # them directly.
    "training": {
        "batch_decisions", "blocks", "checkpoint_every_decisions", "clip",
        "decisions_per_block", "epochs", "lr", "minibatch_size", "seed", "tensorboard",
    },
    "evaluation": {
        "promotion_family_alpha", "promotion_min_games", "champion_games", "champion_seed_offset",
        "deterioration_benchmark_regression", "deterioration_head_to_head_below", "device",
        "enabled", "format", "games", "head_to_head", "max_turns", "opponents", "pool",
        "progress_every", "seed", "viewer_max_benchmark_regression",
        "viewer_min_head_to_head",
    },
    "retention": {
        "backup_checkpoints", "keep_numbered_per_block", "larger_evaluation_games",
        "maximum_fallback_uses", "prune_numbered_after_evaluation",
    },
    "safety": {"auto_refresh", "device", "games", "replay_audit_config"},
    # These sections intentionally have data-dependent keys or non-mapping values.
    "overrides": set(),
    "sources": set(),
}

SCHEMA = ConfigSchema(
    version=1,
    sections={name: frozenset(keys) for name, keys in _SECTION_KEYS.items()},
    scalar_roots=frozenset({
        "base_config", "device", "experiment", "initial_checkpoint", "mixed_precision", "name",
        "output_dir", "seed", "trainer", "version",
    }),
)


def _unknown_message(path: str, key: str, allowed: set[str] | frozenset[str]) -> str:
    close = get_close_matches(key, sorted(allowed), n=1)
    hint = f"; did you mean {close[0]!r}?" if close else ""
    return f"unknown config key {path + '.' if path else ''}{key!s}{hint}"


def validate_config(config: dict[str, Any], schema: ConfigSchema = SCHEMA) -> None:
    """Raise on unknown keys, malformed section types, or invalid stable enum values."""
    if not isinstance(config, dict):
        raise TypeError("resolved config must be a mapping")
    allowed_roots = set(schema.scalar_roots) | set(schema.sections)
    for key in config:
        if key not in allowed_roots:
            raise ValueError(_unknown_message("", str(key), allowed_roots))
    for section, allowed in schema.sections.items():
        if section not in config:
            continue
        value = config[section]
        if section in {"sources"} and not isinstance(value, dict):
            continue
        if not isinstance(value, dict):
            raise TypeError(f"config section {section!r} must be a mapping")
        if section == "overrides":
            continue
        for key in value:
            if key not in allowed:
                raise ValueError(_unknown_message(section, str(key), allowed))

    version = config.get("version", schema.version)
    if not isinstance(version, int) or version != schema.version:
        raise ValueError(f"unsupported config version {version!r}; expected {schema.version}")
    backend = (config.get("env") or {}).get("backend")
    if backend is not None and backend not in {"showdown_direct", "showdown", "engine"}:
        raise ValueError(f"env.backend must be showdown_direct, showdown, or engine; got {backend!r}")
    history = (config.get("encoding") or {}).get("history_encoder")
    if history is not None and history not in {"causal_transformer", "gru"}:
        raise ValueError(f"encoding.history_encoder must be causal_transformer or gru; got {history!r}")
    oracle = (config.get("model") or {}).get("oracle")
    if oracle is not None and oracle not in {"none", "set", "action"}:
        raise ValueError(f"model.oracle must be none, set, or action; got {oracle!r}")


__all__ = ["SCHEMA", "ConfigSchema", "validate_config"]
