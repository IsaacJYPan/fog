"""Strict local loading and native observations for Kakuna checkpoint 34.

Install Metamon in its own environment; importing FOG does not require it.
Upstream code: https://github.com/UT-Austin-RPL/metamon (MIT).
Published weights: https://huggingface.co/jakegrigsby/metamon (Apache-2.0).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import torch

from fog.pretrained.attention import WindowAttention

KAKUNA_34_SHA256 = "7e7474d5b2ec6624afdeb29a7cf1dc5c84817d5aee8ec39c1d35a153b18df26c"
INTERFACE_VERSION = "kakuna34-metamon-default13-window96-fp32-legal-gen9ou-v4"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass
class KakunaRuntime:
    model: Any
    observation_space: Any
    action_space: Any
    reward_function: Any
    source_sha256: str
    interface_version: str = INTERFACE_VERSION

    def distribution(self, obs, rl2s, time_idxs, hidden_state=None):
        latent, hidden = self.model.get_state_embedding(
            obs=obs,
            rl2s=rl2s,
            time_idxs=time_idxs,
            hidden_state=hidden_state,
        )
        distribution = self.model.actor(
            latent,
            straight_from_obs={k: obs[k] for k in self.model.pass_obs_keys_to_actor},
        )
        # The last gamma is the deployed policy, as in native get_actions().
        probabilities = distribution.probs[..., -1, :]
        # Native clipping gives even masked actions a small probability floor.
        # Apply the authoritative legality mask again at deployment, identically
        # for the unchanged baseline and both trained conditions.
        legal = ~obs["illegal_actions"].bool()
        probabilities = torch.where(legal, probabilities, 0.0)
        total = probabilities.sum(-1, keepdim=True)
        if bool((total <= 0).any()):
            raise ValueError("Native deployment state has no legal probability mass")
        return probabilities / total, latent, hidden


def load_kakuna(
    checkpoint: str | Path,
    *,
    device: str = "cpu",
    verify_original: bool = True,
    training: bool = False,
) -> KakunaRuntime:
    """Build native modules directly, avoiding placeholder subprocesses and downloads."""
    import gin
    import gymnasium as gym
    import numpy as np
    import amago.cli_utils
    from amago.agent import MultiTaskAgent
    from amago.nets.traj_encoders import TformerTrajEncoder
    from amago.nets.transformer import SelfAttention
    from metamon.rl.metamon_to_amago import (
        MetamonPerceiverTstepEncoder,
        make_placeholder_env,
    )
    from metamon.rl.pretrained import Kakuna

    path = Path(checkpoint)
    digest = sha256(path)
    if verify_original and digest != KAKUNA_34_SHA256:
        raise ValueError("Expected the protected, published Kakuna checkpoint 34")
    gin.clear_config()
    pretrained = Kakuna()
    amago.cli_utils.use_config(
        pretrained.base_config,
        [pretrained.model_gin_config_path, pretrained.train_gin_config_path],
        finalize=False,
    )

    class NativeWindowAttention(WindowAttention, SelfAttention):
        pass

    gin.bind_parameter("traj_encoders.TformerTrajEncoder.attention_type", NativeWindowAttention)
    # Ordinary finetuning retains the native actor/critic objective and architecture.
    # Both arms receive identical primary losses; auxiliary heads are a separate wrapper.
    if training:
        gin.bind_parameter("agent.MultiTaskAgent.online_coeff", 0.0)
        gin.bind_parameter("agent.MultiTaskAgent.offline_coeff", 1.0)
    environment = make_placeholder_env(pretrained.observation_space, pretrained.action_space)
    try:
        model = MultiTaskAgent(
            obs_space=environment.observation_space,
            rl2_space=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32),
            action_space=environment.action_space,
            max_seq_len=128,
            tstep_encoder_type=MetamonPerceiverTstepEncoder,
            traj_encoder_type=TformerTrajEncoder,
        )
    finally:
        environment.close()
    weights = torch.load(path, map_location="cpu", weights_only=True)
    pretrained._validate_checkpoint(weights, model.state_dict())
    model.load_state_dict(weights, strict=True)
    model.on_checkpoint_loaded(is_resume=False)
    model.to(device)
    model.gammas = model.gammas.to(device)
    model.eval()

    def actor_mask(batch, _loss, mask):
        return mask & ~batch.obs["missing_action_mask"][:, :-1, :, None]

    def critic_mask(batch, _loss, mask):
        return mask & ~batch.obs["missing_action_mask"][:, :-1, :, None, None]

    model.edit_actor_mask = actor_mask
    model.edit_critic_mask = critic_mask
    return KakunaRuntime(
        model,
        pretrained.observation_space,
        pretrained.action_space,
        pretrained.reward_function,
        digest,
    )
