import torch
from torch import nn

from fog.pretrained.joint import JointKakuna


class SmallTrajectory(nn.Module):
    emb_dim = 8

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)

    def forward(self, sequence):
        return self.linear(sequence), None


class Native(nn.Module):
    def __init__(self):
        super().__init__()
        self.tstep_encoder = nn.Linear(3, 8)
        self.traj_encoder = SmallTrajectory()
        self.actor = nn.Linear(8, 3)

    @property
    def trainable_params(self):
        return self.parameters()

    def forward(self, batch, log_step=False):
        latent, _ = self.traj_encoder(self.tstep_encoder(batch))
        return self.actor(latent).square().mean()


def test_joint_auxiliaries_reach_shared_latent_without_using_unverified_prefixes():
    model = JointKakuna(Native(), belief_dim=16).eval()
    primary, latent = model.primary_with_latent(torch.randn(2, 3))
    reveals = torch.full((2, 6, 15), -1)
    truth = reveals.clone()
    verified = torch.zeros_like(truth, dtype=torch.bool)
    truth[:, 0, 7] = 1
    verified[:, 0, 7] = True
    first = model.auxiliary_losses(
        latent,
        reveals,
        truth,
        verified,
        torch.tensor([1.0, -1.0]),
        torch.ones(2, dtype=torch.bool),
        generator=torch.Generator().manual_seed(2),
    )
    # A spurious unverified label must not affect either the target or its teacher prefix.
    corrupt = truth.clone()
    corrupt[:, :, 8] = 999999
    second = model.auxiliary_losses(
        latent,
        reveals,
        corrupt,
        verified,
        torch.tensor([1.0, -1.0]),
        torch.ones(2, dtype=torch.bool),
        generator=torch.Generator().manual_seed(2),
    )
    for key in first:
        torch.testing.assert_close(first[key], second[key])
    sum(first.values()).backward()
    assert model.native.traj_encoder.linear.weight.grad.norm() > 0
    assert model.native.tstep_encoder.weight.grad.norm() > 0
    assert model.context_projection[1].weight.grad.norm() > 0
    assert model.outcome[-1].weight.grad.norm() > 0
    assert model.native.actor.weight.grad is None
    assert torch.isfinite(primary)


def test_auxiliary_head_changes_do_not_change_ordinary_actor_probabilities():
    model = JointKakuna(Native(), belief_dim=16).eval()
    observation = torch.randn(2, 3)
    latent, _ = model.native.traj_encoder(model.native.tstep_encoder(observation))
    before = model.native.actor(latent).softmax(-1)
    with torch.no_grad():
        for module in (model.context_projection, model.belief, model.outcome):
            for parameter in module.parameters():
                parameter.add_(torch.randn_like(parameter))
    latent, _ = model.native.traj_encoder(model.native.tstep_encoder(observation))
    after = model.native.actor(latent).softmax(-1)
    assert torch.equal(before, after)
