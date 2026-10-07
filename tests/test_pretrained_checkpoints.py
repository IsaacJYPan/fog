from types import SimpleNamespace

import pytest
import torch

from fog.pretrained.checkpoints import export_candidates, load_auxiliary
from fog.pretrained.kakuna import sha256


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.native = torch.nn.Linear(2, 1)
        self.outcome = torch.nn.Linear(2, 1)


def test_interrupted_arms_keep_common_update_anchors_and_matching_heads(tmp_path):
    model = Model()
    recipe = {"updates": 20, "condition": "joint"}
    early, complete = tmp_path / "early", tmp_path / "complete"
    for out, completed in ((early, 13), (complete, 20)):
        for update in range(1, completed + 1):
            with torch.no_grad():
                model.outcome.bias.fill_(update)
            export_candidates(model, out, update=update, identity="frozen", config=recipe)
        export_candidates(
            model, out, update=completed, identity="frozen", config=recipe, final=True
        )
        assert sha256(out / "native_final.pt") == sha256(out / f"native_{completed:06d}.pt")
    early_updates = {int(path.stem.split("_")[1]) for path in early.glob("native_[0-9]*.pt")}
    complete_updates = {int(path.stem.split("_")[1]) for path in complete.glob("native_[0-9]*.pt")}
    assert early_updates == {5, 10, 13}
    assert complete_updates == {5, 10, 15, 20}
    assert early_updates.intersection(complete_updates) == {5, 10}
    state = load_auxiliary(model, early / "auxiliary_000010.pt", early / "native_000010.pt")
    assert state["update"] == 10 and model.outcome.bias.item() == 10
    wrong = tmp_path / "wrong-native.pt"
    torch.save(SimpleNamespace(unrelated=True), wrong)
    with pytest.raises(ValueError, match="different native"):
        load_auxiliary(model, early / "auxiliary_000010.pt", wrong)
