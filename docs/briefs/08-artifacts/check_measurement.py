"""Verify timing hooks preserve native update metrics, parameters and RNG on CPU."""
import importlib.util
import json
import os
from pathlib import Path
import runpy
import tempfile
import torch

root = Path(__file__).resolve().parents[3]
fixture = runpy.run_path(str(root / "tests/test_ppo_update.py"))
source = Path(__file__).with_name("measure.py")
with tempfile.TemporaryDirectory() as tmp:
    os.environ["FOG08_OUTPUT"] = tmp
    spec = importlib.util.spec_from_file_location("measure", source)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    from fog.training.ppo import ppo_update
    torch.set_num_threads(1)

    def run(timed):
        torch.manual_seed(61)
        agent = fixture["TinyPolicyAgent"]()
        opt = torch.optim.AdamW(agent.parameters(), lr=3e-4)
        cfg = {"ppo": {"minibatch_size": 32, "epochs": 2}}
        batch = fixture["make_ppo_batch"]()
        if timed:
            with m.Parts(agent, opt, "parts.json"):
                metrics = ppo_update(agent, opt, batch, cfg)
        else:
            metrics = ppo_update(agent, opt, batch, cfg)
        return metrics, agent.state_dict(), torch.random.get_rng_state()

    a, b = run(False), run(True)
    assert a[0] == b[0]
    assert all(torch.equal(a[1][key], b[1][key]) for key in a[1])
    assert torch.equal(a[2], b[2])
    parts = json.loads((Path(tmp) / "parts.json").read_text())
    assert parts["rows"] == 32
    assert abs(parts["partition_seconds"] - parts["total_seconds"]) < 1e-9
    result = {"label": "CPU engineering smoke; no GPU performance claim",
              "metrics_identical": True, "parameters_identical": True, "rng_identical": True,
              "timing_partition_complete": True}
    Path(__file__).with_name("timing-equivalence.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
