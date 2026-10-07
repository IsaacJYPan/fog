from pathlib import Path
import os
import sys
import venv

import pytest

from fog.pretrained.process import DecisionProcess
from fog.pretrained.runtime import interpreter_path


def test_player_boundary_rejects_nested_truth_before_starting_process(tmp_path):
    worker = DecisionProcess(["should-not-start"], cwd=tmp_path, log=tmp_path / "worker.log")
    with pytest.raises(ValueError, match="Complete-team labels"):
        worker.request({"op": "act", "nested": [{"labels": [1, 2, 3]}]})
    assert worker.process is None


def test_timeout_kills_worker_and_restart_does_not_receive_old_responses(tmp_path):
    worker_source = tmp_path / "worker.py"
    worker_source.write_text("""import json,sys,time
print(json.dumps({'status':'ready'}),flush=True)
for line in sys.stdin:
    message=json.loads(line)
    if message.get('slow'): time.sleep(2)
    print(json.dumps({'id':message['id'],'choice':'move 1'}),flush=True)
""")
    worker = DecisionProcess(
        [sys.executable, str(worker_source)],
        cwd=Path.cwd(),
        log=tmp_path / "worker.log",
        deadline=0.05,
    )
    try:
        worker.start()
        process = worker.process
        with pytest.raises(TimeoutError, match="deadline"):
            worker.request({"op": "act", "slow": True})
        assert process.poll() is not None
        response = worker.request({"op": "act"}, timeout=2)
        assert response["choice"] == "move 1"
        assert response["id"] == 2
    finally:
        worker.close(force=True)


def test_virtual_environment_interpreter_keeps_its_prefix(tmp_path):
    environment = tmp_path / "isolated"
    venv.EnvBuilder(with_pip=False, symlinks=os.name != "nt").create(environment)
    executable = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    code = "import json,sys; print(json.dumps({'status':'ready','prefix':sys.prefix}),flush=True); sys.stdin.read()"
    worker = DecisionProcess(
        [interpreter_path(executable), "-c", code],
        cwd=tmp_path,
        log=tmp_path / "venv.log",
    )
    try:
        ready = worker.start()
        assert Path(ready["prefix"]).samefile(environment)
    finally:
        worker.close(force=True)
