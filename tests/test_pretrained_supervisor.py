import ctypes
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import zipfile

import pytest

from fog.scripts import pretrained_kaggle_worker as worker


def bundle(tmp_path, *, corrupt=False, timeout=20):
    root = tmp_path / "inputs"
    root.mkdir()
    record = gzip.compress(b'{"submitted_teams": []}')
    manifest = {
        "stage": "train",
        "timeout_seconds": timeout,
        "files": {"inputs/game.json.gz": hashlib.sha256(record).hexdigest()},
    }
    with zipfile.ZipFile(root / "bundle.bin", "w") as archive:
        archive.writestr("fog/experiment.json", json.dumps(manifest))
        archive.writestr("fog/inputs/game.json.gz", b"changed" if corrupt else record)
    return root, manifest, record


def assert_archive(output):
    with zipfile.ZipFile(output.parent / "fog-experiment-output.zip") as archive:
        hashes = json.loads(archive.read("output_hashes.json"))
        assert set(hashes) == set(archive.namelist()) - {"output_hashes.json"}
        for name, digest in hashes.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
        return json.loads(archive.read("pipeline_state.json"))


def process_running(pid):
    if os.name == "nt":
        kernel = ctypes.windll.kernel32
        kernel.OpenProcess.restype = ctypes.c_void_p
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        code = ctypes.c_ulong()
        try:
            kernel.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code))
            return code.value == 259
        finally:
            kernel.CloseHandle(ctypes.c_void_p(handle))
    try:
        os.kill(pid, 0)
        # A killed orphan can briefly remain a Linux zombie awaiting reaping.
        proc = Path(f"/proc/{pid}/stat")
        return not proc.exists() or proc.read_text().split(") ")[1].split()[0] != "Z"
    except ProcessLookupError:
        return False


def test_opaque_input_keeps_gzip_bytes_and_checks_inventory(tmp_path):
    inputs, manifest, record = bundle(tmp_path)
    output = tmp_path / "results"
    output.mkdir()
    work = tmp_path / "work/fog"
    assert worker.prepare_inputs(inputs, work, output) == manifest
    assert (work / "inputs/game.json.gz").read_bytes() == record


def test_preparation_failure_still_exports_verified_partial_results(tmp_path):
    inputs, manifest, _ = bundle(tmp_path, corrupt=True)
    output = tmp_path / "job/results"
    output.mkdir(parents=True)
    (output / "saved_checkpoint.pt").write_bytes(b"existing periodic state")
    with pytest.raises(RuntimeError, match="subprocess exit code"):
        worker.run(
            input_root=inputs,
            work=tmp_path / "work/fog",
            output=output,
            timeout=20,
            export_reserve=4,
        )
    state = assert_archive(output)
    assert state["status"] == "failed"
    assert state["stage"] == "frozen input verification"
    assert json.loads((output / "experiment.json").read_text()) == manifest
    assert (output / "saved_checkpoint.pt").read_bytes() == b"existing periodic state"
    assert "Frozen file changed" in (output / "supervisor.log").read_text()


def test_watchdog_stops_preparation_and_its_descendants_and_exports(tmp_path):
    output = tmp_path / "job/results"
    output.mkdir(parents=True)
    script = tmp_path / "hang.py"
    script.write_text(
        "import json,subprocess,sys,time\n"
        "from pathlib import Path\n"
        "out=Path(sys.argv[1])\n"
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(300)'])\n"
        "(out/'child.json').write_text(json.dumps(child.pid))\n"
        "(out/'pipeline_progress.json').write_text(json.dumps({'stage':'input preparation'}))\n"
        "(out/'checkpoint.pt').write_bytes(b'periodic full state')\n"
        "time.sleep(300)\n"
    )
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="deadline exceeded"):
        worker.run(
            output=output,
            timeout=12,
            export_reserve=4,
            worker_command=[sys.executable, str(script), str(output)],
        )
    assert time.monotonic() - started < 12
    state = assert_archive(output)
    assert state["status"] == "failed"
    assert state["stage"] == "input preparation"
    assert (output / "checkpoint.pt").read_bytes() == b"periodic full state"
    assert not process_running(json.loads((output / "child.json").read_text()))


def test_successful_process_without_result_is_not_a_successful_experiment(tmp_path):
    output = tmp_path / "results"
    with pytest.raises(RuntimeError, match="without a pipeline result"):
        worker.run(
            output=output,
            timeout=20,
            export_reserve=4,
            worker_command=[sys.executable, "-c", "pass"],
        )
    assert assert_archive(output)["status"] == "failed"


def test_packaged_script_imports_verified_code_from_an_unrelated_directory(tmp_path):
    # This is launched as a file, outside the checkout, like /kaggle/src/script.py.
    # Changing cwd alone does not update that interpreter's sys.path[0].
    inputs, manifest, _ = bundle(tmp_path, timeout=30)
    manifest.update(device="cpu", cpu_preflight=True)
    sources = {
        "fog/__init__.py": "",
        "fog/pretrained/__init__.py": "",
        "fog/scripts/__init__.py": "",
        "fog/pretrained/runtime.py": (
            "import os\nfrom pathlib import Path\n"
            "def native_environment(root):\n"
            "    assert Path(__file__).resolve().is_relative_to(root)\n"
            "    return dict(os.environ,PYTHONPATH=str(root))\n"
        ),
        "fog/scripts/pretrained_setup.py": (
            "import argparse,json,sys\nfrom pathlib import Path\n"
            "p=argparse.ArgumentParser();p.add_argument('--stage');p.add_argument('--device');"
            "p.add_argument('--out',type=Path);a=p.parse_args()\n"
            "assert a.device=='cpu';a.out.mkdir(parents=True)\n"
            "(a.out/'runtimes.json').write_text(json.dumps({'native':sys.executable}))\n"
        ),
        "fog/scripts/pretrained_contract.py": (
            "import argparse,json\nfrom pathlib import Path\n"
            "p=argparse.ArgumentParser();p.add_argument('--out',type=Path);"
            "p.add_argument('--device');a,_=p.parse_known_args()\n"
            "assert a.device=='cpu';a.out.mkdir(parents=True)\n"
            "(a.out/'contract.json').write_text(json.dumps({'passed':True,'device':a.device}))\n"
        ),
        "fog/scripts/pretrained_train.py": (
            "import argparse,json\nfrom pathlib import Path\n"
            "p=argparse.ArgumentParser();p.add_argument('--out',type=Path);"
            "p.add_argument('--device');a,_=p.parse_known_args()\n"
            "assert a.device=='cpu';a.out.mkdir(parents=True)\n"
            "(a.out/'RESULTS.json').write_text(json.dumps({'complete':True}))\n"
            "(a.out/'native_final.pt').write_bytes(b'fixture checkpoint')\n"
        ),
    }
    manifest["files"] = {
        name: hashlib.sha256(code.encode()).hexdigest() for name, code in sources.items()
    }
    with zipfile.ZipFile(inputs / "bundle.bin", "w") as archive:
        for name, code in sources.items():
            archive.writestr("fog/" + name, code)
        archive.writestr("fog/experiment.json", json.dumps(manifest))
    script = tmp_path / "notebook/script.py"
    script.parent.mkdir()
    script.write_text(
        Path(worker.__file__)
        .read_text()
        .replace("EXPORT_RESERVE_SECONDS = 60", "EXPORT_RESERVE_SECONDS = 4")
    )
    output = tmp_path / "job/results"
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    worker.execute(
        [
            sys.executable,
            "-I",
            str(script),
            "--input-root",
            str(inputs),
            "--work",
            str(tmp_path / "work/fog"),
            "--output",
            str(output),
            "--timeout",
            "30",
        ],
        env,
        tmp_path / "launch.log",
        30,
    )
    assert assert_archive(output)["status"] == "complete"
    assert json.loads((output / "contract-trained/cpu/contract.json").read_text())["passed"]
