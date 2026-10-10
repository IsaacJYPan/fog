from concurrent.futures import ThreadPoolExecutor
import json

from fog.scripts.kaggle_results import write_json


def test_concurrent_package_handoffs_leave_one_complete_pointer(tmp_path):
    path = tmp_path / "latest.json"
    payloads = [{"experiment": f"run-{index}", "files": list(range(1000))} for index in range(24)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda row: write_json(path, row), payloads))
    assert json.loads(path.read_text()) in payloads
    assert list(tmp_path.iterdir()) == [path]
