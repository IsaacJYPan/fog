"""PPO restarts retain completed measurements, including when the header grows."""
from fog.training.dashboard import TrainingDashboard


def test_resumed_dashboard_preserves_previous_rows(tmp_path):
    first = TrainingDashboard(str(tmp_path))
    first.log(8000, {"loss": 1.25})
    before = first.csv.read_text()
    resumed = TrainingDashboard(str(tmp_path), resume=True)
    resumed.log(16000, {"loss": .75})
    assert resumed.csv.read_text().startswith(before)
    resumed.log(24000, {"loss": .5, "new_metric": 2.})
    assert resumed.csv.read_text().splitlines() == [
        "step,loss,new_metric", "8000,1.25000,0.00000",
        "16000,0.75000,0.00000", "24000,0.50000,2.00000",
    ]
    if first.tb:
        first.tb.close()
    if resumed.tb:
        resumed.tb.close()


def test_fresh_dashboard_still_starts_a_new_csv(tmp_path):
    first = TrainingDashboard(str(tmp_path))
    first.log(8000, {"loss": 1.25})
    fresh = TrainingDashboard(str(tmp_path))
    fresh.log(0, {"loss": 2.})
    assert fresh.csv.read_text().splitlines() == ["step,loss", "0,2.00000"]
    if first.tb:
        first.tb.close()
    if fresh.tb:
        fresh.tb.close()
