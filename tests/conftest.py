"""Shared pytest fixtures. Import-light so the suite runs without a Showdown server or torch CUDA."""
from __future__ import annotations

import pytest


@pytest.fixture
def golden_battle_fixture():
    """A scripted 10-turn battle with known reveals (spec §2 tests, §10 golden-battle integration).

    STUB: the fixture content (protocol log + expected tensors) is authored once the parser exists. The
    integration test asserting byte-identical tensors across refactors reads from here.
    """
    pytest.skip("Golden-battle fixture not authored yet (needs parser + encoders — §2/§10).")
