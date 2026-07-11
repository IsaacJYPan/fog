"""Shared pytest fixtures. Import-light so the suite runs without a Showdown server or torch CUDA.

The golden-battle integration test does not live here: it builds its scripted battle via
tests/golden_fixture.py and asserts byte-identical tensors in tests/test_golden_battle.py.
"""
from __future__ import annotations
