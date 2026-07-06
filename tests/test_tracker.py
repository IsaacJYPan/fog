"""Tests for encoding/tracker.BattleTracker — the event-derived encoder-fidelity fields."""
from __future__ import annotations

import numpy as np

from fog.encoding.tracker import BattleTracker


def test_last_used_move_and_switch_reset():
    t = BattleTracker()
    t.feed_turn(1, [
        ["", "switch", "p1a: Garchomp", "Garchomp", "100/100"],
        ["", "switch", "p2a: Corviknight", "Corviknight", "100/100"],
    ])
    t.feed_turn(2, [["", "move", "p1a: Garchomp", "Earthquake", "p2a: Corviknight"]])
    assert t.last_used_move["p1: Garchomp"] == "earthquake"
    # switching out resets last_used_move (Encore/choice-lock inference resets on re-entry)
    t.feed_turn(3, [["", "switch", "p1a: Kingambit", "Kingambit", "100/100"]])
    assert "p1: Garchomp" not in t.last_used_move


def test_fallen_count_locks_at_entry():
    t = BattleTracker()
    t.feed_turn(1, [["", "switch", "p1a: Kingambit", "Kingambit", "100/100"]])
    assert t.fallen_for("p1: Kingambit", "p1") == 0.0
    t.feed_turn(2, [
        ["", "faint", "p1a: Kingambit"],
        ["", "switch", "p1a: Garchomp", "Garchomp", "100/100"],
    ])
    # Garchomp entered AFTER one faint; a later faint must not change its locked count
    assert t.fallen_for("p1: Garchomp", "p1") == 1 / 5
    t.feed_turn(3, [["", "faint", "p2a: Corviknight"]])
    assert t.fallen_for("p1: Garchomp", "p1") == 1 / 5
    # re-entry re-locks: Kingambit fainted, so only my side count matters for my mons
    t.feed_turn(4, [["", "switch", "p1a: Dragapult", "Dragapult", "100/100"]])
    assert t.fallen_for("p1: Dragapult", "p1") == 1 / 5


def test_booster_and_volatile_counters():
    t = BattleTracker()
    t.feed_turn(1, [
        ["", "-enditem", "p1a: Great Tusk", "Booster Energy"],
        ["", "-start", "p2a: Gholdengo", "move: Taunt"],
    ])
    assert "p1: Great Tusk" in t.booster_consumed
    assert t.counter_remaining("p2: Gholdengo", "taunt", now=2) == (3 - 1) / 8
    t.feed_turn(2, [["", "-end", "p2a: Gholdengo", "move: Taunt"]])
    assert t.counter_remaining("p2: Gholdengo", "taunt", now=3) == 0.0
    # volatiles end on switch-out too
    t.feed_turn(3, [["", "-start", "p2a: Gholdengo", "Encore"]])
    t.feed_turn(4, [["", "switch", "p2a: Kingambit", "Kingambit", "100/100"],
                    ["", "switch", "p2a: Gholdengo", "Gholdengo", "100/100"]])
    # first switch replaces the previously-active (None tracked -> set on first switch); simulate:
    t2 = BattleTracker()
    t2.feed_turn(1, [["", "switch", "p2a: Gholdengo", "Gholdengo", "100/100"]])
    t2.feed_turn(2, [["", "-start", "p2a: Gholdengo", "Encore"]])
    t2.feed_turn(3, [["", "switch", "p2a: Kingambit", "Kingambit", "100/100"]])
    assert t2.counter_remaining("p2: Gholdengo", "encore", now=4) == 0.0


def test_future_sight_and_wish_delayed_block():
    t = BattleTracker(my_role="p1")
    # opponent's Slowking sets Future Sight (lands on MY side), my Alomomola uses Wish
    t.feed_turn(3, [
        ["", "-start", "p2a: Slowking", "move: Future Sight"],
        ["", "move", "p1a: Alomomola", "Wish", "p1a: Alomomola"],
    ])
    block = t.delayed_block(now=4)
    # my side (first 5): FS pending on ME (set by opponent), turns left (3+2-4)/3
    assert block[0] == 1.0 and np.isclose(block[1], 1 / 3)
    # my Wish resolves end of turn 4: pending with 0 turns left at now=4
    assert block[2] == 1.0 and np.isclose(block[3], 0.0) and block[4] == 0.5
    # opponent side: nothing pending
    assert (block[5:] == 0).all()
    # after the FS resolves the pending flag clears
    t.feed_turn(5, [["", "-end", "p1a: Garchomp", "move: Future Sight"]])
    assert t.delayed_block(now=6)[0] == 0.0


def test_hp_seeding_and_decision_recording():
    t = BattleTracker()
    t.feed_turn(1, [
        ["", "switch", "p1a: Garchomp", "Garchomp", "100/100"],
        ["", "-damage", "p1a: Garchomp", "63/100"],
    ])
    assert np.isclose(t.hp["p1: Garchomp"], 0.63)
    snap = np.arange(30, dtype=np.float32)
    t.record_decision(2, snap, action_idx=3)
    assert t.my_actions[2] == 3
    assert (t.snapshots[2] == snap).all()
    assert np.isclose(t.hp_at_turn_start[2]["p1: Garchomp"], 0.63)
