"""test_regime_causality.py — R1 Part A: strict causal-backward windows, no-future, rates.
Covers the partner's test_no_future_regime + test_window_boundaries requirements too."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import market_regime as mr

COLS = ["timestamp_ms", "event_type", "event_uid", "order_uid", "side", "price",
        "qty", "old_qty", "filled_qty", "maker_uid", "taker_uid",
        "execution_price", "execution_qty", "reason", "n_events_same_ms", "ordering_ambiguous"]


def ev_df(rows):
    df = pd.DataFrame(rows)
    for c in COLS:
        if c not in df:
            df[c] = None
    df["timestamp_ms"] = df["timestamp_ms"].astype("Int64")
    return df[COLS]


def placed(ts, uid, side, price, qty):
    return {"timestamp_ms": ts, "event_type": "OrderPlaced", "order_uid": uid,
            "side": side, "price": price, "qty": qty}


def test_future_events_not_counted():
    df = ev_df([placed(5000, "a", "buy", 100, 1)])
    f = mr.build_regime_features(df, None, np.array([4000]), windows={"1s": 1000})
    assert f.iloc[0]["order_event_count_1s"] == 0
    assert f.iloc[0]["placed_count_1s"] == 0
    f2 = mr.build_regime_features(df, None, np.array([5000]), windows={"1s": 1000})
    assert f2.iloc[0]["order_event_count_1s"] == 1


def test_window_boundary_inclusive_right_exclusive_left():
    # anchor=5000, W=1000 -> window (4000, 5000]; event at 5000 IN, at 4000 OUT
    df = ev_df([placed(4000, "a", "buy", 100, 1), placed(5000, "b", "buy", 100, 1)])
    f = mr.build_regime_features(df, None, np.array([5000]), windows={"1s": 1000})
    assert f.iloc[0]["order_event_count_1s"] == 1  # only ts=5000


def test_rates_definition():
    df = ev_df([placed(t, f"o{t}", "buy", 100, 1) for t in (100, 200, 300, 400, 500)])
    f = mr.build_regime_features(df, None, np.array([1000]), windows={"1s": 1000})
    # 5 events in 1s window -> 5 events/sec
    assert abs(f.iloc[0]["events_per_sec_1s"] - 5.0) < 1e-9


def test_unique_uids_and_levels_distinct():
    df = ev_df([placed(100, "a", "buy", 100, 1), placed(200, "a", "buy", 100, 1),
                placed(300, "b", "buy", 101, 1)])
    f = mr.build_regime_features(df, None, np.array([1000]), windows={"1s": 1000})
    assert f.iloc[0]["unique_order_uids_1s"] == 2   # a, b
    assert f.iloc[0]["levels_touched_1s"] == 2      # 100, 101


def test_taker_side_counts_from_atoms():
    import aggressive_flow as af
    ex = [{"timestamp": 500, "event": {"Execution": {"execution": {
        "takerOrder": {"uid": "t", "direction": "Sell"}, "makerOrder": {"uid": "m", "direction": "Buy"},
        "quantity": "2.0", "price": "100", "usdValue": "200"}}}}]
    atoms = af.build_taker_atoms(ex)
    df = ev_df([placed(500, "a", "buy", 100, 1)])
    f = mr.build_regime_features(df, atoms, np.array([1000]), windows={"1s": 1000})
    assert f.iloc[0]["sell_taker_qty_1s"] == 2.0
    assert f.iloc[0]["buy_taker_qty_1s"] == 0.0
