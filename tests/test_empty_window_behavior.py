"""test_empty_window_behavior.py — R1: windows with no events return zeros/rates=0, not errors."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import market_regime as mr
import aggressive_flow as af
import passive_flow as pf

COLS = ["timestamp_ms", "event_type", "event_uid", "order_uid", "side", "price",
        "qty", "old_qty", "filled_qty", "maker_uid", "taker_uid",
        "execution_price", "execution_qty", "reason", "n_events_same_ms", "ordering_ambiguous"]


def _empty_ev():
    df = pd.DataFrame(columns=COLS)
    df["timestamp_ms"] = df["timestamp_ms"].astype("Int64")
    return df


def test_regime_empty_stream():
    f = mr.build_regime_features(_empty_ev(), None, np.array([1000, 2000]), windows={"1s": 1000})
    assert (f["order_event_count_1s"] == 0).all()
    assert (f["events_per_sec_1s"] == 0).all()
    assert (f["unique_order_uids_1s"] == 0).all()


def test_regime_gap_between_events():
    # events far in the past -> a later anchor's short window is empty
    rows = [{"timestamp_ms": 100, "event_type": "OrderPlaced", "order_uid": "a",
             "side": "buy", "price": 100, "qty": 1}]
    df = pd.DataFrame(rows)
    for c in COLS:
        if c not in df:
            df[c] = None
    df["timestamp_ms"] = df["timestamp_ms"].astype("Int64")
    f = mr.build_regime_features(df[COLS], None, np.array([100000]), windows={"1s": 1000})
    assert f.iloc[0]["order_event_count_1s"] == 0


def test_flow_modules_empty():
    a = af.windowed_aggressive_flow(af.build_taker_atoms([]), np.array([1000]), {"1s": 1000})
    assert a.iloc[0]["buy_taker_qty_1s"] == 0.0
    p = pf.windowed_passive_flow(pf.build_passive_atoms([]), np.array([1000]), {"1s": 1000})
    assert p.iloc[0]["bid_add_qty_1s"] == 0.0
