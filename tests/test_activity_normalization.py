"""test_activity_normalization.py — R1 Part A: relative context must be causal and emit
explicit not_ready/NaN when trailing history is insufficient (never silently shorten)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import market_regime as mr

COLS = ["timestamp_ms", "event_type", "event_uid", "order_uid", "side", "price",
        "qty", "old_qty", "filled_qty", "maker_uid", "taker_uid",
        "execution_price", "execution_qty", "reason", "n_events_same_ms", "ordering_ambiguous"]


def _events_on_grid(anchors, per_anchor):
    """Place `per_anchor[k]` events just before each anchor so base activity is controllable."""
    rows = []
    for k, a in enumerate(anchors):
        for j in range(per_anchor[k]):
            rows.append({"timestamp_ms": a - 1, "event_type": "OrderPlaced",
                         "order_uid": f"o{a}_{j}", "side": "buy", "price": 100, "qty": 1})
    df = pd.DataFrame(rows)
    for c in COLS:
        if c not in df:
            df[c] = None
    df["timestamp_ms"] = df["timestamp_ms"].astype("Int64")
    return df[COLS]


def test_not_ready_when_history_insufficient():
    # context 5m needs 300 x 1s anchors; only 10 anchors -> relative context all NaN
    anchors = np.arange(1000, 11000, 1000)
    df = _events_on_grid(anchors, [5] * len(anchors))
    f = mr.build_regime_features(df, None, anchors, windows={"1s": 1000},
                                 context_windows={"5m": 300_000}, regime_context="5m")
    assert f["activity_rolling_percentile_5m"].isna().all()
    assert (f["regime_state"] == mr.NOT_READY).all()


def test_percentile_and_median_when_history_ready():
    # 40 anchors @1s, context window 10s (needs 10 samples). Steady 5 events, then a burst.
    anchors = np.arange(1000, 41000, 1000)
    per = [5] * len(anchors)
    per[-1] = 50  # burst on last anchor
    df = _events_on_grid(anchors, per)
    f = mr.build_regime_features(df, None, anchors, windows={"1s": 1000},
                                 context_windows={"10s": 10_000}, regime_context="10s")
    last = f.iloc[-1]
    assert not np.isnan(last["activity_rolling_percentile_10s"])
    assert last["activity_rolling_percentile_10s"] == 1.0     # burst is the max in window
    assert last["activity_vs_trailing_median_10s"] > 1.0      # above trailing median
    assert last["activity_robust_z_10s"] > 0                  # positive robust z
    assert last["regime_state"] == "EXTREME"                  # >=0.98 percentile band


def test_regime_bands_are_frozen_methodological():
    # band edges must be the documented percentile cuts, independent of data
    assert mr.REGIME_BANDS == [("QUIET", 0.0), ("NORMAL", 0.20), ("ACTIVE", 0.70),
                               ("BURST", 0.90), ("EXTREME", 0.98)]
    assert mr._regime_label(0.0) == "QUIET"
    assert mr._regime_label(0.5) == "NORMAL"
    assert mr._regime_label(0.80) == "ACTIVE"
    assert mr._regime_label(0.95) == "BURST"
    assert mr._regime_label(0.99) == "EXTREME"
    assert mr._regime_label(float("nan")) == mr.NOT_READY
