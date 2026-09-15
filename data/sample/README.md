# data/sample — Synthetic Schema Example

Real session data (3–18 GB raw JSON per session) is not included in this
repository. Use `download_raw.py` to fetch real data from the Kraken
Futures public API. **The files in this directory are synthetic and contain
no redistributed Kraken raw data.**

This directory contains a minimal synthetic example that exercises the full
pipeline without real exchange data:

```
example_orders.jsonl      — 17 synthetic order events (10 placed, 4 repriced, 2 cancelled, 1 update)
example_executions.jsonl  — 3 synthetic execution events
```

These files use a minimal synthetic subset of the Kraken Futures API v3 event schema
(the fields consumed by this research pipeline), enabling the pipeline to be demonstrated
and tested without downloading gigabytes of real data.

## Download Real Data

```bash
python download_raw.py \
    --symbol PF_XBTUSD \
    --start  2026-09-06T13:00:00Z \
    --end    2026-09-06T15:00:00Z \
    --suffix _S_LOW
```

Output: `data/raw/PF_XBTUSD_orders_S_LOW.jsonl` and
`data/raw/PF_XBTUSD_executions_S_LOW.jsonl`

## Data Redistribution Note

Kraken Futures historical data is accessed via a public unauthenticated
REST API. Redistribution of raw exchange data is subject to Kraken's terms
of service. Verify compliance before publishing downloaded data.
