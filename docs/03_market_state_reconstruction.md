# Market State Reconstruction

## The `OrderBook` Class

`src/book_state.py` implements an **order-level book-state reconstruction** that tracks
individual order UIDs under millisecond-batch semantics.

> It should not be interpreted as an exact matching-engine replay: intra-millisecond
> event sequence is not observable from the public REST API (no `sequence_id` field).

```python
book = OrderBook()
book.replay(event_df)       # apply events in batch-priority order
snap = book.bbo_snapshot()  # {best_bid, best_ask, spread, mid, n_active}
depth = book.depth_bps(10)  # {depth_bid, depth_ask, obi} within 10 bps of mid
```

Key QA counters (accumulated during replay):

| Counter | Meaning | Expected |
|---|---|---|
| `crossed_book` | Batches where best_bid > best_ask | 0 after fix |
| `locked_book` | Batches where best_bid == best_ask | Informational only |
| `negative_qty` | Orders with qty < 0 | 0 |
| `duplicate_active_uid` | Duplicate OrderPlaced for same UID | 0 |

## The Ghost-Order Bug (Pre-Fix)

**Finding during QA1:** the book was systematically crossed (ETH: 206,773 crossed batches,
spread = −24 ticks) due to a specific event-ordering failure.

**Root cause:** the API delivers same-ms events in file order. For many resting limit orders,
an `OrderCancelled` event arrived before its `OrderPlaced` in the same millisecond.
Naive replay processed the cancel (no-op, UID not yet known) then the place (added to book,
never removed) — creating ghost orders that accumulated and crossed the book.

**Fix:** within each ms-batch, apply events in net-terminal order:
`Placed → Updated → Execution → Cancelled/Rejected`.

This makes a same-ms place+cancel resolve to "removed" (the terminal lifecycle state
implied by the observed events). This is a **deterministic terminal-state resolution rule**;
it is not a claim about true intra-millisecond matching-engine order.

**Post-fix results (BTC):**
- Ghost orders: 8,998 → 0
- Crossed batches: 8,998 → 0
- `crossed_book` counter: > 0 → 0 for all sessions

A regression test (`tests/test_same_ms_batch_invariance.py`) verifies that shuffling
events within any ms-batch produces identical post-batch book state.

## Performance

| Session | Events | Replay time |
|---|---|---|
| BTC LOW | 3.53M | ~8s |
| BTC HIGH | 8.42M | ~20s |
| ETH LOW | 1.97M | ~4s |

Vectorized implementation: numpy `lexsort` for batch ordering; lazy heaps replace O(N)
full-book scans for BBO tracking (O(log N) heap updates, O(1) top access before lazy
stale-entry cleanup).

## Near-BBO Completeness

After 1-hour warmup, near-BBO completeness is 100% for L1/L5/L10 and within 5/10/25 bps
in all six sessions. The 1–3% unknown-origin floor from warmup does not reach the active
near-market region.
