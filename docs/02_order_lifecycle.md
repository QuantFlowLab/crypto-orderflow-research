# Order Lifecycle

## Event Types

Each element in the orders stream has the structure:

```json
{
  "uid": "<event-uuid>",
  "timestamp": 1789163998277,
  "event": {
    "OrderPlaced": {
      "order": {
        "uid": "<order-uuid>",
        "direction": "Buy",
        "limitPrice": "77111",
        "quantity": "1.0",
        "orderType": "Post",
        "reduceOnly": false,
        "filled": "0"
      }
    }
  }
}
```

| Event type | Meaning |
|---|---|
| `OrderPlaced` | New order entered the book |
| `OrderCancelled` | Order removed from book |
| `OrderUpdated` | Order modified (price or qty); qty = **replacement total**, not delta |
| `OrderRejected` | Order was rejected (e.g. Post-only would have crossed) |
| `Execution` | Trade matched; contains both `makerOrder` and `takerOrder` with UIDs |

## Quantity Semantics

`OrderUpdated.quantity` = **replacement quantity** (new total remaining), NOT a delta.  
97–98.5% of updates have unchanged quantity — the update typically reprices the order.

## Millisecond-Batch Semantics

**82% of events share a `timestamp_ms` with at least one other event** (up to 120 events/ms).
The API provides no `sequence_id`. Sub-millisecond event ordering is not recoverable.

The canonical treatment: **same-ms events form an unordered batch**. Book state is defined
*after* all events in the batch are applied. Within a batch, events are applied in
net-terminal order to ensure physical consistency:

```
OrderPlaced → OrderUpdated → Execution → OrderCancelled/OrderRejected
```

This resolves the same-ms place+cancel pattern (a common HFT lifecycle) correctly:
the order is placed, then cancelled within the same ms — net result: never active.
This is a **deterministic terminal-state resolution rule**, not a reconstruction of
true intra-millisecond matching-engine order.

## Lifecycle Completeness

With 35-minute warmup + 5-minute target window:

| Metric | BTC | ETH |
|---|---|---|
| Complete lifecycle (placed→closed) | 96.9% | 96.3% |
| Left-truncated (placed before warmup start) | 0.7% | 1.1% |
| Execution linkage (maker UID found in orders) | 100.0% | 99.3% |

The residual left-truncated fraction decreases with longer warmup and is quantified
per-session (see `reports/SESSION_QA_SUMMARY.txt`).

## Warmup Convergence

Unknown-origin orders (first event is cancel/update, not place) as function of warmup depth:

| Warmup | BTC unknown% | ETH unknown% |
|---|---|---|
| 0 min | 3.0% | 3.9% |
| 5 min | 1.7% | 2.9% |
| 30 min | 1.7% | 2.9% |

ETH does not reach <2% within 30 minutes — the residual is consistent with orders
placed before the tested warmup window.
