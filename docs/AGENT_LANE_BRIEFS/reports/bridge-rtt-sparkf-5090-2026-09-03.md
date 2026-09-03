# Bridge RTT: sparkf <-> RTX 5090 (10 GbE point-to-point)

Date: 2026-09-03. Link: sparkf enP7s7 10.10.250.1/30 <-> 5090 10.10.250.2. Farm server
(`farm_server.py`, DFRM + DFT3 on :7793) running on the 5090 with glm-5.3-flash-dflash2 loaded.

Probe: 200 iterations each, TCP_NODELAY, from sparkf (`/tmp/probe_rtt.py` +
`/tmp/draft_protocol.py`).

## Pure transport (connect + immediately-rejected bad-magic request)

p50 0.471 ms, p95 1.081 ms, p99 1.208 ms, max 1.476 ms.

## Full DFT3 proposal, persistent connection

5-token committed prefix, 4 zero tap rows (160 KiB payload), speculator_mask=DFLASH2,
time_budget_ms=5, max_depth=8, max_nodes=64.

- End-to-end RTT: p50 8.49 ms, p95 9.56 ms, p99 9.83 ms, max 297.5 ms (first request after
  server start = CUDA warmup; steady state is tight).
- Server-side generation (footer elapsed): p50 6.54 ms, p95 7.03 ms.
- Transport + Python server overhead over server compute: ~2 ms.
- Tree size: p50 5 nodes (budget-bound from a 5-token context).

## Reading

Transport is not the constraint: ~0.5 ms p50 / ~1.2 ms p99 even with a fresh connection per
request. The budget math is dominated by draft compute (bounded by the caller's time_budget_ms)
plus ~2 ms of stack overhead, which a C client + pinned rings should shrink, not grow.
The B1 fully-serial round trip at a 5 ms budget lands ~8.5 ms; at B8 the proposal overlaps
pipeline revisit slack.
