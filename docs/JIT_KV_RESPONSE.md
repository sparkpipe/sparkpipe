# JIT-KV: the kimi analysis as the build contract (2026-08-30)

THE VERDICT ACCEPTED: the bet is good — queue-driven admission +
whole-lane restore + deadline prefetch-from-enqueue + hard active-set
control makes ~2TB NVMe practically VRAM for queued work (1-2 orders
of magnitude slack). The gap is WIRING (~15%) + named safety bugs,
not physics. This file turns kimi's conditions into the gated build
order; the audit-readiness claim for the NEXT kimi pass includes
every item below.

## The contract (each = a merge-gated item)

C1 ACTIVE-SET NEVER OVERCOMMITS: admission's exact per-lane demand +
overflow checks is THE enforcement; no exception path under load;
backpressure = queue. (Existing dsv4 path is the pattern.)
C2 DISPATCH GATES ON RESTORE COMPLETE — not a hint. REVERSES the
PROPOSAL_KV_SEAM narrowing (WillBeResidentBy lookahead-only): for
queue-driven serving the deadline scheduler works BACKWARD from
dispatch order; a pre-restore dispatch is the cliff. The nvme_tier's
deadline-ordered lookahead becomes the engine.
C3 AGGREGATE BANDWIDTH ACCOUNTING: N restores share ~5-7GB/s; the
scheduler admits by (sum queued restore bytes / bandwidth) <= slack.
MEASURE the drive's real sustained number first — the one number
worth measuring before any tuning.
C4 ASYNC PARK: write-out NEVER blocks decode.
C5 PARK POLICY = reuse value (shared prefixes, multi-turn), not
aggression — endurance is a line item (0.1-0.3 DWPD sane; 10x if
careless).

## The named bugs (disqualifying if a tenant touches it; fix FIRST)

B1 WRITE-BACK WEDGE: eviction retries IO-error forever, block becomes
unevictable, admission wedges (cache/kv_cache.c:1141-1157). ENOSPC
must DEGRADE (drop + recompute) not stall.
B2 GLM5_NEXT ARENA GEOMETRY: pool layout contradicts geometry — an
OOB DMA the moment lanes wire up. Fix or delete before wiring.
B3 TIER CHECKSUMS: no integrity on tier boundaries; nvme_tier keys on
a bare 64-bit hash = silent cross-tenant KV aliasing. Per-slot
CRC/SHA-256 verified on restore (the prefix-cache digest precedent).
B4 BACKING-STORE HYGIENE: slots mode 0644 in predictable /tmp paths
= tenant KV world-readable. 0600 + namespaced paths.

## The wiring (the 85%)

W1 Consume spark_kv_backing from the module/adapter path (JIT_KV_
DESIGN steps 2-5 — the pager exists, unit-tested, zero consumers).
W2 Wire nvme_tier's lookahead to C2's gate.
W3 COLLAPSE the three parallel spill mechanisms (kv_page_store,
kv_backing, nvme_tier) to one + the Mooncake client boundary.
W4 The glm52/glm5_next static-identity-page-table bypass resolves
when lanes go through the page directory (W1's consequence, not a
separate hack).

SEQUENCE: B1-B4 (safety, small) -> C3's measurement -> W1+W2+C2 (the
vertical slice: one family, dsv4, end-to-end restore-gated) -> C1/C4/
C5 hardening -> W3 collapse -> family rollout. NO step ships without
its contract item's gate.
