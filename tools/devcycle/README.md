# DSV4 TP4 B1 fail-fast dev cycle

Tool: `tools/devcycle.sh` (run from the sparkpipe checkout on this MacBook).
All ssh is key-based; all four TP4 ranks (spark4-spark7) are reachable.

## The loop (one candidate = a few commands, compact output)

```text
1. pick a candidate from CANDIDATES below (fail-fast: small, reversible,
   isolated-PoC-first when the kernel is new)
2. edit source locally -> run the fast source-contract tests (seconds)
3. build on a spark (nvcc sm_121a) -> model_driver.so
4. cp model_driver.so tools/devcycle/drivers/<NAME>/model_driver.so
5. ./tools/devcycle.sh spot <NAME> lean 3        # alternating pairs, ~10 min
6. read the VERDICT line:
   - ACCEPT + exact 3/3 + delta > ~+0.5%  -> accuracy gates + full tests + PR
   - REJECT / WRONG-TOKENS / delta < noise -> drop it, next candidate
```

## Why this is fast

- Spot test = build (once) + 3 alternating O128 pairs with driver swaps.
  No CI, no accuracy suite, no docs, until the candidate proves a repeatable
  exact-output speedup.
- The pack (38 GB) is symlinked, configs are templates with the runtime
  root sed'd in, and the driver swap is one 6.4 MB scp per rank.
- Every run is hash-checked against the pinned O128 stream; a wrong-token
  run fails the gate automatically (no silent quality drift).

## Pinned identities (do not change without a new receipt)

| What | Value |
| --- | --- |
| Control runtime | /tmp/dsv4-integrated-lean-3d962820-runtime (spark4-7) |
| Control driver sha256 | 3d962820608fbad251aa50b7650dba2ab4b1d19ec378251c0e0ee36922e7fce4 |
| O24 gate batch | tools/devcycle/batches/o24_batch.json |
| O24 token hash | 6f2cfd844a2c296feaf9dd05a04d7888b87c906dcfee947d5cc6de28f541e538 |
| O128 batch | tools/devcycle/batches/o128_batch.json |
| O128 token hash | a9385d0b296ca083e577e715d2f6335067691dce0e0dd5ab1394a102a3d3631f |
| Control baseline | 40.46 tok/s mean (lean integrated, 2026-08-15) |
| Measurement noise | ~+-0.05% same-driver; ~+-0.5% run-to-run over 3 pairs |

## Candidate list (fail-fast order)

Accepted candidates must: emit the exact O24+O128 token streams, show a
repeatable positive E2E delta, and not change the precision contract
(BF16 spine, FP8 non-expert linears, MXFP4 experts, BF16 KV).

1. **16-byte vectorized GEMV weight loads** (handoff's next experiment).
   The B1 dense W13/projection GEMVs load FP8/BF16 weights at byte/2-byte
   granularity; vectorizing to 16-byte loads improves DRAM efficiency on
   the fixed stream. Isolated exact PoC first, then spot test.

2. **De-alias ffn_accum_bf16** (WAR hazard: projection-shard pack and MoE
   output share one buffer). A second scratch buffer removes a
   serialization point; ~8 KB cost. Spot-testable with one-line change.

3. **Fold the MoE pair-reduce into the HcPost epilogue** (accumulate into
   the residual directly, skip the standalone read-modify-write pass).
   Needs an isolated bitwise PoC first (arithmetic reordering risk).

4. **L2-pin hot fixed tensors** (norms, HC fn/scale/base, FP8 scale planes)
   so the small hot part of the 10.5 GB fixed stream stops re-reading DRAM.
   Bigger change; only after 1-3 are exhausted.

Rejected before (do not re-attempt without new evidence): Query RMS+RoPE
fusion, all-post stack, projection-precollective overlap, cooperative Hc
finalize, persistent expert schedule, persistent dense/projection bundle,
weight read-ahead queue. See DSV4_TP4_B1_HANDOFF.md "Rejected and
quarantined work".

Quarantined: rolling collective Program (needs exact B1024 validation
first; see the handoff).

## What a spot-test pass unlocks next

After ACCEPT: exact O24 gate (already part of spot), then the full flow —
GA validator, source-contract suite, docs update, PR via
tools/sparkpipe_github_pat.sh (never plain gh/git push), merged-main
rebuild + 3-run requalification.

## Cleanup

```sh
./tools/devcycle.sh stop <NAME>     # stop a runtime's residentds
```

Receipts from spot runs live on this MacBook under /tmp/devcycle-spot-*.json;
copy winners into qualification/ when they graduate.
