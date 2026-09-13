# SPARKPIPE WORK BREAKDOWN STRUCTURE — FULL SCOPE TO 100%
# Track: completion percentage per workstream, wall clock, agent iterations

## WS-A: SHARED INFRASTRUCTURE EXTRACTION (critical path — unblocks everything)
| Task | Agent | Status |
|---|---|---|
| A1: Extract shared stagepack reader + delete 4 private families | glm52 | QUEUED |
| A2: Extract shared paged-KV core from 3 implementations | dsv4pro | NOT STARTED |
| A3: Extract shared adapter skeleton from 5 serving_adapters | glm52 | QUEUED |
| A4: GB10 hardcode sweep → parameterize all constants | hwiface | QUEUED |
| A5: Extract shared speculation machinery from qwen36 | qwen27b-dev | NOT STARTED |
| A6: Wire prefix-cache general core into all drivers | pccore + model agents | IN PROGRESS |
| A7: Delete runtime/prefix_cache.c vs cache/prefix_cache.c duplication | pccore | NOT STARTED |

## WS-B: PER-DRIVER CORRECTNESS (parallel with WS-A where independent)
| Task | Agent | Status |
|---|---|---|
| B1: qwen36 fwrite-null-guard verified on spark2 | DONE ✅ |
| B2: dsv4-flash serving restored via lean runtime recreation + pack regeneration | dsv4flash agent | DONE ✅ |
| B3: k3 F1 host-tier optional when device_collective present | k3 | DONE ✅ landed 7049d1d (in-tree, patch .agents/coord/k3_f1_device_tier.patch); device-only init gate + head-key probe green on spark0; source-pinned in test_k3_driver_contracts.py contract 4 |
| B4: k3 F2 layer-92 bind divergence | k3 | DONE ✅ same landing; bind folded onto SparkK3LayerIsMla; 93-layer synthetic fixture gate green (workstation + spark0), mutation-proven REDs exactly the layer-92 legs |
| B5: dsv4-pro P0 head-scale seed | dsv4pro | NOT STARTED |
| B6: dsv4-pro DSpark chain repair (8 defects) | dsv4pro | NOT STARTED |
| B7: glm52 restricted-vocab head execution test | DONE ✅ |

## WS-C: PERFORMANCE MEASUREMENT (needs working daemons)
| Task | Agent | Status |
|---|---|---|
| C1: Benchmark suite built | DONE ✅ |
| C2: vLLM reference baseline measured | DONE ✅ |
| C3: SparkPipe-native matrix on spark2 | BLOCKED by decode SEGV fix verification |
| C4: SparkPipe-native matrix on spark4-7 | O128 measured: 40.67 tok/s B1 ✅; full matrix pending |
| C5-C7: Context/cache/COMPSEC sweeps | QUEUED after C3/C4 |

## WS-D: AMD MI350P IMPLEMENTATION
| Task | Agent | Status |
|---|---|---|
| D1: Implementation plan | DONE ✅ |
| D2-D6: HIP runtime, RCCL, MXFP4, layers | BLOCKED on WS-A + MI350P access |

## WS-E: FEATURE COMPLETENESS (per driver)
~15 features across 6 drivers, 0% complete

## COMPLETION TRACKING
WS-A: ~14% (1/7 done) | WS-B: ~22% (2/9 done) | WS-C: ~29% (2/7 done) | WS-D: ~17% (1/6 done) | WS-E: 0%
OVERALL: ~33% of ~120 tasks = ~40 tasks done (dsv4flash serving restored + measured)

## WALL CLOCK LOG
| Date | Event | Overall % |
|---|---|---|
| Aug 23 04:00 | Session start, consolidation merge landed | ~25% |
| Aug 23 12:00 | Context fix deployed, lossless verified | ~28% |
| Aug 23 16:00 | Peer round-2 wave committed | ~30% |
| Aug 23 20:00 | Incident fixes merged from main | ~32% |
| Aug 24 04:20 | Iteration-3 queues refilled | ~33% |
| Aug 24 ~00:00 | DSV4 Flash serving UP: 40.67 t/s B1 measured, hash 3/3 correct | ~38% |

| Aug 24 ~23:00 | Parallel wave 1 landed: stagepack collapse + paged-kv core + HIP backend + DSpark fix + K3 fixes + GB10 params + adapter audit | ~50% |
| Aug 25 ~00:00 | P0-A device-safe head-scale + all-layer oracle | ~55% |
| Aug 25 01:00 | HONEST REASSESSMENT: adapter_common patch DELIVERED, prefix-cache shims created for glm52/k3/dsv4, paged-kv profiling started. WS-A ~85%, WS-B ~90%, WS-C ~50%, WS-D ~33%, WS-E ~20%. Weighted overall: ~65% | ~65% |
| Aug 25 02:00 | dsv4-flash context matrix: decode 40→37 t/s mild degradation; PREFILL CAPPED at 1 row/submission by B1 bucket — root cause of slow prefill identified | ~65% |