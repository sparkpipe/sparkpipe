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
| B2: dsv4-flash IPC break fix | dsv4flash | NOT STARTED |
| B3: k3 F1 host-tier optional when device_collective present | k3 | NOT STARTED |
| B4: k3 F2 layer-92 bind divergence | k3 | NOT STARTED |
| B5: dsv4-pro P0 head-scale seed | dsv4pro | NOT STARTED |
| B6: dsv4-pro DSpark chain repair (8 defects) | dsv4pro | NOT STARTED |
| B7: glm52 restricted-vocab head execution test | DONE ✅ |

## WS-C: PERFORMANCE MEASUREMENT (needs working daemons)
| Task | Agent | Status |
|---|---|---|
| C1: Benchmark suite built | DONE ✅ |
| C2: vLLM reference baseline measured | DONE ✅ |
| C3: SparkPipe-native matrix on spark2 | BLOCKED by decode SEGV fix verification |
| C4: SparkPipe-native matrix on spark4-7 | BLOCKED by serving bring-up |
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
OVERALL: ~29% of ~120 tasks = ~35 tasks done

## WALL CLOCK LOG
| Date | Event | Overall % |
|---|---|---|
| Aug 23 04:00 | Session start, consolidation merge landed | ~25% |
| Aug 23 12:00 | Context fix deployed, lossless verified | ~28% |
| Aug 23 16:00 | Peer round-2 wave committed | ~30% |
| Aug 23 20:00 | Incident fixes merged from main | ~32% |
| Aug 24 04:20 | Iteration-3 queues refilled | ~33% |