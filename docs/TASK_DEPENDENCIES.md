# TASK LIST WITH DEPENDENCIES — TO 90%
# Updated: 2026-08-25T05:59 UTC

## TIER 1: NO DEPENDENCIES (start immediately)
- [T1-a] qwen36: Fix RANS validation hole (module.c:441 skips payload/scale checks) → pccore
- [T1-b] dsv4-pro: Merge b07b798 residency crash fix (exists upstream, not in HEAD) → dsv4pro
- [T1-c] K3: Provision bin/lib/packs on ranks {1,8,9,a,b,d,e,f} (~25GB each) → k3
- [T1-d] GLM52: Serialize ForgetAllSpeculation (lock-corruption path: memset zeroes held lock word) → glm52
- [T1-e] All drivers: Fix 11 include-path test regressions (-Iinclude missing from host_cuda_compiler.py) → hwiface
- [T1-f] qwen36: Fix serving adapter init returning BUSY instead of SCHEMA_ERROR on stale config → glm52

## TIER 2: DEPENDS ON TIER 1
- [T2-a] dsv4-pro: Wire MTP pack-load C guard (needs entry-count synthesis fix: enumerate full kind space via ExpectedLayerBits) → dsv4pro | depends: T1-b
- [T2-b] K3: Regenerate tp16 rank packs failing loader VALIDATION_FAILED → k3 | depends: T1-c
- [T2-c] GLM52: Stamp SYNC_DELIVERY capability flag (needs driver-side push-vs-sync evidence) → glm52 | depends: T1-d
- [T2-d] All: Implement adapter_common thread-safety fixes F-A1 through F-A6 (atomic orphan counter, DestroyReady latch rule, delivery-school enforcement) → hwiface | depends: T1-e

## TIER 3: DEPENDS ON TIER 2
- [T3-a] K3: Run TP16 boot sequence (tmp/k3_tp16_rollout.sh boot && status) → k3 | depends: T2-b
- [T3-b] dsv4-pro: Run GPU ladder S1→S4 for rank-local MoE execution → dsv4pro | depends: T2-a
- [T3-c] AMD: hipcc gfx950 compile + C2/C3 oracle runs on MI350P hardware → AMD dev | depends: hardware access

## TIER 4: PERFORMANCE (parallel with Tiers 2-3)
- [T4-a] qwen36: Validate multi-row prefill attention numerics past M=8 (tap-stability blocker) → pccore
- [T4-b] qwen36: CUDA-graph the spec path (reduce ~76ms/frame launch overhead) → pccore
- [T4-c] qwen38: Investigate ctx2048 decode REGRESSION (status=17; suspect DFLASH2_WINDOW=2048 boundary vs prompt length) → max
- [T4-d] dsv4-flash: Fix B4/B16 TP graph capture (widths ≠{1,8,1024} fail tp_graph_projection_island) → dsv4flash
- [T4-e] All: Reduce daemon memory footprint (+88% RSS, +3.6GB GPU regression from new binaries) → hwiface
