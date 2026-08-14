# GLM52 Fast Stage Sweep 2026-07-01

This run used the `main` checkout on `spark2` after PR #47 and PR #48.

The stage sweep default no longer rebuilds the firmware package for every
bucket/stage attempt. It uses existing package artifacts and caches one CUDA
validator binary per active sequence count.

## Hardware / Checkout

```text
host=spark2
repo=/tmp/sparkpipe_glm52_live_main_20260630
main=6829271
model_dir=/home/spark2/models/hf/nvidia/GLM-5.2-NVFP4
nvcc=/usr/local/cuda/bin/nvcc
cuda_arch=sm_121a
```

## Accuracy Gate Context

The latest clean-main practical accuracy gate before this sweep had:

```text
dense_reference=passed
restricted_token_stability=passed token=1228
routed_numeric_repeatability=passed
routed_bit_stability=failed
```

So the performance sweep is meaningful for throughput exploration, but bitwise
routed determinism is still not solved.

## Fast-Sweep Change

Default `make glm52_stage_bucket_sweep` now uses:

```text
execution_mode=direct_cached_validator
```

The old package-per-attempt behavior is opt-in:

```text
GLM52_STAGE_SWEEP_PACKAGE_EACH_RUN=1
```

Use `GLM52_STAGE_SWEEP_FORCE_VALIDATOR_REBUILD=1` only when the validator source
or link inputs changed.

## Routed Stage Results

These are hidden-input routed stage timings with graph replay disabled.

```text
validator_cache=build/glm52_stage_bucket_sweep_validators_fast_default
output_root=build/glm52_stage_bucket_sweep_all_routed_fast_default_20260701
```

For stages `27:8`, `35:8`, and `43:8`, the old stage-specific pack
directories failed to bind. The passing runs used:

```text
build/glm52_b12x_resident_moe_0027_0050_v3
```

| stage | B8 ms | B16 ms | B32 ms | B64 ms | B64 tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 11:8 | 35.454 | 36.034 | 56.829 | 60.976 | 1049.6 |
| 19:8 | 34.034 | 37.975 | 45.706 | 62.081 | 1030.9 |
| 27:8 | 33.249 | 38.137 | 47.768 | 63.309 | 1010.9 |
| 35:8 | 36.094 | 38.423 | 46.983 | 61.205 | 1045.7 |
| 43:8 | 34.243 | 39.776 | 48.939 | 63.481 | 1008.2 |
| 51:8 | 36.607 | 38.919 | 46.434 | 61.265 | 1044.6 |
| 59:8 | 36.370 | 37.988 | 50.285 | 63.960 | 1000.6 |
| 67:8 | 35.433 | 41.471 | 46.555 | 63.460 | 1008.5 |
| 75:3 final | 12.515 | 14.383 | 17.258 | 23.108 | 2769.6 |

The routed B64 slowest stage is currently `59:8` at `63.960 ms`, which implies
about `1000 tok/s` for a filled pipe if the front stage were equally balanced.

## Prefix Result

The old dense-prefix timing measured prefill plus current token:

```text
GLM52_DENSE_PREFIX_CURRENT_TOKEN_ONLY=0
B64 total_us=360626.436
submissions=44
ceiling=177.5 tok/s
```

That is not decode-throughput shape. The new validator flag measures the
current token only with resident KV assumed:

```text
GLM52_DENSE_PREFIX_CURRENT_TOKEN_ONLY=1
```

Current-token prefix results:

| prefix mode | B8 ms | B16 ms | B32 ms | B64 ms | B64 tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0-10 current token | 54.165 | 57.619 | 69.966 | 89.263 | 717.0 |

The existing `0-10` front grouping is now the real throughput limiter at B64.

## Stage Grouping Implication

Current grouping:

```text
0-10, 11-18, 19-26, 27-34, 35-42, 43-50, 51-58, 59-66, 67-74, 75-77
```

At B64 this is limited by:

```text
0-10 current-token prefix ~= 89.3 ms
filled-pipeline ceiling ~= 64 * 1000 / 89.3 ~= 717 tok/s
```

The routed layers are already near the 1000 tok/s B64 target with 8-layer
chunks. The front chunk needs to be split.

Recommended next grouping to test for 12 Sparks:

```text
0-5, 6-12, 13-19, 20-26, 27-33, 34-40,
41-47, 48-54, 55-61, 62-68, 69-74, 75-77
```

This is not final. It is the next testable hypothesis: 6-7 layers per Spark
should move the slowest B64 stage toward the 50-65 ms band.

For 13 Sparks, test 6-layer-ish chunks:

```text
0-5, 6-11, 12-17, 18-23, 24-29, 30-35, 36-41,
42-47, 48-53, 54-59, 60-65, 66-71, 72-77
```

For 8 Sparks, feasible but likely lower throughput:

```text
0-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70-77
```

The 8-Spark plan should still work functionally, but B64 is likely bounded near
700-800 tok/s unless per-stage graphing and projection fast paths improve.

## Known Bad Artifacts

These old pack dirs failed B12x binding during the fast sweep:

```text
build/glm52_b12x_resident_moe_0027_0034
build/glm52_b12x_resident_moe_0035_0042
build/glm52_b12x_resident_moe_0043_0050
```

Use the v3 combined pack until fresh per-stage packs are regenerated:

```text
build/glm52_b12x_resident_moe_0027_0050_v3
```

## Next Work

1. Add stage-slice measurement support for arbitrary layer ranges starting at 0.
2. Generate local `.spb12x` packs for the proposed 12-Spark and 13-Spark
   groupings.
3. Measure B16/B32/B64 with graph replay enabled and disabled.
4. Pick the grouping by `max_stage_ms(B)` for the workload target.
5. Keep package validation opt-in unless package artifacts or validator source
   changed.
