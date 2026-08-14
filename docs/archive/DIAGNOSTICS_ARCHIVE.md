# Diagnostics archive

The captured tensors, traces and logs that back every measured claim in `docs/`.

They are no longer in the tree. Two thirds of this repository's FILES were binary
dumps from past debugging runs, which made every ratio meaningless and made
housekeeping look like progress. They are not waste - this codebase's failure mode
has been claims without evidence - so they are archived rather than deleted, and
this file records which run backs which document.

## Fetch

```
sh tools/get_diagnostics.sh          # into ./diagnostics
sh tools/get_diagnostics.sh /tmp/d   # or anywhere
```

| | |
|---|---|
| release | `diagnostics-20260727` |
| asset | `diagnostics-20260727.tar.gz` |
| sha256 | `638c2b8088052b60dfc5756ee92ff741f66f9145d0b46ed97833f152cafb76b1` |
| size | 27,233,555 bytes compressed, 37,932,799 uncompressed |
| contents | 2,533 files across 18 runs |

## Runs

Ordered by whether a document depends on them. A run nothing cites is kept
because absence of a citation today is not evidence it will not be needed.

| run | files | bytes | cited by |
|---|---:|---:|---|
| `glm52_b64_api_performance_20260714` | 14 | 366,117 | `GB10_CUDA_COST_MODEL_CALIBRATION.md`, `GLM52_PP13_MULTIROW_LINEAR_PLAN_FIX_20260718.md`, `GLM52_MEASURED_STATUS.md` |
| `glm52_fp8_scaled_gemm_20260712` | 4 | 13,788 | `GLM52_FP8_SCALED_GEMM_ACTIVATION_20260712.md`, `GLM52_MEASURED_STATUS.md` |
| `glm52_mtp_b1_f536418_20260714` | 8 | 565,638 | `GLM52_MTP_B1_MEASUREMENT_20260714.md`, `GLM52_MEASURED_STATUS.md` |
| `glm52_measured_status_e459d41_20260712` | 7 | 7,376 | `GLM52_MEASURED_STATUS.md` |
| `glm52_pp13_diff_20260708` | 79 | 1,795,521 | `GLM52_PP13_WRONG_TOKENS_ROOT_CAUSE_20260710.md` |
| `glm52_pp13_diff_20260709` | 193 | 2,342,737 | `GLM52_PP13_WRONG_TOKENS_ROOT_CAUSE_20260710.md` |
| `glm52_pp13_diff_20260709_release231_seq1` | 73 | 889,496 | `GLM52_PP13_WRONG_TOKENS_ROOT_CAUSE_20260710.md` |
| `glm52_active_row_plan_20260714` | 907 | 10,883,586 | — |
| `glm52_active_row_ring_equivalence_20260714` | 629 | 11,797,556 | — |
| `glm52_b1_fp8_exact_20260712` | 93 | 1,873,099 | — |
| `glm52_b1_fp8_isolation_20260712` | 39 | 929,473 | — |
| `glm52_b256_compressed_mla_20260715` | 1 | 2,422 | — |
| `glm52_direct_fp8_tiled_20260716` | 18 | 870,354 | — |
| `glm52_fp8_stage0_official_20260710` | 57 | 740,330 | — |
| `glm52_mla_dsa_production_20260710` | 183 | 1,849,254 | — |
| `glm52_mla_dsa_ring_20260710` | 181 | 2,220,915 | — |
| `glm52_mtp_depth_cap_falsification_20260715` | 44 | 756,658 | — |
| `glm52_pp13_attention_contract_20260713` | 3 | 28,479 | — |

## If the release goes away

These documents lose their evidence and become assertions:

- `docs/GB10_CUDA_COST_MODEL_CALIBRATION.md` depends on `glm52_b64_api_performance_20260714`
- `docs/GLM52_PP13_MULTIROW_LINEAR_PLAN_FIX_20260718.md` depends on `glm52_b64_api_performance_20260714`
- `docs/GLM52_MEASURED_STATUS.md` depends on `glm52_b64_api_performance_20260714`
- `docs/GLM52_FP8_SCALED_GEMM_ACTIVATION_20260712.md` depends on `glm52_fp8_scaled_gemm_20260712`
- `docs/GLM52_MEASURED_STATUS.md` depends on `glm52_fp8_scaled_gemm_20260712`
- `docs/GLM52_MEASURED_STATUS.md` depends on `glm52_measured_status_e459d41_20260712`
- `docs/GLM52_MTP_B1_MEASUREMENT_20260714.md` depends on `glm52_mtp_b1_f536418_20260714`
- `docs/GLM52_MEASURED_STATUS.md` depends on `glm52_mtp_b1_f536418_20260714`
- `docs/GLM52_PP13_WRONG_TOKENS_ROOT_CAUSE_20260710.md` depends on `glm52_pp13_diff_20260708`
- `docs/GLM52_PP13_WRONG_TOKENS_ROOT_CAUSE_20260710.md` depends on `glm52_pp13_diff_20260709`
- `docs/GLM52_PP13_WRONG_TOKENS_ROOT_CAUSE_20260710.md` depends on `glm52_pp13_diff_20260709_release231_seq1`
