# Routed-expert codec plumbing and parity (glm5_next)

This page covers two things for the quantization A/B work (lanes/quant-ab-design.md §4.2, lane L4):

- how a routed-expert codec's planes are described to weightd;
- how the routed MLP is checked on real pack layers for every expert path.

## Expert planes and the `.experts` manifest

`include/sparkpipe/spark_expert_planes.h` describes, per expert codec, the byte planes of one routed-expert entry. The glm5_next and glm52 module attach checks (`include/sparkpipe/family/module/spark_module_manifest_check.h`) and the `.experts` generators (`tools/glm5_next_experts_manifest.c`, `tools/glm52_experts_manifest.c`) all use it.

| Codec | Scale encoding | Planes per expert (range kind) | Address of expert `e` (E experts) |
|---|---|---|---|
| bf16 | none | payload (`2k`) | `payload_offset + e*P` |
| fp8, int6/7/8 | f32 | payload (`2k`), scale (`2k+1`) | `scale_offset + e*S` |
| mxfp4 | e8m0 | payload (`2k`), scale (`2k+1`) | `scale_offset + e*S` |
| nvfp4 | ue4m3 + f32 global | payload (`2k`), block scales (`2k+1`), global (`0x10000 + k`) | blocks at `scale_offset + E*4 + e*B`, global at `scale_offset + e*4` |

Here `k` is the entry's tensor kind. The fp8/int/mxfp4 kinds are unchanged, so every existing fp8 `.experts` file is still valid. On sparkf, the new generator reproduces production's rank-f `.experts` byte for byte, and the new check accepts it.

The check requires every plane of every expert to be covered by exactly one range at the plane's address and size, and the manifest's range count to equal the sum over planes. weightd already refuses a duplicate kind inside an expert group. NVFP4 at TP16 needs 42 layers × 288 experts × 6 ranges = 72,576 ranges, under the 131,072 cap, and 6 ranges per expert, under the cap of 16.

To check a placed pack's manifest the same way the module does at attach:

    build/glm5_next_experts_manifest --check <pack.sp>

## Routed-MLP parity on real layers

The validator is `modules/glm5_next_resident_decode_stage/validation/validate_glm5_next_resident_decode_stage_routed_parity.sh <codec> <pack> <out> [ceiling-dump]`.

It compiles the harness for the codec and reads the routed-expert entries of real pack layers (default 3, 23, 44). With seeded BF16 activations and seeded top-8 routes, it runs the routed MLP through:

- the module's natural dispatch (`Glm5NextLayerMoeUp/Down`);
- the skinny path, forced;
- the grouped-skinny path, forced;
- the tensor-core GEMM path, forced.

The row sets are 1, 9 (k+1 verify rows), 16, 64 and 256 (the prefill submission). The harness records which path natural dispatch took and requires its output to be bitwise equal to that path's output.

The oracle is `tools/glm5_next_routed_oracle.py`. It dequantizes the same pack bytes in float64:

- fp8: e4m3 × f32 block scale;
- bf16;
- nvfp4: e2m1 × ue4m3 × f32 global, the modelopt formula.

It computes two references:

- **exact:** the codec's true value;
- **emulated:** BF16 rounding at the kernels' materialization points (gate_up, swiglu output, each expert output, the combined row). For the GEMM path it also rounds each dequantized weight to BF16 before the MMA.

A path passes when all three hold:

- At most max(8, 0.1%) of its elements fall outside the T1 band (|d| ≤ 1e-3 or ≤ 2% relative, docs/T1_REFERENCE_COMPARE.md) against the emulated reference.
- Its relative L2 error against the emulated reference is at most 2^-8.
- Its relative L2 error against the exact reference is at most 1%.

The outlier allowance exists because a one-ulp BF16 flip at gate_up moves a few cancelling output elements.

With a ceiling dump (the BF16-expert arm, already scored), every path is also scored against the ceiling's exact reference. The oracle refuses the whole run unless every run's inputs (`x.bf16`, `route_expert.u32`, `route_weight.f32`) are byte-identical to the ceiling's and the ceiling has a complete `ref.f64`. A ceiling that does not match is never skipped silently.

### Which path each wave shape takes (glm5_next at TP16, main 266b6adbc)

| Codec | 1 row | 9 / 16 / 64 / 256 rows |
|---|---|---|
| fp8, bf16 | skinny | grouped skinny |
| nvfp4 | GEMM | GEMM |

- **fp8 and bf16.** `LmSkinnyGroupedExperts` accepts up to 16 × 288 routed pairs, which is 576 rows at top-8. So fp8 and bf16 reach the tensor-core GEMM only above 576 rows per wave. Teacher-forced prefill waves of at most 256 rows therefore score fp8 through the f32-scale skinny arithmetic, the same as B1 decode.
- **nvfp4 (and int6/7/8).** These have no skinny format, so every row takes the GEMM path. There the dequantized weight is rounded to BF16 before the MMA.

The measured cost of that rounding for fp8 (forced GEMM vs skinny, relative L2 against the exact codec value) is 4.4-5.1e-3 vs 3.6-4.1e-3. For bf16 the two paths are equal (3.5-4.1e-3).
