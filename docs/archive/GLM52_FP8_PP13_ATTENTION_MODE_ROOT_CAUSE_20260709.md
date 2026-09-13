# GLM52 FP8 PP13 rank0 correctness root cause, 2026-07-09

## Verdict

The live PP13 FP8 ring corruption is reproduced by the production attention mode, not by transport, FP8 MoE packs, graph replay, runtime KV tables, physical KV block remapping, MTP frame context, or built-in PP13 unroll.

Production builder was selecting:

```text
SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTENTION_EXECUTION_ABSORBED_LATENT
```

Exact stage0 validation with that same mode reproduces the bad ring rank0 token-0 output byte-for-byte. Exact stage0 validation with tiled online softmax matches the oracle.

The production default is changed to:

```text
SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTENTION_EXECUTION_TILED_ONLINE_SOFTMAX
```

Absorbed-latent attention remains a separate kernel correctness bug and must not be the serving default until it has its own numeric gate.

## Raw artifact paths

```text
Ring dump:
/home/spark0/sparkpipe_trace_runs/b1024_pr219_dump_20260708T1158Z

Oracle stage0 output:
/home/spark0/sparkpipe_trace_runs/spark0_fp8_stage0_oracle_20260708T1245Z/after_layer_5_fp8_stage0.bf16

Exact runtime-KV output:
/home/spark0/sparkpipe_trace_runs/fp8_stage0_b1024_capacity_active1_20260709Tnow/output/after_layer_5_fp8_cutlass_stage0_b1024cap_active1_runtimekv.bf16

Exact runtime-KV + frame context + MTP output:
/home/spark0/sparkpipe_trace_runs/fp8_stage0_b1024_capacity_active1_20260709Tnow/output/after_layer_5_fp8_cutlass_stage0_b1024cap_active1_runtimekv_framectx_mtp.bf16

Exact physical-block-base=1 output:
/home/spark0/sparkpipe_trace_runs/fp8_stage0_b1024_capacity_active1_20260709Tnow/output/after_layer_5_fp8_cutlass_stage0_b1024cap_active1_physbase1.bf16

Exact absorbed-latent output:
/home/spark0/sparkpipe_trace_runs/fp8_stage0_b1024_capacity_active1_20260709Tnow/output/after_layer_5_fp8_cutlass_stage0_b1024cap_active1_attention_absorbed.bf16

Exact tiled-online-softmax output:
/home/spark0/sparkpipe_trace_runs/fp8_stage0_b1024_capacity_active1_20260709Tnow/output/after_layer_5_fp8_cutlass_stage0_b1024cap_active1_attention_tiled.bf16
```

## Numeric discriminator

All comparisons below use BF16 hidden rows, hidden dimension 6144.

```text
runtimekv_vs_oracle
token=0 rel_l2=0.004102960 cos=0.999991583 max_abs=4.882812500e-04

runtimekv_framectx_mtp_vs_oracle
token=0 rel_l2=0.004102960 cos=0.999991583 max_abs=4.882812500e-04

physbase1_vs_oracle
token=0 rel_l2=0.004102960 cos=0.999991583 max_abs=4.882812500e-04

absorbed_vs_oracle
token=0 rel_l2=14.425851185 cos=0.068419255 max_abs=3.955444336e+00

ring_rank0_tok0_vs_absorbed_tok0
token=0 rel_l2=0.000000000 cos=1.000000000 max_abs=0.000000000e+00

tiled_vs_oracle
token=0 rel_l2=0.004102960 cos=0.999991583 max_abs=4.882812500e-04
```

## Eliminated causes

```text
transport: hop hashes were clean
FP8 stagepack non-MoE bytes: rank0 non-MoE payload matched HF FP8 exactly
FP8 MoE pack parser: exact FP8 MoE validation matched oracle when attention mode was tiled
graph replay/built-in PP13 unroll: builtin-vs-loop was byte-identical
runtime KV table: exact runtime-KV matched oracle
MTP frame context: exact runtime-KV+framectx+MTP matched oracle
physical KV block remap: exact physical-block-base=1 matched oracle
```

## Validation commands run

```text
make -j test
make -j glm52_pp13_node_context_builder
make -j -B cuda_glm52_resident_decode_stage
```

All completed successfully; CUDA builds emitted warnings only.
