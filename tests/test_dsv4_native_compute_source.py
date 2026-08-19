#!/usr/bin/env python3
"""Fail-closed source contract for DSV4's native SM121 decode compute.

This test deliberately does not claim numerical or hardware qualification.  It
checks the properties that can be proved without a CUDA toolkit/GPU: the exact
PTX atoms and packed conversion instructions are reachable from the production
launchers; DSV4 admits only the three qualified shapes; B1 has exact-width
expert tiles while B8/B1024 retain tensor-core routes; W13 owns both BF16
rounding boundaries; W2 never routes through the BF16-dequant weight-only GEMM;
strided stores keep the full row stride. tests/test_ptx_capability_gate.py
separately assembles the exact PTX forms when ptxas is installed.
"""

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / "model-families/common/include/sparkpipe/spark_lm_kernels.cuh"
DSV4 = ROOT / "modules/dsv4_resident_decode_stage/source/spark_dsv4_resident_decode_stage_cuda.cu"
MODULE = ROOT / "modules/dsv4_resident_decode_stage/source/spark_dsv4_resident_decode_stage_module.c"
HEAD_SCREEN = ROOT / "include/sparkpipe/spark_head_screen.h"
MMA = ROOT / "inference/kernels/mma.cuh"
ROUTE = ROOT / "inference/kernels/route.cuh"


def body(source: str, name: str) -> str:
    match = re.search(r"\b" + re.escape(name) + r"\s*\([^;]*?\)\s*\{", source, re.S)
    if match is None:
        raise AssertionError(f"missing function {name}")
    start = match.end()
    depth = 1
    for index in range(start, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start:index]
    raise AssertionError(f"unterminated function {name}")


def require(source: str, needle: str, label: str) -> None:
    if needle not in source:
        raise AssertionError(f"missing {label}: {needle}")


def forbid(source: str, needle: str, label: str) -> None:
    if needle in source:
        raise AssertionError(f"forbidden {label}: {needle}")


def main() -> int:
    common = COMMON.read_text(encoding="utf-8")
    dsv4 = DSV4.read_text(encoding="utf-8")
    module = MODULE.read_text(encoding="utf-8")
    head_contract = HEAD_SCREEN.read_text(encoding="utf-8")
    mma = MMA.read_text(encoding="utf-8")
    route = ROUTE.read_text(encoding="utf-8")

    require(mma, "defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1210)",
            "SM121-only device gate")
    require(mma, "m16n8k32.row.col.kind::mxf8f6f4", "documented PTX modifier order")
    require(mma, ".block_scale.scale_vec::1X", "block-scaled family")
    require(mma, ".f32.e4m3.e2m1.f32.ue8m0", "MXFP8 x MXFP4 atom")
    require(mma, ".f32.e4m3.e4m3.f32.ue8m0", "MXFP8 x MXFP8 atom")
    for name in ("LmMmaMxf8Mxf4", "LmMmaMxf8Mxf8"):
        require(body(mma, name), 'asm volatile("trap;', f"{name} fail-closed trap")
    require(body(common, "SparkLmDecodeE2m1x8Half2"),
            "cvt.rn.f16x2.e2m1x2", "packed E2M1 conversion")
    require(body(common, "SparkLmDecodeE4m3x4Half2"),
            "cvt.rn.f16x2.e4m3x2", "packed E4M3 conversion")
    require(body(common, "SparkLmDotRowFp8"), "packed = __ldcs(",
            "streaming immutable FP8 payload load")
    if common.count("__ldcs(") != 1:
        raise AssertionError("streaming cache policy must stay limited to SparkLmDotRowFp8")

    shape = body(common, "SparkLmSm121NativeDecodeShape")
    require(shape, "rows == 1u || rows == 5u || rows == 7u || rows == 8u || rows == 16u || rows == 32u || rows == 64u || rows == 1024u", "exact decode buckets")
    require(dsv4, "SparkDsv4RequireNativeDecodeShape(row_count)", "dense shape gate")
    require(dsv4, "SparkDsv4RequireNativeDecodeShape(rows)", "MoE shape gate")
    require(dsv4, "properties.major != 12", "runtime major check")
    require(dsv4, "properties.minor != 1", "runtime minor check")
    route_build = body(route, "LmRouteBuild")
    require(route_build, "tile_n_up", "independent W13 route tile")
    require(route_build, "tile_n_down", "independent W2 route tile")
    route_launch = body(dsv4, "SparkDsv4LaunchGateRouteCooperative")
    require(route_launch, "SparkLmSm121ExpertW13TileN(1u)",
            "shared W13 tile policy")
    require(route_launch, "SparkLmSm121ExpertW2TileN(1u)",
            "shared W2 tile policy")
    require(common, "#define SPARK_LM_SM121_B1_EXPERT_W2_TILE_N 128u",
            "measured B1 W2 N128 tile")
    require(common,
            "#define SPARK_LM_SM121_B1_EXPERT_W2_BLOCKS_PER_SM 4u",
            "measured B1 W2 four-CTA occupancy")
    require(common,
            "#define SPARK_LM_SM121_B1_DENSE_W13_CTA_THREADS 1024u",
            "measured B1 shared W13 32-warp CTA")
    gate_route = body(dsv4, "SparkDsv4LaunchGateRoute")
    require(gate_route, "if ( row_count == 1u )",
            "runtime cooperative B1 gate-route selection")
    require(gate_route, "SparkDsv4LaunchGateRouteCooperative",
            "cooperative B1 gate-route path")
    require(gate_route, "SparkDsv4LaunchGateRouteBatched",
            "B2-B1024 gate-route path")
    forbid(gate_route, "#if", "compile-time gate-route feature fork")
    cooperative_gate = body(dsv4, "SparkDsv4GateRouteCooperativeKernel")
    require(cooperative_gate, "cooperative_groups::this_grid()",
            "one cooperative gate-score grid")
    require(cooperative_gate, "grid.sync()",
            "score completion before exact route selection")
    require(cooperative_gate, "SparkDsv4GateSelectShared",
            "exact fused top-k selection")
    require(cooperative_gate, "SparkDsv4GateRouteBuildShared",
            "exact fused route table build")
    forbid(cooperative_gate, "SparkDsv4GateScoresKernel<<<",
           "wrapped legacy gate-score launch")
    run_moe = body(module, "SparkDsv4ModuleRunMoe")
    if run_moe.count("SparkDsv4LaunchGateRoute(") != 1:
        raise AssertionError("production MoE must issue one gate-route entrypoint")
    forbid(run_moe, "SparkDsv4LaunchGateScores",
           "split production gate-score launch")
    forbid(run_moe, "SparkDsv4LaunchGateSelect",
           "split production gate-select launch")
    forbid(run_moe, "SparkDsv4LaunchMoeRoute",
           "split production route-build launch")
    validator = (ROOT / "modules/dsv4_resident_decode_stage/validation/"
                 "spark_dsv4_resident_decode_stage_cuda_validation.cu").read_text(
                     encoding="utf-8")
    gate_validation = body(validator, "SparkDsv4ValidationGateRouteRunCase")
    require(gate_validation, "SparkDsv4LaunchGateRoute(cudaStreamPerThread",
            "production-entrypoint gate-route validation")
    require(gate_validation, "score_expected", "score-route tie fixture")
    require(gate_validation, "hash_expected", "hash-route fixture")
    require(body(validator, "main"), "SparkDsv4ValidationGateRouteB1()",
            "mandatory hardware gate-route validation")

    query_kernel = body(dsv4, "SparkDsv4QueryHeadRmsRopeKernel")
    require(dsv4, "uint32_t rope_dim,float epsilon,uint32_t inverse)",
            "query runtime RoPE-direction kernel contract")
    require(query_kernel, "head_dim,rope_dim,inverse",
            "query runtime direction reaches the shared primitive")
    forbid(query_kernel, "head_dim,rope_dim,0u",
           "query constant-folded RoPE direction")
    query_rms = query_kernel.index("SparkDsv4QueryHeadRmsRow")
    query_boundary = query_kernel.index("__syncthreads()", query_rms)
    query_rope = query_kernel.index("SparkDsv4RopeRow", query_boundary)
    if not query_rms < query_boundary < query_rope:
        raise AssertionError("query fusion must preserve the BF16 RMS boundary before RoPE")
    kv_kernel = body(dsv4, "SparkDsv4KvPostKernel")
    require(dsv4, "float epsilon,\n\tuint32_t inverse)",
            "KV runtime RoPE-direction kernel contract")
    require(kv_kernel, "head_dim,rope_dim,lane,inverse",
            "KV runtime direction reaches the shared primitive")
    forbid(kv_kernel, "head_dim,rope_dim,lane,0u",
           "KV constant-folded RoPE direction")
    kv_rms = kv_kernel.index("SparkLmRmsNormRow")
    kv_boundary = kv_kernel.index("__syncthreads()", kv_rms)
    require(kv_kernel, "warp * quant_block < quant_width",
            "disjoint KV quantization warps")
    require(kv_kernel, "warp == SPARK_LM_CTA_WARPS - 1u",
            "disjoint KV RoPE warp")
    require(kv_kernel, "448.0f,0u", "exact KV FP8 quantization simulation")
    if not kv_rms < kv_boundary < kv_kernel.index("SparkDsv4QuantSimGroup"):
        raise AssertionError("KV fusion must preserve the BF16 RMS boundary")
    for launcher in ("SparkDsv4LaunchQueryHeadRmsRope", "SparkDsv4LaunchKvPost"):
        launch_body = body(dsv4, launcher)
        require(launch_body,
                "SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT",
                f"{launcher} B1-B1024 shape gate")
        forbid(launch_body, "#if", f"{launcher} compile-time feature fork")
        require(launch_body, "uint32_t inverse = 0u;",
                f"{launcher} runtime forward-direction launch argument")
    query_production = body(module, "SparkDsv4ModuleRunQueryProjection")
    forbid(query_production, "SparkDsv4LaunchQueryHeadRmsRope(",
           "fused production query path in the KV-indexer bisection")
    require(query_production, "SparkDsv4LaunchQueryHeadRms(",
            "canonical production query RMS")
    require(query_production, "SparkDsv4LaunchRope(",
            "canonical production query RoPE")
    if not query_production.index("SparkDsv4LaunchQueryHeadRms(") < \
            query_production.index("SparkDsv4LaunchRope("):
        raise AssertionError("canonical query RMS must complete before RoPE")
    kv_production = body(module, "SparkDsv4ModuleRunKvPost")
    require(kv_production, "SparkDsv4LaunchKvPost(",
            "one-launch KV post-processing")
    for legacy in ("SparkDsv4LaunchRmsNorm(", "SparkDsv4LaunchRope(",
                   "SparkDsv4LaunchQuantSim("):
        forbid(kv_production, legacy, "sequential production KV post-processing")
    query_validation = body(validator, "SparkDsv4ValidationQueryPostCase")
    require(query_validation, "SparkDsv4LaunchQueryHeadRms(",
            "query fusion control")
    require(query_validation, "SparkDsv4LaunchQueryHeadRmsRope(",
            "query fusion candidate")
    kv_validation = body(validator, "SparkDsv4ValidationKvPostCase")
    for call in ("SparkDsv4LaunchRmsNorm(", "SparkDsv4LaunchRope(",
                 "SparkDsv4LaunchQuantSim(", "SparkDsv4LaunchKvPost("):
        require(kv_validation, call, "KV fusion exact hardware comparison")
    require(body(validator, "main"), "SparkDsv4ValidationPostFusions()",
            "mandatory post-fusion hardware validation")

    fused = body(common, "SparkLmSm121FusedExpertW13Kernel")
    if fused.count("SparkLmSm121StageMxf8<TILE_M>(") != 1:
        raise AssertionError("fused expert W13 must stage each activation K tile once")
    require(fused, "LmMmaMxf8Mxf4(gate_total", "native packed W1")
    require(fused, "LmMmaMxf8Mxf4(up_total", "native packed W3")
    if fused.count("SparkLmSm121Bf16Round(") != 2:
        raise AssertionError("fused expert W13 must BF16-round both projections exactly once")
    gate_round = fused.index("gate = SparkLmSm121Bf16Round")
    up_round = fused.index("up = SparkLmSm121Bf16Round")
    clamp = fused.index("gate = gate > limit ? limit : gate")
    swiglu = fused.index("SparkLmSwish(gate) * up")
    if not (gate_round < clamp and up_round < clamp < swiglu):
        raise AssertionError("W13 boundary must be projection->BF16->clamp->SiLU/product->BF16")
    require(fused, "up < -limit ? -limit : up", "two-sided up clamp")
    forbid(fused, "SparkLmDecodeE2m1", "software MXFP4 decode")
    forbid(fused, "LmMmaBf16", "BF16 MMA fallback")

    b1_fused = body(common, "SparkLmSm121B1ExpertW13Task")
    require(b1_fused, "SparkLmDotRowMxfp4Pair<32u>",
            "B1 interleaved packed W1/W3 GEMV")
    if b1_fused.count("SparkLmSm121Bf16Round(") != 2:
        raise AssertionError("B1 expert W13 must BF16-round both projections")
    b1_clamp = b1_fused.index("gate = gate > limit ? limit : gate")
    if not (b1_fused.index("gate = SparkLmSm121Bf16Round") < b1_clamp and
            b1_fused.index("up = SparkLmSm121Bf16Round") < b1_clamp <
            b1_fused.index("SparkLmSwish(gate) * up")):
        raise AssertionError("B1 W13 lost the BF16/clamp/SwiGLU boundary")

    shared = body(common, "SparkLmSm121FusedDenseW13Kernel")
    if shared.count("SparkLmSm121StageMxf8<TILE_M>(") != 1:
        raise AssertionError("fused shared W13 must stage each activation K tile once")
    require(shared, "SparkLmSm121Mma<WEIGHT_BITS>(gate_total",
            "native shared W1")
    require(shared, "SparkLmSm121Mma<WEIGHT_BITS>(up_total",
            "native shared W3")
    if shared.count("SparkLmSm121Bf16Round(") != 2:
        raise AssertionError("fused shared W13 must BF16-round both projections")
    shared_clamp = shared.index("gate = gate > limit ? limit : gate")
    if not (shared.index("gate = SparkLmSm121Bf16Round") < shared_clamp and
            shared.index("up = SparkLmSm121Bf16Round") < shared_clamp <
            shared.index("SparkLmSwish(gate) * up")):
        raise AssertionError("shared W13 lost the BF16/clamp/SwiGLU boundary")

    w2 = body(common, "SparkLmSm121ExpertW2Kernel")
    require(w2, "SparkLmSm121LoadMxf4B", "packed MXFP4 W2 load")
    require(w2, "LmMmaMxf8Mxf4(total", "native mixed-width W2 MMA")
    mxf4_loader = body(common, "SparkLmSm121LoadMxf4B")
    mxf8_loader = body(common, "SparkLmSm121LoadMxf8B")
    require(mxf4_loader,
            "uint32_t neuron = neuron_base + LmMma8OperandBRow(lane);",
            "MXFP4 B-fragment lane row")
    require(mxf8_loader,
            "uint32_t neuron = neuron_base + LmMma8OperandBRow(lane);",
            "MXFP8 B-fragment lane row")
    require(mxf4_loader, "& 15u) << 2u",
            "E2M1 central-bit PTX byte packing")
    forbid(w2, "SparkLmDecodeE2m1", "software W2 dequant")
    forbid(w2, "LmMmaBf16", "BF16 W2 MMA")
    require(body(common, "SparkLmSm121B1ExpertW2Task"),
            "SparkLmDotRowMxfp4<32u>", "B1 packed W2 GEMV")

    strided = body(common, "SparkLmSm121NativeLinearKernel")
    require(strided, "(uint64_t)row * output_row_stride + output_offset +",
            "B>1 full output row stride")
    require(strided, "group * output_group_stride + column", "grouped output slice")
    routed_reduce = body(common, "SparkLmMoePairReduceStridedKernel")
    require(routed_reduce,
            "(uint64_t)row * accum_row_stride + accum_offset",
            "B>1 routed-reduce full output row stride")
    require(body(dsv4, "SparkDsv4LaunchMoePairReduceStrided"),
            "SparkLmHostLaunchMoePairReduceStrided",
            "DSV4 strided routed-reduce launcher")

    attention = body(module, "SparkDsv4ModuleRunAttentionTail")
    require(attention, "SparkDsv4LaunchLinear(stream,&layer->attn.wo_b",
            "column-parallel WO full-hidden partial")
    forbid(attention, "state->tp_rank * local_hidden",
           "diagonal WO rank-row write")
    routed = body(module, "SparkDsv4ModuleRunMoeRoutedProjection")
    moe = body(module, "SparkDsv4ModuleRunMoe")
    require(moe, "SparkDsv4LaunchMoePairReduce(stream",
            "column-parallel routed full-hidden partial")
    forbid(moe, "SparkDsv4LaunchMoePairReduceStrided",
           "diagonal routed rank-row write")
    shared = body(module, "SparkDsv4ModuleRunMoeShared")
    require(shared, "SparkDsv4LaunchExpertLinear(stream,&moe->shared_w2",
            "column-parallel expert-codec shared-W2 full-hidden partial")
    forbid(moe, "state->tp_rank * hidden_dimension",
           "diagonal shared-W2 rank-row write")
    require(moe, "SparkStageModuleCudaForkBegin",
            "shared-routed MoE fork")
    require(moe, "SparkStageModuleCudaForkJoin",
            "shared-routed MoE join")
    reduce_hidden = body(module, "SparkDsv4ModuleReduceHidden")
    require(reduce_hidden, "submission.local_device = device_bf16;",
            "in-place full-hidden reduction input")
    require(module, "SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16",
            "device BF16 sum collective")

    expert_host = body(common, "SparkLmHostLaunchSm121FusedExpertW13")
    w2_host = body(common, "SparkLmHostLaunchSm121ExpertW2")
    require(expert_host, "if ( rows == 1024u )", "B1024 routed W13 M64 tile")
    require(expert_host, "SPARK_LM_SM121_NATIVE_WIDE_TILE_M",
            "wide routed W13 specialization")
    require(w2_host, "if ( rows == 1024u )", "B1024 routed W2 M64 tile")
    require(w2_host, "SPARK_LM_SM121_NATIVE_WIDE_TILE_M",
            "wide routed W2 specialization")
    require(expert_host, "if ( rows == 1u )", "true-B1 routed W13 dispatch")
    require(expert_host, "SPARK_LM_SM121_B1_EXPERT_W13_TILE_N",
            "B1 W13 N32 tile")
    require(w2_host, "if ( rows == 1u )", "true-B1 routed W2 dispatch")
    require(w2_host, "SPARK_LM_SM121_B1_EXPERT_W2_TILE_N", "B1 W2 N128 tile")
    require(w2_host, "SPARK_LM_SM121_B1_EXPERT_W2_BLOCKS_PER_SM",
            "B1 W2 four-CTA launch")

    up = body(dsv4, "SparkDsv4LaunchExpertUp")
    require(up, "return(cudaErrorInvalidValue);", "retired split-up fail closed")
    fused_launch = body(dsv4, "SparkDsv4LaunchFusedExpertW13Act")
    require(fused_launch, "SparkLmHostLaunchSm121FusedExpertW13", "routed fused launcher")
    shared_launch = body(dsv4, "SparkDsv4LaunchFusedSharedW13Act")
    require(shared_launch, "SparkLmHostLaunchSm121FusedDenseW13", "shared fused launcher")
    down = body(dsv4, "SparkDsv4LaunchExpertDown")
    require(down, "SparkLmHostLaunchSm121ExpertW2", "native routed W2 launcher")
    forbid(down, "LmGemmWeightOnlyLaunch", "BF16-dequant routed W2 launch")

    dense = body(dsv4, "SparkDsv4LaunchLinearCodec")
    require(dense, "SparkLmHostLaunchSm121DecodeLinear", "shape-aware dense route")
    require(body(dsv4, "SparkDsv4LaunchLinear"),
            "SPARK_DSV4_MODEL_NON_EXPERT_ACTIVATION_CODEC",
            "BF16 spine activation specialization")
    require(body(dsv4, "SparkDsv4LaunchExpertLinear"),
            "SPARK_DSV4_MODEL_EXPERT_ACTIVATION_CODEC",
            "expert activation specialization")
    dense_dispatch = body(common, "SparkLmHostLaunchSm121DecodeLinear")
    require(dense_dispatch, "if ( row_count == 1u )", "true-B1 dispatch")
    require(dense_dispatch, "SparkLmHostLaunchBatchedLinear", "B1 GEMV route")
    require(dense_dispatch, "SparkLmHostLaunchSm121NativeLinear", "B8/B1024 native route")
    scalar_dispatch = body(common, "SparkLmHostLaunchBatchedLinear")
    require(scalar_dispatch,
            "SparkLmLinearKernel<GROUP_SIZE,ACTIVATION_CODEC,\n"
            "\t\t\tSPARK_LM_CTA_WARPS>",
            "measured one-neuron B1 projection route")
    scalar_kernel = body(common, "SparkLmLinearKernel")
    require(scalar_kernel, "blockIdx.y * CTA_WARPS",
            "caller-selected scalar projection geometry")
    require(scalar_dispatch, "SPARK_LM_CTA_THREADS",
            "measured 256-thread B1 projection launch")
    forbid(scalar_dispatch, "SPARK_LM_SCALAR_CTA_THREADS",
           "unmeasured 1024-thread B1 projection launch")
    head_screen = body(common, "SparkLmHostLaunchHeadScreenedArgmaxWithScore")
    require(head_screen,
            "SPARK_ACTIVATION_CODEC_NONE,SPARK_LM_SCALAR_CTA_WARPS",
            "screened-head scalar kernel geometry")
    require(head_screen, "SPARK_LM_SCALAR_CTA_WARPS",
            "screened-head scalar grid geometry")
    require(head_screen, "SPARK_LM_SCALAR_CTA_THREADS",
            "screened-head scalar launch geometry")
    forbid(scalar_dispatch, "SPARK_LM_SCALAR_NEURONS_PER_WARP",
           "slower multi-neuron scalar projection route")
    dense_w13_dispatch = body(common, "SparkLmHostLaunchSm121FusedDenseW13")
    require(dense_w13_dispatch, "if ( row_count == 1u )",
            "true-B1 shared W13 dispatch")
    require(dense_w13_dispatch, "SparkLmSm121FusedDenseW13GemvKernel",
            "B1 shared W13 GEMV")
    require(dense_w13_dispatch,
            "SPARK_LM_SM121_B1_DENSE_W13_CTA_THREADS",
            "B1 shared W13 measured launch geometry")
    require(dense_w13_dispatch,
            "SPARK_LM_SM121_B1_DENSE_W13_CTA_WARPS",
            "B1 shared W13 measured grid geometry")
    dense_w13_gemv = body(common, "SparkLmSm121FusedDenseW13GemvKernel")
    require(dense_w13_gemv, "SPARK_LM_SM121_B1_DENSE_W13_CTA_WARPS",
            "B1 shared W13 measured neuron geometry")
    require(dense_w13_dispatch, "SparkLmSm121FusedDenseW13Kernel",
            "B8/B1024 shared W13 tensor route")
    strided_launch = body(dsv4, "SparkDsv4LaunchStridedLinear")
    require(strided_launch, "SparkLmHostLaunchSm121StridedDecodeLinear",
            "shape-aware strided route")
    strided_dispatch = body(common, "SparkLmHostLaunchSm121StridedDecodeLinear")
    require(strided_dispatch, "if ( row_count == 1u )", "true-B1 strided dispatch")
    require(strided_dispatch, "SparkLmStridedLinearKernel", "B1 strided GEMV route")
    require(strided_dispatch, "SparkLmHostLaunchSm121NativeLinear",
            "B8/B1024 native strided route")
    require(strided_dispatch,
            "SparkLmStridedLinearKernel<GROUP_SIZE,ACTIVATION_CODEC>",
            "measured one-neuron strided B1 route")
    forbid(strided_dispatch, "SPARK_LM_SCALAR_NEURONS_PER_WARP",
           "slower multi-neuron scalar strided route")
    head = body(dsv4, "SparkDsv4LaunchHeadScreenedArgmax")
    require(head, "SparkDsv4RequireNativeDecodeShape(row_count)",
            "screened-head exact-shape/native-device gate")
    sharded_head = body(dsv4, "SparkDsv4LaunchHeadScreenedArgmaxSharded")
    require(sharded_head, "SparkLmHostLaunchHeadScreenedArgmaxWithScore",
            "local exact score and vocabulary-offset head")
    head_dispatch = body(common, "SparkLmHostLaunchHeadScreenedArgmaxWithScore")
    require(head_dispatch, "if ( row_count == 1u )",
            "B1 direct exact-head dispatch")
    require(head_dispatch, "SparkLmHostLaunchHeadDirectArgmaxWithScore",
            "B1 skips overflowing shadow screen")
    direct_head = body(common, "SparkLmHostLaunchHeadDirectArgmaxWithScore")
    require(direct_head, "SparkLmHeadFallbackRescoreKernel",
            "parallel exact vocabulary scan")
    require(direct_head, "SparkLmHeadRescoreArgmaxKernel",
            "exact partial argmax reduction")
    certified_quantize = body(common, "SparkLmHostLaunchHeadCertifiedFp8Quantize")
    require(certified_quantize, "SparkLmHeadCertifiedFp8QuantizeKernel",
            "shared certified FP8 shadow construction")
    certified_score = body(common, "SparkLmHeadCertifiedFp8ScoreKernel")
    require(certified_score, "const uint4 *", "contiguous 16-byte FP8 row loads")
    require(certified_score, "LmE4m3PairToFloat2",
            "packed native E4M3 pair decode")
    require(certified_score, "cert_norm_f32", "groupwise certified bounds")
    certified_screen = body(common, "SparkLmHeadCertifiedScreenKernel")
    require(certified_screen, "candidate_ids[atomicAdd(&cursor,1u)]",
            "full B1 rank-local candidate capacity")
    forbid(certified_screen, "SPARK_LM_HEAD_SCREEN_CAP",
           "no overflowing compatibility scan at B1")
    certified_rescore = body(common, "SparkLmHeadCertifiedRescoreKernel")
    require(certified_rescore, "SparkLmDotRowBf16",
            "untouched BF16 target rescore")
    certified_reduce = body(common, "SparkLmHeadCertifiedReduceKernel")
    require(certified_reduce, "*output_score = best_score[0]",
            "exact BF16 score emission")
    certified_launch = body(common, "SparkLmHostLaunchHeadCertifiedFp8B1WithScore")
    require(certified_launch, "row_count != 1u", "explicit B1 shape contract")
    require(certified_launch, "SparkLmHeadCertifiedFp8ScoreKernel",
            "certified B1 screen launch")
    require(certified_launch, "SparkLmHeadCertifiedRescoreKernel",
            "parallel exact B1 candidate rescore")
    require(certified_launch, "SparkLmHeadCertifiedFp8ScratchView",
            "single model-neutral scratch layout")
    forbid(certified_launch, "SPARK_LM_HEAD_FALLBACK",
           "no legacy fallback contract in certified B1 path")
    scratch_size = body(head_contract, "SparkHeadCertifiedFp8ScratchBytes")
    require(scratch_size, "2u * vocabulary_rows", "coarse and bound storage")
    require(scratch_size, "2u * SPARK_HEAD_CERTIFIED_FP8_PARTIAL_COUNT",
            "exact rescore partial storage")
    require(body(module, "SparkDsv4ModuleBuildHeadShadow"),
            "SparkHeadCertifiedFp8PayloadBytes",
            "model-neutral certified-shadow sizing")
    slot_tail = body(module, "SparkDsv4ModuleAllocateSlotTail")
    require(slot_tail, "SparkHeadCertifiedFp8ScratchBytes",
            "dedicated certified-head scratch allocation")
    require(slot_tail, "&slot->head_certified_scratch",
            "dedicated certified-head scratch pointer")
    require(slot_tail, "&slot->head_candidate_ids_u32",
            "separate certified-head candidate allocation")
    require(slot_tail, "SparkHeadCertifiedFp8CandidateBytes",
            "full rank-local candidate allocation size")
    project_head = body(module, "SparkDsv4ModuleProjectHead")
    require(project_head, "if ( error == cudaSuccess && rows == 1u )",
            "runtime B1 algorithm selection")
    require(project_head, "SparkDsv4LaunchHeadCertifiedFp8B1Sharded",
            "certified B1 head route")
    require(project_head, "state->head_certified_fp8_norm_f32,slot->head_certified_scratch",
            "dedicated certified scratch launch argument")
    forbid(project_head, "#if", "no compile-time B1 feature fork")
    require(body(dsv4, "SparkDsv4HeadMaxlocPackKernel"),
            "UINT32_MAX - token_ids[row]", "lower-token maxloc tie break")
    require(body(module, "SparkDsv4ModuleReduceHeadMax"),
            "SparkTpDeviceCollectiveSubmitU64Max",
            "generic device max collective")
    require(body(module, "SparkDsv4ModuleInitializeTpCollective"),
            "configuration.combine_u64_max_function = SparkDsv4ModuleCombineU64Max",
            "hidden-transport maxloc reduction")
    require(body(module, "SparkDsv4ModuleResolvedShape"),
            "state->vocabulary_rows_per_rank", "rank-local head pack shape")
    require(body(module, "SparkDsv4ModuleLaunchTpFinalIsland"),
            "state->participates_final_head", "all-rank final-head work")

    print("PASS DSV4 native SM121 compute source contract")
    print("  static only: live sm_121a compile/PTX and GA tensors remain required")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as error:
        print(f"FAIL {error}", file=sys.stderr)
        raise SystemExit(1)
