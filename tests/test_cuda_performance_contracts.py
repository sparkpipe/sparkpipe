#!/usr/bin/env python3

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CUDA_ROOTS = (
    ROOT / "inference",
    ROOT / "runtime",
    ROOT / "modules",
)


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def require(text: str, needle: str, label: str) -> None:
    if needle not in text:
        raise AssertionError(f"{label} is missing {needle!r}")


def forbid(text: str, needle: str, label: str) -> None:
    if needle in text:
        raise AssertionError(f"{label} contains forbidden {needle!r}")


def strip_comments_and_literals(text: str) -> str:
    output: list[str] = []
    index = 0
    state = "code"
    quote = ""

    while index < len(text):
        current = text[index]
        following = text[index + 1] if index + 1 < len(text) else ""
        if state == "code":
            if current == "/" and following == "/":
                output.extend("  ")
                index += 2
                state = "line_comment"
                continue
            if current == "/" and following == "*":
                output.extend("  ")
                index += 2
                state = "block_comment"
                continue
            if current in ("'", '"'):
                quote = current
                output.append(" ")
                index += 1
                state = "literal"
                continue
            output.append(current)
            index += 1
            continue
        if state == "line_comment":
            if current == "\n":
                output.append("\n")
                state = "code"
            else:
                output.append(" ")
            index += 1
            continue
        if state == "block_comment":
            if current == "*" and following == "/":
                output.extend("  ")
                index += 2
                state = "code"
            else:
                output.append("\n" if current == "\n" else " ")
                index += 1
            continue
        if current == "\\" and index + 1 < len(text):
            output.extend("  ")
            index += 2
            continue
        if current == quote:
            output.append(" ")
            index += 1
            state = "code"
            continue
        output.append("\n" if current == "\n" else " ")
        index += 1

    if state == "block_comment":
        raise AssertionError("unterminated block comment")
    if state == "literal":
        raise AssertionError("unterminated source literal")
    return "".join(output)


def validate_balanced_delimiters(path: Path) -> None:
    source = strip_comments_and_literals(path.read_text(encoding="utf-8"))
    opening = {"(": ")", "[": "]", "{": "}"}
    closing = {value: key for key, value in opening.items()}
    stack: list[tuple[str, int]] = []

    for offset, character in enumerate(source):
        if character in opening:
            stack.append((character, offset))
        elif character in closing:
            if not stack or stack[-1][0] != closing[character]:
                raise AssertionError(
                    f"{path.relative_to(ROOT)} has unmatched {character!r} "
                    f"at byte {offset}"
                )
            stack.pop()
    if stack:
        character, offset = stack[-1]
        raise AssertionError(
            f"{path.relative_to(ROOT)} has unmatched {character!r} "
            f"at byte {offset}"
        )


def owned_cuda_files() -> list[Path]:
    files: list[Path] = []
    for root in CUDA_ROOTS:
        if not root.exists():
            continue
        files.extend(root.rglob("*.cu"))
        files.extend(root.rglob("*.cuh"))
    return sorted(set(files))


def validate_scale_abi() -> None:
    scale = read("inference/kernels/scale.cuh")
    gemm = read("inference/kernels/gemm.cuh")
    runtime = read("runtime/gemm.cuh")

    for encoding in (
        "LM_SCALE_ENCODING_NONE",
        "LM_SCALE_ENCODING_F32",
        "LM_SCALE_ENCODING_UE4M3",
        "LM_SCALE_ENCODING_UE8M0",
    ):
        require(scale, encoding, "shared scale ABI")
    for field in (
        "group_stride_entries",
        "row_group_stride_entries",
        "group_count",
        "row_count",
        "input_dimension",
        "row_group_size",
        "k_group_size",
    ):
        require(scale, field, "shared scale tensor")
    require(scale, "(uint64_t)group_index * scale->group_stride_entries", "expert scale indexing")
    require(scale, "row_index / scale->row_group_size", "row-group scale indexing")
    require(scale, "k_index / scale->k_group_size", "K-group scale indexing")
    require(gemm, "LmScaleTensor scale_a;", "GEMM activation scale descriptor")
    require(gemm, "LmScaleTensor scale_b;", "GEMM weight scale descriptor")
    require(runtime, "LmGemmValidateScaleTensor<FormatA>", "activation scale validation")
    require(runtime, "LmGemmValidateScaleTensor<FormatB>", "weight scale validation")

    cast_pattern = re.compile(r"scale_[ab]\s*=\s*\(const\s+float\s*\*\)")
    for path in owned_cuda_files():
        text = path.read_text(encoding="utf-8")
        match = cast_pattern.search(text)
        if match:
            raise AssertionError(
                f"{path.relative_to(ROOT)} reinterprets a scale plane as float*: "
                f"{match.group(0)!r}"
            )


def validate_tma_contract() -> None:
    tma = read("inference/kernels/tma.cuh")
    tile = read("inference/kernels/tile.cuh")
    gemm = read("inference/kernels/gemm.cuh")
    runtime = read("runtime/gemm.cuh")

    tma_code = strip_comments_and_literals(tma)
    forbid(tma_code, "elect.sync", "TMA producer election")
    require(tile, "threadIdx.x == 0u", "CTA-wide producer selection")
    require(gemm, "LmPipelineInitialise<STAGES>(barrier, 1u)", "single-producer mbarrier")
    require(gemm, "__grid_constant__ const CUtensorMap tensor_map_a", "by-value activation tensor map")
    require(gemm, "__grid_constant__ const CUtensorMap tensor_map_b", "by-value weight tensor map")
    require(runtime, "alignas(64) CUtensorMap activation_map;", "activation tensor-map lifetime")
    require(runtime, "alignas(64) CUtensorMap weight_map;", "weight tensor-map lifetime")
    forbid(gemm, "const CUtensorMap *tensor_map", "device pointer to host tensor map")


def validate_quantizer_writes() -> None:
    mma = read("inference/kernels/mma.cuh")
    norm = read("inference/kernels/norm.cuh")

    require(mma, "void LmStoreCodeOctet(", "exclusive packed-code writer")
    require(mma, "const float values[8]", "eight-code ownership")
    require(mma, "memcpy(base + byte, &packed, Format::kStoredBits);", "byte-disjoint packed store")
    require(norm, "index = threadIdx.x * 8u", "eight-code quantizer scheduling")
    forbid(strip_comments_and_literals(norm), "LmStoreCodePair", "overlapping packed pair writer")

    for width in (4, 6, 7, 8):
        mask = (1 << width) - 1
        codes = [((index * 37) + width) & mask for index in range(40)]
        packed = bytearray((len(codes) * width + 7) // 8)
        for begin in range(0, len(codes), 8):
            word = 0
            for lane, code in enumerate(codes[begin:begin + 8]):
                word |= code << (lane * width)
            byte = (begin * width) // 8
            packed[byte:byte + width] = word.to_bytes(8, "little")[:width]
        decoded = []
        bits = int.from_bytes(packed, "little")
        for index in range(len(codes)):
            decoded.append((bits >> (index * width)) & mask)
        if decoded != codes:
            raise AssertionError(f"{width}-bit octet packing did not round-trip")


def validate_model_precision_contracts() -> None:
    glm = read(
        "modules/glm52_resident_decode_stage/source/cuda/layer.cuh"
    )
    glm_unity = read(
        "modules/glm52_resident_decode_stage/source/cuda/unity.cu"
    )
    glm_codec = read("inference/kernels/weight_codec.cuh")
    glm_module = read(
        "modules/glm52_resident_decode_stage/source/"
        "spark_glm52_resident_decode_stage_module.c"
    )
    k3 = read("inference/llms/kimi_k3/layer.cuh")
    qwen_bind = read("inference/llms/qwen_3_6/bind.cu")
    dsv4 = read(
        "modules/dsv4_resident_decode_stage/source/"
        "spark_dsv4_resident_decode_stage_module.c"
    )
    dsv4_cuda = read(
        "modules/dsv4_resident_decode_stage/source/"
        "spark_dsv4_resident_decode_stage_cuda.cu"
    )
    dsv4_model = read(
        "model-families/dsv4/include/sparkpipe/spark_dsv4_model.h"
    )

    require(
        glm,
        "LmGemmWeightOnlyIndirectLaunch<\n        ExpertFormat,",
        "GLM package-selected expert weights",
    )
    require(
        glm,
        "typename LmWeightCodec<ExpertCodec>::Format",
        "GLM compile-time expert codec",
    )
    require(
        glm_unity,
        '#error "GLM52_EXPERT_WEIGHT_CODEC must name the exact package expert codec"',
        "GLM explicit codec build gate",
    )
    require(
        glm_module,
        "context->expert_weight_codec != GLM52_EXPERT_WEIGHT_CODEC",
        "GLM package/module codec equality",
    )
    for codec in (
        "INT6",
        "INT7",
        "INT8",
        "FP8_E4M3",
        "NVFP4_E2M1",
        "MXFP4_E2M1",
    ):
        require(
            glm_codec,
            f"LM_WEIGHT_CODEC(SPARK_WEIGHT_CODEC_{codec},",
            f"generic {codec} weight codec",
        )
    require(glm, "LmGemmLaunch<\n        LmBf16Format,", "GLM BF16 non-expert execution")
    forbid(glm, "LmQuantiseRowsKernel", "GLM BF16 activation path")
    forbid(glm, "LmGatherRowsKernel", "GLM materialized expert activation gather")
    glm_cuda = read(
        "modules/glm52_resident_decode_stage/source/"
        "spark_glm52_resident_decode_stage_cuda.cu"
    )
    forbid(glm_cuda, "expert_weight_codec", "GLM CUDA runtime codec selection")

    require(k3, "LmGemmWeightOnlyLaunch<", "K3 BF16-activation/MXFP4-weight experts")
    require(k3, "LmScaleTensorBlockUe8m0(", "K3 MXFP4 scale plane")
    require(qwen_bind, "Qwen36LaunchSlice<LmBf16Format>", "Qwen 3.6 BF16 entry point")

    require(dsv4_model, "SPARK_DSV4_MODEL_NON_EXPERT_WEIGHT_CODEC SPARK_WEIGHT_CODEC_FP8_E4M3", "DSV4 FP8 linear codec")
    require(dsv4_model, "SPARK_DSV4_MODEL_EXPERT_WEIGHT_CODEC SPARK_WEIGHT_CODEC_MXFP4_E2M1", "DSV4 MXFP4 expert codec")
    require(dsv4_model, "SPARK_DSV4_MODEL_KV_CACHE_CODEC SPARK_WEIGHT_CODEC_BF16", "DSV4 BF16 KV codec")
    require(dsv4, "SparkDsv4LaunchExpertUp", "DSV4 grouped expert execution")
    require(dsv4_cuda, "LmWeightCodec<SPARK_DSV4_MODEL_EXPERT_WEIGHT_CODEC>::Format", "DSV4 package-selected expert codec")
    require(dsv4_cuda, "LmGemmWeightOnlyIndirectLaunch<SparkDsv4ExpertWeightFormat", "DSV4 indirect grouped up GEMM")
    require(dsv4_cuda, "LmGemmWeightOnlyLaunch<SparkDsv4ExpertWeightFormat", "DSV4 grouped down GEMM")
    forbid(dsv4_cuda, "SparkLmExpertTileAllKernel", "DSV4 runtime-format expert kernel")
    require(dsv4_cuda, "SparkDsv4LaunchQuantSim", "DSV4 checkpoint activation quantization")
    forbid(dsv4, "inference/llms/deepseek_v4", "DSV4 legacy driver dependency")



def validate_k3_exact_replay() -> None:
    layer = read("inference/llms/kimi_k3/layer.cuh")
    slice_source = read("inference/llms/kimi_k3/slice.cuh")
    combined_code = strip_comments_and_literals(layer + "\n" + slice_source)

    require(layer, "float *replay_retention;", "K3 retained decay values")
    require(layer, "float *replay_write_gate;", "K3 retained write gates")
    require(layer, "? b->replay_retention : b->kda_retention", "K3 direct retention capture")
    require(layer, "? b->replay_write_gate : b->kda_write_gate_out", "K3 direct gate capture")
    require(slice_source, "buffers->replay_retention,buffers->replay_write_gate", "K3 exact fold inputs")
    require(slice_source, "K3_KDA_KEY_DIM * K3_KDA_VALUE_DIM * sizeof(float)", "K3 fold shared-state allocation")
    fold_begin = slice_source.index("static int32_t K3FoldAccepted")
    fold_source = strip_comments_and_literals(slice_source[fold_begin:])
    forbid(fold_source, "LmBoundedDecayKernel", "K3 accepted-prefix gate recomputation")
    forbid(fold_source, "LmSigmoidRowsKernel", "K3 accepted-prefix beta recomputation")
    forbid(combined_code, "replay_decay_logit", "raw replay decay logits")
    forbid(combined_code, "replay_beta_logit", "raw replay beta logits")

def validate_grouped_moe_contract() -> None:
    route = read("inference/kernels/route.cuh")
    gemm = read("inference/kernels/gemm.cuh")
    runtime = read("runtime/gemm.cuh")
    glm = read(
        "modules/glm52_resident_decode_stage/source/cuda/layer.cuh"
    )

    require(route, "packed_rows != expected_packed_rows", "route cardinality validation")
    require(route, "LmLaunchGroupedTileM(rows,top_k,EXPERTS)", "token-priced grouped tile")
    # The gather double-touch is retired by reading A rows through
    # route_source_token; the consumer contract and the named mapping are
    # what the gather4 GEMM wave builds against.
    require(route, "ROUTE ROW INDIRECTION CONSUMER CONTRACT", "route row indirection contract")
    require(route, "LmRouteSourceRow", "named route indirection mapping")
    require(gemm, "const uint32_t *source_row_map;", "GEMM source-row map")
    require(gemm, "LmPipelineProduceIndirectA<FormatA>", "GEMM direct indexed activation stage")
    require(runtime, "LmGemmWeightOnlyIndirectLaunch(", "weight-only indirect launcher")
    require(glm, "gemm.source_row_map = buffers->route_source_token;", "GLM route-to-GEMM binding")
    forbid(glm, "LmGatherRowsKernel", "GLM gathered activation buffer")


def validate_stream_ordered_dispatch() -> None:
    module = read(
        "modules/glm52_resident_decode_stage/source/"
        "spark_glm52_resident_decode_stage_module.c"
    )
    cuda = read(
        "modules/glm52_resident_decode_stage/source/"
        "spark_glm52_resident_decode_stage_cuda.cu"
    )
    adapter = read(
        "modules/glm52_resident_decode_stage/source/"
        "spark_glm52_serving_adapter.c"
    )

    require(module, "cudaHostAlloc(", "persistent pinned request metadata")
    require(module, "cudaLaunchHostFunc(", "stream-ordered stage completion")
    require(module, "SparkStageModuleSlotRelease", "callback-owned slot release")
    require(module, "host_callback_completion_count", "callback completion telemetry")
    require(
        adapter,
        "SPARK_MODEL_DRIVER_PROGRAM_FLAG_EXTERNAL_COMPLETION",
        "external completion adapter contract",
    )
    forbid(cuda, "cudaStreamSynchronize(", "successful CUDA wave synchronization")
    if module.count("cudaStreamSynchronize(") != 2:
        raise AssertionError(
            "GLM host module synchronization must be limited to failed-enqueue "
            "cleanup and teardown"
        )

    dsv4_module = read(
        "modules/dsv4_resident_decode_stage/source/"
        "spark_dsv4_resident_decode_stage_module.c"
    )
    dsv4_adapter = read(
        "modules/dsv4_resident_decode_stage/source/"
        "spark_dsv4_serving_adapter.c"
    )
    require(dsv4_module, "cudaHostAlloc(", "DSV4 pinned output staging")
    require(dsv4_module, "cudaLaunchHostFunc(", "DSV4 stream-ordered completion")
    require(dsv4_module, "SparkStageModuleSlotRelease", "DSV4 callback slot release")
    require(dsv4_module, "host_callback_completion_count", "DSV4 callback telemetry")
    require(dsv4_module, "SparkDsv4LaunchBuildAttentionIndices", "DSV4 device index assembly")
    require(dsv4_module, "SparkDsv4LaunchCacheScatter", "DSV4 batched cache scatter")
    forbid(dsv4_module, "host_topk_indices", "DSV4 host top-k matrix")
    require(
        dsv4_adapter,
        "SPARK_MODEL_DRIVER_PROGRAM_FLAG_EXTERNAL_COMPLETION",
        "DSV4 external completion adapter contract",
    )
    if dsv4_module.count("cudaStreamSynchronize(") != 4:
        raise AssertionError(
            "DSV4 synchronization must be limited to three initialization "
            "gates and failed-enqueue cleanup"
        )


def validate_head_selection_contract() -> None:
    head = read("inference/kernels/head.cuh")

    # Sampled decoding must not stream full logits back: the chunked top-k
    # emits per-tile partials and commits rows * K * 8 bytes. The full-logits
    # variant stays for callers that need the distribution.
    require(head, "LmHeadTopkCandidateKernel", "chunked top-k partial pass")
    require(head, "LmHeadTopkCommitKernel", "chunked top-k commit pass")
    require(head, "LmHeadTopk", "top-k launcher")
    require(head, "score == best && token < best_token", "deterministic top-k tie rule")
    require(head, "LmHeadSoftmaxKernel", "full-logits variant retained")


def main() -> int:
    files = owned_cuda_files()
    if not files:
        raise AssertionError("no owned CUDA translation units were found")
    for path in files:
        validate_balanced_delimiters(path)
    validate_scale_abi()
    validate_tma_contract()
    validate_quantizer_writes()
    validate_k3_exact_replay()
    validate_model_precision_contracts()
    validate_grouped_moe_contract()
    validate_stream_ordered_dispatch()
    validate_head_selection_contract()
    print(
        f"PASS CUDA performance source contracts: {len(files)} owned CUDA files"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
