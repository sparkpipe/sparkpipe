#!/usr/bin/env python3

import heapq
import math
import random
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(text: str, fragment: str, label: str) -> None:
    if fragment not in text:
        raise AssertionError(f"{label} lacks required fragment {fragment!r}")


def forbid(text: str, fragment: str, label: str) -> None:
    if fragment in text:
        raise AssertionError(f"{label} contains forbidden fragment {fragment!r}")


def extract_braced_definition(text: str, signature: str) -> str:
    start = text.index(signature)
    brace = text.index("{", start)
    depth = 0
    index = brace
    while index < len(text):
        character = text[index]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
        index += 1
    raise AssertionError(f"unterminated definition for {signature!r}")


def strip_comments_and_literals(text: str) -> str:
    output: list[str] = []
    index = 0
    state = "code"
    while index < len(text):
        character = text[index]
        following = text[index + 1] if index + 1 < len(text) else ""
        if state == "code":
            if character == "/" and following == "/":
                output.extend("  ")
                index += 2
                state = "line_comment"
                continue
            if character == "/" and following == "*":
                output.extend("  ")
                index += 2
                state = "block_comment"
                continue
            if character == '"':
                output.append(" ")
                index += 1
                state = "string"
                continue
            if character == "'":
                output.append(" ")
                index += 1
                state = "character"
                continue
            output.append(character)
            index += 1
            continue
        if state == "line_comment":
            if character == "\n":
                output.append("\n")
                state = "code"
            else:
                output.append(" ")
            index += 1
            continue
        if state == "block_comment":
            if character == "*" and following == "/":
                output.extend("  ")
                index += 2
                state = "code"
            else:
                output.append("\n" if character == "\n" else " ")
                index += 1
            continue
        if character == "\\":
            output.append(" ")
            if index + 1 < len(text):
                output.append("\n" if text[index + 1] == "\n" else " ")
                index += 2
            else:
                index += 1
            continue
        if state == "string" and character == '"':
            output.append(" ")
            index += 1
            state = "code"
            continue
        if state == "character" and character == "'":
            output.append(" ")
            index += 1
            state = "code"
            continue
        output.append("\n" if character == "\n" else " ")
        index += 1
    if state in {"block_comment", "string", "character"}:
        raise AssertionError(f"unterminated lexical state {state}")
    return "".join(output)


def validate_balanced_delimiters(path: Path) -> None:
    text = strip_comments_and_literals(path.read_text(encoding="utf-8"))
    opening = {"(": ")", "[": "]", "{": "}"}
    closing = {value: key for key, value in opening.items()}
    stack: list[tuple[str, int]] = []
    for index, character in enumerate(text):
        if character in opening:
            stack.append((character, index))
        elif character in closing:
            if not stack or stack[-1][0] != closing[character]:
                raise AssertionError(f"{path}: unmatched {character!r} at byte {index}")
            stack.pop()
    if stack:
        character, index = stack[-1]
        raise AssertionError(f"{path}: unmatched {character!r} at byte {index}")


def float32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def ordered_topk_key(score: float, slot: int) -> int:
    score = float32(score)
    if math.isnan(score) or score <= float32(-3.0e38):
        return 0
    if score == 0.0:
        score = 0.0
    bits = struct.unpack("<I", struct.pack("<f", score))[0]
    ordered = bits ^ (0xFFFFFFFF if bits & 0x80000000 else 0x80000000)
    return (ordered << 32) | (0xFFFFFFFF - slot)


def radix_select_reference(scores: list[float], topk: int) -> list[int]:
    keys = [(ordered_topk_key(score, slot), slot) for slot, score in enumerate(scores)]
    keys = [entry for entry in keys if entry[0] != 0]
    keys.sort(key=lambda entry: entry[0], reverse=True)
    return [slot for _, slot in keys[:topk]]


def radix_select_emulation(scores: list[float], topk: int) -> list[int]:
    keys = [ordered_topk_key(score, slot) for slot, score in enumerate(scores)]
    selected_count = min(topk, sum(key != 0 for key in keys))
    if selected_count == 0:
        return []
    remaining_rank = selected_count - 1
    prefix = 0
    prefix_mask = 0
    for radix_pass in range(8):
        shift = 56 - 8 * radix_pass
        histogram = [0] * 256
        for key in keys:
            if key != 0 and (key & prefix_mask) == prefix:
                histogram[(key >> shift) & 0xFF] += 1
        for digit in range(255, -1, -1):
            if remaining_rank < histogram[digit]:
                prefix |= digit << shift
                break
            remaining_rank -= histogram[digit]
        prefix_mask |= 0xFF << shift
    selected = [key for key in keys if key != 0 and key >= prefix]
    selected.sort(reverse=True)
    return [0xFFFFFFFF - (key & 0xFFFFFFFF) for key in selected[:selected_count]]


def validate_radix_selection() -> None:
    generator = random.Random(0x505121A)
    edge_scores = [
        float("nan"),
        -float("inf"),
        -3.0e38,
        -0.0,
        0.0,
        1.0,
        1.0,
        -1.0,
        float("inf"),
    ]
    for topk in range(1, 10):
        expected = radix_select_reference(edge_scores, topk)
        actual = radix_select_emulation(edge_scores, topk)
        if actual != expected:
            raise AssertionError(f"radix edge mismatch topk={topk}: {actual} != {expected}")
    for count in (1, 2, 7, 31, 257, 1024):
        for topk in (1, min(3, count), min(16, count), min(128, count)):
            scores = []
            for _ in range(count):
                selector = generator.randrange(32)
                if selector == 0:
                    scores.append(float("nan"))
                elif selector == 1:
                    scores.append(-3.0e38)
                elif selector < 8:
                    scores.append(float(generator.randrange(-4, 5)))
                else:
                    scores.append(generator.uniform(-20.0, 20.0))
            expected = radix_select_reference(scores, topk)
            actual = radix_select_emulation(scores, topk)
            if actual != expected:
                raise AssertionError(
                    f"radix random mismatch count={count} topk={topk}: "
                    f"{actual[:16]} != {expected[:16]}"
                )




def bitonic_router_topk_emulation(
    choice_scores: list[float],
    route_weights: list[float],
    topk: int,
    capacity: int,
) -> tuple[list[int], list[float]]:
    keys = [0] * capacity
    weights = [0.0] * capacity
    for expert, score in enumerate(choice_scores):
        if math.isfinite(score):
            keys[expert] = ordered_topk_key(score, expert)
            weights[expert] = float32(route_weights[expert])
    bitonic_size = 2
    while bitonic_size <= capacity:
        bitonic_stride = bitonic_size >> 1
        while bitonic_stride:
            previous = keys.copy()
            for expert_index in range(capacity):
                partner_index = expert_index ^ bitonic_stride
                if partner_index > expert_index:
                    current_key = previous[expert_index]
                    partner_key = previous[partner_index]
                    ascending = (expert_index & bitonic_size) == 0
                    if (ascending and current_key > partner_key) or (
                        not ascending and current_key < partner_key
                    ):
                        keys[expert_index] = partner_key
                        keys[partner_index] = current_key
            bitonic_stride >>= 1
        bitonic_size <<= 1
    selected_keys = [
        keys[capacity - 1 - selected_index]
        for selected_index in range(min(topk, len(choice_scores)))
    ]
    selected_experts = [
        0xFFFFFFFF - (key & 0xFFFFFFFF)
        for key in selected_keys
        if key != 0
    ]
    return selected_experts, [weights[expert] for expert in selected_experts]


def validate_bitonic_router_selection() -> None:
    generator = random.Random(0x505B17)
    for expert_count in (1, 3, 17, 128, 256, 384):
        for topk in (1, min(4, expert_count), min(8, expert_count)):
            choice_scores = []
            route_weights = []
            for expert in range(expert_count):
                selector = generator.randrange(16)
                if selector == 0:
                    choice_scores.append(float("nan"))
                elif selector < 5:
                    choice_scores.append(float(generator.randrange(-3, 4)))
                else:
                    choice_scores.append(generator.uniform(-20.0, 20.0))
                route_weights.append(generator.random())
            expected_entries = [
                (ordered_topk_key(score, expert), expert)
                for expert, score in enumerate(choice_scores)
                if math.isfinite(score)
            ]
            expected_entries.sort(reverse=True)
            expected_experts = [expert for _, expert in expected_entries[:topk]]
            actual_experts, actual_weights = bitonic_router_topk_emulation(
                choice_scores,
                route_weights,
                topk,
                512,
            )
            if actual_experts != expected_experts:
                raise AssertionError(
                    f"bitonic router mismatch experts={expert_count} topk={topk}: "
                    f"{actual_experts} != {expected_experts}"
                )
            expected_weights = [float32(route_weights[expert]) for expert in expected_experts]
            if actual_weights != expected_weights:
                raise AssertionError(
                    f"bitonic router weight mismatch experts={expert_count} topk={topk}"
                )


def hierarchical_topk_emulation(
    scores: list[float],
    topk: int,
    partition_size: int,
) -> list[int]:
    partition_lists: list[list[tuple[int, int]]] = []
    for partition_base in range(0, len(scores), partition_size):
        local_entries = [
            (ordered_topk_key(scores[slot], slot), slot)
            for slot in range(
                partition_base,
                min(partition_base + partition_size, len(scores)),
            )
        ]
        local_entries = [entry for entry in local_entries if entry[0] != 0]
        local_entries.sort(key=lambda entry: entry[0], reverse=True)
        partition_lists.append(local_entries[:topk])

    heap: list[tuple[int, int, int, int]] = []
    for partition_index, entries in enumerate(partition_lists):
        if entries:
            key, slot = entries[0]
            heapq.heappush(heap, (-key, partition_index, 0, slot))
    selected: list[int] = []
    while heap and len(selected) < topk:
        negative_key, partition_index, entry_index, slot = heapq.heappop(heap)
        selected.append(slot)
        next_entry_index = entry_index + 1
        if next_entry_index < len(partition_lists[partition_index]):
            key, next_slot = partition_lists[partition_index][next_entry_index]
            heapq.heappush(
                heap,
                (-key, partition_index, next_entry_index, next_slot),
            )
    return selected


def validate_hierarchical_selection() -> None:
    generator = random.Random(0x505D5A)
    for count in (1, 17, 63, 257, 1025):
        for partition_size in (7, 32, 127):
            for topk in (1, min(3, count), min(16, count), min(128, count)):
                scores: list[float] = []
                for _ in range(count):
                    selector = generator.randrange(24)
                    if selector == 0:
                        scores.append(float("nan"))
                    elif selector == 1:
                        scores.append(-3.0e38)
                    elif selector < 7:
                        scores.append(float(generator.randrange(-3, 4)))
                    else:
                        scores.append(generator.uniform(-30.0, 30.0))
                expected = radix_select_reference(scores, topk)
                actual = hierarchical_topk_emulation(
                    scores,
                    topk,
                    partition_size,
                )
                if actual != expected:
                    raise AssertionError(
                        "hierarchical top-k mismatch "
                        f"count={count} partition={partition_size} topk={topk}: "
                        f"{actual[:16]} != {expected[:16]}"
                    )


def validate_fp8_block_scale_accumulation() -> None:
    generator = random.Random(0xF32B128)
    input_dimension = 512
    output_dimension = 256
    activations = [generator.uniform(-2.0, 2.0) for _ in range(input_dimension)]
    weights = [
        generator.uniform(-1.0, 1.0)
        for _ in range(output_dimension * input_dimension)
    ]
    scales = [
        generator.uniform(0.05, 2.0)
        for _ in range(
            (output_dimension // 128) * (input_dimension // 128)
        )
    ]
    input_block_count = input_dimension // 128
    for neuron in (0, 63, 64, 127, 128, 191, 192, 255):
        reference = 0.0
        for element in range(input_dimension):
            scale = scales[
                (neuron // 128) * input_block_count + (element // 128)
            ]
            reference += (
                activations[element] *
                weights[neuron * input_dimension + element] *
                scale
            )
        tiled = 0.0
        block_accumulator = 0.0
        for k_base in range(0, input_dimension, 64):
            for element in range(k_base, k_base + 64):
                block_accumulator += (
                    activations[element] *
                    weights[neuron * input_dimension + element]
                )
            next_k_base = k_base + 64
            if next_k_base % 128 == 0 or next_k_base >= input_dimension:
                scale_index = (
                    (neuron // 128) * input_block_count + (k_base // 128)
                )
                tiled += block_accumulator * scales[scale_index]
                block_accumulator = 0.0
        tolerance = 1.0e-10 * max(1.0, abs(reference))
        if abs(tiled - reference) > tolerance:
            raise AssertionError(
                f"F32B128 block-scale mismatch neuron={neuron}: "
                f"{tiled} != {reference}"
            )


def validate_moe_grouping() -> None:
    generator = random.Random(0x505E7)
    for row_count in (1, 3, 17):
        experts_per_token = 16
        expert_count = 64
        pair_experts = [
            generator.randrange(expert_count)
            for _ in range(row_count * experts_per_token)
        ]
        grouped_slots = sorted(
            range(len(pair_experts)),
            key=lambda pair: (pair_experts[pair], pair),
        )
        inverse_map = [0] * len(pair_experts)
        for grouped_slot, pair in enumerate(grouped_slots):
            inverse_map[pair] = grouped_slot
        pair_values = [generator.uniform(-4.0, 4.0) for _ in pair_experts]
        grouped_values = [pair_values[pair] for pair in grouped_slots]
        for row in range(row_count):
            direct = pair_values[
                row * experts_per_token:(row + 1) * experts_per_token
            ]
            restored = [
                grouped_values[inverse_map[row * experts_per_token + rank]]
                for rank in range(experts_per_token)
            ]
            if restored != direct:
                raise AssertionError("MoE inverse map failed to restore route order")


def validate_k3_contract() -> None:
    path = ROOT / (
        "modules/k3_resident_decode_stage/source/"
        "spark_k3_resident_decode_stage_cuda.cu"
    )
    text = path.read_text(encoding="utf-8")
    decode_launch = extract_braced_definition(
        text,
        'extern "C" SparkStatus SparkK3LaunchKdaDecodeStep(',
    )
    configure = extract_braced_definition(
        text,
        'extern "C" SparkStatus SparkK3ConfigureCudaKernels(',
    )
    materialize = extract_braced_definition(
        text,
        'extern "C" SparkStatus SparkK3LaunchKdaMaterialize(',
    )
    route = extract_braced_definition(
        text,
        "__global__ void SparkK3MoeRouteKernel(",
    )

    if text.count("SparkK3KdaDecodeStepKernel<<<") != 1:
        raise AssertionError("K3 must enqueue exactly one decode state transition")
    require(
        decode_launch,
        "SPARK_K3_KDA_DECODE_SMEM_FLOATS * sizeof(float)",
        "K3 decode dynamic shared memory",
    )
    if text.count("cudaFuncSetAttribute(") != 4:
        raise AssertionError("K3 must configure exactly four large-shared kernels")
    if configure.count("cudaFuncSetAttribute(") != 4:
        raise AssertionError("K3 kernel attributes escaped one-time configuration")
    forbid(decode_launch, "cudaFuncSetAttribute(", "K3 decode launch")
    require(materialize, "activation_pair_count", "K3 activation launch sizing")
    require(materialize, "activation_block_count", "K3 activation launch sizing")
    forbid(materialize, "<<<256u,SPARK_K3_CTA_THREADS", "K3 activation launch")
    require(route, "SparkLmArgmaxReduce", "K3 parallel router selection")
    forbid(route, "14336 shared reads", "K3 stale serial-router description")

    require(text, "SparkLmHostLaunchBatchedLinear", "K3 size-aware dense linear")
    require(text, "SparkK3LinearBundleKernel", "K3 bundled projection kernel")
    require(text, "SPARK_K3_LINEAR_BUNDLE_OUTPUTS_PER_CTA 64u", "K3 projection tile width")
    require(materialize, "SparkK3LaunchLinearBundle(&input_bundle", "K3 bundled input projections")
    require(materialize, "SparkK3LaunchLinearBundle(&low_rank_bundle", "K3 bundled low-rank projections")
    require(text, "SparkLmHostLaunchMoeGroup", "K3 device expert grouping")
    require(text, "SparkK3MoeGroupedExpertInterKernel", "K3 grouped expert execution")
    require(text, "slot->moe_inverse_map", "K3 grouped expert inverse map")


def validate_qwen_contract() -> None:
    path = ROOT / (
        "modules/qwen36_resident_decode_stage/source/"
        "spark_qwen36_resident_decode_stage_cuda.cu"
    )
    text = path.read_text(encoding="utf-8")
    prepare = extract_braced_definition(
        text,
        "static __global__ void SparkQwen36AttnPrepareKernel(",
    )
    attention = extract_braced_definition(
        text,
        "static __global__ void SparkQwen36AttnDecodeKernel(",
    )
    gdn = extract_braced_definition(
        text,
        "static __global__ void SparkQwen36GdnStepKernel(",
    )
    qk_decay = extract_braced_definition(
        text,
        "static __global__ void SparkQwen36ChunkQkDecayKernel(",
    )
    chunk_transform = extract_braced_definition(
        text,
        "static __global__ void SparkQwen36ChunkTransformKernel(",
    )
    chunk_step = extract_braced_definition(
        text,
        "static __global__ void SparkQwen36ChunkStepKernel(",
    )
    configure = extract_braced_definition(
        text,
        'extern "C" cudaError_t SparkQwen36ConfigureCudaKernels(',
    )
    chunk_launch = extract_braced_definition(
        text,
        'extern "C" cudaError_t SparkQwen36LaunchGdnChunk(',
    )

    require(prepare, "query_shared", "Qwen normalized query staging")
    require(prepare, "key_shared", "Qwen normalized key staging")
    require(prepare, "sincosf", "Qwen shared RoPE trigonometry")
    require(
        prepare,
        "query_shared[column]",
        "Qwen single-write normalized query",
    )
    require(
        prepare,
        "key_shared[column]",
        "Qwen single-write normalized key",
    )
    require(text, "SPARK_QWEN36_CUDA_ATTN_VALUE_PAIRS_PER_LANE", "Qwen full-width attention")
    value_pair_loop_count = (
        attention.count("pair < SPARK_QWEN36_CUDA_ATTN_VALUE_PAIRS_PER_LANE") +
        attention.count("pair_index < SPARK_QWEN36_CUDA_ATTN_VALUE_PAIRS_PER_LANE")
    )
    if value_pair_loop_count < 3:
        raise AssertionError("Qwen attention must initialize, accumulate, and publish every value pair")
    require(attention, "heads_per_cta = SPARK_QWEN36_CUDA_ATTN_HEADS_PER_CTA", "Qwen grouped KV reuse")
    require(attention, "float local_logit[SPARK_QWEN36_CUDA_ATTN_HEADS_PER_CTA]", "Qwen grouped QK accumulation")
    require(
        attention,
        "SparkLmWarpReduceSum(local_logit[local_head])",
        "Qwen complete QK warp reduction",
    )
    require(attention, "available_block_count", "Qwen device block-table bound")
    require(attention, "required_block_count", "Qwen required block count")
    forbid(attention, "(void)block_counts", "Qwen ignored block-count metadata")
    require(attention, "if (lane == 0u)", "Qwen lane-zero online softmax")
    require(attention, "rescale[local_head] = __shfl_sync", "Qwen softmax scale broadcast")
    if attention.count("value_pair = SparkLmLoadBf16Pair(") != 1:
        raise AssertionError("Qwen grouped attention must load each V pair once per token")
    if gdn.count("state_f32[state_base + state_index]") != 2:
        raise AssertionError("Qwen decode GDN must read and write each state element once")
    require(gdn, "extern __shared__ float state_shared[]", "Qwen decode state staging")
    require(qk_decay, "extern __shared__ float qk_shared[]", "Qwen chunk QK staging")
    require(qk_decay, "qn_shared[vector_element] = views.qn", "Qwen chunk Q staging")
    require(qk_decay, "kn_shared[vector_element] = views.kn", "Qwen chunk K staging")
    forbid(qk_decay, "fmaf(views.qn[", "Qwen repeated global Q reads")
    forbid(qk_decay, "views.kn[vec_base + ((uint64_t)column", "Qwen repeated global K reads")
    forbid(chunk_step, "float v_new[", "Qwen chunk register spill array")
    require(chunk_step, "v_new_shared", "Qwen chunk value staging")
    require(chunk_step, "state_shared", "Qwen chunk state staging")
    require(
        text,
        "SPARK_QWEN36_CUDA_GDN_STATE_ELEMENTS +",
        "Qwen chunk state shared layout",
    )
    require(
        text,
        "SPARK_QWEN36_CUDA_CHUNK * SPARK_QWEN36_CUDA_DV",
        "Qwen chunk value shared layout",
    )
    require(
        text,
        "2u * SPARK_QWEN36_CUDA_CHUNK",
        "Qwen cached chunk-decay shared layout",
    )
    require(
        text,
        "SPARK_QWEN36_CUDA_GDN_CHUNK_SHARED_BYTES == 98816u",
        "Qwen exact chunk shared-memory assertion",
    )
    require(
        configure,
        "SPARK_QWEN36_CUDA_GDN_CHUNK_SHARED_BYTES",
        "Qwen configured chunk shared bytes",
    )
    require(
        chunk_launch,
        "SPARK_QWEN36_CUDA_GDN_CHUNK_SHARED_BYTES",
        "Qwen launched chunk shared bytes",
    )
    require(
        chunk_transform,
        "__shared__ float exp_cum_g[SPARK_QWEN36_CUDA_CHUNK]",
        "Qwen transform exponent cache",
    )
    require(
        chunk_transform,
        "exp_cum_g[element]",
        "Qwen transform exponent-cache reuse",
    )
    forbid(
        chunk_transform,
        "__expf(views.cum_g[SparkQwen36ChunkHeadOffset",
        "Qwen repeated transform exponentials",
    )
    require(
        chunk_step,
        "exp_cum_g_shared =",
        "Qwen step exponent-cache layout",
    )
    require(
        chunk_step,
        "carry_decay_shared =",
        "Qwen step carry-decay cache layout",
    )
    require(
        chunk_step,
        "exp_cum_g_shared[token_count - 1u]",
        "Qwen cached final chunk decay",
    )
    require(
        chunk_step,
        "carry_decay_shared[row]",
        "Qwen cached state carry decay",
    )
    forbid(
        chunk_step,
        "__expf(views.cum_g[g_base + row])",
        "Qwen repeated output decay exponentials",
    )
    forbid(chunk_step, "__expf(g_last)", "Qwen repeated final decay exponentials")
    forbid(
        chunk_step,
        "__expf(g_last - views.cum_g[g_base + row])",
        "Qwen repeated state-carry exponentials",
    )
    transform_cache_write = chunk_transform.index(
        "exp_cum_g[threadIdx.x] = __expf("
    )
    transform_barrier = chunk_transform.index("__syncthreads()", transform_cache_write)
    transform_cache_read = chunk_transform.index(
        "exp_cum_g[element]", transform_barrier
    )
    if not transform_cache_write < transform_barrier < transform_cache_read:
        raise AssertionError("Qwen transform exponent cache is not synchronized")
    step_cache_write = chunk_step.index(
        "exp_cum_g_shared[column] = __expf("
    )
    step_barrier = chunk_step.index("__syncthreads()", step_cache_write)
    step_cache_read = chunk_step.index(
        "exp_cum_g_shared[row]", step_barrier
    )
    if not step_cache_write < step_barrier < step_cache_read:
        raise AssertionError("Qwen step exponent cache is not synchronized")
    if configure.count("cudaFuncSetAttribute(") != 3:
        raise AssertionError("Qwen must configure all three large-shared kernels once")
    required_launch_order = [
        "SparkQwen36ChunkPrepareKernel<<<",
        "SparkQwen36ChunkSolveKernel<<<",
        "SparkQwen36ChunkTransformKernel<<<",
        "SparkQwen36ChunkQkDecayKernel<<<",
        "SparkQwen36ChunkStepKernel<<<",
    ]
    positions = [chunk_launch.index(fragment) for fragment in required_launch_order]
    if positions != sorted(positions):
        raise AssertionError("Qwen chunk stages are not stream-ordered in dependency order")
    if chunk_launch.count("cudaGetLastError()") != 5:
        raise AssertionError("Qwen chunk launch must report every stage error")
    require(
        chunk_launch,
        "SPARK_QWEN36_CUDA_GDN_QK_SHARED_BYTES",
        "Qwen chunk QK dynamic shared memory",
    )


def validate_shared_kernel_contract() -> None:
    path = ROOT / "model-families/common/include/sparkpipe/spark_lm_kernels.cuh"
    text = path.read_text(encoding="utf-8")
    attention_score = extract_braced_definition(
        text,
        "static __device__ __forceinline__ float SparkLmAttnKeyLogit(",
    )
    rescore = extract_braced_definition(
        text,
        "static __global__ void SparkLmHeadRescoreArgmaxKernel(",
    )
    screened_launch = extract_braced_definition(
        text,
        "static inline cudaError_t SparkLmHostLaunchHeadScreenedArgmax(",
    )
    rms = extract_braced_definition(text, "static __global__ void SparkLmRmsNormKernel(")
    fused_rms = extract_braced_definition(
        text,
        "static __global__ void SparkLmFusedResidualRmsNormKernel(",
    )

    require(attention_score, "__shfl_sync", "shared complete QK warp reduction")
    require(rms, "extern __shared__ float staged_input[]", "shared RMSNorm input staging")
    require(fused_rms, "extern __shared__ float staged_hidden[]", "shared residual RMSNorm staging")
    rms_driver_paths = (
        ROOT / "modules/dsv4_resident_decode_stage/source/spark_dsv4_resident_decode_stage_cuda.cu",
        ROOT / "modules/k3_resident_decode_stage/source/spark_k3_resident_decode_stage_cuda.cu",
        ROOT / "modules/mimo25_resident_decode_stage/source/spark_mimo25_resident_decode_stage_cuda.cu",
        ROOT / "modules/qwen36_resident_decode_stage/source/spark_qwen36_resident_decode_stage_cuda.cu",
    )
    for driver_path in rms_driver_paths:
        driver_text = driver_path.read_text(encoding="utf-8")
        if "SparkLmRmsNormKernel<<<" in driver_text:
            if "SparkLmRmsNormKernel<<<row_count,SPARK_LM_CTA_THREADS,0," in driver_text:
                raise AssertionError(
                    f"{driver_path.name} launches staged RMSNorm with zero dynamic shared memory"
                )
        if "SparkLmFusedResidualRmsNormKernel<<<" in driver_text:
            if "SparkLmFusedResidualRmsNormKernel<<<row_count,SPARK_LM_CTA_THREADS,0," in driver_text:
                raise AssertionError(
                    f"{driver_path.name} launches staged fused RMSNorm with zero dynamic shared memory"
                )
    require(rescore, "candidate_counts[row]", "single screened/full-vocabulary rescore")
    if screened_launch.count("SparkLmHeadRescoreArgmaxKernel<<<") != 1:
        raise AssertionError("screened vocabulary head must enqueue one exact-rescore kernel")
    require(text, "SparkLmHostLaunchMoePairReduceOverwrite", "overwrite MoE reduction")
    require(text, "SparkLmExpertTileBodyAllWarps", "all-warp tensor tile variant")
    require(
        text,
        "SparkLmExpertTileBodySoftwarePipelined",
        "software-pipelined tensor tile variant",
    )
    require(text, "SPARK_LM_EXPERT_TILE_POLICY_AUTOMATIC", "tensor tile policy")
    forbid(
        text,
        "#ifndef SPARK_LM_EXPERT_TILE_POLICY",
        "tensor tile policy",
    )
    require(text, "producer_warp_base", "shared tensor-tile producer warps")
    require(text, "tile_input[2u]", "shared tensor-tile ping-pong input")
    require(text, "tile_weight[2u]", "shared tensor-tile ping-pong weights")


def validate_dsv4_contract() -> None:
    path = ROOT / (
        "modules/dsv4_resident_decode_stage/source/"
        "spark_dsv4_resident_decode_stage_cuda.cu"
    )
    text = path.read_text(encoding="utf-8")
    indexer = extract_braced_definition(text, "static __global__ void SparkDsv4IndexerScoreKernel(")
    topk = extract_braced_definition(text, "static __global__ void SparkDsv4TopKKernel(")
    topk_launch = extract_braced_definition(
        text,
        'extern "C" cudaError_t SparkDsv4LaunchTopK(',
    )
    hadamard_launch = extract_braced_definition(
        text,
        'extern "C" cudaError_t SparkDsv4LaunchHadamard(',
    )
    route = extract_braced_definition(
        text,
        "static __global__ void SparkDsv4GateSelectKernel(",
    )
    shared_bytes = extract_braced_definition(
        text,
        "static size_t SparkDsv4SparseAttnSharedBytes(",
    )
    configure_cuda = extract_braced_definition(
        text,
        'extern "C" cudaError_t SparkDsv4ConfigureCudaKernels(',
    )
    launch_sparse_attention = extract_braced_definition(
        text,
        'extern "C" cudaError_t SparkDsv4LaunchSparseAttn(',
    )
    module_path = ROOT / (
        "modules/dsv4_resident_decode_stage/source/"
        "spark_dsv4_resident_decode_stage_module.c"
    )
    module_text = module_path.read_text(encoding="utf-8")
    initialize = extract_braced_definition(
        module_text,
        "SparkStatus SparkDsv4ResidentDecodeStageInitialize(",
    )

    require(indexer, "extern __shared__ float q_shared[]", "DSV4 indexer Q staging")
    require(indexer, "float2 key_pair[maximum_pairs_per_lane]", "DSV4 indexer register key reuse")
    forbid(indexer, "key_shared", "DSV4 repeated shared-key sweep")
    require(
        indexer,
        "lane == 0u && accumulator > 0.0f",
        "DSV4 lane-zero weighted accumulation",
    )
    require(topk, "for (pass = 0u; pass < 8u; pass++)", "DSV4 byte-radix selection")
    require(topk, "__match_any_sync", "DSV4 grouped radix histogram")
    require(topk, "__activemask()", "DSV4 tail-safe radix mask")
    require(indexer, "slot_counts[row] < max_slots", "DSV4 indexer slot bound")
    require(topk, "slot_counts[row] < max_slots", "DSV4 selector slot bound")
    forbid(topk, "scores[slot] =", "DSV4 destructive top-k marking")
    require(topk_launch, "const float *scores_f32", "DSV4 immutable score buffer")
    require(hadamard_launch, "(width & (width - 1u)) != 0u", "DSV4 Hadamard power-of-two contract")
    require(hadamard_launch, "width > SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION", "DSV4 Hadamard bound")
    require(route, "SparkLmBitonicSortKeysAscending", "DSV4 one-sort gate top-k")
    require(route, "SparkLmOrderedTopKKey", "DSV4 canonical gate ordering")
    forbid(route, "SparkLmArgmaxReduce", "DSV4 repeated-rank gate selection")
    forbid(route, "for (rank = 0u; rank < topk", "DSV4 repeated-rank gate selection")
    require(
        initialize,
        "SparkDsv4ConfigureCudaKernels()",
        "DSV4 initialization-time large-shared kernel configuration",
    )
    if initialize.count("SparkDsv4ConfigureCudaKernels()") != 1:
        raise AssertionError("DSV4 initialization must configure large-shared kernels once")
    module_configure_offset = initialize.find("SparkDsv4ModuleConfigure(state)")
    module_configure_failure_offset = initialize.find(
        "if (status != SPARK_STATUS_OK)",
        module_configure_offset,
    )
    atomic_init_offset = initialize.find(
        "SparkStageModuleAtomicStateArrayInitialize("
    )
    if (
        module_configure_offset < 0
        or module_configure_failure_offset < 0
        or atomic_init_offset < 0
        or not (
            module_configure_offset
            < module_configure_failure_offset
            < atomic_init_offset
        )
    ):
        raise AssertionError(
            "DSV4 configuration failures must return before atomic-array cleanup is possible"
        )
    module_configure_failure = initialize[
        module_configure_failure_offset:atomic_init_offset
    ]
    require(
        module_configure_failure,
        "free(state)",
        "DSV4 pre-resource configuration failure cleanup",
    )
    require(
        module_configure_failure,
        "return status",
        "DSV4 pre-resource configuration failure return",
    )
    configure_offset = initialize.find("SparkDsv4ConfigureCudaKernels()")
    if atomic_init_offset < 0 or configure_offset < 0 or atomic_init_offset > configure_offset:
        raise AssertionError(
            "DSV4 slot atomics must be initialized before fail-closed CUDA configuration"
        )
    require(
        configure_cuda,
        "SparkDsv4SparseAttnKernel",
        "DSV4 configured sparse-attention kernel",
    )
    require(
        configure_cuda,
        "cudaFuncAttributeMaxDynamicSharedMemorySize",
        "DSV4 sparse-attention dynamic-shared opt-in",
    )
    require(
        configure_cuda,
        "SparkDsv4SparseAttnSharedBytes(",
        "DSV4 configured sparse-attention shared bytes",
    )
    require(
        launch_sparse_attention,
        "SparkDsv4SparseAttnSharedBytes(head_dim)",
        "DSV4 launched sparse-attention shared bytes",
    )
    forbid(
        shared_bytes,
        "sizeof(__nv_bfloat16)",
        "DSV4 stale sparse-attention shared allocation",
    )
    require(
        shared_bytes,
        "(size_t)heads_per_cta * head_dimension * sizeof(float)",
        "DSV4 sparse-attention query staging",
    )
    require(
        shared_bytes,
        "(size_t)heads_per_cta * SPARK_LM_CTA_WARPS *",
        "DSV4 sparse-attention reduction staging",
    )
    if shared_bytes.count("sizeof(float)") != 2:
        raise AssertionError("DSV4 sparse-attention shared layout must contain two float regions")


# The guarantee is that a decode never rereads KV per query head. Which grouping
# width delivers that is the dispatcher's business, so assert on the dispatcher
# rather than pinning one template argument at a call site.
def validate_adaptive_attn_dispatch() -> None:
    path = ROOT / "model-families/common/include/sparkpipe/spark_lm_kernels.cuh"
    dispatch = extract_braced_definition(
        path.read_text(encoding="utf-8"),
        "static inline cudaError_t SparkLmHostLaunchAdaptiveAttnDecode(",
    )
    require(dispatch, "SparkLmHostLaunchGroupedAttnDecode<4u>", "adaptive attn widest grouping")
    forbid(dispatch, "SparkLmAttnDecodeKernel", "adaptive attn per-query-head KV reread")
    if "SparkLmHostLaunchGroupedAttnDecode<" not in dispatch:
        raise AssertionError("adaptive attn dispatch must reach only grouped kernels")


def validate_mimo_contract() -> None:
    path = ROOT / (
        "modules/mimo25_resident_decode_stage/source/"
        "spark_mimo25_resident_decode_stage_cuda.cu"
    )
    text = path.read_text(encoding="utf-8")
    pair_reduce = extract_braced_definition(
        text,
        'extern "C" cudaError_t SparkMimo25LaunchMoePairReduce(',
    )
    attention = extract_braced_definition(
        text,
        'extern "C" cudaError_t SparkMimo25LaunchAttnDecode(',
    )
    route = extract_braced_definition(
        text,
        "static __global__ void SparkMimo25GateSelectKernel(",
    )
    module_path = ROOT / (
        "modules/mimo25_resident_decode_stage/source/"
        "spark_mimo25_resident_decode_stage_module.c"
    )
    module_text = module_path.read_text(encoding="utf-8")
    run_ffn = extract_braced_definition(
        module_text,
        "static SparkStatus SparkMimo25ModuleRunFfn(",
    )
    require(pair_reduce, "SparkLmHostLaunchMoePairReduceOverwrite", "MiMo overwrite reduction")
    require(attention, "SparkLmHostLaunchAdaptiveAttnDecode", "MiMo grouped KV reuse")
    validate_adaptive_attn_dispatch()
    forbid(attention, "SparkLmAttnDecodeKernel<<<", "MiMo per-query-head KV reread")
    forbid(pair_reduce, "cudaMemsetAsync", "MiMo zero-then-read accumulation")
    forbid(run_ffn, "cudaMemsetAsync", "MiMo dead routed-FFN clear")
    require(route, "SparkLmBitonicSortKeysAscending", "MiMo one-sort gate top-k")
    require(route, "SparkLmOrderedTopKKey", "MiMo canonical gate ordering")
    forbid(route, "SparkLmArgmaxReduce", "MiMo repeated-rank gate selection")
    forbid(route, "for (rank = 0u; rank < topk", "MiMo repeated-rank gate selection")


def validate_glm_contract() -> None:
    exact_path = ROOT / (
        "modules/glm52_resident_decode_stage/source/"
        "spark_glm52_sm121_required_decode_stage.cu"
    )
    exact_text = exact_path.read_text(encoding="utf-8")
    selector = extract_braced_definition(
        exact_text,
        "void SparkGlm52ResidentDecodeStageDsaSelectRadixTopkKernel(",
    )
    require(
        exact_text,
        "SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_RADIX_BITS_PER_PASS 8u",
        "GLM byte-radix selector",
    )
    require(selector, "__match_any_sync", "GLM grouped radix histogram")
    require(selector, "shared_selected_keys", "GLM canonical selected-key sort")
    forbid(selector, "dsa_token_scores[candidate_index] =", "GLM destructive selector")
    require(exact_text, "#define SPARK_GLM52_ENABLE_STAGE_PHASE_HASH 0", "GLM release hash default")
    require(
        exact_text,
        "SparkGlm52ResidentDecodeStageDsaScoreSelectHierarchicalKernel",
        "GLM fused hierarchical score-select",
    )
    require(
        exact_text,
        "SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_HIERARCHICAL_PARTITION_CANDIDATES",
        "GLM hierarchical partitioning",
    )
    require(
        exact_text,
        "SparkGlm52ResidentDecodeStageDsaMergeHierarchicalTopkKernel",
        "GLM compact hierarchical merge",
    )
    require(
        exact_text,
        "SparkGlm52ResidentDecodeStageDsaHierarchicalMergeSharedStorage",
        "GLM heap-based hierarchical merge",
    )
    merge = extract_braced_definition(
        exact_text,
        "void SparkGlm52ResidentDecodeStageDsaMergeHierarchicalTopkKernel(",
    )
    ordered_float_key = extract_braced_definition(
        exact_text,
        "static __device__ __forceinline__ uint32_t "
        "SparkGlm52ResidentDecodeStageDsaOrderedFloatKey(",
    )
    configure = extract_braced_definition(
        exact_text,
        "static SparkStatus "
        "SparkGlm52Sm121RequiredDecodeStageConfigureCudaKernels(",
    )
    initialize = extract_braced_definition(
        exact_text,
        'extern "C" SparkStatus SparkGlm52Sm121RequiredDecodeStageInitialize(',
    )
    require(merge, "heap_keys", "GLM hierarchical max heap")
    require(merge, "partition_cursors", "GLM one-pass partition cursors")
    require(
        ordered_float_key,
        "if ((value_bits & 0x7fffffffu) == 0u)",
        "GLM signed-zero canonicalization",
    )
    require(
        ordered_float_key,
        "value_bits = 0u",
        "GLM canonical positive-zero key bits",
    )
    require(
        configure,
        "cudaFuncAttributeMaxDynamicSharedMemorySize",
        "GLM hierarchical DSA dynamic-shared opt-in",
    )
    require(
        configure,
        "SparkGlm52ResidentDecodeStageDsaScoreSelectHierarchicalKernel",
        "GLM hierarchical DSA configured kernel",
    )
    require(
        configure,
        "(size_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_HIERARCHICAL_PARTITION_CANDIDATES *",
        "GLM hierarchical DSA shared candidate count",
    )
    require(
        configure,
        "sizeof(uint64_t)",
        "GLM hierarchical DSA key storage",
    )
    require(
        initialize,
        "SparkGlm52Sm121RequiredDecodeStageConfigureCudaKernels()",
        "GLM initialization-time CUDA kernel configuration",
    )
    forbid(
        merge,
        "SparkGlm52ResidentDecodeStageDsaSelectTopKeyArray",
        "GLM redundant global merge radix",
    )
    require(selector, "__activemask()", "GLM tail-safe radix mask")
    router = extract_braced_definition(
        exact_text,
        "void SparkGlm52ResidentDecodeStageMoeRouterTopKFromLogitsKernel(",
    )
    require(router, "shared_ordered_keys", "GLM one-pass router ordering")
    require(router, "bitonic_size", "GLM bitonic router top-k")
    require(router, "SparkGlm52ResidentDecodeStageWarpReduceSum", "GLM router weight reduction")
    forbid(router, "for (selected_index = 0u;", "GLM repeated full router reductions")

    linear_plan_path = ROOT / (
        "modules/glm52_resident_decode_stage/source/"
        "spark_glm52_resident_decode_stage_linear_plan.cu"
    )
    linear_plan_text = linear_plan_path.read_text(encoding="utf-8")
    require(
        linear_plan_text,
        "storage_output_dimension_out = output_dimension",
        "GLM direct FP8 output geometry",
    )
    forbid(
        linear_plan_text,
        "padded_output_dimension",
        "GLM padded FP8 output geometry",
    )
    forbid(
        exact_text,
        "SparkGlm52ResidentDecodeStageCommitFp8LinearStorageOutput",
        "GLM FP8 trim copy",
    )
    require(
        exact_text,
        "fp8_linear_plan_active_rows_mismatch",
        "GLM exact-row FP8 qualification",
    )
    forbid(
        exact_text,
        "launch_count = prepared_active_sequence_count == active_sequence_count",
        "GLM per-row cuBLASLt fallback",
    )
    require(
        exact_text,
        "#define SPARK_GLM52_ENABLE_FP8_AMAX_PROBE 0",
        "GLM release FP8 probe disable",
    )
    phase_enabled = extract_braced_definition(
        exact_text,
        "static uint32_t SparkGlm52ResidentDecodeStagePhaseHashEnabled(void)",
    )
    amax_enabled = extract_braced_definition(
        exact_text,
        "static uint32_t SparkGlm52ResidentDecodeStageFp8AmaxProbeEnabled(void)",
    )
    require(phase_enabled, "#if SPARK_GLM52_ENABLE_STAGE_PHASE_HASH", "GLM hash compile gate")
    require(amax_enabled, "#if SPARK_GLM52_ENABLE_FP8_AMAX_PROBE", "GLM FP8 probe compile gate")

    stage_path = ROOT / (
        "modules/glm52_resident_decode_stage/source/"
        "spark_glm52_resident_decode_stage_cuda.cu"
    )
    stage_text = stage_path.read_text(encoding="utf-8")
    stage_slice = extract_braced_definition(
        stage_text,
        'extern "C" SparkStatus SparkGlm52ResidentDecodeStageBackendSubmitStageSlice(',
    )
    copy_final_tokens = extract_braced_definition(
        stage_text,
        "static SparkStatus SparkGlm52ResidentDecodeStageCudaCopyFinalTokens(",
    )
    fake_backend_path = ROOT / (
        "tests/fixtures/glm52_resident_decode_stage_fake_backend.c"
    )
    fake_backend_text = fake_backend_path.read_text(encoding="utf-8")
    fake_copy_final_tokens = extract_braced_definition(
        fake_backend_text,
        "static SparkStatus SparkGlm52ResidentDecodeStageFakeCopyFinalTokens(",
    )
    fake_stage_slice = extract_braced_definition(
        fake_backend_text,
        "SparkStatus SparkGlm52ResidentDecodeStageBackendSubmitStageSlice(",
    )
    module_path = ROOT / (
        "modules/glm52_resident_decode_stage/source/"
        "spark_glm52_resident_decode_stage_module.c"
    )
    module_text = module_path.read_text(encoding="utf-8")
    decode_token_count_supported = extract_braced_definition(
        module_text,
        "static bool SparkGlm52ResidentDecodeStageDecodeTokenCountIsSupported(",
    )
    frame_shape_supported = extract_braced_definition(
        module_text,
        "static bool SparkGlm52ResidentDecodeStageFrameShapeIsSupported(",
    )
    admit = extract_braced_definition(
        module_text,
        "SparkStatus SparkGlm52ResidentDecodeStageAdmit(",
    )
    require(stage_slice, "cudaLaunchHostFunc", "GLM stream-ordered stage completion")
    if stage_slice.count("cudaStreamSynchronize(") != 3:
        raise AssertionError("GLM stage slice may synchronize only for debug and two error paths")
    forbid(stage_slice, "completion->function(completion->context);", "GLM submit-thread completion")
    forbid(copy_final_tokens, "token_count = 1u", "GLM completion token count")
    forbid(
        copy_final_tokens,
        "token_count = SPARK_MODEL_DRIVER_COMPLETION_TOKEN_CAPACITY",
        "GLM completion token count",
    )
    require(
        copy_final_tokens,
        "return SPARK_STATUS_INVALID_ARGUMENT;",
        "GLM strict completion token count",
    )
    require(
        copy_final_tokens,
        "SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT + 1u",
        "GLM exact completion token capacity",
    )
    require(
        copy_final_tokens,
        "pipeline_slot->mtp_committed_token_ids",
        "GLM committed speculative tokens",
    )
    forbid(
        copy_final_tokens,
        "pipeline_slot->mtp_draft_token_ids",
        "GLM uncommitted speculative tokens",
    )
    require(
        fake_copy_final_tokens,
        "return SPARK_STATUS_INVALID_ARGUMENT;",
        "GLM fake strict completion token count",
    )
    require(
        fake_copy_final_tokens,
        "pipeline_slot->mtp_committed_token_ids",
        "GLM fake committed speculative tokens",
    )
    forbid(
        fake_copy_final_tokens,
        "token_count = 1u",
        "GLM fake completion token count",
    )
    require(
        fake_stage_slice,
        "layer_node_contexts[layer_count - 1u]",
        "GLM fake final-layer completion context",
    )
    require(
        fake_stage_slice,
        "&completion_node_context->pipeline_slots[pipeline_slot_index]",
        "GLM fake final-layer completion buffer",
    )
    require(
        fake_stage_slice,
        "first_node_context->pipeline_slots == 0",
        "GLM fake first-layer pipeline-slot validation",
    )
    require(
        fake_stage_slice,
        "layer_node_contexts[layer_index]->pipeline_slots == 0",
        "GLM fake per-layer pipeline-slot validation",
    )
    require(
        decode_token_count_supported,
        "SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_SPECULATIVE_ROWS_PER_LANE",
        "GLM internal speculative row capacity",
    )
    require(
        decode_token_count_supported,
        "state->stage_slice_final_token_stage == 0u",
        "GLM final-token stage distinction",
    )
    require(
        decode_token_count_supported,
        "SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT + 1u",
        "GLM final completion token capacity",
    )
    for caller, label in (
        (frame_shape_supported, "GLM execution shape validation"),
        (admit, "GLM admission shape validation"),
    ):
        require(
            caller,
            "SparkGlm52ResidentDecodeStageDecodeTokenCountIsSupported(",
            label,
        )


def validate_pr505_graph_configuration_contract() -> None:
    path = ROOT / (
        "modules/glm52_resident_decode_stage/source/"
        "spark_glm52_pp13_node_context_builder_cuda.cu"
    )
    text = path.read_text(encoding="utf-8")
    parser = extract_braced_definition(
        text,
        "static SparkStatus SparkGlm52Pp13BuilderReadOptionalBooleanEnvironment(",
    )
    configure = extract_braced_definition(
        text,
        "static SparkStatus SparkGlm52Pp13BuilderConfigureMtpLayer(",
    )
    require(parser, 'strcmp(text,"1") == 0', "strict graph-enable true value")
    require(parser, 'strcmp(text,"0") == 0', "strict graph-enable false value")
    require(parser, "SPARK_STATUS_INVALID_ARGUMENT", "invalid graph value rejection")
    require(
        configure,
        "SparkGlm52Pp13BuilderReadOptionalBooleanEnvironment",
        "explicit MTP graph configuration",
    )
    forbid(
        configure,
        'getenv("SPARKPIPE_MTP_LAYER_ENABLE_GRAPH") != 0',
        "presence-only MTP graph configuration",
    )



def validate_fp8_tile_contract() -> None:
    path = ROOT / "model-families/common/include/sparkpipe/spark_lm_fp8_tile.cuh"
    text = path.read_text(encoding="utf-8")
    body = extract_braced_definition(
        text,
        "static __device__ void SparkLmExpertTileBodyFp8(",
    )
    fragment_a = extract_braced_definition(
        text,
        "static __device__ __forceinline__ void SparkLmFp8LoadFragA(",
    )
    fragment_b = extract_braced_definition(
        text,
        "static __device__ __forceinline__ void SparkLmFp8LoadFragB(",
    )
    stage_input = extract_braced_definition(
        text,
        "static __device__ void SparkLmFp8StageInput(",
    )
    stage_input_producer = extract_braced_definition(
        text,
        "static __device__ void SparkLmFp8StageInputProducerGroup(",
    )
    dispatch_path = ROOT / "model-families/common/include/sparkpipe/spark_lm_kernels.cuh"
    dispatch_text = dispatch_path.read_text(encoding="utf-8")
    dispatch = extract_braced_definition(
        dispatch_text,
        "static __device__ void SparkLmExpertTileDispatch(",
    )
    batched_linear = extract_braced_definition(
        dispatch_text,
        "static inline cudaError_t SparkLmHostLaunchBatchedLinear(",
    )
    mimo_path = ROOT / (
        "modules/mimo25_resident_decode_stage/source/"
        "spark_mimo25_resident_decode_stage_cuda.cu"
    )
    mimo_text = mimo_path.read_text(encoding="utf-8")
    mimo_expert = extract_braced_definition(
        mimo_text,
        'extern "C" cudaError_t SparkMimo25LaunchExpertTile(',
    )
    mimo_all_experts = extract_braced_definition(
        mimo_text,
        'extern "C" cudaError_t SparkMimo25LaunchExpertTileAll(',
    )

    require(body, "block_accumulator", "FP8 per-scale-block accumulator")
    require(body, "next_k_base % 128u", "FP8 128-wide scale boundary")
    require(body, "neuron_tile_base / 128u", "FP8 output scale block")
    require(body, "k_base / 128u", "FP8 input scale block")
    require(
        body,
        "neuron_tile_offset < SPARK_LM_TILE_N",
        "FP8 complete shared-tile output coverage",
    )
    require(
        body,
        "neuron_tile_offset += SPARK_LM_FP8_TILE_N",
        "FP8 complete shared-tile output coverage",
    )
    require(body, "row_inverse_scale", "FP8 cached activation reciprocal")
    forbid(stage_input, "1.0f /", "FP8 input-staging reciprocal reuse")
    forbid(
        stage_input_producer,
        "1.0f /",
        "FP8 producer-staging reciprocal reuse",
    )
    require(
        stage_input,
        "row_inverse_scale[row_in_tile]",
        "FP8 input-staging reciprocal reuse",
    )
    require(
        stage_input_producer,
        "row_inverse_scale[row_in_tile]",
        "FP8 producer-staging reciprocal reuse",
    )
    fragment_a_order = [
        "a[0] = base[(row_lo * stride_words) + k_word0];",
        "a[1] = base[((row_lo + 8u) * stride_words) + k_word0];",
        "a[2] = base[(row_lo * stride_words) + k_word1];",
        "a[3] = base[((row_lo + 8u) * stride_words) + k_word1];",
    ]
    fragment_a_positions = [
        fragment_a.index(assignment) for assignment in fragment_a_order
    ]
    if fragment_a_positions != sorted(fragment_a_positions):
        raise AssertionError("FP8 A registers do not follow the PTX m16n8k32 layout")
    require(
        fragment_b,
        "b[0] = base[(col * stride_words) + k_word0];",
        "FP8 PTX B-register layout",
    )
    require(
        fragment_b,
        "b[1] = base[(col * stride_words) + k_word0 + 4u];",
        "FP8 PTX B-register layout",
    )
    covered_neurons: list[int] = []
    for neuron_tile_offset in range(0, 128, 64):
        covered_neurons.extend(
            range(neuron_tile_offset, neuron_tile_offset + 64)
        )
    if covered_neurons != list(range(128)):
        raise AssertionError("FP8 subtiles do not cover each shared output column once")
    if body.count("__syncthreads();") != 6:
        raise AssertionError("FP8 tile must retain all six CTA synchronization points")

    neuron_loop = body.index("for (neuron_tile_offset = 0u;")
    k_loop = body.index("for (k_base = 0u;", neuron_loop)
    initial_stage = body[neuron_loop:k_loop]
    initial_stage_order = [
        initial_stage.find("SparkLmFp8StageInput("),
        initial_stage.find("SparkLmFp8StageWeight("),
        initial_stage.find("__syncthreads();"),
    ]
    if -1 in initial_stage_order or initial_stage_order != sorted(initial_stage_order):
        raise AssertionError(
            "FP8 initial shared staging must complete before current-buffer consumers"
        )

    store_start = body.index("SparkLmFp8StoreAccumulator(", k_loop)
    k_pipeline = body[k_loop:store_start]
    first_barrier = k_pipeline.find("__syncthreads();")
    second_barrier = k_pipeline.find("__syncthreads();", first_barrier + 1)
    third_barrier = k_pipeline.find("__syncthreads();", second_barrier + 1)
    # The weight pipeline must retire exactly one group per iteration, never
    # drain. A wait_all here idles the memory system at every iteration boundary,
    # and the weight stream is the bandwidth bound.
    if "SparkLmAsyncWaitAll();" in k_pipeline:
        raise AssertionError(
            "FP8 K pipeline must retire one group, not drain the weight stream"
        )
    # The invariant is the ping-pong itself: the next weight tile is issued
    # before either consumer phase, the two consumer phases alternate across
    # exactly two barriers, and the asynchronous staging is retired before the
    # buffers swap. How the staging is divided among warps is not the contract -
    # it used to be two half-tile producers because the loads blocked, and is now
    # one whole-tile issue because cp.async does not.
    k_pipeline_order = [
        k_pipeline.find("SparkLmFp8StageWeight("),
        k_pipeline.find("if (warp_index < 4u)"),
        k_pipeline.find("SparkLmFp8StageInputProducerGroup("),
        first_barrier,
        k_pipeline.find("if (warp_index >= 4u)"),
        k_pipeline.find("SparkLmAsyncWait<1u>();"),
        second_barrier,
        k_pipeline.find("if ((next_k_base % 128u) == 0u"),
        k_pipeline.find("current_buffer = next_buffer;"),
    ]
    if (
        -1 in k_pipeline_order
        or k_pipeline_order != sorted(k_pipeline_order)
        or third_barrier != -1
    ):
        raise AssertionError(
            "FP8 ping-pong producers and consumers must remain CTA-synchronized"
        )

    output_stage = body[store_start:]
    output_first_barrier = output_stage.find("__syncthreads();")
    output_second_barrier = output_stage.find(
        "__syncthreads();",
        output_first_barrier + 1,
    )
    output_third_barrier = output_stage.find(
        "__syncthreads();",
        output_second_barrier + 1,
    )
    output_stage_order = [
        output_stage.find("SparkLmFp8StoreAccumulator("),
        output_first_barrier,
        output_stage.find("for (entry = threadIdx.x;"),
        output_stage.find("SparkLmFloatToBf16("),
        output_second_barrier,
    ]
    if (
        -1 in output_stage_order
        or output_stage_order != sorted(output_stage_order)
        or output_third_barrier != -1
    ):
        raise AssertionError(
            "FP8 output staging must be fenced before copy and shared-tile reuse"
        )
    forbid(text, "weight_scale[0]", "FP8 single-scale GEMM")
    require(
        dispatch,
        "SPARK_LM_WEIGHT_FORMAT_FP8_E4M3_F32B128",
        "FP8 exact format dispatch",
    )
    forbid(dispatch, "weight_scale != 0", "FP8 silent BF16 fallback")
    forbid(dispatch, "input_dimension % 128u", "FP8 silent BF16 fallback")
    for launcher, label in (
        (batched_linear, "shared batched linear"),
        (mimo_expert, "MiMo expert tile"),
        (mimo_all_experts, "MiMo all-expert tile"),
    ):
        require(
            launcher,
            "SPARK_LM_WEIGHT_FORMAT_FP8_E4M3_F32B128",
            f"{label} FP8 validation",
        )
        require(launcher, "cudaErrorInvalidValue", f"{label} FP8 validation")
    forbid(text, "nvcc 13.1, verified", "unretained CUDA qualification claim")


def validate_gpudirect_transport_contract() -> None:
    path = ROOT / "modules/hidden_transport_spark_host_rdma_verbs.cu"
    text = path.read_text(encoding="utf-8")
    prepare = extract_braced_definition(
        text,
        "static SparkStatus SparkHiddenSparkHostRdmaPreparePacketMemory(",
    )
    exchange = extract_braced_definition(
        text,
        "static SparkStatus SparkHiddenSparkHostRdmaExchangeQueuePairInfo(",
    )
    finalize = extract_braced_definition(
        text,
        "static SparkStatus SparkHiddenSparkHostRdmaFinalizePendingReceive(",
    )
    start_send = extract_braced_definition(
        text,
        "static SparkStatus SparkHiddenSparkHostRdmaPrepareInflightSend(",
    )
    start_batch = extract_braced_definition(
        text,
        "static SparkStatus SparkHiddenSparkHostRdmaStartSendBatch(",
    )
    region_cache = extract_braced_definition(
        text,
        "static SparkStatus SparkHiddenSparkHostRdmaGetCachedMemoryRegion(",
    )
    parse_uint = extract_braced_definition(
        text,
        "static SparkStatus SparkHiddenSparkHostRdmaParseUintEnv(",
    )
    initialize = extract_braced_definition(
        text,
        "static SparkStatus SparkHiddenSparkHostRdmaInitialize(",
    )

    require(text, "SPARK_HIDDEN_SPARK_RDMA_DEVICE_DIRECT", "separate GPUDirect build mode")
    require(text, "SPARK_HIDDEN_SPARK_HOST_RDMA_MEMORY_MODE_DEVICE_DIRECT", "GPUDirect wire mode")
    require(exchange, "remote_infos[lane_index].memory_mode != state->memory_mode", "RDMA memory-mode handshake")
    require(prepare, "SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_DEVICE_POINTER", "RDMA device-visible pointer contract")
    if prepare.count("SparkHiddenSparkHostRdmaGetCachedMemoryRegion") != 2:
        raise AssertionError("hidden and sideband RDMA payloads must use persistent MR registration")
    forbid(prepare, "memcpy(", "mapped-host staging copy")
    forbid(text, "SparkHiddenSparkHostRdmaStagingSlot", "secondary RDMA staging ring")
    require(
        prepare,
        "mapped_host_zero_copy_transfer_count",
        "mapped-host direct-registration receipt",
    )
    require(
        start_send,
        "return SPARK_STATUS_BUSY;",
        "single-send source ownership until CQ completion",
    )
    require(
        start_batch,
        "return SPARK_STATUS_BUSY;",
        "batch source ownership until CQ completion",
    )
    require(finalize, "SparkHiddenSparkHostRdmaFlushGpudirectWrites", "GPUDirect inbound visibility")
    require(
        text,
        "cudaDevAttrGPUDirectRDMAWritesOrdering",
        "GPUDirect native visibility query",
    )
    require(
        text,
        "cudaGPUDirectRDMAWritesOrderingOwner",
        "GPUDirect owner-ordering fast path",
    )
    require(
        text,
        "cudaDevAttrGPUDirectRDMAFlushWritesOptions",
        "GPUDirect flush capability query",
    )
    require(
        text,
        "cudaFlushGPUDirectRDMAWritesOptionHost",
        "GPUDirect host flush support",
    )
    require(text, "state->gpudirect_transfer_bytes", "GPUDirect traffic receipt")
    fast_cache_lookup = region_cache.index(
        "state->cached_regions[index].cuda_visible_pointer == pointer"
    )
    pointer_query = region_cache.index(
        "SparkHiddenSparkHostRdmaResolveRegistrationPointer("
    )
    if fast_cache_lookup >= pointer_query:
        raise AssertionError(
            "RDMA MR cache must bypass CUDA pointer attributes on exact cache hits"
        )
    require(
        region_cache,
        "state->pointer_attribute_query_count += 1u",
        "RDMA CUDA pointer-query receipt",
    )
    require(parse_uint, "errno = 0", "strict RDMA integer parsing")
    require(
        parse_uint,
        "return SPARK_STATUS_INVALID_ARGUMENT;",
        "strict RDMA integer parsing",
    )
    forbid(parse_uint, "return fallback", "strict RDMA integer parsing")
    forbid(initialize, "lane_count = 1u", "RDMA lane configuration")
    forbid(
        initialize,
        "lane_count = SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_LANE_COUNT",
        "RDMA lane configuration",
    )
    require(
        initialize,
        "config_value > 255u",
        "RDMA verbs-port range validation",
    )
    require(
        initialize,
        "state->control_port_base == 0u",
        "RDMA control-port zero rejection",
    )

    header = (ROOT / "model-families/common/include/sparkpipe/spark_hidden_transport.h").read_text(encoding="utf-8")
    host_caps_start = header.index("#define SPARK_HIDDEN_TRANSPORT_REQUIRED_SPARK_HOST_RDMA_CAPS")
    host_caps_end = header.index("#define SPARK_HIDDEN_TRANSPORT_RECOMMENDED_SPARK_HOST_RDMA_CAPS", host_caps_start)
    host_caps = header[host_caps_start:host_caps_end]
    require(
        host_caps,
        "SPARK_HIDDEN_TRANSPORT_REQUIRED_PRODUCTION_CAPS",
        "mapped-host RDMA no-staging capability",
    )

    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    require(
        makefile,
        "hidden_transport_spark_gpudirect_rdma_verbs",
        "separate GPUDirect transport artifact",
    )
    require(
        makefile,
        "-DSPARK_HIDDEN_SPARK_RDMA_DEVICE_DIRECT=1",
        "GPUDirect compile mode",
    )
    require(
        makefile,
        "-DSPARK_HIDDEN_SPARK_RDMA_DEVICE_DIRECT=0",
        "mapped-host compile mode",
    )
    forbid(
        makefile,
        "hidden_transport_spark_gpudirect_rdma_verbs skipped:",
        "GPUDirect build prerequisites",
    )
    require(
        text,
        '#error "SPARK_HIDDEN_SPARK_RDMA_DEVICE_DIRECT must be explicitly set to 0 or 1"',
        "explicit RDMA memory mode",
    )
    require(
        text,
        '#error "GPUDirect RDMA transport requires CUDA runtime 11.3 or newer"',
        "GPUDirect CUDA runtime floor",
    )


def main() -> int:
    changed_cuda_paths = [
        ROOT / "model-families/common/include/sparkpipe/spark_lm_kernels.cuh",
        ROOT / "modules/dsv4_resident_decode_stage/source/spark_dsv4_resident_decode_stage_cuda.cu",
        ROOT / "modules/glm52_resident_decode_stage/source/spark_glm52_resident_decode_stage_cuda.cu",
        ROOT / "modules/glm52_resident_decode_stage/source/spark_glm52_sm121_required_decode_stage.cu",
        ROOT / "modules/k3_resident_decode_stage/source/spark_k3_resident_decode_stage_cuda.cu",
        ROOT / "modules/mimo25_resident_decode_stage/source/spark_mimo25_resident_decode_stage_cuda.cu",
        ROOT / "modules/qwen36_resident_decode_stage/source/spark_qwen36_resident_decode_stage_cuda.cu",
        ROOT / "modules/hidden_transport_spark_host_rdma_verbs.cu",
        ROOT / "modules/glm52_resident_decode_stage/source/spark_glm52_pp13_node_context_builder_cuda.cu",
        ROOT / "model-families/common/include/sparkpipe/spark_lm_fp8_tile.cuh",
    ]
    for path in changed_cuda_paths:
        validate_balanced_delimiters(path)
    validate_radix_selection()
    validate_bitonic_router_selection()
    validate_hierarchical_selection()
    validate_fp8_block_scale_accumulation()
    validate_moe_grouping()
    validate_k3_contract()
    validate_qwen_contract()
    validate_shared_kernel_contract()
    validate_dsv4_contract()
    validate_mimo_contract()
    validate_glm_contract()
    validate_pr505_graph_configuration_contract()
    validate_fp8_tile_contract()
    validate_gpudirect_transport_contract()
    print("PASS CUDA performance source contracts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
