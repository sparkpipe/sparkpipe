"""Algorithmic contracts for kernels/.

What survived a rewrite of this file, and why. It had sixteen contracts; four
named files that no longer exist and eight asserted structural properties of a
tree being replaced - that a source contained a string, that a family called a
shared library a certain number of times. Those said nothing about behaviour and
are now either compile-time static_asserts or meaningless.

What is kept is the part that could not be a static_assert: emulations of an
algorithm checked against an independent reference. Radix selection over float
keys has to handle NaN, both infinities and negative zero, and the monotonic-key
trick that makes it work is exactly the kind of thing that looks right and is
wrong for one input class.

These emulate kernels/topk.cuh and kernels/gemm.cuh in Python. That is a real
duplication and it is the point: an emulation that shares no code with the
kernel disagrees when either is wrong, where a test built from the kernel's own
helpers agrees with it either way.
"""
import heapq
import math
import random
import sys
import struct
from pathlib import Path

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



def main() -> int:
    failures = 0
    for name, fn in [
        ("radix selection", validate_radix_selection),
        ("bitonic router selection", validate_bitonic_router_selection),
        ("hierarchical selection", validate_hierarchical_selection),
        ("fp8 block scale accumulation", validate_fp8_block_scale_accumulation),
        ("moe grouping", validate_moe_grouping),
    ]:
        try:
            fn()
            print(f"  ok   {name}")
        except AssertionError as error:
            print(f"  FAIL {name}: {error}")
            failures += 1
    print(f"\n{'FAIL' if failures else 'PASS'} ({failures} failing)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
