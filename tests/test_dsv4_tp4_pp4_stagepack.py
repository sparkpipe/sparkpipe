#!/usr/bin/env python3
"""Static topology and ownership checks for the TP4 x PP4 pack slicer."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "dsv4_tp16_stagepack.py"
SPEC = importlib.util.spec_from_file_location("dsv4_parallel_pack", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PACK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PACK)


def expect_filtered(entry: tuple[int, ...], pp_stage: int) -> None:
    try:
        PACK.plan_entry(entry, 0, 4, pp_stage)
    except PACK.PackFailure as error:
        assert str(error) == "filtered"
        return
    raise AssertionError("entry should have been filtered")


def dot(weight_row: list[int], vector: list[int]) -> int:
    return sum(weight * value for weight, value in zip(weight_row, vector))


def verify_output_projection_sum(tp_degree: int) -> None:
    group_count = 8
    group_width = 4
    lora_width = 2
    hidden = 3
    attention = [[1 + group * 3 + column
                  for column in range(group_width)]
                 for group in range(group_count)]
    wo_a = [[[1 + output + 2 * column + group
              for column in range(group_width)]
             for output in range(lora_width)]
            for group in range(group_count)]
    wo_b = [[[1 + row + output + 3 * group
              for output in range(lora_width)]
             for group in range(group_count)]
            for row in range(hidden)]
    expected = [sum(wo_b[row][group][output] *
                    dot(wo_a[group][output], attention[group])
                    for group in range(group_count)
                    for output in range(lora_width))
                for row in range(hidden)]
    actual = [0] * hidden
    PACK.TP_DEGREE = tp_degree
    for rank in range(tp_degree):
        group_start, groups, column_start, columns = (
            PACK.output_group_shard(rank)
        )
        column_start = column_start * group_width // PACK.OUTPUT_GROUP_DIM
        columns = columns * group_width // PACK.OUTPUT_GROUP_DIM
        for row in range(hidden):
            for group in range(group_start, group_start + groups):
                for output in range(lora_width):
                    partial = dot(
                        wo_a[group][output][column_start:column_start + columns],
                        attention[group][column_start:column_start + columns],
                    )
                    actual[row] += wo_b[row][group][output] * partial
    assert actual == expected


def verify_down_projection_sum() -> None:
    columns = PACK.EXPERT_WIDTH
    rows = (0, 17, PACK.HIDDEN - 1)
    vector = [1 + (column % 11) for column in range(columns)]
    PACK.TP_DEGREE = 4
    for row in rows:
        expected = sum((1 + ((row * 7 + column * 3) % 19)) * vector[column]
                       for column in range(columns))
        actual = 0
        for rank in range(PACK.TP_DEGREE):
            start, width = PACK.column_slice(PACK.KIND_SHARED_W2, rank,
                                             columns)
            actual += sum((1 + ((row * 7 + column * 3) % 19)) *
                          vector[column]
                          for column in range(start, start + width))
        assert actual == expected


def main() -> int:
    PACK.TP_DEGREE = 4
    assert [PACK.layer_slice(4, stage) for stage in range(4)] == [
        (0, 11), (11, 11), (22, 11), (33, 10)
    ]
    layer_32 = (PACK.KIND_ATTN_SINK, 32, PACK.WEIGHT_F32, 1, 64, 0, 0, 0)
    layer_33 = (PACK.KIND_ATTN_SINK, 33, PACK.WEIGHT_F32, 1, 64, 0, 0, 0)
    expect_filtered(layer_32, 3)
    planned, _, column_start, _ = PACK.plan_entry(layer_33, 2, 4, 3)
    assert planned[1] == 33 and planned[3:5] == (1, 16)
    assert column_start == 32
    embedding = (
        PACK.KIND_EMBEDDING, PACK.GLOBAL_LAYER, PACK.WEIGHT_BF16,
        PACK.VOCAB, PACK.HIDDEN, 0, 0, 0,
    )
    final_norm = (
        PACK.KIND_FINAL_NORM, PACK.GLOBAL_LAYER, PACK.WEIGHT_BF16,
        1, PACK.HIDDEN, 0, 0, 0,
    )
    PACK.plan_entry(embedding, 0, 4, 0)
    expect_filtered(embedding, 1)
    expect_filtered(final_norm, 0)
    expect_filtered(final_norm, 2)
    PACK.plan_entry(final_norm, 0, 4, 3)
    PACK.plan_entry(final_norm, 3, 4, 3)
    lm_head = (
        PACK.KIND_LM_HEAD, PACK.GLOBAL_LAYER, PACK.WEIGHT_BF16,
        PACK.VOCAB, PACK.HIDDEN, 0, 0, 0,
    )
    expected_starts = (0, 32384, 64768, 97024)
    expected_counts = (32384, 32384, 32256, 32256)
    for rank in range(4):
        planned, indices, column_start, _ = PACK.plan_entry(
            lm_head, rank, 4, 3
        )
        assert planned[3:5] == (expected_counts[rank], PACK.HIDDEN)
        assert indices[0] == expected_starts[rank]
        assert indices[-1] + 1 == expected_starts[rank] + expected_counts[rank]
        assert column_start == 0
    wo_a = (PACK.KIND_WO_A, 33, PACK.WEIGHT_FP8,
            PACK.OUTPUT_GROUPS * PACK.OUTPUT_LORA,
            PACK.OUTPUT_GROUP_DIM, 0, 0, 0)
    planned, indices, column_start, _ = PACK.plan_entry(wo_a, 2, 4, 3)
    assert planned[3:5] == (2 * PACK.OUTPUT_LORA,
                            PACK.OUTPUT_GROUP_DIM)
    assert indices[0] == 4 * PACK.OUTPUT_LORA
    assert indices[-1] == 6 * PACK.OUTPUT_LORA - 1
    assert column_start == 0
    wo_b = (PACK.KIND_WO_B, 33, PACK.WEIGHT_FP8, PACK.HIDDEN,
            PACK.OUTPUT_GROUPS * PACK.OUTPUT_LORA, 0, 0, 0)
    planned, indices, column_start, _ = PACK.plan_entry(wo_b, 2, 4, 3)
    assert planned[3:5] == (PACK.HIDDEN, 2 * PACK.OUTPUT_LORA)
    assert len(indices) == PACK.HIDDEN and column_start == 4 * PACK.OUTPUT_LORA
    shared_w2 = (PACK.KIND_SHARED_W2, 33, PACK.WEIGHT_FP4, PACK.HIDDEN,
                 PACK.EXPERT_WIDTH, 0, 0, 0)
    planned, indices, column_start, _ = PACK.plan_entry(shared_w2, 2, 4, 3)
    assert planned[3:5] == (PACK.HIDDEN, PACK.EXPERT_WIDTH // 4)
    assert len(indices) == PACK.HIDDEN
    assert column_start == 2 * PACK.EXPERT_WIDTH // 4
    verify_output_projection_sum(4)
    verify_output_projection_sum(16)
    verify_down_projection_sum()
    PACK.TP_DEGREE = 4
    assert 4 * 1024 == 4096
    print("PASS DSV4 TP packs: exact WO/W2 partitions sum to dense output")
    print("PASS DSV4 TP4 x PP4 packs: 11/11/11/10 and 4096 B1024 requests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
