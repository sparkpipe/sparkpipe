#!/usr/bin/env python3
"""Exercise the DSV4 compact resident-pool arithmetic on the host."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_HEADER = ROOT / "model-families/dsv4/include/sparkpipe/spark_dsv4_model.h"
FIRMWARE_HEADER = (
    ROOT / "modules/dsv4_resident_decode_stage/include/sparkpipe/"
    "spark_dsv4_resident_decode_stage_firmware.h"
)
ADAPTER_SOURCE = (
    ROOT / "modules/dsv4_resident_decode_stage/source/spark_dsv4_serving_adapter.c"
)
DEPLOYMENT = ROOT / "examples/deployments/dsv4_flash_stage.json"


@dataclass(frozen=True)
class LayerRegion:
    layer: int
    ratio: int
    slots: int
    cache_offset: int
    cache_stride: int
    state_offset: int | None
    state_stride: int | None


def uint_define(text: str, name: str) -> int:
    match = re.search(rf"^#define {name} ([0-9]+)u$", text, re.MULTILINE)
    assert match is not None, f"missing integer definition {name}"
    return int(match.group(1))


def compression_ratios(text: str) -> list[int]:
    match = re.search(
        r"SparkDsv4ModelCompressionRatios\[[^]]+\]\s*=\s*\{(.*?)\};",
        text,
        re.DOTALL,
    )
    assert match is not None, "missing generated compression-ratio table"
    return [int(value) for value in re.findall(r"\b([0-9]+)u\b", match.group(1))]


def stage_layer_counts(text: str) -> list[int]:
    match = re.search(r"\.stage_layer_counts\s*=\s*\{([^}]+)\}", text)
    assert match is not None, "missing serving stage partition"
    return [int(value) for value in re.findall(r"\b([0-9]+)u\b", match.group(1))]


def build_layout(
    ratios: list[int], first_layer: int, layer_count: int, resident: int,
    maxseq: int, window: int, head_dim: int, csa_ratio: int,
    hca_ratio: int, csa_overlap: int,
) -> tuple[list[LayerRegion], int, int]:
    cache_cursor = 0
    state_cursor = 0
    regions = []
    for layer in range(first_layer, first_layer + layer_count):
        ratio = ratios[layer]
        assert ratio in (0, csa_ratio, hca_ratio)
        slots = window + (maxseq // ratio if ratio != 0 else 0)
        cache_stride = slots * head_dim
        state_offset = None
        state_stride = None
        if ratio != 0:
            overlap = csa_overlap if ratio == csa_ratio else 1
            state_offset = state_cursor
            state_stride = overlap * ratio * overlap * head_dim
            state_cursor += resident * state_stride
        regions.append(LayerRegion(
            layer, ratio, slots, cache_cursor, cache_stride,
            state_offset, state_stride,
        ))
        cache_cursor += resident * cache_stride
    return regions, cache_cursor, state_cursor


def assert_bounds(regions: list[LayerRegion], resident: int) -> None:
    cache_cursor = 0
    state_cursor = 0
    for region in regions:
        assert region.cache_offset == cache_cursor
        cache_end = region.cache_offset + resident * region.cache_stride
        last_lane = region.cache_offset + (resident - 1) * region.cache_stride
        last_element = last_lane + region.slots * (region.cache_stride // region.slots) - 1
        assert last_element == cache_end - 1
        cache_cursor = cache_end
        if region.ratio == 0:
            assert region.state_offset is None and region.state_stride is None
            continue
        assert region.state_offset == state_cursor
        assert region.state_stride is not None
        reset_begin = region.state_offset + (resident - 1) * region.state_stride
        reset_end = reset_begin + region.state_stride
        state_cursor = region.state_offset + resident * region.state_stride
        assert reset_end == state_cursor


def main() -> int:
    model = MODEL_HEADER.read_text(encoding="utf-8")
    firmware = FIRMWARE_HEADER.read_text(encoding="utf-8")
    adapter = ADAPTER_SOURCE.read_text(encoding="utf-8")
    deployment = json.loads(DEPLOYMENT.read_text(encoding="utf-8"))
    ratios = compression_ratios(model)
    resident = uint_define(
        firmware, "SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT")
    maxseq = deployment["max_sequence_positions"]
    window = uint_define(model, "SPARK_DSV4_MODEL_SLIDING_WINDOW_TOKENS")
    head_dim = uint_define(model, "SPARK_DSV4_MODEL_HEAD_DIMENSION")
    index_dim = uint_define(model, "SPARK_DSV4_MODEL_INDEX_HEAD_DIMENSION")
    bf16_bytes = uint_define(model, "SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES")
    csa_ratio = uint_define(model, "SPARK_DSV4_MODEL_CSA_COMPRESS_RATIO")
    hca_ratio = uint_define(model, "SPARK_DSV4_MODEL_HCA_COMPRESS_RATIO")
    overlap = uint_define(model, "SPARK_DSV4_MODEL_CSA_OVERLAP_FACTOR")
    assert (maxseq, resident) == (4096, 128)
    assert len(ratios) == uint_define(model, "SPARK_DSV4_MODEL_LAYER_COUNT")
    assert maxseq % csa_ratio == 0 and maxseq % hca_ratio == 0

    mixed, cache_elements, state_elements = build_layout(
        ratios, 0, 4, resident, maxseq, window, head_dim,
        csa_ratio, hca_ratio, overlap,
    )
    assert_bounds(mixed, resident)
    assert [region.ratio for region in mixed] == [0, 0, 4, 128]
    assert [(region.cache_offset, region.cache_stride) for region in mixed] == [
        (0, 65536), (8388608, 65536),
        (16777216, 589824), (92274688, 81920),
    ]
    assert [(region.state_offset, region.state_stride) for region in mixed] == [
        (None, None), (None, None), (0, 8192), (1048576, 65536),
    ]
    assert (cache_elements, state_elements) == (102760448, 9437184)

    index_slots = maxseq // csa_ratio
    index_lane_stride = index_slots * index_dim
    index_state_stride = overlap * csa_ratio * overlap * index_dim
    last_index_element = (
        (resident - 1) * index_lane_stride + index_slots * index_dim - 1)
    assert last_index_element == resident * index_lane_stride - 1
    last_index_reset = (
        (resident - 1) * index_state_stride + index_state_stride)
    assert last_index_reset == resident * index_state_stride
    assert (index_lane_stride, index_state_stride) == (131072, 2048)

    first = 0
    stage_bytes = []
    counts = stage_layer_counts(adapter)
    assert sum(counts) == len(ratios)
    for count in counts:
        regions, cache_elements, state_elements = build_layout(
            ratios, first, count, resident, maxseq, window, head_dim,
            csa_ratio, hca_ratio, overlap,
        )
        assert_bounds(regions, resident)
        csa_count = sum(region.ratio == csa_ratio for region in regions)
        resident_bytes = (
            cache_elements * bf16_bytes
            + 2 * state_elements * 4
            + csa_count * resident * (
                index_lane_stride * bf16_bytes + 2 * index_state_stride * 4)
        )
        stage_bytes.append(resident_bytes)
        first += count
    assert first == len(ratios)
    assert max(stage_bytes) == 566231040
    assert [index for index, size in enumerate(stage_bytes) if size == max(stage_bytes)] == [
        7, 8, 9, 10, 11,
    ]
    print("PASS DSV4 compact pool layout: maxseq=4096 resident=128 worst_stage=566231040")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
