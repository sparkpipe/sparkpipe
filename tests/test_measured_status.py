#!/usr/bin/env python3

import json
import pathlib
import re
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "tools"))

from glm52_model_contract import CODECS, description_path
ACTIVE_CUDA_DIRECTORIES = (
    ROOT / "inference" / "llms" / "glm5_2",
    ROOT / "modules" / "glm52_resident_decode_stage",
)
FORBIDDEN_FLAGS = {
    "jit_kv_cache",
    "no_host_staging",
    "no_device_memcpy",
    "driver_private_expert_queues",
    "validated_latency",
}
POSITIVE_LATENCY = re.compile(
    r"(?:validated_maximum_latency_ns|validated_stage_latency_ns|"
    r"estimated_stage_latency_ns)\s*=\s*[1-9][0-9]*"
    r"(?:u|ul|ull)?\b",
    re.I,
)


def main():
    for codec in CODECS:
        model = json.loads(description_path(ROOT,codec).read_text(encoding="utf-8"))
        precision = model["metadata"]["precision_contract"]
        qualification = model["metadata"]["qualification"]
        program = model["stages"][0]["programs"][0]
        scheduling = program["scheduling"]
        flags = set(scheduling["flags"])
        assert precision["expert_weight_codec"] == codec
        assert precision["aot_codec_specialization"] is True
        assert precision["runtime_precision_selection"] == "forbidden"
        assert precision["fallback_allowed"] is False
        assert qualification["status"] == "NOT_MEASURED"
        assert qualification["production_ready"] is False
        assert program["max_inflight"] == 4
        assert scheduling["max_active_slots"] == 1024
        assert scheduling["max_resident_sequences"] == 1024
        assert scheduling["validated_latency_ns"] == 0
        assert scheduling["private_queue_count"] == 0
        assert flags.isdisjoint(FORBIDDEN_FLAGS)

    cuda_sources = []
    for source_directory in ACTIVE_CUDA_DIRECTORIES:
        cuda_sources.extend(sorted(source_directory.rglob("*.cu")))
        cuda_sources.extend(sorted(source_directory.rglob("*.cuh")))
    assert cuda_sources, "no GLM-5.2 CUDA sources found"
    for source_path in cuda_sources:
        source = source_path.read_text(encoding="utf-8")
        assert POSITIVE_LATENCY.search(source) is None, source_path

    status = (ROOT / "docs" / "GLM52_MEASURED_STATUS.md").read_text(
        encoding="utf-8")
    for required in ("MEASURED", "OBSERVED", "NOT_MEASURED", "NOT_WORKING"):
        assert required in status


if __name__ == "__main__":
    main()
