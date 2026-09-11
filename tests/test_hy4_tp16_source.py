#!/usr/bin/env python3
"""hy4 TP16 collective wiring contracts: the module path reaches the async
engine surface, the rung drives the production engine and combine kernels,
and every error path in the touched C sources fails closed."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODULE = ROOT / "modules/hy4_resident_decode_stage/source/\
spark_hy4_resident_decode_stage_module.c"
CUDA_TU = ROOT / "modules/hy4_resident_decode_stage/source/\
spark_hy4_resident_decode_stage_cuda.cu"
FIRMWARE = ROOT / ("modules/hy4_resident_decode_stage/include/sparkpipe/"
                   "spark_hy4_resident_decode_stage_firmware.h")
RUNG = ROOT / "tools/hy4_gpu/hy4_tp16_rung.cu"
RUNNER = ROOT / "tools/hy4_gpu/run_tp16_rung.sh"
SERVING_RUNG = ROOT / "tools/hy4_gpu/hy4_tp16_serving_rung.c"
SERVING_RUNNER = ROOT / "tools/hy4_gpu/run_tp16_serving_rung.sh"


def require(pattern: str, text: str, label: str) -> re.Match:
    found = re.search(pattern, text, re.M)
    assert found, f"missing {label}"
    return found


def main() -> int:
    module = MODULE.read_text(encoding="utf-8")
    cuda_tu = CUDA_TU.read_text(encoding="utf-8")
    firmware = FIRMWARE.read_text(encoding="utf-8")
    rung = RUNG.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")

    require(r"SparkTpDeviceCollectiveCreate\(&configuration,\n\t\t&state->tp_device_collective\)",
            module, "module collective create")
    require(r"SparkTpDeviceCollectivePrepareReceiveBf16", module,
            "module mesh receive prepare")
    require(r"configuration\.combine_bf16_function = SparkHy4ModuleCombineBf16;",
            module, "module bf16 combine hook")
    require(r"configuration\.combine_u64_max_function =\n\t\tSparkHy4ModuleCombineU64Max;",
            module, "module u64 combine hook")
    require(r"SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16",
            module, "module allreduce operation")
    require(r"SparkTpDeviceCollectiveDestroy\(&state->tp_device_collective\)",
            module, "module destroy releases the collective")
    require(r"context->tp_collective_topology\.rank_count !=\n\t    context->tp_degree",
            module, "module rejects degree/topology mismatch")
    require(r"context->tp_collective_mesh_addr == 0u",
            module, "module fails closed without a mesh mapping")

    require(r"extern \"C\" cudaError_t SparkHy4LaunchAccumAddBf16\(",
            cuda_tu, "bf16 combine launcher")
    require(r"extern \"C\" cudaError_t SparkHy4LaunchAccumU64Max\(",
            cuda_tu, "u64 combine launcher")
    require(r"\(width & 1u\) != 0u", cuda_tu, "odd-width guard")

    for field in ("tp_degree", "tp_rank", "tp_collective_backend_kind",
                  "tp_collective_identifier", "tp_connect_timeout_milli",
                  "tp_operation_timeout_milli",
                  "tp_collective_control_port_base",
                  "tp_collective_mesh_addr", "tp_collective_topology",
                  "tp_collective_backend_module_path"):
        require(rf"\b{field};", firmware, f"context field {field}")
    require(r"#include \"sparkpipe/spark_tp_device_collective.h\"", firmware,
            "firmware collective include")

    require(r"SparkTpDeviceCollectiveEnqueue\(&rank->collective,\n\t\t\t&submission,operation\)",
            rung, "rung enqueues through the engine")
    require(
        r"SPARK_TP_DEVICE_COLLECTIVE_SUBMISSION_STREAM_ORDERED_COMPLETION",
        rung, "rung stream-ordered completion")
    require(r"SPARK_HY4_TP16_RUNG_MESH_SLOT_BYTES \(16u \* 1024u \* 1024u\)",
            rung, "rung mesh slot bytes pin")
    require(r"SPARK_HY4_TP16_RUNG_MESH_SLOTS_PER_BAND 32u", rung,
            "rung mesh slots-per-band pin")
    require(r"SPARK_HY4_TP16_RUNG_MESH_BANDS 4u", rung, "rung mesh band pin")
    require(r"SPARK_HY4_TP16_RUNG_MESH_REGION_BYTES", rung, "rung mesh mapping")
    require(r"SparkHy4Tp16RungMeshSequencer", rung, "rung mesh sequencer")
    require(r"SPARK_HY4_TP16_RUNG_RANKS SPARK_TP_DEVICE_COLLECTIVE_MAX_DEGREE",
            rung, "rung pinned to engine degree 16")
    assert "getenv(\"SPARK_WEIGHTD_SOCKET\")" not in rung, (
        "rung must let the engine read the socket contract")

    require(r"make build/sparkpipe_weightd", runner, "runner builds the daemon")
    require(r"--device-bytes-max 1073741824", runner,
            "runner bounds the daemon arena")
    require(r"SPARK_WEIGHTD_SOCKET=", runner, "runner exports the socket")

    serving_rung = SERVING_RUNG.read_text(encoding="utf-8")
    serving_runner = SERVING_RUNNER.read_text(encoding="utf-8")
    require(r"SparkWeightdAttachRequested\(\)", serving_rung,
            "serving rung gates on the attach request contract")
    require(r"SparkWeightdLazyPackCreate\(", serving_rung,
            "serving rung attaches through the weightd lazy pack")
    require(r"pack->attached\.mesh_send_buffer_addr", serving_rung,
            "serving rung sources the mesh addr from the attach reply")
    require(r"pack->attached\.mesh_ready", serving_rung,
            "serving rung reads mesh readiness from the attach reply")
    require(r"SparkHy4ServingRungFillNodeContext\(&context,attach_mesh_addr\)",
            serving_rung, "serving rung fills tp fields from the attach")
    require(r"case 3u: context->tp_collective_mesh_addr = 0u; break;",
            serving_rung, "serving rung proves the zero-mesh fail-closed path")
    require(r"\"attach_mesh_addr_module_init\",\n\t\tattach_mesh_ready != 0u \? SPARK_STATUS_OK :\n\t\tSPARK_STATUS_INVALID_ARGUMENT,got\)",
            serving_rung, "serving rung expects init only on a live mesh")
    require(r"\"engine_unavailable_fail_closed\",\n\t\tSPARK_STATUS_UNSUPPORTED,got\)",
            serving_rung, "serving rung fails closed without the engine socket")
    require(r"\"engine_backed_module_init\",SPARK_STATUS_OK,got\)",
            serving_rung, "serving rung proves init is engine-backed")
    require(r"\"degree_one_dormant\"", serving_rung,
            "serving rung pins the dormant degree-1 contract")
    require(r"\"identifier_zero_dormant\"", serving_rung,
            "serving rung pins the dormant identifier contract")
    for field in ("tp_degree", "tp_rank", "tp_collective_backend_kind",
                  "tp_collective_identifier", "tp_connect_timeout_milli",
                  "tp_operation_timeout_milli",
                  "tp_collective_control_port_base",
                  "tp_collective_mesh_addr", "tp_collective_topology",
                  "tp_collective_backend_module_path"):
        pattern = r"context->tp_collective_topology\.rank_count =" \
            if field == "tp_collective_topology" \
            else rf"context->{field} ="
        require(pattern, serving_rung,
                f"serving rung fills context field {field}")
    require(r"libhy4_resident_decode_stage\.a", serving_runner,
            "serving runner links the rules.mk module archive")
    require(r"cc -shared -fPIC -Wl,--no-undefined", serving_runner,
            "serving runner links a complete module .so")
    require(r"libsparkpipe_transport\.a", serving_runner,
            "serving runner resolves the collective engine into the .so")
    require(r"grep -q \"T SparkTpDeviceCollectiveCreate\" serving_so_symbols",
            serving_runner, "serving runner proves the collective path .so")
    require(r"TP16_SERVING_LINK_ONLY", serving_runner,
            "serving runner stages link-only mode")
    assert "SparkTpDeviceCollectiveEnqueue" not in serving_rung, (
        "serving rung must not drive engine submissions before Execute")
    assert "//" not in serving_rung, "line comment in serving_rung"
    assert "/*" not in serving_rung and "*/" not in serving_rung, (
        "block comment in serving_rung")
    assert not re.search(r"^\s*#(?!!)", serving_runner, re.M), (
        "shell comment in serving_runner")

    for label, text in (("cuda", cuda_tu), ("rung", rung)):
        for marker in ("//", "/*", "*/"):
            assert marker not in text, f"comment marker {marker!r} in {label}"
    assert "//" not in module, "line comment in module"
    assert module.count("/*") == 1, "new comment block in module"
    print("test_hy4_tp16_source: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
