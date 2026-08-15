#!/usr/bin/env python3
"""Source contracts for sealed two-graph DSV4 TP decode programs."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE = (ROOT / "modules/dsv4_resident_decode_stage/source/"
          "spark_dsv4_resident_decode_stage_module.c").read_text()
FIRMWARE = (ROOT / "modules/dsv4_resident_decode_stage/include/sparkpipe/"
            "spark_dsv4_resident_decode_stage_firmware.h").read_text()
ADAPTER = (ROOT / "modules/dsv4_resident_decode_stage/source/"
           "spark_dsv4_serving_adapter.c").read_text()
TP4_CONFIG = (ROOT / "examples/deployments/"
              "dsv4_flash_tp4_stage.json").read_text()
TP4_PP4_CONFIG = (ROOT / "examples/deployments/"
                  "dsv4_flash_tp4_pp4_stage.json").read_text()


def body(name: str, start_at: int = 0) -> str:
    start = MODULE.index(name, start_at)
    opening = MODULE.index("{", start)
    depth = 1
    cursor = opening + 1
    while depth:
        depth += (MODULE[cursor] == "{") - (MODULE[cursor] == "}")
        cursor += 1
    return MODULE[opening:cursor]


assert "SPARK_DSV4_RESIDENT_DECODE_STAGE_TP_PROGRAM_GRAPH_COUNT 2u" in FIRMWARE
assert "SparkDsv4ResidentDecodeStageTpProgramGraphsPerSlot" in FIRMWARE
assert "return(SPARK_DSV4_RESIDENT_DECODE_STAGE_TP_PROGRAM_GRAPH_COUNT);" in FIRMWARE
assert "GraphIslandsPerSlot" not in FIRMWARE
assert '"cuda_graph_count_by_pp_stage"' in ADAPTER
assert '"cuda_graph_count_by_pp_stage": [2]' in TP4_CONFIG
assert '"cuda_graph_count_by_pp_stage": [2, 2, 2, 2]' in TP4_PP4_CONFIG

configure = body("static SparkStatus SparkDsv4ModuleConfigure(")
assert "context->cuda_graph_count !=" in configure
assert "SparkDsv4ResidentDecodeStageTpProgramGraphsPerSlot" in configure

prewarm = body("static SparkStatus SparkDsv4ModulePrewarmTpPrograms(")
assert "state->pipeline_slot_count" in prewarm
assert "SparkDsv4ModulePrepareTpDecodeProgram" in prewarm
assert "state->tp_decode_programs_sealed = 1u;" in prewarm
assert "cudaGraphLaunch" not in prewarm

capture = body("static SparkStatus SparkDsv4ModuleCaptureTpDecodeGraphs(")
assert capture.count("cudaStreamBeginCapture") == 2
assert "SparkDsv4ModuleCaptureProducerBody" in capture
assert "SparkDsv4ModuleCaptureConsumerBody" in capture
assert "program->producer_executable" in capture
assert "program->consumer_executable" in capture

prepare = body("static SparkStatus SparkDsv4ModulePrepareState(")
assert "SparkDsv4ModulePrewarmTpPrograms(state)" in prepare

start_name = "static SparkStatus SparkDsv4ModuleStartTpDecodeProgram("
start = body(start_name, MODULE.rindex(start_name))
assert start.count("cudaGraphLaunch") == 2
assert "SparkTpDeviceCollectiveStartProgram" in start
assert "SparkTpDeviceCollectiveSubmitBf16" not in start

for legacy in (
        "TP_GRAPH_ISLAND", "GraphIslandsPerSlot", "LaunchTpIsland",
        "ReplayTpIsland", "PrewarmTpGraphs", "CaptureTpIsland"):
    assert legacy not in MODULE
assert "island" not in MODULE.lower()

print("dsv4_tp_program_graph_source: ok")
