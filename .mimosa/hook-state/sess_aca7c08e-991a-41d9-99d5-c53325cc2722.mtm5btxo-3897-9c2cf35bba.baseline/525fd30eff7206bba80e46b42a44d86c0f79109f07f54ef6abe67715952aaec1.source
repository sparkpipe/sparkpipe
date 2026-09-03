#!/usr/bin/env python3
"""Source contracts for sealed, prewarmed DSV4 TP graph islands."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE = (ROOT / "modules/dsv4_resident_decode_stage/source/"
          "spark_dsv4_resident_decode_stage_module.c").read_text()
FIRMWARE = (ROOT / "modules/dsv4_resident_decode_stage/include/sparkpipe/"
            "spark_dsv4_resident_decode_stage_firmware.h").read_text()
ADAPTER = (ROOT / "modules/dsv4_resident_decode_stage/source/"
           "spark_dsv4_serving_adapter.c").read_text()
CONFIG = (ROOT / "examples/deployments/"
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


assert "return(3u * local_layer_count + 1u);" in FIRMWARE
assert "width == 1u || width == 8u || width == 16u || width == 32u || width == 64u || width == 1024u" in FIRMWARE
assert '"cuda_graph_count_by_pp_stage"' in ADAPTER
assert '"cuda_graph_count_by_pp_stage": [34, 34, 34, 31]' in CONFIG

configure = body("static SparkStatus SparkDsv4ModuleConfigure(")
assert "context->cuda_graph_count !=" in configure
assert "SparkDsv4ResidentDecodeStageGraphIslandsPerSlot(context->layer_count)" in configure

prewarm = body("static SparkStatus SparkDsv4ModulePrewarmTpGraphs(")
assert "state->pipeline_slot_count" in prewarm
assert "state->tp_graph_islands_per_slot" in prewarm
assert "SparkDsv4ModuleCaptureTpIsland" in prewarm
assert "state->tp_graphs_sealed = 1u;" in prewarm
assert "cudaGraphLaunch" not in prewarm
assert "tp_graph_program" not in MODULE
assert "SparkTpDeviceCollectiveGraphFenceHost" not in MODULE
assert "SparkDsv4LaunchWaitSystemU32" not in MODULE
assert "cuStreamWaitValue32" not in MODULE

# Re-pinned for the 1e38389 stage-module lifecycle cutover: the
# SparkDsv4ModulePrepareState entry became the lifecycle prepare hook
# SparkDsv4ModulePrepare; the prewarm-the-islands requirement is unchanged.
prepare = body("static SparkStatus SparkDsv4ModulePrepare(")
assert prepare.index("SparkDsv4ModulePrewarmTpGraphs(state)") < len(prepare)

start = body("static SparkStatus SparkDsv4ModuleStartLayers(")
continue_name = "static void SparkDsv4ModuleContinueLayers("
continue_layers = body(continue_name, MODULE.index(continue_name) + 1)
finish = body("static SparkStatus SparkDsv4ModuleFinishFrameContinuation(",
              MODULE.index("static SparkStatus SparkDsv4ModuleRunFrame("))
assert "SparkDsv4ModuleReplayTpIsland" in start
assert continue_layers.count("SparkDsv4ModuleReplayTpIsland") >= 1
assert "SparkDsv4ModuleReplayTpIsland" in finish

projection = body("static SparkStatus SparkDsv4ModuleLaunchTpProjectionIsland(")
attention = body("static SparkStatus SparkDsv4ModuleLaunchTpAttentionIsland(")
assert "SparkDsv4ModuleBeginStreams(state,slot,0" in projection
assert "SparkDsv4ModuleRunProjectionShards" in projection
assert "SparkDsv4ModuleRunAttentionProjected" in attention
run_frame = body("static SparkStatus SparkDsv4ModuleRunFrame(")
embedding_guard = ("prefill == 0 && state->tp_degree > 1u && "
                   "state->owns_embedding != 0u")
assert embedding_guard in run_frame
assert "tp_graph_boundary_in" in run_frame

replay = body("static SparkStatus SparkDsv4ModuleReplayTpIsland(",
		      MODULE.index("static SparkStatus SparkDsv4ModuleLaunchTpProjectionIsland("))
assert "state->tp_graphs_sealed == 0u" in replay
assert "rows != SPARK_BATCH_BUCKET" in replay
assert "cudaGraphLaunch" in replay

execute = body("static SparkStatus SparkDsv4ModuleExecuteFrame(")
assert "state->tp_graphs_sealed == 0u" in execute

# Neither TP nor TP1 may silently drop to eager execution after a graph error.
for tp_body in (start, continue_layers, finish):
    assert "SparkDsv4ModuleCaptureDecode" not in tp_body
    assert "SparkDsv4ModuleRunCapturedDecode" not in tp_body
capture_decode = body("static SparkStatus SparkDsv4ModuleCaptureDecode(")
graphed_decode = body("static SparkStatus SparkDsv4ModuleRunGraphedDecode(")
assert capture_decode.count("SparkDsv4ModuleRunCapturedDecode") == 1
assert "return(SparkDsv4ModuleRunCapturedDecode" not in capture_decode
assert "SparkDsv4ModuleRunCapturedDecode" not in graphed_decode

print("dsv4_tp_graph_islands_source: ok")
