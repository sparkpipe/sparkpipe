#!/usr/bin/env python3
"""Contracts for the DSV4 asynchronous TP continuation lifetime."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "modules/dsv4_resident_decode_stage/source/"
          "spark_dsv4_resident_decode_stage_module.c").read_text(encoding="utf-8")


def body(name: str, start_at: int = 0) -> str:
    start = SOURCE.index(name, start_at)
    opening = SOURCE.index("{", start)
    depth = 1
    cursor = opening + 1
    while depth:
        depth += (SOURCE[cursor] == "{") - (SOURCE[cursor] == "}")
        cursor += 1
    return SOURCE[opening:cursor]


continuation = SOURCE[SOURCE.index("struct SparkDsv4TpFrameContinuation"):]
continuation = continuation[:continuation.index("};")]
assert "SparkDsv4PrefillBatchView" not in continuation
assert "SparkModelDriverFrame" not in continuation
assert "SparkDsv4ResidentDecodeStageFrameContext" not in continuation

run_frame = body("static SparkStatus SparkDsv4ModuleRunFrame")
promotion = "if ( prefill != 0 && state->tp_degree > 1u )"
assert promotion in run_frame
assert run_frame.index(promotion) < run_frame.index("SparkDsv4ModuleBeginStreams")
assert "rows != SPARK_BATCH_BUCKET" in run_frame
# The speculative wave stages single-lane prefills to bucket width, so the
# rows gate is the only full-width check left; the per-sequence gate is gone.
assert "active_sequence_count != SPARK_BATCH_BUCKET" not in run_frame
assert run_frame.count("!= SPARK_BATCH_BUCKET") == 1
assert "prefill = 0;" in run_frame
assert run_frame.index("SparkDsv4ModuleRunLocalLayers") < run_frame.index(
    "continuation = slot->tp_continuation")
assert "continuation->prefill" not in SOURCE

initialize_collective = body("static SparkStatus SparkDsv4ModuleInitializeTpCollective")
assert "SPARK_DSV4_TP_COLLECTIVE_CREDITS_PER_SLOT" in initialize_collective
assert "configuration.credit_count = state->pipeline_slot_count *" in \
    initialize_collective

validate_shape = body("static SparkStatus SparkDsv4ModuleValidateFrameShape")
admission_shape = body("static uint32_t SparkDsv4ModuleAdmissionShapeSupported")
for shape in (validate_shape, admission_shape):
    # The speculative wave re-gates the full-width check in equality form:
    # (active == BUCKET && new == BUCKET) or the bucket-8 DSpark lane case.
    # ValidateFrame uses frame->, AdmissionShapeSupported uses request->.
    assert "active_slot_count == SPARK_BATCH_BUCKET &&" in shape
    assert "new_token_count == SPARK_BATCH_BUCKET)" in shape
    assert "is_prefill != 0u ||" not in shape

continue_name = "static void SparkDsv4ModuleContinueLayers"
continue_layers = body(continue_name, SOURCE.index(continue_name) + 1)
assert continue_layers.count("SparkDsv4ModuleFinishContinuationTerminal(") == 2

terminal = body("static void SparkDsv4ModuleFinishContinuationTerminal")
assert "enqueue_status = SparkDsv4ModuleEnqueueAsync" in terminal
assert "if ( enqueue_status != SPARK_STATUS_OK )" in terminal
assert "SparkDsv4ModuleCompleteAfterFailedEnqueue" in terminal

finish = body("static SparkStatus SparkDsv4ModuleFinishFrameContinuation",
              SOURCE.index("static SparkStatus SparkDsv4ModuleRunFrame"))
assert "SparkDsv4ModuleEnqueueAsync" not in finish

fallback_name = "static void SparkDsv4ModuleCompleteAfterFailedEnqueue"
fallback = body(fallback_name, SOURCE.index(fallback_name) + 1)
assert "SparkDsv4ModuleSynchronizeFailedSlot(slot)" in fallback
assert fallback.index("SparkDsv4ModuleSynchronizeFailedSlot(slot)") < \
    fallback.index("SparkDsv4ModuleCompleteAsync(async)")
assert "SparkStageModuleCompleteAndReleaseClaims" not in fallback
assert "async->completion_function = 0;" in fallback

failure_sync_name = "static SparkStatus SparkDsv4ModuleSynchronizeFailedSlot"
failure_sync = body(failure_sync_name, SOURCE.index(failure_sync_name) + 1)
assert failure_sync.count("cudaStreamSynchronize(") == 1

print("dsv4_tp_async_source: ok")
