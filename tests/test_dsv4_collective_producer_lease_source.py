#!/usr/bin/env python3
"""Source contract for transport-owned reusable mixed-reduction programs."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = (ROOT / "include/sparkpipe/spark_tp_device_collective.h").read_text()
TRANSPORT = (ROOT / "ring/transport/tp_device_collective.c").read_text()
FIRMWARE = (ROOT / "modules/dsv4_resident_decode_stage/include/sparkpipe/"
            "spark_dsv4_resident_decode_stage_firmware.h").read_text()
CUDA = (ROOT / "modules/dsv4_resident_decode_stage/source/"
        "spark_dsv4_resident_decode_stage_cuda.cu").read_text()
MODULE = (ROOT / "modules/dsv4_resident_decode_stage/source/"
          "spark_dsv4_resident_decode_stage_module.c").read_text()
CHARACTERIZER = (ROOT / "tools/hardware/"
                 "spark_tp_device_collective_characterize.cu").read_text()


assert "SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION 15u" in HEADER
assert "SparkTpDeviceCollectiveProducerLease" in HEADER
assert "SPARK_TP_DEVICE_COLLECTIVE_PRODUCER_SPAN_COUNT 2u" in HEADER
for function in (
        "SparkTpDeviceCollectiveAcquireBf16ProducerLease",
        "SparkTpDeviceCollectiveRecordBf16ProducerReady",
        "SparkTpDeviceCollectiveCancelBf16ProducerLease",
        "SparkTpDeviceCollectiveSubmitLeasedBf16"):
    assert function in HEADER
    assert function in TRANSPORT

assert "SEND_PREPACKED" not in HEADER
assert "SEND_PREPACKED" not in TRANSPORT
assert "producer_prepacked" in TRANSPORT
assert "SparkTpDeviceCollectiveBuildProducerLease" in TRANSPORT
assert "SparkTpDeviceCollectiveValidateProducerLease" in TRANSPORT
assert "producer_ready_event" in HEADER
assert "consumer_ready_event" in HEADER
assert "cudaStreamWaitEvent" in TRANSPORT

assert "SPARK_DSV4_EXACT_BF16_MIRROR_SPAN_COUNT 2u" in FIRMWARE
assert "SparkDsv4ExactBf16MirrorTarget" in FIRMWARE
assert "SparkDsv4LaunchExactBf16ProducerMirror" in CUDA
mirror_start = CUDA.index("static __global__ void SparkDsv4ExactBf16MirrorKernel")
mirror_end = CUDA.index("static cudaError_t SparkDsv4ExactBf16MirrorValidate")
mirror = CUDA[mirror_start:mirror_end]
assert "destination[word_index] = source[offset_words + word_index];" in mirror
assert "Bf16ToFloat" not in mirror
assert "FloatToBf16" not in mirror
assert MODULE.count("SparkDsv4LaunchExactBf16ProducerMirror") == 3
assert "SparkDsv4LaunchAccumU64MaxTp4" in CUDA
assert "SparkDsv4LaunchAccumU64MaxTp4" in MODULE
assert "SparkTpDeviceCollectiveSubmitBf16" not in MODULE
assert "SparkTpDeviceCollectiveSubmitU64Max" not in MODULE
assert "SparkDsv4ModuleReplayTpIsland" not in MODULE
assert "SparkDsv4ModulePrewarmTpPrograms" in MODULE
assert "SparkDsv4ModuleCaptureProducerBody" in MODULE
assert "SparkDsv4ModuleCaptureConsumerBody" in MODULE

assert "collective-send-producer" not in CHARACTERIZER
assert "SUBMISSION_SEND_PREPACKED" not in CHARACTERIZER
assert "SparkTpDeviceCollectiveAcquireBf16ProducerLease" in CHARACTERIZER
assert "SparkTpDeviceCollectiveRecordBf16ProducerReady" in CHARACTERIZER
assert "SparkTpDeviceCollectiveSubmitLeasedBf16" in CHARACTERIZER

for function in (
        "SparkTpDeviceCollectivePrepareProgram",
        "SparkTpDeviceCollectiveGetProgramProducerStep",
        "SparkTpDeviceCollectiveGetProgramConsumerStep",
        "SparkTpDeviceCollectiveGetProgramConsumerPhase",
        "SparkTpDeviceCollectiveStartProgram",
        "SparkTpDeviceCollectiveCancelProgram",
        "SparkTpDeviceCollectiveRearmProgram",
        "SparkTpDeviceCollectiveDestroyProgram"):
    assert function in HEADER
    assert function in TRANSPORT
assert "producer_ready_host" in HEADER
assert "receive_ready_host" in HEADER
assert "result_ready_host" in HEADER
assert "send_reuse_host" in HEADER
assert "run_state_host" in HEADER
assert "storage_bytes" in HEADER
assert "SparkTpDeviceCollectiveProgressProgram" in TRANSPORT
operation_stream_start = TRANSPORT.index(
    "static void *SparkTpDeviceCollectiveOperationStream(")
operation_stream_end = TRANSPORT.index(
    "static uint32_t SparkTpDeviceCollectiveOperationsAreDrained(",
    operation_stream_start)
operation_stream = TRANSPORT[operation_stream_start:operation_stream_end]
assert "operation->program != 0" in operation_stream
assert "SparkTpDeviceCollectivePublishProgramCompletion" in TRANSPORT
assert "SparkTpDeviceCollectivePublishProgramTerminalCallback" in TRANSPORT
assert "SPARK_TP_DEVICE_COLLECTIVE_PROGRAM_DRAINING" in TRANSPORT
assert "SPARK_TP_DEVICE_COLLECTIVE_PROGRAM_TERMINATING" in TRANSPORT
assert "SparkTpDeviceCollectiveValidateProgramReceiveAliases" in TRANSPORT
assert "SparkDsv4ModuleCaptureSplitConsumerPhase" in MODULE
assert "SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_MAX_U64" in HEADER
assert "combine_tp4_u64_max_function" in HEADER
prepare_start = TRANSPORT.index("SparkTpDeviceCollectivePrepareProgram(")
prepare_end = TRANSPORT.index("SparkTpDeviceCollectiveGetProgramProducerStep(")
prepare = TRANSPORT[prepare_start:prepare_end]
assert "calloc(" not in prepare
assert "free(" not in prepare
progress_start = TRANSPORT.rindex(
    "static void SparkTpDeviceCollectiveProgressProgram(")
progress_end = TRANSPORT.index("SparkTpDeviceCollectiveAcquireBf16ProducerLease(")
progress = TRANSPORT[progress_start:progress_end]
assert "SparkTpDeviceCollectiveActivateProgramStep" in progress
assert "SparkTpDeviceCollectiveAcquireBf16ProducerLease" not in progress
assert "SparkTpDeviceCollectiveSubmitHidden" not in progress
consume_start = TRANSPORT.index(
    "static void SparkTpDeviceCollectiveProgressConsumption(")
consume_end = TRANSPORT.index(
    "static void SparkTpDeviceCollectiveReportProfile(", consume_start)
consume = TRANSPORT[consume_start:consume_end]
assert consume.index("SparkTpDeviceCollectiveStateHasFailure(state_word)") < \
    consume.index("result_ready_host")
completion_start = TRANSPORT.index(
    "static void SparkTpDeviceCollectivePublishProgramCompletion(")
completion_end = TRANSPORT.index(
    "static void SparkTpDeviceCollectivePublishCompletion(", completion_start)
completion = TRANSPORT[completion_start:completion_end]
assert "completion->status == SPARK_STATUS_OK &&" in completion
assert "SparkTpDeviceCollectiveSetProgramTerminalStep" in completion
assert 'strcmp(argv[12],"rolling-program")' in CHARACTERIZER
assert "terminal_callbacks" in CHARACTERIZER
assert "producer.completion_generation" in CHARACTERIZER
assert "SparkTpDeviceCollectiveGetProgramConsumerPhase" in CHARACTERIZER

start_start = MODULE.rindex("static SparkStatus SparkDsv4ModuleStartTpDecodeProgram(")
start_end = MODULE.index("static SparkStatus SparkDsv4ModuleRunPrefillHead(")
start = MODULE[start_start:start_end]
for forbidden in (
        "malloc(", "calloc(", "realloc(", "free(", "cudaMalloc(",
        "cudaFree(", "cudaMemcpy", "cudaGraphInstantiate(",
        "cudaGraphExecKernelNodeSetParams("):
    assert forbidden not in start
assert "SparkTpDeviceCollectiveRearmProgram" in start
assert "SparkTpDeviceCollectiveStartProgram" in start
assert start.count("cudaGraphLaunch(") == 2
launch_cancel = start.index("SparkDsv4ModuleCancelTpProgramLaunch")
assert "status = SPARK_STATUS_OK;" in start[launch_cancel:]
assert "#if" not in MODULE

print("dsv4_collective_producer_lease_source: ok")
