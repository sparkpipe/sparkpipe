#!/usr/bin/env python3
"""Enforce the model-neutral resident and model-owned adapter boundary."""

from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NEUTRAL_FILES = (
    "include/sparkpipe/spark_model_driver.h",
    "include/sparkpipe/spark_model_driver_support.h",
    "include/sparkpipe/spark_module_abi.h",
    "include/sparkpipe/spark_model_serving_adapter.h",
    "include/sparkpipe/spark_model_resident_ipc.h",
    "include/sparkpipe/spark_model_resident_endpoint.h",
    "include/sparkpipe/spark_model_resident_deployment.h",
    "include/sparkpipe/spark_model_resident_client.h",
    "include/sparkpipe/spark_model_pipeline_client.h",
    "include/sparkpipe/spark_model_batch_engine.h",
    "include/sparkpipe/spark_kv_cache.h",
    "include/sparkpipe/spark_kv_page_cache.h",
    "include/sparkpipe/spark_kv_page_store.h",
    "include/sparkpipe/spark_prefix_cache.h",
    "include/sparkpipe/spark_pipeline_runtime.h",
    "runtime/model_serving_adapter.c",
    "runtime/model_resident_ipc.c",
    "runtime/model_resident_endpoint.c",
    "runtime/model_resident_deployment.c",
    "runtime/model_resident_client.c",
    "runtime/model_pipeline_client.c",
    "runtime/model_batch_engine.c",
    "runtime/model_batch_scheduler.h",
    "runtime/pipeline_runtime.c",
    "cache/kv_cache.c",
    "cache/kv_page_cache.c",
    "cache/kv_page_store.c",
    "cache/prefix_cache.c",
    "node/model_residentd.c",
)
MODEL_TOKEN = re.compile(
    r"glm(?:52)?|dsv4|deepseek|qwen|kimi|mimo|dspark|mtp|moe|\bmla\b",
    re.IGNORECASE,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    for relative in NEUTRAL_FILES:
        text = (ROOT / relative).read_text(encoding="utf-8")
        match = MODEL_TOKEN.search(text)
        if match is not None:
            require(False, f"{relative} contains model token {match.group()!r}")

    for makefile in sorted((ROOT / "modules").glob("*/Makefile")):
        text = makefile.read_text(encoding="utf-8")
        require(
            "REPOSITORY_ROOT := $(abspath" not in text,
            f"{makefile.relative_to(ROOT)} embeds a space-unsafe target root",
        )
    module_rules = (ROOT / "modules/resident_decode_stage_rules.mk").read_text(
        encoding="utf-8"
    )
    require(
        "MODULE_POSIX_FLAGS := -D_GNU_SOURCE" in module_rules,
        "forced model headers hide GNU production APIs from module sources",
    )

    sources = (ROOT / "sources.mk").read_text(encoding="utf-8")
    common = sources.split("SPARKPIPE_MODEL_COMMON_SOURCES :=", 1)[1].split(
        "SPARKPIPE_GLM52_SOURCES :=", 1
    )[0]
    require(
        "scheduler/stage_plan.c" not in common,
        "model-neutral runtime still links the legacy routed-layer planner",
    )
    require(
        "scheduler/stage_plan.c" not in sources,
        "deleted legacy stage planner is still linked by a model path",
    )
    serving_header = (
        ROOT / "include/sparkpipe/spark_model_serving_adapter.h"
    ).read_text(encoding="utf-8")
    serving_source = (ROOT / "runtime/model_serving_adapter.c").read_text(
        encoding="utf-8"
    )
    pipeline_runtime = (ROOT / "runtime/pipeline_runtime.c").read_text(
        encoding="utf-8"
    )
    require(
        "spark_stage_plan.h" not in serving_header
        and "first_routed_layer" not in serving_header
        and "SparkStagePlan" not in pipeline_runtime,
        "model-neutral serving still inherits legacy stage-cut policy",
    )

    resident = (ROOT / "node/model_residentd.c").read_text(encoding="utf-8")
    deployment_source = (ROOT / "runtime/model_resident_deployment.c").read_text(
        encoding="utf-8"
    )
    require(
        'strcmp(deployment->transport_mode,"host-rdma")' in deployment_source
        and 'strcmp(deployment->transport_mode,"gpudirect-rdma")'
        in deployment_source,
        "deployment transport modes are not explicit",
    )
    require('strcmp(mode,"host-rdma")' in resident, "host RDMA mode missing")
    require('strcmp(mode,"gpudirect-rdma")' in resident, "GPUDirect mode missing")
    require('strcmp(mode,"device")' not in resident, "device alias reintroduces fallback")
    require(
        "SPARKPIPE_RING_TRANSPORT_PORT_BASE" not in resident,
        "resident uses an obsolete transport port variable",
    )
    require(
        "SparkModelResidentdApplyTransportCompletionLocked" in resident,
        "resident does not correlate transport completion before slot reuse",
    )
    require(
        resident.count("SparkModelServingAdapterValidateRuntimeSubmission") >= 2,
        "resident acknowledges work before enforcing configured runtime limits",
    )
    require(
        "SparkModelServingAdapterPrepareSubmission" in resident
        and resident.index("SparkModelServingAdapterPrepareSubmission")
        < resident.index("route = SparkModelResidentdReserveRoute"),
        "model-owned preflight does not run before route reservation",
    )
    require(
        "SPARK_MODEL_RESIDENTD_ROUTE_RESERVED" in resident
        and "SPARK_MODEL_RESIDENT_IPC_KIND_PREPARE" in resident
        and "SPARK_MODEL_RESIDENT_IPC_KIND_DECISION" in resident
        and "SparkModelResidentIpcInitializeDecisionResult" in resident,
        "resident can execute before pipeline-wide admission commits",
    )
    submission_path = resident.split(
        "static SparkStatus SparkModelResidentdProcessSubmission", 1
    )[1].split("static SparkStatus SparkModelResidentdProcessDecision", 1)[0]
    require(
        "SparkModelResidentIpcValidateDirectSubmitDescriptor" in submission_path
        and submission_path.index(
            "SparkModelResidentIpcValidateDirectSubmitDescriptor"
        )
        < submission_path.index(
            "SparkModelServingAdapterValidateRuntimeSubmission"
        )
        < submission_path.index("SparkModelServingAdapterPrepareSubmission"),
        "distributed direct SUBMIT reaches adapter/cache admission",
    )
    decision_path = resident.split(
        "static SparkStatus SparkModelResidentdProcessDecision", 1
    )[1].split("static SparkStatus SparkModelResidentdProcessMessage", 1)[0]
    require(
        "SPARK_MODEL_RESIDENTD_ROUTE_RESOLVING" in decision_path
        and "pthread_mutex_unlock(&runtime->mutex);" in decision_path
        and decision_path.index("pthread_mutex_unlock(&runtime->mutex);")
        < decision_path.index("SparkModelServingAdapterResolvePrefetch")
        < decision_path.index("SparkModelResidentdEnqueueCommittedLocked"),
        "prepared cache is resolved under the resident mutex or after commit publication",
    )
    require(
        "committed_fifo_head" in resident
        and "SparkModelResidentdEnqueueCommittedLocked" in resident
        and "SparkModelResidentdRemoveCommittedLocked" in resident,
        "resident lacks strict committed adapter-entry FIFO ownership",
    )
    require(
        "SparkModelResidentdQueueDeadlineCompletionLocked" in resident
        and "deadline_completion_queued" in resident
        and "deadline_wait_state" in resident
        and "stay quarantined until transport reports a terminal" in resident,
        "transport expiry can hang completion or recycle a live boundary",
    )
    resident_client = (
        ROOT / "runtime/model_resident_client.c"
    ).read_text(encoding="utf-8")
    pipeline_client = (
        ROOT / "runtime/model_pipeline_client.c"
    ).read_text(encoding="utf-8")
    require(
        "SparkModelResidentClientCanQueueDecision" in resident_client
        and pipeline_client.count("SparkModelResidentClientCanQueueDecision") >= 2,
        "pipeline decisions can partially publish before all-rank preflight",
    )
    require(
        "packet->sequence_id = submission->submission_id" in resident
        and "packet->token_index = submission->dispatch_generation" in resident,
        "transport packet identity is not derived from the global submission",
    )
    require(
        "route->submission.hidden_input_address" in resident
        and "route->submission.hidden_output_address" in resident,
        "resident does not own the model boundary pointers",
    )
    require(
        "SparkModelServingAdapterValidateRuntimeLimits" in resident
        and "runtime->route_capacity = "
        "runtime->runtime_limits.max_inflight_submission_count" in resident
        and "runtime->runtime_limits.max_input_row_count" in resident,
        "resident storage is not bounded by validated runtime limits",
    )
    require(
        "--deployment PATH --rank-index N" in resident
        and "--adapter-so" not in resident
        and "--driver-so" not in resident
        and "--transport-so" not in resident,
        "resident has multiple deployment configuration sources",
    )
    require(
        "transport_host" in deployment_source
        and "host_name_prefix" not in deployment_source,
        "data-plane hosts are inferred instead of declared per node",
    )
    pipeline_runtime = (ROOT / "runtime/pipeline_runtime.c").read_text(
        encoding="utf-8"
    )
    pipeline_runtime_header = (
        ROOT / "include/sparkpipe/spark_pipeline_runtime.h"
    ).read_text(encoding="utf-8")
    require(
        "SPARK_HIDDEN_TRANSPORT_ENDPOINT_FLAG_EXPLICIT_ROUTE_CONFIGURATION"
        in pipeline_runtime
        and "endpoint->local_rank_index = rank_plan->rank_index"
        in pipeline_runtime
        and "endpoint->source_rank_index = source_rank_index" in pipeline_runtime
        and "endpoint->sink_rank_index = sink_rank_index" in pipeline_runtime
        and "endpoint->source_host = source_host" in pipeline_runtime
        and "endpoint->sink_host = sink_host" in pipeline_runtime
        and "endpoint->control_port_base" in pipeline_runtime
        and "setenv(" not in resident,
        "model resident routes transport through process environment",
    )
    require(
        "SparkPipelineRuntimeBuildInputEndpoint" in pipeline_runtime_header
        and "SparkPipelineRuntimeBuildOutputEndpoint" in pipeline_runtime_header
        and "SparkHiddenTransportEndpoint input_endpoint"
        not in pipeline_runtime_header,
        "rank plans embed endpoint pointers and are not safe value objects",
    )
    rdma = (ROOT / "ring/transport/rdma.cu").read_text(encoding="utf-8")
    require(
        "state->source_rank = (int32_t)endpoint->source_rank_index" in rdma
        and "state->sink_rank = (int32_t)endpoint->sink_rank_index" in rdma
        and 'endpoint->source_host' in rdma
        and 'endpoint->sink_host' in rdma,
        "explicit RDMA topology is inferred from host naming",
    )
    require(
        "SparkHiddenSparkHostRdmaRetireCompletedReceives" in rdma
        and "return receive->complete != 0u ? SPARK_STATUS_OK"
        not in rdma
        and "return SparkHiddenSparkHostRdmaFinalizePendingReceive(state,receive);"
        not in rdma,
        "RDMA keeps accepted receive work in caller retry state",
    )
    doorbell_completion = rdma.split(
        "static SparkStatus SparkHiddenSparkHostRdmaApplyDoorbellCompletion", 1
    )[1].split("static SparkHiddenSparkHostRdmaInflightSend", 1)[0]
    require(
        "receive_index = immediate &" in doorbell_completion
        and "completion_receive_index = (uint32_t)(work_completion->wr_id &"
        in doorbell_completion
        and "receive_credit_index = completion_receive_index"
        in doorbell_completion
        and "state,receive_index,remote_receive->generation"
        in doorbell_completion
        and "work_completion->wr_id !=" not in doorbell_completion,
        "RDMA separates FIFO WQE repost ownership from logical immediate credit",
    )
    pipeline_header = (
        ROOT / "include/sparkpipe/spark_model_pipeline_client.h"
    ).read_text(encoding="utf-8")
    pipeline_client = (ROOT / "runtime/model_pipeline_client.c").read_text(
        encoding="utf-8"
    )
    model_batch = (ROOT / "node/model_batch.c").read_text(encoding="utf-8")
    client = (ROOT / "runtime/model_resident_client.c").read_text(
        encoding="utf-8"
    )
    require(
        "const SparkModelResidentDeployment *deployment" in pipeline_header
        and "const char *runtime_root" in pipeline_header
        and "SparkResolveRuntimePath(configuration->runtime_root" in pipeline_client
        and "rank_endpoints" not in pipeline_header
        and "configuration->runtime_limits" not in pipeline_client
        and "configuration->rank_endpoints" not in pipeline_client,
        "pipeline client retains a second deployment configuration source",
    )
    require(
        'strcmp(argv[index],"--runtime-root")' in model_batch
        and "--deployment PATH --runtime-root PATH --batch PATH" in model_batch,
        "batch client does not require its local packaged runtime root",
    )
    require(
        "SparkModelResidentClientPrepare" in pipeline_client
        and "SparkModelResidentClientCommit" in pipeline_client
        and "SparkModelResidentClientAbort" in pipeline_client
        and "for (rank=pipeline->rank_count" in pipeline_client,
        "pipeline admission is not two-phase and downstream-first",
    )
    batch_engine = (ROOT / "runtime/model_batch_engine.c").read_text(
        encoding="utf-8"
    )
    require(
        "SparkModelPipelineClientSubmit" in batch_engine
        and "SPARK_MODEL_SERVING_WORK_KIND_PREFILL" in batch_engine
        and "SPARK_MODEL_SERVING_WORK_KIND_DECODE" in batch_engine
        and "SPARK_MODEL_SERVING_WORK_KIND_RELEASE" in batch_engine
        and "SPARK_MODEL_SERVING_SLOT_REUSE_REQUIRES_RELEASE" in batch_engine,
        "neutral request engine does not drive every adapter lifecycle phase",
    )
    require(
        "SparkService" not in batch_engine
        and "SparkTokenizer" not in batch_engine
        and "getenv(" not in batch_engine
        and "fallback" not in batch_engine.lower(),
        "neutral request engine imports a model application or hidden mode",
    )
    require(
        "SparkModelBatchSchedulerPlanCacheBoundLaneCount" in batch_engine
        and "SparkModelBatchSchedulerCacheDemandFits" in batch_engine
        and "SparkModelBatchCacheDemandAdditional" in batch_engine
        and "request->cache_published_identity" in batch_engine
        and "engine->kv_physical_page_capacity" in batch_engine
        and "engine->inflight_kv_page_count" in batch_engine,
        "neutral scheduler does not account for shared-prefix KV page demand",
    )
    require(
        "runtime/model_batch_engine.c" in sources.split(
            "SPARKPIPE_MODEL_COMMON_SOURCES :=", 1
        )[0],
        "neutral request engine is not part of the runtime library",
    )
    require(
        "transaction->active_sequence_count = submission->active_sequence_count"
        in pipeline_client
        and "SparkModelServingAdapterValidateStageCompletion" in pipeline_client
        and "completion->token_count == active_sequence_count" in serving_source
        and "work_kind == SPARK_MODEL_SERVING_WORK_KIND_RELEASE" in serving_source,
        "pipeline does not enforce terminal token ownership and cardinality",
    )
    require(
        "SparkModelServingAdapterValidateStageCompletion" in serving_header
        and "&transaction->residency" in pipeline_client
        and "&pending->residency" in client
        and "&route->submission.residency" in resident,
        "distributed completion residency is not checked at every boundary",
    )
    for relative in (
        "tests/test_model_resident_end_to_end.c",
        "tests/test_model_pipeline_client.c",
    ):
        process_test = (ROOT / relative).read_text(encoding="utf-8")
        require(
            '"--deployment",deployment_path' in process_test
            and '"--adapter-so"' not in process_test
            and '"--driver-so"' not in process_test
            and '"--transport-so"' not in process_test,
            f"{relative} bypasses the deployment manifest",
        )

    adapter_header = (
        ROOT / "include/sparkpipe/spark_model_serving_adapter.h"
    ).read_text(encoding="utf-8")
    require(
        "SparkHiddenTransport" not in adapter_header,
        "model adapter ABI owns a transport session or callback",
    )
    require(
        "resident_sequence_slot" in adapter_header
        and "SPARK_MODEL_SERVING_SLOT_REUSE_REQUIRES_RELEASE" in adapter_header
        and "SPARK_MODEL_SERVING_SLOT_REUSE_AT_POSITION_ZERO" in adapter_header,
        "persistent KV slot ownership or reuse policy is implicit",
    )
    require(
        "SparkModelServingAdapterValidateSubmissionFunction" in adapter_header
        and "validate_submission" in adapter_header,
        "model adapters cannot preflight model-specific submissions",
    )
    require(
        "SparkModelServingAdapterQuiesceFunction" in adapter_header
        and "quiesce" in adapter_header,
        "model adapters do not expose a mandatory quiescence boundary",
    )
    require(
        "const char *runtime_root" in adapter_header,
        "model adapters do not receive the manifest runtime root",
    )
    require(
        "uint32_t kv_logical_page_capacity;" in adapter_header
        and "uint32_t kv_physical_page_capacity;" in adapter_header
        and '"kv_logical_page_capacity"' in deployment_source
        and '"kv_physical_page_capacity"' in deployment_source
        and "runtime_limits->kv_physical_page_capacity" in serving_source,
        "KV page capacity policy is not a validated neutral runtime limit",
    )
    driver_header = (ROOT / "include/sparkpipe/spark_model_driver.h").read_text(
        encoding="utf-8"
    )
    driver_support = (
        ROOT / "include/sparkpipe/spark_model_driver_support.h"
    ).read_text(encoding="utf-8")
    module_abi = (ROOT / "include/sparkpipe/spark_module_abi.h").read_text(
        encoding="utf-8"
    )
    driver_compiler = (ROOT / "runtime/pack/driver_compiler.c").read_text(
        encoding="utf-8"
    )
    require(
        "uint32_t abi_version;" in driver_header
        and "uint32_t descriptor_bytes;" in driver_header
        and "void *execution_stream;" in driver_header
        and "SparkModelDriverCreateRequestIsValid" in driver_support,
        "model-driver creation is not a versioned resource contract",
    )
    require(
        "void *execution_stream;" in module_abi
        and "host_services.execution_stream = request->execution_stream"
        in driver_compiler
        and "SparkModelDriverCreateRequestIsValid(request)" in driver_compiler,
        "generated drivers do not validate or forward the resident stream",
    )
    require(
        "uint32_t kv_logical_page_capacity;" in driver_header
        and "uint32_t kv_physical_page_capacity;" in driver_header
        and "uint32_t kv_logical_page_capacity;" in module_abi
        and "uint32_t kv_physical_page_capacity;" in module_abi
        and "host_services.kv_logical_page_capacity = request->kv_logical_page_capacity"
        in driver_compiler
        and "host_services.kv_physical_page_capacity = request->kv_physical_page_capacity"
        in driver_compiler,
        "generated driver boundary does not forward neutral KV page budgets",
    )
    require(
        "#define SPARK_MODEL_DRIVER_ABI_VERSION 12u" in driver_header
        and "uint64_t submission_id;" in driver_header
        and "uint64_t control_generation;" in driver_header
        and "uint64_t transaction_id;" in driver_header
        and driver_header.count("uint64_t request_generation;") >= 2
        and driver_header.count("uint64_t step_generation;") >= 2
        and "lane->request_generation == 0u" in driver_support
        and "lane->step_generation == 0u" in driver_support
        and "destination->request_generation = source->request_generation"
        in serving_source
        and "destination->step_generation = source->step_generation"
        in serving_source,
        "prepared driver cache admission lacks collision-safe request identity",
    )

    require(
        "configuration->stage_index != configuration->rank_index" not in client,
        "resident client conflates physical rank with pipeline stage",
    )
    ipc = (ROOT / "runtime/model_resident_ipc.c").read_text(encoding="utf-8")
    for relative, text in (("resident client", client), ("resident IPC", ipc)):
        require(
            "hidden_input_address != 0" in text
            and "hidden_output_address != 0" in text,
            f"{relative} can smuggle process-local boundary pointers",
        )
    require(
        "control_generation" in ipc
        and "transaction_id" in ipc
        and "dispatch_generation" in ipc
        and "wire->residency = submission->residency" in ipc
        and "completion_out->residency = wire->residency" in ipc,
        "resident IPC does not preserve distributed completion identity",
    )
    resident = (ROOT / "node/model_residentd.c").read_text(encoding="utf-8")
    require(
        "adapter_interface.quiesce" in resident
        and resident.index("adapter_interface.quiesce")
        < resident.index("cudaStreamSynchronize")
        < resident.index("SparkHiddenTransportClose")
        < resident.index("adapter_interface.destroy")
        < resident.index("SparkModelServingAdapterUnloadInterface"),
        "resident teardown does not quiesce CUDA and transport before unload",
    )
    require(
        "cudaStream_t execution_stream;" in resident
        and "cudaStream_t transport_stream;" in resident
        and "adapter_configuration.execution_stream = runtime->execution_stream"
        in resident
        and "packet->cuda_stream = runtime->transport_stream" in resident
        and "cudaStreamCreateWithFlags(&runtime->transport_stream,cudaStreamNonBlocking)"
        in resident
        and "cudaStreamSynchronize(runtime->transport_stream)" in resident
        and "cudaStreamDestroy(runtime->transport_stream)" in resident,
        "resident does not isolate model execution from stream-ordered transport",
    )
    drain_routes = "status = SparkModelResidentdProgressRoutes(runtime,0u);"
    submit_route = "status = SparkModelResidentdProgressRoutes(runtime,1u);"
    poll_input = "status = SparkModelResidentdProgressTransport(runtime,runtime->input_transport,1u);"
    poll_output = "status = SparkModelResidentdProgressTransport(runtime,runtime->output_transport,0u);"
    drain_before = resident.index(drain_routes)
    poll_input_before = resident.index(poll_input, drain_before + 1)
    poll_output_before = resident.index(poll_output, poll_input_before + 1)
    drain_middle = resident.index(drain_routes, poll_output_before + 1)
    submit_one = resident.index(submit_route, drain_middle + 1)
    poll_input_after = resident.index(poll_input, submit_one + 1)
    poll_output_after = resident.index(poll_output, poll_input_after + 1)
    drain_after = resident.index(drain_routes, poll_output_after + 1)
    require(
        drain_before < poll_input_before < poll_output_before < drain_middle
        < submit_one < poll_input_after < poll_output_after < drain_after
        and resident.count(drain_routes) == 3
        and resident.count(submit_route) == 1
        and resident.count(poll_input) == 2
        and resident.count(poll_output) == 2
        and "uint32_t next_adapter_route;" in resident
        and "start = allow_adapter != 0u ? runtime->next_adapter_route : 0u;"
        in resident
        and "adapter_submitted == 0u" in resident
        and "adapter_submitted = 0u;" in resident
        and "allow_adapter != 0u ? &adapter_submitted : 0"
        in resident
        and "if ( adapter_submitted == 0 || *adapter_submitted != 0u )\n\t\t\t\treturn(SPARK_STATUS_OK);"
        in resident
        and "if ( status == SPARK_STATUS_BUSY )\n\t\t\t\treturn(SPARK_STATUS_OK);\n\t\t\tif ( status != SPARK_STATUS_OK )\n\t\t\t\treturn(status);\n\t\t\t*adapter_submitted = 1u;\n\t\t\tSparkModelResidentdWake(runtime);"
        in resident,
        "resident reactor does not bound adapter work between transport polls",
    )
    require(
        "SparkModelResidentdClaimResidentSlotsLocked" in resident
        and "SparkModelResidentdReleaseResidentSlotsLocked" in resident
        and "SparkModelResidentdCompleteResidentSlotsLocked" in resident
        and "SparkModelResidentdValidatePersistentSlot" in resident
        and "sequence_slots" in resident,
        "resident does not enforce active and persistent KV slot ownership",
    )

    module = (
        ROOT
        / "modules/dsv4_resident_decode_stage/source/"
        "spark_dsv4_resident_decode_stage_module.c"
    ).read_text(encoding="utf-8")
    require(
        "SparkStageModuleEnvironment" not in module,
        "DSV4 production geometry still comes from process environment",
    )
    require(
        re.findall(r'getenv\("([^"]+)"\)', module) == [],
        "DSV4 production module reads process environment",
    )
    require(
        "SparkDsv4ResidentDecodeStageNodeContext" in module,
        "DSV4 module does not consume its typed node context",
    )
    require(
        "resident_sequence_capacity" in module
        and "state->max_active_sequence_count" not in module,
        "DSV4 still conflates dispatch width with persistent KV capacity",
    )
    require(
        "SparkHiddenTransport" not in module,
        "DSV4 compute module owns pipeline transport",
    )
    require(
        "SPARK_DSV4_STAGE_GRAPHS" not in module,
        "DSV4 module retains a silent graph-selection path",
    )
    require(
        "SparkDsv4ModuleRunCausalAttention" in module
        and "SparkDsv4ModulePrefillWaveRowCount" in module
        and "SparkDsv4ModuleRunFrame(" in module
        and "SparkDsv4ModuleRunFrameWaves" not in module
        and "SparkRowLayoutValidateRoundMajor" in module,
        "DSV4 prefill does not bulk dense work around causal attention waves",
    )
    require(
        "cudaStreamCreate(" not in module
        and module.count("cudaStreamCreateWithFlags") == 1
        and "kv_page_store_stream" in module
        and "cudaDeviceSynchronize" not in module
        and "state->execution_stream = host_services->execution_stream" in module
        and "state->execution_stream != frame->execution_stream" in module,
        "DSV4 execution stream and isolated KV transfer stream are not separated",
    )
    stage_runner = (
        ROOT
        / "modules/dsv4_resident_decode_stage/source/spark_dsv4_stage_runner.c"
    ).read_text(encoding="utf-8")
    require(
        "frame->execution_stream = runner->execution_stream" in stage_runner,
        "DSV4 runner drops the resident-owned CUDA stream",
    )

    for relative in (
        "modules/dsv4_resident_decode_stage/source/spark_dsv4_stage_runner.c",
        "modules/dsv4_resident_decode_stage/source/spark_dsv4_serving_adapter.c",
        "modules/dsv4_resident_decode_stage/include/sparkpipe/"
        "spark_dsv4_resident_decode_stage_firmware.h",
    ):
        text = (ROOT / relative).read_text(encoding="utf-8")
        require(
            "SparkHiddenTransportSession" not in text
            and "SparkHiddenTransportSend" not in text
            and "SparkHiddenTransportPostReceive" not in text,
            f"{relative} crosses the model/transport ownership boundary",
        )

    description_path = (
        ROOT
        / "examples/model_descriptions/"
        "dsv4_resident_decode_stage_firmware.json"
    )
    description = json.loads(description_path.read_text(encoding="utf-8"))
    runtime_contract = description["metadata"]["runtime_contract"]
    require(runtime_contract["required_environment"] == [], "DSV4 requires model env")
    require(runtime_contract["fallback_allowed"] is False, "DSV4 permits fallback")
    require(
        runtime_contract["configuration_source"].startswith("typed "),
        "DSV4 configuration source is not typed",
    )
    require(
        runtime_contract["completion"] == "external"
        and runtime_contract["runtime_backend_selection"] == "forbidden"
        and runtime_contract["runtime_precision_selection"] == "forbidden",
        "DSV4 description does not bind the generalized execution contract",
    )
    require(
        description["stages"][0]["programs"][0]["max_inflight"] == 13,
        "DSV4 driver capacity disagrees with its PP13 serving adapter",
    )
    adapter = (
        ROOT
        / "modules/dsv4_resident_decode_stage/source/"
        "spark_dsv4_serving_adapter.c"
    ).read_text(encoding="utf-8")
    model_header = (
        ROOT
        / "model-families/dsv4/include/sparkpipe/spark_dsv4_model.h"
    ).read_text(encoding="utf-8")
    firmware_header = (
        ROOT
        / "modules/dsv4_resident_decode_stage/include/sparkpipe/"
        "spark_dsv4_resident_decode_stage_firmware.h"
    ).read_text(encoding="utf-8")
    description_sha256 = hashlib.sha256(description_path.read_bytes()).hexdigest()
    require(
        description_sha256 in model_header
        and "SPARK_DSV4_MODEL_DESCRIPTION_SHA256" in adapter,
        "DSV4 generated contract and adapter are not bound to the model description",
    )
    require(
        "resident_row_lane_indices[row] = "
        "submission->lanes[lane].resident_sequence_slot" in adapter,
        "DSV4 adapter does not translate batch lanes to persistent KV slots",
    )
    require(
        "request.kv_logical_page_capacity =" in adapter
        and "request.kv_physical_page_capacity =" in adapter
        and "state->logical_page_capacity = host_services->kv_logical_page_capacity"
        in module
        and "state->physical_page_capacity = host_services->kv_physical_page_capacity"
        in module,
        "DSV4 device layer does not consume neutral KV page budgets",
    )
    deployment_header = (
        ROOT / "include/sparkpipe/spark_model_resident_deployment.h"
    ).read_text(encoding="utf-8")
    require(
        "char *kv_backing_directory;" in deployment_header
        and "uint64_t kv_backing_maximum_bytes;" in deployment_header
        and "configuration->kv_backing_directory = node->kv_backing_directory"
        in resident
        and "getenv(" not in resident,
        "KV backing policy is not owned by the typed generic deployment",
    )
    require(
        "node_context.logical_page_capacity" not in adapter
        and "node_context.physical_page_capacity" not in adapter
        and "physical_page_capacity = state->max_active_sequence_count" not in adapter
        and "uint32_t logical_page_capacity;" not in firmware_header
        and "uint32_t physical_page_capacity;" not in firmware_header,
        "DSV4 adapter or node context owns generic cache sizing policy",
    )
    require(
        "SparkRowLayoutValidateRoundMajor" in adapter,
        "DSV4 adapter does not fail closed on non-wavefront prefill order",
    )
    require(
        "SPARK_MODEL_DRIVER_PROGRAM_FLAG_EXTERNAL_COMPLETION" in adapter
        and "cudaLaunchHostFunc" in module,
        "DSV4 does not use stream-ordered external completion",
    )
    require(
        "SparkResolveRuntimePath" in adapter,
        "DSV4 stage packs can bypass the manifest runtime root",
    )
    require(
        ".quiesce = SparkDsv4ServingQuiesce" in adapter
        and "state->quiescing = 1u" in adapter
        and "snapshot.active_submission_count == 0u" in adapter,
        "DSV4 adapter cannot prove model work is quiescent before unload",
    )
    require(
        "allow_unqualified_execution" not in adapter
        and "ALLOW_UNQUALIFIED" not in adapter,
        "DSV4 production adapter exposes a qualification bypass",
    )
    require(
        "ALLOW_UNQUALIFIED" not in module
        and "NODE_CONTEXT_FLAG_ALLOW_UNQUALIFIED" not in firmware_header,
        "DSV4 module exposes a runtime qualification bypass",
    )
    for config_name in (
        "dsv4_serving_adapter_config.json",
        "dsv4_serving_adapter_config_absolute.json",
        "dsv4_serving_adapter_config_stale.json",
    ):
        config_text = (ROOT / "tests/fixtures" / config_name).read_text(
            encoding="utf-8"
        )
        require(
            "allow_unqualified_execution" not in config_text,
            f"{config_name} exposes a qualification bypass",
        )

    deployment_example = json.loads(
        (
            ROOT
            / "examples/deployments/dsv4_flash_pp13_host_rdma.json"
        ).read_text(encoding="utf-8")
    )
    dsv4_description = json.loads(
        (
            ROOT
            / "examples/model_descriptions/dsv4_resident_decode_stage_firmware.json"
        ).read_text(encoding="utf-8")
    )
    dsv4_target = dsv4_description["stages"][0]["target"]
    expected_hosts = [f"spark{rank}-fabric" for rank in range(10)] + [
        "sparka-fabric",
        "sparkb-fabric",
        "sparkc-fabric",
    ]
    require(
        deployment_example["coordinator_rank_index"] == 0
        and [node["transport_host"] for node in deployment_example["nodes"]]
        == expected_hosts
        and all(
            not component["shared_object_path"].startswith("/")
            for component in (
                deployment_example["adapter"],
                deployment_example["driver"],
                deployment_example["transport"],
            )
        ),
        "DSV4 deployment does not use explicit hosts and node-local release roots",
    )
    require(
        all(node["node_target"] == dsv4_target
            for node in deployment_example["nodes"]),
        "DSV4 deployment target drifts from the generated AOT package",
    )

    fixture_adapter = (
        ROOT / "tests/fixtures/model_serving_adapter_module.c"
    ).read_text(encoding="utf-8")
    pipeline_process_test = (
        ROOT / "tests/test_model_pipeline_client.c"
    ).read_text(encoding="utf-8")
    transport_fixture = (
        ROOT / "tests/fixtures/model_resident_transport_module.c"
    ).read_text(encoding="utf-8")
    require(
        "SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_RELEASE" in fixture_adapter
        and "SPARK_MODEL_SERVING_WORK_KIND_RELEASE" in pipeline_process_test
        and "SPARK_STATUS_UNSUPPORTED" in pipeline_process_test,
        "generic process test misses release or partial rejection abort",
    )
    require(
        "SparkModelBatchEngineConnect" in pipeline_process_test
        and "max_prefill_rows_per_submission = max_prefill_rows"
        in pipeline_process_test
        and "TestModelBatchConnect(deployment,&state,0u,0u,1u)"
        in pipeline_process_test
        and "view.pipeline.submitted_count >= 7u" in pipeline_process_test,
        "generic process test misses chunking, decode width, or release",
    )
    require(
        "completion_delays" in transport_fixture
        and "TestModelResidentTransportFind(state,packet) != 0"
        in transport_fixture,
        "resident test hides duplicate posts with immediate completion",
    )

    print("model-neutral resident boundary and DSV4 adapter ownership hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
