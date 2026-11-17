include $(dir $(lastword $(MAKEFILE_LIST)))runtime/weightd_sources.mk


SPARKPIPE_CORE_SOURCES := \
    src/spark_status.c \
    src/spark_sha256.c \
    src/spark_io.c \
    src/spark_quant_arm.c \
    src/spark_ck128.c \
    src/spark_admission.c \
    src/spark_speculation_lookup_draft.c \
    src/spark_speculation_frame.c \
    src/spark_speculation_tree_frame.c \
    src/spark_tap_stream.c \
    ring/transport/draft_bridge.c \
    runtime/filesystem.c \
    runtime/json.c

SPARKPIPE_COMPILER_SOURCES := \
    runtime/pack/model_description.c \
    runtime/pack/module_library.c \
    runtime/pack/driver_compiler.c

SPARKPIPE_RUNTIME_SOURCES := \
    src/spark_driver_loader.c \
    src/spark_orchestrator.c \
    runtime/stagepack_format.c \
    runtime/named_pack.c \
    runtime/spark_expert_working_set.c \
	runtime/model_serving_adapter.c \
	runtime/model_resident_endpoint.c \
	runtime/model_resident_deployment.c \
	runtime/chat_template.c \
	runtime/model_continuation_lease.c \
	runtime/model_resident_ipc.c \
	runtime/model_resident_client.c \
	runtime/model_pipeline_client.c \
	$(SPARKPIPE_WEIGHTD_SOURCES) \
	runtime/model_batch_engine.c \
	runtime/prefix_index_file.c \
	runtime/pipeline_runtime.c

SPARKPIPE_TRANSPORT_SOURCES := \
    ring/transport/hidden_transport.c \
    ring/transport/fabric_topology.c \
    ring/transport/tp_device_collective.c

SPARKPIPE_CACHE_SOURCES := \
    cache/kv_cache.c \
    cache/kv_page_cache.c \
    cache/kv_page_store.c \
    cache/kv_snapshot.c \
    cache/kv_model_table.c \
    cache/prefix_cache.c

SPARKPIPE_MODEL_COMMON_SOURCES := \
    $(SPARKPIPE_WEIGHTD_SOURCES) \
    $(SPARKPIPE_TRANSPORT_SOURCES) \
    $(SPARKPIPE_CACHE_SOURCES) \
    text/tokenizer.c \
    text/tokenizer_sidecar.c \
    runtime/stage_module_claims.c \
    runtime/stage_kv_binding.c \
    runtime/kv_device_copy.c

SPARKPIPE_GLM52_SOURCES := \
	model-families/glm52/src/spark_glm52_chat_template.c

SPARKPIPE_K3_SOURCES := \
    model-families/k3/src/spark_k3_pack_load.c \
    model-families/k3/src/spark_k3_bind.c \
    model-families/k3/src/spark_k3_resident_decode_stage_module.c

SPARKPIPE_DEPLOYMENT_SOURCES := \
    deployment/src/spark_release.c
