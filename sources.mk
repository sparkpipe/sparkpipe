# Every host source, by the library it lands in, at the path it actually lives.
#
# This replaced four per-directory sources.mk files. They encoded the directory
# twice - once in the file's own location and once in the paths inside it - and
# the reorganisation moved twelve sources without updating either. The Makefile
# then mapped source to object with $(patsubst src/%.c,...), which returns a
# non-matching entry UNCHANGED rather than failing, so seven .c paths ended up
# in *_OBJECTS and one of them was fed to -include as a makefile:
#
#   runtime/filesystem.c:19: *** missing separator.  Stop.
#
# One list and one path-preserving object map means a source can move to any
# directory and only this file changes.

SPARKPIPE_CORE_SOURCES := \
    src/spark_status.c \
    src/spark_sha256.c \
    runtime/filesystem.c \
    runtime/json.c

SPARKPIPE_COMPILER_SOURCES := \
    runtime/pack/model_description.c \
    runtime/pack/module_library.c \
    runtime/pack/driver_compiler.c

SPARKPIPE_RUNTIME_SOURCES := \
    src/spark_driver_loader.c \
    src/spark_orchestrator.c \
    runtime/runtime_completion.c \
    runtime/model_runtime.c \
	runtime/model_serving_adapter.c \
	runtime/model_resident_endpoint.c \
	runtime/model_resident_deployment.c \
	runtime/model_continuation_lease.c \
	runtime/model_resident_ipc.c \
	runtime/model_resident_client.c \
	runtime/model_pipeline_client.c \
	runtime/model_batch_engine.c \
	runtime/pipeline_runtime.c

# Transport, memory link, collectives, tokenizer, KV store, stage module ABI.
# Formerly model-families/common/src, which is the directory the handoff records
# as deleted twice with every gate green, because it was named for who shared it
# rather than what it is.
SPARKPIPE_MODEL_COMMON_SOURCES := \
    ring/transport/hidden_transport.c \
    ring/transport/fabric_topology.c \
    ring/transport/memlink.c \
    ring/transport/tp_collective.c \
    ring/transport/tp_device_collective.c \
    ring/transport/tp_device_collective_nccl.c \
    text/tokenizer.c \
    cache/kv_cache.c \
	cache/kv_page_cache.c \
	cache/kv_page_store.c \
    cache/prefix_cache.c \
    cache/store/kv_store.c \
    cache/store/stage_kv_client.c \
    cache/nvme_tier.c \
    runtime/stage_module_common.c \
    runtime/work_transaction.c

SPARKPIPE_GLM52_SOURCES := \
	model-families/glm52/src/spark_glm52_chat_template.c \
	modules/glm52_dspark_draft_backend/source/spark_glm52_dspark_dispatch_policy.c

SPARKPIPE_QWEN36_SOURCES := \
    model-families/qwen36/src/spark_qwen36_work_control.c

SPARKPIPE_QWEN38_SOURCES := \
    model-families/qwen38/src/spark_qwen38_work_control.c

SPARKPIPE_DSV4_SOURCES := \
    model-families/dsv4/src/spark_dsv4_parallel_shape.c \
    model-families/dsv4/src/spark_dsv4_cache_plan.c \
    model-families/dsv4/src/spark_dsv4_cache_arena.c

SPARKPIPE_K3_SOURCES := \
    modules/k3_resident_decode_stage/source/spark_k3_pack_load.c \
    modules/k3_resident_decode_stage/source/spark_k3_bind.c \
    modules/k3_resident_decode_stage/source/spark_k3_resident_decode_stage_module.c

SPARKPIPE_DEPLOYMENT_SOURCES := \
    deployment/src/spark_release.c
