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
    src/spark_orchestrator.c

# Transport, memory link, collectives, tokenizer, KV store, stage module ABI.
# Formerly model-families/common/src, which is the directory the handoff records
# as deleted twice with every gate green, because it was named for who shared it
# rather than what it is.
SPARKPIPE_MODEL_COMMON_SOURCES := \
    ring/transport/hidden_transport.c \
    ring/transport/memlink.c \
    ring/transport/tp_collective.c \
    api/text/tokenizer.c \
    cache/store/kv_store.c \
    cache/store/stage_kv_client.c \
    runtime/stage_module_common.c

SPARKPIPE_GLM52_SOURCES := \
    api/compat_api.c \
    api/http_gateway.c \
    api/request.c \
    api/service.c \
    api/serving_engine.c \
    cache/kv_cache.c \
    cache/prefix_cache.c \
    model-families/glm52/src/spark_glm52_batch_sequence_table.c \
    model-families/glm52/src/spark_glm52_chat_template.c \
    model-families/glm52/src/spark_glm52_cuda_resident_ipc.c \
    model-families/glm52/src/spark_glm52_expert_queue.c \
    model-families/glm52/src/spark_glm52_ring_node_context_builder.c \
    model-families/glm52/src/spark_glm52_production_topology.c \
    model-families/glm52/src/spark_glm52_prompt_pipeline.c \
    model-families/glm52/src/spark_glm52_row_allocator.c \
    model-families/glm52/src/spark_glm52_service_backend.c \
    model-families/glm52/src/spark_glm52_shape_config.c \
    model-families/glm52/src/spark_glm52_text_prompt.c \
    model-families/glm52/src/spark_glm52_tp_shard.c \
    node/rank_runtime.c \
    runtime/pack/stagepack.c \
    scheduler/long_context.c \
    scheduler/scheduler.c \
    scheduler/speculation.c \
    scheduler/stage_plan.c \
    scheduler/work_control.c

SPARKPIPE_QWEN36_SOURCES := \
    model-families/qwen36/src/spark_qwen36_work_control.c

SPARKPIPE_DEPLOYMENT_SOURCES := \
    deployment/src/spark_release.c
