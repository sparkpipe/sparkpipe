CC ?= cc
CXX ?= c++
AR ?= ar
NVCC ?= nvcc
CUDA_HOME ?= /usr/local/cuda
CFLAGS ?= -std=c11 -Wall -Wextra -Werror -O3 -g -pthread
CXXFLAGS ?= -std=c++20 -Wall -Wextra -Werror -O3 -g -pthread
CPPFLAGS ?= -Iinclude -Isrc
LDFLAGS ?=
LDLIBS ?= -ldl -pthread
CUDA_ARCH ?= sm_121a
NVCCFLAGS ?= -O3 --use_fast_math -arch=$(CUDA_ARCH)
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# Architecture tuning for auto-vectorization. The tokenizer and other hot loops
# are written as portable scalar C that the compiler auto-vectorizes; these flags
# tell it which vector units the build host actually has, so it emits AVX2 on x86
# and NEON or SVE2 on an Arm Grace host, from one source with no intrinsics.
# ARCH_TUNE_FLAGS can be overridden or set empty for reproducible or cross builds.
# On Arm the Grace core is Neoverse V2; older compilers that do not know that name
# fall back to -mcpu=native. The flag is probed so an unsupported one is dropped
# rather than breaking the build.
spark-cc-supports = $(shell printf 'int main(void){return 0;}' | $(CC) $(1) -x c -c - -o /dev/null >/dev/null 2>&1 && printf '%s' '$(1)')
ifeq ($(UNAME_M),aarch64)
ARCH_TUNE_FLAGS ?= $(or $(call spark-cc-supports,-mcpu=neoverse-v2),$(call spark-cc-supports,-mcpu=native))
else ifeq ($(UNAME_M),arm64)
ARCH_TUNE_FLAGS ?= $(or $(call spark-cc-supports,-mcpu=native),$(call spark-cc-supports,-mcpu=apple-m1))
else ifeq ($(UNAME_M),x86_64)
ARCH_TUNE_FLAGS ?= $(call spark-cc-supports,-march=native)
else
ARCH_TUNE_FLAGS ?=
endif
CFLAGS += $(ARCH_TUNE_FLAGS)
CXXFLAGS += $(ARCH_TUNE_FLAGS)
ifeq ($(UNAME_S),Darwin)
SHARED_LIBRARY_FLAGS ?= -dynamiclib
SHARED_LIBRARY_EXT ?= dylib
else
SHARED_LIBRARY_FLAGS ?= -shared
SHARED_LIBRARY_EXT ?= so
endif
GLM52_PP13_NODE_CONTEXT_BUILDER_RPATH :=
ifneq ($(UNAME_S),Darwin)
GLM52_PP13_NODE_CONTEXT_BUILDER_RPATH := -Xlinker -rpath -Xlinker '$$ORIGIN/runtime_libs'
endif
SPARKPIPE_B12X_AOT_ENV ?= $(HOME)/.config/sparkpipe/glm52_b12x_aot_env.sh
B12X_AOT_TOKENS ?= 1,2,4,6,8,12,16,24,32,48,64,96,128,192,256,384,512,576,768,1024
B12X_AOT_WARMUP ?= 5
B12X_AOT_ITERATIONS ?= 20
B12X_AOT_OUTPUT_DIR ?= build/glm52_b12x_aot_prompt
GLM52_MOE_BACKEND ?= fp8
MOONCAKE_ROOT ?=
MOONCAKE_LIB ?= $(MOONCAKE_ROOT)/build/mooncake-store/src
MOONCAKE_DEP_INCLUDE ?= $(MOONCAKE_ROOT)/local/include
GLM52_EXACT_PP13_MODEL_QUANTIZATION ?= $(GLM52_MOE_BACKEND)
GLM52_MODEL_DIR ?= $(HOME)/models/hf/zai-org/GLM-5.2-FP8
GLM52_STAGE_PACK_DIR ?= $(HOME)/models/sparkpipe/glm52_fp8_pp13_stage_payload_v1
GLM52_FP8_MOE_PACK_DIR ?= $(GLM52_STAGE_PACK_DIR)
B12X_AOT_BENCHMARK ?= --benchmark
B12X_MOE_PACK_OUTPUT_DIR ?= build/glm52_b12x_resident_moe
B12X_MOE_PACK_LAYERS ?= 3,4,5,6,7,8,9,10
B12X_MOE_PACK_REQUIRE_REUSE ?= 1
B12X_MOE_PACK_VERIFY_REUSED_SHA256 ?= 0
B12X_MOE_PACK_JOBS ?= 1
B12X_MOE_PACK_PACKAGE_MODE ?= hardlink
GLM52_VALIDATION_MODE ?= dense_to_layer3_routed
GLM52_VALIDATION_ACTIVE_SEQUENCE_COUNT ?= 1
GLM52_VALIDATION_FIRST_ROUTED_LAYER_INDEX ?= 3
GLM52_VALIDATION_ROUTED_CHAIN_LAYER_COUNT ?= 1
GLM52_PIPELINE_INPUT_HIDDEN_BF16 ?=
GLM52_PIPELINE_OUTPUT_HIDDEN_BF16 ?= build/glm52_pipeline_validation/output_hidden.bf16
GLM52_ENABLE_CUDA_GRAPH_REPLAY ?= 0
GLM52_STAGE_SWEEP_BUCKETS ?= 8,16,32,64,128
GLM52_STAGE_SWEEP_STAGE_ARGS ?=
GLM52_STAGE_SWEEP_MAX_STAGE_US ?= 1000000
GLM52_STAGE_SWEEP_OUTPUT_DIR ?= build/glm52_stage_bucket_sweep
GLM52_STAGE_SWEEP_WARMUP_RUNS ?= 0
GLM52_STAGE_SWEEP_MEASURE_RUNS ?= 1
GLM52_STAGE_SWEEP_PACKAGE_EACH_RUN ?= 0
GLM52_STAGE_SWEEP_MODULE_ARCHIVE ?= build/modules/glm52_resident_decode_stage/libglm52_resident_decode_stage.a
GLM52_STAGE_SWEEP_DRIVER_SO ?=
GLM52_STAGE_SWEEP_VALIDATOR_CACHE_DIR ?= $(GLM52_STAGE_SWEEP_OUTPUT_DIR)/validators
GLM52_STAGE_SWEEP_REQUIRED_CUDA_LINK_ARGS ?=
GLM52_STAGE_SWEEP_FORCE_VALIDATOR_REBUILD ?= 0
GLM52_REQUIRED_CUDA_LINK_ARGS ?=
REQUIRED_CUDA_CC_ARGS ?=
B12X_ADAPTER_ARCHIVE := $(abspath build/modules/glm52_sm121_flashinfer_b12x_moe/libglm52_sm121_flashinfer_b12x_moe_adapter.a)
B12X_COMPILED_BACKEND_ARCHIVE := $(abspath build/modules/glm52_sm121_b12x_compiled_backend/libglm52_sm121_b12x_compiled_backend.a)
B12X_GENERATED_KERNEL_TABLE_ARCHIVE := $(abspath build/modules/glm52_sm121_b12x_compiled_backend/libglm52_sm121_b12x_generated_kernel_table.a)
DSPARK_DRAFT_BACKEND_ARCHIVE := $(abspath build/modules/glm52_dspark_draft_backend/libglm52_dspark_draft_backend.a)
B12X_RUNTIME_LINK_ARGS_FILE := $(abspath $(B12X_AOT_OUTPUT_DIR))/generated/runtime_link_args.txt
ifeq ($(GLM52_MOE_BACKEND),nvfp4)
GLM52_PP13_NODE_CONTEXT_BUILDER_DEFAULT_LINK_ARGS := $(B12X_ADAPTER_ARCHIVE) $(B12X_COMPILED_BACKEND_ARCHIVE) $(B12X_GENERATED_KERNEL_TABLE_ARCHIVE) $(shell cat "$(B12X_RUNTIME_LINK_ARGS_FILE)" 2>/dev/null)
else
GLM52_PP13_NODE_CONTEXT_BUILDER_DEFAULT_LINK_ARGS := $(B12X_ADAPTER_ARCHIVE) $(B12X_COMPILED_BACKEND_ARCHIVE) $(B12X_GENERATED_KERNEL_TABLE_ARCHIVE)
endif
GLM52_PP13_NODE_CONTEXT_BUILDER_LINK_ARGS ?= $(if $(GLM52_REQUIRED_CUDA_LINK_ARGS),$(GLM52_REQUIRED_CUDA_LINK_ARGS),$(GLM52_PP13_NODE_CONTEXT_BUILDER_DEFAULT_LINK_ARGS))
GLM52_PP13_NODE_CONTEXT_BUILDER := build/libglm52_pp13_node_context_builder.$(SHARED_LIBRARY_EXT)
GLM52_FP8_SCALED_GEMM_CUDA_GATE := build/glm52_fp8_scaled_gemm_cuda_gate
HIDDEN_TRANSPORT_TCP_CUDA := build/libhidden_transport_tcp_cuda.$(SHARED_LIBRARY_EXT)
HIDDEN_TRANSPORT_SPARK_HOST_RDMA := build/libhidden_transport_spark_host_rdma_verbs.$(SHARED_LIBRARY_EXT)

COMMON_SOURCES := \
    src/spark_status.c \
    src/spark_filesystem.c \
    src/spark_json.c \
    src/spark_sha256.c \
    src/spark_release.c \
    src/spark_hidden_transport.c \
    src/spark_memlink.c \
    src/spark_glm52_kv_cache.c \
    src/spark_kv_store.c \
    src/spark_glm52_dspark.c \
    src/spark_glm52_stage_plan.c \
    src/spark_glm52_stagepack.c \
    src/spark_glm52_production_topology.c \
    src/spark_glm52_pp13_runtime.c \
    src/spark_glm52_pp13_work_control.c \
    src/spark_glm52_cuda_resident_ipc.c \
    src/spark_glm52_pp13_node_context_builder.c \
    src/spark_glm52_scheduler.c \
    src/spark_glm52_prefix_cache.c \
    src/spark_glm52_request_api.c \
    src/spark_glm52_long_context.c \
    src/spark_tokenizer.c \
    src/spark_glm52_chat_template.c \
    src/spark_glm52_text_prompt.c \
    src/spark_glm52_prompt_pipeline.c \
    src/spark_glm52_serving_engine.c \
    src/spark_glm52_service.c \
    src/spark_glm52_service_backend.c \
    src/spark_glm52_compat_api.c \
    src/spark_glm52_http_gateway.c \
    src/spark_stage_kv_client.c \
    src/spark_qwen36_work_control.c

COMPILER_SOURCES := \
    src/spark_model_description.c \
    src/spark_module_library.c \
    src/spark_driver_compiler.c

RUNTIME_SOURCES := \
    src/spark_driver_loader.c \
    src/spark_orchestrator.c

COMMON_OBJECTS := $(patsubst src/%.c,build/%.o,$(COMMON_SOURCES))
COMPILER_OBJECTS := $(patsubst src/%.c,build/%.o,$(COMPILER_SOURCES))
RUNTIME_OBJECTS := $(patsubst src/%.c,build/%.o,$(RUNTIME_SOURCES))
COMMON_LIBRARY := build/libsparkpipe_common.a
COMPILER_LIBRARY := build/libsparkpipe_compiler.a
RUNTIME_LIBRARY := build/libsparkpipe_runtime.a
LIBRARIES := $(COMMON_LIBRARY) $(COMPILER_LIBRARY) $(RUNTIME_LIBRARY)
GLM52_PP13_SERVICE_BACKEND := build/libglm52_pp13_service_backend.$(SHARED_LIBRARY_EXT)

TOOL_NAMES := \
    sparkpipe_module_publish \
    sparkpipe_model_compile \
    sparkpipe_driver_inspect \
    sparkpipe_glm52_prefill_dryrun \
    sparkpipe_hidden_transport_preflight \
    sparkpipe_glm52_pp13_rank_gate \
    sparkpipe_glm52_pp13_loopback_probe \
    sparkpipe_glm52_pipesim \
    sparkpipe_glm52_pp13_rank_daemon \
    sparkpipe_glm52_cuda_residentd \
    sparkpipe_glm52_cuda_resident_gate \
    sparkpipe_glm52_kv_jit_budget \
    sparkpipe_glm52_tokenize \
    sparkpipe_tokenize_prompt \
    sparkpipe_tokenizer_benchmark \
    sparkpipe_glm52_http_gateway \
    sparkpipe_memlink \
    sparkpipe_prevcp \
    sparkpipe_nextcp \
    sparkpipe_release_manager

TOOL_BINARIES := $(addprefix build/,$(TOOL_NAMES))

TEST_NAMES := \
    test_json \
    test_hidden_transport \
    test_memlink \
    test_release \
    test_glm52_kv_cache \
    test_kv_store \
    test_kv_mooncake \
    test_qwen36_work_control \
    test_glm52_dspark \
    test_glm52_stage_plan \
    test_glm52_mtp_tree \
    test_glm52_stagepack \
    test_glm52_production_topology \
    test_glm52_pp13_runtime \
    test_glm52_cuda_resident_ipc \
    test_glm52_cuda_resident_gate \
    test_glm52_pp13_work_control \
    test_glm52_scheduler \
    test_glm52_prefix_cache \
    test_glm52_request_api \
    test_glm52_long_context \
    test_tokenizer \
    test_glm52_prompt_pipeline \
    test_glm52_serving_engine \
    test_glm52_service \
    test_glm52_service_backend \
    test_glm52_compat_api \
    test_glm52_http_gateway \
    test_glm52_pp13_rank_daemon \
    test_model_description \
    test_module_library \
    test_driver_compiler \
    test_orchestrator \
    test_glm52_resident_decode_stage_firmware \
    test_glm52_resident_decode_stage_production_runner

TEST_BINARIES := $(addprefix build/,$(TEST_NAMES))
PYTHON_TESTS := \
	tests/test_api_stress.py \
	tests/test_memory_contracts.py \
	tests/test_b12x_scale_layout.py \
	tests/test_glm52_dspark_manifest.py \
	tests/test_glm52_dspark_artifact_preflight.py \
	tests/test_glm52_dspark_trace_quality.py \
	tests/test_glm52_b12x_pack_worker.py \
	tests/test_glm52_b12x_resident_manifest.py \
	tests/test_glm52_fp8_pack_layout.py \
	tests/test_glm52_nvfp4_artifact_preflight.py \
	tests/test_glm52_quantized_cuda_contract.py \
	tests/test_glm52_b12x_relocate_aot_bundle.py \
	tests/test_glm52_b12x_deterministic_finalize.py \
	tests/test_glm52_final_from_hidden_mode.py \
	tests/test_glm52_exact_pp13_prefill_hidden.py \
	tests/test_glm52_firmware_package.py \
	tests/test_measured_status.py \
	tests/test_release_assemble.py \
	tests/test_glm52_stage_pack.py \
	tests/test_glm52_stage_bucket_sweep.py \
	tests/test_glm52_prompt_pipeline_input.py
TEST_SUPPORT_OBJECT := build/test_support.o
TEST_MODULE_OBJECTS := \
    build/test_modules/module_add_one.o \
    build/test_modules/module_add_two.o \
    build/test_modules/module_double.o \
    build/test_modules/module_affine_entry.o \
    build/test_modules/module_affine_helper.o
TEST_MODULE_ARCHIVES := \
    build/test_modules/module_affine.a
TEST_MODULE_LINK_UNITS := $(TEST_MODULE_OBJECTS) $(TEST_MODULE_ARCHIVES)
TEST_MODULE_DEPENDENCIES := $(TEST_MODULE_OBJECTS:.o=.d)
TEST_HIDDEN_TRANSPORT_MODULE := \
    build/test_modules/libhidden_transport_module.$(SHARED_LIBRARY_EXT)
TEST_SERVICE_BACKEND_MODULE := \
    build/test_modules/libglm52_service_backend_module.$(SHARED_LIBRARY_EXT)
TEST_VALIDATOR := build/test_module_validator
GLM52_RESIDENT_DECODE_STAGE_TEST_DIRECTORY := \
    build/glm52_resident_decode_stage_test
GLM52_RESIDENT_DECODE_STAGE_TEST_OBJECTS := \
    $(GLM52_RESIDENT_DECODE_STAGE_TEST_DIRECTORY)/spark_glm52_resident_decode_stage_module.o \
    $(GLM52_RESIDENT_DECODE_STAGE_TEST_DIRECTORY)/glm52_resident_decode_stage_fake_backend.o
GLM52_RESIDENT_DECODE_STAGE_TEST_DEPENDENCIES := \
    $(GLM52_RESIDENT_DECODE_STAGE_TEST_OBJECTS:.o=.d)
GLM52_RESIDENT_DECODE_STAGE_TEST_ARCHIVE := \
    $(GLM52_RESIDENT_DECODE_STAGE_TEST_DIRECTORY)/libglm52_resident_decode_stage_test.a

.PHONY: all clean test tools demo FORCE \
    cuda_glm52_resident_decode_stage \
    cuda_glm52_resident_decode_stage_publish \
    glm52_flashinfer_b12x_moe_adapter \
    glm52_b12x_prepare_spark_env \
    glm52_b12x_aot_compile \
    glm52_b12x_resident_pack \
    glm52_b12x_compiled_backend \
    glm52_quantized_readiness_test \
    glm52_required_cuda_link_args \
    glm52_stage_bucket_sweep \
    glm52_spark2_accuracy_gate \
    glm52_spark2_local_pipeline_gate \
    glm52_pp13_service_backend \
    hidden_transport_spark_host_rdma_verbs \
    glm52_pp13_node_context_builder \
    kv_mooncake \
    glm52_resident_decode_stage_firmware_package \
    tree_summary

all: $(LIBRARIES) tools $(GLM52_PP13_SERVICE_BACKEND)

tools: $(TOOL_BINARIES) $(GLM52_PP13_SERVICE_BACKEND)

build:
	mkdir -p build

build/test_modules:
	mkdir -p build/test_modules

$(GLM52_RESIDENT_DECODE_STAGE_TEST_DIRECTORY):
	mkdir -p $(GLM52_RESIDENT_DECODE_STAGE_TEST_DIRECTORY)

build/%.o: src/%.c | build
	$(CC) $(CPPFLAGS) $(CFLAGS) -fPIC -MMD -MP -c $< -o $@

$(COMMON_LIBRARY): $(COMMON_OBJECTS)
	$(AR) rcs $@ $^

$(COMPILER_LIBRARY): $(COMPILER_OBJECTS)
	$(AR) rcs $@ $^

$(RUNTIME_LIBRARY): $(RUNTIME_OBJECTS)
	$(AR) rcs $@ $^

build/sparkpipe_module_publish: tools/sparkpipe_module_publish.c $(COMPILER_LIBRARY) $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(COMPILER_LIBRARY) $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/sparkpipe_model_compile: tools/sparkpipe_model_compile.c $(COMPILER_LIBRARY) $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(COMPILER_LIBRARY) $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/sparkpipe_driver_inspect: tools/sparkpipe_driver_inspect.c $(RUNTIME_LIBRARY) $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(RUNTIME_LIBRARY) $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/sparkpipe_glm52_prefill_dryrun: tools/sparkpipe_glm52_prefill_dryrun.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/sparkpipe_hidden_transport_preflight: tools/sparkpipe_hidden_transport_preflight.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/sparkpipe_glm52_pp13_rank_gate: tools/sparkpipe_glm52_pp13_rank_gate.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/sparkpipe_glm52_pp13_loopback_probe: tools/sparkpipe_glm52_pp13_loopback_probe.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/sparkpipe_glm52_pipesim: tools/sparkpipe_glm52_pipesim.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_glm52_batch_plane: tests/test_glm52_batch_plane.c src/spark_glm52_expert_queue.c src/spark_glm52_jit_kv_pool.c src/spark_glm52_batch_sequence_table.c src/spark_glm52_kv_dedup.c
	$(CC) $(CFLAGS) -Iinclude -o $@ tests/test_glm52_batch_plane.c src/spark_glm52_expert_queue.c src/spark_glm52_jit_kv_pool.c src/spark_glm52_batch_sequence_table.c src/spark_glm52_kv_dedup.c

build/sparkpipe_glm52_batchplane_sim: tools/sparkpipe_glm52_batchplane_sim.c src/spark_glm52_expert_queue.c src/spark_glm52_jit_kv_pool.c src/spark_glm52_kv_dedup.c
	$(CC) $(CFLAGS) -Iinclude -o $@ tools/sparkpipe_glm52_batchplane_sim.c src/spark_glm52_expert_queue.c src/spark_glm52_jit_kv_pool.c src/spark_glm52_kv_dedup.c

build/sparkpipe_glm52_batchplane_model: tools/sparkpipe_glm52_batchplane_model.c 
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

build/sparkpipe_glm52_tokenize: tools/sparkpipe_glm52_tokenize.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/sparkpipe_tokenize_prompt: tools/sparkpipe_tokenize_prompt.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/sparkpipe_tokenizer_benchmark: tools/sparkpipe_tokenizer_benchmark.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/sparkpipe_glm52_http_gateway: tools/sparkpipe_glm52_http_gateway.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/sparkpipe_memlink: tools/sparkpipe_memlink.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/sparkpipe_prevcp: tools/sparkpipe_memlink.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DSPARK_MEMLINK_FIXED_COMMAND=\"prevcp\" $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/sparkpipe_nextcp: tools/sparkpipe_memlink.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DSPARK_MEMLINK_FIXED_COMMAND=\"nextcp\" $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/sparkpipe_release_manager: tools/sparkpipe_release_manager.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/sparkpipe_glm52_pp13_ring_check: tools/sparkpipe_glm52_pp13_ring_check.c $(RUNTIME_LIBRARY) $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -I$(CUDA_HOME)/include $(CFLAGS) tools/sparkpipe_glm52_pp13_ring_check.c $(RUNTIME_LIBRARY) $(COMMON_LIBRARY) $(LDFLAGS) -L$(CUDA_HOME)/lib64 $(LDLIBS) -lcudart -o $@

build/sparkpipe_glm52_pp13_rank_daemon: tools/sparkpipe_glm52_pp13_rank_daemon.c modules/glm52_resident_decode_stage/source/spark_glm52_resident_decode_stage_production_runner.c modules/glm52_resident_decode_stage/include/sparkpipe/spark_glm52_resident_decode_stage_production_runner.h $(RUNTIME_LIBRARY) $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Imodules/glm52_resident_decode_stage/include $(CFLAGS) tools/sparkpipe_glm52_pp13_rank_daemon.c modules/glm52_resident_decode_stage/source/spark_glm52_resident_decode_stage_production_runner.c $(RUNTIME_LIBRARY) $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/sparkpipe_glm52_cuda_residentd: tools/sparkpipe_glm52_cuda_residentd.c $(RUNTIME_LIBRARY) $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Imodules/glm52_resident_decode_stage/include $(CFLAGS) tools/sparkpipe_glm52_cuda_residentd.c $(RUNTIME_LIBRARY) $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/sparkpipe_glm52_cuda_resident_gate: tools/sparkpipe_glm52_cuda_resident_gate.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) tools/sparkpipe_glm52_cuda_resident_gate.c $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/sparkpipe_glm52_kv_jit_budget: tools/sparkpipe_glm52_kv_jit_budget.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) tools/sparkpipe_glm52_kv_jit_budget.c $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

$(GLM52_PP13_SERVICE_BACKEND): src/spark_glm52_pp13_service_backend.c modules/glm52_resident_decode_stage/source/spark_glm52_resident_decode_stage_production_runner.c modules/glm52_resident_decode_stage/include/sparkpipe/spark_glm52_resident_decode_stage_production_runner.h $(RUNTIME_LIBRARY) $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Imodules/glm52_resident_decode_stage/include $(CFLAGS) -fPIC $(SHARED_LIBRARY_FLAGS) src/spark_glm52_pp13_service_backend.c modules/glm52_resident_decode_stage/source/spark_glm52_resident_decode_stage_production_runner.c $(RUNTIME_LIBRARY) $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

glm52_pp13_service_backend: $(GLM52_PP13_SERVICE_BACKEND)

$(HIDDEN_TRANSPORT_TCP_CUDA): modules/hidden_transport_tcp_cuda.cu $(COMMON_LIBRARY)
	@if ! command -v $(NVCC) >/dev/null 2>&1; then \
		echo "hidden_transport_tcp_cuda skipped: nvcc unavailable"; \
	else \
		$(NVCC) $(NVCCFLAGS) $(SHARED_LIBRARY_FLAGS) -Xcompiler -fPIC -Xcompiler -pthread -Iinclude -Isrc modules/hidden_transport_tcp_cuda.cu $(COMMON_LIBRARY) $(LDFLAGS) -L$(CUDA_HOME)/lib64 -lcudart -ldl -lpthread -o $@; \
	fi

hidden_transport_tcp_cuda: $(HIDDEN_TRANSPORT_TCP_CUDA)

$(HIDDEN_TRANSPORT_SPARK_HOST_RDMA): modules/hidden_transport_spark_host_rdma_verbs.cu include/sparkpipe/spark_hidden_transport.h include/sparkpipe/spark_memlink.h $(COMMON_LIBRARY)
	@if ! command -v $(NVCC) >/dev/null 2>&1; then \
		echo "hidden_transport_spark_host_rdma_verbs skipped: nvcc unavailable"; \
	elif [ ! -f "$(CUDA_HOME)/include/cuda_runtime_api.h" ]; then \
		echo "hidden_transport_spark_host_rdma_verbs skipped: CUDA headers unavailable"; \
	elif [ ! -f "/usr/include/infiniband/verbs.h" ]; then \
		echo "hidden_transport_spark_host_rdma_verbs skipped: libibverbs headers unavailable"; \
	else \
		$(NVCC) $(NVCCFLAGS) $(SHARED_LIBRARY_FLAGS) -Xcompiler -fPIC -Xcompiler -pthread -Iinclude -Isrc modules/hidden_transport_spark_host_rdma_verbs.cu $(COMMON_LIBRARY) $(LDFLAGS) -L$(CUDA_HOME)/lib64 -lcudart -libverbs -ldl -lpthread -o $@; \
	fi

hidden_transport_spark_host_rdma_verbs: $(HIDDEN_TRANSPORT_SPARK_HOST_RDMA)

FORCE:

$(GLM52_STAGE_SWEEP_MODULE_ARCHIVE): FORCE
	@if ! command -v $(NVCC) >/dev/null 2>&1; then \
		echo "glm52_resident_decode_stage archive skipped: nvcc unavailable"; \
	else \
		$(MAKE) -C modules/glm52_resident_decode_stage archive NVCC=$(NVCC) CUDA_ARCH=sm_121a; \
	fi

$(DSPARK_DRAFT_BACKEND_ARCHIVE): FORCE
	@if ! command -v $(NVCC) >/dev/null 2>&1; then \
		echo "glm52_dspark_draft_backend archive skipped: nvcc unavailable"; \
	else \
		$(MAKE) -C modules/glm52_dspark_draft_backend archive NVCC=$(NVCC) CUDA_ARCH=sm_121a; \
	fi

$(GLM52_PP13_NODE_CONTEXT_BUILDER): modules/glm52_resident_decode_stage/source/spark_glm52_pp13_node_context_builder_cuda.cu modules/glm52_resident_decode_stage/source/spark_glm52_resident_decode_stage_production_runner.c modules/glm52_resident_decode_stage/include/sparkpipe/spark_glm52_resident_decode_stage_production_runner.h modules/glm52_resident_decode_stage/include/sparkpipe/spark_glm52_resident_decode_stage_firmware.h include/sparkpipe/spark_glm52_kv_cache.h include/sparkpipe/spark_glm52_pp13_work_control.h $(GLM52_STAGE_SWEEP_MODULE_ARCHIVE) $(DSPARK_DRAFT_BACKEND_ARCHIVE) $(B12X_ADAPTER_ARCHIVE) $(B12X_COMPILED_BACKEND_ARCHIVE) $(B12X_GENERATED_KERNEL_TABLE_ARCHIVE) $(COMMON_LIBRARY) $(RUNTIME_LIBRARY)
	@if ! command -v $(NVCC) >/dev/null 2>&1; then \
		echo "glm52_pp13_node_context_builder skipped: nvcc unavailable"; \
	else \
		if [ "$(GLM52_MOE_BACKEND)" = "nvfp4" ] && [ -z "$(GLM52_REQUIRED_CUDA_LINK_ARGS)" ] && [ ! -s "$(B12X_RUNTIME_LINK_ARGS_FILE)" ]; then \
			echo "missing $(B12X_RUNTIME_LINK_ARGS_FILE); run make glm52_b12x_aot_compile first (GLM52_MOE_BACKEND=nvfp4)" >&2; \
			exit 2; \
		fi; \
		$(NVCC) $(NVCCFLAGS) $(SHARED_LIBRARY_FLAGS) -Xcompiler -fPIC -Xcompiler -pthread -Iinclude -Isrc -Imodules/glm52_resident_decode_stage/include -Imodules/glm52_resident_decode_stage/source -Imodules/glm52_dspark_draft_backend/include modules/glm52_resident_decode_stage/source/spark_glm52_pp13_node_context_builder_cuda.cu modules/glm52_resident_decode_stage/source/spark_glm52_resident_decode_stage_production_runner.c $(GLM52_STAGE_SWEEP_MODULE_ARCHIVE) $(DSPARK_DRAFT_BACKEND_ARCHIVE) $(COMMON_LIBRARY) $(RUNTIME_LIBRARY) $(LDFLAGS) -L$(CUDA_HOME)/lib64 -lcudart -lcublasLt -lcublas -lm -ldl $(GLM52_PP13_NODE_CONTEXT_BUILDER_RPATH) $(GLM52_PP13_NODE_CONTEXT_BUILDER_LINK_ARGS) -o $@; \
	fi

glm52_pp13_node_context_builder: $(GLM52_PP13_NODE_CONTEXT_BUILDER)

kv_mooncake:
	$(MAKE) -C modules/kv_mooncake MOONCAKE_ROOT="$(MOONCAKE_ROOT)" MOONCAKE_LIB="$(MOONCAKE_LIB)" MOONCAKE_DEP_INCLUDE="$(MOONCAKE_DEP_INCLUDE)"

$(TEST_SUPPORT_OBJECT): tests/test_support.c tests/test_support.h $(COMPILER_LIBRARY) $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) -MMD -MP -c tests/test_support.c -o $@

build/test_modules/module_add_one.o: tests/fixtures/module_add_one.c | build/test_modules
	$(CC) -Iinclude $(CFLAGS) -fPIC -MMD -MP -c $< -o $@

build/test_modules/module_add_two.o: tests/fixtures/module_add_two.c | build/test_modules
	$(CC) -Iinclude $(CFLAGS) -fPIC -MMD -MP -c $< -o $@

build/test_modules/module_double.o: tests/fixtures/module_double.c | build/test_modules
	$(CC) -Iinclude $(CFLAGS) -fPIC -MMD -MP -c $< -o $@

build/test_modules/module_affine_entry.o: tests/fixtures/module_affine_entry.c | build/test_modules
	$(CC) -Iinclude $(CFLAGS) -fPIC -MMD -MP -c $< -o $@

build/test_modules/module_affine_helper.o: tests/fixtures/module_affine_helper.c | build/test_modules
	$(CC) -Iinclude $(CFLAGS) -fPIC -fvisibility=hidden -MMD -MP -c $< -o $@

build/test_modules/module_affine.a: build/test_modules/module_affine_entry.o build/test_modules/module_affine_helper.o
	$(AR) rcs $@ $^

$(TEST_HIDDEN_TRANSPORT_MODULE): tests/fixtures/hidden_transport_module.c | build/test_modules
	$(CC) $(CPPFLAGS) $(CFLAGS) -fPIC $(SHARED_LIBRARY_FLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

$(TEST_SERVICE_BACKEND_MODULE): tests/fixtures/glm52_service_backend_module.c include/sparkpipe/spark_glm52_service_backend.h | build/test_modules
	$(CC) $(CPPFLAGS) $(CFLAGS) -fPIC $(SHARED_LIBRARY_FLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

$(TEST_VALIDATOR): tests/fixtures/module_validator.c | build
	$(CC) $(CFLAGS) $< -o $@


$(GLM52_RESIDENT_DECODE_STAGE_TEST_DIRECTORY)/spark_glm52_resident_decode_stage_module.o: modules/glm52_resident_decode_stage/source/spark_glm52_resident_decode_stage_module.c modules/glm52_resident_decode_stage/include/sparkpipe/spark_glm52_resident_decode_stage_firmware.h modules/glm52_resident_decode_stage/source/spark_glm52_resident_decode_stage_backend.h | $(GLM52_RESIDENT_DECODE_STAGE_TEST_DIRECTORY)
	$(CC) $(CPPFLAGS) -Imodules/glm52_resident_decode_stage/include -Imodules/glm52_resident_decode_stage/source $(CFLAGS) -fPIC -fvisibility=hidden -MMD -MP -c $< -o $@

$(GLM52_RESIDENT_DECODE_STAGE_TEST_DIRECTORY)/glm52_resident_decode_stage_fake_backend.o: tests/fixtures/glm52_resident_decode_stage_fake_backend.c tests/fixtures/glm52_resident_decode_stage_fake_backend.h modules/glm52_resident_decode_stage/include/sparkpipe/spark_glm52_resident_decode_stage_firmware.h modules/glm52_resident_decode_stage/source/spark_glm52_resident_decode_stage_backend.h | $(GLM52_RESIDENT_DECODE_STAGE_TEST_DIRECTORY)
	$(CC) $(CPPFLAGS) -Itests/fixtures -Imodules/glm52_resident_decode_stage/include -Imodules/glm52_resident_decode_stage/source $(CFLAGS) -fPIC -fvisibility=hidden -MMD -MP -c $< -o $@

$(GLM52_RESIDENT_DECODE_STAGE_TEST_ARCHIVE): $(GLM52_RESIDENT_DECODE_STAGE_TEST_OBJECTS)
	rm -f $@
	$(AR) rcs $@ $^

build/test_json: tests/test_json.c $(COMPILER_LIBRARY) $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(COMPILER_LIBRARY) $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_hidden_transport: tests/test_hidden_transport.c $(COMMON_LIBRARY) $(TEST_HIDDEN_TRANSPORT_MODULE)
	$(CC) $(CPPFLAGS) -Itests -DSPARK_TEST_HIDDEN_TRANSPORT_MODULE_PATH=\"$(TEST_HIDDEN_TRANSPORT_MODULE)\" $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_memlink: tests/test_memlink.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_release: tests/test_release.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_glm52_kv_cache: tests/test_glm52_kv_cache.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_kv_store: tests/test_kv_store.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_kv_mooncake: tests/test_kv_mooncake.cpp tests/fixtures/mooncake/dummy_client.cpp modules/kv_mooncake/spark_kv_mooncake.cpp $(COMMON_LIBRARY)
	$(CXX) $(CPPFLAGS) -Itests/fixtures/mooncake $(CXXFLAGS) $^ $(LDFLAGS) $(LDLIBS) -o $@

build/test_qwen36_work_control: tests/test_qwen36_work_control.cpp tests/fixtures/mooncake/dummy_client.cpp modules/kv_mooncake/spark_kv_mooncake.cpp $(COMMON_LIBRARY)
	$(CXX) $(CPPFLAGS) -Itests/fixtures/mooncake $(CXXFLAGS) $^ $(LDFLAGS) $(LDLIBS) -o $@

build/test_glm52_stage_plan: tests/test_glm52_stage_plan.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_glm52_mtp_tree: tests/test_glm52_mtp_tree.c include/sparkpipe/spark_glm52_mtp_tree.h $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_glm52_stagepack: tests/test_glm52_stagepack.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_glm52_dspark: tests/test_glm52_dspark.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_glm52_production_topology: tests/test_glm52_production_topology.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_glm52_pp13_runtime: tests/test_glm52_pp13_runtime.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_glm52_pp13_work_control: tests/test_glm52_pp13_work_control.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_glm52_scheduler: tests/test_glm52_scheduler.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_glm52_prefix_cache: tests/test_glm52_prefix_cache.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_glm52_request_api: tests/test_glm52_request_api.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_glm52_cuda_resident_ipc: tests/test_glm52_cuda_resident_ipc.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_glm52_cuda_resident_gate: tests/test_glm52_cuda_resident_gate.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@


build/test_glm52_long_context: tests/test_glm52_long_context.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@


build/test_tokenizer: tests/test_tokenizer.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_glm52_prompt_pipeline: tests/test_glm52_prompt_pipeline.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@


build/test_glm52_serving_engine: tests/test_glm52_serving_engine.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_glm52_service: tests/test_glm52_service.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_glm52_service_backend: tests/test_glm52_service_backend.c $(COMMON_LIBRARY) $(TEST_SERVICE_BACKEND_MODULE)
	$(CC) $(CPPFLAGS) -Itests -DTEST_SERVICE_BACKEND_MODULE_PATH=\"$(TEST_SERVICE_BACKEND_MODULE)\" $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_glm52_compat_api: tests/test_glm52_compat_api.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_glm52_http_gateway: tests/test_glm52_http_gateway.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_glm52_pp13_rank_daemon: tests/test_glm52_pp13_rank_daemon.c modules/glm52_resident_decode_stage/source/spark_glm52_resident_decode_stage_production_runner.c $(RUNTIME_LIBRARY) $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests -Imodules/glm52_resident_decode_stage/include $(CFLAGS) tests/test_glm52_pp13_rank_daemon.c modules/glm52_resident_decode_stage/source/spark_glm52_resident_decode_stage_production_runner.c $(RUNTIME_LIBRARY) $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@


build/test_model_description: tests/test_model_description.c $(COMPILER_LIBRARY) $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(COMPILER_LIBRARY) $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_module_library: tests/test_module_library.c $(TEST_SUPPORT_OBJECT) $(TEST_MODULE_LINK_UNITS) $(TEST_VALIDATOR) $(COMPILER_LIBRARY) $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(TEST_SUPPORT_OBJECT) $(COMPILER_LIBRARY) $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_driver_compiler: tests/test_driver_compiler.c $(TEST_SUPPORT_OBJECT) $(TEST_MODULE_LINK_UNITS) $(TEST_VALIDATOR) $(COMPILER_LIBRARY) $(RUNTIME_LIBRARY) $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(TEST_SUPPORT_OBJECT) $(COMPILER_LIBRARY) $(RUNTIME_LIBRARY) $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_orchestrator: tests/test_orchestrator.c $(TEST_SUPPORT_OBJECT) $(TEST_MODULE_LINK_UNITS) $(TEST_VALIDATOR) $(COMPILER_LIBRARY) $(RUNTIME_LIBRARY) $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(TEST_SUPPORT_OBJECT) $(COMPILER_LIBRARY) $(RUNTIME_LIBRARY) $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_glm52_resident_decode_stage_firmware: tests/test_glm52_resident_decode_stage_firmware.c modules/glm52_resident_decode_stage/include/sparkpipe/spark_glm52_resident_decode_stage_firmware.h tests/fixtures/glm52_resident_decode_stage_fake_backend.h $(GLM52_RESIDENT_DECODE_STAGE_TEST_ARCHIVE) $(TEST_SUPPORT_OBJECT) $(TEST_VALIDATOR) $(COMPILER_LIBRARY) $(RUNTIME_LIBRARY) $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests -Itests/fixtures -Imodules/glm52_resident_decode_stage/include $(CFLAGS) $< $(TEST_SUPPORT_OBJECT) $(GLM52_RESIDENT_DECODE_STAGE_TEST_ARCHIVE) $(COMPILER_LIBRARY) $(RUNTIME_LIBRARY) $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_glm52_resident_decode_stage_production_runner: tests/test_glm52_resident_decode_stage_production_runner.c modules/glm52_resident_decode_stage/source/spark_glm52_resident_decode_stage_production_runner.c modules/glm52_resident_decode_stage/include/sparkpipe/spark_glm52_resident_decode_stage_production_runner.h $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests -Imodules/glm52_resident_decode_stage/include $(CFLAGS) tests/test_glm52_resident_decode_stage_production_runner.c modules/glm52_resident_decode_stage/source/spark_glm52_resident_decode_stage_production_runner.c $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

glm52_quantized_readiness_test: build/test_glm52_pp13_runtime build/test_glm52_stagepack build/test_glm52_cuda_resident_gate build/test_model_description
	./build/test_glm52_pp13_runtime
	./build/test_glm52_stagepack
	./build/test_glm52_cuda_resident_gate
	./build/test_model_description
	python3 tests/test_glm52_stage_pack.py
	python3 tests/test_glm52_nvfp4_artifact_preflight.py
	python3 tests/test_glm52_quantized_cuda_contract.py
	python3 tests/test_glm52_b12x_resident_manifest.py
	python3 tests/test_release_assemble.py

test: $(TEST_BINARIES) build/sparkpipe_glm52_prefill_dryrun
	@set -e; \
	for test_binary in $(TEST_BINARIES); do \
		echo "RUN $$test_binary"; \
		./$$test_binary; \
	done; \
	for python_test in $(PYTHON_TESTS); do \
		echo "RUN $$python_test"; \
		python3 $$python_test; \
	done

demo: all $(TEST_MODULE_OBJECTS) $(TEST_VALIDATOR)
	rm -rf build/demo
	mkdir -p build/demo
	build/sparkpipe_module_publish --library build/demo/library --module spark.test.add_one.v1 --target host.cpu --link-unit build/test_modules/module_add_one.o --recipe test.module.validator.v1 --initialize SparkTestAddOneInitialize --execute SparkTestAddOneExecute --destroy SparkTestAddOneDestroy --validator build/test_module_validator --validator-arg build/demo/validator_count.txt
	build/sparkpipe_module_publish --library build/demo/library --module spark.test.add_one.v1 --target host.cpu --link-unit build/test_modules/module_add_one.o --recipe test.module.validator.v1 --initialize SparkTestAddOneInitialize --execute SparkTestAddOneExecute --destroy SparkTestAddOneDestroy --validator build/test_module_validator --validator-arg build/demo/validator_count.txt
	build/sparkpipe_module_publish --library build/demo/library --module spark.test.double.v1 --target host.cpu --link-unit build/test_modules/module_double.o --recipe test.module.validator.v1 --execute SparkTestDoubleExecute --validator build/test_module_validator --validator-arg build/demo/validator_count.txt
	build/sparkpipe_module_publish --library build/demo/library --module spark.test.double.v1 --target host.accelerator --link-unit build/test_modules/module_double.o --recipe test.module.validator.v1 --execute SparkTestDoubleExecute --validator build/test_module_validator --validator-arg build/demo/validator_count.txt
	test "$$(cat build/demo/validator_count.txt)" = "3"
	build/sparkpipe_model_compile --model examples/model_descriptions/firmware_demo.json --library build/demo/library --output build/demo/package --include include
	build/sparkpipe_driver_inspect build/demo/package/stages/stage_000/model_driver.so host.cpu
	build/sparkpipe_driver_inspect build/demo/package/stages/stage_001/model_driver.so host.accelerator

cuda_glm52_resident_decode_stage:
	@if ! command -v $(NVCC) >/dev/null 2>&1; then \
		echo "cuda_glm52_resident_decode_stage skipped: nvcc unavailable"; \
	else \
		$(MAKE) -C modules/glm52_resident_decode_stage archive NVCC=$(NVCC) CUDA_ARCH=sm_121a; \
	fi

glm52_fp8_scaled_gemm_cuda_gate: cuda_glm52_resident_decode_stage $(B12X_ADAPTER_ARCHIVE) $(B12X_COMPILED_BACKEND_ARCHIVE) $(B12X_GENERATED_KERNEL_TABLE_ARCHIVE)
	mkdir -p build
	$(NVCC) -std=c++17 $(NVCCFLAGS) \
		-Iinclude \
		-Imodules/glm52_resident_decode_stage/include \
		-Ithird_party/flashinfer/include \
		-Ithird_party/flashinfer/3rdparty/cutlass/include \
		-Ithird_party/flashinfer/3rdparty/cutlass/tools/util/include \
		tools/glm52_fp8_scaled_gemm_cuda_gate.cu \
		$(GLM52_STAGE_SWEEP_MODULE_ARCHIVE) \
		$(GLM52_PP13_NODE_CONTEXT_BUILDER_LINK_ARGS) \
		-lcublasLt -lcublas -ldl \
		-o $(GLM52_FP8_SCALED_GEMM_CUDA_GATE)




cuda_glm52_resident_decode_stage_publish: $(B12X_ADAPTER_ARCHIVE) $(B12X_COMPILED_BACKEND_ARCHIVE) $(B12X_GENERATED_KERNEL_TABLE_ARCHIVE)
	@if ! command -v $(NVCC) >/dev/null 2>&1; then \
		echo "cuda_glm52_resident_decode_stage_publish skipped: nvcc unavailable"; \
	else \
		if [ -n "$(GLM52_REQUIRED_CUDA_LINK_ARGS)" ]; then \
			required_cuda_link_args='$(GLM52_REQUIRED_CUDA_LINK_ARGS)'; \
		else \
			required_cuda_link_args='$(GLM52_PP13_NODE_CONTEXT_BUILDER_DEFAULT_LINK_ARGS)'; \
		fi; \
		$(MAKE) -C modules/glm52_resident_decode_stage publish \
			NVCC=$(NVCC) \
			CUDA_ARCH=sm_121a \
			MAX_STAGE_MICROSECONDS=$(MAX_STAGE_MICROSECONDS) \
			REQUIRED_CUDA_CC_ARGS='$(REQUIRED_CUDA_CC_ARGS)' \
			GLM52_REQUIRED_CUDA_LINK_ARGS="$$required_cuda_link_args" \
			GLM52_REQUIRE_B12X_RESIDENT_PACK='$(if $(filter nvfp4,$(GLM52_MOE_BACKEND)),1,0)' \
			GLM52_MODEL_DIR='$(GLM52_MODEL_DIR)' \
			GLM52_ALLOW_REMOTE_MODEL_DIR='$(GLM52_ALLOW_REMOTE_MODEL_DIR)' \
			GLM52_VALIDATION_MODE='$(GLM52_VALIDATION_MODE)' \
			GLM52_VALIDATION_ACTIVE_SEQUENCE_COUNT='$(GLM52_VALIDATION_ACTIVE_SEQUENCE_COUNT)' \
			GLM52_VALIDATION_FIRST_ROUTED_LAYER_INDEX='$(GLM52_VALIDATION_FIRST_ROUTED_LAYER_INDEX)' \
			GLM52_VALIDATION_ROUTED_CHAIN_LAYER_COUNT='$(GLM52_VALIDATION_ROUTED_CHAIN_LAYER_COUNT)' \
			GLM52_PIPELINE_INPUT_HIDDEN_BF16='$(GLM52_PIPELINE_INPUT_HIDDEN_BF16)' \
			GLM52_PIPELINE_OUTPUT_HIDDEN_BF16='$(GLM52_PIPELINE_OUTPUT_HIDDEN_BF16)' \
			GLM52_ENABLE_CUDA_GRAPH_REPLAY='$(GLM52_ENABLE_CUDA_GRAPH_REPLAY)' \
			GLM52_EXACT_PP13_MODEL_QUANTIZATION='$(GLM52_EXACT_PP13_MODEL_QUANTIZATION)' \
			GLM52_STAGE_PACK_DIR='$(GLM52_STAGE_PACK_DIR)' \
			GLM52_FP8_MOE_PACK_DIR='$(GLM52_FP8_MOE_PACK_DIR)' \
			B12X_MOE_PACK_DIR='$(abspath $(B12X_MOE_PACK_OUTPUT_DIR))' \
			B12X_MOE_PACK_LAYERS='$(B12X_MOE_PACK_LAYERS)' \
			B12X_MOE_PACK_REQUIRE_REUSE='$(B12X_MOE_PACK_REQUIRE_REUSE)' \
			B12X_MOE_PACK_VERIFY_REUSED_SHA256='$(B12X_MOE_PACK_VERIFY_REUSED_SHA256)' \
			B12X_MOE_PACK_JOBS='$(B12X_MOE_PACK_JOBS)' \
			B12X_MOE_PACK_PACKAGE_MODE='$(B12X_MOE_PACK_PACKAGE_MODE)' \
			B12X_PACK_PYTHON='$(B12X_PACK_PYTHON)' \
			AOT_MANIFEST='$(abspath $(B12X_AOT_OUTPUT_DIR))/generated/aot_manifest.json'; \
	fi

$(B12X_ADAPTER_ARCHIVE):
	$(MAKE) -C modules/glm52_sm121_flashinfer_b12x_moe archive

glm52_flashinfer_b12x_moe_adapter: $(B12X_ADAPTER_ARCHIVE)

glm52_b12x_prepare_spark_env:
	tools/glm52_b12x_prepare_spark_env.sh

glm52_b12x_aot_compile:
	@test -f "$(SPARKPIPE_B12X_AOT_ENV)" || \
		{ echo "missing $(SPARKPIPE_B12X_AOT_ENV); run make glm52_b12x_prepare_spark_env first" >&2; exit 2; }
	. "$(SPARKPIPE_B12X_AOT_ENV)" && \
		"$$SPARKPIPE_B12X_AOT_PYTHON" ./tools/glm52_b12x_aot_compile.py \
		--tokens "$(B12X_AOT_TOKENS)" \
		--warmup "$(B12X_AOT_WARMUP)" \
		--iterations "$(B12X_AOT_ITERATIONS)" \
		$(B12X_AOT_BENCHMARK) \
		--output-dir "$(B12X_AOT_OUTPUT_DIR)"

glm52_b12x_resident_pack:
	@test -n "$(GLM52_MODEL_DIR)" || \
		{ echo "set GLM52_MODEL_DIR to the live GLM artifact directory" >&2; exit 2; }
	@test -s "$(B12X_AOT_OUTPUT_DIR)/generated/aot_manifest.json" || \
		{ echo "missing $(B12X_AOT_OUTPUT_DIR)/generated/aot_manifest.json; run make glm52_b12x_aot_compile first" >&2; exit 2; }
	@test -f "$(SPARKPIPE_B12X_AOT_ENV)" || \
		{ echo "missing $(SPARKPIPE_B12X_AOT_ENV); run make glm52_b12x_prepare_spark_env first" >&2; exit 2; }
	. "$(SPARKPIPE_B12X_AOT_ENV)" && \
		"$$SPARKPIPE_B12X_AOT_PYTHON" ./tools/glm52_b12x_resident_pack.py \
			--model-dir "$(GLM52_MODEL_DIR)" \
				--aot-manifest "$(B12X_AOT_OUTPUT_DIR)/generated/aot_manifest.json" \
					--layers "$(B12X_MOE_PACK_LAYERS)" \
					--output-dir "$(B12X_MOE_PACK_OUTPUT_DIR)" \
					--jobs "$(B12X_MOE_PACK_JOBS)" \
					--reuse-valid \
				$(if $(filter 1,$(B12X_MOE_PACK_VERIFY_REUSED_SHA256)),--verify-reused-sha256,)

glm52_dspark_draft_backend:
	$(MAKE) -C modules/glm52_dspark_draft_backend archive NVCC=$(NVCC) CUDA_ARCH=sm_121a

$(B12X_COMPILED_BACKEND_ARCHIVE):
	$(MAKE) -C modules/glm52_sm121_b12x_compiled_backend archive NVCC=$(NVCC) CUDA_ARCH=sm_121a

$(B12X_GENERATED_KERNEL_TABLE_ARCHIVE):
ifeq ($(GLM52_MOE_BACKEND),nvfp4)
	$(MAKE) -C modules/glm52_sm121_b12x_compiled_backend generated_archive NVCC=$(NVCC) CUDA_ARCH=sm_121a GENERATED_DIRECTORY=$(abspath $(B12X_AOT_OUTPUT_DIR)/generated)
else
	$(MAKE) -C modules/glm52_sm121_b12x_compiled_backend unavailable_archive
endif

glm52_b12x_compiled_backend: $(B12X_COMPILED_BACKEND_ARCHIVE) $(B12X_GENERATED_KERNEL_TABLE_ARCHIVE)

glm52_required_cuda_link_args: $(B12X_ADAPTER_ARCHIVE) $(B12X_COMPILED_BACKEND_ARCHIVE) $(B12X_GENERATED_KERNEL_TABLE_ARCHIVE)
ifeq ($(GLM52_MOE_BACKEND),nvfp4)
	@test -s "$(B12X_RUNTIME_LINK_ARGS_FILE)" || \
		{ echo "missing $(B12X_RUNTIME_LINK_ARGS_FILE); run make glm52_b12x_aot_compile first" >&2; exit 2; }
	@printf "%s %s %s " "$(B12X_ADAPTER_ARCHIVE)" "$(B12X_COMPILED_BACKEND_ARCHIVE)" "$(B12X_GENERATED_KERNEL_TABLE_ARCHIVE)"
	@cat "$(B12X_RUNTIME_LINK_ARGS_FILE)"
else
	@printf "%s %s %s\n" "$(B12X_ADAPTER_ARCHIVE)" "$(B12X_COMPILED_BACKEND_ARCHIVE)" "$(B12X_GENERATED_KERNEL_TABLE_ARCHIVE)"
endif

glm52_stage_bucket_sweep:
	@test -n "$(GLM52_PIPELINE_INPUT_HIDDEN_BF16)" || \
		{ echo "set GLM52_PIPELINE_INPUT_HIDDEN_BF16 to a one-vector or B-vector hidden BF16 file" >&2; exit 2; }
	@test -s "$(GLM52_PIPELINE_INPUT_HIDDEN_BF16)" || \
		{ echo "missing GLM52_PIPELINE_INPUT_HIDDEN_BF16: $(GLM52_PIPELINE_INPUT_HIDDEN_BF16)" >&2; exit 2; }
	python3 ./tools/glm52_stage_bucket_sweep.py \
		--buckets "$(GLM52_STAGE_SWEEP_BUCKETS)" \
		--input-hidden "$(GLM52_PIPELINE_INPUT_HIDDEN_BF16)" \
		--output-dir "$(GLM52_STAGE_SWEEP_OUTPUT_DIR)" \
		--max-stage-us "$(GLM52_STAGE_SWEEP_MAX_STAGE_US)" \
		--warmup-runs "$(GLM52_STAGE_SWEEP_WARMUP_RUNS)" \
		--measure-runs "$(GLM52_STAGE_SWEEP_MEASURE_RUNS)" \
		--model-dir "$(GLM52_MODEL_DIR)" \
		--nvcc "$(NVCC)" \
		--aot-env "$(SPARKPIPE_B12X_AOT_ENV)" \
		--aot-output-dir "$(B12X_AOT_OUTPUT_DIR)" \
		--b12x-moe-pack-dir "$(B12X_MOE_PACK_OUTPUT_DIR)" \
		--b12x-moe-pack-layers "$(B12X_MOE_PACK_LAYERS)" \
		--module-archive "$(GLM52_STAGE_SWEEP_MODULE_ARCHIVE)" \
		$(if $(GLM52_STAGE_SWEEP_DRIVER_SO),--driver-so "$(GLM52_STAGE_SWEEP_DRIVER_SO)",) \
		--validator-cache-dir "$(GLM52_STAGE_SWEEP_VALIDATOR_CACHE_DIR)" \
		$(if $(GLM52_STAGE_SWEEP_REQUIRED_CUDA_LINK_ARGS),--required-cuda-link-args "$(GLM52_STAGE_SWEEP_REQUIRED_CUDA_LINK_ARGS)",) \
		$(if $(filter 1,$(GLM52_STAGE_SWEEP_PACKAGE_EACH_RUN)),--package-each-run,) \
		$(if $(filter 1,$(GLM52_STAGE_SWEEP_FORCE_VALIDATOR_REBUILD)),--force-validator-rebuild,) \
		$(if $(filter 1,$(B12X_MOE_PACK_REQUIRE_REUSE)),--require-pack-reuse,--allow-pack-build) \
		$(if $(filter 1,$(B12X_MOE_PACK_VERIFY_REUSED_SHA256)),--verify-reused-sha256,) \
		$(GLM52_STAGE_SWEEP_STAGE_ARGS) \
		$(if $(filter 1,$(GLM52_ENABLE_CUDA_GRAPH_REPLAY)),--graph,)

glm52_spark2_accuracy_gate:
	bash tools/glm52_spark2_accuracy_gate.sh

glm52_spark2_local_pipeline_gate:
	bash tools/glm52_spark2_local_pipeline_gate.sh

glm52_resident_decode_stage_firmware_package: $(B12X_ADAPTER_ARCHIVE) $(B12X_COMPILED_BACKEND_ARCHIVE) $(B12X_GENERATED_KERNEL_TABLE_ARCHIVE)
	@command -v $(NVCC) >/dev/null 2>&1 || \
		{ echo "missing nvcc for required GLM52 SM121 package build" >&2; exit 2; }
ifeq ($(GLM52_MOE_BACKEND),nvfp4)
	@test -f "$(SPARKPIPE_B12X_AOT_ENV)" || \
		{ echo "missing $(SPARKPIPE_B12X_AOT_ENV); run make glm52_b12x_prepare_spark_env first" >&2; exit 2; }
	@test -s "$(B12X_RUNTIME_LINK_ARGS_FILE)" || \
		{ echo "missing $(B12X_RUNTIME_LINK_ARGS_FILE); run make glm52_b12x_aot_compile first" >&2; exit 2; }
	. "$(SPARKPIPE_B12X_AOT_ENV)" && \
		if [ -n "$(GLM52_REQUIRED_CUDA_LINK_ARGS)" ]; then \
			required_cuda_link_args='$(GLM52_REQUIRED_CUDA_LINK_ARGS)'; \
		else \
			required_cuda_link_args='$(B12X_ADAPTER_ARCHIVE) $(B12X_COMPILED_BACKEND_ARCHIVE) $(B12X_GENERATED_KERNEL_TABLE_ARCHIVE) '"$$(cat "$(B12X_RUNTIME_LINK_ARGS_FILE)")"; \
		fi; \
		$(MAKE) -C modules/glm52_resident_decode_stage package NVCC=$(NVCC) CUDA_HOME='$(CUDA_HOME)' CUDA_ARCH=sm_121a MAX_STAGE_MICROSECONDS=$(MAX_STAGE_MICROSECONDS) REQUIRED_CUDA_CC_ARGS='$(REQUIRED_CUDA_CC_ARGS)' GLM52_REQUIRED_CUDA_LINK_ARGS="$$required_cuda_link_args" B12X_MOE_PACK_DIR='$(abspath $(B12X_MOE_PACK_OUTPUT_DIR))' B12X_MOE_PACK_LAYERS='$(B12X_MOE_PACK_LAYERS)' B12X_MOE_PACK_REQUIRE_REUSE='$(B12X_MOE_PACK_REQUIRE_REUSE)' B12X_MOE_PACK_VERIFY_REUSED_SHA256='$(B12X_MOE_PACK_VERIFY_REUSED_SHA256)' B12X_MOE_PACK_JOBS='$(B12X_MOE_PACK_JOBS)' B12X_MOE_PACK_PACKAGE_MODE='$(B12X_MOE_PACK_PACKAGE_MODE)' GLM52_MODEL_DIR='$(GLM52_MODEL_DIR)' GLM52_VALIDATION_MODE='$(GLM52_VALIDATION_MODE)' GLM52_VALIDATION_ACTIVE_SEQUENCE_COUNT='$(GLM52_VALIDATION_ACTIVE_SEQUENCE_COUNT)' GLM52_VALIDATION_FIRST_ROUTED_LAYER_INDEX='$(GLM52_VALIDATION_FIRST_ROUTED_LAYER_INDEX)' GLM52_VALIDATION_ROUTED_CHAIN_LAYER_COUNT='$(GLM52_VALIDATION_ROUTED_CHAIN_LAYER_COUNT)' GLM52_PIPELINE_INPUT_HIDDEN_BF16='$(GLM52_PIPELINE_INPUT_HIDDEN_BF16)' GLM52_PIPELINE_OUTPUT_HIDDEN_BF16='$(GLM52_PIPELINE_OUTPUT_HIDDEN_BF16)' GLM52_ENABLE_CUDA_GRAPH_REPLAY='$(GLM52_ENABLE_CUDA_GRAPH_REPLAY)' GLM52_EXACT_PP13_MODEL_QUANTIZATION='$(GLM52_EXACT_PP13_MODEL_QUANTIZATION)' GLM52_STAGE_PACK_DIR='$(GLM52_STAGE_PACK_DIR)' GLM52_FP8_MOE_PACK_DIR='$(GLM52_FP8_MOE_PACK_DIR)' B12X_PACK_PYTHON="$$SPARKPIPE_B12X_AOT_PYTHON" AOT_MANIFEST='$(abspath $(B12X_AOT_OUTPUT_DIR))/generated/aot_manifest.json'
else
	@if [ -n "$(GLM52_REQUIRED_CUDA_LINK_ARGS)" ]; then \
		required_cuda_link_args='$(GLM52_REQUIRED_CUDA_LINK_ARGS)'; \
	else \
		required_cuda_link_args='$(B12X_ADAPTER_ARCHIVE) $(B12X_COMPILED_BACKEND_ARCHIVE) $(B12X_GENERATED_KERNEL_TABLE_ARCHIVE)'; \
	fi; \
	$(MAKE) -C modules/glm52_resident_decode_stage package NVCC=$(NVCC) CUDA_HOME='$(CUDA_HOME)' CUDA_ARCH=sm_121a MAX_STAGE_MICROSECONDS=$(MAX_STAGE_MICROSECONDS) REQUIRED_CUDA_CC_ARGS='$(REQUIRED_CUDA_CC_ARGS)' GLM52_REQUIRED_CUDA_LINK_ARGS="$$required_cuda_link_args" B12X_MOE_PACK_DIR='$(abspath $(B12X_MOE_PACK_OUTPUT_DIR))' B12X_MOE_PACK_LAYERS='$(B12X_MOE_PACK_LAYERS)' B12X_MOE_PACK_REQUIRE_REUSE='$(B12X_MOE_PACK_REQUIRE_REUSE)' B12X_MOE_PACK_VERIFY_REUSED_SHA256='$(B12X_MOE_PACK_VERIFY_REUSED_SHA256)' B12X_MOE_PACK_JOBS='$(B12X_MOE_PACK_JOBS)' B12X_MOE_PACK_PACKAGE_MODE='$(B12X_MOE_PACK_PACKAGE_MODE)' GLM52_MODEL_DIR='$(GLM52_MODEL_DIR)' GLM52_VALIDATION_MODE='$(GLM52_VALIDATION_MODE)' GLM52_VALIDATION_ACTIVE_SEQUENCE_COUNT='$(GLM52_VALIDATION_ACTIVE_SEQUENCE_COUNT)' GLM52_VALIDATION_FIRST_ROUTED_LAYER_INDEX='$(GLM52_VALIDATION_FIRST_ROUTED_LAYER_INDEX)' GLM52_VALIDATION_ROUTED_CHAIN_LAYER_COUNT='$(GLM52_VALIDATION_ROUTED_CHAIN_LAYER_COUNT)' GLM52_PIPELINE_INPUT_HIDDEN_BF16='$(GLM52_PIPELINE_INPUT_HIDDEN_BF16)' GLM52_PIPELINE_OUTPUT_HIDDEN_BF16='$(GLM52_PIPELINE_OUTPUT_HIDDEN_BF16)' GLM52_ENABLE_CUDA_GRAPH_REPLAY='$(GLM52_ENABLE_CUDA_GRAPH_REPLAY)' GLM52_EXACT_PP13_MODEL_QUANTIZATION='$(GLM52_EXACT_PP13_MODEL_QUANTIZATION)' GLM52_STAGE_PACK_DIR='$(GLM52_STAGE_PACK_DIR)' GLM52_FP8_MOE_PACK_DIR='$(GLM52_FP8_MOE_PACK_DIR)' GLM52_REQUIRE_B12X_RESIDENT_PACK=0 B12X_PACK_PYTHON="$(B12X_PACK_PYTHON)" AOT_MANIFEST=''
endif

tree_summary:
	@printf "public_headers="; find include/sparkpipe -type f | wc -l
	@printf "common_sources="; printf '%s\n' $(COMMON_SOURCES) | wc -l
	@printf "compiler_sources="; printf '%s\n' $(COMPILER_SOURCES) | wc -l
	@printf "runtime_sources="; printf '%s\n' $(RUNTIME_SOURCES) | wc -l
	@printf "test_executables="; printf '%s\n' $(TEST_NAMES) | wc -l

clean:
	rm -rf build

-include $(COMMON_OBJECTS:.o=.d) $(COMPILER_OBJECTS:.o=.d) \
    $(RUNTIME_OBJECTS:.o=.d) $(TEST_SUPPORT_OBJECT:.o=.d) \
    $(TEST_MODULE_DEPENDENCIES) \
    $(GLM52_RESIDENT_DECODE_STAGE_TEST_DEPENDENCIES)
