CC ?= cc
CXX ?= c++
AR ?= ar
NVCC ?= nvcc
CUDA_HOME ?= /usr/local/cuda
CFLAGS ?= -std=c11 -Wall -Wextra -Werror -O3 -g -pthread
CXXFLAGS ?= -std=c++20 -Wall -Wextra -Werror -O3 -g -pthread
CORE_INCLUDE_FLAGS := -I. -Iinclude -Isrc
# stage_module_common.c reaches <cuda_runtime.h> through spark_stage_module_common.h.
# Nothing in this group supplied that path, so the archive could not have built;
# the tests hid it behind -Itests/cuda_stub.
CUDA_RUNTIME_HEADER := $(wildcard $(CUDA_HOME)/include/cuda_runtime.h)
ifeq ($(CUDA_RUNTIME_HEADER),)
MODEL_COMMON_INCLUDE_FLAGS := $(CORE_INCLUDE_FLAGS) -Itests/cuda_stub
SPARKPIPE_HOST_CUDA_STUB_SOURCE := tests/cuda_stub/cuda_runtime_stub.c
else
MODEL_COMMON_INCLUDE_FLAGS := $(CORE_INCLUDE_FLAGS) -I$(CUDA_HOME)/include
SPARKPIPE_HOST_CUDA_STUB_SOURCE :=
endif
GLM52_INCLUDE_FLAGS := $(MODEL_COMMON_INCLUDE_FLAGS) -Imodel-families/glm52/include
QWEN36_INCLUDE_FLAGS := $(MODEL_COMMON_INCLUDE_FLAGS) -Imodel-families/qwen36/include
DSV4_INCLUDE_FLAGS := $(MODEL_COMMON_INCLUDE_FLAGS) -Imodel-families/dsv4/include
K3_INCLUDE_FLAGS := $(MODEL_COMMON_INCLUDE_FLAGS) -Imodel-families/k3/include
MIMO25_INCLUDE_FLAGS := $(MODEL_COMMON_INCLUDE_FLAGS) -Imodel-families/mimo25/include
MODEL_FAMILY_INCLUDE_FLAGS := \
    \
    -Imodel-families/glm52/include \
    -Imodel-families/qwen36/include \
    -Imodel-families/dsv4/include \
    -Imodel-families/k3/include \
    -Imodel-families/mimo25/include
DEPLOYMENT_INCLUDE_FLAGS := $(CORE_INCLUDE_FLAGS) -Ideployment/include -Ideployment/src
CPPFLAGS ?= $(CORE_INCLUDE_FLAGS) $(MODEL_FAMILY_INCLUDE_FLAGS) -Ideployment/include -Ideployment/src
LDFLAGS ?=
LDLIBS ?= -ldl -pthread
CUDA_ARCH ?= sm_121a
CUDA_COMPUTE_ARCH ?= $(subst sm_,compute_,$(CUDA_ARCH))
NVCCFLAGS ?= -O3 -gencode arch=$(CUDA_COMPUTE_ARCH),code=$(CUDA_ARCH)
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
GLM52_RING_NODE_CONTEXT_BUILDER_RPATH :=
ifneq ($(UNAME_S),Darwin)
GLM52_RING_NODE_CONTEXT_BUILDER_RPATH := -Xlinker -rpath -Xlinker '$$ORIGIN/runtime_libs'
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
GLM52_EXACT_RING_MODEL_QUANTIZATION ?= $(GLM52_MOE_BACKEND)
GLM52_MODEL_DIR ?= $(HOME)/models/hf/zai-org/GLM-5.2-FP8
GLM52_STAGE_PACK_DIR ?= $(HOME)/models/sparkpipe/glm52_fp8_ring_stage_payload_v1
GLM52_FP8_MOE_PACK_DIR ?= $(GLM52_STAGE_PACK_DIR)
GLM52_W8LUT_MOE_PACK_DIR ?=
B12X_AOT_BENCHMARK ?= --benchmark
B12X_MOE_PACK_OUTPUT_DIR ?= build/glm52_b12x_resident_moe
B12X_MOE_PACK_LAYERS ?= 3,4,5,6,7,8,9,10
B12X_MOE_PACK_REQUIRE_REUSE ?= 1
B12X_MOE_PACK_VERIFY_REUSED_SHA256 ?= 0
B12X_MOE_PACK_JOBS ?= 1
B12X_MOE_PACK_PACKAGE_MODE ?= hardlink
W8LUT_MODEL_DIR ?=
W8LUT_MOE_PACK_OUTPUT_DIR ?=
W8LUT_MOE_PACK_LAYERS ?= 3,4,5,6,7,8
W8LUT_MOE_PACK_JOBS ?= 1
W8LUT_MOE_PACK_MAX_ACTIVE ?= 1024
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
GLM52_RING_NODE_CONTEXT_BUILDER_DEFAULT_LINK_ARGS := $(B12X_ADAPTER_ARCHIVE) $(B12X_COMPILED_BACKEND_ARCHIVE) $(B12X_GENERATED_KERNEL_TABLE_ARCHIVE) $(shell cat "$(B12X_RUNTIME_LINK_ARGS_FILE)" 2>/dev/null)
else
GLM52_RING_NODE_CONTEXT_BUILDER_DEFAULT_LINK_ARGS := $(B12X_ADAPTER_ARCHIVE) $(B12X_COMPILED_BACKEND_ARCHIVE) $(B12X_GENERATED_KERNEL_TABLE_ARCHIVE)
endif
GLM52_RING_NODE_CONTEXT_BUILDER_LINK_ARGS ?= $(if $(GLM52_REQUIRED_CUDA_LINK_ARGS),$(GLM52_REQUIRED_CUDA_LINK_ARGS),$(GLM52_RING_NODE_CONTEXT_BUILDER_DEFAULT_LINK_ARGS))
GLM52_RING_NODE_CONTEXT_BUILDER := build/libglm52_ring_node_context_builder.$(SHARED_LIBRARY_EXT)
HIDDEN_TRANSPORT_SPARK_HOST_RDMA := build/libhidden_transport_spark_host_rdma_verbs.$(SHARED_LIBRARY_EXT)
HIDDEN_TRANSPORT_SPARK_GPUDIRECT_RDMA := build/libhidden_transport_spark_gpudirect_rdma_verbs.$(SHARED_LIBRARY_EXT)

include sources.mk

CORE_SOURCES := $(SPARKPIPE_CORE_SOURCES)
MODEL_COMMON_SOURCES := $(SPARKPIPE_MODEL_COMMON_SOURCES) $(SPARKPIPE_HOST_CUDA_STUB_SOURCE)
DEPLOYMENT_SOURCES := $(SPARKPIPE_DEPLOYMENT_SOURCES)
GLM52_HOST_SOURCES := $(SPARKPIPE_GLM52_SOURCES)
QWEN36_HOST_SOURCES := $(SPARKPIPE_QWEN36_SOURCES)
DSV4_HOST_SOURCES := $(SPARKPIPE_DSV4_SOURCES)
COMPILER_SOURCES := $(SPARKPIPE_COMPILER_SOURCES)
RUNTIME_SOURCES := $(SPARKPIPE_RUNTIME_SOURCES)

# Source path to object path, for any source under any directory. The full
# source path is preserved under build/obj so two files with the same basename
# in different directories cannot collide. A prefix-stripping patsubst cannot
# do this: it returns a non-matching entry unchanged, which is how seven .c
# paths came to sit in *_OBJECTS and abort the parse via -include.
sp_objects = $(patsubst %.c,build/obj/%.o,$(1))

CORE_OBJECTS := $(call sp_objects,$(CORE_SOURCES))
MODEL_COMMON_OBJECTS := $(call sp_objects,$(MODEL_COMMON_SOURCES))
DEPLOYMENT_OBJECTS := $(call sp_objects,$(DEPLOYMENT_SOURCES))
GLM52_HOST_OBJECTS := $(call sp_objects,$(GLM52_HOST_SOURCES))
QWEN36_HOST_OBJECTS := $(call sp_objects,$(QWEN36_HOST_SOURCES))
DSV4_HOST_OBJECTS := $(call sp_objects,$(DSV4_HOST_SOURCES))
COMPILER_OBJECTS := $(call sp_objects,$(COMPILER_SOURCES))
RUNTIME_OBJECTS := $(call sp_objects,$(RUNTIME_SOURCES))
ALL_HOST_OBJECTS := $(CORE_OBJECTS) $(MODEL_COMMON_OBJECTS) $(DEPLOYMENT_OBJECTS) \
    $(GLM52_HOST_OBJECTS) $(QWEN36_HOST_OBJECTS) $(DSV4_HOST_OBJECTS) $(COMPILER_OBJECTS) $(RUNTIME_OBJECTS)

# Every object is built by the one rule below. Include flags are attached per
# object list rather than per directory, so moving a source does not change how
# it is compiled.
$(CORE_OBJECTS): SP_INCLUDE_FLAGS = $(CORE_INCLUDE_FLAGS)
$(COMPILER_OBJECTS): SP_INCLUDE_FLAGS = $(CORE_INCLUDE_FLAGS)
$(RUNTIME_OBJECTS): SP_INCLUDE_FLAGS = $(CORE_INCLUDE_FLAGS)
$(MODEL_COMMON_OBJECTS): SP_INCLUDE_FLAGS = $(MODEL_COMMON_INCLUDE_FLAGS)
$(DEPLOYMENT_OBJECTS): SP_INCLUDE_FLAGS = $(DEPLOYMENT_INCLUDE_FLAGS)
$(GLM52_HOST_OBJECTS): SP_INCLUDE_FLAGS = $(GLM52_INCLUDE_FLAGS)
$(QWEN36_HOST_OBJECTS): SP_INCLUDE_FLAGS = $(QWEN36_INCLUDE_FLAGS)
$(DSV4_HOST_OBJECTS): SP_INCLUDE_FLAGS = $(DSV4_INCLUDE_FLAGS)
CORE_LIBRARY := build/libsparkpipe_core.a
MODEL_COMMON_LIBRARY := build/libsparkpipe_model_common.a
DEPLOYMENT_LIBRARY := build/libsparkpipe_deployment.a
GLM52_HOST_LIBRARY := build/libglm52_host.a
QWEN36_HOST_LIBRARY := build/libqwen36_host.a
DSV4_HOST_LIBRARY := build/libdsv4_host.a
COMPILER_LIBRARY := build/libsparkpipe_compiler.a
RUNTIME_LIBRARY := build/libsparkpipe_runtime.a
COMMON_LIBRARY := $(CORE_LIBRARY)
LIBRARIES := $(CORE_LIBRARY) $(MODEL_COMMON_LIBRARY) $(COMPILER_LIBRARY) $(RUNTIME_LIBRARY) $(DEPLOYMENT_LIBRARY) $(GLM52_HOST_LIBRARY) $(QWEN36_HOST_LIBRARY) $(DSV4_HOST_LIBRARY)

GLM52_RING_SERVICE_BACKEND := build/libglm52_ring_service_backend.$(SHARED_LIBRARY_EXT)

TOOL_NAMES := \
    sparkpipe_module_publish \
    sparkpipe_model_compile \
    sparkpipe_driver_inspect \
    sparkpipe_glm52_pipesim \
    sparkpipe_glm52_ring_rank_daemon \
    sparkpipe_glm52_cuda_residentd \
    sparkpipe_glm52_kv_jit_budget \
    sparkpipe_glm52_tokenize \
    sparkpipe_tokenize_prompt \
    sparkpipe_tokenizer_benchmark \
    sparkpipe_glm52_http_gateway \
    sparkpipe_memlink \
    sparkpipe_prevcp \
    sparkpipe_nextcp \
    sparkpipe_release_manager \
    sparkpipe_dsv4_cache_plan_report \
    spark_model_kernel_characterize \
    spark_transport_characterize \
    spark_topology_characterize \
    spark_pmtu_characterize

TOOL_BINARIES := $(addprefix build/,$(TOOL_NAMES))

TEST_NAMES := \
    test_gemm_descriptor_cache \
    test_arena \
    test_work_transaction \
    test_runtime_completion \
    test_model_runtime \
    test_distributed_work \
    test_json \
    test_hidden_transport \
    test_fabric_topology \
    test_memlink \
    test_release \
    test_glm52_kv_cache \
    test_kv_store \
    test_nvme_tier \
    test_kv_mooncake \
    test_qwen36_work_control \
    test_dsv4_cache_plan \
    test_glm52_dspark \
    test_glm52_stage_plan \
    test_glm52_mtp_tree \
    test_glm52_tp_shard \
    test_glm52_shape_config \
    test_tp_collective \
    test_glm52_row_allocator \
    test_glm52_stagepack \
    test_glm52_production_topology \
    test_glm52_ring_runtime \
    test_glm52_cuda_resident_ipc \
    test_glm52_ring_work_control \
    test_glm52_shared_prefix_admission \
    test_glm52_scheduler \
    test_glm52_prefix_cache \
    test_glm52_request_api \
    test_glm52_long_context \
    test_tokenizer \
    test_glm52_prompt_pipeline \
    test_glm52_serving_engine \
    test_glm52_service \
    test_glm52_service_backend \
    test_ring_service_backend_transactions \
    test_glm52_ring_service_backend_internal \
    test_glm52_compat_api \
    test_glm52_http_gateway \
    test_glm52_ring_rank_daemon \
    test_model_description \
    test_stage_module_common \
    test_module_library \
    test_driver_compiler \
    test_orchestrator \
    test_glm52_resident_decode_stage_firmware \
    test_glm52_resident_decode_stage_production_runner \
    test_stage_graph_replay \
    test_topology_switch

TEST_BINARIES := $(addprefix build/,$(TEST_NAMES))
PYTHON_TESTS := \
	tests/test_api_stress.py \
	tests/test_batch_variants.py \
	tests/test_code_size.py \
	tests/test_config_coverage.py \
	tests/test_cuda_performance_contracts.py \
	tests/test_cuda_math_policy.py \
	tests/test_dry_law.py \
	tests/test_dsv4_contracts.py \
	tests/test_dsv4_driver_source_contracts.py \
	tests/test_dsv4_layer_host.py \
	tests/test_expert_grouping.py \
	tests/test_fast_defaults.py \
	tests/test_gemm_k_alignment.py \
	tests/test_glm52_dspark_manifest.py \
	tests/test_glm52_dspark_trace_quality.py \
	tests/test_glm52_firmware_package.py \
	tests/test_glm52_fp8_pack_layout.py \
	tests/test_glm52_final_artifact_tools.py \
	tests/test_glm52_layer_host.py \
	tests/test_glm52_prompt_pipeline_input.py \
	tests/test_glm52_quantized_cuda_contract.py \
	tests/test_glm52_stage_pack.py \
	tests/test_glm52_unity_precision_contract.py \
	tests/test_gqa_host.py \
	tests/test_grouped_moe_source_contracts.py \
	tests/test_hardware_topology.py \
	tests/test_hardware_assumption_bindings.py \
	tests/test_hardware_job_runner.py \
	tests/test_hardware_handoff_preflight.py \
	tests/test_hardware_policy_closure.py \
	tests/test_hardware_probe_coverage.py \
	tests/test_hardware_probe_source_contracts.py \
	tests/test_hardware_runner_configs.py \
	tests/test_spark_model_kernel_probe.py \
	tests/test_spark_transport_probe.py \
	tests/test_spark_topology_probe.py \
	tests/test_spark_pmtu_probe.py \
	tests/test_k3_driver_contracts.py \
	tests/test_k3_engine.py \
	tests/test_k3_kv_geometry.py \
	tests/test_k3_layer_host.py \
	tests/test_k3_pack.py \
	tests/test_k3_pack_layout.py \
	tests/test_k3_quant_recipe.py \
	tests/test_k3_shard.py \
	tests/test_k3_shard_table.py \
	tests/test_k3_slice_host.py \
	tests/test_kda_decay.py \
	tests/test_kda_host.py \
	tests/test_kv_failure_host.py \
	tests/test_kernel_algorithms.py \
	tests/test_kernel_launches.py \
	tests/test_layer_dataflow.py \
	tests/test_layer_host.py \
	tests/test_layer_kinds.py \
	tests/test_measured_status.py \
	tests/test_memory_contracts.py \
	tests/test_mimo25_layer_host.py \
	tests/test_mla_absorption.py \
	tests/test_mla_host.py \
	tests/test_model_driver_contracts.py \
	tests/test_model_families.py \
	tests/test_must_work_targets.py \
	tests/test_nvme_kv_estimate.py \
	tests/test_package_manifest.py \
	tests/test_perf_estimate.py \
	tests/test_ptx_capability_gate.py \
	tests/test_python_syntax.py \
	tests/test_qwen36_bf16_contract.py \
	tests/test_qwen36_layer_host.py \
	tests/test_recipe_generation.py \
	tests/test_release_assemble.py \
	tests/test_rope_pairing.py \
	tests/test_router_host.py \
	tests/test_router_precision_contract.py \
	tests/test_situ_activation.py \
	tests/test_sources_exist.py \
	tests/test_status_truth.py
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
    $(GLM52_RESIDENT_DECODE_STAGE_TEST_DIRECTORY)/glm52_resident_decode_stage_fake_backend.o \
    $(GLM52_RESIDENT_DECODE_STAGE_TEST_DIRECTORY)/spark_glm52_stage_validation.o
GLM52_RESIDENT_DECODE_STAGE_TEST_DEPENDENCIES := \
    $(GLM52_RESIDENT_DECODE_STAGE_TEST_OBJECTS:.o=.d)
GLM52_RESIDENT_DECODE_STAGE_TEST_ARCHIVE := \
    $(GLM52_RESIDENT_DECODE_STAGE_TEST_DIRECTORY)/libglm52_resident_decode_stage_test.a

.PHONY: all clean test tools hardware_tools hardware_cuda_tools hardware_handoff runtime_completion_tests demo FORCE \
    cuda_glm52_resident_decode_stage \
    cuda_glm52_resident_decode_stage_variants \
    cuda_glm52_resident_decode_stage_publish \
    glm52_w8lut_quality_cuda_gate \
    glm52_w8lut_quality_reference \
    glm52_w8lut_resident_pack \
    glm52_flashinfer_b12x_moe_adapter \
    glm52_b12x_prepare_spark_env \
    glm52_b12x_aot_compile \
    glm52_b12x_resident_pack \
    glm52_b12x_compiled_backend \
    glm52_quantized_readiness_test \
    glm52_required_cuda_link_args \
    glm52_ring_service_backend \
    hidden_transport_spark_host_rdma_verbs \
    hidden_transport_spark_gpudirect_rdma_verbs \
    glm52_ring_node_context_builder \
    kv_mooncake \
    glm52_resident_decode_stage_firmware_package \
    tree_summary \
    architecture_audit \
    model_driver_contracts

# The batch-variant set is part of all: the variant IS the module, so a build
# that stops at the unbucketed archive is a partial build. The nvcc guard
# keeps host-only checkouts building - no CUDA toolchain, no variant archives,
# one skip line, same contract as the other cuda_* targets.
all: $(LIBRARIES) tools $(GLM52_RING_SERVICE_BACKEND) cuda_glm52_resident_decode_stage_variants

tools: $(TOOL_BINARIES) $(GLM52_RING_SERVICE_BACKEND)

.PHONY: core model_common deployment audit-boundaries architecture_audit model_driver_contracts non_glm_model_driver_contracts

core: $(CORE_LIBRARY) $(COMPILER_LIBRARY) $(RUNTIME_LIBRARY)

model_common: $(MODEL_COMMON_LIBRARY)

deployment: $(DEPLOYMENT_LIBRARY)

audit-boundaries: core
	python3 tools/audit_core_boundaries.py --repository . --core-archive $(CORE_LIBRARY) --compiler-archive $(COMPILER_LIBRARY) --runtime-archive $(RUNTIME_LIBRARY)

architecture_audit: audit-boundaries

non_glm_model_driver_contracts:
	@mkdir -p build/model_driver_contracts
	$(CC) -Iinclude -Wall -Wextra -Werror -DNDEBUG -c modules/dsv4_resident_decode_stage/source/spark_dsv4_resident_decode_stage_validation.c -o build/model_driver_contracts/dsv4_validation.o
	$(CC) -Iinclude -Imodel-families/k3/include -Imodel-families/glm52/include -Wall -Wextra -Werror -DNDEBUG -c modules/k3_resident_decode_stage/source/spark_k3_resident_decode_stage_validation.c -o build/model_driver_contracts/k3_validation.o
	$(CC) -Iinclude -Wall -Wextra -Werror -DNDEBUG -c modules/mimo25_resident_decode_stage/source/spark_mimo25_resident_decode_stage_validation.c -o build/model_driver_contracts/mimo25_validation.o
	$(CC) -Iinclude -Wall -Wextra -Werror -DNDEBUG -c modules/qwen36_resident_decode_stage/source/spark_qwen36_resident_decode_stage_validation.c -o build/model_driver_contracts/qwen36_validation.o

model_driver_contracts: build/test_model_description build/test_stage_module_common non_glm_model_driver_contracts
	./build/test_stage_module_common
	python3 tests/test_model_driver_contracts.py

MODEL_COMMON_LINK_TARGETS := \
    build/sparkpipe_tokenize_prompt \
    build/sparkpipe_tokenizer_benchmark \
    build/sparkpipe_memlink \
    build/sparkpipe_prevcp \
    build/sparkpipe_nextcp \
    build/test_hidden_transport \
    build/test_memlink \
    build/test_kv_store \
    build/test_kv_mooncake \
    build/test_tp_collective \
    build/test_tokenizer \
    $(HIDDEN_TRANSPORT_SPARK_HOST_RDMA) \
    $(HIDDEN_TRANSPORT_SPARK_GPUDIRECT_RDMA)
GLM52_LINK_TARGETS := \
    $(filter build/sparkpipe_glm52_% build/test_glm52_%,$(TOOL_BINARIES) $(TEST_BINARIES)) \
    build/sparkpipe_glm52_batchplane_model \
    build/test_glm52_batch_plane \
    build/test_model_description \
    $(GLM52_RING_SERVICE_BACKEND) \
    $(GLM52_RING_NODE_CONTEXT_BUILDER)
NULL_REQUEST_MODEL_OBJECT := build/obj/serving/spark_request_model_null.o

$(NULL_REQUEST_MODEL_OBJECT): serving/spark_request_model_null.c include/sparkpipe/spark_request_api.h
	@mkdir -p $(dir $@) && $(CC) -I. -Iinclude -Imodel-families/glm52/include -std=c11 -Wall -Wextra -Werror -O3 -g -c serving/spark_request_model_null.c -o $@

build/test_null_seam_link: tests/test_null_seam_link.c $(NULL_REQUEST_MODEL_OBJECT) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) -I. -Iinclude -Imodel-families/glm52/include -std=c11 -Wall -Wextra -Werror -O3 -g -pthread tests/test_null_seam_link.c $(NULL_REQUEST_MODEL_OBJECT) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) -o $@

QWEN36_LINK_TARGETS := build/test_qwen36_work_control
DSV4_LINK_TARGETS := build/test_dsv4_cache_plan build/sparkpipe_dsv4_cache_plan_report


DEPLOYMENT_LINK_TARGETS := build/sparkpipe_release_manager build/test_release

$(MODEL_COMMON_LINK_TARGETS): COMMON_LIBRARY = $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
$(MODEL_COMMON_LINK_TARGETS): $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
$(GLM52_LINK_TARGETS): COMMON_LIBRARY = $(GLM52_HOST_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
$(GLM52_LINK_TARGETS): $(GLM52_HOST_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
$(DSV4_LINK_TARGETS): $(DSV4_HOST_LIBRARY) $(CORE_LIBRARY)

$(QWEN36_LINK_TARGETS): COMMON_LIBRARY = $(QWEN36_HOST_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(NULL_REQUEST_MODEL_OBJECT)
$(QWEN36_LINK_TARGETS): $(QWEN36_HOST_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(NULL_REQUEST_MODEL_OBJECT)
$(DEPLOYMENT_LINK_TARGETS): COMMON_LIBRARY = $(DEPLOYMENT_LIBRARY) $(CORE_LIBRARY)
$(DEPLOYMENT_LINK_TARGETS): $(DEPLOYMENT_LIBRARY) $(CORE_LIBRARY)

build:
	mkdir -p build

build/test_modules:
	mkdir -p build/test_modules

$(GLM52_RESIDENT_DECODE_STAGE_TEST_DIRECTORY):
	mkdir -p $(GLM52_RESIDENT_DECODE_STAGE_TEST_DIRECTORY)

build/obj/%.o: %.c | build
	@mkdir -p $(dir $@) && $(CC) $(SP_INCLUDE_FLAGS) $(CFLAGS) -fPIC -MMD -MP -c $< -o $@

$(CORE_LIBRARY): $(CORE_OBJECTS)
	$(AR) rcs $@ $^

$(MODEL_COMMON_LIBRARY): $(MODEL_COMMON_OBJECTS)
	$(AR) rcs $@ $^

$(DEPLOYMENT_LIBRARY): $(DEPLOYMENT_OBJECTS)
	$(AR) rcs $@ $^

$(GLM52_HOST_LIBRARY): $(GLM52_HOST_OBJECTS)
	$(AR) rcs $@ $^

$(QWEN36_HOST_LIBRARY): $(QWEN36_HOST_OBJECTS)
	$(AR) rcs $@ $^

$(DSV4_HOST_LIBRARY): $(DSV4_HOST_OBJECTS)
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

build/sparkpipe_dsv4_driver_cuda_smoke: tools/sparkpipe_dsv4_driver_cuda_smoke.c $(RUNTIME_LIBRARY) $(COMMON_LIBRARY)
	$(CC) $(DSV4_INCLUDE_FLAGS) -Imodules/dsv4_resident_decode_stage/include $(CFLAGS) -I$(CUDA_HOME)/include $< $(RUNTIME_LIBRARY) $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -L$(CUDA_HOME)/lib64 -lcudart -lstdc++ -lm -o $@

build/sparkpipe_glm52_pipesim: tests/studies/sparkpipe_glm52_pipesim.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@


build/sparkpipe_dsv4_cache_plan_report: tests/studies/sparkpipe_dsv4_cache_plan_report.c $(DSV4_HOST_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(DSV4_INCLUDE_FLAGS) $(CFLAGS) $< $(DSV4_HOST_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/spark_model_kernel_characterize: tools/hardware/spark_model_kernel_characterize.c include/sparkpipe/spark_hardware_kernel_probe.h
	@mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LDFLAGS) $(LDLIBS) -lm -o $@

build/spark_transport_characterize: tools/hardware/spark_transport_characterize.c include/sparkpipe/spark_hardware_transport_probe.h
	@mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LDFLAGS) $(LDLIBS) -lm -o $@

build/spark_topology_characterize: tools/hardware/spark_topology_characterize.c include/sparkpipe/spark_hardware_topology_probe.h
	@mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LDFLAGS) $(LDLIBS) -lm -o $@

build/spark_pmtu_characterize: tools/hardware/spark_pmtu_characterize.c tools/hardware/spark_probe_common.h
	@mkdir -p build
	$(CC) -Itools/hardware $(CFLAGS) $< $(LDFLAGS) -o $@

hardware_tools: build/spark_model_kernel_characterize build/spark_transport_characterize build/spark_topology_characterize build/spark_pmtu_characterize

hardware_cuda_tools:
	@if ! command -v $(NVCC) >/dev/null 2>&1; then \
		echo "hardware_cuda_tools skipped: nvcc unavailable"; \
	else \
		set -e; \
		mkdir -p build; \
		$(NVCC) -std=c++17 -O3 -arch=sm_121a -Xptxas=-v -Itools/hardware tools/hardware/spark_cuda_characterize.cu -o build/spark_cuda_characterize; \
		$(NVCC) -std=c++17 -O3 -arch=sm_121a -Xptxas=-v -Itools/hardware -Xcompiler=-pthread tools/hardware/spark_nvme_characterize.cu -o build/spark_nvme_characterize -lpthread; \
	fi

hardware_handoff: hardware_tools
	python3 tests/test_hardware_probe_coverage.py
	python3 tests/test_hardware_assumption_bindings.py
	python3 tests/test_hardware_policy_closure.py
	python3 tests/test_hardware_job_runner.py
	python3 tests/test_hardware_handoff_preflight.py
	python3 tests/test_hardware_runner_configs.py
	python3 tests/test_hardware_probe_source_contracts.py
	python3 tests/test_spark_model_kernel_probe.py
	python3 tests/test_spark_transport_probe.py
	python3 tests/test_spark_topology_probe.py
	python3 tests/test_spark_pmtu_probe.py

build/test_dsv4_cache_plan: tests/test_dsv4_cache_plan.c $(DSV4_HOST_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(DSV4_INCLUDE_FLAGS) $(CFLAGS) $< $(DSV4_HOST_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_glm52_batch_plane: tests/test_glm52_batch_plane.c $(GLM52_HOST_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(GLM52_INCLUDE_FLAGS) $(CFLAGS) $< $(GLM52_HOST_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/sparkpipe_glm52_batchplane_model: tests/studies/sparkpipe_glm52_batchplane_model.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LDFLAGS) $(LDLIBS) -lm -o $@

build/sparkpipe_glm52_tokenize: text/tokenize_main.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/sparkpipe_tokenize_prompt: tools/sparkpipe_tokenize_prompt.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/sparkpipe_tokenizer_benchmark: tools/sparkpipe_tokenizer_benchmark.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/sparkpipe_glm52_http_gateway: api/gateway/http_server.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/sparkpipe_memlink: node/memlink_tool.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/sparkpipe_prevcp: node/memlink_tool.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DSPARK_MEMLINK_FIXED_COMMAND=\"prevcp\" $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/sparkpipe_nextcp: node/memlink_tool.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DSPARK_MEMLINK_FIXED_COMMAND=\"nextcp\" $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/sparkpipe_release_manager: deployment/tools/sparkpipe_release_manager.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/sparkpipe_glm52_ring_rank_daemon: node/rank_daemon.c inference/stage/runner.c modules/glm52_resident_decode_stage/include/sparkpipe/spark_glm52_resident_decode_stage_production_runner.h $(RUNTIME_LIBRARY) $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Imodules/glm52_resident_decode_stage/include $(CFLAGS) -I$(CUDA_HOME)/include node/rank_daemon.c inference/stage/runner.c $(RUNTIME_LIBRARY) $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/sparkpipe_glm52_cuda_residentd: node/residentd.c $(RUNTIME_LIBRARY) $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Imodules/glm52_resident_decode_stage/include $(CFLAGS) node/residentd.c $(RUNTIME_LIBRARY) $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/sparkpipe_glm52_kv_jit_budget: tests/studies/sparkpipe_glm52_kv_jit_budget.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/studies/sparkpipe_glm52_kv_jit_budget.c $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

$(GLM52_RING_SERVICE_BACKEND): node/backend.c inference/stage/runner.c modules/glm52_resident_decode_stage/include/sparkpipe/spark_glm52_resident_decode_stage_production_runner.h $(RUNTIME_LIBRARY) $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Imodules/glm52_resident_decode_stage/include $(CFLAGS) -fPIC $(SHARED_LIBRARY_FLAGS) node/backend.c inference/stage/runner.c $(RUNTIME_LIBRARY) $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

glm52_ring_service_backend: $(GLM52_RING_SERVICE_BACKEND)


$(HIDDEN_TRANSPORT_SPARK_HOST_RDMA): ring/transport/rdma.cu include/sparkpipe/spark_hidden_transport.h include/sparkpipe/spark_memlink.h $(COMMON_LIBRARY)
	@if ! command -v $(NVCC) >/dev/null 2>&1; then \
		echo "hidden_transport_spark_host_rdma_verbs failed: nvcc unavailable" >&2; exit 1; \
	elif [ ! -f "$(CUDA_HOME)/include/cuda_runtime_api.h" ]; then \
		echo "hidden_transport_spark_host_rdma_verbs failed: CUDA headers unavailable" >&2; exit 1; \
	elif [ ! -f "/usr/include/infiniband/verbs.h" ]; then \
		echo "hidden_transport_spark_host_rdma_verbs failed: libibverbs headers unavailable" >&2; exit 1; \
	else \
		$(NVCC) $(NVCCFLAGS) -DSPARK_HIDDEN_SPARK_RDMA_DEVICE_DIRECT=0 $(SHARED_LIBRARY_FLAGS) -Xcompiler -fPIC -Xcompiler -pthread $(MODEL_COMMON_INCLUDE_FLAGS) ring/transport/rdma.cu $(COMMON_LIBRARY) $(LDFLAGS) -L$(CUDA_HOME)/lib64 -lcudart -libverbs -ldl -lpthread -o $@; \
	fi

hidden_transport_spark_host_rdma_verbs: $(HIDDEN_TRANSPORT_SPARK_HOST_RDMA)

$(HIDDEN_TRANSPORT_SPARK_GPUDIRECT_RDMA): ring/transport/rdma.cu include/sparkpipe/spark_hidden_transport.h include/sparkpipe/spark_memlink.h $(COMMON_LIBRARY)
	@if ! command -v $(NVCC) >/dev/null 2>&1; then \
		echo "hidden_transport_spark_gpudirect_rdma_verbs failed: nvcc unavailable" >&2; exit 1; \
	elif [ ! -f "$(CUDA_HOME)/include/cuda_runtime_api.h" ]; then \
		echo "hidden_transport_spark_gpudirect_rdma_verbs failed: CUDA headers unavailable" >&2; exit 1; \
	elif [ ! -f "/usr/include/infiniband/verbs.h" ]; then \
		echo "hidden_transport_spark_gpudirect_rdma_verbs failed: libibverbs headers unavailable" >&2; exit 1; \
	else \
		$(NVCC) $(NVCCFLAGS) -DSPARK_HIDDEN_SPARK_RDMA_DEVICE_DIRECT=1 $(SHARED_LIBRARY_FLAGS) -Xcompiler -fPIC -Xcompiler -pthread $(MODEL_COMMON_INCLUDE_FLAGS) ring/transport/rdma.cu $(COMMON_LIBRARY) $(LDFLAGS) -L$(CUDA_HOME)/lib64 -lcudart -libverbs -ldl -lpthread -o $@; \
	fi

hidden_transport_spark_gpudirect_rdma_verbs: $(HIDDEN_TRANSPORT_SPARK_GPUDIRECT_RDMA)

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

# UNRESOLVED. The CUDA translation unit this shared library was built from,
# modules/glm52_resident_decode_stage/source/spark_glm52_ring_node_context_builder_cuda.cu,
# was deleted with the 10,628-line node context builder. Its replacement in
# inference/stage/ is not wired to a link line yet, so this target names the gap
# instead of failing on a missing prerequisite. .updaterepo-policy runs it when
# nvcc is present; it will stop here until an owner supplies the source.
$(GLM52_RING_NODE_CONTEXT_BUILDER): inference/stage/runner.c modules/glm52_resident_decode_stage/include/sparkpipe/spark_glm52_resident_decode_stage_production_runner.h modules/glm52_resident_decode_stage/include/sparkpipe/spark_glm52_resident_decode_stage_firmware.h model-families/glm52/include/sparkpipe/spark_glm52_kv_cache.h include/sparkpipe/spark_ring_work_control.h $(GLM52_STAGE_SWEEP_MODULE_ARCHIVE) $(DSPARK_DRAFT_BACKEND_ARCHIVE) $(B12X_ADAPTER_ARCHIVE) $(B12X_COMPILED_BACKEND_ARCHIVE) $(B12X_GENERATED_KERNEL_TABLE_ARCHIVE) $(COMMON_LIBRARY) $(RUNTIME_LIBRARY)
	@echo "glm52_ring_node_context_builder: source deleted, replacement not wired; see HANDOFF.md" >&2; exit 2
	@if ! command -v $(NVCC) >/dev/null 2>&1; then \
		echo "glm52_ring_node_context_builder skipped: nvcc unavailable"; \
	else \
		if [ "$(GLM52_MOE_BACKEND)" = "nvfp4" ] && [ -z "$(GLM52_REQUIRED_CUDA_LINK_ARGS)" ] && [ ! -s "$(B12X_RUNTIME_LINK_ARGS_FILE)" ]; then \
			echo "missing $(B12X_RUNTIME_LINK_ARGS_FILE); run make glm52_b12x_aot_compile first (GLM52_MOE_BACKEND=nvfp4)" >&2; \
			exit 2; \
		fi; \
	fi

glm52_ring_node_context_builder: $(GLM52_RING_NODE_CONTEXT_BUILDER)

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

$(TEST_SERVICE_BACKEND_MODULE): tests/fixtures/glm52_service_backend_module.c include/sparkpipe/spark_service_backend.h | build/test_modules
	$(CC) $(CPPFLAGS) $(CFLAGS) -fPIC $(SHARED_LIBRARY_FLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

$(TEST_VALIDATOR): tests/fixtures/module_validator.c | build
	$(CC) $(CFLAGS) $< -o $@


$(GLM52_RESIDENT_DECODE_STAGE_TEST_DIRECTORY)/spark_glm52_resident_decode_stage_module.o: inference/stage/module.c modules/glm52_resident_decode_stage/include/sparkpipe/spark_glm52_resident_decode_stage_firmware.h modules/glm52_resident_decode_stage/source/spark_glm52_resident_decode_stage_backend.h | $(GLM52_RESIDENT_DECODE_STAGE_TEST_DIRECTORY)
	$(CC) $(CPPFLAGS) -Imodules/glm52_resident_decode_stage/include -Imodules/glm52_resident_decode_stage/source $(CFLAGS) -fPIC -fvisibility=hidden -MMD -MP -c $< -o $@
$(GLM52_RESIDENT_DECODE_STAGE_TEST_DIRECTORY)/spark_glm52_stage_validation.o: modules/glm52_resident_decode_stage/source/spark_glm52_resident_decode_stage_validation.c modules/glm52_resident_decode_stage/include/sparkpipe/spark_glm52_resident_decode_stage_firmware.h modules/glm52_resident_decode_stage/source/spark_glm52_resident_decode_stage_backend.h | $(GLM52_RESIDENT_DECODE_STAGE_TEST_DIRECTORY)
	$(CC) $(CPPFLAGS) -Imodules/glm52_resident_decode_stage/include -Imodules/glm52_resident_decode_stage/source $(CFLAGS) -fPIC -fvisibility=hidden -MMD -MP -c $< -o $@

$(GLM52_RESIDENT_DECODE_STAGE_TEST_DIRECTORY)/glm52_resident_decode_stage_fake_backend.o: tests/fixtures/glm52_resident_decode_stage_fake_backend.c tests/fixtures/glm52_resident_decode_stage_fake_backend.h modules/glm52_resident_decode_stage/include/sparkpipe/spark_glm52_resident_decode_stage_firmware.h modules/glm52_resident_decode_stage/source/spark_glm52_resident_decode_stage_backend.h | $(GLM52_RESIDENT_DECODE_STAGE_TEST_DIRECTORY)
	$(CC) $(CPPFLAGS) -Itests/fixtures -Imodules/glm52_resident_decode_stage/include -Imodules/glm52_resident_decode_stage/source $(CFLAGS) -fPIC -fvisibility=hidden -MMD -MP -c $< -o $@

$(GLM52_RESIDENT_DECODE_STAGE_TEST_ARCHIVE): $(GLM52_RESIDENT_DECODE_STAGE_TEST_OBJECTS)
	rm -f $@
	$(AR) rcs $@ $^

build/test_uniform_profile_admit: tests/test_uniform_profile_admit.c $(GLM52_HOST_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(GLM52_HOST_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_hybrid_kv_arithmetic: tests/test_hybrid_kv_arithmetic.c $(GLM52_HOST_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests -Imodel-families/k3/include -Imodel-families/qwen36/include $(CFLAGS) $< $(GLM52_HOST_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_arena: tests/test_arena.c runtime/arena.h
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_arena.c $(LDFLAGS) $(LDLIBS) -o $@

build/test_work_transaction: tests/test_work_transaction.c $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_work_transaction.c $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_runtime_completion: tests/test_runtime_completion.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_runtime_completion.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_model_runtime: tests/test_model_runtime.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_model_runtime.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

runtime_completion_tests: build/test_runtime_completion build/test_model_runtime
	./build/test_runtime_completion
	./build/test_model_runtime
	python3 tests/test_glm52_final_artifact_tools.py

# Header-only cache plus a mock encode; the driver stub's cuda.h is what lets
# runtime/tensor_map.h compile with no CUDA toolkit, and its stub.c stands in
# for cuTensorMapEncodeTiled so the production entry point is exercised too.
# -x c++ on the stub keeps the symbol mangling identical to the test TU's
# (the stub predates C and C++ disagreeing on the name).
build/test_gemm_descriptor_cache: tests/test_gemm_descriptor_cache.cpp tests/cuda_driver_stub/stub.c runtime/gemm_descriptor_cache.h
	$(CXX) -x c++ -Itests/cuda_driver_stub -O2 -Wall -Wextra -c tests/cuda_driver_stub/stub.c -o build/test_gemm_descriptor_cache_stub.o
	$(CXX) $(CPPFLAGS) -I. -Itests/cuda_driver_stub $(CXXFLAGS) tests/test_gemm_descriptor_cache.cpp build/test_gemm_descriptor_cache_stub.o $(LDFLAGS) $(LDLIBS) -o $@

build/test_distributed_work: tests/test_distributed_work.c $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_distributed_work.c $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_json: tests/test_json.c $(COMPILER_LIBRARY) $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(COMPILER_LIBRARY) $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

# The stage-side graph cache and its capture/replay/fallback decision, driven
# through a recording mock of the five CUDA calls dispatch.cu provides.
build/test_stage_graph_replay: tests/test_stage_graph_replay.c inference/stage/graph_replay.h include/sparkpipe/spark_resident_decode_stage.h
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LDFLAGS) -o $@

build/test_fabric_topology: tests/test_fabric_topology.c $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(MODEL_COMMON_INCLUDE_FLAGS) $(CFLAGS) $< $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

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

# The tier and its mock device, compiled together directly: the test's device
# is a vtable implementation, so there is no library boundary to cross and no
# archive to link for two translation units.
build/test_nvme_tier: tests/test_nvme_tier.c cache/nvme_tier.c include/sparkpipe/spark_nvme_tier.h
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_nvme_tier.c cache/nvme_tier.c $(LDFLAGS) $(LDLIBS) -o $@

# The switch machine sits on the same mock-drive tier: two translation units,
# vtable devices, compiled directly like the tier test above.
build/test_topology_switch: tests/test_topology_switch.c scheduler/topology_switch.c cache/nvme_tier.c include/sparkpipe/spark_topology_switch.h include/sparkpipe/spark_nvme_tier.h
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_topology_switch.c scheduler/topology_switch.c cache/nvme_tier.c $(LDFLAGS) $(LDLIBS) -o $@

build/test_kv_mooncake: tests/test_kv_mooncake.cpp tests/fixtures/mooncake/dummy_client.cpp modules/kv_mooncake/spark_kv_mooncake.cpp $(COMMON_LIBRARY)
	$(CXX) $(CPPFLAGS) -Itests/fixtures/mooncake $(CXXFLAGS) $^ $(LDFLAGS) $(LDLIBS) -o $@

build/test_qwen36_work_control: tests/test_qwen36_work_control.cpp tests/fixtures/mooncake/dummy_client.cpp modules/kv_mooncake/spark_kv_mooncake.cpp $(COMMON_LIBRARY)
	$(CXX) $(CPPFLAGS) -Itests/fixtures/mooncake $(CXXFLAGS) $^ $(LDFLAGS) $(LDLIBS) -o $@

build/test_glm52_stage_plan: tests/test_glm52_stage_plan.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_glm52_shape_config: tests/test_glm52_shape_config.c model-families/glm52/include/sparkpipe/spark_glm52_shape_config.h $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_tp_collective: tests/test_tp_collective.c include/sparkpipe/spark_tp_collective.h $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -lpthread -o $@

build/test_glm52_tp_shard: tests/test_glm52_tp_shard.c model-families/glm52/include/sparkpipe/spark_glm52_tp_shard.h $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_glm52_row_allocator: tests/test_glm52_row_allocator.c include/sparkpipe/spark_row_allocator.h $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_glm52_mtp_tree: tests/test_glm52_mtp_tree.c include/sparkpipe/spark_mtp_tree.h $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_glm52_stagepack: tests/test_glm52_stagepack.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_glm52_dspark: tests/test_glm52_dspark.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_glm52_production_topology: tests/test_glm52_production_topology.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_glm52_ring_runtime: tests/test_glm52_ring_runtime.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_glm52_ring_work_control: tests/test_glm52_ring_work_control.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_glm52_shared_prefix_admission: tests/test_glm52_shared_prefix_admission.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_glm52_scheduler: tests/test_glm52_scheduler.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_glm52_prefix_cache: tests/test_glm52_prefix_cache.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_glm52_request_api: tests/test_glm52_request_api.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_glm52_cuda_resident_ipc: tests/test_glm52_cuda_resident_ipc.c $(COMMON_LIBRARY)
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

build/test_ring_service_backend_transactions: tests/test_ring_service_backend_transactions.c inference/stage/runner.c $(LIBRARIES)
	$(CC) $(CPPFLAGS) -Itests -Imodules/glm52_resident_decode_stage/include $(CFLAGS) tests/test_ring_service_backend_transactions.c inference/stage/runner.c $(LIBRARIES) $(LIBRARIES) $(LDFLAGS) $(LDLIBS) -o $@

build/test_glm52_ring_service_backend_internal: tests/test_glm52_ring_service_backend_internal.c inference/stage/runner.c $(LIBRARIES)
	$(CC) $(CPPFLAGS) -Itests -Imodules/glm52_resident_decode_stage/include $(CFLAGS) tests/test_glm52_ring_service_backend_internal.c inference/stage/runner.c $(LIBRARIES) $(LIBRARIES) $(LDFLAGS) $(LDLIBS) -o $@

build/test_glm52_compat_api: tests/test_glm52_compat_api.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_glm52_http_gateway: tests/test_glm52_http_gateway.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_glm52_ring_rank_daemon: tests/test_glm52_ring_rank_daemon.c inference/stage/runner.c $(RUNTIME_LIBRARY) $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests -Imodules/glm52_resident_decode_stage/include $(CFLAGS) tests/test_glm52_ring_rank_daemon.c inference/stage/runner.c $(RUNTIME_LIBRARY) $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@


build/test_model_description: tests/test_model_description.c $(COMPILER_LIBRARY) $(GLM52_HOST_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(GLM52_INCLUDE_FLAGS) -Itests $(CFLAGS) $< $(COMPILER_LIBRARY) $(GLM52_HOST_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_stage_module_common: tests/test_stage_module_common.c runtime/stage_module_common.c tests/cuda_stub/cuda_runtime_stub.c | build
	$(CC) $(MODEL_COMMON_INCLUDE_FLAGS) -Itests/cuda_stub -Itests $(CFLAGS) $^ $(LDFLAGS) -o $@

build/test_module_library: tests/test_module_library.c $(TEST_SUPPORT_OBJECT) $(TEST_MODULE_LINK_UNITS) $(TEST_VALIDATOR) $(COMPILER_LIBRARY) $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(TEST_SUPPORT_OBJECT) $(COMPILER_LIBRARY) $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_driver_compiler: tests/test_driver_compiler.c $(TEST_SUPPORT_OBJECT) $(TEST_MODULE_LINK_UNITS) $(TEST_VALIDATOR) $(COMPILER_LIBRARY) $(RUNTIME_LIBRARY) $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(TEST_SUPPORT_OBJECT) $(COMPILER_LIBRARY) $(RUNTIME_LIBRARY) $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_orchestrator: tests/test_orchestrator.c $(TEST_SUPPORT_OBJECT) $(TEST_MODULE_LINK_UNITS) $(TEST_VALIDATOR) $(COMPILER_LIBRARY) $(RUNTIME_LIBRARY) $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(TEST_SUPPORT_OBJECT) $(COMPILER_LIBRARY) $(RUNTIME_LIBRARY) $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_glm52_resident_decode_stage_firmware: tests/test_glm52_resident_decode_stage_firmware.c modules/glm52_resident_decode_stage/include/sparkpipe/spark_glm52_resident_decode_stage_firmware.h tests/fixtures/glm52_resident_decode_stage_fake_backend.h $(GLM52_RESIDENT_DECODE_STAGE_TEST_ARCHIVE) $(TEST_SUPPORT_OBJECT) $(TEST_VALIDATOR) $(COMPILER_LIBRARY) $(RUNTIME_LIBRARY) $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests -Itests/fixtures -Imodules/glm52_resident_decode_stage/include $(CFLAGS) $< $(TEST_SUPPORT_OBJECT) $(GLM52_RESIDENT_DECODE_STAGE_TEST_ARCHIVE) $(COMPILER_LIBRARY) $(RUNTIME_LIBRARY) $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_glm52_resident_decode_stage_production_runner: tests/test_glm52_resident_decode_stage_production_runner.c inference/stage/runner.c modules/glm52_resident_decode_stage/include/sparkpipe/spark_glm52_resident_decode_stage_production_runner.h $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests -Imodules/glm52_resident_decode_stage/include $(CFLAGS) tests/test_glm52_resident_decode_stage_production_runner.c inference/stage/runner.c $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

glm52_quantized_readiness_test: build/test_glm52_ring_runtime build/test_glm52_stagepack build/test_model_description
	./build/test_glm52_ring_runtime
	./build/test_glm52_stagepack
	./build/test_model_description
	python3 tests/test_glm52_stage_pack.py
	python3 tests/test_glm52_w8lut_artifact_preflight.py
	python3 tests/test_glm52_quantized_cuda_contract.py
	python3 tests/test_glm52_b12x_resident_manifest.py
	python3 tests/test_release_assemble.py

test: $(TEST_BINARIES)
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

# One source tree, four capacity-ceiling modules (b8 chat through b1024, the
# planner maximum): the module Makefile's variants target compiles every
# bucket from the single SPARK_GLM52_BATCH_VARIANT_RULES template. Trim the
# set with GLM52_BATCH_VARIANT_BUCKETS="8 1024", never by editing a variant.
cuda_glm52_resident_decode_stage_variants:
	@if ! command -v $(NVCC) >/dev/null 2>&1; then \
		echo "cuda_glm52_resident_decode_stage_variants skipped: nvcc unavailable"; \
	else \
		$(MAKE) -C modules/glm52_resident_decode_stage variants NVCC=$(NVCC) CUDA_ARCH=sm_121a; \
	fi


glm52_w8lut_quality_reference:
	@test -d modules/glm52_w8lut_quality_weights || { echo "modules/glm52_w8lut_quality_weights is not in this tree; target unavailable" >&2; exit 2; }
	$(MAKE) -C modules/glm52_w8lut_quality_weights test_ref crosscheck

glm52_w8lut_quality_cuda_gate:
	@command -v $(NVCC) >/dev/null 2>&1 || \
		{ echo "glm52_w8lut_quality_cuda_gate requires nvcc" >&2; exit 2; }
	@test -d modules/glm52_w8lut_quality_weights || { echo "modules/glm52_w8lut_quality_weights is not in this tree; target unavailable" >&2; exit 2; }
	$(MAKE) -C modules/glm52_w8lut_quality_weights test_gpu NVCC=$(NVCC) CUDA_ARCH=sm_121a

glm52_w8lut_resident_pack:
	@test -n "$(W8LUT_MODEL_DIR)" || \
		{ echo "set W8LUT_MODEL_DIR to the BF16 GLM-5.2 model directory" >&2; exit 2; }
	@test -n "$(W8LUT_MOE_PACK_OUTPUT_DIR)" || \
		{ echo "set W8LUT_MOE_PACK_OUTPUT_DIR to a dedicated W8LUT pack directory" >&2; exit 2; }
	python3 tools/glm52_w8lut_resident_pack.py \
		--model-dir "$(W8LUT_MODEL_DIR)" \
		--output-dir "$(W8LUT_MOE_PACK_OUTPUT_DIR)" \
		--layers "$(W8LUT_MOE_PACK_LAYERS)" \
		--jobs "$(W8LUT_MOE_PACK_JOBS)" \
		--max-active "$(W8LUT_MOE_PACK_MAX_ACTIVE)"

cuda_glm52_resident_decode_stage_publish: $(B12X_ADAPTER_ARCHIVE) $(B12X_COMPILED_BACKEND_ARCHIVE) $(B12X_GENERATED_KERNEL_TABLE_ARCHIVE)
	@if ! command -v $(NVCC) >/dev/null 2>&1; then \
		echo "cuda_glm52_resident_decode_stage_publish skipped: nvcc unavailable"; \
	else \
		if [ -n "$(GLM52_REQUIRED_CUDA_LINK_ARGS)" ]; then \
			required_cuda_link_args='$(GLM52_REQUIRED_CUDA_LINK_ARGS)'; \
		else \
			required_cuda_link_args='$(GLM52_RING_NODE_CONTEXT_BUILDER_DEFAULT_LINK_ARGS)'; \
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
			GLM52_EXACT_RING_MODEL_QUANTIZATION='$(GLM52_EXACT_RING_MODEL_QUANTIZATION)' \
			GLM52_STAGE_PACK_DIR='$(GLM52_STAGE_PACK_DIR)' \
			GLM52_FP8_MOE_PACK_DIR='$(GLM52_FP8_MOE_PACK_DIR)' \
			GLM52_W8LUT_MOE_PACK_DIR='$(GLM52_W8LUT_MOE_PACK_DIR)' \
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
	@test -d modules/glm52_sm121_flashinfer_b12x_moe || { echo "modules/glm52_sm121_flashinfer_b12x_moe is not in this tree; target unavailable" >&2; exit 2; }
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
	@test -d modules/glm52_sm121_b12x_compiled_backend || { echo "modules/glm52_sm121_b12x_compiled_backend is not in this tree; target unavailable" >&2; exit 2; }
	$(MAKE) -C modules/glm52_sm121_b12x_compiled_backend archive NVCC=$(NVCC) CUDA_ARCH=sm_121a

$(B12X_GENERATED_KERNEL_TABLE_ARCHIVE):
ifeq ($(GLM52_MOE_BACKEND),nvfp4)
	@test -d modules/glm52_sm121_b12x_compiled_backend || { echo "modules/glm52_sm121_b12x_compiled_backend is not in this tree; target unavailable" >&2; exit 2; }
	$(MAKE) -C modules/glm52_sm121_b12x_compiled_backend generated_archive NVCC=$(NVCC) CUDA_ARCH=sm_121a GENERATED_DIRECTORY=$(abspath $(B12X_AOT_OUTPUT_DIR)/generated)
else
	@test -d modules/glm52_sm121_b12x_compiled_backend || { echo "modules/glm52_sm121_b12x_compiled_backend is not in this tree; target unavailable" >&2; exit 2; }
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

	@test -n "$(GLM52_PIPELINE_INPUT_HIDDEN_BF16)" || \
		{ echo "set GLM52_PIPELINE_INPUT_HIDDEN_BF16 to a one-vector or B-vector hidden BF16 file" >&2; exit 2; }
	@test -s "$(GLM52_PIPELINE_INPUT_HIDDEN_BF16)" || \
		{ echo "missing GLM52_PIPELINE_INPUT_HIDDEN_BF16: $(GLM52_PIPELINE_INPUT_HIDDEN_BF16)" >&2; exit 2; }
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
		$(MAKE) -C modules/glm52_resident_decode_stage package NVCC=$(NVCC) CUDA_HOME='$(CUDA_HOME)' CUDA_ARCH=sm_121a MAX_STAGE_MICROSECONDS=$(MAX_STAGE_MICROSECONDS) REQUIRED_CUDA_CC_ARGS='$(REQUIRED_CUDA_CC_ARGS)' GLM52_REQUIRED_CUDA_LINK_ARGS="$$required_cuda_link_args" B12X_MOE_PACK_DIR='$(abspath $(B12X_MOE_PACK_OUTPUT_DIR))' B12X_MOE_PACK_LAYERS='$(B12X_MOE_PACK_LAYERS)' B12X_MOE_PACK_REQUIRE_REUSE='$(B12X_MOE_PACK_REQUIRE_REUSE)' B12X_MOE_PACK_VERIFY_REUSED_SHA256='$(B12X_MOE_PACK_VERIFY_REUSED_SHA256)' B12X_MOE_PACK_JOBS='$(B12X_MOE_PACK_JOBS)' B12X_MOE_PACK_PACKAGE_MODE='$(B12X_MOE_PACK_PACKAGE_MODE)' GLM52_MODEL_DIR='$(GLM52_MODEL_DIR)' GLM52_VALIDATION_MODE='$(GLM52_VALIDATION_MODE)' GLM52_VALIDATION_ACTIVE_SEQUENCE_COUNT='$(GLM52_VALIDATION_ACTIVE_SEQUENCE_COUNT)' GLM52_VALIDATION_FIRST_ROUTED_LAYER_INDEX='$(GLM52_VALIDATION_FIRST_ROUTED_LAYER_INDEX)' GLM52_VALIDATION_ROUTED_CHAIN_LAYER_COUNT='$(GLM52_VALIDATION_ROUTED_CHAIN_LAYER_COUNT)' GLM52_PIPELINE_INPUT_HIDDEN_BF16='$(GLM52_PIPELINE_INPUT_HIDDEN_BF16)' GLM52_PIPELINE_OUTPUT_HIDDEN_BF16='$(GLM52_PIPELINE_OUTPUT_HIDDEN_BF16)' GLM52_ENABLE_CUDA_GRAPH_REPLAY='$(GLM52_ENABLE_CUDA_GRAPH_REPLAY)' GLM52_EXACT_RING_MODEL_QUANTIZATION='$(GLM52_EXACT_RING_MODEL_QUANTIZATION)' GLM52_STAGE_PACK_DIR='$(GLM52_STAGE_PACK_DIR)' GLM52_FP8_MOE_PACK_DIR='$(GLM52_FP8_MOE_PACK_DIR)' GLM52_W8LUT_MOE_PACK_DIR='$(GLM52_W8LUT_MOE_PACK_DIR)' B12X_PACK_PYTHON="$$SPARKPIPE_B12X_AOT_PYTHON" AOT_MANIFEST='$(abspath $(B12X_AOT_OUTPUT_DIR))/generated/aot_manifest.json'
else
	@if [ -n "$(GLM52_REQUIRED_CUDA_LINK_ARGS)" ]; then \
		required_cuda_link_args='$(GLM52_REQUIRED_CUDA_LINK_ARGS)'; \
	else \
		required_cuda_link_args='$(B12X_ADAPTER_ARCHIVE) $(B12X_COMPILED_BACKEND_ARCHIVE) $(B12X_GENERATED_KERNEL_TABLE_ARCHIVE)'; \
	fi; \
	$(MAKE) -C modules/glm52_resident_decode_stage package NVCC=$(NVCC) CUDA_HOME='$(CUDA_HOME)' CUDA_ARCH=sm_121a MAX_STAGE_MICROSECONDS=$(MAX_STAGE_MICROSECONDS) REQUIRED_CUDA_CC_ARGS='$(REQUIRED_CUDA_CC_ARGS)' GLM52_REQUIRED_CUDA_LINK_ARGS="$$required_cuda_link_args" B12X_MOE_PACK_DIR='$(abspath $(B12X_MOE_PACK_OUTPUT_DIR))' B12X_MOE_PACK_LAYERS='$(B12X_MOE_PACK_LAYERS)' B12X_MOE_PACK_REQUIRE_REUSE='$(B12X_MOE_PACK_REQUIRE_REUSE)' B12X_MOE_PACK_VERIFY_REUSED_SHA256='$(B12X_MOE_PACK_VERIFY_REUSED_SHA256)' B12X_MOE_PACK_JOBS='$(B12X_MOE_PACK_JOBS)' B12X_MOE_PACK_PACKAGE_MODE='$(B12X_MOE_PACK_PACKAGE_MODE)' GLM52_MODEL_DIR='$(GLM52_MODEL_DIR)' GLM52_VALIDATION_MODE='$(GLM52_VALIDATION_MODE)' GLM52_VALIDATION_ACTIVE_SEQUENCE_COUNT='$(GLM52_VALIDATION_ACTIVE_SEQUENCE_COUNT)' GLM52_VALIDATION_FIRST_ROUTED_LAYER_INDEX='$(GLM52_VALIDATION_FIRST_ROUTED_LAYER_INDEX)' GLM52_VALIDATION_ROUTED_CHAIN_LAYER_COUNT='$(GLM52_VALIDATION_ROUTED_CHAIN_LAYER_COUNT)' GLM52_PIPELINE_INPUT_HIDDEN_BF16='$(GLM52_PIPELINE_INPUT_HIDDEN_BF16)' GLM52_PIPELINE_OUTPUT_HIDDEN_BF16='$(GLM52_PIPELINE_OUTPUT_HIDDEN_BF16)' GLM52_ENABLE_CUDA_GRAPH_REPLAY='$(GLM52_ENABLE_CUDA_GRAPH_REPLAY)' GLM52_EXACT_RING_MODEL_QUANTIZATION='$(GLM52_EXACT_RING_MODEL_QUANTIZATION)' GLM52_STAGE_PACK_DIR='$(GLM52_STAGE_PACK_DIR)' GLM52_FP8_MOE_PACK_DIR='$(GLM52_FP8_MOE_PACK_DIR)' GLM52_W8LUT_MOE_PACK_DIR='$(GLM52_W8LUT_MOE_PACK_DIR)' GLM52_REQUIRE_B12X_RESIDENT_PACK=0 B12X_PACK_PYTHON="$(B12X_PACK_PYTHON)" AOT_MANIFEST=''
endif

tree_summary:
	@printf "core_public_headers="; find include/sparkpipe -type f | wc -l
	@printf "core_support_sources="; printf '%s\n' $(CORE_SOURCES) | wc -l
	@printf "compiler_sources="; printf '%s\n' $(COMPILER_SOURCES) | wc -l
	@printf "runtime_sources="; printf '%s\n' $(RUNTIME_SOURCES) | wc -l
	@printf "model_common_sources="; printf '%s\n' $(MODEL_COMMON_SOURCES) | wc -l
	@printf "glm52_host_sources="; printf '%s\n' $(GLM52_HOST_SOURCES) | wc -l
	@printf "qwen36_host_sources="; printf '%s\n' $(QWEN36_HOST_SOURCES) | wc -l
	@printf "deployment_sources="; printf '%s\n' $(DEPLOYMENT_SOURCES) | wc -l
	@printf "test_executables="; printf '%s\n' $(TEST_NAMES) | wc -l

clean:
	rm -rf build

-include $(ALL_HOST_OBJECTS:.o=.d) $(TEST_SUPPORT_OBJECT:.o=.d) \
    $(TEST_MODULE_DEPENDENCIES) \
    $(GLM52_RESIDENT_DECODE_STAGE_TEST_DEPENDENCIES)
