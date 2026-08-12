CC ?= cc
CXX ?= c++
AR ?= ar
NVCC ?= nvcc
CUDA_HOME ?= /usr/local/cuda
CFLAGS ?= -std=c11 -Wall -Wextra -Werror -O3 -g -pthread
CFLAGS += -D_GNU_SOURCE
CXXFLAGS ?= -std=c++20 -Wall -Wextra -Werror -O3 -g -pthread
CORE_INCLUDE_FLAGS := -I. -Iinclude -Isrc
# stage_module_common.c reaches <cuda_runtime.h> through spark_stage_module_common.h.
# Nothing in this group supplied that path, so the archive could not have built;
# the tests hid it behind -Itests/cuda_stub.
CUDA_RUNTIME_HEADER := $(wildcard $(CUDA_HOME)/include/cuda_runtime.h)
ifeq ($(CUDA_RUNTIME_HEADER),)
MODEL_COMMON_INCLUDE_FLAGS := $(CORE_INCLUDE_FLAGS) -Itests/cuda_stub
SPARKPIPE_HOST_CUDA_STUB_SOURCE := tests/cuda_stub/cuda_runtime_stub.c
SPARKPIPE_TP_DEVICE_TEST_CUDA_STUB_SOURCE :=
SPARKPIPE_CUDA_RUNTIME_LINK :=
else
MODEL_COMMON_INCLUDE_FLAGS := $(CORE_INCLUDE_FLAGS) -I$(CUDA_HOME)/include
SPARKPIPE_HOST_CUDA_STUB_SOURCE :=
SPARKPIPE_TP_DEVICE_TEST_CUDA_STUB_SOURCE := tests/cuda_stub/cuda_runtime_stub.c
SPARKPIPE_CUDA_RUNTIME_LINK := -L$(CUDA_HOME)/lib64 -lcudart
endif
GLM52_INCLUDE_FLAGS := $(MODEL_COMMON_INCLUDE_FLAGS) -Imodel-families/glm52/include
QWEN36_INCLUDE_FLAGS := $(MODEL_COMMON_INCLUDE_FLAGS) -Imodel-families/qwen36/include
DSV4_DEFAULT_BATCH_FLAGS := -DSPARK_BATCH_BUCKET=1024u
DSV4_INCLUDE_FLAGS := $(MODEL_COMMON_INCLUDE_FLAGS) -Imodel-families/dsv4/include $(DSV4_DEFAULT_BATCH_FLAGS)
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
CPPFLAGS ?= $(CORE_INCLUDE_FLAGS) $(MODEL_FAMILY_INCLUDE_FLAGS) -Ideployment/include -Ideployment/src $(DSV4_DEFAULT_BATCH_FLAGS)
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
MOONCAKE_ROOT ?=
MOONCAKE_LIB ?= $(MOONCAKE_ROOT)/build/mooncake-store/src
MOONCAKE_DEP_INCLUDE ?= $(MOONCAKE_ROOT)/local/include
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

DSV4_SERVING_ADAPTER := build/libdsv4_serving_adapter.$(SHARED_LIBRARY_EXT)
DSV4_TP16_SERVING_ADAPTER := build/libdsv4_tp16_serving_adapter.$(SHARED_LIBRARY_EXT)
DSV4_TP4_SERVING_ADAPTER := build/libdsv4_tp4_serving_adapter.$(SHARED_LIBRARY_EXT)
DSV4_TP4_PP4_SERVING_ADAPTER := build/libdsv4_tp4_pp4_serving_adapter.$(SHARED_LIBRARY_EXT)
DSV4_TP4_PP4_B1_SERVING_ADAPTER := build/libdsv4_tp4_pp4_b1_serving_adapter.$(SHARED_LIBRARY_EXT)
DSV4_SERVING_TOPOLOGY_FLAGS := -DSPARK_DSV4_SERVING_TOPOLOGY=13
DSV4_TP16_SERVING_TOPOLOGY_FLAGS := -DSPARK_DSV4_SERVING_TOPOLOGY=16
DSV4_TP4_SERVING_TOPOLOGY_FLAGS := -DSPARK_DSV4_SERVING_TOPOLOGY=4
DSV4_TP4_PP4_SERVING_TOPOLOGY_FLAGS := -DSPARK_DSV4_SERVING_TOPOLOGY=404
DSV4_TP4_PP4_B1_SERVING_TOPOLOGY_FLAGS := $(DSV4_TP4_PP4_SERVING_TOPOLOGY_FLAGS) -USPARK_BATCH_BUCKET -DSPARK_BATCH_BUCKET=1u
QWEN36_SERVING_ADAPTER := build/libqwen36_serving_adapter.$(SHARED_LIBRARY_EXT)
QWEN36_MODEL_DESCRIPTION := examples/model_descriptions/qwen36_resident_decode_stage_firmware.json
QWEN36_MODEL_REVISION ?= bf16-h5120-l64-gdn48-full16-v248320-mtp1-v1
QWEN36_CONTRACT_SHA256 ?= $(shell if command -v sha256sum >/dev/null 2>&1; then sha256sum "$(QWEN36_MODEL_DESCRIPTION)"; else shasum -a 256 "$(QWEN36_MODEL_DESCRIPTION)"; fi | awk '{print $$1}')
QWEN36_SERVING_ADAPTER_FLAGS := -D_POSIX_C_SOURCE=200809L -DQWEN36_MODEL_REVISION=\"$(QWEN36_MODEL_REVISION)\" -DQWEN36_CONTRACT_SHA256=\"$(QWEN36_CONTRACT_SHA256)\"

TOOL_NAMES := \
    sparkpipe_module_publish \
    sparkpipe_model_compile \
    sparkpipe_driver_inspect \
    sparkpipe_model_residentd \
    sparkpipe_model_batch \
    sparkpipe_glm52_tokenize \
    sparkpipe_tokenize_prompt \
    sparkpipe_tokenizer_benchmark \
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
	test_status \
    test_gemm_descriptor_cache \
    test_launch \
    test_arena \
    test_work_transaction \
    test_runtime_completion \
    test_model_runtime \
    test_model_serving_adapter \
	test_model_resident_deployment \
    test_model_resident_ipc \
	test_model_resident_deadline \
    test_model_pipeline_client \
    test_pipeline_runtime \
    test_dsv4_serving_adapter \
    test_dsv4_tp16_serving_adapter \
	test_dsv4_tp4_pp4_serving_adapter \
    test_qwen36_serving_adapter \
    test_model_resident_end_to_end \
    test_distributed_work \
	    test_json \
	    test_hidden_transport \
	    test_hidden_transport_rdma_control \
    test_fabric_topology \
    test_memlink \
    test_release \
    test_kv_store \
    test_kv_cache \
	test_k3_kv_cache \
    test_nvme_tier \
    test_kv_mooncake \
    test_qwen36_work_control \
    test_dsv4_cache_plan \
	test_dsv4_parallel_shape \
    test_glm52_dspark \
    test_glm52_mtp_tree \
    test_tp_collective \
    test_tp_device_collective \
    test_tp_device_collective_nccl \
    test_glm52_stagepack \
    test_tokenizer \
    test_model_description \
    test_stage_module_common \
    test_module_library \
    test_driver_compiler \
    test_orchestrator \
	test_dsv4_lane_continuity \
	test_dsv4_tp_graph_contract \
	test_dsv4_pool_layout \
	test_dsv4_paged_cache \
    test_dsv4_stage_runner \
    test_tensor_map_geometry \
    test_weight_codec \
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
	tests/test_dsv4_ga_reference_fixture.py \
	tests/test_dsv4_native_compute_source.py \
	tests/test_dsv4_module_host_syntax.py \
	tests/test_dsv4_stage_source.py \
	tests/test_dsv4_stagepack.py \
	tests/test_dsv4_tp_async_source.py \
	tests/test_dsv4_tp_graph_islands_source.py \
	tests/test_dsv4_tp4_decode_roofline.py \
	tests/test_dsv4_tp4_pp4_perf_estimate.py \
	tests/test_dsv4_tp4_pp4_stagepack.py \
	tests/test_expert_grouping.py \
	tests/test_production_selection_contract.py \
	tests/test_gemm_k_alignment.py \
	tests/test_glm52_dspark_manifest.py \
	tests/test_glm52_dspark_trace_quality.py \
	tests/test_glm52_module_contract.py \
	tests/test_glm52_layer_host.py \
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
		tests/test_hidden_transport_rdma_source.py \
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
	tests/test_k3_slice_host.py \
	tests/test_kda_bf16_state.py \
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
	tests/test_generate_model_resident_deployment.py \
	tests/test_model_serving_architecture.py \
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
	tests/test_qwen36_stagepack.py \
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
TEST_TP_DEVICE_COLLECTIVE_MODULE := \
    build/test_modules/libtp_device_collective_module.$(SHARED_LIBRARY_EXT)
TEST_TP_DEVICE_COLLECTIVE_NCCL_MODULE := \
    build/test_modules/libtp_device_collective_nccl_module.$(SHARED_LIBRARY_EXT)
TEST_MODEL_SERVING_ADAPTER_MODULE := \
    build/test_modules/libmodel_serving_adapter_module.$(SHARED_LIBRARY_EXT)
TEST_DSV4_SERVING_DRIVER_MODULE := \
    build/test_modules/libdsv4_serving_driver_module.$(SHARED_LIBRARY_EXT)
TEST_DSV4_TP16_SERVING_DRIVER_MODULE := \
    build/test_modules/libdsv4_tp16_serving_driver_module.$(SHARED_LIBRARY_EXT)
TEST_DSV4_TP4_PP4_SERVING_DRIVER_MODULE := \
    build/test_modules/libdsv4_tp4_pp4_serving_driver_module.$(SHARED_LIBRARY_EXT)
TEST_QWEN36_SERVING_DRIVER_MODULE := \
    build/test_modules/libqwen36_serving_driver_module.$(SHARED_LIBRARY_EXT)
TEST_MODEL_RESIDENT_TRANSPORT_MODULE := \
    build/test_modules/libmodel_resident_transport_module.$(SHARED_LIBRARY_EXT)
TEST_VALIDATOR := build/test_module_validator
TEST_VALIDATOR_CHANGED := build/test_module_validator_identity_changed
.PHONY: all clean test tools hardware_tools hardware_cuda_tools hardware_handoff runtime_completion_tests demo FORCE \
    cuda_glm52_resident_decode_stage_variants \
    cuda_dsv4_resident_decode_stage_variants \
    glm52_resident_decode_stage_contract \
    glm52_resident_decode_stage_archive \
    glm52_resident_decode_stage_publish \
    glm52_serving_adapter \
    hidden_transport_spark_host_rdma_verbs \
    hidden_transport_spark_gpudirect_rdma_verbs \
    kv_mooncake \
    tree_summary \
    architecture_audit \
    model_driver_contracts

# Model CUDA modules are immutable artifacts selected by an explicit model
# package. Host builds never guess a codec or silently skip a CUDA artifact.
all: $(LIBRARIES) tools $(DSV4_SERVING_ADAPTER) $(DSV4_TP4_PP4_SERVING_ADAPTER) $(DSV4_TP4_PP4_B1_SERVING_ADAPTER) $(QWEN36_SERVING_ADAPTER)

tools: $(TOOL_BINARIES) $(DSV4_SERVING_ADAPTER) $(DSV4_TP4_PP4_SERVING_ADAPTER) $(QWEN36_SERVING_ADAPTER)

.PHONY: core model_common deployment audit-boundaries architecture_audit model_driver_contracts non_glm_model_driver_contracts

core: $(CORE_LIBRARY) $(COMPILER_LIBRARY) $(RUNTIME_LIBRARY)

model_common: $(MODEL_COMMON_LIBRARY)

deployment: $(DEPLOYMENT_LIBRARY)

audit-boundaries: core
	python3 tools/audit_core_boundaries.py --repository . --core-archive $(CORE_LIBRARY) --compiler-archive $(COMPILER_LIBRARY) --runtime-archive $(RUNTIME_LIBRARY)

architecture_audit: audit-boundaries

non_glm_model_driver_contracts:
	$(MAKE) -C modules/dsv4_resident_decode_stage contract

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
    build/test_model_description

QWEN36_LINK_TARGETS := build/test_qwen36_work_control
DSV4_LINK_TARGETS := build/test_dsv4_cache_plan build/sparkpipe_dsv4_cache_plan_report


DEPLOYMENT_LINK_TARGETS := build/sparkpipe_release_manager build/test_release

$(MODEL_COMMON_LINK_TARGETS): COMMON_LIBRARY = $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
$(MODEL_COMMON_LINK_TARGETS): $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
$(GLM52_LINK_TARGETS): COMMON_LIBRARY = $(GLM52_HOST_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
$(GLM52_LINK_TARGETS): $(GLM52_HOST_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
$(DSV4_LINK_TARGETS): $(DSV4_HOST_LIBRARY) $(CORE_LIBRARY)

$(QWEN36_LINK_TARGETS): COMMON_LIBRARY = $(QWEN36_HOST_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
$(QWEN36_LINK_TARGETS): $(QWEN36_HOST_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
$(DEPLOYMENT_LINK_TARGETS): COMMON_LIBRARY = $(DEPLOYMENT_LIBRARY) $(CORE_LIBRARY)
$(DEPLOYMENT_LINK_TARGETS): $(DEPLOYMENT_LIBRARY) $(CORE_LIBRARY)

build:
	mkdir -p build

build/test_modules:
	mkdir -p build/test_modules

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

build/sparkpipe_dsv4_driver_cuda_smoke: tools/sparkpipe_dsv4_driver_cuda_smoke.c modules/dsv4_resident_decode_stage/source/spark_dsv4_stage_runner.c $(RUNTIME_LIBRARY) $(COMMON_LIBRARY) $(MODEL_COMMON_LIBRARY)
	$(CC) $(DSV4_INCLUDE_FLAGS) -Imodules/dsv4_resident_decode_stage/include $(CFLAGS) -I$(CUDA_HOME)/include $< modules/dsv4_resident_decode_stage/source/spark_dsv4_stage_runner.c $(RUNTIME_LIBRARY) $(COMMON_LIBRARY) $(MODEL_COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -L$(CUDA_HOME)/lib64 -lcudart -lstdc++ -lm -o $@

build/sparkpipe_dsv4_cache_plan_report: tests/studies/sparkpipe_dsv4_cache_plan_report.c $(DSV4_HOST_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(DSV4_INCLUDE_FLAGS) $(CFLAGS) $< $(DSV4_HOST_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) $(SPARKPIPE_CUDA_RUNTIME_LINK) -o $@

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
	$(CC) $(DSV4_INCLUDE_FLAGS) $(CFLAGS) $< $(DSV4_HOST_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) $(SPARKPIPE_CUDA_RUNTIME_LINK) -o $@

build/test_dsv4_parallel_shape: tests/test_dsv4_parallel_shape.c $(DSV4_HOST_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(DSV4_INCLUDE_FLAGS) $(CFLAGS) $< $(DSV4_HOST_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_kv_cache: tests/test_kv_cache.c $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(MODEL_COMMON_INCLUDE_FLAGS) $(CFLAGS) $< $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) $(SPARKPIPE_CUDA_RUNTIME_LINK) -o $@

build/test_k3_kv_cache: tests/test_k3_kv_cache.c $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(MODEL_COMMON_INCLUDE_FLAGS) -Imodel-families/k3/include $(CFLAGS) $< $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/sparkpipe_glm52_batchplane_model: tests/studies/sparkpipe_glm52_batchplane_model.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LDFLAGS) $(LDLIBS) -lm -o $@

build/sparkpipe_glm52_tokenize: tools/sparkpipe_glm52_tokenize.c $(GLM52_HOST_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(GLM52_INCLUDE_FLAGS) $(CFLAGS) $< $(GLM52_HOST_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/sparkpipe_tokenize_prompt: tools/sparkpipe_tokenize_prompt.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/sparkpipe_tokenizer_benchmark: tools/sparkpipe_tokenizer_benchmark.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/sparkpipe_memlink: node/memlink_tool.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/sparkpipe_prevcp: node/memlink_tool.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DSPARK_MEMLINK_FIXED_COMMAND=\"prevcp\" $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/sparkpipe_nextcp: node/memlink_tool.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DSPARK_MEMLINK_FIXED_COMMAND=\"nextcp\" $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/sparkpipe_release_manager: deployment/tools/sparkpipe_release_manager.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/sparkpipe_model_residentd: node/model_residentd.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(MODEL_COMMON_INCLUDE_FLAGS) $(CFLAGS) node/model_residentd.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) $(SPARKPIPE_CUDA_RUNTIME_LINK) -o $@

build/sparkpipe_model_batch: node/model_batch.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(MODEL_COMMON_INCLUDE_FLAGS) $(CFLAGS) node/model_batch.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

$(DSV4_SERVING_ADAPTER): modules/dsv4_resident_decode_stage/source/spark_dsv4_serving_adapter.c modules/dsv4_resident_decode_stage/source/spark_dsv4_stage_runner.c modules/dsv4_resident_decode_stage/include/sparkpipe/spark_dsv4_serving_adapter.h modules/dsv4_resident_decode_stage/include/sparkpipe/spark_dsv4_resident_decode_stage_firmware.h $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(DSV4_HOST_LIBRARY)
	$(CC) $(CPPFLAGS) $(DSV4_SERVING_TOPOLOGY_FLAGS) -Imodules/dsv4_resident_decode_stage/include $(CFLAGS) -fPIC $(SHARED_LIBRARY_FLAGS) modules/dsv4_resident_decode_stage/source/spark_dsv4_serving_adapter.c modules/dsv4_resident_decode_stage/source/spark_dsv4_stage_runner.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(DSV4_HOST_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

$(DSV4_TP16_SERVING_ADAPTER): modules/dsv4_resident_decode_stage/source/spark_dsv4_serving_adapter.c modules/dsv4_resident_decode_stage/source/spark_dsv4_stage_runner.c modules/dsv4_resident_decode_stage/include/sparkpipe/spark_dsv4_serving_adapter.h modules/dsv4_resident_decode_stage/include/sparkpipe/spark_dsv4_resident_decode_stage_firmware.h $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(DSV4_HOST_LIBRARY)
	$(CC) $(CPPFLAGS) $(DSV4_TP16_SERVING_TOPOLOGY_FLAGS) -Imodules/dsv4_resident_decode_stage/include $(CFLAGS) -fPIC $(SHARED_LIBRARY_FLAGS) modules/dsv4_resident_decode_stage/source/spark_dsv4_serving_adapter.c modules/dsv4_resident_decode_stage/source/spark_dsv4_stage_runner.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(DSV4_HOST_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

$(DSV4_TP4_SERVING_ADAPTER): modules/dsv4_resident_decode_stage/source/spark_dsv4_serving_adapter.c modules/dsv4_resident_decode_stage/source/spark_dsv4_stage_runner.c modules/dsv4_resident_decode_stage/include/sparkpipe/spark_dsv4_serving_adapter.h modules/dsv4_resident_decode_stage/include/sparkpipe/spark_dsv4_resident_decode_stage_firmware.h $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(DSV4_HOST_LIBRARY)
	$(CC) $(CPPFLAGS) $(DSV4_TP4_SERVING_TOPOLOGY_FLAGS) -Imodules/dsv4_resident_decode_stage/include $(CFLAGS) -fPIC $(SHARED_LIBRARY_FLAGS) modules/dsv4_resident_decode_stage/source/spark_dsv4_serving_adapter.c modules/dsv4_resident_decode_stage/source/spark_dsv4_stage_runner.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(DSV4_HOST_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

$(DSV4_TP4_PP4_SERVING_ADAPTER): modules/dsv4_resident_decode_stage/source/spark_dsv4_serving_adapter.c modules/dsv4_resident_decode_stage/source/spark_dsv4_stage_runner.c modules/dsv4_resident_decode_stage/include/sparkpipe/spark_dsv4_serving_adapter.h modules/dsv4_resident_decode_stage/include/sparkpipe/spark_dsv4_resident_decode_stage_firmware.h $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(DSV4_HOST_LIBRARY)
	$(CC) $(CPPFLAGS) $(DSV4_TP4_PP4_SERVING_TOPOLOGY_FLAGS) -Imodules/dsv4_resident_decode_stage/include $(CFLAGS) -fPIC $(SHARED_LIBRARY_FLAGS) modules/dsv4_resident_decode_stage/source/spark_dsv4_serving_adapter.c modules/dsv4_resident_decode_stage/source/spark_dsv4_stage_runner.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(DSV4_HOST_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

$(DSV4_TP4_PP4_B1_SERVING_ADAPTER): modules/dsv4_resident_decode_stage/source/spark_dsv4_serving_adapter.c modules/dsv4_resident_decode_stage/source/spark_dsv4_stage_runner.c modules/dsv4_resident_decode_stage/include/sparkpipe/spark_dsv4_serving_adapter.h modules/dsv4_resident_decode_stage/include/sparkpipe/spark_dsv4_resident_decode_stage_firmware.h $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(DSV4_HOST_LIBRARY)
	$(CC) $(CPPFLAGS) $(DSV4_TP4_PP4_B1_SERVING_TOPOLOGY_FLAGS) -Imodules/dsv4_resident_decode_stage/include $(CFLAGS) -fPIC $(SHARED_LIBRARY_FLAGS) modules/dsv4_resident_decode_stage/source/spark_dsv4_serving_adapter.c modules/dsv4_resident_decode_stage/source/spark_dsv4_stage_runner.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(DSV4_HOST_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

$(QWEN36_SERVING_ADAPTER): modules/qwen36_resident_decode_stage/source/spark_qwen36_serving_adapter.c modules/qwen36_resident_decode_stage/include/sparkpipe/spark_qwen36_serving_adapter.h modules/qwen36_resident_decode_stage/include/sparkpipe/spark_qwen36_resident_decode_stage_firmware.h $(QWEN36_MODEL_DESCRIPTION) $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(CPPFLAGS) $(QWEN36_INCLUDE_FLAGS) -Imodules/qwen36_resident_decode_stage/include $(CFLAGS) $(QWEN36_SERVING_ADAPTER_FLAGS) -fPIC $(SHARED_LIBRARY_FLAGS) modules/qwen36_resident_decode_stage/source/spark_qwen36_serving_adapter.c $(SPARKPIPE_HOST_CUDA_STUB_SOURCE) $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) $(SPARKPIPE_CUDA_RUNTIME_LINK) -o $@

$(HIDDEN_TRANSPORT_SPARK_HOST_RDMA): ring/transport/rdma.cu ring/transport/rdma_control.c include/sparkpipe/spark_hidden_transport.h include/sparkpipe/spark_hidden_transport_rdma_control.h include/sparkpipe/spark_memlink.h $(COMMON_LIBRARY)
	@if ! command -v $(NVCC) >/dev/null 2>&1; then \
		echo "hidden_transport_spark_host_rdma_verbs failed: nvcc unavailable" >&2; exit 1; \
	elif [ ! -f "$(CUDA_HOME)/include/cuda_runtime_api.h" ]; then \
		echo "hidden_transport_spark_host_rdma_verbs failed: CUDA headers unavailable" >&2; exit 1; \
	elif [ ! -f "/usr/include/infiniband/verbs.h" ]; then \
		echo "hidden_transport_spark_host_rdma_verbs failed: libibverbs headers unavailable" >&2; exit 1; \
	else \
		$(NVCC) $(NVCCFLAGS) -DSPARK_HIDDEN_SPARK_RDMA_DEVICE_DIRECT=0 $(SHARED_LIBRARY_FLAGS) -Xcompiler -fPIC -Xcompiler -pthread $(MODEL_COMMON_INCLUDE_FLAGS) ring/transport/rdma.cu ring/transport/rdma_control.c $(COMMON_LIBRARY) $(LDFLAGS) -L$(CUDA_HOME)/lib64 -lcudart -libverbs -ldl -lpthread -o $@; \
	fi

hidden_transport_spark_host_rdma_verbs: $(HIDDEN_TRANSPORT_SPARK_HOST_RDMA)

$(HIDDEN_TRANSPORT_SPARK_GPUDIRECT_RDMA): ring/transport/rdma.cu ring/transport/rdma_control.c include/sparkpipe/spark_hidden_transport.h include/sparkpipe/spark_hidden_transport_rdma_control.h include/sparkpipe/spark_memlink.h $(COMMON_LIBRARY)
	@if ! command -v $(NVCC) >/dev/null 2>&1; then \
		echo "hidden_transport_spark_gpudirect_rdma_verbs failed: nvcc unavailable" >&2; exit 1; \
	elif [ ! -f "$(CUDA_HOME)/include/cuda_runtime_api.h" ]; then \
		echo "hidden_transport_spark_gpudirect_rdma_verbs failed: CUDA headers unavailable" >&2; exit 1; \
	elif [ ! -f "/usr/include/infiniband/verbs.h" ]; then \
		echo "hidden_transport_spark_gpudirect_rdma_verbs failed: libibverbs headers unavailable" >&2; exit 1; \
	else \
		$(NVCC) $(NVCCFLAGS) -DSPARK_HIDDEN_SPARK_RDMA_DEVICE_DIRECT=1 $(SHARED_LIBRARY_FLAGS) -Xcompiler -fPIC -Xcompiler -pthread $(MODEL_COMMON_INCLUDE_FLAGS) ring/transport/rdma.cu ring/transport/rdma_control.c $(COMMON_LIBRARY) $(LDFLAGS) -L$(CUDA_HOME)/lib64 -lcudart -libverbs -ldl -lpthread -o $@; \
	fi

hidden_transport_spark_gpudirect_rdma_verbs: $(HIDDEN_TRANSPORT_SPARK_GPUDIRECT_RDMA)

FORCE:

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

$(TEST_TP_DEVICE_COLLECTIVE_MODULE): tests/fixtures/tp_device_collective_module.c | build/test_modules
	$(CC) $(CPPFLAGS) $(CFLAGS) -fPIC $(SHARED_LIBRARY_FLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

$(TEST_TP_DEVICE_COLLECTIVE_NCCL_MODULE): tests/fixtures/tp_device_collective_nccl_module.c | build/test_modules
	$(CC) $(CPPFLAGS) $(CFLAGS) -fPIC $(SHARED_LIBRARY_FLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

$(TEST_MODEL_SERVING_ADAPTER_MODULE): tests/fixtures/model_serving_adapter_module.c include/sparkpipe/spark_model_serving_adapter.h $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) | build/test_modules
	$(CC) $(CPPFLAGS) $(CFLAGS) -fPIC $(SHARED_LIBRARY_FLAGS) $< $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

$(TEST_DSV4_SERVING_DRIVER_MODULE): tests/fixtures/dsv4_serving_adapter_driver.c modules/dsv4_resident_decode_stage/include/sparkpipe/spark_dsv4_resident_decode_stage_firmware.h include/sparkpipe/spark_model_driver.h include/sparkpipe/spark_model_driver_support.h | build/test_modules
	$(CC) $(CPPFLAGS) -Imodules/dsv4_resident_decode_stage/include $(CFLAGS) -fPIC $(SHARED_LIBRARY_FLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

$(TEST_DSV4_TP16_SERVING_DRIVER_MODULE): tests/fixtures/dsv4_serving_adapter_driver.c modules/dsv4_resident_decode_stage/include/sparkpipe/spark_dsv4_resident_decode_stage_firmware.h include/sparkpipe/spark_model_driver.h include/sparkpipe/spark_model_driver_support.h | build/test_modules
	$(CC) $(CPPFLAGS) -DTEST_DSV4_SERVING_TP16=1 -Imodules/dsv4_resident_decode_stage/include $(CFLAGS) -fPIC $(SHARED_LIBRARY_FLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

$(TEST_DSV4_TP4_PP4_SERVING_DRIVER_MODULE): tests/fixtures/dsv4_serving_adapter_driver.c modules/dsv4_resident_decode_stage/include/sparkpipe/spark_dsv4_resident_decode_stage_firmware.h include/sparkpipe/spark_model_driver.h include/sparkpipe/spark_model_driver_support.h | build/test_modules
	$(CC) $(CPPFLAGS) -DTEST_DSV4_SERVING_HYBRID=1 -Imodules/dsv4_resident_decode_stage/include $(CFLAGS) -fPIC $(SHARED_LIBRARY_FLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

$(TEST_QWEN36_SERVING_DRIVER_MODULE): tests/fixtures/qwen36_serving_adapter_driver.c modules/qwen36_resident_decode_stage/include/sparkpipe/spark_qwen36_resident_decode_stage_firmware.h include/sparkpipe/spark_model_driver.h include/sparkpipe/spark_model_driver_support.h $(QWEN36_MODEL_DESCRIPTION) | build/test_modules
	$(CC) $(CPPFLAGS) $(QWEN36_INCLUDE_FLAGS) -Imodules/qwen36_resident_decode_stage/include $(CFLAGS) $(QWEN36_SERVING_ADAPTER_FLAGS) -fPIC $(SHARED_LIBRARY_FLAGS) $< $(SPARKPIPE_HOST_CUDA_STUB_SOURCE) $(LDFLAGS) $(LDLIBS) $(SPARKPIPE_CUDA_RUNTIME_LINK) -o $@

$(TEST_MODEL_RESIDENT_TRANSPORT_MODULE): tests/fixtures/model_resident_transport_module.c include/sparkpipe/spark_hidden_transport.h | build/test_modules
	$(CC) $(CPPFLAGS) $(CFLAGS) -fPIC $(SHARED_LIBRARY_FLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

$(TEST_VALIDATOR): tests/fixtures/module_validator.c | build
	$(CC) $(CFLAGS) $< -o $@

$(TEST_VALIDATOR_CHANGED): tests/fixtures/module_validator.c | build
	$(CC) $(CFLAGS) -DSPARK_TEST_MODULE_VALIDATOR_IDENTITY=1u $< -o $@


build/test_arena: tests/test_arena.c runtime/arena.h
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_arena.c $(LDFLAGS) $(LDLIBS) -o $@

build/test_status: tests/test_status.c $(CORE_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_status.c $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_work_transaction: tests/test_work_transaction.c $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_work_transaction.c $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_runtime_completion: tests/test_runtime_completion.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_runtime_completion.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_model_runtime: tests/test_model_runtime.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_model_runtime.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_model_serving_adapter: tests/test_model_serving_adapter.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(TEST_MODEL_SERVING_ADAPTER_MODULE)
	$(CC) $(CPPFLAGS) -DTEST_MODEL_SERVING_ADAPTER_MODULE_PATH=\"$(TEST_MODEL_SERVING_ADAPTER_MODULE)\" $(CFLAGS) tests/test_model_serving_adapter.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_model_resident_deployment: tests/test_model_resident_deployment.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_model_resident_deployment.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_model_resident_ipc: tests/test_model_resident_ipc.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_model_resident_ipc.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_model_resident_deadline: tests/test_model_resident_deadline.c node/model_residentd.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(MODEL_COMMON_INCLUDE_FLAGS) $(CFLAGS) tests/test_model_resident_deadline.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) $(SPARKPIPE_CUDA_RUNTIME_LINK) -o $@

build/test_model_pipeline_client: tests/test_model_pipeline_client.c tests/fixtures/model_resident_deployment_fixture.c tests/fixtures/model_serving_adapter_config.json build/sparkpipe_model_residentd build/sparkpipe_model_batch $(TEST_MODEL_SERVING_ADAPTER_MODULE) $(TEST_MODEL_RESIDENT_TRANSPORT_MODULE) $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(CPPFLAGS) -DTEST_MODEL_RESIDENTD_PATH=\"build/sparkpipe_model_residentd\" -DTEST_MODEL_BATCH_PATH=\"build/sparkpipe_model_batch\" -DTEST_MODEL_SERVING_ADAPTER_PATH=\"$(TEST_MODEL_SERVING_ADAPTER_MODULE)\" -DTEST_MODEL_RESIDENT_TRANSPORT_PATH=\"$(TEST_MODEL_RESIDENT_TRANSPORT_MODULE)\" $(CFLAGS) tests/test_model_pipeline_client.c tests/fixtures/model_resident_deployment_fixture.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_pipeline_runtime: tests/test_pipeline_runtime.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_pipeline_runtime.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_dsv4_serving_adapter: tests/test_dsv4_serving_adapter.c tests/fixtures/dsv4_serving_adapter_config_absolute.json tests/fixtures/dsv4_serving_adapter_config_graphs.json tests/fixtures/dsv4_serving_adapter_config_graphs_overrun.json $(DSV4_SERVING_ADAPTER) $(TEST_DSV4_SERVING_DRIVER_MODULE) $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(CPPFLAGS) -Imodules/dsv4_resident_decode_stage/include -DTEST_DSV4_SERVING_ADAPTER_PATH=\"$(DSV4_SERVING_ADAPTER)\" -DTEST_DSV4_SERVING_DRIVER_PATH=\"$(TEST_DSV4_SERVING_DRIVER_MODULE)\" -DTEST_DSV4_SERVING_CONFIG_PATH=\"tests/fixtures/dsv4_serving_adapter_config.json\" -DTEST_DSV4_SERVING_STALE_CONFIG_PATH=\"tests/fixtures/dsv4_serving_adapter_config_stale.json\" -DTEST_DSV4_SERVING_ABSOLUTE_CONFIG_PATH=\"tests/fixtures/dsv4_serving_adapter_config_absolute.json\" -DTEST_DSV4_SERVING_GRAPHS_CONFIG_PATH=\"tests/fixtures/dsv4_serving_adapter_config_graphs.json\" -DTEST_DSV4_SERVING_GRAPHS_OVERRUN_CONFIG_PATH=\"tests/fixtures/dsv4_serving_adapter_config_graphs_overrun.json\" $(CFLAGS) tests/test_dsv4_serving_adapter.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_dsv4_tp16_serving_adapter: tests/test_dsv4_tp16_serving_adapter.c tests/fixtures/dsv4_tp16_serving_adapter_config.json $(DSV4_TP16_SERVING_ADAPTER) $(TEST_DSV4_TP16_SERVING_DRIVER_MODULE) $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(CPPFLAGS) -Imodules/dsv4_resident_decode_stage/include -DTEST_DSV4_TP16_ADAPTER_PATH=\"$(DSV4_TP16_SERVING_ADAPTER)\" -DTEST_DSV4_TP16_DRIVER_PATH=\"$(TEST_DSV4_TP16_SERVING_DRIVER_MODULE)\" -DTEST_DSV4_TP16_CONFIG_PATH=\"tests/fixtures/dsv4_tp16_serving_adapter_config.json\" $(CFLAGS) tests/test_dsv4_tp16_serving_adapter.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_dsv4_tp4_pp4_serving_adapter: tests/test_dsv4_tp4_pp4_serving_adapter.c tests/fixtures/dsv4_tp4_pp4_serving_adapter_config.json tests/fixtures/dsv4_tp4_pp4_serving_adapter_config_missing_graphs.json tests/fixtures/dsv4_tp4_pp4_serving_adapter_config_short_graphs.json $(DSV4_TP4_PP4_SERVING_ADAPTER) $(TEST_DSV4_TP4_PP4_SERVING_DRIVER_MODULE) $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(CPPFLAGS) -Imodules/dsv4_resident_decode_stage/include -DTEST_DSV4_TP4_PP4_ADAPTER_PATH=\"$(DSV4_TP4_PP4_SERVING_ADAPTER)\" -DTEST_DSV4_TP4_PP4_DRIVER_PATH=\"$(TEST_DSV4_TP4_PP4_SERVING_DRIVER_MODULE)\" -DTEST_DSV4_TP4_PP4_CONFIG_PATH=\"tests/fixtures/dsv4_tp4_pp4_serving_adapter_config.json\" -DTEST_DSV4_TP4_PP4_MISSING_GRAPHS_CONFIG_PATH=\"tests/fixtures/dsv4_tp4_pp4_serving_adapter_config_missing_graphs.json\" -DTEST_DSV4_TP4_PP4_SHORT_GRAPHS_CONFIG_PATH=\"tests/fixtures/dsv4_tp4_pp4_serving_adapter_config_short_graphs.json\" $(CFLAGS) tests/test_dsv4_tp4_pp4_serving_adapter.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_qwen36_serving_adapter: tests/test_qwen36_serving_adapter.c tests/fixtures/qwen36_serving_adapter_config.json tests/fixtures/qwen36_serving_adapter_config_stale.json tests/fixtures/qwen36_serving_adapter_config_absolute.json tests/fixtures/qwen36_serving_adapter_config_overrun.json $(QWEN36_SERVING_ADAPTER) $(TEST_QWEN36_SERVING_DRIVER_MODULE) $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(CPPFLAGS) $(QWEN36_INCLUDE_FLAGS) -Imodules/qwen36_resident_decode_stage/include -DTEST_QWEN36_SERVING_ADAPTER_PATH=\"$(QWEN36_SERVING_ADAPTER)\" -DTEST_QWEN36_SERVING_DRIVER_PATH=\"$(TEST_QWEN36_SERVING_DRIVER_MODULE)\" -DTEST_QWEN36_SERVING_CONFIG_PATH=\"tests/fixtures/qwen36_serving_adapter_config.json\" -DTEST_QWEN36_SERVING_STALE_CONFIG_PATH=\"tests/fixtures/qwen36_serving_adapter_config_stale.json\" -DTEST_QWEN36_SERVING_ABSOLUTE_CONFIG_PATH=\"tests/fixtures/qwen36_serving_adapter_config_absolute.json\" -DTEST_QWEN36_SERVING_OVERRUN_CONFIG_PATH=\"tests/fixtures/qwen36_serving_adapter_config_overrun.json\" $(CFLAGS) tests/test_qwen36_serving_adapter.c $(SPARKPIPE_HOST_CUDA_STUB_SOURCE) $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) $(SPARKPIPE_CUDA_RUNTIME_LINK) -o $@

build/test_model_resident_end_to_end: tests/test_model_resident_end_to_end.c tests/fixtures/model_resident_deployment_fixture.c build/sparkpipe_model_residentd $(DSV4_SERVING_ADAPTER) $(TEST_DSV4_SERVING_DRIVER_MODULE) $(TEST_MODEL_RESIDENT_TRANSPORT_MODULE) $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(CPPFLAGS) -DTEST_MODEL_RESIDENTD_PATH=\"build/sparkpipe_model_residentd\" -DTEST_DSV4_SERVING_ADAPTER_PATH=\"$(DSV4_SERVING_ADAPTER)\" -DTEST_DSV4_SERVING_DRIVER_PATH=\"$(TEST_DSV4_SERVING_DRIVER_MODULE)\" -DTEST_DSV4_SERVING_CONFIG_PATH=\"tests/fixtures/dsv4_serving_adapter_config.json\" -DTEST_MODEL_RESIDENT_TRANSPORT_PATH=\"$(TEST_MODEL_RESIDENT_TRANSPORT_MODULE)\" $(CFLAGS) tests/test_model_resident_end_to_end.c tests/fixtures/model_resident_deployment_fixture.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

runtime_completion_tests: build/test_runtime_completion build/test_model_runtime
	./build/test_runtime_completion
	./build/test_model_runtime

# Header-only cache plus a mock encode; the driver stub's cuda.h is what lets
# runtime/tensor_map.h compile with no CUDA toolkit, and its stub.c stands in
# for cuTensorMapEncodeTiled so the production entry point is exercised too.
# -x c++ on the stub keeps the symbol mangling identical to the test TU's
# (the stub predates C and C++ disagreeing on the name).
build/test_gemm_descriptor_cache: tests/test_gemm_descriptor_cache.cpp tests/cuda_driver_stub/stub.c runtime/gemm_descriptor_cache.h | build
	$(CXX) -x c++ -Itests/cuda_driver_stub -O2 -Wall -Wextra -c tests/cuda_driver_stub/stub.c -o build/test_gemm_descriptor_cache_stub.o
	$(CXX) $(CPPFLAGS) -I. -Itests/cuda_driver_stub $(CXXFLAGS) tests/test_gemm_descriptor_cache.cpp build/test_gemm_descriptor_cache_stub.o $(LDFLAGS) $(LDLIBS) -o $@

build/test_launch: tests/test_launch.c runtime/launch.h inference/kernels/layout.cuh | build
	$(CXX) -x c++ -D__host__= -D__device__= $(CPPFLAGS) $(CXXFLAGS) -Wno-unused-function $< $(LDFLAGS) $(LDLIBS) -o $@

build/test_tensor_map_geometry: tests/test_tensor_map_geometry.c inference/kernels/tensor_map.cuh | build
	$(CC) $(CPPFLAGS) -I. $(CFLAGS) tests/test_tensor_map_geometry.c $(LDFLAGS) $(LDLIBS) -o $@

build/test_distributed_work: tests/test_distributed_work.c $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_distributed_work.c $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_json: tests/test_json.c $(COMPILER_LIBRARY) $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(COMPILER_LIBRARY) $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_fabric_topology: tests/test_fabric_topology.c $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(MODEL_COMMON_INCLUDE_FLAGS) $(CFLAGS) $< $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_hidden_transport: tests/test_hidden_transport.c $(COMMON_LIBRARY) $(TEST_HIDDEN_TRANSPORT_MODULE)
	$(CC) $(CPPFLAGS) -Itests -DSPARK_TEST_HIDDEN_TRANSPORT_MODULE_PATH=\"$(TEST_HIDDEN_TRANSPORT_MODULE)\" $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_hidden_transport_rdma_control: tests/test_hidden_transport_rdma_control.c ring/transport/rdma_control.c include/sparkpipe/spark_hidden_transport_rdma_control.h
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_hidden_transport_rdma_control.c ring/transport/rdma_control.c $(LDFLAGS) $(LDLIBS) -lpthread -o $@

build/test_memlink: tests/test_memlink.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_release: tests/test_release.c $(COMMON_LIBRARY)
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

build/test_tp_collective: tests/test_tp_collective.c include/sparkpipe/spark_tp_collective.h $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -lpthread -o $@

build/test_tp_device_collective: tests/test_tp_device_collective.c include/sparkpipe/spark_tp_device_collective.h $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(TEST_TP_DEVICE_COLLECTIVE_MODULE)
	$(CC) $(MODEL_COMMON_INCLUDE_FLAGS) -DSPARK_TEST_TP_DEVICE_COLLECTIVE_MODULE_PATH=\"$(TEST_TP_DEVICE_COLLECTIVE_MODULE)\" $(CFLAGS) $< $(SPARKPIPE_TP_DEVICE_TEST_CUDA_STUB_SOURCE) tests/fixtures/tp_device_collective_cuda_query_stub.c $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_tp_device_collective_nccl: tests/test_tp_device_collective_nccl.c include/sparkpipe/spark_tp_device_collective.h $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(TEST_TP_DEVICE_COLLECTIVE_NCCL_MODULE)
	$(CC) $(MODEL_COMMON_INCLUDE_FLAGS) -DSPARK_TEST_TP_DEVICE_COLLECTIVE_NCCL_MODULE_PATH=\"$(TEST_TP_DEVICE_COLLECTIVE_NCCL_MODULE)\" $(CFLAGS) $< $(SPARKPIPE_TP_DEVICE_TEST_CUDA_STUB_SOURCE) tests/fixtures/tp_device_collective_cuda_query_stub.c $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_glm52_mtp_tree: tests/test_glm52_mtp_tree.c model-families/glm52/include/sparkpipe/spark_glm52_mtp_tree.h $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_glm52_stagepack: tests/test_glm52_stagepack.c modules/glm52_resident_decode_stage/source/spark_glm52_stagepack_format.h
	$(CC) $(GLM52_INCLUDE_FLAGS) -Imodules/glm52_resident_decode_stage/source \
		$(CFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

build/test_glm52_dspark: tests/test_glm52_dspark.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_tokenizer: tests/test_tokenizer.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_model_description: tests/test_model_description.c $(COMPILER_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(CORE_INCLUDE_FLAGS) -Itests $(CFLAGS) $< $(COMPILER_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_stage_module_common: tests/test_stage_module_common.c runtime/stage_module_common.c tests/cuda_stub/cuda_runtime_stub.c | build
	$(CC) $(MODEL_COMMON_INCLUDE_FLAGS) -Itests/cuda_stub -Itests $(CFLAGS) $^ $(LDFLAGS) -o $@

build/test_module_library: tests/test_module_library.c $(TEST_SUPPORT_OBJECT) $(TEST_MODULE_LINK_UNITS) $(TEST_VALIDATOR) $(TEST_VALIDATOR_CHANGED) $(COMPILER_LIBRARY) $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(TEST_SUPPORT_OBJECT) $(COMPILER_LIBRARY) $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_driver_compiler: tests/test_driver_compiler.c $(TEST_SUPPORT_OBJECT) $(TEST_MODULE_LINK_UNITS) $(TEST_VALIDATOR) $(COMPILER_LIBRARY) $(RUNTIME_LIBRARY) $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(TEST_SUPPORT_OBJECT) $(COMPILER_LIBRARY) $(RUNTIME_LIBRARY) $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_orchestrator: tests/test_orchestrator.c $(TEST_SUPPORT_OBJECT) $(TEST_MODULE_LINK_UNITS) $(TEST_VALIDATOR) $(COMPILER_LIBRARY) $(RUNTIME_LIBRARY) $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(TEST_SUPPORT_OBJECT) $(COMPILER_LIBRARY) $(RUNTIME_LIBRARY) $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_dsv4_stage_runner: tests/test_dsv4_stage_runner.c modules/dsv4_resident_decode_stage/source/spark_dsv4_stage_runner.c modules/dsv4_resident_decode_stage/include/sparkpipe/spark_dsv4_resident_decode_stage_runner.h modules/dsv4_resident_decode_stage/include/sparkpipe/spark_dsv4_resident_decode_stage_firmware.h $(COMMON_LIBRARY) $(MODEL_COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests -Imodules/dsv4_resident_decode_stage/include $(CFLAGS) tests/test_dsv4_stage_runner.c modules/dsv4_resident_decode_stage/source/spark_dsv4_stage_runner.c $(COMMON_LIBRARY) $(MODEL_COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_dsv4_lane_continuity: tests/test_dsv4_lane_continuity.c modules/dsv4_resident_decode_stage/source/spark_dsv4_lane_continuity.h
	$(CC) $(CPPFLAGS) -Imodules/dsv4_resident_decode_stage/source $(CFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

build/test_dsv4_tp_graph_contract: tests/test_dsv4_tp_graph_contract.c modules/dsv4_resident_decode_stage/include/sparkpipe/spark_dsv4_resident_decode_stage_firmware.h
	$(CC) $(DSV4_INCLUDE_FLAGS) -Imodules/dsv4_resident_decode_stage/include $(CFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

build/test_dsv4_pool_layout: tests/test_dsv4_pool_layout.c modules/dsv4_resident_decode_stage/source/spark_dsv4_pool_layout.h
	$(CC) $(DSV4_INCLUDE_FLAGS) -Imodules/dsv4_resident_decode_stage/include -Imodules/dsv4_resident_decode_stage/source $(CFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

build/test_dsv4_paged_cache: tests/test_dsv4_paged_cache.c modules/dsv4_resident_decode_stage/source/spark_dsv4_paged_cache.c modules/dsv4_resident_decode_stage/source/spark_dsv4_paged_cache.h $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(DSV4_INCLUDE_FLAGS) -Imodules/dsv4_resident_decode_stage/include -Imodules/dsv4_resident_decode_stage/source $(CFLAGS) tests/test_dsv4_paged_cache.c modules/dsv4_resident_decode_stage/source/spark_dsv4_paged_cache.c $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) $(SPARKPIPE_CUDA_RUNTIME_LINK) -o $@

build/test_weight_codec: tests/test_weight_codec.c include/sparkpipe/spark_weight_codec.h
	$(CC) $(CORE_INCLUDE_FLAGS) $(CFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

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

glm52_resident_decode_stage_contract:
	$(MAKE) -C modules/glm52_resident_decode_stage contract \
		EXPERT_CODEC='$(EXPERT_CODEC)' MODEL_REVISION='$(MODEL_REVISION)' \
		CONTRACT_SHA256='$(CONTRACT_SHA256)'

glm52_resident_decode_stage_archive:
	$(MAKE) -C modules/glm52_resident_decode_stage archive \
		EXPERT_CODEC='$(EXPERT_CODEC)' MODEL_REVISION='$(MODEL_REVISION)' \
		CONTRACT_SHA256='$(CONTRACT_SHA256)' NVCC='$(NVCC)' CUDA_ARCH='$(CUDA_ARCH)'

# The batch-variant sets: one archive per power-of-two bucket from each
# module Makefile's variants target, compiled from the single
# SPARK_MODULE_BATCH_VARIANT_RULES template in
# modules/resident_decode_stage_rules.mk. Trim a set with the family's
# MODULE_BATCH_VARIANT_BUCKETS, never by editing a variant.
cuda_glm52_resident_decode_stage_variants:
	$(MAKE) -C modules/glm52_resident_decode_stage variants \
		EXPERT_CODEC='$(EXPERT_CODEC)' MODEL_REVISION='$(MODEL_REVISION)' \
		CONTRACT_SHA256='$(CONTRACT_SHA256)' NVCC='$(NVCC)' CUDA_ARCH='$(CUDA_ARCH)'

cuda_dsv4_resident_decode_stage_variants:
	$(MAKE) -C modules/dsv4_resident_decode_stage variants \
		NVCC='$(NVCC)' CUDA_ARCH='$(CUDA_ARCH)'

glm52_resident_decode_stage_publish:
	$(MAKE) -C modules/glm52_resident_decode_stage publish \
		EXPERT_CODEC='$(EXPERT_CODEC)' MODEL_REVISION='$(MODEL_REVISION)' \
		CONTRACT_SHA256='$(CONTRACT_SHA256)' STAGE_PACK_PATH='$(STAGE_PACK_PATH)' \
		GPU_VALIDATOR='$(GPU_VALIDATOR)' NVCC='$(NVCC)' CUDA_ARCH='$(CUDA_ARCH)'

glm52_serving_adapter:
	$(MAKE) -C modules/glm52_resident_decode_stage adapter \
		EXPERT_CODEC='$(EXPERT_CODEC)' MODEL_REVISION='$(MODEL_REVISION)' \
		CONTRACT_SHA256='$(CONTRACT_SHA256)'

glm52_dspark_draft_backend:
	$(MAKE) -C modules/glm52_dspark_draft_backend archive NVCC=$(NVCC) CUDA_ARCH=sm_121a

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
    $(TEST_MODULE_DEPENDENCIES)
