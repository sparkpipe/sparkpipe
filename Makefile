CC ?= cc
CXX ?= c++
AR ?= ar
NVCC ?= nvcc

CUDA_HOME ?= /usr/local/cuda
CFLAGS ?= -std=c11 -Wall -Wextra -Werror -O3 -g -pthread
CFLAGS += -D_GNU_SOURCE
CXXFLAGS ?= -std=c++20 -Wall -Wextra -Werror -O3 -g -pthread
CORE_INCLUDE_FLAGS := -I. -Iinclude -Isrc
CUDA_STUB_INCLUDE_FLAGS := -Itests/cuda_stub -Imodel-families/common/include
CUDA_RUNTIME_HEADER := $(wildcard $(CUDA_HOME)/include/cuda_runtime.h)
ifeq ($(CUDA_RUNTIME_HEADER),)
MODEL_COMMON_INCLUDE_FLAGS := $(CORE_INCLUDE_FLAGS) $(CUDA_STUB_INCLUDE_FLAGS)
SPARKPIPE_HOST_CUDA_STUB_SOURCE := tests/cuda_stub/cuda_runtime_stub.c
SPARKPIPE_TP_DEVICE_TEST_CUDA_STUB_SOURCE :=
SPARKPIPE_CUDA_RUNTIME_LINK :=
SPARKPIPE_CUDA_DRIVER_LINK :=
else
MODEL_COMMON_INCLUDE_FLAGS := $(CORE_INCLUDE_FLAGS) -I$(CUDA_HOME)/include
SPARKPIPE_HOST_CUDA_STUB_SOURCE :=
SPARKPIPE_TP_DEVICE_TEST_CUDA_STUB_SOURCE := tests/cuda_stub/cuda_runtime_stub.c
SPARKPIPE_CUDA_RUNTIME_LINK := -L$(CUDA_HOME)/lib64 -lcudart
SPARKPIPE_CUDA_DRIVER_LINK := -L$(CUDA_HOME)/lib64 -lcuda
endif
MODEL_COMMON_INCLUDE_FLAGS += -Imodel-families/common/include
GLM52_INCLUDE_FLAGS := $(MODEL_COMMON_INCLUDE_FLAGS) -Imodel-families/glm52/include
DEFAULT_BATCH_FLAGS := -DSPARK_BATCH_BUCKET=1024u
K3_INCLUDE_FLAGS := $(MODEL_COMMON_INCLUDE_FLAGS) -Imodel-families/k3/include
MODEL_FAMILY_INCLUDE_FLAGS := \
    -Imodel-families/common/include \
    -Imodel-families/glm52/include \
    -Imodel-families/k3/include
DEPLOYMENT_INCLUDE_FLAGS := $(CORE_INCLUDE_FLAGS) -Ideployment/include -Ideployment/src
CPPFLAGS ?= $(CORE_INCLUDE_FLAGS) $(MODEL_FAMILY_INCLUDE_FLAGS) -Ideployment/include -Ideployment/src $(DEFAULT_BATCH_FLAGS)
LDFLAGS ?=
LDLIBS ?= -ldl -pthread
LDFLAGS += $(SPARKPIPE_CUDA_DRIVER_LINK)
CUDA_ARCH ?= sm_121a
CUDA_COMPUTE_ARCH ?= $(subst sm_,compute_,$(CUDA_ARCH))
NVCCFLAGS ?= -O3 -gencode arch=$(CUDA_COMPUTE_ARCH),code=$(CUDA_ARCH)
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

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
HIDDEN_TRANSPORT_HOST_STAGED_TCP := build/libhidden_transport_host_staged_tcp.$(SHARED_LIBRARY_EXT)

include sources.mk

CORE_SOURCES := $(SPARKPIPE_CORE_SOURCES)
MODEL_COMMON_SOURCES := $(SPARKPIPE_MODEL_COMMON_SOURCES) $(SPARKPIPE_HOST_CUDA_STUB_SOURCE)
DEPLOYMENT_SOURCES := $(SPARKPIPE_DEPLOYMENT_SOURCES)
GLM52_HOST_SOURCES := $(SPARKPIPE_GLM52_SOURCES)
COMPILER_SOURCES := $(SPARKPIPE_COMPILER_SOURCES)
RUNTIME_SOURCES := $(SPARKPIPE_RUNTIME_SOURCES)

sp_objects = $(patsubst %.c,build/obj/%.o,$(1))

CORE_OBJECTS := $(call sp_objects,$(CORE_SOURCES))
TRANSPORT_SOURCES := $(SPARKPIPE_TRANSPORT_SOURCES)
CACHE_SOURCES := $(SPARKPIPE_CACHE_SOURCES)
TRANSPORT_OBJECTS := $(call sp_objects,$(TRANSPORT_SOURCES))
CACHE_OBJECTS := $(call sp_objects,$(CACHE_SOURCES))
MODEL_COMMON_OBJECTS := $(call sp_objects,$(MODEL_COMMON_SOURCES))
DEPLOYMENT_OBJECTS := $(call sp_objects,$(DEPLOYMENT_SOURCES))
GLM52_HOST_OBJECTS := $(call sp_objects,$(GLM52_HOST_SOURCES))
COMPILER_OBJECTS := $(call sp_objects,$(COMPILER_SOURCES))
RUNTIME_OBJECTS := $(call sp_objects,$(RUNTIME_SOURCES))
ALL_HOST_OBJECTS := $(CORE_OBJECTS) $(MODEL_COMMON_OBJECTS) $(DEPLOYMENT_OBJECTS) \
    $(GLM52_HOST_OBJECTS) $(COMPILER_OBJECTS) $(RUNTIME_OBJECTS)

$(CORE_OBJECTS): SP_INCLUDE_FLAGS = $(CORE_INCLUDE_FLAGS)
$(COMPILER_OBJECTS): SP_INCLUDE_FLAGS = $(CORE_INCLUDE_FLAGS)
$(RUNTIME_OBJECTS): SP_INCLUDE_FLAGS = $(CORE_INCLUDE_FLAGS)
$(MODEL_COMMON_OBJECTS): SP_INCLUDE_FLAGS = $(MODEL_COMMON_INCLUDE_FLAGS)
build/obj/runtime/spark_weightd.o build/obj/runtime/spark_weightd_attach.o build/obj/runtime/spark_weightd_kv_pool.o build/obj/runtime/spark_weightd_map.o build/obj/runtime/spark_weightd_spine.o build/obj/runtime/spark_weightd_receipt.o build/obj/runtime/spark_weightd_worker.o build/obj/runtime/spark_weightd_lazy_pack.o: SP_INCLUDE_FLAGS = $(MODEL_COMMON_INCLUDE_FLAGS)
$(DEPLOYMENT_OBJECTS): SP_INCLUDE_FLAGS = $(DEPLOYMENT_INCLUDE_FLAGS)
$(GLM52_HOST_OBJECTS): SP_INCLUDE_FLAGS = $(GLM52_INCLUDE_FLAGS)
CORE_LIBRARY := build/libsparkpipe_core.a
MODEL_COMMON_LIBRARY := build/libsparkpipe_model_common.a
TRANSPORT_LIBRARY := build/libsparkpipe_transport.a
DEPLOYMENT_LIBRARY := build/libsparkpipe_deployment.a
GLM52_HOST_LIBRARY := build/libglm52_host.a
COMPILER_LIBRARY := build/libsparkpipe_compiler.a
RUNTIME_LIBRARY := build/libsparkpipe_runtime.a
COMMON_LIBRARY := $(CORE_LIBRARY)
LIBRARIES := $(CORE_LIBRARY) $(MODEL_COMMON_LIBRARY) $(COMPILER_LIBRARY) $(RUNTIME_LIBRARY) $(DEPLOYMENT_LIBRARY) $(GLM52_HOST_LIBRARY)

K3_SERVING_ADAPTER := build/libk3_serving_adapter.$(SHARED_LIBRARY_EXT)
K3_TP16_SERVING_ADAPTER := build/libk3_tp16_serving_adapter.$(SHARED_LIBRARY_EXT)
QWEN38_27B_TP16_SERVING_ADAPTER := build/libqwen38_27b_tp16_serving_adapter.$(SHARED_LIBRARY_EXT)
DSV41_FLASH_TP16_SERVING_ADAPTER := build/libdsv41_flash_tp16_serving_adapter.$(SHARED_LIBRARY_EXT)
DSV41_FLASH_TP16_CONTRACT := model_contracts/dsv41_flash_authoritative.json
DSV41_FLASH_TP16_CONTRACT_SHA256 ?= $(shell if command -v sha256sum >/dev/null 2>&1; then sha256sum "$(DSV41_FLASH_TP16_CONTRACT)"; else shasum -a 256 "$(DSV41_FLASH_TP16_CONTRACT)"; fi | awk '{print $$1}')
DSV41_FLASH_TP16_SOURCES := modules/dsv41_flash_resident_decode_stage/source/spark_dsv41_flash_stage_serving.c inference/runner/stage_serving_adapter.c inference/runner/stage_runner.cu modules/dsv41_flash_resident_decode_stage/source/spark_dsv41_flash_stage_model.cu
QWEN38_27B_BF16_TP16_SERVING_ADAPTER := build/libqwen38_27b_bf16_tp16_serving_adapter.$(SHARED_LIBRARY_EXT)
QWEN38_27B_TP16_CONTRACT := model_contracts/qwen38_27b_authoritative.json
QWEN38_27B_TP16_CONTRACT_SHA256 ?= $(shell if command -v sha256sum >/dev/null 2>&1; then sha256sum "$(QWEN38_27B_TP16_CONTRACT)"; else shasum -a 256 "$(QWEN38_27B_TP16_CONTRACT)"; fi | awk '{print $$1}')
QWEN38_27B_TP16_SOURCES := modules/qwen38_27b_resident_decode_stage/source/spark_qwen38_27b_stage_serving.c inference/runner/stage_serving_adapter.c inference/runner/stage_runner.cu modules/qwen38_27b_resident_decode_stage/source/spark_qwen38_27b_stage_model.cu
QWEN38_MAX_TP16_SERVING_ADAPTER := build/libqwen38_max_tp16_serving_adapter.$(SHARED_LIBRARY_EXT)
QWEN38_MAX_TP16_CONTRACT := model_contracts/qwen38_authoritative.json
QWEN38_MAX_TP16_CONTRACT_SHA256 ?= $(shell if command -v sha256sum >/dev/null 2>&1; then sha256sum "$(QWEN38_MAX_TP16_CONTRACT)"; else shasum -a 256 "$(QWEN38_MAX_TP16_CONTRACT)"; fi | awk '{print $$1}')
QWEN38_MAX_TP16_SOURCES := modules/qwen38_max_resident_decode_stage/source/spark_qwen38_max_stage_serving.c inference/runner/stage_serving_adapter.c inference/runner/stage_runner.cu modules/qwen38_max_resident_decode_stage/source/spark_qwen38_max_stage_model.cu
QWEN38_FLASH_TP16_SERVING_ADAPTER := build/libqwen38_flash_tp16_serving_adapter.$(SHARED_LIBRARY_EXT)
QWEN38_FLASH_TP16_CONTRACT := model_contracts/qwen4_flash_authoritative.json
QWEN38_FLASH_TP16_CONTRACT_SHA256 ?= $(shell if command -v sha256sum >/dev/null 2>&1; then sha256sum "$(QWEN38_FLASH_TP16_CONTRACT)"; else shasum -a 256 "$(QWEN38_FLASH_TP16_CONTRACT)"; fi | awk '{print $$1}')
QWEN38_FLASH_TP16_SOURCES := modules/qwen38_flash_resident_decode_stage/source/spark_qwen38_flash_stage_serving.c inference/runner/stage_serving_adapter.c inference/runner/stage_runner.cu modules/qwen38_flash_resident_decode_stage/source/spark_qwen38_flash_stage_model.cu
GEMMA4_31B_TP16_SERVING_ADAPTER := build/libgemma4_31b_tp16_serving_adapter.$(SHARED_LIBRARY_EXT)
GEMMA4_31B_TP16_CONTRACT := model_contracts/gemma4_31b_authoritative.json
GEMMA4_31B_TP16_CONTRACT_SHA256 ?= $(shell if command -v sha256sum >/dev/null 2>&1; then sha256sum "$(GEMMA4_31B_TP16_CONTRACT)"; else shasum -a 256 "$(GEMMA4_31B_TP16_CONTRACT)"; fi | awk '{print $$1}')
GEMMA4_31B_TP16_SOURCES := modules/gemma4_31b_resident_decode_stage/source/spark_gemma4_31b_stage_serving.c inference/runner/stage_serving_adapter.c inference/runner/stage_runner.cu modules/gemma4_31b_resident_decode_stage/source/spark_gemma4_31b_stage_model.cu
GEMMA4_26B_TP16_SERVING_ADAPTER := build/libgemma4_26b_tp16_serving_adapter.$(SHARED_LIBRARY_EXT)
GEMMA4_26B_TP16_CONTRACT := model_contracts/gemma4_26b_a4b_authoritative.json
GEMMA4_26B_TP16_CONTRACT_SHA256 ?= $(shell if command -v sha256sum >/dev/null 2>&1; then sha256sum "$(GEMMA4_26B_TP16_CONTRACT)"; else shasum -a 256 "$(GEMMA4_26B_TP16_CONTRACT)"; fi | awk '{print $$1}')
GEMMA4_26B_TP16_SOURCES := modules/gemma4_26b_resident_decode_stage/source/spark_gemma4_26b_stage_serving.c inference/runner/stage_serving_adapter.c inference/runner/stage_runner.cu modules/gemma4_26b_resident_decode_stage/source/spark_gemma4_26b_stage_model.cu
LING3_FLASH_TP16_SERVING_ADAPTER := build/libling3_flash_tp16_serving_adapter.$(SHARED_LIBRARY_EXT)
LING3_FIN_TP16_SERVING_ADAPTER := build/libling3_fin_tp16_serving_adapter.$(SHARED_LIBRARY_EXT)
LING3_FLASH_TP16_CONTRACT := model_contracts/ling_authoritative.json
LING3_FIN_TP16_CONTRACT := model_contracts/lingfin_authoritative.json
LING3_FLASH_TP16_CONTRACT_SHA256 ?= $(shell if command -v sha256sum >/dev/null 2>&1; then sha256sum "$(LING3_FLASH_TP16_CONTRACT)"; else shasum -a 256 "$(LING3_FLASH_TP16_CONTRACT)"; fi | awk '{print $$1}')
LING3_FIN_TP16_CONTRACT_SHA256 ?= $(shell if command -v sha256sum >/dev/null 2>&1; then sha256sum "$(LING3_FIN_TP16_CONTRACT)"; else shasum -a 256 "$(LING3_FIN_TP16_CONTRACT)"; fi | awk '{print $$1}')
LING3_TP16_SOURCES := modules/ling3_resident_decode_stage/source/spark_ling3_stage_serving.c inference/runner/stage_serving_adapter.c inference/runner/stage_runner.cu modules/ling3_resident_decode_stage/source/spark_ling3_stage_model.cu
HY4_TP16_SERVING_ADAPTER := build/libhy4_tp16_serving_adapter.$(SHARED_LIBRARY_EXT)
HY4_TP16_CONTRACT := model_contracts/hy4_authoritative.json
HY4_TP16_CONTRACT_SHA256 ?= $(shell if command -v sha256sum >/dev/null 2>&1; then sha256sum "$(HY4_TP16_CONTRACT)"; else shasum -a 256 "$(HY4_TP16_CONTRACT)"; fi | awk '{print $$1}')
HY4_TP16_SOURCES := modules/hy4_tp16_resident_decode_stage/source/spark_hy4_tp16_stage_serving.c inference/runner/stage_serving_adapter.c inference/runner/stage_runner.cu modules/hy4_tp16_resident_decode_stage/source/spark_hy4_tp16_stage_model.cu
GLM53_FULL_TP16_SERVING_ADAPTER := build/libglm53_full_tp16_serving_adapter.$(SHARED_LIBRARY_EXT)
GLM53_FULL_TP16_IDENTITY := $(shell python3 tools/glm52_model_contract.py --print-build-identity fp8 --expert-codec fp8 2>/dev/null)
GLM53_FULL_TP16_REVISION := $(word 1,$(GLM53_FULL_TP16_IDENTITY))
GLM53_FULL_TP16_CONTRACT_SHA256 := $(word 2,$(GLM53_FULL_TP16_IDENTITY))
GLM53_FULL_TP16_SOURCES := modules/glm53_full_tp16_resident_decode_stage/source/spark_glm53_full_tp16_stage_serving.c inference/runner/stage_serving_adapter.c inference/runner/stage_runner.cu modules/glm53_full_tp16_resident_decode_stage/source/spark_glm53_full_tp16_stage_model.cu
MUSE_TP16_SERVING_ADAPTER := build/libmuse_glimmer_tp16_serving_adapter.$(SHARED_LIBRARY_EXT)
MUSE_TP16_CONTRACT := model_contracts/muse_glimmer_authoritative.json
MUSE_TP16_CONTRACT_SHA256 ?= $(shell if command -v sha256sum >/dev/null 2>&1; then sha256sum "$(MUSE_TP16_CONTRACT)"; else shasum -a 256 "$(MUSE_TP16_CONTRACT)"; fi | awk '{print $$1}')
MUSE_TP16_SOURCES := modules/muse_glimmer_tp16_resident_decode_stage/source/spark_muse_tp16_stage_serving.c inference/runner/stage_serving_adapter.c inference/runner/stage_runner.cu modules/muse_glimmer_tp16_resident_decode_stage/source/spark_muse_tp16_stage_model.cu
GLM53_FLASH_TP16_SERVING_ADAPTER := build/libglm53_flash_tp16_serving_adapter.$(SHARED_LIBRARY_EXT)
GLM53_FLASH_TP16_CONTRACT := model_contracts/glm53_flash_authoritative.json
GLM53_FLASH_TP16_CONTRACT_SHA256 ?= $(shell if command -v sha256sum >/dev/null 2>&1; then sha256sum "$(GLM53_FLASH_TP16_CONTRACT)"; else shasum -a 256 "$(GLM53_FLASH_TP16_CONTRACT)"; fi | awk '{print $$1}')
GLM53_FLASH_TP16_SOURCES := modules/glm53_flash_tp16_resident_decode_stage/source/spark_glm53_flash_tp16_stage_serving.c inference/runner/stage_serving_adapter.c inference/runner/stage_runner.cu modules/glm53_flash_tp16_resident_decode_stage/source/spark_glm53_flash_tp16_stage_model.cu
LAGUNA_TP16_SERVING_ADAPTER := build/liblaguna_tp16_serving_adapter.$(SHARED_LIBRARY_EXT)
LAGUNA_TP16_CONTRACT := model_contracts/laguna_authoritative.json
LAGUNA_TP16_CONTRACT_SHA256 ?= $(shell if command -v sha256sum >/dev/null 2>&1; then sha256sum "$(LAGUNA_TP16_CONTRACT)"; else shasum -a 256 "$(LAGUNA_TP16_CONTRACT)"; fi | awk '{print $$1}')
LAGUNA_TP16_SOURCES := modules/laguna_tp16_resident_decode_stage/source/spark_laguna_tp16_stage_serving.c inference/runner/stage_serving_adapter.c inference/runner/stage_runner.cu modules/laguna_tp16_resident_decode_stage/source/spark_laguna_tp16_stage_model.cu
MIMO26_FLASH_TP16_SERVING_ADAPTER := build/libmimo26_flash_tp16_serving_adapter.$(SHARED_LIBRARY_EXT)
MIMO26_FLASH_TP16_CONTRACT := model_contracts/mimo26_flash_authoritative.json
MIMO26_FLASH_TP16_CONTRACT_SHA256 ?= $(shell if command -v sha256sum >/dev/null 2>&1; then sha256sum "$(MIMO26_FLASH_TP16_CONTRACT)"; else shasum -a 256 "$(MIMO26_FLASH_TP16_CONTRACT)"; fi | awk '{print $$1}')
MIMO26_FLASH_TP16_SOURCES := modules/mimo26_flash_tp16_resident_decode_stage/source/spark_mimo26_flash_tp16_stage_serving.c inference/runner/stage_serving_adapter.c inference/runner/stage_runner.cu modules/mimo26_flash_tp16_resident_decode_stage/source/spark_mimo26_flash_tp16_stage_model.cu
MIMO26_PRO_TP16_SERVING_ADAPTER := build/libmimo26_pro_tp16_serving_adapter.$(SHARED_LIBRARY_EXT)
MIMO26_PRO_TP16_CONTRACT := model_contracts/mimo26_pro_authoritative.json
MIMO26_PRO_TP16_CONTRACT_SHA256 ?= $(shell if command -v sha256sum >/dev/null 2>&1; then sha256sum "$(MIMO26_PRO_TP16_CONTRACT)"; else shasum -a 256 "$(MIMO26_PRO_TP16_CONTRACT)"; fi | awk '{print $$1}')
MIMO26_PRO_TP16_SOURCES := modules/mimo26_pro_tp16_resident_decode_stage/source/spark_mimo26_pro_tp16_stage_serving.c inference/runner/stage_serving_adapter.c inference/runner/stage_runner.cu modules/mimo26_pro_tp16_resident_decode_stage/source/spark_mimo26_pro_tp16_stage_model.cu

TOOL_NAMES := \
    sparkpipe_module_publish \
    sparkpipe_model_compile \
    sparkpipe_driver_inspect \
    sparkpipe_model_residentd \
    sparkpipe_model_batch \
    sparkpipe_model_api \
    sparkpipe_glm52_tokenize \
    sparkpipe_tokenize_prompt \
    sparkpipe_quant_arm \
    sparkpipe_nfc \
    sparkpipe_split \
    sparkpipe_tokenizer_benchmark \
    sparkpipe_memlink \
    sparkpipe_prevcp \
    sparkpipe_nextcp \
    sparkpipe_release_manager \
    spark_model_kernel_characterize \
    spark_transport_characterize \
    spark_topology_characterize \
    spark_pmtu_characterize \
    sparkpipe_registrar \
    sparkpipe_weightd \
    sparkpipe_weightsd \
    sparkpipe_mesh_status

TOOL_BINARIES := $(addprefix build/,$(TOOL_NAMES))

TEST_NAMES := \
	test_status \
	test_score_dump_keys \
    test_gemm_descriptor_cache \
    test_launch \
    test_arena \
    test_work_transaction \
    test_runtime_completion \
	test_memory_buffer \
    test_model_runtime \
    test_model_serving_adapter \
	test_model_resident_deployment \
    test_model_resident_ipc \
	test_model_resident_deadline \
	test_model_resident_adapter_budget \
	test_model_resident_session \
    test_model_pipeline_client \
    test_model_batch_engine_mock \
    test_model_batch_engine_mock_debug \
    test_model_pipeline_client_mock \
    test_tp_device_collective_mock \
    test_tp_chain_graph \
    test_tp_allreduce_fuzz \
    test_steploop_admission \
    test_model_api_text \
    test_system_loopback \
    test_child_guard \
    test_tokenizer_sidecar \
    test_pipeline_runtime \
    test_serving_fault_fuzz \
    test_kv_lane_fuzz \
    test_kv_shared_index \
    test_kv_page_cache_lru \
    test_model_resident_end_to_end \
    test_model_resident_reconnect \
    test_distributed_work \
	    test_json \
	    test_quant_arm \
	    test_hidden_transport \
	    test_hidden_transport_rdma_control \
	    test_draft_bridge \
    test_speculation_seam \
    test_row_layout_round_span \
    test_fabric_topology \
    test_memlink \
    test_release \
    test_kv_store \
    test_serving_cache_admission \
    test_kda_reference \
    test_gdn_reference \
    test_numerical_metrics \
    test_kv_cache \
    test_kv_snapshot \
    test_kv_page_layout \
	test_k3_kv_cache \
	test_k3_llm_defines \
	test_k3_run_equivalence \
	test_k3_attach_contract \
	test_k3_pool_sizing \
	test_k3_pack_bind \
	test_k3_serving_adapter \
	test_k3_tp16_serving_adapter \
	test_kv_model_table \
    test_nvme_tier \
	test_continuous_batch \
    test_gemm_tile_k_fallback \
    test_serial_tp_replay \
    test_speculation_tree_resolve \
    test_serving_tp_config \
    test_rope_plan \
    test_glm52_stagepack \
    test_tokenizer \
    test_model_description \
    test_stage_module_common \
    test_weightd \
    test_weightd_lease \
    test_weightd_working_set \
    test_weightd_dense_manifest \
    test_weightd_churn \
    test_weightd_expert_stress \
    test_glm5_next_index_cp_math \
    test_weightd_worker \
    test_host_staged_tcp \
    test_weightd_direct \
    test_weightd_fd_frames \
    test_weightd_attach \
    test_weightd_receipt \
    test_weightd_expert \
    test_stage_module_weightd \
    test_weightd_map \
    test_expert_working_set \
    test_graph_reloc \
    test_weightd_mesh_mock \
    test_module_library \
    test_speculation_reference_draft \
    test_speculation_recorded_draft \
    test_speculation_lookup_draft \
    test_weightd_pacer \
    test_speculation_ngram_draft \
    test_speculation_drafter_mix \
    test_speculation_relay_draft \
    test_speculation_tap \
    test_speculation_relay_link \
    test_speculation_depth \
    test_speculation_frame \
    test_driver_compiler \
    test_orchestrator \
    test_tensor_map_geometry \
    test_tensor_map_encode \
    test_kv_geometry \
    test_mma_fragment_mapping \
    test_weight_codec \
    test_topology_switch \
    test_stage_kv_layout_digest

ifneq ($(UNAME_S),Darwin)
TEST_NAMES += test_stage_kv_binding test_stage_kv_pool_resize test_stage_kv_shared_prefix test_k3_kv_binding
endif

TEST_BINARIES := $(addprefix build/,$(TEST_NAMES))
SHELL_TESTS := \
	tests/fuzz_system_loopback.sh \
	tests/test_deploy_restart_scope.sh
PYTHON_TESTS := \
	tests/test_stray_fixture_processes.py \
	tests/test_kv_binding_async_source.py \
	tests/test_kv_snapshot_law.py \
	tests/test_ab_stats.py \
	tests/test_ab_verdict.py \
	tests/test_ab_arm.py \
	tests/test_ab_receipt.py \
	tests/test_ab_suite_compare.py \
	tests/test_ab_corpus.py \
	tests/test_ab_dry_run.py \
	tests/test_pack_spine_sha.py \
	tests/test_glm5_next_compsec17.py \
	tests/test_compsec17.py \
	tests/test_ling_compsec17.py \
	tests/test_weightd_supervised.py \
	tests/test_weightd_supervision.py \
	tests/test_fleet_agent_doctor.py \
	tests/test_fleet_agent_mesh_exchange.py \
	tests/test_fleet_agent_boot_breaker.py \
	tests/test_roce_qos_contract.py \
	tests/test_fleet_agent_multi_root.py \
	tests/test_ab_fleet.py \
	tests/test_ab_campaign_plans.py \
	tests/test_spark_queue.py \
	tests/test_multi_dev_orchestrate.py \
	tests/test_inference_smoke.py \
	tests/test_gemma4_smoke_manifest.py \
	tests/test_ling_contract_freeze.py \
	tests/test_laguna_reference_fixture.py \
	tests/test_api_stress.py \
	tests/test_code_size.py \
	tests/test_complexity_ceiling.py \
	tests/test_config_coverage.py \
	tests/test_cuda_performance_contracts.py \
	tests/test_dry_law.py \
	tests/test_family_defines_single_source.py \
	tests/test_no_source_comments.py \
	tests/test_third_party_notices.py \
	tests/test_unicode_nfc.py \
	tests/test_tokenizer_unicode_split.py \
	tests/test_fleet_registrar.py \
	tests/test_expert_grouping.py \
	tests/test_ds4_parallel_pxe_rescue.py \
	tests/test_sparkpipe_fsck_health_automount.py \
	tests/test_gemm_k_alignment.py \
	tests/test_glm52_dspark_manifest.py \
	tests/test_glm52_dspark_trace_quality.py \
	tests/test_glm52_pack_fp8_source.py \
	tests/test_glm52_stage_pack.py \
	tests/test_glm52_validate_pack_mirror.py \
	tests/test_glm52_validate_pack_stage.py \
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
	tests/test_k3_checkpoint_contract.py \
	tests/test_k3_driver_contracts.py \
	tests/test_k3_device_combine.py \
	tests/test_k3_stage_ordering.py \
	tests/test_k3_engine.py \
	tests/test_k3_head_rank_host.py \
	tests/test_k3_kv_geometry.py \
	tests/test_k3_layer_host.py \
	tests/test_k3_multidev_lane.py \
	tests/test_k3_pack.py \
	tests/test_k3_pack_layout.py \
	tests/test_k3_quant_recipe.py \
	tests/test_k3_shard.py \
	tests/test_k3_slice_host.py \
	tests/test_k3_kda_rank_heads.py \
	tests/test_k3_kda_row_order.py \
	tests/test_k3_smoke_experts.py \
	tests/test_kda_bf16_state.py \
	tests/test_kda_decay.py \
	tests/test_kda_host.py \
	tests/test_tp_reduce_host.py \
	tests/test_hc_post_host.py \
	tests/test_bf16_rms_host.py \
	tests/test_bf16_conv_host.py \
	tests/test_bf16_sigmoid_host.py \
	tests/test_tp_f32_tree_host.py \
	tests/test_kv_failure_host.py \
	tests/test_frame_error_host.py \
	tests/test_weight_prefetch_host.py \
	tests/test_kernel_frame_error_source.py \
	tests/test_ue8m0_encoder_oracle.py \
	tests/test_kernel_algorithms.py \
	tests/test_kernel_launches.py \
	tests/test_layer_dataflow.py \
	tests/test_layer_host.py \
	tests/test_layer_kinds.py \
	tests/test_measured_status.py \
	tests/test_memory_contracts.py \
	tests/test_generate_model_resident_deployment.py \
	tests/test_deployment_config_drift.py \
	tests/test_k3_deployment_config.py \
	tests/test_model_serving_architecture.py \
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
	tests/test_p3_batched_small_rows_host.py \
	tests/test_glm53full_compsec17.py \
	tests/test_ling_stagepack_resume.py \
	tests/test_mimo26_emit_order.py \
	tests/test_recipe_generation.py \
	tests/test_release_assemble.py \
	tests/test_release_agent.py \
	tests/test_rope_pairing.py \
	tests/test_router_host.py \
	tests/test_router_precision_contract.py \
	tests/test_situ_activation.py \
	tests/test_sources_exist.py \
	tests/test_unlinked_components.py \
	tests/test_staging_manifest.py \
	tests/test_status_truth.py \
	tests/test_site.py \
	tests/test_weightd_manifest.py \
	tests/test_weightd_spine_budget.py \
	tests/test_glm5_next_range_manifest.py \
	tests/test_glm5_next_routed_oracle.py \
	tests/test_glm52_experts_manifest.py \
	tests/test_weightd_lazy_pair.py \
	tests/test_weightd_cli_counts.py \
	tests/test_generated_control_admission.py \
	tests/test_topk_exact_host.py \
	tests/test_glm5_next_graph_regime.py \
	tests/test_glm5_next_verify_regime.py \
	tests/test_spec_verify_bench.py \
	tests/test_spec_tap_dump.py \
	tests/test_glm5_next_spec_replay.py \
	tests/test_spec_offline.py \
	tests/test_spec_bakeoff.py \
	tests/test_spec_bakeoff_corpus.py \
	tests/test_glm5_next_spec_ab.py \
	tests/test_glm5_next_rows_kernels_host.py \
	tests/test_head_sampling_host.py \
	tests/test_skinny_grouped_host.py \
	tests/test_skinny_mxfp4_host.py \
	tests/test_skinny_scale_host.py \
	tests/test_release_rollout.py \
	tests/test_hub_paths_on_workspace.py \
	tests/test_skinny_mxfp4_cuda.py \
	tests/test_kv_shard_host.py \
	tests/test_index_shard_host.py \
	tests/test_kv_shard_cuda.py \
	tests/test_gqa_shard_cuda.py \
	tests/test_head_certified_cuda.py \
	tests/test_latent_gather_prefill_cuda.py \
	tests/test_delta_rule_column_cuda.py \
	tests/test_index_shard_cuda.py \
	tests/test_latent_rope_heads_cuda.py \
	tests/test_project_chain_cuda.py \
	tests/test_topk_warp_cuda.py \
	tests/test_rms_norm_cuda.py \
	tests/test_quantise_rows_cuda.py \
	tests/test_index_select_bound_cuda.py \
	tests/test_mesh_copy_down_cuda.py \
	tests/test_k3_tp_context_cuda.py \
	tests/test_runner_capture_flags_cuda.py \
	tests/test_glm_attach_kv_cuda.py \
	tests/test_topk_exact_cuda.py \
	tests/test_hyper_inject_cuda.py \
	tests/test_layer_norm_cuda.py \
	tests/test_hidden_transport_host_rdma_load.py \
	tests/test_skinny_dependent_cuda.py \
	tests/test_stream_gemm_cuda.py \
	tests/test_attn_prefill_cuda.py \
	tests/test_score_merge.py \
	tests/test_score_export.py \
	tests/test_score_dump_cuda.py \
	tests/test_kernel_codegen_diff.py \
	tests/test_host_codegen_diff.py \
	tests/test_timing_reports.py \
	tests/test_glm5_next_bench_wrap.py \
	tests/test_i27_session.py \
	tests/test_qualification_bundle.py \
	tests/test_glm5_next_expert_pack_layout.py \
	tests/test_glm5_next_expert_shard_math.py \
	tests/test_glm5_next_pack_header_codec.py \
	tests/test_glm5_next_expert_graft.py \
	tests/test_exl3_expert_dequant.py \
	tests/test_glm5_next_pack_tp_all.py \
	tests/test_glm5_next_pack_tool_refusals.py \
	tests/test_glm5_next_nvfp4_spine_scale.py \
	tests/test_checkpoint_row_read.py \
	tests/test_glm5_next_pack_regions.py \
	tests/test_tp_mesh_cancel_lifetime.py \
	tests/test_clamped_up_gate.py \
	tests/test_glm52_expert_graft.py \
	tests/test_glm52_spine_source_verify.py \
	tests/test_glm52_routed_parity.py \
	tests/test_glm5_next_wset_from_trace.py \
	tests/test_acc_parity_oracle.py \
	tests/test_ds4_spark_brickproof.py \
	tests/test_dsv41_flash_geometry.py \
	tests/test_dsv41_flash_kernels.py \
	tests/test_dsv41_official_harness.py \
	tests/test_dsv41_flash_layer.py \
	tests/test_dsv41_flash_layer0_anchor.py \
	tests/test_dsv41_flash_pack_contract.py \
	tests/test_dsv41_flash_smoke_experts.py \
	tests/test_dsv41_flash_stagepack.py \
	tests/test_gemma4_verify_existing.py \
	tests/test_glm52_pack_bf16_passthrough.py \
	tests/test_glm52_pack_nvfp4_passthrough.py \
	tests/test_glm53_contract.py \
	tests/test_draftd_mtp_g8.py \
	tests/test_glm53flash_mtp_reference.py \
	tests/test_spec_recorded_drafts.py \
	tests/test_glm5_next_geometry.py \
	tests/test_head_host.py \
	tests/test_sample_host.py \
	tests/test_chat_frontend_sampling.py \
	tests/test_chat_frontend_control_tokens.py \
	tests/test_chat_frontend_engine_stream.py \
	tests/test_chat_frontend_request_fields.py \
	tests/test_chat_frontend_output_cap.py \
	tests/test_chat_frontend_sampling_defaults.py \
	tests/test_open_webui_github_tool.py \
	tests/test_model_swap_drain_quiet.py \
	tests/test_model_swap_force.py \
	tests/test_model_swap_token.py \
	tests/test_swap_token_shell.py \
	tests/test_chat_frontend_switch_now.py \
	tests/test_fleet_gateway.py \
	tests/test_open_webui_fleet_tool.py \
	tests/test_chat_memory_core.py \
	tests/test_chat_memory_session.py \
	tests/test_chat_memory_frontend.py \
	tests/test_chat_frontend_trace.py \
	tests/test_chat_frontend_citations.py \
	tests/test_chat_frontend_warm_queue.py \
	tests/test_chat_frontend_tagged_calls.py \
	tests/test_chat_frontend_function_xml_calls.py \
	tests/test_model_swap_memory_guard.py \
	tests/test_model_swap_shared_label.py \
	tests/test_model_swap_failures.py \
	tests/test_stage_lane_kv_labels.py \
	tests/test_stage_lane_stop.py \
	tests/test_chat_frontend_object_and_channel_calls.py \
	tests/test_draftd_kernels.py \
	tests/test_draftd_mtp_kv.py \
	tests/test_k3_reshard_cells.py \
	tests/test_laguna_reference_torch.py \
	tests/test_python_tests_registered.py \
	tests/test_spark_pack_common_extents.py \
	tests/test_tokenizer_text_prefix.py \
	tests/test_k3_spec_verify.py \
	tests/test_ling_verify_pack.py \
	tests/test_ling_reference_agreement.py \
	tests/test_mesh_lane_ladder_receipt.py \
	tests/test_mimo26_census.py \
	tests/test_mimo26_model_inputs.py \
	tests/test_mimo26_stagepack.py \
	tests/test_mimo26_stagepack_windows.py \
	tests/test_model_api_queue_lifetime.py \
	tests/test_no_python_in_production.py \
	tests/test_rtx5090_spec_node.py \
	tests/test_safetensors_subset.py \
	tests/test_serving_profile_derivation.py \
	tests/test_spark_ssh_failover.py \
	tests/test_spark_station.py \
	tests/test_spark_tiktoken_compile.py \
	tests/test_t1_reference_decoder.py \
	tests/test_t1_reference_dsa_cache.py \
	tests/test_t1_reference_dsv41.py \
	tests/test_t1_reference_engines.py \
	tests/test_t1_reference_glm53flash.py \
	tests/test_t1_reference_hy4.py \
	tests/test_t1_reference_k3.py \
	tests/test_t1_reference_mimo26.py \
	tests/test_t1_reference_minimax.py \
	tests/test_t1_reference_muse.py \
	tests/test_t1_reference_quarantine.py \
	tests/test_gen_geometry_header.py \
	tests/test_lane_run_log.py \
	tests/test_no_build_defaults.py \
	tests/test_tp_cupti_trace_report.py \
	tests/test_weightd_map_fd_ownership.py \
	tests/test_weightd_mesh_source.py \
	tests/test_mesh_status_tool.py \
	tests/test_weightd_warm_family.py
PYTHON_TEST_BINARIES := build/test_tiktoken_compiled \
	build/weightd_lazy_consumer \
	build/sparkpipe_weightd \
	build/sparkpipe_mesh_status \
	build/dsv41_flash_pack_synthesize \
	build/dsv41_flash_experts_manifest \
	build/test_dsv41_flash_pack_contract
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
TEST_MODEL_SERVING_ADAPTER_MODULE := \
    build/test_modules/libmodel_serving_adapter_module.$(SHARED_LIBRARY_EXT)
TEST_MODEL_SERVING_WITHOUT_PREFIX_REUSE_MODULE := \
    build/test_modules/libmodel_serving_adapter_without_prefix_reuse_module.$(SHARED_LIBRARY_EXT)
TEST_MODEL_SERVING_SPECULATIVE_INLINE_MODULE := \
    build/test_modules/libmodel_serving_adapter_speculative_inline_module.$(SHARED_LIBRARY_EXT)
TEST_MODEL_SERVING_SPECULATIVE_DEFERRED_MODULE := \
    build/test_modules/libmodel_serving_adapter_speculative_deferred_module.$(SHARED_LIBRARY_EXT)
TEST_MODEL_SERVING_SPECULATIVE_VERIFY_MODULE := \
    build/test_modules/libmodel_serving_adapter_speculative_verify_module.$(SHARED_LIBRARY_EXT)
TEST_MODEL_SERVING_SPECULATIVE_LONG_VERIFY_MODULE := \
    build/test_modules/libmodel_serving_adapter_speculative_long_verify_module.$(SHARED_LIBRARY_EXT)
TEST_MODEL_SERVING_RECURRENT_CHECKPOINTS_MODULE := \
    build/test_modules/libmodel_serving_adapter_recurrent_checkpoints_module.$(SHARED_LIBRARY_EXT)
TEST_MODEL_SERVING_ADAPTER_VARIANT_MODULES := \
    $(TEST_MODEL_SERVING_WITHOUT_PREFIX_REUSE_MODULE) \
    $(TEST_MODEL_SERVING_SPECULATIVE_INLINE_MODULE) \
    $(TEST_MODEL_SERVING_SPECULATIVE_DEFERRED_MODULE) \
    $(TEST_MODEL_SERVING_SPECULATIVE_VERIFY_MODULE) \
    $(TEST_MODEL_SERVING_SPECULATIVE_LONG_VERIFY_MODULE) \
    $(TEST_MODEL_SERVING_RECURRENT_CHECKPOINTS_MODULE)
TEST_MODEL_RESIDENT_TRANSPORT_MODULE := \
    build/test_modules/libmodel_resident_transport_module.$(SHARED_LIBRARY_EXT)
TEST_VALIDATOR := build/test_module_validator
TEST_VALIDATOR_CHANGED := build/test_module_validator_identity_changed
.PHONY: all clean test tools hardware_tools hardware_cuda_tools hardware_handoff runtime_completion_tests demo FORCE \
    hidden_transport_spark_host_rdma_verbs \
    hidden_transport_spark_gpudirect_rdma_verbs \
    tree_summary \
    architecture_audit \
    model_driver_contracts

all: $(LIBRARIES) tools $(K3_SERVING_ADAPTER) $(K3_TP16_SERVING_ADAPTER)

tools: $(TOOL_BINARIES)

.PHONY: core model_common deployment audit-boundaries architecture_audit model_driver_contracts non_glm_model_driver_contracts

core: $(CORE_LIBRARY) $(COMPILER_LIBRARY) $(RUNTIME_LIBRARY)

model_common: $(MODEL_COMMON_LIBRARY)

deployment: $(DEPLOYMENT_LIBRARY)

audit-boundaries: core
	python3 tools/audit_core_boundaries.py --repository . --core-archive $(CORE_LIBRARY) --compiler-archive $(COMPILER_LIBRARY) --runtime-archive $(RUNTIME_LIBRARY)

architecture_audit: audit-boundaries

non_glm_model_driver_contracts:

model_driver_contracts: build/test_model_description build/test_stage_module_common non_glm_model_driver_contracts
	./build/test_stage_module_common
	python3 tests/test_model_driver_contracts.py

MODEL_COMMON_LINK_TARGETS := \
    build/sparkpipe_tokenize_prompt \
    build/sparkpipe_nfc \
    build/sparkpipe_split \
    build/sparkpipe_tokenizer_benchmark \
    build/sparkpipe_memlink \
    build/sparkpipe_prevcp \
    build/sparkpipe_nextcp \
    build/test_hidden_transport \
    build/test_draft_bridge \
    build/test_speculation_seam \
    build/test_memlink \
    build/test_kv_store \
    build/test_tokenizer \
    build/test_tokenizer_sidecar \
    $(HIDDEN_TRANSPORT_SPARK_HOST_RDMA) \
    $(HIDDEN_TRANSPORT_SPARK_GPUDIRECT_RDMA)
GLM52_LINK_TARGETS := \
    $(filter build/sparkpipe_glm52_% build/test_glm52_%,$(TOOL_BINARIES) $(TEST_BINARIES)) \
    build/sparkpipe_glm52_batchplane_model \
    build/test_model_description

DEPLOYMENT_LINK_TARGETS := build/sparkpipe_release_manager build/test_release

$(MODEL_COMMON_LINK_TARGETS): COMMON_LIBRARY = $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
$(MODEL_COMMON_LINK_TARGETS): $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
$(GLM52_LINK_TARGETS): COMMON_LIBRARY = $(GLM52_HOST_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
$(GLM52_LINK_TARGETS): $(GLM52_HOST_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)

$(DEPLOYMENT_LINK_TARGETS): COMMON_LIBRARY = $(DEPLOYMENT_LIBRARY) $(CORE_LIBRARY)
$(DEPLOYMENT_LINK_TARGETS): $(DEPLOYMENT_LIBRARY) $(CORE_LIBRARY)

KERNEL_TEST_NVCC = $(NVCC) -std=c++17 $(NVCCFLAGS) -I. -Iinclude -Imodel-families/common/include $(DEFAULT_BATCH_FLAGS)
KERNEL_TEST_CUDA_LINK = -L$(CUDA_HOME)/lib64 -lcudart -lcuda

build:
	mkdir -p build

.PHONY: test-glm-head-offset

.PHONY: test-glm-mtp-join

.PHONY: test-glm-hc-mix

.PHONY: test-glm-l2-prefetch

.PHONY: test-skinny-gemv

.PHONY: test-glm5-next-index-cp

build/test_glm5_next_index_cp_math: tests/test_glm5_next_index_cp_math.c model-families/glm5_next/include/sparkpipe/spark_glm5_next_index_cp.h | build
	$(CC) -I. -Iinclude -Imodel-families/glm5_next/include $(CFLAGS) $< -o $@

build/test_k3_tp16_expert_gemm: tests/test_k3_tp16_expert_gemm.cu runtime/gemm.cuh runtime/tensor_map.h inference/kernels/tile.cuh inference/kernels/gemm.cuh inference/kernels/tensor_map.cuh inference/kernels/formats/mxfp4.cuh | build
	$(NVCC) -std=c++17 $(NVCCFLAGS) -I. -Iinclude -Isrc $< -L$(CUDA_HOME)/lib64 -lcudart -lcuda -o $@

.PHONY: test-k3-tp16-expert-gemm
test-k3-tp16-expert-gemm: build/test_k3_tp16_expert_gemm
	./build/test_k3_tp16_expert_gemm

build/test_k3_interleave_gemm: tests/test_k3_interleave_gemm.cu runtime/gemm.cuh inference/kernels/formats/mxfp4.cuh | build
	$(NVCC) -std=c++17 $(NVCCFLAGS) -I. -Iinclude -Isrc $< -L$(CUDA_HOME)/lib64 -lcudart -lcuda -o $@

build/test_k3_swizzle_probe: tests/test_k3_swizzle_probe.cu runtime/gemm.cuh | build
	$(NVCC) -std=c++17 $(NVCCFLAGS) -I. -Iinclude -Isrc $< -L$(CUDA_HOME)/lib64 -lcudart -lcuda -o $@

.PHONY: test-k3-probes
test-k3-probes: build/test_k3_interleave_gemm build/test_k3_swizzle_probe
	./build/test_k3_interleave_gemm
	./build/test_k3_swizzle_probe

.PHONY: test-glm5-next-kv-shard

.PHONY: test-glm5-next-rows-kernels

build/test_head_candidate_stage: tests/test_head_candidate_stage.cu inference/kernels/head.cuh | build
	$(KERNEL_TEST_NVCC) $< $(KERNEL_TEST_CUDA_LINK) -o $@

.PHONY: test-head-candidate-stage
test-head-candidate-stage: build/test_head_candidate_stage
	./build/test_head_candidate_stage --run

build/test_delta_rule_columns: tests/test_delta_rule_columns.cu inference/kernels/linear_attn.cuh | build
	$(KERNEL_TEST_NVCC) $< $(KERNEL_TEST_CUDA_LINK) -o $@

.PHONY: test-delta-rule-columns
test-delta-rule-columns: build/test_delta_rule_columns
	./build/test_delta_rule_columns --run

build/test_route_build_scan: tests/test_route_build_scan.cu inference/kernels/route.cuh | build
	$(KERNEL_TEST_NVCC) $< $(KERNEL_TEST_CUDA_LINK) -o $@

.PHONY: test-route-build-scan
test-route-build-scan: build/test_route_build_scan
	./build/test_route_build_scan --run

build/test_skinny_cell: tests/test_skinny_cell.cu inference/kernels/skinny.cuh | build
	$(KERNEL_TEST_NVCC) $< $(KERNEL_TEST_CUDA_LINK) -o $@

.PHONY: test-skinny-cell
test-skinny-cell: build/test_skinny_cell
	./build/test_skinny_cell --run

.PHONY: test-latent-attention-split-group

build/test_rms_norm_staged: tests/test_rms_norm_staged.cu inference/kernels/norm.cuh | build
	$(KERNEL_TEST_NVCC) $< $(KERNEL_TEST_CUDA_LINK) -o $@

.PHONY: test-rms-norm-staged
test-rms-norm-staged: build/test_rms_norm_staged
	./build/test_rms_norm_staged --run

build/test_causal_conv_streams: tests/test_causal_conv_streams.cu inference/kernels/linear_attn.cuh | build
	$(KERNEL_TEST_NVCC) $< $(KERNEL_TEST_CUDA_LINK) -o $@

.PHONY: test-causal-conv-streams
test-causal-conv-streams: build/test_causal_conv_streams
	./build/test_causal_conv_streams --run

build/test_skinny_dense_multi: tests/test_skinny_dense_multi.cu inference/kernels/skinny.cuh | build
	$(KERNEL_TEST_NVCC) $< $(KERNEL_TEST_CUDA_LINK) -o $@

.PHONY: test-skinny-dense-multi
test-skinny-dense-multi: build/test_skinny_dense_multi
	./build/test_skinny_dense_multi --run

build/test_skinny_rows_reference: tests/test_skinny_rows_reference.cu inference/kernels/skinny.cuh | build
	$(KERNEL_TEST_NVCC) $< $(KERNEL_TEST_CUDA_LINK) -o $@

.PHONY: test-skinny-rows-reference
test-skinny-rows-reference: build/test_skinny_rows_reference
	./build/test_skinny_rows_reference --run

build/test_skinny_grouped_reference: tests/test_skinny_grouped_reference.cu inference/kernels/skinny.cuh inference/kernels/route.cuh | build
	$(KERNEL_TEST_NVCC) $< $(KERNEL_TEST_CUDA_LINK) -o $@

.PHONY: test-skinny-grouped-reference
test-skinny-grouped-reference: build/test_skinny_grouped_reference
	./build/test_skinny_grouped_reference --run

WORKING_SET_NVCC = $(NVCC) -std=c++17 $(NVCCFLAGS) -I. -Iinclude

build/test_working_set_rollback: tests/test_working_set_rollback.cu runtime/spark_expert_working_set.c include/sparkpipe/spark_expert_working_set.h include/sparkpipe/spark_step_verdict.h inference/kernels/expert_cover.cuh inference/kernels/state_snapshot.cuh include/sparkpipe/spark_state_span.h | build
	$(WORKING_SET_NVCC) tests/test_working_set_rollback.cu runtime/spark_expert_working_set.c -L$(CUDA_HOME)/lib64 -lcudart -o $@

.PHONY: test-working-set-rollback
test-working-set-rollback: build/test_working_set_rollback
	./build/test_working_set_rollback

.PHONY: bench-glm5-next-batch

.PHONY: kernel-codegen-diff

build/test_cuda_stream_receipt: tests/test_cuda_stream_receipt.c $(MODEL_COMMON_LIBRARY) $(RUNTIME_LIBRARY) $(CORE_LIBRARY) | build
	test -f $(CUDA_HOME)/include/cuda_runtime.h
	$(CC) $(MODEL_COMMON_INCLUDE_FLAGS) $(CFLAGS) $^ $(LDFLAGS) $(LDLIBS) $(SPARKPIPE_CUDA_RUNTIME_LINK) $(SPARKPIPE_CUDA_DRIVER_LINK) -o $@

.PHONY: test-cuda-stream-receipt
test-cuda-stream-receipt: build/test_cuda_stream_receipt
	./build/test_cuda_stream_receipt --run

build/test_modules:
	mkdir -p build/test_modules

build/obj/%.o: %.c | build
	@mkdir -p $(dir $@) && $(CC) $(SP_INCLUDE_FLAGS) $(CFLAGS) -fPIC -MMD -MP -c $< -o $@

$(CORE_LIBRARY): $(CORE_OBJECTS)
	$(AR) rcs $@.$$$$.tmp $^ && mv $@.$$$$.tmp $@

$(MODEL_COMMON_LIBRARY): $(MODEL_COMMON_OBJECTS)
	$(AR) rcs $@.$$$$.tmp $^ && mv $@.$$$$.tmp $@

$(TRANSPORT_LIBRARY): $(TRANSPORT_OBJECTS)
	$(AR) rcs $@.$$$$.tmp $^ && mv $@.$$$$.tmp $@

$(DEPLOYMENT_LIBRARY): $(DEPLOYMENT_OBJECTS)
	$(AR) rcs $@.$$$$.tmp $^ && mv $@.$$$$.tmp $@

$(GLM52_HOST_LIBRARY): $(GLM52_HOST_OBJECTS)
	$(AR) rcs $@.$$$$.tmp $^ && mv $@.$$$$.tmp $@

$(COMPILER_LIBRARY): $(COMPILER_OBJECTS)
	$(AR) rcs $@.$$$$.tmp $^ && mv $@.$$$$.tmp $@

$(RUNTIME_LIBRARY): $(RUNTIME_OBJECTS)
	$(AR) rcs $@.$$$$.tmp $^ && mv $@.$$$$.tmp $@

build/sparkpipe_module_publish: tools/sparkpipe_module_publish.c $(COMPILER_LIBRARY) $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(COMPILER_LIBRARY) $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/sparkpipe_model_compile: tools/sparkpipe_model_compile.c $(COMPILER_LIBRARY) $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(COMPILER_LIBRARY) $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/sparkpipe_driver_inspect: tools/sparkpipe_driver_inspect.c $(RUNTIME_LIBRARY) $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(RUNTIME_LIBRARY) $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

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

build/rdma_alltoall_probe: tools/hardware/rdma_alltoall_probe.c
	@mkdir -p build
	$(CC) $(CFLAGS) $< $(LDFLAGS) -libverbs -o $@

hardware_tools: build/spark_model_kernel_characterize build/spark_transport_characterize build/spark_topology_characterize build/spark_pmtu_characterize build/rdma_alltoall_probe

hardware_cuda_tools:
	@if ! command -v $(NVCC) >/dev/null 2>&1; then \
		echo "hardware_cuda_tools skipped: nvcc unavailable"; \
	else \
		set -e; \
		mkdir -p build; \
		$(NVCC) -std=c++17 -O3 -arch=sm_121a -Xptxas=-v -Itools/hardware tools/hardware/spark_cuda_characterize.cu -o build/spark_cuda_characterize; \
		$(NVCC) -std=c++17 -O3 -arch=sm_121a -Xptxas=-v -Itools/hardware -Xcompiler=-pthread tools/hardware/spark_nvme_characterize.cu -o build/spark_nvme_characterize -lpthread; \
		$(MAKE) build/spark_tp_device_collective_characterize; \
	fi

.PHONY: test-tp-f32-arithmetic-gpu
test-tp-f32-arithmetic-gpu: build/tp_f32_arithmetic
	./build/tp_f32_arithmetic

build/tp_mesh_hardware_daemon.o: tests/fixtures/tp_mesh_hardware_daemon.c tests/fixtures/tp_mesh_hardware_fixture.h node/weightd_mesh.c include/sparkpipe/spark_weightd.h | build
	$(CC) -std=c11 -D_GNU_SOURCE -O2 -ffunction-sections -fdata-sections $(MODEL_COMMON_INCLUDE_FLAGS) -c $< -o $@

build/tp_mesh_hardware_probe: tools/tp_mesh_hardware_probe.cu tests/fixtures/tp_mesh_hardware_fixture.h model-families/common/include/sparkpipe/spark_tp_mesh_kernels.cuh model-families/common/include/sparkpipe/spark_tp_mesh_round_control.h build/tp_mesh_hardware_daemon.o | build
	$(NVCC) -std=c++17 $(NVCCFLAGS) $(MODEL_COMMON_INCLUDE_FLAGS) -Xcompiler=-pthread $< build/tp_mesh_hardware_daemon.o -Xlinker --gc-sections -L$(CUDA_HOME)/lib64 -lcuda -libverbs -lmlx5 -o $@

build/tp_f32_arithmetic: tools/hardware/tp_f32_arithmetic.cu inference/kernels/tp_reduce.cuh inference/kernels/dtype.cuh | build
	$(NVCC) -std=c++17 $(NVCCFLAGS) -I. $< -o $@

build/mesh_round_bench: tools/mesh_round_bench.cu runtime/stage_module_common.c $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(RUNTIME_LIBRARY) | build
	$(NVCC) -std=c++17 $(NVCCFLAGS) $(MODEL_COMMON_INCLUDE_FLAGS) -Xcompiler=-pthread $^ -L$(CUDA_HOME)/lib64 -lcuda -ldl -lcudart -o $@

build/spark_tp_device_collective_characterize: tools/hardware/spark_tp_device_collective_characterize.cu $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) | build
	$(NVCC) -std=c++17 $(NVCCFLAGS) $(MODEL_COMMON_INCLUDE_FLAGS) -Xcompiler=-pthread $< $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) -L$(CUDA_HOME)/lib64 -lcudart -lcuda -ldl -lpthread -o $@

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

build/test_kv_page_layout: tests/test_kv_page_layout.c include/sparkpipe/spark_kv_page_store.h | build
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

build/test_kda_reference: tests/test_kda_reference.c include/sparkpipe/spark_kda_reference.h | build
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -o $@

build/test_gdn_reference: model-families/common/validation/spark_gdn_reference.c | build
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -lm -o $@

build/test_numerical_metrics: tests/test_numerical_metrics.c include/sparkpipe/spark_numerical_metrics.h | build
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -lm -o $@

build/test_kv_cache: tests/test_kv_cache.c $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(MODEL_COMMON_INCLUDE_FLAGS) $(CFLAGS) $< $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) $(SPARKPIPE_CUDA_RUNTIME_LINK) -o $@

build/test_kv_snapshot: tests/test_kv_snapshot.c $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(MODEL_COMMON_INCLUDE_FLAGS) $(CFLAGS) $< $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_kv_snapshot_cuda: tests/test_kv_snapshot_cuda.c tests/test_kv_snapshot_cuda_kernels.cu $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) | build
	$(NVCC) -std=c++17 $(NVCCFLAGS) -c tests/test_kv_snapshot_cuda_kernels.cu -o build/test_kv_snapshot_cuda_kernels.o
	$(CC) $(MODEL_COMMON_INCLUDE_FLAGS) $(CFLAGS) tests/test_kv_snapshot_cuda.c build/test_kv_snapshot_cuda_kernels.o $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -L$(CUDA_HOME)/lib64 -lcudart -lstdc++ -o $@

.PHONY: test-kv-snapshot-cuda
test-kv-snapshot-cuda: build/test_kv_snapshot_cuda
	./build/test_kv_snapshot_cuda $(CURDIR)/build

build/test_k3_kv_cache: tests/test_k3_kv_cache.c $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(MODEL_COMMON_INCLUDE_FLAGS) -Imodel-families/k3/include $(CFLAGS) $< $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_k3_llm_defines: tests/test_k3_llm_defines.c inference/llms/kimi_k3/config.h inference/llms/kimi_k3/generated_config.h model-families/k3/include/sparkpipe/spark_k3_llm_defines.h | build
	$(CC) -I. -Iinclude -Imodel-families/k3/include $(CFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

HOST_CUDA_CXX := $(shell for v in 20 19 18 17 16 15 14 13 12 11; do command -v g++-$$v >/dev/null 2>&1 && echo g++-$$v && break; done)
ifeq ($(strip $(HOST_CUDA_CXX)),)
HOST_CUDA_CXX := g++
endif

build/test_k3_attach_contract: tests/test_k3_attach_contract.c include/sparkpipe/spark_stage_runner.h include/sparkpipe/spark_stage_runner_model.h include/sparkpipe/spark_status.h $(K3_SERVING_ADAPTER) | build
	@if [ -f "$(K3_SERVING_ADAPTER)" ]; then $(CC) $(CPPFLAGS) -I. -Iinclude -Imodules/k3_resident_decode_stage/include -DTEST_K3_SERVING_ADAPTER_PATH=\"$(K3_SERVING_ADAPTER)\" $(CFLAGS) $< $(LDFLAGS) -ldl -o $@; else echo "SKIP $@ ($(K3_SERVING_ADAPTER) not built on this host; spark-gated test)"; fi

build/test_k3_pool_sizing: tests/test_k3_pool_sizing.c modules/k3_resident_decode_stage/include/sparkpipe/spark_k3_pool_sizing.h model-families/k3/include/sparkpipe/spark_k3_llm_defines.h | build
	$(CC) -I. -Iinclude -Imodel-families/k3/include -Imodules/k3_resident_decode_stage/include $(CFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

K3_PACK_BIND_SOURCES := runtime/named_pack.c modules/k3_resident_decode_stage/source/spark_k3_pack_load.c modules/k3_resident_decode_stage/source/spark_k3_bind.c modules/k3_resident_decode_stage/source/spark_k3_resident_decode_stage_module.c

build/test_k3_pack_bind: tests/test_k3_pack_bind.c $(K3_PACK_BIND_SOURCES) $(CORE_LIBRARY) | build
	$(CC) $(CORE_INCLUDE_FLAGS) -Imodel-families/common/include -Imodel-families/k3/include -Imodules/k3_resident_decode_stage/include $(CFLAGS) $< $(K3_PACK_BIND_SOURCES) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_k3_serving_adapter: tests/test_k3_serving_adapter.c tests/test_k3_runner_stub.h modules/k3_resident_decode_stage/source/spark_k3_serving_adapter.c inference/runner/stage_serving_adapter.c include/sparkpipe/spark_stage_serving_adapter.h modules/k3_resident_decode_stage/configs/model_resident.json $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) | build
	$(CC) $(CPPFLAGS) -Itests/cuda_stub -Imodules/k3_resident_decode_stage/include $(CFLAGS) -Wno-unused-function -DSPARK_K3_SERVING_TOPOLOGY=404 $< $(SPARKPIPE_HOST_CUDA_STUB_SOURCE) $(SPARKPIPE_TP_DEVICE_TEST_CUDA_STUB_SOURCE) $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) $(SPARKPIPE_CUDA_RUNTIME_LINK) -o $@

build/test_k3_kv_binding: tests/test_k3_kv_binding.c tests/test_k3_runner_stub.h tests/test_weightd_kv_server.h modules/k3_resident_decode_stage/source/spark_k3_serving_adapter.c inference/runner/stage_serving_adapter.c include/sparkpipe/spark_stage_serving_adapter.h $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) | build
	$(CC) $(CPPFLAGS) -Itests/cuda_stub -Imodules/k3_resident_decode_stage/include $(CFLAGS) -Wno-unused-function -DSPARK_K3_SERVING_TOPOLOGY=404 $< $(SPARKPIPE_HOST_CUDA_STUB_SOURCE) $(SPARKPIPE_TP_DEVICE_TEST_CUDA_STUB_SOURCE) $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) $(SPARKPIPE_CUDA_RUNTIME_LINK) -o $@

build/test_k3_tp16_serving_adapter: tests/test_k3_serving_adapter.c tests/test_k3_runner_stub.h modules/k3_resident_decode_stage/source/spark_k3_serving_adapter.c inference/runner/stage_serving_adapter.c include/sparkpipe/spark_stage_serving_adapter.h modules/k3_resident_decode_stage/configs/model_resident_tp16.json $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) | build
	$(CC) $(CPPFLAGS) -Itests/cuda_stub -Imodules/k3_resident_decode_stage/include $(CFLAGS) -Wno-unused-function -DSPARK_K3_SERVING_TOPOLOGY=16 $< $(SPARKPIPE_HOST_CUDA_STUB_SOURCE) $(SPARKPIPE_TP_DEVICE_TEST_CUDA_STUB_SOURCE) $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) $(SPARKPIPE_CUDA_RUNTIME_LINK) -o $@

build/test_k3_run_equivalence: tests/host_cuda/k3_run_equivalence.cu tests/host_cuda/lm_host_cuda.cuh inference/kernels/linear_attn.cuh inference/kernels/norm.cuh inference/kernels/dtype.cuh
	$(HOST_CUDA_CXX) -std=c++17 -O0 -Itests/host_cuda/shim -I. -Itests/host_cuda -Imodel-families/common/include -Iinclude -x c++ $< -o $@

build/test_kv_model_table: tests/test_kv_model_table.c $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(MODEL_COMMON_INCLUDE_FLAGS) $(CFLAGS) $< $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) $(SPARKPIPE_CUDA_RUNTIME_LINK) -o $@

build/sparkpipe_glm52_batchplane_model: tests/studies/sparkpipe_glm52_batchplane_model.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LDFLAGS) $(LDLIBS) -lm -o $@

build/sparkpipe_glm52_tokenize: tools/sparkpipe_glm52_tokenize.c $(GLM52_HOST_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(GLM52_INCLUDE_FLAGS) $(CFLAGS) $< $(GLM52_HOST_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/sparkpipe_tokenize_prompt: tools/sparkpipe_tokenize_prompt.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/sparkpipe_quant_arm: tools/sparkpipe_quant_arm.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/sparkpipe_nfc: tools/sparkpipe_nfc.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/sparkpipe_split: tools/sparkpipe_split.c $(COMMON_LIBRARY)
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

build/sparkpipe_model_residentd: node/model_residentd.c node/weightd_spawn.c node/weightd_spawn.h runtime/model_resident_socket.h $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(MODEL_COMMON_INCLUDE_FLAGS) $(CFLAGS) node/model_residentd.c node/weightd_spawn.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) $(SPARKPIPE_CUDA_RUNTIME_LINK) -o $@

build/sparkpipe_model_batch: node/model_batch.c scheduler/continuous_batch.c include/sparkpipe/spark_continuous_batch.h $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(MODEL_COMMON_INCLUDE_FLAGS) $(CFLAGS) node/model_batch.c scheduler/continuous_batch.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/debug/sparkpipe_model_batch: node/model_batch.c scheduler/continuous_batch.c runtime/model_batch_engine.c include/sparkpipe/spark_continuous_batch.h $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	@mkdir -p $(dir $@)
	$(CC) $(MODEL_COMMON_INCLUDE_FLAGS) $(CFLAGS) -DDEBUG node/model_batch.c scheduler/continuous_batch.c runtime/model_batch_engine.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/sparkpipe_registrar: tools/sparkpipe_registrar.c | build
	$(CC) $(CFLAGS) -D_POSIX_C_SOURCE=200809L -D_DEFAULT_SOURCE tools/sparkpipe_registrar.c $(LDFLAGS) -o $@

build/sparkpipe_model_api: node/model_api.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(MODEL_COMMON_INCLUDE_FLAGS) $(CFLAGS) node/model_api.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -lpthread -o $@

$(HIDDEN_TRANSPORT_SPARK_HOST_RDMA): ring/transport/rdma.cu ring/transport/rdma_control.c include/sparkpipe/spark_hidden_transport.h include/sparkpipe/spark_hidden_transport_rdma_control.h include/sparkpipe/spark_memlink.h include/sparkpipe/spark_weightd.h $(RUNTIME_LIBRARY) $(COMMON_LIBRARY)
	@if ! command -v $(NVCC) >/dev/null 2>&1; then \
		echo "hidden_transport_spark_host_rdma_verbs failed: nvcc unavailable" >&2; exit 1; \
	elif [ ! -f "$(CUDA_HOME)/include/cuda_runtime_api.h" ]; then \
		echo "hidden_transport_spark_host_rdma_verbs failed: CUDA headers unavailable" >&2; exit 1; \
	elif [ ! -f "/usr/include/infiniband/verbs.h" ]; then \
		echo "hidden_transport_spark_host_rdma_verbs failed: libibverbs headers unavailable" >&2; exit 1; \
	else \
		$(NVCC) $(NVCCFLAGS) -DSPARK_HIDDEN_SPARK_RDMA_DEVICE_DIRECT=0 $(SHARED_LIBRARY_FLAGS) -Xcompiler -fPIC -Xcompiler -pthread $(MODEL_COMMON_INCLUDE_FLAGS) ring/transport/rdma.cu ring/transport/rdma_control.c $(RUNTIME_LIBRARY) $(COMMON_LIBRARY) $(LDFLAGS) -L$(CUDA_HOME)/lib64 -lcudart -libverbs -ldl -lpthread -o $@; \
	fi

$(HIDDEN_TRANSPORT_HOST_STAGED_TCP): ring/transport/host_staged_tcp.c include/sparkpipe/spark_hidden_transport.h | build
	$(CC) $(CORE_INCLUDE_FLAGS) -I$(CUDA_HOME)/include $(CFLAGS) -fPIC $(SHARED_LIBRARY_FLAGS) ring/transport/host_staged_tcp.c $(LDFLAGS) -L$(CUDA_HOME)/lib64 -lcudart -lpthread -o $@

build/test_host_staged_tcp: tests/test_host_staged_tcp.c ring/transport/host_staged_tcp.c $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) tests/cuda_stub/cuda_runtime_stub.c | build
	$(CC) $(CORE_INCLUDE_FLAGS) $(CUDA_STUB_INCLUDE_FLAGS) $(CFLAGS) tests/test_host_staged_tcp.c ring/transport/host_staged_tcp.c $(SPARKPIPE_TP_DEVICE_TEST_CUDA_STUB_SOURCE) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

hidden_transport_spark_host_rdma_verbs: $(HIDDEN_TRANSPORT_SPARK_HOST_RDMA)

$(HIDDEN_TRANSPORT_SPARK_GPUDIRECT_RDMA): ring/transport/rdma.cu ring/transport/rdma_control.c include/sparkpipe/spark_hidden_transport.h include/sparkpipe/spark_hidden_transport_rdma_control.h include/sparkpipe/spark_memlink.h include/sparkpipe/spark_weightd.h $(RUNTIME_LIBRARY) $(COMMON_LIBRARY)
	@if ! command -v $(NVCC) >/dev/null 2>&1; then \
		echo "hidden_transport_spark_gpudirect_rdma_verbs failed: nvcc unavailable" >&2; exit 1; \
	elif [ ! -f "$(CUDA_HOME)/include/cuda_runtime_api.h" ]; then \
		echo "hidden_transport_spark_gpudirect_rdma_verbs failed: CUDA headers unavailable" >&2; exit 1; \
	elif [ ! -f "/usr/include/infiniband/verbs.h" ]; then \
		echo "hidden_transport_spark_gpudirect_rdma_verbs failed: libibverbs headers unavailable" >&2; exit 1; \
	else \
		$(NVCC) $(NVCCFLAGS) -DSPARK_HIDDEN_SPARK_RDMA_DEVICE_DIRECT=1 $(SHARED_LIBRARY_FLAGS) -Xcompiler -fPIC -Xcompiler -pthread $(MODEL_COMMON_INCLUDE_FLAGS) ring/transport/rdma.cu ring/transport/rdma_control.c $(RUNTIME_LIBRARY) $(COMMON_LIBRARY) $(LDFLAGS) -L$(CUDA_HOME)/lib64 -lcudart -libverbs -ldl -lpthread -o $@; \
	fi

hidden_transport_spark_gpudirect_rdma_verbs: $(HIDDEN_TRANSPORT_SPARK_GPUDIRECT_RDMA)

FORCE:

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
	$(AR) rcs $@.$$$$.tmp $^ && mv $@.$$$$.tmp $@

$(TEST_HIDDEN_TRANSPORT_MODULE): tests/fixtures/hidden_transport_module.c include/sparkpipe/spark_hidden_transport.h | build/test_modules
	$(CC) $(CPPFLAGS) $(CFLAGS) -fPIC $(SHARED_LIBRARY_FLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

$(TEST_MODEL_SERVING_ADAPTER_MODULE): tests/fixtures/model_serving_adapter_module.c include/sparkpipe/spark_model_serving_adapter.h $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) | build/test_modules
	$(CC) $(CPPFLAGS) $(CFLAGS) -fPIC $(SHARED_LIBRARY_FLAGS) $< $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

$(TEST_MODEL_SERVING_WITHOUT_PREFIX_REUSE_MODULE): tests/fixtures/model_serving_adapter_module.c include/sparkpipe/spark_model_serving_adapter.h $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) | build/test_modules
	$(CC) $(CPPFLAGS) -DTEST_MODEL_SERVING_WITHOUT_PREFIX_REUSE $(CFLAGS) -fPIC $(SHARED_LIBRARY_FLAGS) $< $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

$(TEST_MODEL_SERVING_SPECULATIVE_INLINE_MODULE): tests/fixtures/model_serving_adapter_module.c include/sparkpipe/spark_model_serving_adapter.h $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) | build/test_modules
	$(CC) $(CPPFLAGS) -DTEST_MODEL_SERVING_SPECULATIVE_INLINE_CHECKPOINTS $(CFLAGS) -fPIC $(SHARED_LIBRARY_FLAGS) $< $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

$(TEST_MODEL_SERVING_SPECULATIVE_DEFERRED_MODULE): tests/fixtures/model_serving_adapter_module.c include/sparkpipe/spark_model_serving_adapter.h $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) | build/test_modules
	$(CC) $(CPPFLAGS) -DTEST_MODEL_SERVING_SPECULATIVE_DEFERRED_CHECKPOINTS $(CFLAGS) -fPIC $(SHARED_LIBRARY_FLAGS) $< $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

$(TEST_MODEL_SERVING_SPECULATIVE_VERIFY_MODULE): tests/fixtures/model_serving_adapter_module.c include/sparkpipe/spark_model_serving_adapter.h $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) | build/test_modules
	$(CC) $(CPPFLAGS) -DTEST_MODEL_SERVING_SPECULATIVE_VERIFY $(CFLAGS) -fPIC $(SHARED_LIBRARY_FLAGS) $< $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

$(TEST_MODEL_SERVING_SPECULATIVE_LONG_VERIFY_MODULE): tests/fixtures/model_serving_adapter_module.c include/sparkpipe/spark_model_serving_adapter.h $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) | build/test_modules
	$(CC) $(CPPFLAGS) -DTEST_MODEL_SERVING_SPECULATIVE_LONG_VERIFY $(CFLAGS) -fPIC $(SHARED_LIBRARY_FLAGS) $< $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

$(TEST_MODEL_SERVING_RECURRENT_CHECKPOINTS_MODULE): tests/fixtures/model_serving_adapter_module.c include/sparkpipe/spark_model_serving_adapter.h $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) | build/test_modules
	$(CC) $(CPPFLAGS) -DTEST_MODEL_SERVING_RECURRENT_CHECKPOINTS $(CFLAGS) -fPIC $(SHARED_LIBRARY_FLAGS) $< $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

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

build/test_score_dump_keys: tests/test_score_dump_keys.c src/spark_score_dump.c include/sparkpipe/spark_score_dump.h $(CORE_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_score_dump_keys.c src/spark_score_dump.c $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_work_transaction: tests/test_work_transaction.c $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_work_transaction.c $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_runtime_completion: tests/test_runtime_completion.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_runtime_completion.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_model_runtime: tests/test_model_runtime.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_model_runtime.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_serving_cache_admission: tests/test_serving_cache_admission.c include/sparkpipe/spark_serving_cache_admission.h $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_model_serving_adapter: tests/test_model_serving_adapter.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(TEST_MODEL_SERVING_ADAPTER_MODULE)
	$(CC) $(CPPFLAGS) -DTEST_MODEL_SERVING_ADAPTER_MODULE_PATH=\"$(TEST_MODEL_SERVING_ADAPTER_MODULE)\" $(CFLAGS) tests/test_model_serving_adapter.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/probe_adapter: tools/devcycle/probe_adapter.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_model_resident_deployment: tests/test_model_resident_deployment.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_model_resident_deployment.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_model_resident_ipc: tests/test_model_resident_ipc.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_model_resident_ipc.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_model_resident_deadline: tests/test_model_resident_deadline.c node/model_residentd.c node/weightd_spawn.c node/weightd_spawn.h runtime/model_resident_socket.h $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(MODEL_COMMON_INCLUDE_FLAGS) $(CFLAGS) tests/test_model_resident_deadline.c node/weightd_spawn.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) $(SPARKPIPE_CUDA_RUNTIME_LINK) -o $@

build/test_model_resident_adapter_budget: tests/test_model_resident_adapter_budget.c node/model_residentd.c node/weightd_spawn.c node/weightd_spawn.h runtime/model_resident_socket.h $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(MODEL_COMMON_INCLUDE_FLAGS) $(CFLAGS) tests/test_model_resident_adapter_budget.c node/weightd_spawn.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) $(SPARKPIPE_CUDA_RUNTIME_LINK) -o $@

build/test_memory_buffer: tests/test_memory_buffer.c runtime/memory_buffer.c include/sparkpipe/spark_memory_buffer.h tests/cuda_stub/cuda_runtime_stub.c | build
	$(CC) $(CORE_INCLUDE_FLAGS) $(CUDA_STUB_INCLUDE_FLAGS) $(CFLAGS) tests/test_memory_buffer.c tests/cuda_stub/cuda_runtime_stub.c $(LDFLAGS) $(LDLIBS) -o $@

build/test_model_resident_session: tests/test_model_resident_session.c node/model_residentd.c node/weightd_spawn.c node/weightd_spawn.h runtime/model_resident_socket.h $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(MODEL_COMMON_INCLUDE_FLAGS) $(CFLAGS) tests/test_model_resident_session.c node/weightd_spawn.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) $(SPARKPIPE_CUDA_RUNTIME_LINK) -o $@

build/test_model_pipeline_client: tests/test_model_pipeline_client.c tests/fixtures/model_resident_deployment_fixture.c tests/fixtures/test_child_guard.c tests/fixtures/model_serving_adapter_config.json build/sparkpipe_model_residentd build/sparkpipe_model_batch $(TEST_MODEL_SERVING_ADAPTER_MODULE) $(TEST_MODEL_RESIDENT_TRANSPORT_MODULE) $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(CPPFLAGS) -DTEST_MODEL_RESIDENTD_PATH=\"build/sparkpipe_model_residentd\" -DTEST_MODEL_BATCH_PATH=\"build/sparkpipe_model_batch\" -DTEST_MODEL_SERVING_ADAPTER_PATH=\"$(TEST_MODEL_SERVING_ADAPTER_MODULE)\" -DTEST_MODEL_RESIDENT_TRANSPORT_PATH=\"$(TEST_MODEL_RESIDENT_TRANSPORT_MODULE)\" $(CFLAGS) tests/test_model_pipeline_client.c tests/fixtures/model_resident_deployment_fixture.c tests/fixtures/test_child_guard.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_model_batch_engine_mock: tests/test_model_batch_engine_mock.c tests/mock_model_resident_client.c tests/fixtures/model_resident_deployment_fixture.c $(TEST_MODEL_SERVING_ADAPTER_MODULE) $(TEST_MODEL_SERVING_ADAPTER_VARIANT_MODULES) $(TEST_MODEL_RESIDENT_TRANSPORT_MODULE) $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(CPPFLAGS) -DTEST_MODEL_SERVING_ADAPTER_PATH=\"$(TEST_MODEL_SERVING_ADAPTER_MODULE)\" -DTEST_MODEL_SERVING_WITHOUT_PREFIX_REUSE_PATH=\"$(TEST_MODEL_SERVING_WITHOUT_PREFIX_REUSE_MODULE)\" -DTEST_MODEL_SERVING_SPECULATIVE_INLINE_PATH=\"$(TEST_MODEL_SERVING_SPECULATIVE_INLINE_MODULE)\" -DTEST_MODEL_SERVING_SPECULATIVE_DEFERRED_PATH=\"$(TEST_MODEL_SERVING_SPECULATIVE_DEFERRED_MODULE)\" -DTEST_MODEL_SERVING_SPECULATIVE_VERIFY_PATH=\"$(TEST_MODEL_SERVING_SPECULATIVE_VERIFY_MODULE)\" -DTEST_MODEL_SERVING_SPECULATIVE_LONG_VERIFY_PATH=\"$(TEST_MODEL_SERVING_SPECULATIVE_LONG_VERIFY_MODULE)\" -DTEST_MODEL_SERVING_RECURRENT_CHECKPOINTS_PATH=\"$(TEST_MODEL_SERVING_RECURRENT_CHECKPOINTS_MODULE)\" -DTEST_MODEL_RESIDENT_TRANSPORT_PATH=\"$(TEST_MODEL_RESIDENT_TRANSPORT_MODULE)\" $(CFLAGS) tests/test_model_batch_engine_mock.c tests/mock_model_resident_client.c tests/fixtures/model_resident_deployment_fixture.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_model_batch_engine_mock_debug: tests/test_model_batch_engine_mock.c runtime/model_batch_engine.c tests/mock_model_resident_client.c tests/fixtures/model_resident_deployment_fixture.c $(TEST_MODEL_SERVING_ADAPTER_MODULE) $(TEST_MODEL_SERVING_ADAPTER_VARIANT_MODULES) $(TEST_MODEL_RESIDENT_TRANSPORT_MODULE) $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(CPPFLAGS) -DTEST_MODEL_SERVING_ADAPTER_PATH=\"$(TEST_MODEL_SERVING_ADAPTER_MODULE)\" -DTEST_MODEL_SERVING_WITHOUT_PREFIX_REUSE_PATH=\"$(TEST_MODEL_SERVING_WITHOUT_PREFIX_REUSE_MODULE)\" -DTEST_MODEL_SERVING_SPECULATIVE_INLINE_PATH=\"$(TEST_MODEL_SERVING_SPECULATIVE_INLINE_MODULE)\" -DTEST_MODEL_SERVING_SPECULATIVE_DEFERRED_PATH=\"$(TEST_MODEL_SERVING_SPECULATIVE_DEFERRED_MODULE)\" -DTEST_MODEL_SERVING_SPECULATIVE_VERIFY_PATH=\"$(TEST_MODEL_SERVING_SPECULATIVE_VERIFY_MODULE)\" -DTEST_MODEL_SERVING_SPECULATIVE_LONG_VERIFY_PATH=\"$(TEST_MODEL_SERVING_SPECULATIVE_LONG_VERIFY_MODULE)\" -DTEST_MODEL_SERVING_RECURRENT_CHECKPOINTS_PATH=\"$(TEST_MODEL_SERVING_RECURRENT_CHECKPOINTS_MODULE)\" -DTEST_MODEL_RESIDENT_TRANSPORT_PATH=\"$(TEST_MODEL_RESIDENT_TRANSPORT_MODULE)\" $(CFLAGS) -DDEBUG tests/test_model_batch_engine_mock.c runtime/model_batch_engine.c tests/mock_model_resident_client.c tests/fixtures/model_resident_deployment_fixture.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_tp_chain_graph: tests/test_tp_chain_graph.c include/sparkpipe/spark_tp_chain_graph.h | build
	$(CC) $(CPPFLAGS) $(CUDA_STUB_INCLUDE_FLAGS) $(CFLAGS) tests/test_tp_chain_graph.c $(LDFLAGS) -o $@

build/test_tp_device_collective_mock: tests/test_tp_device_collective_mock.c ring/transport/tp_device_collective.c tests/cuda_stub/cuda_runtime_stub.c $(CORE_LIBRARY)
	$(CC) $(CPPFLAGS) $(CUDA_STUB_INCLUDE_FLAGS) $(CFLAGS) tests/test_tp_device_collective_mock.c ring/transport/tp_device_collective.c tests/cuda_stub/cuda_runtime_stub.c $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_tp_allreduce_fuzz: tests/test_tp_allreduce_fuzz.c ring/transport/tp_device_collective.c tests/cuda_stub/cuda_runtime_stub.c $(CORE_LIBRARY)
	$(CC) $(CPPFLAGS) $(CUDA_STUB_INCLUDE_FLAGS) $(CFLAGS) tests/test_tp_allreduce_fuzz.c tests/cuda_stub/cuda_runtime_stub.c $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_kv_lane_fuzz build/test_kv_page_cache_lru build/test_kv_shared_index: build/test_%: tests/test_%.c build/libsparkpipe_model_common.a build/libsparkpipe_core.a
	$(CC) $(CPPFLAGS) $(CFLAGS) $< build/libsparkpipe_model_common.a build/libsparkpipe_core.a $(LDFLAGS) $(LDLIBS) -o $@

build/test_serving_fault_fuzz: tests/test_serving_fault_fuzz.c tests/mock_model_resident_client.c tests/fixtures/model_resident_deployment_fixture.c $(TEST_MODEL_SERVING_ADAPTER_MODULE) $(TEST_MODEL_RESIDENT_TRANSPORT_MODULE) $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(CPPFLAGS) -DTEST_MODEL_SERVING_ADAPTER_PATH=\"$(TEST_MODEL_SERVING_ADAPTER_MODULE)\" -DTEST_MODEL_RESIDENT_TRANSPORT_PATH=\"$(TEST_MODEL_RESIDENT_TRANSPORT_MODULE)\" $(CFLAGS) tests/test_serving_fault_fuzz.c tests/mock_model_resident_client.c tests/fixtures/model_resident_deployment_fixture.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_model_pipeline_client_mock: tests/test_model_pipeline_client_mock.c tests/mock_model_resident_client.c tests/fixtures/model_resident_deployment_fixture.c $(TEST_MODEL_SERVING_ADAPTER_MODULE) $(TEST_MODEL_RESIDENT_TRANSPORT_MODULE) $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(CPPFLAGS) -DTEST_MODEL_SERVING_ADAPTER_PATH=\"$(TEST_MODEL_SERVING_ADAPTER_MODULE)\" -DTEST_MODEL_RESIDENT_TRANSPORT_PATH=\"$(TEST_MODEL_RESIDENT_TRANSPORT_MODULE)\" $(CFLAGS) tests/test_model_pipeline_client_mock.c tests/mock_model_resident_client.c tests/fixtures/model_resident_deployment_fixture.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_steploop_admission: tests/test_steploop_admission.c tests/fixtures/model_resident_deployment_fixture.c tests/fixtures/test_child_guard.c tests/fixtures/model_serving_adapter_config.json tests/fixtures/model_serving_adapter_config_hold.json build/sparkpipe_model_residentd $(TEST_MODEL_SERVING_ADAPTER_MODULE) $(TEST_MODEL_RESIDENT_TRANSPORT_MODULE) $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(CPPFLAGS) -DTEST_MODEL_RESIDENTD_PATH=\"build/sparkpipe_model_residentd\" -DTEST_MODEL_SERVING_ADAPTER_PATH=\"$(TEST_MODEL_SERVING_ADAPTER_MODULE)\" -DTEST_MODEL_RESIDENT_TRANSPORT_PATH=\"$(TEST_MODEL_RESIDENT_TRANSPORT_MODULE)\" $(CFLAGS) tests/test_steploop_admission.c tests/fixtures/model_resident_deployment_fixture.c tests/fixtures/test_child_guard.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_pipeline_runtime: tests/test_pipeline_runtime.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_pipeline_runtime.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

$(K3_SERVING_ADAPTER): modules/k3_resident_decode_stage/source/spark_k3_serving_adapter.c inference/runner/stage_serving_adapter.c inference/runner/stage_runner.cu modules/k3_resident_decode_stage/source/spark_k3_stage_model.cu modules/k3_resident_decode_stage/source/spark_k3_resident_decode_stage_cuda.cu modules/k3_resident_decode_stage/include/sparkpipe/spark_k3_serving_adapter.h include/sparkpipe/spark_stage_runner.h include/sparkpipe/spark_stage_runner_model.h include/sparkpipe/spark_stage_serving_adapter.h $(wildcard inference/llms/kimi_k3/*.cu inference/llms/kimi_k3/*.cuh inference/llms/kimi_k3/*.h inference/kernels/*.cuh inference/kernels/*.h modules/k3_resident_decode_stage/source/*.c modules/k3_resident_decode_stage/source/*.h modules/k3_resident_decode_stage/include/sparkpipe/*.h model-families/k3/include/sparkpipe/*.h) $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) | build
	@if command -v $(NVCC) >/dev/null 2>&1; then $(NVCC) $(NVCCFLAGS) -DSPARK_K3_SERVING_TOPOLOGY=404 -I. -Iinclude -Isrc -Imodules/k3_resident_decode_stage/include -Imodel-families/common/include -Imodel-families/k3/include -Xcompiler -fPIC $(SHARED_LIBRARY_FLAGS) modules/k3_resident_decode_stage/source/spark_k3_serving_adapter.c inference/runner/stage_serving_adapter.c inference/runner/stage_runner.cu modules/k3_resident_decode_stage/source/spark_k3_stage_model.cu modules/k3_resident_decode_stage/source/spark_k3_resident_decode_stage_cuda.cu inference/llms/kimi_k3/bind.cu inference/llms/kimi_k3/unity.cu modules/k3_resident_decode_stage/source/spark_k3_pack_load.c modules/k3_resident_decode_stage/source/spark_k3_bind.c modules/k3_resident_decode_stage/source/spark_k3_resident_decode_stage_module.c runtime/json.c runtime/filesystem.c src/spark_status.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) -Xcompiler -pthread -ldl $(SPARKPIPE_CUDA_RUNTIME_LINK) -lcuda -o $@; else echo "SKIP $@ (nvcc unavailable on this host; spark-gated artifact)"; fi

build/runner_capture_flags_cuda: tests/cuda/runner_capture_flags_cuda.cu inference/runner/stage_runner.cu $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) | build
	$(NVCC) $(NVCCFLAGS) -I. -Iinclude -Isrc -Imodel-families/common/include tests/cuda/runner_capture_flags_cuda.cu runtime/json.c runtime/filesystem.c src/spark_status.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) -Xcompiler -pthread -ldl $(SPARKPIPE_CUDA_RUNTIME_LINK) -lcuda -o $@

build/k3_tp_context_cuda: tests/cuda/k3_tp_context_cuda.cu modules/k3_resident_decode_stage/source/spark_k3_stage_model.cu modules/k3_resident_decode_stage/source/spark_k3_resident_decode_stage_cuda.cu inference/llms/kimi_k3/bind.cu inference/llms/kimi_k3/unity.cu $(K3_PACK_BIND_SOURCES) $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) | build
	$(NVCC) $(NVCCFLAGS) -DSPARK_K3_SERVING_TOPOLOGY=404 -I. -Iinclude -Isrc -Imodules/k3_resident_decode_stage/include -Imodel-families/common/include -Imodel-families/k3/include tests/cuda/k3_tp_context_cuda.cu modules/k3_resident_decode_stage/source/spark_k3_resident_decode_stage_cuda.cu inference/llms/kimi_k3/bind.cu inference/llms/kimi_k3/unity.cu modules/k3_resident_decode_stage/source/spark_k3_pack_load.c modules/k3_resident_decode_stage/source/spark_k3_bind.c modules/k3_resident_decode_stage/source/spark_k3_resident_decode_stage_module.c runtime/json.c runtime/filesystem.c src/spark_status.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) -Xcompiler -pthread -ldl $(SPARKPIPE_CUDA_RUNTIME_LINK) -lcuda -o $@

$(K3_TP16_SERVING_ADAPTER): modules/k3_resident_decode_stage/source/spark_k3_serving_adapter.c inference/runner/stage_serving_adapter.c inference/runner/stage_runner.cu modules/k3_resident_decode_stage/source/spark_k3_stage_model.cu modules/k3_resident_decode_stage/source/spark_k3_resident_decode_stage_cuda.cu modules/k3_resident_decode_stage/include/sparkpipe/spark_k3_serving_adapter.h include/sparkpipe/spark_stage_runner.h include/sparkpipe/spark_stage_runner_model.h include/sparkpipe/spark_stage_serving_adapter.h $(wildcard inference/llms/kimi_k3/*.cu inference/llms/kimi_k3/*.cuh inference/llms/kimi_k3/*.h inference/kernels/*.cuh inference/kernels/*.h modules/k3_resident_decode_stage/source/*.c modules/k3_resident_decode_stage/source/*.h modules/k3_resident_decode_stage/include/sparkpipe/*.h model-families/k3/include/sparkpipe/*.h) $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) | build
	@if command -v $(NVCC) >/dev/null 2>&1; then $(NVCC) $(NVCCFLAGS) -DSPARK_K3_SERVING_TOPOLOGY=16 -I. -Iinclude -Isrc -Imodules/k3_resident_decode_stage/include -Imodel-families/common/include -Imodel-families/k3/include -Xcompiler -fPIC $(SHARED_LIBRARY_FLAGS) modules/k3_resident_decode_stage/source/spark_k3_serving_adapter.c inference/runner/stage_serving_adapter.c inference/runner/stage_runner.cu modules/k3_resident_decode_stage/source/spark_k3_stage_model.cu modules/k3_resident_decode_stage/source/spark_k3_resident_decode_stage_cuda.cu inference/llms/kimi_k3/bind.cu inference/llms/kimi_k3/unity.cu modules/k3_resident_decode_stage/source/spark_k3_pack_load.c modules/k3_resident_decode_stage/source/spark_k3_bind.c modules/k3_resident_decode_stage/source/spark_k3_resident_decode_stage_module.c runtime/json.c runtime/filesystem.c src/spark_status.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) -Xcompiler -pthread -ldl $(SPARKPIPE_CUDA_RUNTIME_LINK) -lcuda -o $@; else echo "SKIP $@ (nvcc unavailable on this host; spark-gated artifact)"; fi

define QWEN38_27B_TP16_LINK
	@if command -v $(NVCC) >/dev/null 2>&1; then $(NVCC) $(NVCCFLAGS) $(1) -DQWEN38_27B_CONTRACT_SHA256=\"$(QWEN38_27B_TP16_CONTRACT_SHA256)\" -I. -Iinclude -Isrc -Imodules/qwen38_27b_resident_decode_stage/include -Imodel-families/qwen38/source -Imodel-families/common/include -Xcompiler -fPIC $(SHARED_LIBRARY_FLAGS) $(QWEN38_27B_TP16_SOURCES) runtime/json.c runtime/filesystem.c src/spark_status.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) -Xcompiler -pthread -ldl $(SPARKPIPE_CUDA_RUNTIME_LINK) -lcuda -o $@; else echo "SKIP $@ (nvcc unavailable on this host; spark-gated artifact)"; fi
endef

$(QWEN38_27B_TP16_SERVING_ADAPTER): $(QWEN38_27B_TP16_SOURCES) $(QWEN38_27B_TP16_CONTRACT) modules/qwen38_27b_resident_decode_stage/include/sparkpipe/spark_qwen38_27b_stage_model.h model-families/qwen38/source/qwen38_hybrid_stage.cuh include/sparkpipe/spark_stage_runner.h include/sparkpipe/spark_stage_runner_model.h include/sparkpipe/spark_stage_serving_adapter.h $(wildcard inference/kernels/*.cuh inference/kernels/*.h) $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) | build
	$(call QWEN38_27B_TP16_LINK,)

$(QWEN38_27B_BF16_TP16_SERVING_ADAPTER): $(QWEN38_27B_TP16_SOURCES) $(QWEN38_27B_TP16_CONTRACT) modules/qwen38_27b_resident_decode_stage/include/sparkpipe/spark_qwen38_27b_stage_model.h model-families/qwen38/source/qwen38_hybrid_stage.cuh include/sparkpipe/spark_stage_runner.h include/sparkpipe/spark_stage_runner_model.h include/sparkpipe/spark_stage_serving_adapter.h $(wildcard inference/kernels/*.cuh inference/kernels/*.h) $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) | build
	$(call QWEN38_27B_TP16_LINK,-DQWEN38_27B_BF16=1)

$(QWEN38_MAX_TP16_SERVING_ADAPTER): $(QWEN38_MAX_TP16_SOURCES) $(QWEN38_MAX_TP16_CONTRACT) modules/qwen38_max_resident_decode_stage/include/sparkpipe/spark_qwen38_max_stage_model.h model-families/qwen38/source/qwen38_hybrid_stage.cuh include/sparkpipe/spark_stage_runner.h include/sparkpipe/spark_stage_runner_model.h include/sparkpipe/spark_stage_serving_adapter.h $(wildcard inference/kernels/*.cuh inference/kernels/*.h) $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) | build
	@if command -v $(NVCC) >/dev/null 2>&1; then $(NVCC) $(NVCCFLAGS) -DQWEN38_MAX_CONTRACT_SHA256=\"$(QWEN38_MAX_TP16_CONTRACT_SHA256)\" -I. -Iinclude -Isrc -Imodules/qwen38_max_resident_decode_stage/include -Imodel-families/qwen38/source -Imodel-families/common/include -Xcompiler -fPIC $(SHARED_LIBRARY_FLAGS) $(QWEN38_MAX_TP16_SOURCES) runtime/json.c runtime/filesystem.c src/spark_status.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) -Xcompiler -pthread -ldl $(SPARKPIPE_CUDA_RUNTIME_LINK) -lcuda -o $@; else echo "SKIP $@ (nvcc unavailable on this host; spark-gated artifact)"; fi

$(QWEN38_FLASH_TP16_SERVING_ADAPTER): $(QWEN38_FLASH_TP16_SOURCES) $(QWEN38_FLASH_TP16_CONTRACT) modules/qwen38_flash_resident_decode_stage/include/sparkpipe/spark_qwen38_flash_stage_model.h model-families/qwen38/source/qwen38_hybrid_stage.cuh include/sparkpipe/spark_stage_runner.h include/sparkpipe/spark_stage_runner_model.h include/sparkpipe/spark_stage_serving_adapter.h $(wildcard inference/kernels/*.cuh inference/kernels/*.h) $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) | build
	@if command -v $(NVCC) >/dev/null 2>&1; then $(NVCC) $(NVCCFLAGS) -DQWEN38_FLASH_CONTRACT_SHA256=\"$(QWEN38_FLASH_TP16_CONTRACT_SHA256)\" -I. -Iinclude -Isrc -Imodules/qwen38_flash_resident_decode_stage/include -Imodel-families/qwen38/source -Imodel-families/common/include -Xcompiler -fPIC $(SHARED_LIBRARY_FLAGS) $(QWEN38_FLASH_TP16_SOURCES) runtime/json.c runtime/filesystem.c src/spark_status.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) -Xcompiler -pthread -ldl $(SPARKPIPE_CUDA_RUNTIME_LINK) -lcuda -o $@; else echo "SKIP $@ (nvcc unavailable on this host; spark-gated artifact)"; fi

$(GEMMA4_31B_TP16_SERVING_ADAPTER): $(GEMMA4_31B_TP16_SOURCES) $(GEMMA4_31B_TP16_CONTRACT) modules/gemma4_31b_resident_decode_stage/include/sparkpipe/spark_gemma4_31b_stage_model.h model-families/gemma4/source/gemma4_stage.cuh include/sparkpipe/spark_stage_runner.h include/sparkpipe/spark_stage_runner_model.h include/sparkpipe/spark_stage_serving_adapter.h $(wildcard inference/kernels/*.cuh inference/kernels/*.h) $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) | build
	@if command -v $(NVCC) >/dev/null 2>&1; then $(NVCC) $(NVCCFLAGS) -DGEMMA4_31B_CONTRACT_SHA256=\"$(GEMMA4_31B_TP16_CONTRACT_SHA256)\" -I. -Iinclude -Isrc -Imodules/gemma4_31b_resident_decode_stage/include -Imodel-families/gemma4/source -Imodel-families/common/include -Xcompiler -fPIC $(SHARED_LIBRARY_FLAGS) $(GEMMA4_31B_TP16_SOURCES) runtime/json.c runtime/filesystem.c src/spark_status.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) -Xcompiler -pthread -ldl $(SPARKPIPE_CUDA_RUNTIME_LINK) -lcuda -o $@; else echo "SKIP $@ (nvcc unavailable on this host; spark-gated artifact)"; fi

$(MUSE_TP16_SERVING_ADAPTER): $(MUSE_TP16_SOURCES) $(MUSE_TP16_CONTRACT) modules/muse_glimmer_tp16_resident_decode_stage/include/sparkpipe/spark_muse_tp16_stage_model.h model-families/gemma4/source/gemma4_stage.cuh include/sparkpipe/spark_stage_runner.h include/sparkpipe/spark_stage_runner_model.h include/sparkpipe/spark_stage_serving_adapter.h $(wildcard inference/kernels/*.cuh inference/kernels/*.h) $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) | build
	@if command -v $(NVCC) >/dev/null 2>&1; then $(NVCC) $(NVCCFLAGS) -DMUSE_TP16_CONTRACT_SHA256=\"$(MUSE_TP16_CONTRACT_SHA256)\" -I. -Iinclude -Isrc -Imodules/muse_glimmer_tp16_resident_decode_stage/include -Imodel-families/gemma4/source -Imodel-families/common/include -Xcompiler -fPIC $(SHARED_LIBRARY_FLAGS) $(MUSE_TP16_SOURCES) runtime/json.c runtime/filesystem.c src/spark_status.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) -Xcompiler -pthread -ldl $(SPARKPIPE_CUDA_RUNTIME_LINK) -lcuda -o $@; else echo "SKIP $@ (nvcc unavailable on this host; spark-gated artifact)"; fi
$(GLM53_FLASH_TP16_SERVING_ADAPTER): $(GLM53_FLASH_TP16_SOURCES) $(GLM53_FLASH_TP16_CONTRACT) modules/glm53_flash_tp16_resident_decode_stage/include/sparkpipe/spark_glm53_flash_tp16_stage_model.h model-families/glm5_next/source/glm5_next_stage.cuh model-families/glm5_next/source/glm5_next_stage_model.cuh model-families/glm5_next/include/sparkpipe/spark_glm5_next_stagepack.h include/sparkpipe/spark_stage_runner.h include/sparkpipe/spark_stage_runner_model.h include/sparkpipe/spark_stage_serving_adapter.h $(wildcard inference/kernels/*.cuh inference/kernels/*.h) $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) | build
	@if command -v $(NVCC) >/dev/null 2>&1; then $(NVCC) $(NVCCFLAGS) -DGLM53_FLASH_TP16_CONTRACT_SHA256=\"$(GLM53_FLASH_TP16_CONTRACT_SHA256)\" -I. -Iinclude -Isrc -Imodules/glm53_flash_tp16_resident_decode_stage/include -Imodel-families/glm5_next/source -Imodel-families/glm5_next/include -Imodel-families/common/include -Xcompiler -fPIC $(SHARED_LIBRARY_FLAGS) $(GLM53_FLASH_TP16_SOURCES) runtime/json.c runtime/filesystem.c src/spark_status.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) -Xcompiler -pthread -ldl $(SPARKPIPE_CUDA_RUNTIME_LINK) -lcuda -o $@; else echo "SKIP $@ (nvcc unavailable on this host; spark-gated artifact)"; fi

$(LAGUNA_TP16_SERVING_ADAPTER): $(LAGUNA_TP16_SOURCES) $(LAGUNA_TP16_CONTRACT) modules/laguna_tp16_resident_decode_stage/include/sparkpipe/spark_laguna_tp16_stage_model.h model-families/gemma4/source/gemma4_stage.cuh include/sparkpipe/spark_stage_runner.h include/sparkpipe/spark_stage_runner_model.h include/sparkpipe/spark_stage_serving_adapter.h $(wildcard inference/kernels/*.cuh inference/kernels/*.h) $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) | build
	@if command -v $(NVCC) >/dev/null 2>&1; then $(NVCC) $(NVCCFLAGS) -DLAGUNA_TP16_CONTRACT_SHA256=\"$(LAGUNA_TP16_CONTRACT_SHA256)\" -I. -Iinclude -Isrc -Imodules/laguna_tp16_resident_decode_stage/include -Imodel-families/gemma4/source -Imodel-families/common/include -Xcompiler -fPIC $(SHARED_LIBRARY_FLAGS) $(LAGUNA_TP16_SOURCES) runtime/json.c runtime/filesystem.c src/spark_status.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) -Xcompiler -pthread -ldl $(SPARKPIPE_CUDA_RUNTIME_LINK) -lcuda -o $@; else echo "SKIP $@ (nvcc unavailable on this host; spark-gated artifact)"; fi

$(MIMO26_FLASH_TP16_SERVING_ADAPTER): $(MIMO26_FLASH_TP16_SOURCES) $(MIMO26_FLASH_TP16_CONTRACT) modules/mimo26_flash_tp16_resident_decode_stage/include/sparkpipe/spark_mimo26_flash_tp16_stage_model.h model-families/mimo26/source/mimo26_stage.cuh include/sparkpipe/spark_stage_runner.h include/sparkpipe/spark_stage_runner_model.h include/sparkpipe/spark_stage_serving_adapter.h $(wildcard inference/kernels/*.cuh inference/kernels/*.h) $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) | build
	@if command -v $(NVCC) >/dev/null 2>&1; then $(NVCC) $(NVCCFLAGS) -DMIMO26_FLASH_TP16_CONTRACT_SHA256=\"$(MIMO26_FLASH_TP16_CONTRACT_SHA256)\" -I. -Iinclude -Isrc -Imodules/mimo26_flash_tp16_resident_decode_stage/include -Imodel-families/mimo26/source -Imodel-families/common/include -Xcompiler -fPIC $(SHARED_LIBRARY_FLAGS) $(MIMO26_FLASH_TP16_SOURCES) runtime/json.c runtime/filesystem.c src/spark_status.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) -Xcompiler -pthread -ldl $(SPARKPIPE_CUDA_RUNTIME_LINK) -lcuda -o $@; else echo "SKIP $@ (nvcc unavailable on this host; spark-gated artifact)"; fi

$(MIMO26_PRO_TP16_SERVING_ADAPTER): $(MIMO26_PRO_TP16_SOURCES) $(MIMO26_PRO_TP16_CONTRACT) modules/mimo26_pro_tp16_resident_decode_stage/include/sparkpipe/spark_mimo26_pro_tp16_stage_model.h model-families/mimo26/source/mimo26_stage.cuh include/sparkpipe/spark_stage_runner.h include/sparkpipe/spark_stage_runner_model.h include/sparkpipe/spark_stage_serving_adapter.h $(wildcard inference/kernels/*.cuh inference/kernels/*.h) $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) | build
	@if command -v $(NVCC) >/dev/null 2>&1; then $(NVCC) $(NVCCFLAGS) -DMIMO26_PRO_TP16_CONTRACT_SHA256=\"$(MIMO26_PRO_TP16_CONTRACT_SHA256)\" -I. -Iinclude -Isrc -Imodules/mimo26_pro_tp16_resident_decode_stage/include -Imodel-families/mimo26/source -Imodel-families/common/include -Xcompiler -fPIC $(SHARED_LIBRARY_FLAGS) $(MIMO26_PRO_TP16_SOURCES) runtime/json.c runtime/filesystem.c src/spark_status.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) -Xcompiler -pthread -ldl $(SPARKPIPE_CUDA_RUNTIME_LINK) -lcuda -o $@; else echo "SKIP $@ (nvcc unavailable on this host; spark-gated artifact)"; fi

$(HY4_TP16_SERVING_ADAPTER): $(HY4_TP16_SOURCES) $(HY4_TP16_CONTRACT) modules/hy4_tp16_resident_decode_stage/include/sparkpipe/spark_hy4_tp16_stage_model.h model-families/hy4/source/hy4_stage.cuh include/sparkpipe/spark_stage_runner.h include/sparkpipe/spark_stage_runner_model.h include/sparkpipe/spark_stage_serving_adapter.h $(wildcard inference/kernels/*.cuh inference/kernels/*.h) $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) | build
	@if command -v $(NVCC) >/dev/null 2>&1; then $(NVCC) $(NVCCFLAGS) -DHY4_TP16_CONTRACT_SHA256=\"$(HY4_TP16_CONTRACT_SHA256)\" -I. -Iinclude -Isrc -Imodules/hy4_tp16_resident_decode_stage/include -Imodel-families/hy4/source -Imodel-families/common/include -Xcompiler -fPIC $(SHARED_LIBRARY_FLAGS) $(HY4_TP16_SOURCES) runtime/json.c runtime/filesystem.c src/spark_status.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) -Xcompiler -pthread -ldl $(SPARKPIPE_CUDA_RUNTIME_LINK) -lcuda -o $@; else echo "SKIP $@ (nvcc unavailable on this host; spark-gated artifact)"; fi

$(GLM53_FULL_TP16_SERVING_ADAPTER): $(GLM53_FULL_TP16_SOURCES) model_contracts/glm52.json tools/glm52_model_contract.py modules/glm53_full_tp16_resident_decode_stage/include/sparkpipe/spark_glm53_full_tp16_stage_model.h model-families/glm52/source/glm52_stage.cuh model-families/glm52/source/glm52_row_shard.cuh model-families/glm52/include/sparkpipe/spark_glm52_stagepack.h model-families/glm52/include/sparkpipe/spark_glm52_model.h include/sparkpipe/spark_stage_runner.h include/sparkpipe/spark_stage_runner_model.h include/sparkpipe/spark_stage_serving_adapter.h $(wildcard inference/kernels/*.cuh inference/kernels/*.h) $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) | build
	@test -n "$(GLM53_FULL_TP16_CONTRACT_SHA256)" || { echo "glm53_full: tools/glm52_model_contract.py gave no fp8 build identity" >&2; exit 1; }
	@if command -v $(NVCC) >/dev/null 2>&1; then $(NVCC) $(NVCCFLAGS) -DGLM53_FULL_CONTRACT_SHA256=\"$(GLM53_FULL_TP16_CONTRACT_SHA256)\" -DGLM53_FULL_MODEL_REVISION=\"$(GLM53_FULL_TP16_REVISION)\" -I. -Iinclude -Isrc -Imodules/glm53_full_tp16_resident_decode_stage/include -Imodel-families/glm52/source -Imodel-families/glm52/include -Imodel-families/common/include -Xcompiler -fPIC $(SHARED_LIBRARY_FLAGS) $(GLM53_FULL_TP16_SOURCES) runtime/json.c runtime/filesystem.c src/spark_status.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) -Xcompiler -pthread -ldl $(SPARKPIPE_CUDA_RUNTIME_LINK) -lcuda -o $@; else echo "SKIP $@ (nvcc unavailable on this host; spark-gated artifact)"; fi

$(GEMMA4_26B_TP16_SERVING_ADAPTER): $(GEMMA4_26B_TP16_SOURCES) $(GEMMA4_26B_TP16_CONTRACT) modules/gemma4_26b_resident_decode_stage/include/sparkpipe/spark_gemma4_26b_stage_model.h model-families/gemma4/source/gemma4_stage.cuh include/sparkpipe/spark_stage_runner.h include/sparkpipe/spark_stage_runner_model.h include/sparkpipe/spark_stage_serving_adapter.h $(wildcard inference/kernels/*.cuh inference/kernels/*.h) $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) | build
	@if command -v $(NVCC) >/dev/null 2>&1; then $(NVCC) $(NVCCFLAGS) -DGEMMA4_26B_CONTRACT_SHA256=\"$(GEMMA4_26B_TP16_CONTRACT_SHA256)\" -I. -Iinclude -Isrc -Imodules/gemma4_26b_resident_decode_stage/include -Imodel-families/gemma4/source -Imodel-families/common/include -Xcompiler -fPIC $(SHARED_LIBRARY_FLAGS) $(GEMMA4_26B_TP16_SOURCES) runtime/json.c runtime/filesystem.c src/spark_status.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) -Xcompiler -pthread -ldl $(SPARKPIPE_CUDA_RUNTIME_LINK) -lcuda -o $@; else echo "SKIP $@ (nvcc unavailable on this host; spark-gated artifact)"; fi

define LING3_TP16_LINK
	@if command -v $(NVCC) >/dev/null 2>&1; then $(NVCC) $(NVCCFLAGS) $(1) -I. -Iinclude -Isrc -Imodules/ling3_resident_decode_stage/include -Imodel-families/ling/source -Imodel-families/common/include -Xcompiler -fPIC $(SHARED_LIBRARY_FLAGS) $(LING3_TP16_SOURCES) runtime/json.c runtime/filesystem.c src/spark_status.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) -Xcompiler -pthread -ldl $(SPARKPIPE_CUDA_RUNTIME_LINK) -lcuda -o $@; else echo "SKIP $@ (nvcc unavailable on this host; spark-gated artifact)"; fi
endef

$(LING3_FLASH_TP16_SERVING_ADAPTER): $(LING3_FLASH_TP16_CONTRACT) $(LING3_TP16_SOURCES) modules/ling3_resident_decode_stage/include/sparkpipe/spark_ling3_stage_model.h model-families/ling/source/ling_stage.cuh include/sparkpipe/spark_stage_runner.h include/sparkpipe/spark_stage_runner_model.h include/sparkpipe/spark_stage_serving_adapter.h $(wildcard inference/kernels/*.cuh inference/kernels/*.h) $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) | build
	$(call LING3_TP16_LINK,-DLING3_CONTRACT_SHA256=\"$(LING3_FLASH_TP16_CONTRACT_SHA256)\")

$(LING3_FIN_TP16_SERVING_ADAPTER): $(LING3_FIN_TP16_CONTRACT) $(LING3_TP16_SOURCES) modules/ling3_resident_decode_stage/include/sparkpipe/spark_ling3_stage_model.h model-families/ling/source/ling_stage.cuh include/sparkpipe/spark_stage_runner.h include/sparkpipe/spark_stage_runner_model.h include/sparkpipe/spark_stage_serving_adapter.h $(wildcard inference/kernels/*.cuh inference/kernels/*.h) $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) | build
	$(call LING3_TP16_LINK,-DLING3_FIN=1 -DLING3_CONTRACT_SHA256=\"$(LING3_FIN_TP16_CONTRACT_SHA256)\")

$(DSV41_FLASH_TP16_SERVING_ADAPTER): $(DSV41_FLASH_TP16_SOURCES) $(DSV41_FLASH_TP16_CONTRACT) modules/dsv41_flash_resident_decode_stage/include/sparkpipe/spark_dsv41_flash_stage_model.h modules/dsv41_flash_resident_decode_stage/source/spark_dsv41_flash_stage_kernels.cuh modules/dsv41_flash_resident_decode_stage/source/spark_dsv41_flash_kernels.cuh include/sparkpipe/spark_stage_runner.h include/sparkpipe/spark_stage_runner_model.h include/sparkpipe/spark_stage_serving_adapter.h $(wildcard inference/kernels/*.cuh inference/kernels/*.h) $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) | build
	@if command -v $(NVCC) >/dev/null 2>&1; then $(NVCC) $(NVCCFLAGS) -DDSV41_FLASH_CONTRACT_SHA256=\"$(DSV41_FLASH_TP16_CONTRACT_SHA256)\" -I. -Iinclude -Isrc -Imodules/dsv41_flash_resident_decode_stage/include -Imodules/dsv41_flash_resident_decode_stage/source -Imodel-families/dsv41_flash/include -Imodel-families/common/include -Xcompiler -fPIC $(SHARED_LIBRARY_FLAGS) $(DSV41_FLASH_TP16_SOURCES) runtime/json.c runtime/filesystem.c src/spark_status.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) -Xcompiler -pthread -ldl $(SPARKPIPE_CUDA_RUNTIME_LINK) -lcuda -o $@; else echo "SKIP $@ (nvcc unavailable on this host; spark-gated artifact)"; fi

build/test_model_resident_reconnect: tests/test_model_resident_reconnect.c tests/fixtures/model_resident_deployment_fixture.c tests/fixtures/test_child_guard.c build/sparkpipe_model_residentd $(TEST_MODEL_SERVING_ADAPTER_MODULE) $(TEST_MODEL_RESIDENT_TRANSPORT_MODULE) $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(CPPFLAGS) -DTEST_MODEL_RESIDENTD_PATH=\"build/sparkpipe_model_residentd\" -DTEST_MODEL_SERVING_ADAPTER_PATH=\"$(TEST_MODEL_SERVING_ADAPTER_MODULE)\" -DTEST_MODEL_RESIDENT_TRANSPORT_PATH=\"$(TEST_MODEL_RESIDENT_TRANSPORT_MODULE)\" $(CFLAGS) tests/test_model_resident_reconnect.c tests/fixtures/model_resident_deployment_fixture.c tests/fixtures/test_child_guard.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_model_resident_end_to_end: tests/test_model_resident_end_to_end.c tests/fixtures/model_resident_deployment_fixture.c tests/fixtures/test_child_guard.c build/sparkpipe_model_residentd $(TEST_MODEL_SERVING_ADAPTER_MODULE) $(TEST_MODEL_RESIDENT_TRANSPORT_MODULE) $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(CPPFLAGS) -DTEST_MODEL_RESIDENTD_PATH=\"build/sparkpipe_model_residentd\" -DTEST_MODEL_SERVING_ADAPTER_PATH=\"$(TEST_MODEL_SERVING_ADAPTER_MODULE)\" -DTEST_MODEL_RESIDENT_TRANSPORT_PATH=\"$(TEST_MODEL_RESIDENT_TRANSPORT_MODULE)\" $(CFLAGS) tests/test_model_resident_end_to_end.c tests/fixtures/model_resident_deployment_fixture.c tests/fixtures/test_child_guard.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

runtime_completion_tests: build/test_runtime_completion build/test_model_runtime
	./build/test_runtime_completion
	./build/test_model_runtime

build/test_gemm_descriptor_cache: tests/test_gemm_descriptor_cache.cpp tests/cuda_driver_stub/stub.c runtime/gemm_descriptor_cache.h | build
	$(CXX) -x c++ -Itests/cuda_driver_stub -O2 -Wall -Wextra -c tests/cuda_driver_stub/stub.c -o build/test_gemm_descriptor_cache_stub.o
	$(CXX) $(CPPFLAGS) -I. -Itests/cuda_driver_stub $(CXXFLAGS) tests/test_gemm_descriptor_cache.cpp build/test_gemm_descriptor_cache_stub.o $(LDFLAGS) $(LDLIBS) -o $@

build/test_tensor_map_encode: tests/test_tensor_map_encode.c tests/cuda_driver_stub/stub.c runtime/tensor_map.h inference/kernels/tensor_map.cuh | build
	$(CXX) -x c++ -Itests/cuda_driver_stub -O2 -Wall -Wextra -c tests/cuda_driver_stub/stub.c -o build/test_tensor_map_encode_stub.o
	$(CXX) -x c++ -D__host__= -D__device__= -D__forceinline__=inline $(CORE_INCLUDE_FLAGS) -Itests/cuda_driver_stub $(CXXFLAGS) -Wno-unused-function tests/test_tensor_map_encode.c -x none build/test_tensor_map_encode_stub.o $(LDFLAGS) $(LDLIBS) -o $@

build/test_kv_geometry: tests/test_kv_geometry.cc inference/kernels/kv.cuh | build
	$(CXX) $(CORE_INCLUDE_FLAGS) -Imodel-families/common/include -Imodel-families/glm52/include $(CXXFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

build/test_mma_fragment_mapping: tests/test_mma_fragment_mapping.c tests/host_cuda/lm_host_cuda.cuh inference/kernels/mma.cuh inference/kernels/layout.cuh inference/kernels/dtype.cuh | build
	$(HOST_CUDA_CXX) -std=c++17 -O2 -Wall -Wextra -Werror -Wno-unused-function -Itests/host_cuda/shim -I. -Itests/host_cuda -Imodel-families/common/include -Iinclude -x c++ $< -o $@

build/test_launch: tests/test_launch.c runtime/launch.h inference/kernels/layout.cuh | build
	$(CXX) -x c++ -D__host__= -D__device__= $(CPPFLAGS) $(CXXFLAGS) -Wno-unused-function $< $(LDFLAGS) $(LDLIBS) -o $@

build/test_tensor_map_geometry: tests/test_tensor_map_geometry.c inference/kernels/tensor_map.cuh | build
	$(CC) $(CPPFLAGS) -I. $(CFLAGS) tests/test_tensor_map_geometry.c $(LDFLAGS) $(LDLIBS) -o $@

build/test_distributed_work: tests/test_distributed_work.c $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_distributed_work.c $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_json: tests/test_json.c $(COMPILER_LIBRARY) $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(COMPILER_LIBRARY) $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_quant_arm: tests/test_quant_arm.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_fabric_topology: tests/test_fabric_topology.c $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(MODEL_COMMON_INCLUDE_FLAGS) $(CFLAGS) $< $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_hidden_transport: tests/test_hidden_transport.c $(COMMON_LIBRARY) $(TEST_HIDDEN_TRANSPORT_MODULE)
	$(CC) $(CPPFLAGS) -Itests -DSPARK_TEST_HIDDEN_TRANSPORT_MODULE_PATH=\"$(TEST_HIDDEN_TRANSPORT_MODULE)\" $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_hidden_transport_rdma_control: tests/test_hidden_transport_rdma_control.c ring/transport/rdma_control.c include/sparkpipe/spark_hidden_transport_rdma_control.h
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_hidden_transport_rdma_control.c ring/transport/rdma_control.c $(LDFLAGS) $(LDLIBS) -lpthread -o $@

build/test_draft_bridge: tests/test_draft_bridge.c ring/transport/draft_bridge.c include/sparkpipe/spark_draft_bridge.h
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_draft_bridge.c ring/transport/draft_bridge.c $(LDFLAGS) $(LDLIBS) -lpthread -o $@

build/test_row_layout_round_span: tests/test_row_layout_round_span.c include/sparkpipe/spark_row_layout.h $(CORE_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_row_layout_round_span.c $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_speculation_seam: tests/test_speculation_seam.c ring/transport/draft_bridge.c include/sparkpipe/spark_speculation_seam.h $(CORE_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_speculation_seam.c ring/transport/draft_bridge.c $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -lpthread -o $@

build/test_memlink: tests/test_memlink.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_release: tests/test_release.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_kv_store: tests/test_kv_store.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_nvme_tier: tests/test_nvme_tier.c cache/nvme_tier.c src/spark_sha256.c include/sparkpipe/spark_nvme_tier.h
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_nvme_tier.c cache/nvme_tier.c src/spark_sha256.c $(LDFLAGS) $(LDLIBS) -o $@

build/test_topology_switch: tests/test_topology_switch.c scheduler/topology_switch.c cache/nvme_tier.c src/spark_sha256.c include/sparkpipe/spark_topology_switch.h include/sparkpipe/spark_nvme_tier.h
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_topology_switch.c scheduler/topology_switch.c cache/nvme_tier.c src/spark_sha256.c $(LDFLAGS) $(LDLIBS) -o $@

build/test_continuous_batch: tests/test_continuous_batch.c scheduler/continuous_batch.c include/sparkpipe/spark_continuous_batch.h include/sparkpipe/spark_status.h
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_continuous_batch.c scheduler/continuous_batch.c $(LDFLAGS) $(LDLIBS) -o $@


build/libdsv41_flash_kernels.so: tests/dsv41_flash_kernel_lib.cu modules/dsv41_flash_resident_decode_stage/source/spark_dsv41_flash_kernels.cuh inference/kernels/dtype.cuh | build
	@if command -v $(NVCC) >/dev/null 2>&1; then $(NVCC) -std=c++17 $(NVCCFLAGS) -I. -Xcompiler -fPIC -shared $< -L$(CUDA_HOME)/lib64 -lcudart -o $@; else echo "SKIP libdsv41_flash_kernels (no nvcc on this host)"; fi

build/test_dsv41_flash_layer: tests/test_dsv41_flash_layer.cu modules/dsv41_flash_resident_decode_stage/source/spark_dsv41_flash_kernels.cuh inference/kernels/dtype.cuh | build
	@if command -v $(NVCC) >/dev/null 2>&1; then $(NVCC) -std=c++17 $(NVCCFLAGS) -I. $< -L$(CUDA_HOME)/lib64 -lcudart -o $@; else echo "SKIP test_dsv41_flash_layer (no nvcc on this host)"; fi

build/test_dsv41_flash_kernels: tests/test_dsv41_flash_kernels.cu modules/dsv41_flash_resident_decode_stage/source/spark_dsv41_flash_kernels.cuh inference/kernels/dtype.cuh | build
	@if command -v $(NVCC) >/dev/null 2>&1; then $(NVCC) -std=c++17 $(NVCCFLAGS) -I. $< -L$(CUDA_HOME)/lib64 -lcudart -o $@; else echo "SKIP test_dsv41_flash_kernels (no nvcc on this host)"; fi

build/test_rope_plan: tests/test_rope_plan.c model-families/common/include/sparkpipe/spark_rope_plan.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -I. -Imodel-families/common/include $< -lm -o $@

build/test_serving_tp_config: tests/test_serving_tp_config.c include/sparkpipe/spark_serving_adapter_template.h $(MODEL_COMMON_LIBRARY) $(RUNTIME_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(MODEL_COMMON_LIBRARY) $(RUNTIME_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -lpthread -o $@

build/mb_doorbell: tools/mb_doorbell.cu $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)

build/test_glm52_stagepack: tests/test_glm52_stagepack.c model-families/glm52/include/sparkpipe/spark_glm52_stagepack.h
	$(CC) $(GLM52_INCLUDE_FLAGS) \
		$(CFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

build/test_gemm_tile_k_fallback: tests/test_gemm_tile_k_fallback.c runtime/launch.h inference/kernels/layout.cuh | build
	$(CXX) -x c++ -D__host__= -D__device__= $(CPPFLAGS) $(CXXFLAGS) -Wno-unused-function $< $(LDFLAGS) $(LDLIBS) -o $@

build/test_serial_tp_replay: tests/test_serial_tp_replay.c tests/serial_tp_replay.c tests/serial_tp_replay.h | build
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) tests/test_serial_tp_replay.c tests/serial_tp_replay.c $(LDFLAGS) $(LDLIBS) -o $@

build/test_speculation_tree_resolve: tests/test_speculation_tree_resolve.c $(CORE_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_tokenizer: tests/test_tokenizer.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_tiktoken_compiled: tests/test_tiktoken_compiled.c $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_tokenizer_sidecar: tests/test_tokenizer_sidecar.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_model_api_text: tests/test_model_api_text.c tests/fixtures/model_resident_deployment_fixture.c tests/fixtures/test_child_guard.c tests/fixtures/model_serving_adapter_config.json build/sparkpipe_model_api build/sparkpipe_model_residentd $(TEST_MODEL_SERVING_ADAPTER_MODULE) $(TEST_MODEL_RESIDENT_TRANSPORT_MODULE) $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(CPPFLAGS) -DTEST_MODEL_API_PATH=\"build/sparkpipe_model_api\" -DTEST_MODEL_RESIDENTD_PATH=\"build/sparkpipe_model_residentd\" -DTEST_MODEL_SERVING_ADAPTER_PATH=\"$(TEST_MODEL_SERVING_ADAPTER_MODULE)\" -DTEST_MODEL_RESIDENT_TRANSPORT_PATH=\"$(TEST_MODEL_RESIDENT_TRANSPORT_MODULE)\" $(CFLAGS) tests/test_model_api_text.c tests/fixtures/model_resident_deployment_fixture.c tests/fixtures/test_child_guard.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_system_loopback: tests/test_system_loopback.c tests/fixtures/model_resident_deployment_fixture.c tests/fixtures/test_child_guard.c tests/fixtures/model_serving_adapter_config.json build/sparkpipe_model_api build/sparkpipe_model_residentd $(TEST_MODEL_SERVING_ADAPTER_MODULE) $(TEST_MODEL_RESIDENT_TRANSPORT_MODULE) $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(CPPFLAGS) -DTEST_MODEL_API_PATH=\"build/sparkpipe_model_api\" -DTEST_MODEL_RESIDENTD_PATH=\"build/sparkpipe_model_residentd\" -DTEST_MODEL_SERVING_ADAPTER_PATH=\"$(TEST_MODEL_SERVING_ADAPTER_MODULE)\" -DTEST_MODEL_RESIDENT_TRANSPORT_PATH=\"$(TEST_MODEL_RESIDENT_TRANSPORT_MODULE)\" $(CFLAGS) tests/test_system_loopback.c tests/fixtures/model_resident_deployment_fixture.c tests/fixtures/test_child_guard.c $(RUNTIME_LIBRARY) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_model_description: tests/test_model_description.c $(COMPILER_LIBRARY) $(CORE_LIBRARY)
	$(CC) $(CORE_INCLUDE_FLAGS) -Itests $(CFLAGS) $< $(COMPILER_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_stage_module_common: tests/test_stage_module_common.c runtime/stage_module_common.c $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) tests/cuda_stub/cuda_runtime_stub.c | build
	$(CC) $(CORE_INCLUDE_FLAGS) $(CUDA_STUB_INCLUDE_FLAGS) $(CFLAGS) tests/test_stage_module_common.c runtime/stage_module_common.c $(SPARKPIPE_TP_DEVICE_TEST_CUDA_STUB_SOURCE) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

STAGE_KV_LAYOUT_DIGEST_TEST_SOURCES := tests/test_stage_kv_layout_digest.c runtime/stage_kv_binding.c runtime/kv_device_copy.c runtime/stage_module_common.c cache/kv_cache.c cache/kv_page_cache.c cache/kv_page_store.c cache/kv_snapshot.c cache/kv_model_table.c
build/test_stage_kv_layout_digest: $(STAGE_KV_LAYOUT_DIGEST_TEST_SOURCES) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) tests/cuda_stub/cuda_runtime_stub.c | build
	$(CC) $(CORE_INCLUDE_FLAGS) $(CUDA_STUB_INCLUDE_FLAGS) $(CFLAGS) $(STAGE_KV_LAYOUT_DIGEST_TEST_SOURCES) $(SPARKPIPE_TP_DEVICE_TEST_CUDA_STUB_SOURCE) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

STAGE_KV_BINDING_TEST_SOURCES := tests/test_stage_kv_binding.c runtime/stage_kv_binding.c runtime/kv_device_copy.c runtime/stage_module_common.c cache/kv_cache.c cache/kv_page_cache.c cache/kv_page_store.c cache/kv_snapshot.c cache/kv_model_table.c $(SPARKPIPE_WEIGHTD_SOURCES)
build/test_stage_kv_binding: $(STAGE_KV_BINDING_TEST_SOURCES) tests/test_weightd_kv_server.h $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) tests/cuda_stub/cuda_runtime_stub.c | build
	$(CC) $(CORE_INCLUDE_FLAGS) $(CUDA_STUB_INCLUDE_FLAGS) -Imodel-families/common/include $(CFLAGS) $(STAGE_KV_BINDING_TEST_SOURCES) $(SPARKPIPE_TP_DEVICE_TEST_CUDA_STUB_SOURCE) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

STAGE_KV_SHARED_PREFIX_TEST_SOURCES := tests/test_stage_kv_shared_prefix.c runtime/stage_kv_binding.c runtime/kv_device_copy.c runtime/stage_module_common.c cache/kv_cache.c cache/kv_page_cache.c cache/kv_page_store.c cache/kv_snapshot.c cache/kv_model_table.c $(SPARKPIPE_WEIGHTD_SOURCES)
build/test_stage_kv_shared_prefix: $(STAGE_KV_SHARED_PREFIX_TEST_SOURCES) tests/test_weightd_kv_server.h $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) tests/cuda_stub/cuda_runtime_stub.c | build
	$(CC) $(CORE_INCLUDE_FLAGS) $(CUDA_STUB_INCLUDE_FLAGS) -Imodel-families/common/include $(CFLAGS) $(STAGE_KV_SHARED_PREFIX_TEST_SOURCES) $(SPARKPIPE_TP_DEVICE_TEST_CUDA_STUB_SOURCE) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

STAGE_KV_POOL_RESIZE_TEST_SOURCES := tests/test_stage_kv_pool_resize.c runtime/stage_kv_binding.c runtime/kv_device_copy.c runtime/stage_module_common.c cache/kv_cache.c cache/kv_page_cache.c cache/kv_page_store.c cache/kv_snapshot.c cache/kv_model_table.c $(SPARKPIPE_WEIGHTD_SOURCES)
build/test_stage_kv_pool_resize: $(STAGE_KV_POOL_RESIZE_TEST_SOURCES) tests/test_weightd_kv_server.h $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) tests/cuda_stub/cuda_runtime_stub.c | build
	$(CC) $(CORE_INCLUDE_FLAGS) $(CUDA_STUB_INCLUDE_FLAGS) -Imodel-families/common/include $(CFLAGS) $(STAGE_KV_POOL_RESIZE_TEST_SOURCES) $(SPARKPIPE_TP_DEVICE_TEST_CUDA_STUB_SOURCE) $(MODEL_COMMON_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/sparkpipe_weightd: node/weightd.c node/weightd_mesh.c $(RUNTIME_LIBRARY) $(CORE_LIBRARY) $(SPARKPIPE_HOST_CUDA_STUB_SOURCE) | build
	$(CC) $(MODEL_COMMON_INCLUDE_FLAGS) $(CFLAGS) $^ $(LDFLAGS) $(SPARKPIPE_CUDA_RUNTIME_LINK) $(SPARKPIPE_CUDA_DRIVER_LINK) -libverbs -lmlx5 -o $@

build/sparkpipe_weightsd: node/weightd.c node/weightd_mesh.c $(RUNTIME_LIBRARY) $(CORE_LIBRARY) $(SPARKPIPE_HOST_CUDA_STUB_SOURCE) | build
	$(CC) $(MODEL_COMMON_INCLUDE_FLAGS) $(CFLAGS) $^ $(LDFLAGS) $(SPARKPIPE_CUDA_RUNTIME_LINK) $(SPARKPIPE_CUDA_DRIVER_LINK) -libverbs -lmlx5 -o $@

build/glm5_next_experts_manifest: tools/glm5_next_experts_manifest.c runtime/spark_weightd_manifest.c include/sparkpipe/spark_weightd_manifest.h include/sparkpipe/spark_expert_planes.h $(CORE_LIBRARY) | build
	$(CC) $(CORE_INCLUDE_FLAGS) -Imodel-families/glm5_next/include $(CFLAGS) tools/glm5_next_experts_manifest.c runtime/spark_weightd_manifest.c $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

DSV41_FLASH_PACK_TOOL_FLAGS := $(CORE_INCLUDE_FLAGS) -Imodel-families/common/include -Imodel-families/dsv41_flash/include -Imodules/dsv41_flash_resident_decode_stage/include -Imodules/dsv41_flash_resident_decode_stage/source -D_FILE_OFFSET_BITS=64
DSV41_FLASH_PACK_FORMAT_HEADER := modules/dsv41_flash_resident_decode_stage/source/spark_dsv41_flash_stagepack_format.h

build/dsv41_flash_pack_synthesize: modules/dsv41_flash_resident_decode_stage/tools/dsv41_flash_pack_synthesize.c $(DSV41_FLASH_PACK_FORMAT_HEADER) | build
	$(CC) $(DSV41_FLASH_PACK_TOOL_FLAGS) $(CFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

build/dsv41_flash_experts_manifest: modules/dsv41_flash_resident_decode_stage/tools/dsv41_flash_experts_manifest.c src/spark_ck128.c $(DSV41_FLASH_PACK_FORMAT_HEADER) | build
	$(CC) $(DSV41_FLASH_PACK_TOOL_FLAGS) $(CFLAGS) $< src/spark_ck128.c $(LDFLAGS) $(LDLIBS) -o $@

build/test_dsv41_flash_pack_contract: tests/test_dsv41_flash_pack_contract.c src/spark_ck128.c $(DSV41_FLASH_PACK_FORMAT_HEADER) | build
	$(CC) $(DSV41_FLASH_PACK_TOOL_FLAGS) $(CFLAGS) $< src/spark_ck128.c $(LDFLAGS) $(LDLIBS) -o $@

build/weightd_lazy_consumer: tools/weightd_lazy_consumer.c $(RUNTIME_LIBRARY) $(CORE_LIBRARY) $(SPARKPIPE_HOST_CUDA_STUB_SOURCE) | build
	$(CC) $(MODEL_COMMON_INCLUDE_FLAGS) $(CFLAGS) $^ $(LDFLAGS) $(LDLIBS) $(SPARKPIPE_CUDA_RUNTIME_LINK) $(SPARKPIPE_CUDA_DRIVER_LINK) -o $@

build/weightd_receipt: tools/weightd_receipt.c runtime/spark_weightd_receipt.c $(CORE_LIBRARY) | build
	$(CC) $(CORE_INCLUDE_FLAGS) $(CFLAGS) $^ $(LDFLAGS) -o $@

build/sparkpipe_mesh_status: tools/mesh_status.c $(RUNTIME_LIBRARY) $(CORE_LIBRARY) $(SPARKPIPE_HOST_CUDA_STUB_SOURCE) | build
	$(CC) $(MODEL_COMMON_INCLUDE_FLAGS) $(CFLAGS) $^ $(LDFLAGS) $(LDLIBS) $(SPARKPIPE_CUDA_RUNTIME_LINK) $(SPARKPIPE_CUDA_DRIVER_LINK) -o $@

build/weightdctl: tools/weightdctl.c $(RUNTIME_LIBRARY) $(CORE_LIBRARY) $(SPARKPIPE_HOST_CUDA_STUB_SOURCE) | build
	$(CC) $(MODEL_COMMON_INCLUDE_FLAGS) $(CFLAGS) $^ $(LDFLAGS) $(LDLIBS) $(SPARKPIPE_CUDA_RUNTIME_LINK) $(SPARKPIPE_CUDA_DRIVER_LINK) -o $@

build/weightd_execute_probe: tools/weightd_execute_probe.c $(RUNTIME_LIBRARY) $(CORE_LIBRARY) $(SPARKPIPE_HOST_CUDA_STUB_SOURCE) | build
	$(CC) $(MODEL_COMMON_INCLUDE_FLAGS) $(CFLAGS) $^ $(LDFLAGS) $(LDLIBS) $(SPARKPIPE_CUDA_RUNTIME_LINK) $(SPARKPIPE_CUDA_DRIVER_LINK) -o $@

build/weightd_route_probe: tools/weightd_route_probe.c $(RUNTIME_LIBRARY) $(CORE_LIBRARY) $(SPARKPIPE_HOST_CUDA_STUB_SOURCE) | build
	$(CC) $(MODEL_COMMON_INCLUDE_FLAGS) $(CFLAGS) $^ $(LDFLAGS) $(LDLIBS) $(SPARKPIPE_CUDA_RUNTIME_LINK) $(SPARKPIPE_CUDA_DRIVER_LINK) -o $@

build/multi_dev_smoke: tools/multi_dev_smoke.c $(RUNTIME_LIBRARY) $(CORE_LIBRARY) $(SPARKPIPE_HOST_CUDA_STUB_SOURCE) | build
	$(CC) $(MODEL_COMMON_INCLUDE_FLAGS) $(CFLAGS) $^ $(LDFLAGS) $(LDLIBS) $(SPARKPIPE_CUDA_RUNTIME_LINK) $(SPARKPIPE_CUDA_DRIVER_LINK) -o $@

build/test_weightd_lease: tests/test_weightd_lease.c runtime/spark_weightd_lease.c runtime/spark_weightd_manifest.c include/sparkpipe/spark_weightd_lease.h include/sparkpipe/spark_weightd_manifest.h | build
	$(CC) $(CORE_INCLUDE_FLAGS) $(CFLAGS) tests/test_weightd_lease.c runtime/spark_weightd_lease.c runtime/spark_weightd_manifest.c $(LDFLAGS) -o $@

build/test_weightd_dense_manifest: tests/test_weightd_dense_manifest.c $(RUNTIME_LIBRARY) $(CORE_LIBRARY) tests/cuda_stub/cuda_runtime_stub.c | build
	$(CC) $(CORE_INCLUDE_FLAGS) $(CUDA_STUB_INCLUDE_FLAGS) $(CFLAGS) $^ $(LDFLAGS) -o $@

build/test_weightd_working_set: tests/test_weightd_working_set.c $(RUNTIME_LIBRARY) $(CORE_LIBRARY) tests/cuda_stub/cuda_runtime_stub.c | build
	$(CC) $(CORE_INCLUDE_FLAGS) $(CUDA_STUB_INCLUDE_FLAGS) $(CFLAGS) $^ $(LDFLAGS) -o $@

build/test_child_guard: tests/test_child_guard.c tests/fixtures/test_child_guard.c tests/fixtures/test_child_guard.h | build
	$(CC) $(CFLAGS) -Itests tests/test_child_guard.c tests/fixtures/test_child_guard.c $(LDFLAGS) -o $@

build/test_weightd_churn: tests/test_weightd_churn.c tests/fixtures/test_child_guard.c $(RUNTIME_LIBRARY) $(CORE_LIBRARY) tests/cuda_stub/cuda_runtime_stub.c | build
	$(CC) $(CORE_INCLUDE_FLAGS) $(CUDA_STUB_INCLUDE_FLAGS) $(CFLAGS) $^ $(LDFLAGS) -o $@

build/test_weightd_expert_stress: tests/test_weightd_expert_stress.c tests/fixtures/test_child_guard.c $(RUNTIME_LIBRARY) $(CORE_LIBRARY) tests/cuda_stub/cuda_runtime_stub.c | build
	$(CC) $(CORE_INCLUDE_FLAGS) $(CUDA_STUB_INCLUDE_FLAGS) $(CFLAGS) $^ $(LDFLAGS) -o $@

build/test_weightd_fd_frames: tests/test_weightd_fd_frames.c runtime/spark_weightd_spine.c runtime/spark_weightd_direct.c runtime/spark_weightd_receipt.c runtime/spark_weightd.c runtime/spark_weightd_worker.c runtime/spark_weightd_pacer.c runtime/spark_weightd_manifest.c runtime/spark_weightd_lease.c include/sparkpipe/spark_weightd.h include/sparkpipe/spark_weightd_manifest.h include/sparkpipe/spark_weightd_lease.h $(CORE_LIBRARY) tests/cuda_stub/cuda_runtime_stub.c | build
	$(CC) $(CORE_INCLUDE_FLAGS) $(CUDA_STUB_INCLUDE_FLAGS) $(CFLAGS) $(filter %.c %.a,$(filter-out runtime/spark_weightd.c runtime/spark_weightd_direct.c,$^)) $(LDFLAGS) -o $@

build/test_weightd: tests/test_weightd.c $(RUNTIME_LIBRARY) $(CORE_LIBRARY) tests/cuda_stub/cuda_runtime_stub.c | build build/sparkpipe_weightd
	$(CC) $(CORE_INCLUDE_FLAGS) $(CUDA_STUB_INCLUDE_FLAGS) -DSPARK_TEST_WEIGHTD_BINARY=\"build/sparkpipe_weightd\" $(CFLAGS) $^ $(LDFLAGS) -o $@

build/test_weightd_receipt: tests/test_weightd_receipt.c runtime/spark_weightd_receipt.c $(CORE_LIBRARY) | build
	$(CC) $(CORE_INCLUDE_FLAGS) $(CFLAGS) $^ $(LDFLAGS) -o $@

build/test_weightd_attach: tests/test_weightd_attach.c $(RUNTIME_LIBRARY) $(CORE_LIBRARY) tests/cuda_stub/cuda_runtime_stub.c | build
	$(CC) $(CORE_INCLUDE_FLAGS) $(CUDA_STUB_INCLUDE_FLAGS) $(CFLAGS) $^ $(LDFLAGS) -o $@
build/test_weightd_expert: tests/test_weightd_expert.c $(RUNTIME_LIBRARY) $(CORE_LIBRARY) tests/cuda_stub/cuda_runtime_stub.c | build
	$(CC) $(CORE_INCLUDE_FLAGS) $(CUDA_STUB_INCLUDE_FLAGS) $(CFLAGS) $^ $(LDFLAGS) -o $@
build/test_stage_module_weightd: tests/test_stage_module_weightd.c runtime/stage_module_common.c $(RUNTIME_LIBRARY) $(CORE_LIBRARY) tests/cuda_stub/cuda_runtime_stub.c | build
	$(CC) $(CORE_INCLUDE_FLAGS) $(CUDA_STUB_INCLUDE_FLAGS) $(CFLAGS) $^ $(LDFLAGS) -o $@

build/test_expert_working_set: tests/test_expert_working_set.c $(RUNTIME_LIBRARY) $(CORE_LIBRARY) | build
	$(CC) $(CORE_INCLUDE_FLAGS) $(CFLAGS) $^ $(LDFLAGS) -o $@

build/test_graph_reloc: tests/test_graph_reloc.c runtime/spark_graph_reloc.c runtime/spark_graph_reloc_cuda.c tests/cuda_stub/cuda_runtime_stub.c include/sparkpipe/spark_graph_reloc.h $(CORE_LIBRARY) | build
	$(CC) $(CORE_INCLUDE_FLAGS) $(CUDA_STUB_INCLUDE_FLAGS) $(CFLAGS) $(filter %.c %.a,$^) $(LDFLAGS) $(LDLIBS) -o $@

build/test_graph_reloc_gpu: tests/test_graph_reloc_gpu.cu runtime/spark_graph_reloc.c runtime/spark_graph_reloc_cuda.c include/sparkpipe/spark_graph_reloc.h | build
	$(NVCC) -std=c++17 $(NVCCFLAGS) -I. -Iinclude tests/test_graph_reloc_gpu.cu runtime/spark_graph_reloc.c runtime/spark_graph_reloc_cuda.c -L$(CUDA_HOME)/lib64 -lcudart -lcuda -o $@

.PHONY: test-graph-reloc-gpu
test-graph-reloc-gpu: build/test_graph_reloc_gpu
	./build/test_graph_reloc_gpu

build/test_weightd_map: tests/test_weightd_map.c $(RUNTIME_LIBRARY) $(CORE_LIBRARY) tests/cuda_stub/cuda_runtime_stub.c | build
	$(CC) $(CORE_INCLUDE_FLAGS) $(CUDA_STUB_INCLUDE_FLAGS) $(CFLAGS) $^ $(LDFLAGS) -o $@

build/test_weightd_mesh_mock: tests/test_weightd_mesh_mock.c node/weightd_mesh.c tests/ibv_stub/verbs.c tests/cuda_stub/cuda_runtime_stub.c tests/ibv_stub/infiniband/verbs.h tests/ibv_stub/sys/mman.h $(RUNTIME_LIBRARY) $(CORE_LIBRARY) | build
	$(CC) $(CORE_INCLUDE_FLAGS) $(CUDA_STUB_INCLUDE_FLAGS) -Itests/ibv_stub -DSPARK_WEIGHTD_MESH_DIR=\"/tmp/spark-weightd-mesh-mock\" $(CFLAGS) tests/test_weightd_mesh_mock.c tests/ibv_stub/verbs.c tests/cuda_stub/cuda_runtime_stub.c $(RUNTIME_LIBRARY) $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_module_library: tests/test_module_library.c $(TEST_SUPPORT_OBJECT) $(TEST_MODULE_LINK_UNITS) $(TEST_VALIDATOR) $(TEST_VALIDATOR_CHANGED) $(COMPILER_LIBRARY) $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(TEST_SUPPORT_OBJECT) $(COMPILER_LIBRARY) $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@
build/test_speculation_reference_draft: tests/test_speculation_reference_draft.c include/sparkpipe/spark_speculation_reference_draft.h $(CORE_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_speculation_reference_draft.c $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_speculation_recorded_draft: tests/test_speculation_recorded_draft.c include/sparkpipe/spark_speculation_recorded_draft.h $(CORE_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_speculation_recorded_draft.c $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_speculation_lookup_draft: tests/test_speculation_lookup_draft.c include/sparkpipe/spark_speculation_lookup_draft.h $(CORE_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_speculation_lookup_draft.c $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_speculation_ngram_draft: tests/test_speculation_ngram_draft.c include/sparkpipe/spark_speculation_ngram_draft.h $(CORE_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_speculation_ngram_draft.c $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_speculation_drafter_mix: tests/test_speculation_drafter_mix.c include/sparkpipe/spark_speculation_drafter_mix.h include/sparkpipe/spark_speculation_lookup_draft.h $(CORE_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_speculation_drafter_mix.c $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_speculation_frame: tests/test_speculation_frame.c include/sparkpipe/spark_speculation_frame.h $(CORE_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_speculation_frame.c $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -lm -o $@

build/test_speculation_depth: tests/test_speculation_depth.c include/sparkpipe/spark_speculation_depth.h
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_speculation_depth.c $(LDFLAGS) $(LDLIBS) -o $@

build/test_speculation_relay_draft: tests/test_speculation_relay_draft.c include/sparkpipe/spark_speculation_relay_draft.h $(CORE_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_speculation_relay_draft.c $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_speculation_tap: tests/test_speculation_tap.c include/sparkpipe/spark_speculation_tap.h $(CORE_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_speculation_tap.c $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_speculation_relay_link: tests/test_speculation_relay_link.c include/sparkpipe/spark_speculation_relay_link.h include/sparkpipe/spark_speculation_tap.h $(CORE_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_speculation_relay_link.c $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/spark_speculation_relay_probe: tools/spec_relay_probe.c include/sparkpipe/spark_speculation_relay_link.h include/sparkpipe/spark_speculation_tap.h $(CORE_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) tools/spec_relay_probe.c $(CORE_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@


build/test_driver_compiler: tests/test_driver_compiler.c $(TEST_SUPPORT_OBJECT) $(TEST_MODULE_LINK_UNITS) $(TEST_VALIDATOR) $(COMPILER_LIBRARY) $(RUNTIME_LIBRARY) $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(TEST_SUPPORT_OBJECT) $(COMPILER_LIBRARY) $(RUNTIME_LIBRARY) $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_orchestrator: tests/test_orchestrator.c $(TEST_SUPPORT_OBJECT) $(TEST_MODULE_LINK_UNITS) $(TEST_VALIDATOR) $(COMPILER_LIBRARY) $(RUNTIME_LIBRARY) $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(TEST_SUPPORT_OBJECT) $(COMPILER_LIBRARY) $(RUNTIME_LIBRARY) $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_weight_codec: tests/test_weight_codec.c include/sparkpipe/spark_weight_codec.h
	$(CC) $(CORE_INCLUDE_FLAGS) $(CFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

TEST_EXCLUDE ?=

stray-fixtures:
	python3 tools/stray_fixture_processes.py $(if $(STRAY_UNDER),--under $(STRAY_UNDER))

.PHONY: stray-fixtures

test: all $(TEST_BINARIES) $(PYTHON_TEST_BINARIES)
	@set -e; \
	for test_binary in $(TEST_BINARIES); do \
		if [ ! -x "$$test_binary" ]; then echo "SKIP $$test_binary (host does not build it)"; continue; fi; \
		case " $(TEST_EXCLUDE) " in *" $$test_binary "*) echo "SKIP $$test_binary (TEST_EXCLUDE)"; continue;; esac; \
		echo "RUN $$test_binary"; \
		./$$test_binary; \
	done; \
	for python_test in $(PYTHON_TESTS); do \
		echo "RUN $$python_test"; \
		python3 $$python_test; \
	done; \
	for shell_test in $(SHELL_TESTS); do \
		echo "RUN $$shell_test"; \
		bash $$shell_test; \
	done

OFFLINE_GATES := build-all run-tests package-manifest

.PHONY: fleet-test
fleet-test:
	bash tools/run_fleet_tests.sh

.PHONY: offline-gates
offline-gates:
	@set -e; \
	for gate in $(OFFLINE_GATES); do \
		echo "OFFLINE GATE: $$gate"; \
		case $$gate in \
			build-all) $(MAKE) -j1 all;; \
			run-tests) $(MAKE) -j1 test;; \
			package-manifest) tools/source_package_gate.sh;; \
			*) echo "unknown offline gate: $$gate" >&2; exit 2;; \
		esac; \
	done; \
	echo "OFFLINE GATE PASS: $(OFFLINE_GATES)"

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

tree_summary:
	@printf "core_public_headers="; find include/sparkpipe -type f | wc -l
	@printf "core_support_sources="; printf '%s\n' $(CORE_SOURCES) | wc -l
	@printf "compiler_sources="; printf '%s\n' $(COMPILER_SOURCES) | wc -l
	@printf "runtime_sources="; printf '%s\n' $(RUNTIME_SOURCES) | wc -l
	@printf "model_common_sources="; printf '%s\n' $(MODEL_COMMON_SOURCES) | wc -l
	@printf "glm52_host_sources="; printf '%s\n' $(GLM52_HOST_SOURCES) | wc -l
	@printf "deployment_sources="; printf '%s\n' $(DEPLOYMENT_SOURCES) | wc -l
	@printf "test_executables="; printf '%s\n' $(TEST_NAMES) | wc -l

clean:
	rm -rf build

-include $(ALL_HOST_OBJECTS:.o=.d) $(TEST_SUPPORT_OBJECT:.o=.d) \
    $(TEST_MODULE_DEPENDENCIES)

build/test_weightd_direct: tests/test_weightd_direct.c runtime/spark_weightd_direct.c $(CORE_LIBRARY) | build
	$(CC) $(CORE_INCLUDE_FLAGS) $(CFLAGS) $^ $(LDFLAGS) -o $@

build/pack_stream_bench: tools/pack_stream_bench.c runtime/spark_weightd_direct.c runtime/spark_weightd_manifest.c $(CORE_LIBRARY) | build
	$(CC) $(CORE_INCLUDE_FLAGS) $(CFLAGS) $^ $(LDFLAGS) -o $@

build/test_weightd_worker: tests/test_weightd_worker.c runtime/spark_weightd_worker.c tests/cuda_stub/cuda_runtime_stub.c | build
	$(CC) $(CORE_INCLUDE_FLAGS) $(CUDA_STUB_INCLUDE_FLAGS) $(CFLAGS) $^ $(LDFLAGS) -o $@

build/test_weightd_pacer: tests/test_weightd_pacer.c runtime/spark_weightd_pacer.c include/sparkpipe/spark_weightd_pacer.h $(CORE_LIBRARY) | build
	$(CC) $(CORE_INCLUDE_FLAGS) $(CFLAGS) $(filter %.c %.a,$^) $(LDFLAGS) -o $@

build/weightd_peer_route_probe: tools/weightd_peer_route_probe.c $(RUNTIME_LIBRARY) $(CORE_LIBRARY) $(SPARKPIPE_HOST_CUDA_STUB_SOURCE) | build
	$(CC) $(MODEL_COMMON_INCLUDE_FLAGS) $(CFLAGS) $^ $(LDFLAGS) $(LDLIBS) $(SPARKPIPE_CUDA_RUNTIME_LINK) $(SPARKPIPE_CUDA_DRIVER_LINK) -o $@

build/weightd_warm: tools/weightd_warm.c $(RUNTIME_LIBRARY) $(CORE_LIBRARY) $(SPARKPIPE_HOST_CUDA_STUB_SOURCE) | build
	$(CC) $(MODEL_COMMON_INCLUDE_FLAGS) $(CFLAGS) $^ $(LDFLAGS) $(LDLIBS) $(SPARKPIPE_CUDA_RUNTIME_LINK) $(SPARKPIPE_CUDA_DRIVER_LINK) -o $@

build/mesh_register_attach_repro: tools/mesh_register_attach_repro.c $(RUNTIME_LIBRARY) $(CORE_LIBRARY) $(SPARKPIPE_HOST_CUDA_STUB_SOURCE) | build
	$(CC) $(MODEL_COMMON_INCLUDE_FLAGS) $(CFLAGS) $^ $(LDFLAGS) $(LDLIBS) $(SPARKPIPE_CUDA_RUNTIME_LINK) $(SPARKPIPE_CUDA_DRIVER_LINK) -o $@

.PHONY: test_station
test_station: build/test_tiktoken_compiled  $(RUNTIME_LIBRARY)
	python3 tests/test_spark_station.py
	python3 tests/test_spark_tiktoken_compile.py
	python3 tests/test_kv_failure_host.py
	python3 tests/test_frame_error_host.py
