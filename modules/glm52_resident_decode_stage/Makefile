REPOSITORY_ROOT := $(abspath ../..)
CC ?= cc
AR ?= ar
NVCC ?= nvcc
CUDA_ARCH ?= sm_121
CFLAGS ?= -std=c11 -Wall -Wextra -Werror -O3
NVCCFLAGS ?= -std=c++17 -O3 --use_fast_math -lineinfo -arch=$(CUDA_ARCH)
MODULE_LIBRARY_ROOT ?= $(REPOSITORY_ROOT)/build/module_library
MAX_STAGE_MICROSECONDS ?= 0
ARTIFACT_CHECK_ARGS ?=
BUILD_DIRECTORY := $(REPOSITORY_ROOT)/build/modules/glm52_resident_decode_stage
HOST_OBJECT := $(BUILD_DIRECTORY)/spark_glm52_resident_decode_stage_module.o
CUDA_OBJECT := $(BUILD_DIRECTORY)/spark_glm52_resident_decode_stage_cuda.o
CUBLASLT_PLAN_OBJECT := $(BUILD_DIRECTORY)/spark_glm52_resident_decode_stage_cublaslt_plans.o
SOTA_V6_SOURCE_DIRECTORY := source/sota_v6
SOTA_V6_CUDA_SOURCES := \
	$(SOTA_V6_SOURCE_DIRECTORY)/spark_glm52_sota_production_plan_sm121.cu \
	$(SOTA_V6_SOURCE_DIRECTORY)/spark_glm52_sota_cutlass_nvfp4_grouped_gemm_sm121.cu \
	$(SOTA_V6_SOURCE_DIRECTORY)/spark_glm52_sota_grouped_nvfp4_moe_production_sm121.cu \
	$(SOTA_V6_SOURCE_DIRECTORY)/spark_glm52_sota_router_topk_fused_sm121.cu \
	$(SOTA_V6_SOURCE_DIRECTORY)/spark_glm52_sota_flash_mla_sm121.cu \
	$(SOTA_V6_SOURCE_DIRECTORY)/spark_glm52_sota_cublaslt_linear_plans_sm121.cu \
	$(SOTA_V6_SOURCE_DIRECTORY)/spark_glm52_sota_glm52_fixed_linear_plans_sm121.cu \
	$(SOTA_V6_SOURCE_DIRECTORY)/spark_glm52_sota_rmsnorm_residual_sm121.cu \
	$(SOTA_V6_SOURCE_DIRECTORY)/spark_glm52_sota_fused_rope_kv_write_sm121.cu \
	$(SOTA_V6_SOURCE_DIRECTORY)/spark_glm52_sota_dense_mlp_epilogue_sm121.cu \
	$(SOTA_V6_SOURCE_DIRECTORY)/spark_glm52_sota_restricted_logits_sampler_sm121.cu \
	$(SOTA_V6_SOURCE_DIRECTORY)/spark_glm52_sota_mtp_mxfp4_verify_sm121.cu \
	$(SOTA_V6_SOURCE_DIRECTORY)/spark_glm52_sota_decode_graph_sm121.cu
SOTA_V6_OBJECTS := $(patsubst $(SOTA_V6_SOURCE_DIRECTORY)/%.cu,$(BUILD_DIRECTORY)/sota_v6/%.o,$(SOTA_V6_CUDA_SOURCES))
CUTLASS_MOE_BENCHMARK_DEPENDENCIES := \
	$(BUILD_DIRECTORY)/sota_v6/spark_glm52_sota_production_plan_sm121.o \
	$(BUILD_DIRECTORY)/sota_v6/spark_glm52_sota_cutlass_nvfp4_grouped_gemm_sm121.o \
	$(BUILD_DIRECTORY)/sota_v6/spark_glm52_sota_grouped_nvfp4_moe_production_sm121.o
DENSE_ALPHA_MOE_BENCHMARK_DEPENDENCIES := \
	$(BUILD_DIRECTORY)/sota_v6/spark_glm52_sota_production_plan_sm121.o \
	$(BUILD_DIRECTORY)/sota_v6/spark_glm52_sota_cutlass_nvfp4_grouped_gemm_sm121.o \
	$(BUILD_DIRECTORY)/sota_v6/spark_glm52_sota_grouped_nvfp4_moe_production_sm121.o
HOST_DEPENDENCY := $(HOST_OBJECT:.o=.d)
CUDA_DEPENDENCY := $(CUDA_OBJECT:.o=.d)
CUBLASLT_PLAN_DEPENDENCY := $(CUBLASLT_PLAN_OBJECT:.o=.d)
SOTA_V6_DEPENDENCIES := $(SOTA_V6_OBJECTS:.o=.d)
MODULE_ARCHIVE := $(BUILD_DIRECTORY)/libglm52_resident_decode_stage.a
ARTIFACT_CHECK_TOOL := $(BUILD_DIRECTORY)/glm52_artifact_check
CUTLASS_MOE_BENCHMARK := $(BUILD_DIRECTORY)/spark_glm52_sota_cutlass_moe_benchmark
CUTLASS_MOE_BENCHMARK_OBJECT := $(BUILD_DIRECTORY)/spark_glm52_sota_cutlass_moe_benchmark.o
CUTLASS_ROUTE_MAJOR_MOE_BENCHMARK := $(BUILD_DIRECTORY)/spark_glm52_sota_cutlass_route_major_moe_benchmark
CUTLASS_ROUTE_MAJOR_MOE_BENCHMARK_OBJECT := $(BUILD_DIRECTORY)/spark_glm52_sota_cutlass_route_major_moe_benchmark.o
DENSE_ALPHA_MOE_BENCHMARK := $(BUILD_DIRECTORY)/spark_glm52_sota_dense_alpha_moe_benchmark
DENSE_ALPHA_MOE_BENCHMARK_OBJECT := $(BUILD_DIRECTORY)/spark_glm52_sota_dense_alpha_moe_benchmark.o
PACKAGE_DIRECTORY ?= $(REPOSITORY_ROOT)/build/packages/glm52_resident_decode_stage
CUDA_HOME ?= $(abspath $(dir $(shell command -v $(NVCC)))/..)
VALIDATION_RECIPE_VARIANT ?= default
VALIDATION_RECIPE := glm52.resident_decode_stage.$(CUDA_ARCH).$(VALIDATION_RECIPE_VARIANT).max_us_$(MAX_STAGE_MICROSECONDS).v1
SOTA_V6_ENABLE_CUTLASS ?= 0
SOTA_V6_CUTLASS_INCLUDE ?=
SOTA_V6_CUTLASS_UTIL_INCLUDE ?=
SOTA_V6_CUTLASS_FLAGS :=
ifeq ($(SOTA_V6_ENABLE_CUTLASS),1)
SOTA_V6_CUTLASS_FLAGS += -DSPARK_GLM52_ENABLE_CUTLASS_NVFP4_SM121 --expt-relaxed-constexpr
ifneq ($(SOTA_V6_CUTLASS_INCLUDE),)
SOTA_V6_CUTLASS_FLAGS += -I$(SOTA_V6_CUTLASS_INCLUDE)
endif
ifneq ($(SOTA_V6_CUTLASS_UTIL_INCLUDE),)
SOTA_V6_CUTLASS_FLAGS += -I$(SOTA_V6_CUTLASS_UTIL_INCLUDE)
endif
endif
CUTLASS_MOE_BENCH_ARGS ?= --tokens 32 --groups 256 --capacity 8 --warmup 5 --iterations 20
CUTLASS_ROUTE_MAJOR_MOE_BENCH_ARGS ?= --tokens 96 --groups 256 --capacity 3 --warmup 5 --iterations 20 --workspace-mb 1024
DENSE_ALPHA_MOE_BENCH_ARGS ?= --tokens 96 --groups 256 --capacity 96 --warmup 5 --iterations 20 --workspace-mb 1024

.PHONY: archive sota_v6_archive cold_archive artifact_check validate validate_layer0_attention_bf16 validate_layer0_dense_bf16 validate_layer0_bf16 validate_layer0_embedding_bf16 validate_layer0_prefill_bf16 validate_layer0_reference_bf16 validate_layer0_full_reference_bf16 publish package package_layer0_attention_bf16 package_layer0_dense_bf16 package_layer0_bf16 package_layer0_embedding_bf16 package_layer0_prefill_bf16 package_layer0_reference_bf16 package_layer0_full_reference_bf16 clean require_sm121 require_cutlass require_cutlass_sm121a require_latency_limit require_model_dir require_input_token benchmark_cutlass_moe benchmark_cutlass_route_major_moe benchmark_dense_alpha_moe validate_layer1_full_reference_bf16 validate_layer2_full_reference_bf16 validate_dense_chain_bf16 validate_layer3_router_bf16 validate_layer3_shared_expert_bf16 validate_layer3_routed_expert_nvfp4 validate_layer3_routed_expert_nvfp4_topk validate_dense_chain_layer3_routed_expert_nvfp4_topk validate_dense_chain_two_routed_expert_nvfp4_topk validate_dense_chain_four_routed_expert_nvfp4_topk validate_dense_chain_eight_routed_expert_nvfp4_topk package_layer1_full_reference_bf16 package_layer2_full_reference_bf16 package_dense_chain_bf16 package_layer3_router_bf16 package_layer3_shared_expert_bf16 package_layer3_routed_expert_nvfp4 package_layer3_routed_expert_nvfp4_topk package_dense_chain_layer3_routed_expert_nvfp4_topk package_dense_chain_two_routed_expert_nvfp4_topk package_dense_chain_four_routed_expert_nvfp4_topk package_dense_chain_eight_routed_expert_nvfp4_topk

archive: require_sm121 $(MODULE_ARCHIVE)

require_sm121:
	@test "$(CUDA_ARCH)" = "sm_121" || test "$(CUDA_ARCH)" = "sm_121a" || \
		{ echo "this firmware identity requires CUDA_ARCH=sm_121 or sm_121a" >&2; exit 2; }

require_cutlass_sm121a:
	@test "$(CUDA_ARCH)" = "sm_121a" || \
		{ echo "CUTLASS SM121 FP4 grouped kernels require CUDA_ARCH=sm_121a" >&2; exit 2; }

require_cutlass:
	@test "$(SOTA_V6_ENABLE_CUTLASS)" = "1" || \
		{ echo "set SOTA_V6_ENABLE_CUTLASS=1 for CUTLASS MoE benchmark" >&2; exit 2; }

require_latency_limit:
	@test "$(MAX_STAGE_MICROSECONDS)" != "0" || \
		{ echo "set MAX_STAGE_MICROSECONDS to the qualified stage limit" >&2; exit 2; }

require_model_dir:
	@test -n "$(GLM52_MODEL_DIR)" || \
		{ echo "set GLM52_MODEL_DIR to the live GLM artifact directory" >&2; exit 2; }

require_input_token:
	@test -n "$(GLM52_INPUT_TOKEN_ID)" || \
		{ echo "set GLM52_INPUT_TOKEN_ID to the embedding token id" >&2; exit 2; }

$(BUILD_DIRECTORY):
	mkdir -p $(BUILD_DIRECTORY)

$(BUILD_DIRECTORY)/sota_v6:
	mkdir -p $(BUILD_DIRECTORY)/sota_v6

$(HOST_OBJECT): source/spark_glm52_resident_decode_stage_module.c | $(BUILD_DIRECTORY)
	$(CC) -I$(REPOSITORY_ROOT)/include -Iinclude -Isource $(CFLAGS) \
		-DNDEBUG -fPIC -fvisibility=hidden -MMD -MP -c $< -o $@

$(CUDA_OBJECT): source/spark_glm52_resident_decode_stage_cuda.cu | $(BUILD_DIRECTORY)
	$(NVCC) $(NVCCFLAGS) -I$(REPOSITORY_ROOT)/include -Iinclude -Isource \
		-Xcompiler -fPIC -Xcompiler -fvisibility=hidden -MMD -MP -c $< -o $@

$(CUBLASLT_PLAN_OBJECT): source/spark_glm52_resident_decode_stage_cublaslt_plans.cu | $(BUILD_DIRECTORY)
	$(NVCC) $(NVCCFLAGS) -I$(REPOSITORY_ROOT)/include -Iinclude -Isource \
		-Xcompiler -fPIC -Xcompiler -fvisibility=hidden -MMD -MP -c $< -o $@

$(BUILD_DIRECTORY)/sota_v6/%.o: $(SOTA_V6_SOURCE_DIRECTORY)/%.cu | $(BUILD_DIRECTORY)/sota_v6
	$(NVCC) $(NVCCFLAGS) $(SOTA_V6_CUTLASS_FLAGS) -I$(REPOSITORY_ROOT)/include -Iinclude -Isource -I$(SOTA_V6_SOURCE_DIRECTORY) \
		-Xcompiler -fPIC -Xcompiler -fvisibility=hidden -MMD -MP -c $< -o $@

sota_v6_archive: require_sm121 $(SOTA_V6_OBJECTS)

$(CUTLASS_MOE_BENCHMARK_OBJECT): validation/spark_glm52_sota_cutlass_moe_benchmark.cu | $(BUILD_DIRECTORY)
	$(NVCC) $(NVCCFLAGS) $(SOTA_V6_CUTLASS_FLAGS) -I$(REPOSITORY_ROOT)/include -Iinclude -Isource -I$(SOTA_V6_SOURCE_DIRECTORY) \
		-MMD -MP -c $< -o $@

$(CUTLASS_MOE_BENCHMARK): $(CUTLASS_MOE_BENCHMARK_OBJECT) $(CUTLASS_MOE_BENCHMARK_DEPENDENCIES)
	$(NVCC) $(NVCCFLAGS) $(SOTA_V6_CUTLASS_FLAGS) $^ \
		-L$(CUDA_HOME)/lib64 -lcublasLt -lcublas -lcudart -o $@

$(CUTLASS_ROUTE_MAJOR_MOE_BENCHMARK_OBJECT): validation/spark_glm52_sota_cutlass_route_major_moe_benchmark.cu | $(BUILD_DIRECTORY)
	$(NVCC) $(NVCCFLAGS) $(SOTA_V6_CUTLASS_FLAGS) -I$(REPOSITORY_ROOT)/include -Iinclude -Isource -I$(SOTA_V6_SOURCE_DIRECTORY) \
		-MMD -MP -c $< -o $@

$(CUTLASS_ROUTE_MAJOR_MOE_BENCHMARK): $(CUTLASS_ROUTE_MAJOR_MOE_BENCHMARK_OBJECT)
	$(NVCC) $(NVCCFLAGS) $(SOTA_V6_CUTLASS_FLAGS) $^ \
		-L$(CUDA_HOME)/lib64 -lcudart -o $@

$(DENSE_ALPHA_MOE_BENCHMARK_OBJECT): validation/spark_glm52_sota_dense_alpha_moe_benchmark.cu | $(BUILD_DIRECTORY)
	$(NVCC) $(NVCCFLAGS) $(SOTA_V6_CUTLASS_FLAGS) -I$(REPOSITORY_ROOT)/include -Iinclude -Isource -I$(SOTA_V6_SOURCE_DIRECTORY) \
		-MMD -MP -c $< -o $@

$(DENSE_ALPHA_MOE_BENCHMARK): $(DENSE_ALPHA_MOE_BENCHMARK_OBJECT) $(DENSE_ALPHA_MOE_BENCHMARK_DEPENDENCIES)
	$(NVCC) $(NVCCFLAGS) $(SOTA_V6_CUTLASS_FLAGS) $^ \
		-L$(CUDA_HOME)/lib64 -lcublasLt -lcublas -lcudart -o $@

benchmark_cutlass_moe: require_sm121 require_cutlass require_cutlass_sm121a $(CUTLASS_MOE_BENCHMARK)
	$(CUTLASS_MOE_BENCHMARK) $(CUTLASS_MOE_BENCH_ARGS)

benchmark_cutlass_route_major_moe: require_sm121 require_cutlass require_cutlass_sm121a $(CUTLASS_ROUTE_MAJOR_MOE_BENCHMARK)
	$(CUTLASS_ROUTE_MAJOR_MOE_BENCHMARK) $(CUTLASS_ROUTE_MAJOR_MOE_BENCH_ARGS)

benchmark_dense_alpha_moe: require_sm121 require_cutlass require_cutlass_sm121a $(DENSE_ALPHA_MOE_BENCHMARK)
	$(DENSE_ALPHA_MOE_BENCHMARK) $(DENSE_ALPHA_MOE_BENCH_ARGS)

$(MODULE_ARCHIVE): $(HOST_OBJECT) $(CUDA_OBJECT) $(CUBLASLT_PLAN_OBJECT) $(SOTA_V6_OBJECTS)
	rm -f $@
	$(AR) rcs $@ $^

$(ARTIFACT_CHECK_TOOL): tools/glm52_artifact_check.c | $(BUILD_DIRECTORY)
	$(MAKE) -C $(REPOSITORY_ROOT) build/libsparkpipe_compiler.a build/libsparkpipe_common.a
	$(CC) -I$(REPOSITORY_ROOT)/include -I$(REPOSITORY_ROOT)/src -Iinclude $(CFLAGS) \
		$< $(REPOSITORY_ROOT)/build/libsparkpipe_compiler.a \
		$(REPOSITORY_ROOT)/build/libsparkpipe_common.a -ldl -o $@

cold_archive: require_sm121
	rm -rf $(BUILD_DIRECTORY)
	$(MAKE) archive

artifact_check: require_model_dir $(ARTIFACT_CHECK_TOOL)
	$(ARTIFACT_CHECK_TOOL) \
		--model-dir "$(GLM52_MODEL_DIR)" \
		--model "$(REPOSITORY_ROOT)/examples/model_descriptions/glm52_resident_decode_stage_firmware.json" \
		$(ARTIFACT_CHECK_ARGS)

validate: require_sm121 require_latency_limit cold_archive
	NVCC=$(NVCC) CUDA_ARCH=$(CUDA_ARCH) \
		validation/validate_glm52_resident_decode_stage_cuda.sh \
		$(MAX_STAGE_MICROSECONDS) $(MODULE_ARCHIVE)

validate_layer0_attention_bf16: require_model_dir
	GLM52_LOAD_LAYER0_ATTENTION_BF16=1 $(MAKE) validate

validate_layer0_dense_bf16: require_model_dir
	GLM52_LOAD_LAYER0_DENSE_BF16=1 $(MAKE) validate

validate_layer0_bf16: require_model_dir
	GLM52_LOAD_LAYER0_ATTENTION_BF16=1 GLM52_LOAD_LAYER0_DENSE_BF16=1 $(MAKE) validate

validate_layer0_embedding_bf16: require_model_dir require_input_token
	GLM52_LOAD_LAYER0_ATTENTION_BF16=1 GLM52_LOAD_LAYER0_DENSE_BF16=1 $(MAKE) validate

validate_layer0_prefill_bf16: require_model_dir require_input_token
	GLM52_LOAD_LAYER0_ATTENTION_BF16=1 GLM52_LOAD_LAYER0_DENSE_BF16=1 GLM52_PREFILL_KV_FROM_EMBEDDINGS=1 $(MAKE) validate

validate_layer0_reference_bf16: require_model_dir require_input_token
	GLM52_LOAD_LAYER0_ATTENTION_BF16=1 GLM52_LOAD_LAYER0_DENSE_BF16=1 GLM52_PREFILL_KV_FROM_EMBEDDINGS=1 GLM52_CHECK_LAYER0_REFERENCE=1 $(MAKE) validate

validate_layer0_full_reference_bf16: require_model_dir require_input_token
	GLM52_LOAD_LAYER0_ATTENTION_BF16=1 GLM52_LOAD_LAYER0_DENSE_BF16=1 GLM52_PREFILL_KV_FROM_EMBEDDINGS=1 GLM52_CHECK_LAYER0_REFERENCE=1 GLM52_CHECK_LAYER0_FULL_REFERENCE=1 $(MAKE) validate

validate_layer1_full_reference_bf16: require_model_dir require_input_token
	GLM52_DENSE_LAYER_INDEX=1 GLM52_LOAD_LAYER0_ATTENTION_BF16=1 GLM52_LOAD_LAYER0_DENSE_BF16=1 GLM52_PREFILL_KV_FROM_EMBEDDINGS=1 GLM52_CHECK_LAYER0_REFERENCE=1 GLM52_CHECK_LAYER0_FULL_REFERENCE=1 $(MAKE) validate

validate_layer2_full_reference_bf16: require_model_dir require_input_token
	GLM52_DENSE_LAYER_INDEX=2 GLM52_LOAD_LAYER0_ATTENTION_BF16=1 GLM52_LOAD_LAYER0_DENSE_BF16=1 GLM52_PREFILL_KV_FROM_EMBEDDINGS=1 GLM52_CHECK_LAYER0_REFERENCE=1 GLM52_CHECK_LAYER0_FULL_REFERENCE=1 $(MAKE) validate

validate_dense_chain_bf16: require_model_dir require_input_token
	GLM52_CHAIN_DENSE_LAYERS=1 VALIDATION_RECIPE_VARIANT=dense_chain_bf16 $(MAKE) validate

validate_layer3_router_bf16: require_model_dir require_input_token
	GLM52_LOAD_LAYER3_ROUTER_BF16=1 VALIDATION_RECIPE_VARIANT=layer3_router_bf16 $(MAKE) validate

validate_layer3_shared_expert_bf16: require_model_dir require_input_token
	GLM52_LOAD_LAYER3_SHARED_EXPERT_BF16=1 VALIDATION_RECIPE_VARIANT=layer3_shared_expert_bf16 $(MAKE) validate

validate_layer3_routed_expert_nvfp4: require_model_dir require_input_token
	GLM52_LOAD_LAYER3_ROUTED_EXPERT_NVFP4=1 VALIDATION_RECIPE_VARIANT=layer3_routed_expert_nvfp4 $(MAKE) validate

validate_layer3_routed_expert_nvfp4_topk: require_model_dir require_input_token
	GLM52_LOAD_LAYER3_ROUTED_EXPERT_NVFP4_TOPK=1 VALIDATION_RECIPE_VARIANT=layer3_routed_expert_nvfp4_topk $(MAKE) validate

validate_dense_chain_layer3_routed_expert_nvfp4_topk: require_model_dir require_input_token
	GLM52_CHAIN_DENSE_TO_LAYER3_ROUTED_EXPERT_NVFP4_TOPK=1 VALIDATION_RECIPE_VARIANT=dense_chain_layer3_routed_expert_nvfp4_topk $(MAKE) validate

validate_dense_chain_two_routed_expert_nvfp4_topk: require_model_dir require_input_token
	GLM52_CHAIN_DENSE_TO_LAYER3_ROUTED_EXPERT_NVFP4_TOPK=1 GLM52_ROUTED_CHAIN_LAYER_COUNT=2 VALIDATION_RECIPE_VARIANT=dense_chain_two_routed_expert_nvfp4_topk $(MAKE) validate

validate_dense_chain_four_routed_expert_nvfp4_topk: require_model_dir require_input_token
	GLM52_CHAIN_DENSE_TO_LAYER3_ROUTED_EXPERT_NVFP4_TOPK=1 GLM52_ROUTED_CHAIN_LAYER_COUNT=4 VALIDATION_RECIPE_VARIANT=dense_chain_four_routed_expert_nvfp4_topk $(MAKE) validate

validate_dense_chain_eight_routed_expert_nvfp4_topk: require_model_dir require_input_token
	GLM52_CHAIN_DENSE_TO_LAYER3_ROUTED_EXPERT_NVFP4_TOPK=1 GLM52_ROUTED_CHAIN_LAYER_COUNT=8 VALIDATION_RECIPE_VARIANT=dense_chain_eight_routed_expert_nvfp4_topk $(MAKE) validate

package_layer0_attention_bf16: require_model_dir
	GLM52_LOAD_LAYER0_ATTENTION_BF16=1 VALIDATION_RECIPE_VARIANT=layer0_attention_bf16 $(MAKE) package

package_layer0_dense_bf16: require_model_dir
	GLM52_LOAD_LAYER0_DENSE_BF16=1 VALIDATION_RECIPE_VARIANT=layer0_dense_bf16 $(MAKE) package

package_layer0_bf16: require_model_dir
	GLM52_LOAD_LAYER0_ATTENTION_BF16=1 GLM52_LOAD_LAYER0_DENSE_BF16=1 VALIDATION_RECIPE_VARIANT=layer0_bf16 $(MAKE) package

package_layer0_embedding_bf16: require_model_dir require_input_token
	GLM52_LOAD_LAYER0_ATTENTION_BF16=1 GLM52_LOAD_LAYER0_DENSE_BF16=1 VALIDATION_RECIPE_VARIANT=layer0_embedding_bf16 $(MAKE) package

package_layer0_prefill_bf16: require_model_dir require_input_token
	GLM52_LOAD_LAYER0_ATTENTION_BF16=1 GLM52_LOAD_LAYER0_DENSE_BF16=1 GLM52_PREFILL_KV_FROM_EMBEDDINGS=1 VALIDATION_RECIPE_VARIANT=layer0_prefill_bf16 $(MAKE) package

package_layer0_reference_bf16: require_model_dir require_input_token
	GLM52_LOAD_LAYER0_ATTENTION_BF16=1 GLM52_LOAD_LAYER0_DENSE_BF16=1 GLM52_PREFILL_KV_FROM_EMBEDDINGS=1 GLM52_CHECK_LAYER0_REFERENCE=1 VALIDATION_RECIPE_VARIANT=layer0_reference_bf16 $(MAKE) package

package_layer0_full_reference_bf16: require_model_dir require_input_token
	GLM52_LOAD_LAYER0_ATTENTION_BF16=1 GLM52_LOAD_LAYER0_DENSE_BF16=1 GLM52_PREFILL_KV_FROM_EMBEDDINGS=1 GLM52_CHECK_LAYER0_REFERENCE=1 GLM52_CHECK_LAYER0_FULL_REFERENCE=1 VALIDATION_RECIPE_VARIANT=layer0_full_reference_bf16 $(MAKE) package

package_layer1_full_reference_bf16: require_model_dir require_input_token
	GLM52_DENSE_LAYER_INDEX=1 GLM52_LOAD_LAYER0_ATTENTION_BF16=1 GLM52_LOAD_LAYER0_DENSE_BF16=1 GLM52_PREFILL_KV_FROM_EMBEDDINGS=1 GLM52_CHECK_LAYER0_REFERENCE=1 GLM52_CHECK_LAYER0_FULL_REFERENCE=1 VALIDATION_RECIPE_VARIANT=layer1_full_reference_bf16 $(MAKE) package

package_layer2_full_reference_bf16: require_model_dir require_input_token
	GLM52_DENSE_LAYER_INDEX=2 GLM52_LOAD_LAYER0_ATTENTION_BF16=1 GLM52_LOAD_LAYER0_DENSE_BF16=1 GLM52_PREFILL_KV_FROM_EMBEDDINGS=1 GLM52_CHECK_LAYER0_REFERENCE=1 GLM52_CHECK_LAYER0_FULL_REFERENCE=1 VALIDATION_RECIPE_VARIANT=layer2_full_reference_bf16 $(MAKE) package

package_dense_chain_bf16: require_model_dir require_input_token
	GLM52_CHAIN_DENSE_LAYERS=1 VALIDATION_RECIPE_VARIANT=dense_chain_bf16 $(MAKE) package

package_layer3_router_bf16: require_model_dir require_input_token
	GLM52_LOAD_LAYER3_ROUTER_BF16=1 VALIDATION_RECIPE_VARIANT=layer3_router_bf16 $(MAKE) package

package_layer3_shared_expert_bf16: require_model_dir require_input_token
	GLM52_LOAD_LAYER3_SHARED_EXPERT_BF16=1 VALIDATION_RECIPE_VARIANT=layer3_shared_expert_bf16 $(MAKE) package

package_layer3_routed_expert_nvfp4: require_model_dir require_input_token
	GLM52_LOAD_LAYER3_ROUTED_EXPERT_NVFP4=1 VALIDATION_RECIPE_VARIANT=layer3_routed_expert_nvfp4 $(MAKE) package

package_layer3_routed_expert_nvfp4_topk: require_model_dir require_input_token
	GLM52_LOAD_LAYER3_ROUTED_EXPERT_NVFP4_TOPK=1 VALIDATION_RECIPE_VARIANT=layer3_routed_expert_nvfp4_topk $(MAKE) package

package_dense_chain_layer3_routed_expert_nvfp4_topk: require_model_dir require_input_token
	GLM52_CHAIN_DENSE_TO_LAYER3_ROUTED_EXPERT_NVFP4_TOPK=1 VALIDATION_RECIPE_VARIANT=dense_chain_layer3_routed_expert_nvfp4_topk $(MAKE) package

package_dense_chain_two_routed_expert_nvfp4_topk: require_model_dir require_input_token
	GLM52_CHAIN_DENSE_TO_LAYER3_ROUTED_EXPERT_NVFP4_TOPK=1 GLM52_ROUTED_CHAIN_LAYER_COUNT=2 VALIDATION_RECIPE_VARIANT=dense_chain_two_routed_expert_nvfp4_topk $(MAKE) package

package_dense_chain_four_routed_expert_nvfp4_topk: require_model_dir require_input_token
	GLM52_CHAIN_DENSE_TO_LAYER3_ROUTED_EXPERT_NVFP4_TOPK=1 GLM52_ROUTED_CHAIN_LAYER_COUNT=4 VALIDATION_RECIPE_VARIANT=dense_chain_four_routed_expert_nvfp4_topk $(MAKE) package

package_dense_chain_eight_routed_expert_nvfp4_topk: require_model_dir require_input_token
	GLM52_CHAIN_DENSE_TO_LAYER3_ROUTED_EXPERT_NVFP4_TOPK=1 GLM52_ROUTED_CHAIN_LAYER_COUNT=8 VALIDATION_RECIPE_VARIANT=dense_chain_eight_routed_expert_nvfp4_topk $(MAKE) package

publish: require_sm121 require_latency_limit cold_archive
	$(MAKE) -C $(REPOSITORY_ROOT) build/sparkpipe_module_publish
	NVCC=$(NVCC) CUDA_ARCH=$(CUDA_ARCH) \
		$(REPOSITORY_ROOT)/build/sparkpipe_module_publish \
		--library $(MODULE_LIBRARY_ROOT) \
		--module spark.glm52.resident_decode_stage.bf16.h6144.h64.d512.r64.k2048.b64.rv256.mtp2.v1 \
		--target cuda.sm121.glm52.resident_decode_stage.bf16 \
		--link-unit $(MODULE_ARCHIVE) \
		--recipe $(VALIDATION_RECIPE) \
		--initialize SparkGlm52ResidentDecodeStageInitialize \
		--execute SparkGlm52ResidentDecodeStageExecute \
		--admit SparkGlm52ResidentDecodeStageAdmit \
		--snapshot SparkGlm52ResidentDecodeStageSnapshot \
		--destroy SparkGlm52ResidentDecodeStageDestroy \
		--validator validation/validate_glm52_resident_decode_stage_cuda.sh \
		--validator-arg $(MAX_STAGE_MICROSECONDS)

package: publish
	$(MAKE) -C $(REPOSITORY_ROOT) build/sparkpipe_model_compile
	rm -rf $(PACKAGE_DIRECTORY)
	$(REPOSITORY_ROOT)/build/sparkpipe_model_compile \
		--model $(REPOSITORY_ROOT)/examples/model_descriptions/glm52_resident_decode_stage_firmware.json \
		--library $(MODULE_LIBRARY_ROOT) \
		--output $(PACKAGE_DIRECTORY) \
		--include $(REPOSITORY_ROOT)/include \
		--cc cc \
		--cc-arg -L$(CUDA_HOME)/lib64 \
		--cc-arg -lcudart \
		--cc-arg -lcublasLt \
		--cc-arg -lcublas \
		--cc-arg -lstdc++
	NVCC=$(NVCC) CUDA_ARCH=$(CUDA_ARCH) \
		validation/validate_glm52_resident_decode_stage_cuda.sh \
		$(MAX_STAGE_MICROSECONDS) $(MODULE_ARCHIVE) \
		$(PACKAGE_DIRECTORY)/stages/stage_000/model_driver.so

clean:
	rm -rf $(BUILD_DIRECTORY)

-include $(HOST_DEPENDENCY) $(CUDA_DEPENDENCY) $(CUBLASLT_PLAN_DEPENDENCY) $(SOTA_V6_DEPENDENCIES) $(CUTLASS_MOE_BENCHMARK_OBJECT:.o=.d) $(CUTLASS_ROUTE_MAJOR_MOE_BENCHMARK_OBJECT:.o=.d) $(DENSE_ALPHA_MOE_BENCHMARK_OBJECT:.o=.d)
