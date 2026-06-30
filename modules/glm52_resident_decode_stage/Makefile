REPOSITORY_ROOT := $(abspath ../..)
CC ?= cc
AR ?= ar
NVCC ?= nvcc
CUDA_ARCH ?= sm_121a
CFLAGS ?= -std=c11 -Wall -Wextra -Werror -O3
NVCCFLAGS ?= -std=c++17 -O3 --use_fast_math -lineinfo -arch=$(CUDA_ARCH)
MODULE_LIBRARY_ROOT ?= $(REPOSITORY_ROOT)/build/module_library
MAX_STAGE_MICROSECONDS ?= 0
ARTIFACT_CHECK_ARGS ?=
AOT_MANIFEST ?= $(REPOSITORY_ROOT)/build/glm52_b12x_aot/generated/aot_manifest.json
B12X_MOE_PACK_DIR ?= $(REPOSITORY_ROOT)/build/glm52_b12x_resident_moe
B12X_MOE_PACK_LAYERS ?= 3,4,5,6,7,8,9,10
B12X_PACK_PYTHON ?= python3
GLM52_VALIDATION_INPUT_TOKEN_ID ?= 1000
GLM52_VALIDATION_ROUTED_CHAIN_LAYER_COUNT ?= 1
BUILD_DIRECTORY := $(REPOSITORY_ROOT)/build/modules/glm52_resident_decode_stage
HOST_OBJECT := $(BUILD_DIRECTORY)/spark_glm52_resident_decode_stage_module.o
CUDA_OBJECT := $(BUILD_DIRECTORY)/spark_glm52_resident_decode_stage_cuda.o
REQUIRED_STAGE_OBJECT := $(BUILD_DIRECTORY)/spark_glm52_sm121_required_decode_stage.o
B12X_MOE_PLAN_OBJECT := $(BUILD_DIRECTORY)/spark_glm52_resident_decode_stage_b12x_moe_plan.o
LINEAR_PLAN_OBJECT := $(BUILD_DIRECTORY)/spark_glm52_resident_decode_stage_linear_plan.o
MODULE_ARCHIVE := $(BUILD_DIRECTORY)/libglm52_resident_decode_stage.a
ARTIFACT_CHECK_TOOL := $(BUILD_DIRECTORY)/glm52_artifact_check
PACKAGE_DIRECTORY ?= $(REPOSITORY_ROOT)/build/packages/glm52_resident_decode_stage
CUDA_HOME ?= $(abspath $(dir $(shell command -v $(NVCC)))/..)
VALIDATION_RECIPE_VARIANT ?= required_cuda
VALIDATION_RECIPE := glm52.resident_decode_stage.$(CUDA_ARCH).$(VALIDATION_RECIPE_VARIANT).max_us_$(MAX_STAGE_MICROSECONDS).v1
REQUIRED_CUDA_CC_ARGS ?=
GLM52_REQUIRED_CUDA_LINK_ARGS ?=
HOST_DEPENDENCY := $(HOST_OBJECT:.o=.d)
CUDA_DEPENDENCY := $(CUDA_OBJECT:.o=.d)
REQUIRED_STAGE_DEPENDENCY := $(REQUIRED_STAGE_OBJECT:.o=.d)
B12X_MOE_PLAN_DEPENDENCY := $(B12X_MOE_PLAN_OBJECT:.o=.d)
LINEAR_PLAN_DEPENDENCY := $(LINEAR_PLAN_OBJECT:.o=.d)

.PHONY: archive cold_archive artifact_check b12x_resident_pack validate publish package clean require_sm121a require_latency_limit require_model_dir

archive: require_sm121a $(MODULE_ARCHIVE)

cold_archive: clean archive

require_sm121a:
	@test "$(CUDA_ARCH)" = "sm_121a" || \
		{ echo "GLM52 required CUDA module contract requires CUDA_ARCH=sm_121a" >&2; exit 2; }

require_latency_limit:
	@test "$(MAX_STAGE_MICROSECONDS)" != "0" || \
		{ echo "set MAX_STAGE_MICROSECONDS to the qualified stage limit" >&2; exit 2; }

require_model_dir:
	@test -n "$(GLM52_MODEL_DIR)" || \
		{ echo "set GLM52_MODEL_DIR to the live GLM artifact directory" >&2; exit 2; }

$(BUILD_DIRECTORY):
	mkdir -p $(BUILD_DIRECTORY)

$(HOST_OBJECT): source/spark_glm52_resident_decode_stage_module.c | $(BUILD_DIRECTORY)
	$(CC) -I$(REPOSITORY_ROOT)/include -Iinclude -Isource $(CFLAGS) \
		-DNDEBUG -DSPARK_GLM52_RESIDENT_DECODE_STAGE_REQUIRE_EXTERNAL_CUDA_MODULES \
		-fPIC -fvisibility=hidden -MMD -MP -c $< -o $@

$(CUDA_OBJECT): source/spark_glm52_resident_decode_stage_cuda.cu | $(BUILD_DIRECTORY)
	$(NVCC) $(NVCCFLAGS) -I$(REPOSITORY_ROOT)/include -Iinclude -Isource \
		-Xcompiler -fPIC -Xcompiler -fvisibility=hidden -MMD -MP -c $< -o $@

$(REQUIRED_STAGE_OBJECT): source/spark_glm52_sm121_required_decode_stage.cu | $(BUILD_DIRECTORY)
	$(NVCC) $(NVCCFLAGS) -I$(REPOSITORY_ROOT)/include -Iinclude -Isource \
		-Xcompiler -fPIC -Xcompiler -fvisibility=hidden -MMD -MP -c $< -o $@

$(B12X_MOE_PLAN_OBJECT): source/spark_glm52_resident_decode_stage_b12x_moe_plan.cu | $(BUILD_DIRECTORY)
	$(NVCC) $(NVCCFLAGS) -I$(REPOSITORY_ROOT)/include -Iinclude -Isource \
		-Xcompiler -fPIC -Xcompiler -fvisibility=hidden -MMD -MP -c $< -o $@

$(LINEAR_PLAN_OBJECT): source/spark_glm52_resident_decode_stage_linear_plan.cu | $(BUILD_DIRECTORY)
	$(NVCC) $(NVCCFLAGS) -I$(REPOSITORY_ROOT)/include -Iinclude -Isource \
		-Xcompiler -fPIC -Xcompiler -fvisibility=hidden -MMD -MP -c $< -o $@

$(MODULE_ARCHIVE): $(HOST_OBJECT) $(CUDA_OBJECT) $(REQUIRED_STAGE_OBJECT) $(B12X_MOE_PLAN_OBJECT) $(LINEAR_PLAN_OBJECT)
	rm -f $@
	$(AR) rcs $@ $^

$(ARTIFACT_CHECK_TOOL): tools/glm52_artifact_check.c | $(BUILD_DIRECTORY)
	$(CC) -I$(REPOSITORY_ROOT)/include -Iinclude -Isource $(CFLAGS) \
		$< $(REPOSITORY_ROOT)/build/libsparkpipe_compiler.a $(REPOSITORY_ROOT)/build/libsparkpipe_common.a \
		-ldl -o $@

artifact_check: require_model_dir $(ARTIFACT_CHECK_TOOL)
	$(ARTIFACT_CHECK_TOOL) \
		--model-dir "$(GLM52_MODEL_DIR)" \
		--model "$(REPOSITORY_ROOT)/examples/model_descriptions/glm52_resident_decode_stage_firmware.json" \
		$(ARTIFACT_CHECK_ARGS)

b12x_resident_pack: require_model_dir
	@test -s "$(AOT_MANIFEST)" || \
		{ echo "missing B12x AOT manifest: $(AOT_MANIFEST)" >&2; exit 2; }
	$(B12X_PACK_PYTHON) $(REPOSITORY_ROOT)/tools/glm52_b12x_resident_pack.py \
		--model-dir "$(GLM52_MODEL_DIR)" \
		--aot-manifest "$(AOT_MANIFEST)" \
		--layers "$(B12X_MOE_PACK_LAYERS)" \
		--output-dir "$(B12X_MOE_PACK_DIR)"

validate: require_sm121a require_latency_limit b12x_resident_pack cold_archive
	NVCC=$(NVCC) CUDA_ARCH=$(CUDA_ARCH) GLM52_REQUIRED_CUDA_LINK_ARGS="$(GLM52_REQUIRED_CUDA_LINK_ARGS)" GLM52_B12X_MOE_PACK_DIR="$(B12X_MOE_PACK_DIR)" GLM52_MODEL_DIR="$(GLM52_MODEL_DIR)" GLM52_INPUT_TOKEN_ID="$(GLM52_VALIDATION_INPUT_TOKEN_ID)" GLM52_CHAIN_DENSE_TO_LAYER3_ROUTED_EXPERT_NVFP4_TOPK=1 GLM52_ROUTED_CHAIN_LAYER_COUNT="$(GLM52_VALIDATION_ROUTED_CHAIN_LAYER_COUNT)" \
		validation/validate_glm52_resident_decode_stage_cuda.sh \
		$(MAX_STAGE_MICROSECONDS) $(MODULE_ARCHIVE)

publish: require_sm121a require_latency_limit b12x_resident_pack cold_archive
	$(MAKE) -C $(REPOSITORY_ROOT) build/sparkpipe_module_publish
	NVCC=$(NVCC) CUDA_ARCH=$(CUDA_ARCH) GLM52_REQUIRED_CUDA_LINK_ARGS="$(GLM52_REQUIRED_CUDA_LINK_ARGS)" GLM52_B12X_MOE_PACK_DIR="$(B12X_MOE_PACK_DIR)" GLM52_MODEL_DIR="$(GLM52_MODEL_DIR)" GLM52_INPUT_TOKEN_ID="$(GLM52_VALIDATION_INPUT_TOKEN_ID)" GLM52_CHAIN_DENSE_TO_LAYER3_ROUTED_EXPERT_NVFP4_TOPK=1 GLM52_ROUTED_CHAIN_LAYER_COUNT="$(GLM52_VALIDATION_ROUTED_CHAIN_LAYER_COUNT)" \
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
		--cc-arg -lstdc++ \
		$(REQUIRED_CUDA_CC_ARGS)
	mkdir -p $(PACKAGE_DIRECTORY)/artifacts/glm52_b12x_resident_moe
	cp -f $(B12X_MOE_PACK_DIR)/*.spb12x $(PACKAGE_DIRECTORY)/artifacts/glm52_b12x_resident_moe/
	cp -f $(B12X_MOE_PACK_DIR)/resident_moe_pack_manifest.json $(PACKAGE_DIRECTORY)/artifacts/glm52_b12x_resident_moe/
	NVCC=$(NVCC) CUDA_ARCH=$(CUDA_ARCH) GLM52_REQUIRED_CUDA_LINK_ARGS="$(GLM52_REQUIRED_CUDA_LINK_ARGS)" GLM52_B12X_MOE_PACK_DIR="$(B12X_MOE_PACK_DIR)" GLM52_MODEL_DIR="$(GLM52_MODEL_DIR)" GLM52_INPUT_TOKEN_ID="$(GLM52_VALIDATION_INPUT_TOKEN_ID)" GLM52_CHAIN_DENSE_TO_LAYER3_ROUTED_EXPERT_NVFP4_TOPK=1 GLM52_ROUTED_CHAIN_LAYER_COUNT="$(GLM52_VALIDATION_ROUTED_CHAIN_LAYER_COUNT)" \
		validation/validate_glm52_resident_decode_stage_cuda.sh \
		$(MAX_STAGE_MICROSECONDS) $(MODULE_ARCHIVE) \
		$(PACKAGE_DIRECTORY)/stages/stage_000/model_driver.so

clean:
	rm -rf $(BUILD_DIRECTORY)

-include $(HOST_DEPENDENCY) $(CUDA_DEPENDENCY) $(REQUIRED_STAGE_DEPENDENCY) $(B12X_MOE_PLAN_DEPENDENCY) $(LINEAR_PLAN_DEPENDENCY)
