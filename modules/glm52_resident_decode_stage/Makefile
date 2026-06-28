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
HOST_DEPENDENCY := $(HOST_OBJECT:.o=.d)
CUDA_DEPENDENCY := $(CUDA_OBJECT:.o=.d)
MODULE_ARCHIVE := $(BUILD_DIRECTORY)/libglm52_resident_decode_stage.a
ARTIFACT_CHECK_TOOL := $(BUILD_DIRECTORY)/glm52_artifact_check
PACKAGE_DIRECTORY ?= $(REPOSITORY_ROOT)/build/packages/glm52_resident_decode_stage
CUDA_HOME ?= $(abspath $(dir $(shell command -v $(NVCC)))/..)
VALIDATION_RECIPE_VARIANT ?= default
VALIDATION_RECIPE := glm52.resident_decode_stage.$(CUDA_ARCH).$(VALIDATION_RECIPE_VARIANT).max_us_$(MAX_STAGE_MICROSECONDS).v1

.PHONY: archive cold_archive artifact_check validate validate_layer0_attention_bf16 validate_layer0_dense_bf16 validate_layer0_bf16 validate_layer0_embedding_bf16 validate_layer0_prefill_bf16 validate_layer0_reference_bf16 validate_layer0_full_reference_bf16 publish package package_layer0_attention_bf16 package_layer0_dense_bf16 package_layer0_bf16 package_layer0_embedding_bf16 package_layer0_prefill_bf16 package_layer0_reference_bf16 package_layer0_full_reference_bf16 clean require_sm121 require_latency_limit require_model_dir require_input_token

archive: require_sm121 $(MODULE_ARCHIVE)

require_sm121:
	@test "$(CUDA_ARCH)" = "sm_121" || \
		{ echo "this firmware identity requires CUDA_ARCH=sm_121" >&2; exit 2; }

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

$(HOST_OBJECT): source/spark_glm52_resident_decode_stage_module.c | $(BUILD_DIRECTORY)
	$(CC) -I$(REPOSITORY_ROOT)/include -Iinclude -Isource $(CFLAGS) \
		-DNDEBUG -fPIC -fvisibility=hidden -MMD -MP -c $< -o $@

$(CUDA_OBJECT): source/spark_glm52_resident_decode_stage_cuda.cu | $(BUILD_DIRECTORY)
	$(NVCC) $(NVCCFLAGS) -I$(REPOSITORY_ROOT)/include -Iinclude -Isource \
		-Xcompiler -fPIC -Xcompiler -fvisibility=hidden -MMD -MP -c $< -o $@

$(MODULE_ARCHIVE): $(HOST_OBJECT) $(CUDA_OBJECT)
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
		--cc-arg -lstdc++
	NVCC=$(NVCC) CUDA_ARCH=$(CUDA_ARCH) \
		validation/validate_glm52_resident_decode_stage_cuda.sh \
		$(MAX_STAGE_MICROSECONDS) $(MODULE_ARCHIVE) \
		$(PACKAGE_DIRECTORY)/stages/stage_000/model_driver.so

clean:
	rm -rf $(BUILD_DIRECTORY)

-include $(HOST_DEPENDENCY) $(CUDA_DEPENDENCY)
