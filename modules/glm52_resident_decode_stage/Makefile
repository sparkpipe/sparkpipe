include ../../runtime/weightd_sources.mk

GLM_FAMILY := glm52
MODULE_FAMILY := glm52
GLM_REPO_ROOT := ../..
GLM_BUILD_ROOT := ../../build/modules
GLM_MODULE_ROOT := .
GLM_EXPERT_CODECS := bf16 int6 int7 int8 fp8 nvfp4 mxfp4
MODULE_IDENTIFIER_PREFIX := spark.glm52.resident_decode_stage.bf16.expert_$(EXPERT_CODEC).h6144.l78.e256.k8
MODULE_IDENTIFIER_SUFFIX := v2
MODULE_TARGET := cuda.sm121.glm52.resident_decode_stage.bf16.expert_$(EXPERT_CODEC)
MODEL_HEADER := ../../model-families/glm52/include/sparkpipe/spark_glm52_model.h
MODULE_ENTRY_PREFIX := SparkGlm52ResidentDecodeStage
MODULE_HOST_SOURCE := source/spark_glm52_resident_decode_stage_module.c
MODULE_ADDITIONAL_HOST_SOURCES := \
	../../runtime/stage_module_lifecycle.c \
	../../src/spark_status.c \
	../../src/spark_sha256.c \
	../../src/spark_ck128.c \
	../../ring/transport/hidden_transport.c \
	../../ring/transport/tp_collective.c \
	../../ring/transport/tp_device_collective.c \
	$(addprefix ../../,$(SPARKPIPE_WEIGHTD_SOURCES)) \
	../../cache/kv_cache.c \
	../../cache/kv_page_cache.c \
	../../cache/kv_page_store.c \
	../../cache/kv_model_table.c
MODULE_CUDA_SOURCE := source/spark_glm52_resident_decode_stage_cuda.cu

include ../../common/glm_resident_stage_wrapper.mk

MODULE_POSIX_FLAGS := -D_GNU_SOURCE -D_FILE_OFFSET_BITS=64
MODEL_DESCRIPTION ?= ../../examples/model_descriptions/glm52_resident_decode_stage_$(EXPERT_CODEC)_firmware.json
MODEL_DESCRIPTION_SHA256 ?= $(shell sha256sum "$(MODEL_DESCRIPTION)" | awk '{print $$1}')
MODULE_COMPILE_FLAGS += -DGLM_MODEL_DESCRIPTION_SHA256=\"$(MODEL_DESCRIPTION_SHA256)\"

STAGE_PACK_PATH ?= $(REPOSITORY_ROOT)/build/stagepacks/glm52_stage.$(EXPERT_CODEC).glm52sp
STAGE_INDEX ?= 0
MAX_ACTIVE_SEQUENCES ?= 64
MAX_SEQUENCE_POSITIONS ?= 32768
EXECUTION_ROW_CAPACITY ?= 64
PIPELINE_SLOT_COUNT ?= 2

override GLM52_CUDA_VALIDATOR_SHA256 := $(shell sha256sum validation/spark_glm52_resident_decode_stage_cuda_validation.cu | awk '{print $$1}')

RUNTIME_CONFIGURATION := \
	SPARK_GLM52_EXPERT_CODEC=$(EXPERT_CODEC) \
	SPARK_GLM52_STAGE_PACK_PATH=$(STAGE_PACK_PATH) \
	SPARK_GLM52_STAGE_INDEX=$(STAGE_INDEX) \
	SPARK_GLM52_STAGE_MAX_ACTIVE_SEQUENCES=$(MAX_ACTIVE_SEQUENCES) \
	SPARK_GLM52_STAGE_MAX_SEQUENCE_POSITIONS=$(MAX_SEQUENCE_POSITIONS) \
	SPARK_GLM52_STAGE_EXECUTION_ROW_CAPACITY=$(EXECUTION_ROW_CAPACITY) \
	SPARK_GLM52_STAGE_PIPELINE_SLOTS=$(PIPELINE_SLOT_COUNT) \
	SPARK_GLM52_CUDA_VALIDATOR_SHA256=$(GLM52_CUDA_VALIDATOR_SHA256)

GPU_VALIDATOR := validation/validate_glm52_resident_decode_stage_cuda.sh

include ../resident_decode_stage_rules.mk
