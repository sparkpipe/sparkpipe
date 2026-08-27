ifndef EXPERT_CODEC
$(error EXPERT_CODEC is required: int6, int7, int8, fp8, nvfp4, or mxfp4)
endif
ifndef MODEL_REVISION
$(error MODEL_REVISION is required and must be the exact source snapshot revision)
endif
ifndef CONTRACT_SHA256
$(error CONTRACT_SHA256 is required and must identify the exact model package contract)
endif

GLM5_NEXT_EXPERT_CODECS := int6 int7 int8 fp8 nvfp4 mxfp4
ifeq ($(filter $(EXPERT_CODEC),$(GLM5_NEXT_EXPERT_CODECS)),)
$(error unsupported EXPERT_CODEC '$(EXPERT_CODEC)': choose $(GLM5_NEXT_EXPERT_CODECS))
endif

ifeq ($(EXPERT_CODEC),int6)
GLM5_NEXT_EXPERT_CODEC_ID := 2
endif
ifeq ($(EXPERT_CODEC),int7)
GLM5_NEXT_EXPERT_CODEC_ID := 3
endif
ifeq ($(EXPERT_CODEC),int8)
GLM5_NEXT_EXPERT_CODEC_ID := 4
endif
ifeq ($(EXPERT_CODEC),fp8)
GLM5_NEXT_EXPERT_CODEC_ID := 5
endif
ifeq ($(EXPERT_CODEC),nvfp4)
GLM5_NEXT_EXPERT_CODEC_ID := 6
endif
ifeq ($(EXPERT_CODEC),mxfp4)
GLM5_NEXT_EXPERT_CODEC_ID := 7
endif

MODULE_FAMILY := glm5_next
MODULE_IDENTIFIER_PREFIX := spark.glm5_next.resident_decode_stage.bf16.expert_$(EXPERT_CODEC).h4096.l45.kda34.e288.k8
MODULE_IDENTIFIER_SUFFIX := v2
MODULE_IDENTIFIER := $(MODULE_IDENTIFIER_PREFIX).$(MODULE_IDENTIFIER_SUFFIX)
# The batch-variant ladder: one archive per power of two, emitted by
# resident_decode_stage_rules.mk from the single template there. Trim for
# local iteration here, never by editing a variant.
MODULE_BATCH_VARIANT_BUCKETS ?= 1 2 4 8 16 32 64 128 256 512 1024
MODULE_TARGET := cuda.sm121.glm5_next.resident_decode_stage.bf16.expert_$(EXPERT_CODEC)
MODEL_HEADER := ../../model-families/glm5_next/include/sparkpipe/spark_glm5_next_model.h
MODULE_ENTRY_PREFIX := SparkGlm5NextResidentDecodeStage
MODULE_HOST_SOURCE := source/spark_glm5_next_resident_decode_stage_module.c
MODULE_ADDITIONAL_HOST_SOURCES := \
	../../ring/transport/hidden_transport.c \
	../../ring/transport/tp_collective.c \
	../../ring/transport/tp_device_collective.c \
	../../ring/transport/tp_device_collective_nccl.c
MODULE_CUDA_SOURCE := source/spark_glm5_next_resident_decode_stage_cuda.cu
MODULE_COMPILE_FLAGS := -DGLM5_NEXT_EXPERT_WEIGHT_CODEC=$(GLM5_NEXT_EXPERT_CODEC_ID) -DGLM5_NEXT_EXPERT_CODEC_NAME=\"$(EXPERT_CODEC)\" -DGLM5_NEXT_MODEL_REVISION=\"$(MODEL_REVISION)\" -DGLM5_NEXT_CONTRACT_SHA256=\"$(CONTRACT_SHA256)\"

BUILD_DIRECTORY ?= ../../build/modules/glm5_next_resident_decode_stage/$(EXPERT_CODEC)
MODULE_ARCHIVE ?= $(BUILD_DIRECTORY)/libglm5_next_resident_decode_stage_$(EXPERT_CODEC).a
MODULE_COMMON_HOST_SOURCES := ../../runtime/stage_module_common.c
MODULE_INCLUDE_FLAGS := \
	-I../../include \
	-I../../model-families/common/include \
	-I../../model-families/glm5_next/include \
	-I../../modules/glm5_next_resident_decode_stage/include \
	-I../../modules/glm5_next_resident_decode_stage/source \
	-I../..

STAGE_PACK_PATH ?= $(REPOSITORY_ROOT)/build/stagepacks/glm5_next_stage.$(EXPERT_CODEC).g5nsp
STAGE_INDEX ?= 0
MAX_ACTIVE_SEQUENCES ?= 64
MAX_SEQUENCE_POSITIONS ?= 32768
EXECUTION_ROW_CAPACITY ?= 64
PIPELINE_SLOT_COUNT ?= 2

RUNTIME_CONFIGURATION := \
	SPARK_GLM5_NEXT_EXPERT_CODEC=$(EXPERT_CODEC) \
	SPARK_GLM5_NEXT_STAGE_PACK_PATH=$(STAGE_PACK_PATH) \
	SPARK_GLM5_NEXT_STAGE_INDEX=$(STAGE_INDEX) \
	SPARK_GLM5_NEXT_STAGE_MAX_ACTIVE_SEQUENCES=$(MAX_ACTIVE_SEQUENCES) \
	SPARK_GLM5_NEXT_STAGE_MAX_SEQUENCE_POSITIONS=$(MAX_SEQUENCE_POSITIONS) \
	SPARK_GLM5_NEXT_STAGE_EXECUTION_ROW_CAPACITY=$(EXECUTION_ROW_CAPACITY) \
	SPARK_GLM5_NEXT_STAGE_PIPELINE_SLOTS=$(PIPELINE_SLOT_COUNT)

GPU_VALIDATOR := validation/validate_glm5_next_resident_decode_stage_cuda.sh

include ../resident_decode_stage_rules.mk

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
ADAPTER_SHARED_FLAGS := -dynamiclib
ADAPTER_LIBRARY_EXT := dylib
else
ADAPTER_SHARED_FLAGS := -shared
ADAPTER_LIBRARY_EXT := so
endif
ADAPTER_LIBRARY := $(BUILD_DIRECTORY)/libglm5_next_serving_adapter_$(EXPERT_CODEC).$(ADAPTER_LIBRARY_EXT)

.PHONY: adapter

adapter:
	$(MAKE) -C $(REPOSITORY_ROOT) core model_common
	@mkdir -p "$(BUILD_DIRECTORY)"
	$(CC) $(CFLAGS) $(MODULE_POSIX_FLAGS) -fPIC $(ADAPTER_SHARED_FLAGS) \
		-I$(REPOSITORY_ROOT)/src $(MODULE_INCLUDE_FLAGS) $(MODULE_COMPILE_FLAGS) -USPARK_BATCH_BUCKET -DSPARK_BATCH_BUCKET=$(MODULE_BATCH_VARIANT_BUCKETS) \
		source/spark_glm5_next_serving_adapter.c \
		$(REPOSITORY_ROOT)/build/libsparkpipe_runtime.a \
		$(REPOSITORY_ROOT)/build/libsparkpipe_model_common.a \
		$(REPOSITORY_ROOT)/build/libsparkpipe_core.a -ldl -pthread \
		-o "$(ADAPTER_LIBRARY)"
