ifndef EXPERT_CODEC
$(error EXPERT_CODEC is required: bf16, int6, int7, int8, fp8, nvfp4, or mxfp4)
endif
ifndef MODEL_REVISION
$(error MODEL_REVISION is required and must be the exact source snapshot revision)
endif
ifndef CONTRACT_SHA256
$(error CONTRACT_SHA256 is required and must identify the exact model package contract)
endif

LING_EXPERT_CODECS := bf16 int6 int7 int8 fp8 nvfp4 mxfp4
ifeq ($(filter $(EXPERT_CODEC),$(LING_EXPERT_CODECS)),)
$(error unsupported EXPERT_CODEC '$(EXPERT_CODEC)': choose $(LING_EXPERT_CODECS))
endif

ifeq ($(EXPERT_CODEC),bf16)
LING_EXPERT_CODEC_ID := 1
endif
ifeq ($(EXPERT_CODEC),int6)
LING_EXPERT_CODEC_ID := 2
endif
ifeq ($(EXPERT_CODEC),int7)
LING_EXPERT_CODEC_ID := 3
endif
ifeq ($(EXPERT_CODEC),int8)
LING_EXPERT_CODEC_ID := 4
endif
ifeq ($(EXPERT_CODEC),fp8)
LING_EXPERT_CODEC_ID := 5
endif
ifeq ($(EXPERT_CODEC),nvfp4)
LING_EXPERT_CODEC_ID := 6
endif
ifeq ($(EXPERT_CODEC),mxfp4)
LING_EXPERT_CODEC_ID := 7
endif

MODULE_FAMILY := ling
MODULE_IDENTIFIER_PREFIX := spark.ling.resident_decode_stage.bf16.expert_$(EXPERT_CODEC).h2560.l42.kda35.e512.k8
MODULE_IDENTIFIER_SUFFIX := v2
MODULE_IDENTIFIER := $(MODULE_IDENTIFIER_PREFIX).$(MODULE_IDENTIFIER_SUFFIX)
# The batch-variant ladder: one archive per power of two, emitted by
# resident_decode_stage_rules.mk from the single template there. Trim for
# local iteration here, never by editing a variant.
MODULE_BATCH_VARIANT_BUCKETS ?= 1 2 4 8 16 32 64 128 256 512 1024
MODULE_TARGET := cuda.sm121.ling.resident_decode_stage.bf16.expert_$(EXPERT_CODEC)
MODEL_HEADER := ../../model-families/ling/include/sparkpipe/spark_ling_model.h
MODULE_ENTRY_PREFIX := SparkLingResidentDecodeStage
MODULE_HOST_SOURCE := source/spark_ling_resident_decode_stage_module.c
MODULE_ADDITIONAL_HOST_SOURCES := \
	../../runtime/stage_module_lifecycle.c \
	../../src/spark_speculation_policy.c \
	../../runtime/spark_weightd.c \
	../../runtime/spark_weightd_attach.c \
	../../src/spark_status.c \
	../../src/spark_sha256.c \
	../../src/spark_ck128.c \
	../../ring/transport/hidden_transport.c \
	../../ring/transport/tp_collective.c \
	../../ring/transport/tp_device_collective.c \
	../../ring/transport/tp_device_collective_nccl.c \
	../../cache/kv_cache.c \
	../../cache/kv_page_cache.c \
	../../cache/kv_page_store.c \
	../../cache/kv_model_table.c
MODULE_CUDA_SOURCE := source/spark_ling_resident_decode_stage_cuda.cu
MODULE_COMPILE_FLAGS := -DLING_EXPERT_WEIGHT_CODEC=$(LING_EXPERT_CODEC_ID) -DLING_EXPERT_CODEC_NAME=\"$(EXPERT_CODEC)\" -DLING_MODEL_REVISION=\"$(MODEL_REVISION)\" -DLING_CONTRACT_SHA256=\"$(CONTRACT_SHA256)\"

BUILD_DIRECTORY ?= ../../build/modules/ling_resident_decode_stage/$(EXPERT_CODEC)
MODULE_ARCHIVE ?= $(BUILD_DIRECTORY)/libling_resident_decode_stage_$(EXPERT_CODEC).a
MODULE_COMMON_HOST_SOURCES := ../../runtime/stage_module_common.c
MODULE_INCLUDE_FLAGS := \
	-I../../include \
	-I../../model-families/common/include \
	-I../../model-families/ling/include \
	-I../../modules/ling_resident_decode_stage/include \
	-I../../modules/ling_resident_decode_stage/source \
	-I../..

STAGE_INDEX ?= 0
MAX_ACTIVE_SEQUENCES ?= 64
MAX_SEQUENCE_POSITIONS ?= 32768
EXECUTION_ROW_CAPACITY ?= 64
PIPELINE_SLOT_COUNT ?= 2

RUNTIME_CONFIGURATION := \
	SPARK_LING_EXPERT_CODEC=$(EXPERT_CODEC) \
	SPARK_LING_STAGE_INDEX=$(STAGE_INDEX) \
	SPARK_LING_STAGE_MAX_ACTIVE_SEQUENCES=$(MAX_ACTIVE_SEQUENCES) \
	SPARK_LING_STAGE_MAX_SEQUENCE_POSITIONS=$(MAX_SEQUENCE_POSITIONS) \
	SPARK_LING_STAGE_EXECUTION_ROW_CAPACITY=$(EXECUTION_ROW_CAPACITY) \
	SPARK_LING_STAGE_PIPELINE_SLOTS=$(PIPELINE_SLOT_COUNT)

GPU_VALIDATOR := validation/validate_ling_resident_decode_stage_cuda.sh

include ../resident_decode_stage_rules.mk

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
ADAPTER_SHARED_FLAGS := -dynamiclib
ADAPTER_LIBRARY_EXT := dylib
else
ADAPTER_SHARED_FLAGS := -shared
ADAPTER_LIBRARY_EXT := so
endif
ADAPTER_LIBRARY := $(BUILD_DIRECTORY)/libling_serving_adapter_$(EXPERT_CODEC).$(ADAPTER_LIBRARY_EXT)

.PHONY: adapter

adapter:
	$(MAKE) -C $(REPOSITORY_ROOT) core model_common
	@mkdir -p "$(BUILD_DIRECTORY)"
	$(CC) $(CFLAGS) $(MODULE_POSIX_FLAGS) -fPIC $(ADAPTER_SHARED_FLAGS) \
		-I$(REPOSITORY_ROOT)/src $(MODULE_INCLUDE_FLAGS) $(MODULE_COMPILE_FLAGS) -USPARK_BATCH_BUCKET -DSPARK_BATCH_BUCKET=$(lastword $(MODULE_BATCH_VARIANT_BUCKETS)) \
		source/spark_ling_serving_adapter.c \
		$(REPOSITORY_ROOT)/build/libsparkpipe_runtime.a \
		$(REPOSITORY_ROOT)/build/libsparkpipe_model_common.a \
		$(REPOSITORY_ROOT)/build/libsparkpipe_core.a -ldl -pthread \
		-o "$(ADAPTER_LIBRARY)"
