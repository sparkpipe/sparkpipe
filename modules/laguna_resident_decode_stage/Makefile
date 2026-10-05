include ../../runtime/weightd_sources.mk

ifndef EXPERT_CODEC
$(error EXPERT_CODEC is required: bf16, int6, int7, int8, fp8, nvfp4, or mxfp4)
endif
ifndef MODEL_REVISION
$(error MODEL_REVISION is required and must be the exact source snapshot revision)
endif
ifndef CONTRACT_SHA256
$(error CONTRACT_SHA256 is required and must identify the exact model package contract)
endif

LAGUNA_EXPERT_CODECS := bf16 int6 int7 int8 fp8 nvfp4 mxfp4
ifeq ($(filter $(EXPERT_CODEC),$(LAGUNA_EXPERT_CODECS)),)
$(error unsupported EXPERT_CODEC '$(EXPERT_CODEC)': choose $(LAGUNA_EXPERT_CODECS))
endif

ifeq ($(EXPERT_CODEC),bf16)
LAGUNA_EXPERT_CODEC_ID := 1
endif
ifeq ($(EXPERT_CODEC),int6)
LAGUNA_EXPERT_CODEC_ID := 2
endif
ifeq ($(EXPERT_CODEC),int7)
LAGUNA_EXPERT_CODEC_ID := 3
endif
ifeq ($(EXPERT_CODEC),int8)
LAGUNA_EXPERT_CODEC_ID := 4
endif
ifeq ($(EXPERT_CODEC),fp8)
LAGUNA_EXPERT_CODEC_ID := 5
endif
ifeq ($(EXPERT_CODEC),nvfp4)
LAGUNA_EXPERT_CODEC_ID := 6
endif
ifeq ($(EXPERT_CODEC),mxfp4)
LAGUNA_EXPERT_CODEC_ID := 7
endif

MODULE_FAMILY := laguna
MODULE_IDENTIFIER_PREFIX := spark.laguna.resident_decode_stage.bf16.expert_$(EXPERT_CODEC).h3072.l48.gqa.e256.k10
MODULE_IDENTIFIER_SUFFIX := v2
MODULE_IDENTIFIER := $(MODULE_IDENTIFIER_PREFIX).$(MODULE_IDENTIFIER_SUFFIX)
MODULE_BATCH_VARIANT_BUCKETS ?= 1 2 4 8 16 32 64 128 256 512 1024
MODULE_TARGET := cuda.sm121.laguna.resident_decode_stage.bf16.expert_$(EXPERT_CODEC)
MODEL_HEADER := ../../model-families/laguna/include/sparkpipe/spark_laguna_model.h
MODULE_ENTRY_PREFIX := SparkLagunaResidentDecodeStage
MODULE_HOST_SOURCE := source/spark_laguna_resident_decode_stage_module.c
MODULE_ADDITIONAL_HOST_SOURCES := \
	../../runtime/stage_module_lifecycle.c \
	$(addprefix ../../,$(SPARKPIPE_WEIGHTD_SOURCES)) \
	../../src/spark_status.c \
	../../src/spark_sha256.c \
	../../src/spark_ck128.c \
	../../ring/transport/hidden_transport.c \
	../../ring/transport/tp_device_collective.c \
	../../cache/kv_cache.c \
	../../cache/kv_page_cache.c \
	../../cache/kv_page_store.c \
	../../cache/kv_snapshot.c \
	../../cache/kv_model_table.c
MODULE_CUDA_SOURCE := source/spark_laguna_resident_decode_stage_cuda.cu
MODULE_COMPILE_FLAGS := -DSPARK_BATCH_BUCKET=1024u -DLAGUNA_EXPERT_WEIGHT_CODEC=$(LAGUNA_EXPERT_CODEC_ID) -DLAGUNA_EXPERT_CODEC_NAME=\"$(EXPERT_CODEC)\" -DLAGUNA_MODEL_REVISION=\"$(MODEL_REVISION)\" -DLAGUNA_CONTRACT_SHA256=\"$(CONTRACT_SHA256)\"

BUILD_DIRECTORY ?= ../../build/modules/laguna_resident_decode_stage/$(EXPERT_CODEC)
MODULE_ARCHIVE ?= $(BUILD_DIRECTORY)/liblaguna_resident_decode_stage_$(EXPERT_CODEC).a
MODULE_COMMON_HOST_SOURCES := ../../runtime/stage_module_common.c
MODULE_INCLUDE_FLAGS := \
	-I../../include \
	-I../../model-families/common/include \
	-I../../model-families/laguna/include \
	-I../../modules/laguna_resident_decode_stage/include \
	-I../../modules/laguna_resident_decode_stage/source \
	-I../..

STAGE_INDEX ?= 0
MAX_ACTIVE_SEQUENCES ?= 64
MAX_SEQUENCE_POSITIONS ?= 32768
EXECUTION_ROW_CAPACITY ?= 64
PIPELINE_SLOT_COUNT ?= 2

RUNTIME_CONFIGURATION := \
	SPARK_LAGUNA_EXPERT_CODEC=$(EXPERT_CODEC) \
	SPARK_LAGUNA_STAGE_INDEX=$(STAGE_INDEX) \
	SPARK_LAGUNA_STAGE_MAX_ACTIVE_SEQUENCES=$(MAX_ACTIVE_SEQUENCES) \
	SPARK_LAGUNA_STAGE_MAX_SEQUENCE_POSITIONS=$(MAX_SEQUENCE_POSITIONS) \
	SPARK_LAGUNA_STAGE_EXECUTION_ROW_CAPACITY=$(EXECUTION_ROW_CAPACITY) \
	SPARK_LAGUNA_STAGE_PIPELINE_SLOTS=$(PIPELINE_SLOT_COUNT)

GPU_VALIDATOR := validation/validate_laguna_resident_decode_stage_cuda.sh

include ../resident_decode_stage_rules.mk

LAGUNA_MODEL_DESCRIPTION := ../../examples/model_descriptions/laguna_resident_decode_stage_$(EXPERT_CODEC)_firmware.json
LAGUNA_MODEL_DESCRIPTION_SHA256 = $(shell if command -v sha256sum >/dev/null 2>&1; then sha256sum "$(LAGUNA_MODEL_DESCRIPTION)"; else shasum -a 256 "$(LAGUNA_MODEL_DESCRIPTION)"; fi | awk '{print $$1}')

ADAPTER_SOURCE := source/spark_laguna_serving_adapter.c
ADAPTER_FLAGS = $(ADAPTER_VARIANT_FLAGS) -DLAGUNA_MODEL_DESCRIPTION_SHA256=\"$(LAGUNA_MODEL_DESCRIPTION_SHA256)\"

.PHONY: laguna_require_model_description
adapter: laguna_require_model_description
laguna_require_model_description:
	@test -r "$(LAGUNA_MODEL_DESCRIPTION)" || { echo "laguna: EXPERT_CODEC=$(EXPERT_CODEC) has no firmware description: $(LAGUNA_MODEL_DESCRIPTION)" >&2; exit 1; }
