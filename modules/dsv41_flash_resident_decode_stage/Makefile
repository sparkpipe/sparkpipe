include ../../runtime/weightd_sources.mk

ifndef EXPERT_CODEC
$(error EXPERT_CODEC is required: fp8 or mxfp4)
endif
ifndef MODEL_REVISION
$(error MODEL_REVISION is required and must be the exact source snapshot revision)
endif
ifndef CONTRACT_SHA256
$(error CONTRACT_SHA256 is required and must identify the exact model package contract)
endif

DSV41_FLASH_EXPERT_CODECS := fp8 mxfp4
ifeq ($(filter $(EXPERT_CODEC),$(DSV41_FLASH_EXPERT_CODECS)),)
$(error unsupported EXPERT_CODEC '$(EXPERT_CODEC)': choose $(DSV41_FLASH_EXPERT_CODECS))
endif

ifeq ($(EXPERT_CODEC),fp8)
DSV41_FLASH_EXPERT_CODEC_ID := 5
endif
ifeq ($(EXPERT_CODEC),mxfp4)
DSV41_FLASH_EXPERT_CODEC_ID := 7
endif

MODULE_FAMILY := dsv41_flash
MODULE_IDENTIFIER_PREFIX := spark.dsv41_flash.resident_decode_stage.bf16.expert_$(EXPERT_CODEC).h5120.l40.e384
MODULE_IDENTIFIER_SUFFIX := v1
MODULE_IDENTIFIER := $(MODULE_IDENTIFIER_PREFIX).$(MODULE_IDENTIFIER_SUFFIX)
MODULE_BATCH_VARIANT_BUCKETS ?= 1 2 4 8 16 32 64 128 256 512 1024
MODULE_TARGET := cuda.sm121.dsv41_flash.resident_decode_stage.bf16.expert_$(EXPERT_CODEC)
MODEL_HEADER := ../../model-families/dsv41_flash/include/sparkpipe/spark_dsv41_flash_model.h
MODULE_ENTRY_PREFIX := SparkDsv41FlashResidentDecodeStage
MODULE_HOST_SOURCE := source/spark_dsv41_flash_resident_decode_stage_module.c
MODULE_ADDITIONAL_HOST_SOURCES := \
	../../runtime/stage_module_lifecycle.c \
	$(addprefix ../../,$(SPARKPIPE_WEIGHTD_SOURCES)) \
	../../src/spark_status.c \
	../../src/spark_sha256.c \
	../../src/spark_ck128.c
MODULE_CUDA_SOURCE := source/spark_dsv41_flash_resident_decode_stage_cuda.cu
MODULE_COMPILE_FLAGS := -DDSV41_FLASH_EXPERT_WEIGHT_CODEC=$(DSV41_FLASH_EXPERT_CODEC_ID) -DDSV41_FLASH_EXPERT_CODEC_NAME=\"$(EXPERT_CODEC)\" -DSPARK_DSV41_FLASH_MODEL_REVISION=\"$(MODEL_REVISION)\" -DSPARK_DSV41_FLASH_CONTRACT_SHA256=\"$(CONTRACT_SHA256)\"

BUILD_DIRECTORY ?= ../../build/modules/dsv41_flash_resident_decode_stage/$(EXPERT_CODEC)
MODULE_ARCHIVE ?= $(BUILD_DIRECTORY)/libdsv41_flash_resident_decode_stage_$(EXPERT_CODEC).a
MODULE_COMMON_HOST_SOURCES := ../../runtime/stage_module_common.c
MODULE_INCLUDE_FLAGS := \
	-I../../include \
	-I../../model-families/common/include \
	-I../../model-families/dsv41_flash/include \
	-I../../modules/dsv41_flash_resident_decode_stage/include \
	-I../../modules/dsv41_flash_resident_decode_stage/source \
	-I../..

STAGE_INDEX ?= 0
STAGE_COUNT ?= 1
FIRST_LAYER_INDEX ?= 0
LAYER_COUNT ?= 40
MAX_ACTIVE_SEQUENCES ?= 64
MAX_SEQUENCE_POSITIONS ?= 1048576
EXECUTION_ROW_CAPACITY ?= 64
PIPELINE_SLOT_COUNT ?= 2
TP_DEGREE ?= 1
TP_RANK ?= 0
STAGE_PACK_PATH ?=

RUNTIME_CONFIGURATION := \
	SPARK_DSV41_FLASH_EXPERT_CODEC=$(EXPERT_CODEC) \
	SPARK_DSV41_FLASH_STAGE_INDEX=$(STAGE_INDEX) \
	SPARK_DSV41_FLASH_STAGE_MAX_ACTIVE_SEQUENCES=$(MAX_ACTIVE_SEQUENCES) \
	SPARK_DSV41_FLASH_STAGE_MAX_SEQUENCE_POSITIONS=$(MAX_SEQUENCE_POSITIONS) \
	SPARK_DSV41_FLASH_STAGE_EXECUTION_ROW_CAPACITY=$(EXECUTION_ROW_CAPACITY) \
	SPARK_DSV41_FLASH_STAGE_PIPELINE_SLOTS=$(PIPELINE_SLOT_COUNT)

include ../resident_decode_stage_rules.mk

DSV41_TEST_PACK ?= $(BUILD_DIRECTORY)/test_host_pack.bin
DSV41_TEST_MANIFEST ?= $(BUILD_DIRECTORY)/test_host_pack.experts
DSV41_TEST_LAYERS ?= 2

.PHONY: test_host
test_host: contract
	$(CC) $(CFLAGS) $(MODULE_POSIX_FLAGS) $(MODULE_INCLUDE_FLAGS) \
		tools/dsv41_flash_pack_synthesize.c -o $(BUILD_DIRECTORY)/dsv41_flash_pack_synthesize
	$(CC) $(CFLAGS) $(MODULE_POSIX_FLAGS) $(MODULE_INCLUDE_FLAGS) \
		tools/dsv41_flash_experts_manifest.c $(REPOSITORY_ROOT)/src/spark_ck128.c -o $(BUILD_DIRECTORY)/dsv41_flash_experts_manifest
	$(CC) $(CFLAGS) $(MODULE_POSIX_FLAGS) $(MODULE_INCLUDE_FLAGS) \
		$(REPOSITORY_ROOT)/tests/test_dsv41_flash_pack_contract.c $(REPOSITORY_ROOT)/src/spark_ck128.c -o $(BUILD_DIRECTORY)/dsv41_flash_pack_contract_test
	$(BUILD_DIRECTORY)/dsv41_flash_pack_synthesize $(DSV41_TEST_PACK) $(MODEL_REVISION) $(CONTRACT_SHA256) 0000000000000000000000000000000000000000000000000000000000000000 0000000000000000000000000000000000000000000000000000000000000000 $(TP_DEGREE) $(TP_RANK) $(EXPERT_CODEC) $(DSV41_TEST_LAYERS)
	$(BUILD_DIRECTORY)/dsv41_flash_experts_manifest $(DSV41_TEST_PACK) $(DSV41_TEST_MANIFEST)
	$(BUILD_DIRECTORY)/dsv41_flash_pack_contract_test $(DSV41_TEST_PACK) $(DSV41_TEST_MANIFEST)
