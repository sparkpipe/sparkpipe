MODULE_FAMILY := dsv4
# Expert weight codec selection: mxfp4 (default) | fp8_e4m3 (variant builds;
# requires the FP8 expert kernel variant and an FP8-expert pack).
PRO_EXPERT_CODEC ?= mxfp4
ifeq ($(PRO_EXPERT_CODEC),fp8_e4m3)
PRO_EXPERT_CODEC_FRAGMENT := expert_fp8
PRO_EXPERT_CODEC_FLAG := -DSPARK_DSV4_PRO_EXPERT_CODEC_FP8_E4M3=1
else
PRO_EXPERT_CODEC_FRAGMENT := expert_mxfp4
PRO_EXPERT_CODEC_FLAG :=
endif
# KV cache codec selection: bf16 (default, the validated set) |
# fp8_e4m3 (variant builds; requires the FP8-KV cache kernels).
PRO_KV_CODEC ?= bf16
ifeq ($(PRO_KV_CODEC),fp8_e4m3)
PRO_KV_CODEC_FRAGMENT := kv_fp8
PRO_KV_CODEC_FLAG := -DSPARK_DSV4_PRO_KV_CODEC_FP8_E4M3=1
else
PRO_KV_CODEC_FRAGMENT := kv_bf16
PRO_KV_CODEC_FLAG :=
endif
MODULE_IDENTIFIER_PREFIX := spark.dsv4.pro.resident_decode_stage.linear_fp8.$(PRO_EXPERT_CODEC_FRAGMENT).$(PRO_KV_CODEC_FRAGMENT).h7168.l61.e384.k6.ga0731
MODULE_IDENTIFIER_SUFFIX := v1
MODULE_IDENTIFIER := $(MODULE_IDENTIFIER_PREFIX).$(MODULE_IDENTIFIER_SUFFIX)
# The batch-variant ladder: one archive per power of two, emitted by
# resident_decode_stage_rules.mk from the single template there. Trim for
# local iteration here, never by editing a variant.
MODULE_BATCH_VARIANT_BUCKETS ?= 1 2 4 8 16 32 64 128 256 512 1024
MODULE_TARGET := cuda.sm121.dsv4.pro.resident_decode_stage.linear_fp8.$(PRO_EXPERT_CODEC_FRAGMENT).$(PRO_KV_CODEC_FRAGMENT)
MODEL_HEADER := ../../model-families/dsv4/include/sparkpipe/spark_dsv4_model.h
MODULE_ENTRY_PREFIX := SparkDsv4ResidentDecodeStage
MODULE_HOST_SOURCE := source/spark_dsv4_resident_decode_stage_module.c
MODULE_ADDITIONAL_HOST_SOURCES := \
	source/spark_dsv4_stage_runner.c \
	source/spark_dsv4_paged_cache.c \
	../../model-families/dsv4/src/spark_dsv4_parallel_shape.c \
	../../ring/transport/hidden_transport.c \
	../../ring/transport/tp_collective.c \
	../../ring/transport/tp_device_collective.c \
	../../ring/transport/tp_device_collective_nccl.c \
	../../cache/kv_cache.c \
	../../cache/kv_page_cache.c \
	../../cache/kv_page_store.c
MODULE_CUDA_SOURCE := source/spark_dsv4_resident_decode_stage_cuda.cu
MODULE_COMPILE_FLAGS := -DSPARK_DSV4_MODULE_BUILD=1 -DSPARK_BATCH_BUCKET=1024u -DSPARK_DSV4_PRO_BUILD=1 $(PRO_EXPERT_CODEC_FLAG) $(PRO_KV_CODEC_FLAG)

# Keep the module graph relative to this directory. The checkout path contains
# spaces on the development Mac, so absolute source lists are not shell-safe.
BUILD_DIRECTORY ?= ../../build/modules/dsv4_pro_resident_decode_stage
MODULE_ARCHIVE ?= $(BUILD_DIRECTORY)/libdsv4_resident_decode_stage.a
MODULE_COMMON_HOST_SOURCES := ../../runtime/stage_module_common.c
MODULE_INCLUDE_FLAGS := \
	-I../../include \
	-I../../model-families/common/include \
	-I../../model-families/dsv4/include \
	-I../../modules/dsv4_resident_decode_stage/include \
	-I../../modules/dsv4_resident_decode_stage/source \
	-I../..

STAGE_PACK_PATH ?= $(REPOSITORY_ROOT)/build/stagepacks/dsv4_stage.dsv4sp
STAGE_COUNT ?= 1
STAGE_INDEX ?= 0
STAGE_FIRST_LAYER ?= 0
STAGE_LAYER_COUNT ?= 61
MAX_ACTIVE_SEQUENCES ?= 32
MAX_SEQUENCE_POSITIONS ?= 4736
PIPELINE_SLOT_COUNT ?= 13
PHYSICAL_PAGE_CAPACITY ?= 1024
LOGICAL_PAGE_CAPACITY ?= 16384
MTP_LAYER_COUNT ?= 1
CUDA_GRAPH_COUNT ?= 0
override DSV4_GA_STAGE0_REFERENCE_MANIFEST_SHA256 := 9ef837975bc4ddbd3cf0de0ea19c59c2c4c8a3750a8b8f302a19df0e09f39fa3
override DSV4_CUDA_VALIDATOR_SHA256 := $(shell sha256sum validation/spark_dsv4_resident_decode_stage_cuda_validation.cu | awk '{print $$1}')
override DSV4_REFERENCE_VERIFIER_SHA256 := $(shell sha256sum ../../tools/verify_dsv4_ga_reference_fixture.py | awk '{print $$1}')

RUNTIME_CONFIGURATION := \
	SPARK_DSV4_STAGE_PACK_PATH=$(STAGE_PACK_PATH) \
	SPARK_DSV4_STAGE_COUNT=$(STAGE_COUNT) \
	SPARK_DSV4_STAGE_INDEX=$(STAGE_INDEX) \
	SPARK_DSV4_STAGE_FIRST_LAYER=$(STAGE_FIRST_LAYER) \
	SPARK_DSV4_STAGE_LAYER_COUNT=$(STAGE_LAYER_COUNT) \
	SPARK_DSV4_STAGE_MAX_ACTIVE_SEQUENCES=$(MAX_ACTIVE_SEQUENCES) \
	SPARK_DSV4_STAGE_MAX_SEQ=$(MAX_SEQUENCE_POSITIONS) \
	SPARK_DSV4_STAGE_PIPELINE_SLOTS=$(PIPELINE_SLOT_COUNT) \
	SPARK_DSV4_STAGE_PHYSICAL_PAGES=$(PHYSICAL_PAGE_CAPACITY) \
	SPARK_DSV4_STAGE_LOGICAL_PAGES=$(LOGICAL_PAGE_CAPACITY) \
	SPARK_DSV4_STAGE_MTP=$(MTP_LAYER_COUNT) \
	SPARK_DSV4_STAGE_GRAPHS=$(CUDA_GRAPH_COUNT) \
	SPARK_DSV4_REFERENCE_MANIFEST_SHA256=$(DSV4_GA_STAGE0_REFERENCE_MANIFEST_SHA256) \
	SPARK_DSV4_CUDA_VALIDATOR_SHA256=$(DSV4_CUDA_VALIDATOR_SHA256) \
	SPARK_DSV4_REFERENCE_VERIFIER_SHA256=$(DSV4_REFERENCE_VERIFIER_SHA256)

GPU_VALIDATOR := validation/validate_dsv4_resident_decode_stage_cuda.sh

include ../resident_decode_stage_rules.mk
