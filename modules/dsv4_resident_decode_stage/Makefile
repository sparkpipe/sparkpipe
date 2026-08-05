MODULE_FAMILY := dsv4
MODULE_IDENTIFIER := spark.dsv4.flash.resident_decode_stage.linear_fp8.expert_mxfp4.kv_bf16.h4096.l43.e256.k6.v2
MODULE_TARGET := cuda.sm121.dsv4.flash.resident_decode_stage.linear_fp8.expert_mxfp4.kv_bf16
MODEL_HEADER := ../../model-families/dsv4/include/sparkpipe/spark_dsv4_model.h
MODULE_ENTRY_PREFIX := SparkDsv4ResidentDecodeStage
MODULE_HOST_SOURCE := source/spark_dsv4_resident_decode_stage_module.c
MODULE_ADDITIONAL_HOST_SOURCES := source/spark_dsv4_stage_runner.c
MODULE_CUDA_SOURCE := source/spark_dsv4_resident_decode_stage_cuda.cu
MODULE_COMPILE_FLAGS := -DSPARK_DSV4_MODULE_BUILD=1

# Keep the module graph relative to this directory. The checkout path contains
# spaces on the development Mac, so absolute source lists are not shell-safe.
BUILD_DIRECTORY ?= ../../build/modules/dsv4_resident_decode_stage
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
STAGE_LAYER_COUNT ?= 43
MAX_ACTIVE_SEQUENCES ?= 32
MAX_SEQUENCE_POSITIONS ?= 4096
PIPELINE_SLOT_COUNT ?= 2
MTP_LAYER_COUNT ?= 0
CUDA_GRAPH_COUNT ?= 0

RUNTIME_CONFIGURATION := \
	SPARK_DSV4_STAGE_PACK_PATH=$(STAGE_PACK_PATH) \
	SPARK_DSV4_STAGE_COUNT=$(STAGE_COUNT) \
	SPARK_DSV4_STAGE_INDEX=$(STAGE_INDEX) \
	SPARK_DSV4_STAGE_FIRST_LAYER=$(STAGE_FIRST_LAYER) \
	SPARK_DSV4_STAGE_LAYER_COUNT=$(STAGE_LAYER_COUNT) \
	SPARK_DSV4_STAGE_MAX_ACTIVE_SEQUENCES=$(MAX_ACTIVE_SEQUENCES) \
	SPARK_DSV4_STAGE_MAX_SEQ=$(MAX_SEQUENCE_POSITIONS) \
	SPARK_DSV4_STAGE_PIPELINE_SLOTS=$(PIPELINE_SLOT_COUNT) \
	SPARK_DSV4_STAGE_MTP=$(MTP_LAYER_COUNT) \
	SPARK_DSV4_STAGE_GRAPHS=$(CUDA_GRAPH_COUNT)

GPU_VALIDATOR := validation/validate_dsv4_resident_decode_stage_cuda.sh

include ../resident_decode_stage_rules.mk
