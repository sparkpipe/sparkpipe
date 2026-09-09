MODULE_FAMILY := minimax_h3
MODULE_IDENTIFIER := spark.minimax_h3.resident_media_stage.bf16.h5376.l50.t4pp4.v1
MODULE_TARGET := cuda.sm121.minimax_h3.resident_media_stage.bf16
MODEL_HEADER := ../../model-families/minimax_h3/include/sparkpipe/spark_minimax_h3_model.h
MODULE_ENTRY_PREFIX := SparkMinimaxH3ResidentMediaStage
MODULE_HOST_SOURCE := $(CURDIR)/source/spark_minimax_h3_resident_media_stage_module.c
MODULE_CUDA_SOURCE := $(CURDIR)/source/spark_minimax_h3_resident_media_stage_cuda.cu
MODULE_ADDITIONAL_HOST_SOURCES = \
	$(REPOSITORY_ROOT)/runtime/stage_module_lifecycle.c \
	$(REPOSITORY_ROOT)/model-families/minimax_h3/src/spark_minimax_h3_scheduler.c \
	$(REPOSITORY_ROOT)/ring/transport/tp_collective.c \
	$(REPOSITORY_ROOT)/ring/transport/tp_device_collective.c \
	$(REPOSITORY_ROOT)/ring/transport/tp_device_collective_nccl.c
MINIMAX_H3_MODEL_REVISION := 42ed227ee7df40d41602854ae760620d6eb651fe
MINIMAX_H3_DIFFUSERS_COMMIT := 3c22124636a332a0df02910f609315d4c2898f57
MINIMAX_H3_CONTRACT_SHA256 := $(shell shasum -a 256 $(CURDIR)/../../model_contracts/h3_authoritative.json | awk '{print $$1}')
MODULE_COMPILE_FLAGS := \
	-DSPARK_MINIMAX_H3_MODULE_BUILD=1 \
	-DMINIMAX_H3_MODEL_REVISION=\"$(MINIMAX_H3_MODEL_REVISION)\" \
	-DMINIMAX_H3_DIFFUSERS_COMMIT=\"$(MINIMAX_H3_DIFFUSERS_COMMIT)\" \
	-DMINIMAX_H3_CONTRACT_SHA256=\"$(MINIMAX_H3_CONTRACT_SHA256)\"
MODULE_INCLUDE_FLAGS = \
	-I"$(REPOSITORY_ROOT)/include" \
	-I"$(REPOSITORY_ROOT)/src" \
	-I"$(REPOSITORY_ROOT)/model-families/common/include" \
	-I"$(REPOSITORY_ROOT)/model-families/minimax_h3/include" \
	-I"$(REPOSITORY_ROOT)/modules/minimax_h3_resident_media_stage/include" \
	-I"$(REPOSITORY_ROOT)/modules/minimax_h3_resident_media_stage/source" \
	-I"$(REPOSITORY_ROOT)"

STAGE_PACK_PATH ?= $(REPOSITORY_ROOT)/build/stagepacks/minimax_h3_stage.rank$(TP_RANK).sp
STAGE_COUNT ?= 4
STAGE_INDEX ?= 0
MAX_ACTIVE_SEQUENCES ?= 4
PIPELINE_SLOT_COUNT ?= 2
KV_BLOCK_COUNT ?= 2
KV_PROVIDER ?= none
KV_SERVICE_ADDRESS ?= none
KV_SOCKET_PATH ?= none
KV_POOL_BYTES ?= 0
KV_WORKER_COUNT ?= 0
ALLOW_UNQUALIFIED_EXECUTION ?= 0
TP_DEGREE ?= 4
TP_RANK ?= 0
TP_STANDALONE ?= 0
override MINIMAX_H3_CUDA_VALIDATOR_SHA256 := $(shell shasum -a 256 validation/spark_minimax_h3_resident_media_stage_cuda_validation.cu 2>/dev/null | awk '{print $$1}')
override MINIMAX_H3_CPU_ORACLE_SHA256 := $(shell shasum -a 256 validation/spark_minimax_h3_reference.c 2>/dev/null | awk '{print $$1}')

RUNTIME_CONFIGURATION := \
	SPARK_MINIMAX_H3_ALLOW_UNQUALIFIED_EXECUTION=$(ALLOW_UNQUALIFIED_EXECUTION) \
	SPARK_MINIMAX_H3_TP_DEGREE=$(TP_DEGREE) \
	SPARK_MINIMAX_H3_TP_RANK=$(TP_RANK) \
	SPARK_MINIMAX_H3_TP_STANDALONE=$(TP_STANDALONE) \
	SPARK_MINIMAX_H3_STAGE_PACK_PATH=$(STAGE_PACK_PATH) \
	SPARK_MINIMAX_H3_STAGE_COUNT=$(STAGE_COUNT) \
	SPARK_MINIMAX_H3_STAGE_INDEX=$(STAGE_INDEX) \
	SPARK_MINIMAX_H3_STAGE_MAX_ACTIVE_SEQUENCES=$(MAX_ACTIVE_SEQUENCES) \
	SPARK_MINIMAX_H3_STAGE_PIPELINE_SLOTS=$(PIPELINE_SLOT_COUNT) \
	SPARK_MINIMAX_H3_STAGE_KV_BLOCKS=$(KV_BLOCK_COUNT) \
	SPARK_MINIMAX_H3_STAGE_KV_STORE=$(KV_PROVIDER) \
	SPARK_MINIMAX_H3_STAGE_KV_SERVICE=$(KV_SERVICE_ADDRESS) \
	SPARK_MINIMAX_H3_STAGE_KV_SOCKET=$(KV_SOCKET_PATH) \
	SPARK_MINIMAX_H3_STAGE_KV_POOL_BYTES=$(KV_POOL_BYTES) \
	SPARK_MINIMAX_H3_STAGE_KV_WORKERS=$(KV_WORKER_COUNT) \
	MINIMAX_H3_CUDA_VALIDATOR_SHA256=$(MINIMAX_H3_CUDA_VALIDATOR_SHA256) \
	MINIMAX_H3_CPU_ORACLE_SHA256=$(MINIMAX_H3_CPU_ORACLE_SHA256)

GPU_VALIDATOR := validation/validate_minimax_h3_resident_media_stage_cuda.sh

include ../resident_decode_stage_rules.mk
