MODULE_FAMILY := minimax
MODULE_IDENTIFIER := spark.minimax.resident_decode_stage.bf16.h5120.l64.tp4.v1
MODULE_TARGET := cuda.sm121.minimax.resident_decode_stage.bf16
MODEL_HEADER := ../../modules/minimax_resident_decode_stage/include/sparkpipe/spark_minimax_model.h
MODULE_ENTRY_PREFIX := SparkMinimaxResidentDecodeStage
MODULE_HOST_SOURCE := $(CURDIR)/source/spark_minimax_resident_decode_stage_module.c
MODULE_CUDA_SOURCE := $(CURDIR)/source/spark_minimax_resident_decode_stage_cuda.cu
MODULE_ADDITIONAL_HOST_SOURCES = \
	$(REPOSITORY_ROOT)/runtime/stage_module_lifecycle.c \
	$(REPOSITORY_ROOT)/runtime/stagepack_format.c \
	$(REPOSITORY_ROOT)/ring/transport/hidden_transport.c \
	$(REPOSITORY_ROOT)/ring/transport/tp_collective.c \
	$(REPOSITORY_ROOT)/ring/transport/tp_device_collective.c
MINIMAX_MODEL_REVISION := $(shell git -C $(REPOSITORY_ROOT) rev-parse HEAD 2>/dev/null || echo unknown)
MINIMAX_CONTRACT_SHA256 := $(shell sha256sum $(CURDIR)/../../model-families/minimax/include/sparkpipe/llm_defines.h 2>/dev/null | awk '{print $$1}')
MODULE_COMPILE_FLAGS := \
	-DSPARK_MINIMAX_MODULE_BUILD=1 \
	-DMINIMAX_MODEL_REVISION=\"$(MINIMAX_MODEL_REVISION)\" \
	-DMINIMAX_CONTRACT_SHA256=\"$(MINIMAX_CONTRACT_SHA256)\"
MODULE_INCLUDE_FLAGS = \
	-I"$(REPOSITORY_ROOT)/include" \
	-I"$(REPOSITORY_ROOT)/src" \
	-I"$(REPOSITORY_ROOT)/model-families/common/include" \
	-I"$(REPOSITORY_ROOT)/model-families/minimax/include" \
	-I"$(REPOSITORY_ROOT)/modules/minimax_resident_decode_stage/include" \
	-I"$(REPOSITORY_ROOT)/modules/minimax_resident_decode_stage/source" \
	-I"$(REPOSITORY_ROOT)"

STAGE_PACK_PATH ?= $(REPOSITORY_ROOT)/build/stagepacks/minimax.text.bf16.tp4.rank0.mntx
STAGE_COUNT ?= 1
STAGE_INDEX ?= 0
MAX_ACTIVE_SEQUENCES ?= 32
PIPELINE_SLOT_COUNT ?= 2
KV_BLOCK_COUNT ?= 256
KV_PROVIDER ?= none
KV_SERVICE_ADDRESS ?= none
KV_SOCKET_PATH ?= none
KV_POOL_BYTES ?= 0
KV_WORKER_COUNT ?= 0
ALLOW_UNQUALIFIED_EXECUTION ?= 0
TP_DEGREE ?= 4
TP_RANK ?= 0
TP_STANDALONE ?= 0

RUNTIME_CONFIGURATION := \
	SPARK_MINIMAX_ALLOW_UNQUALIFIED_EXECUTION=$(ALLOW_UNQUALIFIED_EXECUTION) \
	SPARK_MINIMAX_TP_DEGREE=$(TP_DEGREE) \
	SPARK_MINIMAX_TP_RANK=$(TP_RANK) \
	SPARK_MINIMAX_TP_STANDALONE=$(TP_STANDALONE) \
	SPARK_MINIMAX_STAGE_PACK_PATH=$(STAGE_PACK_PATH) \
	SPARK_MINIMAX_STAGE_COUNT=$(STAGE_COUNT) \
	SPARK_MINIMAX_STAGE_INDEX=$(STAGE_INDEX) \
	SPARK_MINIMAX_STAGE_MAX_ACTIVE_SEQUENCES=$(MAX_ACTIVE_SEQUENCES) \
	SPARK_MINIMAX_STAGE_PIPELINE_SLOTS=$(PIPELINE_SLOT_COUNT) \
	SPARK_MINIMAX_STAGE_KV_BLOCKS=$(KV_BLOCK_COUNT) \
	SPARK_MINIMAX_STAGE_KV_STORE=$(KV_PROVIDER) \
	SPARK_MINIMAX_STAGE_KV_SERVICE=$(KV_SERVICE_ADDRESS) \
	SPARK_MINIMAX_STAGE_KV_SOCKET=$(KV_SOCKET_PATH) \
	SPARK_MINIMAX_STAGE_KV_POOL_BYTES=$(KV_POOL_BYTES) \
	SPARK_MINIMAX_STAGE_KV_WORKERS=$(KV_WORKER_COUNT)

GPU_VALIDATOR := validation/validate_minimax_resident_decode_stage_cuda.sh

include ../resident_decode_stage_rules.mk
