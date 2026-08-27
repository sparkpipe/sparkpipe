MODULE_FAMILY := qwen4_flash
MODULE_IDENTIFIER := spark.qwen4_flash.resident_decode_stage.fp8.h2560.l48.gdn36.e512k10.v1
MODULE_TARGET := cuda.sm121.qwen4_flash.resident_decode_stage.fp8
MODEL_HEADER := ../../model-families/qwen4_flash/include/sparkpipe/spark_qwen4_flash_model.h
MODULE_ENTRY_PREFIX := SparkQwen4FlashResidentDecodeStage
MODULE_HOST_SOURCE := $(CURDIR)/source/spark_qwen4_flash_resident_decode_stage_module.c
MODULE_CUDA_SOURCE := $(CURDIR)/source/spark_qwen4_flash_resident_decode_stage_cuda.cu
MODULE_ADDITIONAL_HOST_SOURCES = \
	$(REPOSITORY_ROOT)/cache/store/stage_kv_client.c \
	$(REPOSITORY_ROOT)/cache/store/kv_store.c \
	$(REPOSITORY_ROOT)/model-families/qwen4_flash/src/spark_qwen4_flash_work_control.c \
	$(REPOSITORY_ROOT)/ring/transport/hidden_transport.c \
	$(REPOSITORY_ROOT)/ring/transport/tp_collective.c \
	$(REPOSITORY_ROOT)/ring/transport/tp_device_collective.c \
	$(REPOSITORY_ROOT)/ring/transport/tp_device_collective_nccl.c
# Adapter provenance pins: the Qwen3.8-Flash-Next checkpoint revision and the
# exact package contract, checked at adapter compile time.
QWEN4_FLASH_MODEL_REVISION := f5d08274bafd880402bd16f5e3e6c514136ec06c
QWEN4_FLASH_CONTRACT_SHA256 := $(shell sha256sum $(REPOSITORY_ROOT)/model_contracts/qwen4_flash_authoritative.json | awk '{print $$1}')
MODULE_COMPILE_FLAGS := \
	-DSPARK_QWEN4_FLASH_MODULE_BUILD=1 \
	-DQWEN4_FLASH_MODEL_REVISION=\"$(QWEN4_FLASH_MODEL_REVISION)\" \
	-DQWEN4_FLASH_CONTRACT_SHA256=\"$(QWEN4_FLASH_CONTRACT_SHA256)\"
MODULE_INCLUDE_FLAGS = \
	-I"$(REPOSITORY_ROOT)/include" \
	-I"$(REPOSITORY_ROOT)/src" \
	-I"$(REPOSITORY_ROOT)/model-families/common/include" \
	-I"$(REPOSITORY_ROOT)/model-families/qwen4_flash/include" \
	-I"$(REPOSITORY_ROOT)/modules/qwen4_flash_resident_decode_stage/include" \
	-I"$(REPOSITORY_ROOT)/modules/qwen4_flash_resident_decode_stage/source" \
	-I"$(REPOSITORY_ROOT)"

STAGE_PACK_PATH ?= $(REPOSITORY_ROOT)/build/stagepacks/qwen4_flash_stage.qwen4_flashsp
STAGE_COUNT ?= 12
STAGE_INDEX ?= 0
STAGE_FIRST_LAYER ?= 0
STAGE_LAYER_COUNT ?= 4
MAX_ACTIVE_SEQUENCES ?= 32
PIPELINE_SLOT_COUNT ?= 2
KV_BLOCK_COUNT ?= 4096
MTP_LAYER_COUNT ?= 0
GDN_SNAPSHOT_SLOT_COUNT ?= 0
KV_PROVIDER ?= none
KV_SERVICE_ADDRESS ?= none
KV_SOCKET_PATH ?= none
KV_POOL_BYTES ?= 0
KV_WORKER_COUNT ?= 0
ALLOW_UNQUALIFIED_EXECUTION ?= 0
TP_DEGREE ?= 1
TP_RANK ?= 0
TP_STANDALONE ?= 0
override QWEN4_FLASH_CUDA_VALIDATOR_SHA256 := $(shell sha256sum validation/spark_qwen4_flash_resident_decode_stage_cuda_validation.cu 2>/dev/null | awk '{print $$1}')
override QWEN4_FLASH_CPU_ORACLE_SHA256 := $(shell sha256sum validation/spark_qwen4_flash_reference.c 2>/dev/null | awk '{print $$1}')

RUNTIME_CONFIGURATION := \
	SPARK_QWEN4_FLASH_ALLOW_UNQUALIFIED_EXECUTION=$(ALLOW_UNQUALIFIED_EXECUTION) \
	SPARK_QWEN4_FLASH_TP_DEGREE=$(TP_DEGREE) \
	SPARK_QWEN4_FLASH_TP_RANK=$(TP_RANK) \
	SPARK_QWEN4_FLASH_TP_STANDALONE=$(TP_STANDALONE) \
	SPARK_QWEN4_FLASH_STAGE_PACK_PATH=$(STAGE_PACK_PATH) \
	SPARK_QWEN4_FLASH_STAGE_COUNT=$(STAGE_COUNT) \
	SPARK_QWEN4_FLASH_STAGE_INDEX=$(STAGE_INDEX) \
	SPARK_QWEN4_FLASH_STAGE_FIRST_LAYER=$(STAGE_FIRST_LAYER) \
	SPARK_QWEN4_FLASH_STAGE_LAYER_COUNT=$(STAGE_LAYER_COUNT) \
	SPARK_QWEN4_FLASH_STAGE_MAX_ACTIVE_SEQUENCES=$(MAX_ACTIVE_SEQUENCES) \
	SPARK_QWEN4_FLASH_STAGE_PIPELINE_SLOTS=$(PIPELINE_SLOT_COUNT) \
	SPARK_QWEN4_FLASH_STAGE_KV_BLOCKS=$(KV_BLOCK_COUNT) \
	SPARK_QWEN4_FLASH_STAGE_KV_STORE=$(KV_PROVIDER) \
	SPARK_QWEN4_FLASH_STAGE_KV_SERVICE=$(KV_SERVICE_ADDRESS) \
	SPARK_QWEN4_FLASH_STAGE_KV_SOCKET=$(KV_SOCKET_PATH) \
	SPARK_QWEN4_FLASH_STAGE_KV_POOL_BYTES=$(KV_POOL_BYTES) \
	SPARK_QWEN4_FLASH_STAGE_KV_WORKERS=$(KV_WORKER_COUNT) \
	SPARK_QWEN4_FLASH_STAGE_MTP=$(MTP_LAYER_COUNT) \
	SPARK_QWEN4_FLASH_STAGE_GDN_SNAPSHOT_SLOTS=$(GDN_SNAPSHOT_SLOT_COUNT) \
	SPARK_QWEN4_FLASH_CUDA_VALIDATOR_SHA256=$(QWEN4_FLASH_CUDA_VALIDATOR_SHA256) \
	SPARK_QWEN4_FLASH_CPU_ORACLE_SHA256=$(QWEN4_FLASH_CPU_ORACLE_SHA256)

GPU_VALIDATOR := validation/validate_qwen4_flash_resident_decode_stage_cuda.sh

include ../resident_decode_stage_rules.mk
