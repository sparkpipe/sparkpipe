MODULE_FAMILY := qwen38
MODULE_IDENTIFIER := spark.qwen38.resident_decode_stage.fp8.h8192.l92.gdn69.e512k10.v1
MODULE_TARGET := cuda.sm121.qwen38.resident_decode_stage.fp8
MODEL_HEADER := ../../model-families/qwen38/include/sparkpipe/spark_qwen38_max_model.h
MODULE_ENTRY_PREFIX := SparkQwen38MaxResidentDecodeStage
MODULE_HOST_SOURCE := $(CURDIR)/source/spark_qwen38_max_resident_decode_stage_module.c
MODULE_CUDA_SOURCE := $(CURDIR)/source/spark_qwen38_max_resident_decode_stage_cuda.cu
MODULE_ADDITIONAL_HOST_SOURCES = \
	$(REPOSITORY_ROOT)/cache/store/stage_kv_client.c \
	$(REPOSITORY_ROOT)/cache/store/kv_store.c \
	$(REPOSITORY_ROOT)/model-families/qwen38/src/spark_qwen38_max_work_control.c \
	$(REPOSITORY_ROOT)/ring/transport/hidden_transport.c \
	$(REPOSITORY_ROOT)/ring/transport/tp_collective.c \
	$(REPOSITORY_ROOT)/ring/transport/tp_device_collective.c \
	$(REPOSITORY_ROOT)/ring/transport/tp_device_collective_nccl.c
# Adapter provenance pins: the vendor FP8 checkpoint revision and the exact
# package contract, checked at adapter compile time.
QWEN38_MODEL_REVISION := d2dc35658bcf77e66643428cb52e774cc3b5bd29
QWEN38_CONTRACT_SHA256 := $(shell sha256sum $(REPOSITORY_ROOT)/model_contracts/qwen38_authoritative.json | awk '{print $$1}')
MODULE_COMPILE_FLAGS := \
	-DSPARK_QWEN38_MAX_MODULE_BUILD=1 \
	-DQWEN38_MODEL_REVISION=\"$(QWEN38_MODEL_REVISION)\" \
	-DQWEN38_CONTRACT_SHA256=\"$(QWEN38_CONTRACT_SHA256)\"
MODULE_INCLUDE_FLAGS = \
	-I"$(REPOSITORY_ROOT)/include" \
	-I"$(REPOSITORY_ROOT)/src" \
	-I"$(REPOSITORY_ROOT)/model-families/common/include" \
	-I"$(REPOSITORY_ROOT)/model-families/qwen38/include" \
	-I"$(REPOSITORY_ROOT)/modules/qwen38_resident_decode_stage/include" \
	-I"$(REPOSITORY_ROOT)/modules/qwen38_resident_decode_stage/source" \
	-I"$(REPOSITORY_ROOT)"

STAGE_PACK_PATH ?= $(REPOSITORY_ROOT)/build/stagepacks/qwen38_stage.qwen38sp
STAGE_COUNT ?= 16
STAGE_INDEX ?= 0
STAGE_FIRST_LAYER ?= 0
STAGE_LAYER_COUNT ?= 23
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

RUNTIME_CONFIGURATION := \
	SPARK_QWEN38_MAX_ALLOW_UNQUALIFIED_EXECUTION=$(ALLOW_UNQUALIFIED_EXECUTION) \
	SPARK_QWEN38_MAX_STAGE_PACK_PATH=$(STAGE_PACK_PATH) \
	SPARK_QWEN38_MAX_STAGE_COUNT=$(STAGE_COUNT) \
	SPARK_QWEN38_MAX_STAGE_INDEX=$(STAGE_INDEX) \
	SPARK_QWEN38_MAX_STAGE_FIRST_LAYER=$(STAGE_FIRST_LAYER) \
	SPARK_QWEN38_MAX_STAGE_LAYER_COUNT=$(STAGE_LAYER_COUNT) \
	SPARK_QWEN38_MAX_STAGE_MAX_ACTIVE_SEQUENCES=$(MAX_ACTIVE_SEQUENCES) \
	SPARK_QWEN38_MAX_STAGE_PIPELINE_SLOTS=$(PIPELINE_SLOT_COUNT) \
	SPARK_QWEN38_MAX_STAGE_KV_BLOCKS=$(KV_BLOCK_COUNT) \
	SPARK_QWEN38_MAX_STAGE_KV_STORE=$(KV_PROVIDER) \
	SPARK_QWEN38_MAX_STAGE_KV_SERVICE=$(KV_SERVICE_ADDRESS) \
	SPARK_QWEN38_MAX_STAGE_KV_SOCKET=$(KV_SOCKET_PATH) \
	SPARK_QWEN38_MAX_STAGE_KV_POOL_BYTES=$(KV_POOL_BYTES) \
	SPARK_QWEN38_MAX_STAGE_KV_WORKERS=$(KV_WORKER_COUNT) \
	SPARK_QWEN38_MAX_STAGE_MTP=$(MTP_LAYER_COUNT) \
	SPARK_QWEN38_MAX_STAGE_GDN_SNAPSHOT_SLOTS=$(GDN_SNAPSHOT_SLOT_COUNT)

include $(REPOSITORY_ROOT)/modules/resident_decode_stage_rules.mk
