MODULE_FAMILY := qwen38_27b
MODULE_IDENTIFIER := spark.qwen38_27b.resident_decode_stage.bf16.h5120.l64.gdn48.mtp1.v1
MODULE_TARGET := cuda.sm121.qwen38_27b.resident_decode_stage.bf16
MODEL_HEADER := ../../model-families/qwen38_27b/include/sparkpipe/spark_qwen38_27b_model.h
MODULE_ENTRY_PREFIX := SparkQwen38_27bResidentDecodeStage
MODULE_HOST_SOURCE := $(CURDIR)/source/spark_qwen38_27b_resident_decode_stage_module.c
MODULE_CUDA_SOURCE := $(CURDIR)/source/spark_qwen38_27b_resident_decode_stage_cuda.cu
# model-families/common/src was deleted with the library rewrite; the KV
# client and store implementations now live under cache/store (same ABI).
MODULE_ADDITIONAL_HOST_SOURCES = \
	$(REPOSITORY_ROOT)/runtime/stage_module_lifecycle.c \
	$(REPOSITORY_ROOT)/cache/store/stage_kv_client.c \
	$(REPOSITORY_ROOT)/cache/store/kv_store.c \
	$(REPOSITORY_ROOT)/ring/transport/hidden_transport.c \
	$(REPOSITORY_ROOT)/ring/transport/tp_device_collective.c \
	$(REPOSITORY_ROOT)/ring/transport/tp_device_collective_nccl.c \
	$(CURDIR)/source/spark_qwen38_27b_tp.c
# The rewritten rules.mk requires a non-empty MODULE_COMPILE_FLAGS; mirror the
# dsv4 convention so a future module-build define has a home.
MODULE_COMPILE_FLAGS := -DSPARK_QWEN38_27B_MODULE_BUILD=1
# spark_lm_kernels.cuh reaches inference/kernels/*.cuh by repo-root-relative
# path; the default flags in rules.mk stop at the family include trees.
# Recursive assignment: REPOSITORY_ROOT is only set inside rules.mk.
MODULE_INCLUDE_FLAGS = \
	-I"$(REPOSITORY_ROOT)/include" \
	-I"$(REPOSITORY_ROOT)/model-families/common/include" \
	-I"$(REPOSITORY_ROOT)/model-families/qwen38_27b/include" \
	-I"$(REPOSITORY_ROOT)/modules/qwen38_27b_resident_decode_stage/include" \
	-I"$(REPOSITORY_ROOT)/modules/qwen38_27b_resident_decode_stage/source" \
	-I"$(REPOSITORY_ROOT)"

STAGE_PACK_PATH ?= $(abspath ../../build/stagepacks/qwen38_27b_stage.qwen38_27bsp)
STAGE_COUNT ?= 1
STAGE_INDEX ?= 0
STAGE_FIRST_LAYER ?= 0
STAGE_LAYER_COUNT ?= 64
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
override QWEN38_27B_CUDA_VALIDATOR_SHA256 := $(shell sha256sum validation/spark_qwen38_27b_resident_decode_stage_cuda_validation.cu | awk '{print $$1}')
override QWEN38_27B_CPU_ORACLE_SHA256 := $(shell sha256sum validation/spark_qwen38_27b_reference.c | awk '{print $$1}')

RUNTIME_CONFIGURATION := \
	SPARK_QWEN38_27B_ALLOW_UNQUALIFIED_EXECUTION=$(ALLOW_UNQUALIFIED_EXECUTION) \
	SPARK_QWEN38_27B_TP_DEGREE=$(TP_DEGREE) \
	SPARK_QWEN38_27B_TP_RANK=$(TP_RANK) \
	SPARK_QWEN38_27B_TP_STANDALONE=$(TP_STANDALONE) \
	SPARK_QWEN38_27B_STAGE_PACK_PATH=$(STAGE_PACK_PATH) \
	SPARK_QWEN38_27B_STAGE_COUNT=$(STAGE_COUNT) \
	SPARK_QWEN38_27B_STAGE_INDEX=$(STAGE_INDEX) \
	SPARK_QWEN38_27B_STAGE_FIRST_LAYER=$(STAGE_FIRST_LAYER) \
	SPARK_QWEN38_27B_STAGE_LAYER_COUNT=$(STAGE_LAYER_COUNT) \
	SPARK_QWEN38_27B_STAGE_MAX_ACTIVE_SEQUENCES=$(MAX_ACTIVE_SEQUENCES) \
	SPARK_QWEN38_27B_STAGE_PIPELINE_SLOTS=$(PIPELINE_SLOT_COUNT) \
	SPARK_QWEN38_27B_STAGE_KV_BLOCKS=$(KV_BLOCK_COUNT) \
	SPARK_QWEN38_27B_STAGE_MTP=$(MTP_LAYER_COUNT) \
	SPARK_QWEN38_27B_STAGE_GDN_SNAPSHOT_SLOTS=$(GDN_SNAPSHOT_SLOT_COUNT) \
	SPARK_QWEN38_27B_STAGE_KV_STORE=$(KV_PROVIDER) \
	SPARK_QWEN38_27B_STAGE_KV_SERVICE=$(KV_SERVICE_ADDRESS) \
	SPARK_QWEN38_27B_STAGE_KV_SOCKET=$(KV_SOCKET_PATH) \
	SPARK_QWEN38_27B_STAGE_KV_POOL_BYTES=$(KV_POOL_BYTES) \
	SPARK_QWEN38_27B_STAGE_KV_WORKERS=$(KV_WORKER_COUNT) \
	SPARK_QWEN38_27B_CUDA_VALIDATOR_SHA256=$(QWEN38_27B_CUDA_VALIDATOR_SHA256) \
	SPARK_QWEN38_27B_CPU_ORACLE_SHA256=$(QWEN38_27B_CPU_ORACLE_SHA256)

GPU_VALIDATOR := validation/validate_qwen38_27b_resident_decode_stage_cuda.sh

include ../resident_decode_stage_rules.mk
