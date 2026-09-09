MODULE_FAMILY := muse
MODULE_IDENTIFIER := spark.muse.resident_decode_stage.bf16.h6656.l52.a52.v1
MODULE_TARGET := cuda.sm121.muse.resident_decode_stage.bf16
MODEL_HEADER := ../../model-families/muse_glimmer/include/sparkpipe/spark_muse_glimmer_model.h
MODULE_ENTRY_PREFIX := SparkMuseGlimmerResidentDecodeStage
MODULE_HOST_SOURCE := $(CURDIR)/source/spark_muse_glimmer_resident_decode_stage_module.c
MODULE_CUDA_SOURCE := $(CURDIR)/source/spark_muse_glimmer_resident_decode_stage_cuda.cu
MODULE_ADDITIONAL_HOST_SOURCES = \
	$(REPOSITORY_ROOT)/runtime/stagepack_format.c \
	$(REPOSITORY_ROOT)/cache/store/stage_kv_client.c \
	$(REPOSITORY_ROOT)/cache/store/kv_store.c \
	$(REPOSITORY_ROOT)/model-families/muse_glimmer/src/spark_muse_glimmer_work_control.c \
	$(REPOSITORY_ROOT)/ring/transport/hidden_transport.c \
	$(REPOSITORY_ROOT)/ring/transport/tp_collective.c \
	$(REPOSITORY_ROOT)/ring/transport/tp_device_collective.c \
	$(REPOSITORY_ROOT)/ring/transport/tp_device_collective_nccl.c
# Provenance pins: the pinned HF modeling-source revision and the exact
# package contract, checked at adapter compile time.
MUSE_MODEL_REVISION := 4177486a9f199bd7be520eff14431071d5d41ec5
MUSE_CONTRACT_SHA256 := $(shell sha256sum $(REPOSITORY_ROOT)/model_contracts/muse_glimmer_authoritative.json | awk '{print $$1}')
MODULE_COMPILE_FLAGS := \
	-DSPARK_MUSE_GLIMMER_MODULE_BUILD=1 \
	-DMUSE_MODEL_REVISION=\"$(MUSE_MODEL_REVISION)\" \
	-DMUSE_CONTRACT_SHA256=\"$(MUSE_CONTRACT_SHA256)\"
MODULE_INCLUDE_FLAGS = \
	-I"$(REPOSITORY_ROOT)/include" \
	-I"$(REPOSITORY_ROOT)/src" \
	-I"$(REPOSITORY_ROOT)/model-families/common/include" \
	-I"$(REPOSITORY_ROOT)/model-families/muse_glimmer/include" \
	-I"$(REPOSITORY_ROOT)/modules/muse_glimmer_resident_decode_stage/include" \
	-I"$(REPOSITORY_ROOT)/modules/muse_glimmer_resident_decode_stage/source" \
	-I"$(REPOSITORY_ROOT)"

STAGE_PACK_PATH ?= $(REPOSITORY_ROOT)/build/stagepacks/muse_stage.gsmu
STAGE_COUNT ?= 1
STAGE_INDEX ?= 0
STAGE_FIRST_LAYER ?= 0
STAGE_LAYER_COUNT ?= 52
MAX_ACTIVE_SEQUENCES ?= 32
PIPELINE_SLOT_COUNT ?= 2
KV_BLOCK_COUNT ?= 4096
KV_PROVIDER ?= none
KV_SERVICE_ADDRESS ?= none
KV_SOCKET_PATH ?= none
KV_POOL_BYTES ?= 0
KV_WORKER_COUNT ?= 0
ALLOW_UNQUALIFIED_EXECUTION ?= 0

RUNTIME_CONFIGURATION := \
	SPARK_MUSE_GLIMMER_ALLOW_UNQUALIFIED_EXECUTION=$(ALLOW_UNQUALIFIED_EXECUTION) \
	SPARK_MUSE_GLIMMER_STAGE_PACK_PATH=$(STAGE_PACK_PATH) \
	SPARK_MUSE_GLIMMER_STAGE_COUNT=$(STAGE_COUNT) \
	SPARK_MUSE_GLIMMER_STAGE_INDEX=$(STAGE_INDEX) \
	SPARK_MUSE_GLIMMER_STAGE_FIRST_LAYER=$(STAGE_FIRST_LAYER) \
	SPARK_MUSE_GLIMMER_STAGE_LAYER_COUNT=$(STAGE_LAYER_COUNT) \
	SPARK_MUSE_GLIMMER_STAGE_MAX_ACTIVE_SEQUENCES=$(MAX_ACTIVE_SEQUENCES) \
	SPARK_MUSE_GLIMMER_STAGE_PIPELINE_SLOTS=$(PIPELINE_SLOT_COUNT) \
	SPARK_MUSE_GLIMMER_STAGE_KV_BLOCKS=$(KV_BLOCK_COUNT) \
	SPARK_MUSE_GLIMMER_STAGE_KV_STORE=$(KV_PROVIDER) \
	SPARK_MUSE_GLIMMER_STAGE_KV_SERVICE=$(KV_SERVICE_ADDRESS) \
	SPARK_MUSE_GLIMMER_STAGE_KV_SOCKET=$(KV_SOCKET_PATH) \
	SPARK_MUSE_GLIMMER_STAGE_KV_POOL_BYTES=$(KV_POOL_BYTES) \
	SPARK_MUSE_GLIMMER_STAGE_KV_WORKERS=$(KV_WORKER_COUNT)

include $(REPOSITORY_ROOT)/modules/resident_decode_stage_rules.mk
