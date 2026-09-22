MODULE_FAMILY := minimax
MODULE_IDENTIFIER := spark.minimax.resident_decode_stage.bf16.h5120.l64.tp4.v1
MODULE_TARGET := cuda.sm121.minimax.resident_decode_stage.bf16
MODEL_HEADER := ../../modules/minimax_resident_decode_stage/include/sparkpipe/spark_minimax_model.h
MODULE_ENTRY_PREFIX := SparkMinimaxResidentDecodeStage
MODULE_HOST_SOURCE := $(CURDIR)/source/spark_minimax_resident_decode_stage_module.c
MODULE_CUDA_SOURCE := $(CURDIR)/source/spark_minimax_resident_decode_stage_cuda.cu
include ../../runtime/weightd_sources.mk
MODULE_ADDITIONAL_HOST_SOURCES = \
	$(REPOSITORY_ROOT)/runtime/stage_module_lifecycle.c \
	$(REPOSITORY_ROOT)/runtime/stagepack_format.c \
	$(addprefix $(REPOSITORY_ROOT)/,$(SPARKPIPE_WEIGHTD_SOURCES)) \
	$(REPOSITORY_ROOT)/src/spark_status.c \
	$(REPOSITORY_ROOT)/src/spark_sha256.c \
	$(REPOSITORY_ROOT)/src/spark_ck128.c \
	$(REPOSITORY_ROOT)/ring/transport/hidden_transport.c \
	$(REPOSITORY_ROOT)/ring/transport/tp_collective.c \
	$(REPOSITORY_ROOT)/ring/transport/tp_device_collective.c
MINIMAX_MODEL_REVISION ?= minimax-h3-text-bf16-h5120-l64-mrope24-20-20-tp4-v1
# The package contract is the firmware description the driver compiles from:
# the adapter's baked contract sha must equal the compiled driver's
# model_description_sha256 (loader equality), so both sides derive from this file.
MINIMAX_CONTRACT_SHA256 ?= $(shell sha256sum $(CURDIR)/../../examples/model_descriptions/minimax_resident_decode_stage_bf16_firmware.json 2>/dev/null | awk '{print $$1}')
MODULE_COMPILE_FLAGS := \
	-DSPARK_MINIMAX_MODULE_BUILD=1 \
	-DMINIMAX_MODEL_REVISION=\"$(MINIMAX_MODEL_REVISION)\" \
	-DMINIMAX_CONTRACT_SHA256=\"$(MINIMAX_CONTRACT_SHA256)\"
MODULE_INCLUDE_FLAGS = \
	-I"$(REPOSITORY_ROOT)/include" \
	-I"$(REPOSITORY_ROOT)/src" \
	-I"$(REPOSITORY_ROOT)/model-families/common/include" \
	-I"$(REPOSITORY_ROOT)/model-families/minimax/include" \
	-I"$(REPOSITORY_ROOT)/model-families/minimax/include/sparkpipe" \
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

# Digest pin must precede RUNTIME_CONFIGURATION (simple := expansion).
override MINIMAX_CUDA_VALIDATOR_SHA256 := $(shell sha256sum validation/spark_minimax_resident_decode_stage_cuda_validation.cu 2>/dev/null | awk '{print $$1}')

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
		SPARK_MINIMAX_STAGE_KV_WORKERS=$(KV_WORKER_COUNT) \
		SPARK_MINIMAX_CUDA_VALIDATOR_SHA256=$(MINIMAX_CUDA_VALIDATOR_SHA256)

GPU_VALIDATOR := validation/validate_minimax_resident_decode_stage_cuda.sh

# Dense text tower: bf16 is the only codec this module builds (no routed
# experts, no quantization arms). EXPERT_CODEC still names the build slot so
# tools/module_build_release.sh and the shared adapter/library layout stay
# uniform across families (gemma4 dense precedent). These MUST precede the
# rules include: the rules bind BUILD_DIRECTORY eagerly.
EXPERT_CODEC ?= bf16
ifeq ($(filter $(EXPERT_CODEC),bf16),)
$(error unsupported EXPERT_CODEC '$(EXPERT_CODEC)': the dense minimax text-tower module builds bf16 only)
endif
BUILD_DIRECTORY ?= $(REPOSITORY_ROOT)/build/modules/minimax_resident_decode_stage/$(EXPERT_CODEC)

include ../resident_decode_stage_rules.mk

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
ADAPTER_SHARED_FLAGS := -dynamiclib
ADAPTER_LIBRARY_EXT := dylib
else
ADAPTER_SHARED_FLAGS := -shared
ADAPTER_LIBRARY_EXT := so
endif
ADAPTER_LIBRARY := $(BUILD_DIRECTORY)/libminimax_serving_adapter_$(EXPERT_CODEC).$(ADAPTER_LIBRARY_EXT)

.PHONY: adapter

# TP4 text-tower serving adapter in the codec-scoped module build slot
# tools/module_build_release.sh expects (gemma4 dense precedent).
adapter:
	$(MAKE) -C $(REPOSITORY_ROOT) core model_common
	@mkdir -p "$(BUILD_DIRECTORY)"
	$(CC) $(CFLAGS) $(MODULE_POSIX_FLAGS) -fPIC $(ADAPTER_SHARED_FLAGS) \
		-D_POSIX_C_SOURCE=200809L \
		-DMINIMAX_MODEL_REVISION=\"$(MINIMAX_MODEL_REVISION)\" \
		-DMINIMAX_CONTRACT_SHA256=\"$(MINIMAX_CONTRACT_SHA256)\" \
		-I"$(CUDA_HOME)/include" \
		$(MODULE_INCLUDE_FLAGS) -DSPARK_BATCH_BUCKET=1024u \
		source/spark_minimax_serving_adapter.c \
		$(REPOSITORY_ROOT)/build/libsparkpipe_runtime.a \
		$(REPOSITORY_ROOT)/build/libsparkpipe_model_common.a \
		$(REPOSITORY_ROOT)/build/libsparkpipe_core.a -ldl -pthread \
		-o "$(ADAPTER_LIBRARY)"
