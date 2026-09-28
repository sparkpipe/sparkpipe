MODULE_FAMILY := mimo26
MODULE_IDENTIFIER := spark.mimo26.flash.resident_decode_stage.mxfp4.h4096.l48.tp4.v1
MODULE_TARGET := cuda.sm121.mimo26.resident_decode_stage.mxfp4
MODEL_HEADER := ../../model-families/mimo26/include/sparkpipe/spark_mimo26_model.h
MODULE_ENTRY_PREFIX := SparkMimo26ResidentDecodeStage
MODULE_HOST_SOURCE := $(CURDIR)/source/spark_mimo26_resident_decode_stage_module.c
MODULE_CUDA_SOURCE := $(CURDIR)/source/spark_mimo26_rank_engine.cu
include ../../runtime/weightd_sources.mk
MODULE_ADDITIONAL_HOST_SOURCES = \
	$(REPOSITORY_ROOT)/runtime/stage_module_lifecycle.c \
	$(addprefix $(REPOSITORY_ROOT)/,$(SPARKPIPE_WEIGHTD_SOURCES)) \
	$(REPOSITORY_ROOT)/ring/transport/hidden_transport.c \
	$(REPOSITORY_ROOT)/ring/transport/tp_collective.c \
	$(REPOSITORY_ROOT)/ring/transport/tp_device_collective.c
MIMO26_MODEL_REVISION ?= mimo26flash.mxfp4.tp4.v2
MIMO26_MODEL_DESCRIPTION_SHA256 := $(shell sha256sum $(CURDIR)/../../examples/model_descriptions/mimo26_resident_decode_stage_mxfp4_firmware.json 2>/dev/null | awk '{print $$1}')
MIMO26_CONTRACT_SHA256 ?= $(MIMO26_MODEL_DESCRIPTION_SHA256)
MODULE_COMPILE_FLAGS := \
	-DSPARK_MIMO26_MODULE_BUILD=1 \
	-DMIMO26_MODEL_REVISION=\"$(MIMO26_MODEL_REVISION)\" \
	-DMIMO26_CONTRACT_SHA256=\"$(MIMO26_CONTRACT_SHA256)\"
MODULE_INCLUDE_FLAGS = \
	-I"$(REPOSITORY_ROOT)/include" \
	-I"$(REPOSITORY_ROOT)/src" \
	-I"$(REPOSITORY_ROOT)/model-families/common/include" \
	-I"$(REPOSITORY_ROOT)/model-families/mimo26/include" \
	-I"$(REPOSITORY_ROOT)/modules/mimo26_resident_decode_stage/include" \
	-I"$(REPOSITORY_ROOT)/modules/mimo26_resident_decode_stage/source" \
	-I"$(REPOSITORY_ROOT)"

STAGE_PACK_PATH ?= $(REPOSITORY_ROOT)/build/stagepacks/mimo26flash.mxfp4.tp4.rank0.sp
MAX_ACTIVE_SEQUENCES ?= 4
PIPELINE_SLOT_COUNT ?= 1
MAX_POSITIONS ?= 8192
TP_RANK ?= 0
GRAPH ?= 0

RUNTIME_CONFIGURATION := \
	SPARK_MIMO26_ALLOW_UNQUALIFIED_EXECUTION=1 \
	SPARK_MIMO26_TP_RANK=$(TP_RANK) \
	SPARK_MIMO26_STAGE_PACK_PATH=$(STAGE_PACK_PATH) \
	SPARK_MIMO26_STAGE_MAX_ACTIVE_SEQUENCES=$(MAX_ACTIVE_SEQUENCES) \
	SPARK_MIMO26_STAGE_PIPELINE_SLOTS=$(PIPELINE_SLOT_COUNT) \
	SPARK_MIMO26_STAGE_MAX_POSITIONS=$(MAX_POSITIONS) \
	SPARK_MIMO26_STAGE_GRAPH=$(GRAPH)

EXPERT_CODEC ?= mxfp4
ifeq ($(filter $(EXPERT_CODEC),mxfp4),)
$(error unsupported EXPERT_CODEC '$(EXPERT_CODEC)': the mimo26 flash module builds the checkpoint's native mxfp4 experts only)
endif
BUILD_DIRECTORY ?= $(REPOSITORY_ROOT)/build/modules/mimo26_resident_decode_stage/$(EXPERT_CODEC)

include ../resident_decode_stage_rules.mk

ADAPTER_SOURCE := source/spark_mimo26_serving_adapter.c
ADAPTER_FLAGS = -D_POSIX_C_SOURCE=200809L -DMIMO26_MODEL_REVISION=\"$(MIMO26_MODEL_REVISION)\" -DMIMO26_CONTRACT_SHA256=\"$(MIMO26_CONTRACT_SHA256)\" \
	-DMIMO26_MODEL_DESCRIPTION_SHA256=\"$(MIMO26_MODEL_DESCRIPTION_SHA256)\" -I"$(CUDA_HOME)/include" $(MODULE_INCLUDE_FLAGS) -DSPARK_BATCH_BUCKET=1024u
ADAPTER_LINK = $(ADAPTER_CUDA_LINK)
