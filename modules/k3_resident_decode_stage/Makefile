MODULE_FAMILY := k3
MODULE_IDENTIFIER := spark.k3.resident_decode_stage.mxfp4.h7168.l72.e896.k16.kda128.mla512.rv256.v1
MODULE_TARGET := cuda.sm121.k3.resident_decode_stage.mxfp4
MODEL_HEADER := sparkpipe/spark_k3_model.h
MODULE_ENTRY_PREFIX := SparkK3ResidentDecodeStage
MODULE_HOST_SOURCE := $(CURDIR)/source/spark_k3_resident_decode_stage_module.c
MODULE_CUDA_SOURCE := $(CURDIR)/source/spark_k3_resident_decode_stage_cuda.cu

STAGE_PACK_PATH ?= $(REPOSITORY_ROOT)/build/stagepacks/k3_stage.k3sp
STAGE_COUNT ?= 1
STAGE_INDEX ?= 0
STAGE_FIRST_LAYER ?= 0
STAGE_LAYER_COUNT ?= 72
MAX_ACTIVE_LANES ?= 4
MAX_CONTEXT_TOKENS ?= 4096
PIPELINE_SLOT_COUNT ?= 2
ALLOW_UNQUALIFIED_EXECUTION ?= 0
PACK_SEED ?= 20260727
PACK_SYNTHESIZER := $(REPOSITORY_ROOT)/build/modules/k3_resident_decode_stage/k3_pack_synthesize
CPU_REFERENCE := $(REPOSITORY_ROOT)/build/modules/k3_resident_decode_stage/spark_k3_reference

RUNTIME_CONFIGURATION := \
	K3_ALLOW_UNQUALIFIED_EXECUTION=$(ALLOW_UNQUALIFIED_EXECUTION) \
	K3_STAGE_PACK=$(STAGE_PACK_PATH) \
	K3_MAX_LANES=$(MAX_ACTIVE_LANES) \
	K3_MAX_CONTEXT_TOKENS=$(MAX_CONTEXT_TOKENS) \
	K3_PIPELINE_SLOTS=$(PIPELINE_SLOT_COUNT) \
	K3_STAGE_COUNT=$(STAGE_COUNT) \
	K3_STAGE_INDEX=$(STAGE_INDEX) \
	K3_STAGE_FIRST_LAYER=$(STAGE_FIRST_LAYER) \
	K3_STAGE_LAYER_COUNT=$(STAGE_LAYER_COUNT)

include ../resident_decode_stage_rules.mk

.PHONY: tools pack reference

$(PACK_SYNTHESIZER): tools/k3_pack_synthesize.c | $(BUILD_DIRECTORY)
	$(CC) $(CFLAGS) $(MODULE_INCLUDE_FLAGS) $< -o $@ -lm

$(CPU_REFERENCE): validation/spark_k3_reference.c | $(BUILD_DIRECTORY)
	$(CC) $(CFLAGS) $(MODULE_INCLUDE_FLAGS) $< -o $@ -lm

tools: $(PACK_SYNTHESIZER) $(CPU_REFERENCE)

pack: $(PACK_SYNTHESIZER)
	mkdir -p $(dir $(STAGE_PACK_PATH))
	$(PACK_SYNTHESIZER) --output $(STAGE_PACK_PATH) --seed $(PACK_SEED)

reference: $(CPU_REFERENCE)
	$(CPU_REFERENCE)
