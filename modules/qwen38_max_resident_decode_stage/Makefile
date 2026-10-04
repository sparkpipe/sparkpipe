MODULE_FAMILY := qwen38
include ../../runtime/weightd_sources.mk

ifndef EXPERT_CODEC
$(error EXPERT_CODEC is required: fp8)
endif
QWEN38_MAX_EXPERT_CODECS := fp8
ifeq ($(filter $(EXPERT_CODEC),$(QWEN38_MAX_EXPERT_CODECS)),)
$(error unsupported EXPERT_CODEC '$(EXPERT_CODEC)': choose $(QWEN38_MAX_EXPERT_CODECS))
endif
ifndef MODEL_REVISION
$(error MODEL_REVISION is required and must be the exact source snapshot revision)
endif
ifndef CONTRACT_SHA256
$(error CONTRACT_SHA256 is required and must identify the exact model package contract)
endif

MODULE_IDENTIFIER := spark.qwen38.resident_decode_stage.fp8.h8192.l92.gdn69.e512k10.v1
MODULE_TARGET := cuda.sm121.qwen38.resident_decode_stage.fp8
MODEL_HEADER := ../../model-families/qwen38_max/include/sparkpipe/spark_qwen38_max_model.h
MODULE_ENTRY_PREFIX := SparkQwen38MaxResidentDecodeStage
MODULE_HOST_SOURCE := $(CURDIR)/source/spark_qwen38_max_resident_decode_stage_module.c
MODULE_CUDA_SOURCE := $(CURDIR)/source/spark_qwen38_max_resident_decode_stage_cuda.cu
MODULE_ADDITIONAL_HOST_SOURCES = \
	$(REPOSITORY_ROOT)/runtime/stage_module_lifecycle.c \
	$(REPOSITORY_ROOT)/runtime/stagepack_format.c \
	$(addprefix $(REPOSITORY_ROOT)/,$(SPARKPIPE_WEIGHTD_SOURCES)) \
	$(REPOSITORY_ROOT)/src/spark_status.c \
	$(REPOSITORY_ROOT)/src/spark_sha256.c \
	$(REPOSITORY_ROOT)/src/spark_ck128.c \
	$(REPOSITORY_ROOT)/cache/store/stage_kv_client.c \
	$(REPOSITORY_ROOT)/cache/store/kv_store.c \
	$(REPOSITORY_ROOT)/model-families/qwen38_max/src/spark_qwen38_max_work_control.c \
	$(REPOSITORY_ROOT)/ring/transport/hidden_transport.c \
	$(REPOSITORY_ROOT)/ring/transport/tp_device_collective.c
# Adapter provenance pins: the vendor FP8 checkpoint revision and the exact
# package contract, checked at adapter compile time (the adapter #errors
# without both, so a drifted build cannot link silently). Passed in by the
# build invocation (MODEL_REVISION / CONTRACT_SHA256), the glm5_next
# convention. REPOSITORY_ROOT is only defined inside
# resident_decode_stage_rules.mk, so this Makefile must never expand it
# eagerly: the previous in-Makefile contract digest ran $(shell sha256sum
# $(REPOSITORY_ROOT)/...) at parse time - before the include - which pinned
# an EMPTY contract sha into every artifact (build-r2 attempt 1c1077ce).
# The serving-adapter request contract must carry the model-description
# sha the driver compile embeds (serving_adapter_template.c compares both
# sides at adapter_initialize; the qwen38-common default passes the
# package contract sha, which can never match a driver built from a
# separate firmware description - the gemma4 launch-9 lesson). CURDIR is
# a make builtin, so this stays eager-safe under the REPOSITORY_ROOT law.
QWEN38_MAX_MODEL_DESCRIPTION_SHA256 := $(shell sha256sum $(CURDIR)/../../examples/model_descriptions/qwen38_max_resident_decode_stage_firmware.json 2>/dev/null | awk '{print $$1}')
MTP_LAYER_COUNT ?= 0
MODULE_COMPILE_FLAGS := \
	-DSPARK_QWEN38_MAX_MODULE_BUILD=1 \
	-DSPARK_LLM_MTP_LAYER_COUNT=$(MTP_LAYER_COUNT)u \
	-DQWEN38_MODEL_REVISION=\"$(MODEL_REVISION)\" \
	-DQWEN38_CONTRACT_SHA256=\"$(CONTRACT_SHA256)\"
MODULE_INCLUDE_FLAGS = \
	-I"$(REPOSITORY_ROOT)/include" \
	-I"$(REPOSITORY_ROOT)/src" \
	-I"$(REPOSITORY_ROOT)/model-families/common/include" \
	-I"$(REPOSITORY_ROOT)/model-families/qwen38_max/include" \
	-I"$(REPOSITORY_ROOT)/modules/qwen38_max_resident_decode_stage/include" \
	-I"$(REPOSITORY_ROOT)/modules/qwen38_max_resident_decode_stage/source" \
	-I"$(REPOSITORY_ROOT)"

# Codec-partitioned build outputs (glm5_next layout): the wrapper resolves the
# adapter at build/modules/qwen38_max_resident_decode_stage/<codec>/lib..._so.
# ?= keeps the expansion deferred until resident_decode_stage_rules.mk has
# defined REPOSITORY_ROOT; the rules default would name the family
# (qwen38_resident_decode_stage, without the _max) and drop the codec leg.
BUILD_DIRECTORY ?= $(REPOSITORY_ROOT)/build/modules/qwen38_max_resident_decode_stage/$(EXPERT_CODEC)
MODULE_ARCHIVE ?= $(BUILD_DIRECTORY)/libqwen38_max_resident_decode_stage_$(EXPERT_CODEC).a
# The wrapper (the dsv4 pattern) computes the validation .cu digest and
# exports SPARK_QWEN38_MAX_CUDA_VALIDATOR_SHA256 before exec'ing the real
# validator; pointing GPU_VALIDATOR at the raw validator fails its source
# digest check with no digest in the publish environment (build-r5b, all
# 16 nodes: "Qwen38_max CUDA validator expected SHA-256 is invalid").
GPU_VALIDATOR := validation/publish_validator_wrapper.sh

STAGE_PACK_PATH ?= $(REPOSITORY_ROOT)/build/stagepacks/qwen38_stage.qwen38sp
STAGE_COUNT ?= 16
STAGE_INDEX ?= 0
STAGE_FIRST_LAYER ?= 0
STAGE_LAYER_COUNT ?= 23
MAX_ACTIVE_SEQUENCES ?= 32
PIPELINE_SLOT_COUNT ?= 2
KV_BLOCK_COUNT ?= 4096
GDN_SNAPSHOT_SLOT_COUNT ?= 0
KV_PROVIDER ?= none
KV_SERVICE_ADDRESS ?= none
KV_SOCKET_PATH ?= none
KV_POOL_BYTES ?= 0
KV_WORKER_COUNT ?= 0
TP_STANDALONE ?= 0

RUNTIME_CONFIGURATION := \
	SPARK_QWEN38_MAX_TP_STANDALONE=$(TP_STANDALONE) \
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

include ../resident_decode_stage_rules.mk

# Publish-validation TP geometry (AFTER the include: REPOSITORY_ROOT is
# defined by the rules): the placed packs are TP16 rank shards (512/16
# = 32 experts per layer), so the module's manifest check and rank-local
# math need degree 16 + THIS node's rank even though the single-node
# validation never opens the inter-node collective (the module skips it
# under TP_STANDALONE=1). Derived from the host name so
# the per-node fleet form needs no env plumbing.
VALIDATION_HOST_SUFFIX := $(shell hostname | sed 's/^spark//')
VALIDATION_NODE_RANK := $(shell printf '%d' 0x$(VALIDATION_HOST_SUFFIX))
ifeq ($(TP_STANDALONE),1)
RUNTIME_CONFIGURATION += \
	SPARK_QWEN38_MAX_STAGE_TP_DEGREE=16 \
	SPARK_QWEN38_MAX_STAGE_TP_RANK=$(VALIDATION_NODE_RANK)
endif

ADAPTER_SOURCE := source/spark_qwen38_max_serving_adapter.c
ADAPTER_FLAGS = $(ADAPTER_CUDA_FLAGS) -DQWEN38_MAX_MODEL_DESCRIPTION_SHA256=\"$(QWEN38_MAX_MODEL_DESCRIPTION_SHA256)\"
ADAPTER_LINK = $(ADAPTER_CUDA_LINK)
