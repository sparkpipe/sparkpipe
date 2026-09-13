ifndef GLM_FAMILY
$(error GLM_FAMILY is required before including glm_resident_stage_wrapper.mk)
endif
ifndef GLM_MODULE_ROOT
$(error GLM_MODULE_ROOT is required before including glm_resident_stage_wrapper.mk)
endif
ifndef EXPERT_CODEC
$(error EXPERT_CODEC is required: $(GLM_EXPERT_CODECS))
endif
ifeq ($(filter $(EXPERT_CODEC),$(GLM_EXPERT_CODECS)),)
$(error unsupported EXPERT_CODEC '$(EXPERT_CODEC)': choose $(GLM_EXPERT_CODECS))
endif
ifndef MODEL_REVISION
$(error MODEL_REVISION is required and must be the exact source snapshot revision)
endif
ifndef CONTRACT_SHA256
$(error CONTRACT_SHA256 is required and must identify the exact model package contract)
endif

GLM_CODEC_ID_BF16 := 1
GLM_CODEC_ID_INT6 := 2
GLM_CODEC_ID_INT7 := 3
GLM_CODEC_ID_INT8 := 4
GLM_CODEC_ID_FP8 := 5
GLM_CODEC_ID_NVFP4 := 6
GLM_CODEC_ID_MXFP4 := 7
GLM_EXPERT_CODEC_ID := $(GLM_CODEC_ID_$(shell echo $(EXPERT_CODEC) | tr a-z A-Z))

MODULE_IDENTIFIER := $(MODULE_IDENTIFIER_PREFIX).$(MODULE_IDENTIFIER_SUFFIX)
MODULE_BATCH_VARIANT_BUCKETS ?= 1 2 4 8 16 32 64 128 256 512 1024
MODULE_COMPILE_FLAGS := \
	-DGLM_EXPERT_WEIGHT_CODEC=$(GLM_EXPERT_CODEC_ID) \
	-DGLM_EXPERT_CODEC_NAME=\"$(EXPERT_CODEC)\" \
	-DGLM_MODEL_REVISION=\"$(MODEL_REVISION)\" \
	-DGLM_CONTRACT_SHA256=\"$(CONTRACT_SHA256)\"
MODULE_INCLUDE_FLAGS := \
	-I$(GLM_REPO_ROOT)/include \
	-I$(GLM_REPO_ROOT)/model-families/common/include \
	-I$(GLM_REPO_ROOT)/model-families/$(GLM_FAMILY)/include \
	-I$(GLM_REPO_ROOT)/modules/$(GLM_FAMILY)_resident_decode_stage/include \
	-I$(GLM_REPO_ROOT)/modules/$(GLM_FAMILY)_resident_decode_stage/source \
	-I$(GLM_REPO_ROOT)
BUILD_DIRECTORY ?= $(GLM_BUILD_ROOT)/$(GLM_FAMILY)_resident_decode_stage/$(EXPERT_CODEC)
MODULE_ARCHIVE ?= $(BUILD_DIRECTORY)/lib$(GLM_FAMILY)_resident_decode_stage_$(EXPERT_CODEC).a
MODULE_COMMON_HOST_SOURCES := $(GLM_REPO_ROOT)/runtime/stage_module_common.c

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
GLM_ADAPTER_SHARED_FLAGS := -dynamiclib
GLM_ADAPTER_LIBRARY_EXT := dylib
else
GLM_ADAPTER_SHARED_FLAGS := -shared
GLM_ADAPTER_LIBRARY_EXT := so
endif
GLM_ADAPTER_SOURCE := $(GLM_MODULE_ROOT)/source/spark_$(GLM_FAMILY)_serving_adapter.c
ADAPTER_LIBRARY := $(BUILD_DIRECTORY)/lib$(GLM_FAMILY)_serving_adapter_$(EXPERT_CODEC).$(GLM_ADAPTER_LIBRARY_EXT)

.PHONY: adapter

adapter:
	$(MAKE) -C $(GLM_REPO_ROOT) core model_common
	@mkdir -p "$(BUILD_DIRECTORY)"
	$(CC) $(CFLAGS) $(MODULE_POSIX_FLAGS) -fPIC $(GLM_ADAPTER_SHARED_FLAGS) \
		-I$(GLM_REPO_ROOT)/src $(MODULE_INCLUDE_FLAGS) $(MODULE_COMPILE_FLAGS) -USPARK_BATCH_BUCKET -DSPARK_BATCH_BUCKET=$(lastword $(MODULE_BATCH_VARIANT_BUCKETS)) \
		$(GLM_ADAPTER_SOURCE) \
		$(GLM_REPO_ROOT)/build/libsparkpipe_runtime.a \
		$(GLM_REPO_ROOT)/build/libsparkpipe_model_common.a \
		$(GLM_REPO_ROOT)/build/libsparkpipe_core.a -ldl -pthread \
		-o "$(ADAPTER_LIBRARY)"
