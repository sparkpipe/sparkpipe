# Keep target and prerequisite names relative to this module.  GNU Make treats
# spaces in absolute path names as target separators, so expanding abspath here
# breaks builds from otherwise valid workspaces whose names contain a space.
REPOSITORY_ROOT := ../..
NVCC ?= nvcc
AR ?= ar
CUDA_ARCH ?= sm_121a
CUDA_COMPUTE_ARCH := $(subst sm_,compute_,$(CUDA_ARCH))
NVCCFLAGS ?= -std=c++17 -O3 -Xcompiler=-fPIC -Xcompiler=-fvisibility=hidden
BUILD_DIRECTORY := $(REPOSITORY_ROOT)/build/modules/glm52_dspark_draft_backend
BACKEND_OBJECT := $(BUILD_DIRECTORY)/spark_glm52_dspark_draft_backend.o
BACKEND_ARCHIVE := $(BUILD_DIRECTORY)/libglm52_dspark_draft_backend.a
VALIDATOR := $(BUILD_DIRECTORY)/validate_glm52_dspark_epoch3_cuda
DSPARK_MANIFEST ?=
DSPARK_CONFIG ?=
DSPARK_SAFETENSORS ?=
DSPARK_ORACLE_DIR ?=
DSPARK_VALIDATION_LANE_COUNT ?= 2

.PHONY: archive validate_epoch3 clean

archive: $(BACKEND_ARCHIVE)

$(BUILD_DIRECTORY):
	mkdir -p $(BUILD_DIRECTORY)

# Include paths: the drafter constants live in the neutral
# spark_dspark_drafter.h (model-families/common) behind the GLM52 target table,
# and the backend launches GEMMs through runtime/gemm.cuh, whose includes are
# repo-root relative. The validator links -lcuda (the shared stack encodes TMA
# descriptors through the driver API); cuBLAS is gone from the tree.
$(BACKEND_OBJECT): source/spark_glm52_dspark_draft_backend.cu include/sparkpipe/spark_glm52_dspark_draft_backend.h | $(BUILD_DIRECTORY)
	$(NVCC) -gencode arch=$(CUDA_COMPUTE_ARCH),code=$(CUDA_ARCH) $(NVCCFLAGS) -I$(REPOSITORY_ROOT) \
		-I$(REPOSITORY_ROOT)/model-families/common/include \
		-I$(REPOSITORY_ROOT)/model-families/glm52/include \
		-I$(REPOSITORY_ROOT)/include -Iinclude -c $< -o $@

$(BACKEND_ARCHIVE): $(BACKEND_OBJECT)
	rm -f $@
	$(AR) rcs $@ $^
