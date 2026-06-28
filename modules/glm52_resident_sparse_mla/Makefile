REPOSITORY_ROOT := $(abspath ../..)
CC ?= cc
AR ?= ar
NVCC ?= nvcc
CUDA_ARCH ?= sm_121
CFLAGS ?= -std=c11 -Wall -Wextra -Werror -O3
NVCCFLAGS ?= -std=c++17 -O3 --use_fast_math -lineinfo -arch=$(CUDA_ARCH)
MODULE_LIBRARY_ROOT ?= $(REPOSITORY_ROOT)/build/module_library
MAX_STAGE_MICROSECONDS ?= 0
BUILD_DIRECTORY := $(REPOSITORY_ROOT)/build/modules/glm52_resident_sparse_mla
HOST_OBJECT := $(BUILD_DIRECTORY)/spark_glm52_resident_sparse_mla_module.o
CUDA_OBJECT := $(BUILD_DIRECTORY)/spark_glm52_resident_sparse_mla_cuda.o
HOST_DEPENDENCY := $(HOST_OBJECT:.o=.d)
CUDA_DEPENDENCY := $(CUDA_OBJECT:.o=.d)
MODULE_ARCHIVE := $(BUILD_DIRECTORY)/libglm52_resident_sparse_mla.a
PACKAGE_DIRECTORY ?= $(REPOSITORY_ROOT)/build/packages/glm52_resident_sparse_mla
CUDA_HOME ?= $(abspath $(dir $(shell command -v $(NVCC)))/..)
VALIDATION_RECIPE := glm52.resident_sparse_mla.$(CUDA_ARCH).max_us_$(MAX_STAGE_MICROSECONDS).v1

.PHONY: archive cold_archive validate publish package clean require_sm121 require_latency_limit

archive: require_sm121 $(MODULE_ARCHIVE)

require_sm121:
	@test "$(CUDA_ARCH)" = "sm_121" || \
		{ echo "this firmware identity requires CUDA_ARCH=sm_121" >&2; exit 2; }

require_latency_limit:
	@test "$(MAX_STAGE_MICROSECONDS)" != "0" || \
		{ echo "set MAX_STAGE_MICROSECONDS to the qualified stage limit" >&2; exit 2; }

$(BUILD_DIRECTORY):
	mkdir -p $(BUILD_DIRECTORY)

$(HOST_OBJECT): source/spark_glm52_resident_sparse_mla_module.c | $(BUILD_DIRECTORY)
	$(CC) -I$(REPOSITORY_ROOT)/include -Iinclude -Isource $(CFLAGS) \
		-DNDEBUG -fPIC -fvisibility=hidden -MMD -MP -c $< -o $@

$(CUDA_OBJECT): source/spark_glm52_resident_sparse_mla_cuda.cu | $(BUILD_DIRECTORY)
	$(NVCC) $(NVCCFLAGS) -I$(REPOSITORY_ROOT)/include -Iinclude -Isource \
		-Xcompiler -fPIC -Xcompiler -fvisibility=hidden -MMD -MP -c $< -o $@

$(MODULE_ARCHIVE): $(HOST_OBJECT) $(CUDA_OBJECT)
	rm -f $@
	$(AR) rcs $@ $^

cold_archive: require_sm121
	rm -rf $(BUILD_DIRECTORY)
	$(MAKE) archive

validate: require_sm121 require_latency_limit cold_archive
	NVCC=$(NVCC) CUDA_ARCH=$(CUDA_ARCH) \
		validation/validate_glm52_resident_sparse_mla_cuda.sh \
		$(MAX_STAGE_MICROSECONDS) $(MODULE_ARCHIVE)

publish: require_sm121 require_latency_limit cold_archive
	$(MAKE) -C $(REPOSITORY_ROOT) build/sparkpipe_module_publish
	NVCC=$(NVCC) CUDA_ARCH=$(CUDA_ARCH) \
		$(REPOSITORY_ROOT)/build/sparkpipe_module_publish \
		--library $(MODULE_LIBRARY_ROOT) \
		--module spark.glm52.resident_sparse_mla.bf16.h64.d512.r64.k2048.b64.rope_adjacent.v1 \
		--target cuda.sm121.glm52.resident_sparse_mla.bf16 \
		--link-unit $(MODULE_ARCHIVE) \
		--recipe $(VALIDATION_RECIPE) \
		--initialize SparkGlm52ResidentSparseMlaInitialize \
		--execute SparkGlm52ResidentSparseMlaExecute \
		--admit SparkGlm52ResidentSparseMlaAdmit \
		--snapshot SparkGlm52ResidentSparseMlaSnapshot \
		--destroy SparkGlm52ResidentSparseMlaDestroy \
		--validator validation/validate_glm52_resident_sparse_mla_cuda.sh \
		--validator-arg $(MAX_STAGE_MICROSECONDS)

package: publish
	$(MAKE) -C $(REPOSITORY_ROOT) build/sparkpipe_model_compile
	rm -rf $(PACKAGE_DIRECTORY)
	$(REPOSITORY_ROOT)/build/sparkpipe_model_compile \
		--model $(REPOSITORY_ROOT)/examples/model_descriptions/glm52_resident_sparse_mla_firmware.json \
		--library $(MODULE_LIBRARY_ROOT) \
		--output $(PACKAGE_DIRECTORY) \
		--include $(REPOSITORY_ROOT)/include \
		--cc cc \
		--cc-arg -L$(CUDA_HOME)/lib64 \
		--cc-arg -lcudart \
		--cc-arg -lstdc++

clean:
	rm -rf $(BUILD_DIRECTORY)

-include $(HOST_DEPENDENCY) $(CUDA_DEPENDENCY)
