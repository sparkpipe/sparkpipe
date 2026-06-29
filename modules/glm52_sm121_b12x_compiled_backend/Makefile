REPOSITORY_ROOT := $(abspath ../..)
NVCC ?= nvcc
AR ?= ar
CUDA_ARCH ?= sm_121a
NVCCFLAGS ?= -std=c++17 -O3 -Xcompiler=-fPIC -Xcompiler=-fvisibility=hidden
BUILD_DIRECTORY := $(REPOSITORY_ROOT)/build/modules/glm52_sm121_b12x_compiled_backend
BACKEND_OBJECT := $(BUILD_DIRECTORY)/spark_flashinfer_b12x_compiled_moe_backend.o
BACKEND_ARCHIVE := $(BUILD_DIRECTORY)/libglm52_sm121_b12x_compiled_backend.a
GENERATED_DIRECTORY ?= $(REPOSITORY_ROOT)/build/glm52_b12x_aot/generated
GENERATED_FLAGS := $(GENERATED_DIRECTORY)/tvm_ffi_flags.mk
GENERATED_SOURCE := $(GENERATED_DIRECTORY)/spark_glm52_sm121_b12x_generated_kernel_table.cu
GENERATED_OBJECT := $(BUILD_DIRECTORY)/spark_glm52_sm121_b12x_generated_kernel_table.o
GENERATED_ARCHIVE := $(BUILD_DIRECTORY)/libglm52_sm121_b12x_generated_kernel_table.a
GENERATED_EXPORTED_OBJECTS := $(wildcard $(GENERATED_DIRECTORY)/objects/*.o)

-include $(GENERATED_FLAGS)

.PHONY: archive generated_archive clean require_generated_source

archive: $(BACKEND_ARCHIVE)

generated_archive: require_generated_source $(GENERATED_ARCHIVE)

$(BUILD_DIRECTORY):
	mkdir -p $(BUILD_DIRECTORY)

$(BACKEND_OBJECT): source/spark_flashinfer_b12x_compiled_moe_backend.cu include/sparkpipe/spark_glm52_sm121_b12x_generated_kernel_table.h | $(BUILD_DIRECTORY)
	$(NVCC) -arch=$(CUDA_ARCH) $(NVCCFLAGS) -I$(REPOSITORY_ROOT)/include -I../glm52_sm121_flashinfer_b12x_moe/include -Iinclude -c $< -o $@

$(BACKEND_ARCHIVE): $(BACKEND_OBJECT)
	rm -f $@
	$(AR) rcs $@ $^

require_generated_source:
	@if [ ! -f "$(GENERATED_SOURCE)" ]; then \
		echo "missing generated B12x kernel table: $(GENERATED_SOURCE)" >&2; \
		echo "run tools/glm52_b12x_aot_compile.py on spark1 first" >&2; \
		exit 1; \
	fi
	@if [ ! -f "$(GENERATED_FLAGS)" ]; then \
		echo "missing generated TVM FFI flags: $(GENERATED_FLAGS)" >&2; \
		echo "run tools/glm52_b12x_aot_compile.py on spark1 first" >&2; \
		exit 1; \
	fi
	@if [ -z "$(GENERATED_EXPORTED_OBJECTS)" ]; then \
		echo "missing exported FlashInfer B12x TVM FFI objects under $(GENERATED_DIRECTORY)/objects" >&2; \
		echo "run tools/glm52_b12x_aot_compile.py on spark1 first" >&2; \
		exit 1; \
	fi

$(GENERATED_OBJECT): $(GENERATED_SOURCE) $(GENERATED_FLAGS) | $(BUILD_DIRECTORY)
	$(NVCC) -arch=$(CUDA_ARCH) $(NVCCFLAGS) $(TVM_FFI_CFLAGS) -I$(REPOSITORY_ROOT)/include -I../glm52_sm121_flashinfer_b12x_moe/include -Iinclude -c $(GENERATED_SOURCE) -o $@

$(GENERATED_ARCHIVE): $(GENERATED_OBJECT) $(GENERATED_EXPORTED_OBJECTS)
	rm -f $@
	$(AR) rcs $@ $^

clean:
	rm -rf $(BUILD_DIRECTORY)
