CC ?= cc
AR ?= ar
NVCC ?= nvcc
CFLAGS ?= -std=c11 -Wall -Wextra -Werror -O3 -g
CPPFLAGS ?= -Iinclude -Isrc
LDFLAGS ?=
LDLIBS ?= -ldl
CUDA_ARCH ?= sm_121
NVCCFLAGS ?= -O3 --use_fast_math -arch=$(CUDA_ARCH)

COMMON_SOURCES := \
    src/spark_status.c \
    src/spark_filesystem.c

COMPILER_SOURCES := \
    src/spark_sha256.c \
    src/spark_json.c \
    src/spark_model_description.c \
    src/spark_module_library.c \
    src/spark_driver_compiler.c

RUNTIME_SOURCES := \
    src/spark_driver_loader.c \
    src/spark_orchestrator.c

COMMON_OBJECTS := $(patsubst src/%.c,build/%.o,$(COMMON_SOURCES))
COMPILER_OBJECTS := $(patsubst src/%.c,build/%.o,$(COMPILER_SOURCES))
RUNTIME_OBJECTS := $(patsubst src/%.c,build/%.o,$(RUNTIME_SOURCES))
COMMON_LIBRARY := build/libsparkpipe_common.a
COMPILER_LIBRARY := build/libsparkpipe_compiler.a
RUNTIME_LIBRARY := build/libsparkpipe_runtime.a
LIBRARIES := $(COMMON_LIBRARY) $(COMPILER_LIBRARY) $(RUNTIME_LIBRARY)

TOOL_NAMES := \
    sparkpipe_module_publish \
    sparkpipe_model_compile \
    sparkpipe_driver_inspect

TOOL_BINARIES := $(addprefix build/,$(TOOL_NAMES))

TEST_NAMES := \
    test_json \
    test_model_description \
    test_module_library \
    test_driver_compiler \
    test_orchestrator \
    test_glm52_resident_sparse_mla_firmware \
    test_glm52_resident_decode_stage_firmware

TEST_BINARIES := $(addprefix build/,$(TEST_NAMES))
TEST_SUPPORT_OBJECT := build/test_support.o
TEST_MODULE_OBJECTS := \
    build/test_modules/module_add_one.o \
    build/test_modules/module_add_two.o \
    build/test_modules/module_double.o \
    build/test_modules/module_affine_entry.o \
    build/test_modules/module_affine_helper.o
TEST_MODULE_ARCHIVES := \
    build/test_modules/module_affine.a
TEST_MODULE_LINK_UNITS := $(TEST_MODULE_OBJECTS) $(TEST_MODULE_ARCHIVES)
TEST_VALIDATOR := build/test_module_validator
GLM52_RESIDENT_SPARSE_MLA_TEST_DIRECTORY := \
    build/glm52_resident_sparse_mla_test
GLM52_RESIDENT_SPARSE_MLA_TEST_OBJECTS := \
    $(GLM52_RESIDENT_SPARSE_MLA_TEST_DIRECTORY)/spark_glm52_resident_sparse_mla_module.o \
    $(GLM52_RESIDENT_SPARSE_MLA_TEST_DIRECTORY)/glm52_resident_sparse_mla_fake_backend.o
GLM52_RESIDENT_SPARSE_MLA_TEST_ARCHIVE := \
    $(GLM52_RESIDENT_SPARSE_MLA_TEST_DIRECTORY)/libglm52_resident_sparse_mla_test.a
GLM52_RESIDENT_DECODE_STAGE_TEST_DIRECTORY := \
    build/glm52_resident_decode_stage_test
GLM52_RESIDENT_DECODE_STAGE_TEST_OBJECTS := \
    $(GLM52_RESIDENT_DECODE_STAGE_TEST_DIRECTORY)/spark_glm52_resident_decode_stage_module.o \
    $(GLM52_RESIDENT_DECODE_STAGE_TEST_DIRECTORY)/glm52_resident_decode_stage_fake_backend.o
GLM52_RESIDENT_DECODE_STAGE_TEST_ARCHIVE := \
    $(GLM52_RESIDENT_DECODE_STAGE_TEST_DIRECTORY)/libglm52_resident_decode_stage_test.a

.PHONY: all clean test tools demo cuda_dummy \
    cuda_glm52_resident_sparse_mla \
    cuda_glm52_resident_sparse_mla_publish \
    glm52_resident_sparse_mla_firmware_package \
    cuda_glm52_resident_decode_stage \
    cuda_glm52_resident_decode_stage_publish \
    glm52_resident_decode_stage_firmware_package \
    tree_summary

all: $(LIBRARIES) tools

tools: $(TOOL_BINARIES)

build:
	mkdir -p build

build/test_modules:
	mkdir -p build/test_modules

$(GLM52_RESIDENT_SPARSE_MLA_TEST_DIRECTORY):
	mkdir -p $(GLM52_RESIDENT_SPARSE_MLA_TEST_DIRECTORY)

$(GLM52_RESIDENT_DECODE_STAGE_TEST_DIRECTORY):
	mkdir -p $(GLM52_RESIDENT_DECODE_STAGE_TEST_DIRECTORY)

build/%.o: src/%.c | build
	$(CC) $(CPPFLAGS) $(CFLAGS) -fPIC -MMD -MP -c $< -o $@

$(COMMON_LIBRARY): $(COMMON_OBJECTS)
	$(AR) rcs $@ $^

$(COMPILER_LIBRARY): $(COMPILER_OBJECTS)
	$(AR) rcs $@ $^

$(RUNTIME_LIBRARY): $(RUNTIME_OBJECTS)
	$(AR) rcs $@ $^

build/sparkpipe_module_publish: tools/sparkpipe_module_publish.c $(COMPILER_LIBRARY) $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(COMPILER_LIBRARY) $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/sparkpipe_model_compile: tools/sparkpipe_model_compile.c $(COMPILER_LIBRARY) $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(COMPILER_LIBRARY) $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/sparkpipe_driver_inspect: tools/sparkpipe_driver_inspect.c $(RUNTIME_LIBRARY) $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(RUNTIME_LIBRARY) $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

$(TEST_SUPPORT_OBJECT): tests/test_support.c tests/test_support.h $(COMPILER_LIBRARY) $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) -c tests/test_support.c -o $@

build/test_modules/module_add_one.o: tests/fixtures/module_add_one.c | build/test_modules
	$(CC) -Iinclude $(CFLAGS) -fPIC -c $< -o $@

build/test_modules/module_add_two.o: tests/fixtures/module_add_two.c | build/test_modules
	$(CC) -Iinclude $(CFLAGS) -fPIC -c $< -o $@

build/test_modules/module_double.o: tests/fixtures/module_double.c | build/test_modules
	$(CC) -Iinclude $(CFLAGS) -fPIC -c $< -o $@

build/test_modules/module_affine_entry.o: tests/fixtures/module_affine_entry.c | build/test_modules
	$(CC) -Iinclude $(CFLAGS) -fPIC -c $< -o $@

build/test_modules/module_affine_helper.o: tests/fixtures/module_affine_helper.c | build/test_modules
	$(CC) -Iinclude $(CFLAGS) -fPIC -fvisibility=hidden -c $< -o $@

build/test_modules/module_affine.a: build/test_modules/module_affine_entry.o build/test_modules/module_affine_helper.o
	$(AR) rcs $@ $^

$(TEST_VALIDATOR): tests/fixtures/module_validator.c | build
	$(CC) $(CFLAGS) $< -o $@


$(GLM52_RESIDENT_SPARSE_MLA_TEST_DIRECTORY)/spark_glm52_resident_sparse_mla_module.o: modules/glm52_resident_sparse_mla/source/spark_glm52_resident_sparse_mla_module.c | $(GLM52_RESIDENT_SPARSE_MLA_TEST_DIRECTORY)
	$(CC) $(CPPFLAGS) -Imodules/glm52_resident_sparse_mla/include -Imodules/glm52_resident_sparse_mla/source $(CFLAGS) -fPIC -fvisibility=hidden -c $< -o $@

$(GLM52_RESIDENT_SPARSE_MLA_TEST_DIRECTORY)/glm52_resident_sparse_mla_fake_backend.o: tests/fixtures/glm52_resident_sparse_mla_fake_backend.c | $(GLM52_RESIDENT_SPARSE_MLA_TEST_DIRECTORY)
	$(CC) $(CPPFLAGS) -Itests/fixtures -Imodules/glm52_resident_sparse_mla/include -Imodules/glm52_resident_sparse_mla/source $(CFLAGS) -fPIC -fvisibility=hidden -c $< -o $@

$(GLM52_RESIDENT_SPARSE_MLA_TEST_ARCHIVE): $(GLM52_RESIDENT_SPARSE_MLA_TEST_OBJECTS)
	rm -f $@
	$(AR) rcs $@ $^

$(GLM52_RESIDENT_DECODE_STAGE_TEST_DIRECTORY)/spark_glm52_resident_decode_stage_module.o: modules/glm52_resident_decode_stage/source/spark_glm52_resident_decode_stage_module.c | $(GLM52_RESIDENT_DECODE_STAGE_TEST_DIRECTORY)
	$(CC) $(CPPFLAGS) -Imodules/glm52_resident_decode_stage/include -Imodules/glm52_resident_decode_stage/source $(CFLAGS) -fPIC -fvisibility=hidden -c $< -o $@

$(GLM52_RESIDENT_DECODE_STAGE_TEST_DIRECTORY)/glm52_resident_decode_stage_fake_backend.o: tests/fixtures/glm52_resident_decode_stage_fake_backend.c | $(GLM52_RESIDENT_DECODE_STAGE_TEST_DIRECTORY)
	$(CC) $(CPPFLAGS) -Itests/fixtures -Imodules/glm52_resident_decode_stage/include -Imodules/glm52_resident_decode_stage/source $(CFLAGS) -fPIC -fvisibility=hidden -c $< -o $@

$(GLM52_RESIDENT_DECODE_STAGE_TEST_ARCHIVE): $(GLM52_RESIDENT_DECODE_STAGE_TEST_OBJECTS)
	rm -f $@
	$(AR) rcs $@ $^

build/test_json: tests/test_json.c $(COMPILER_LIBRARY) $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(COMPILER_LIBRARY) $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_model_description: tests/test_model_description.c $(COMPILER_LIBRARY) $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(COMPILER_LIBRARY) $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_module_library: tests/test_module_library.c $(TEST_SUPPORT_OBJECT) $(TEST_MODULE_LINK_UNITS) $(TEST_VALIDATOR) $(COMPILER_LIBRARY) $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(TEST_SUPPORT_OBJECT) $(COMPILER_LIBRARY) $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_driver_compiler: tests/test_driver_compiler.c $(TEST_SUPPORT_OBJECT) $(TEST_MODULE_LINK_UNITS) $(TEST_VALIDATOR) $(COMPILER_LIBRARY) $(RUNTIME_LIBRARY) $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(TEST_SUPPORT_OBJECT) $(COMPILER_LIBRARY) $(RUNTIME_LIBRARY) $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_orchestrator: tests/test_orchestrator.c $(TEST_SUPPORT_OBJECT) $(TEST_MODULE_LINK_UNITS) $(TEST_VALIDATOR) $(COMPILER_LIBRARY) $(RUNTIME_LIBRARY) $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(TEST_SUPPORT_OBJECT) $(COMPILER_LIBRARY) $(RUNTIME_LIBRARY) $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_glm52_resident_sparse_mla_firmware: tests/test_glm52_resident_sparse_mla_firmware.c $(GLM52_RESIDENT_SPARSE_MLA_TEST_ARCHIVE) $(TEST_SUPPORT_OBJECT) $(TEST_VALIDATOR) $(COMPILER_LIBRARY) $(RUNTIME_LIBRARY) $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests -Itests/fixtures -Imodules/glm52_resident_sparse_mla/include $(CFLAGS) $< $(TEST_SUPPORT_OBJECT) $(COMPILER_LIBRARY) $(RUNTIME_LIBRARY) $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_glm52_resident_decode_stage_firmware: tests/test_glm52_resident_decode_stage_firmware.c $(GLM52_RESIDENT_DECODE_STAGE_TEST_ARCHIVE) $(TEST_SUPPORT_OBJECT) $(TEST_VALIDATOR) $(COMPILER_LIBRARY) $(RUNTIME_LIBRARY) $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests -Itests/fixtures -Imodules/glm52_resident_decode_stage/include $(CFLAGS) $< $(TEST_SUPPORT_OBJECT) $(COMPILER_LIBRARY) $(RUNTIME_LIBRARY) $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

test: $(TEST_BINARIES)
	@set -e; \
	for test_binary in $(TEST_BINARIES); do \
		echo "RUN $$test_binary"; \
		./$$test_binary; \
	done

demo: all $(TEST_MODULE_OBJECTS) $(TEST_VALIDATOR)
	rm -rf build/demo
	mkdir -p build/demo
	build/sparkpipe_module_publish --library build/demo/library --module spark.test.add_one.v1 --target host.cpu --link-unit build/test_modules/module_add_one.o --recipe test.module.validator.v1 --initialize SparkTestAddOneInitialize --execute SparkTestAddOneExecute --destroy SparkTestAddOneDestroy --validator build/test_module_validator --validator-arg build/demo/validator_count.txt
	build/sparkpipe_module_publish --library build/demo/library --module spark.test.add_one.v1 --target host.cpu --link-unit build/test_modules/module_add_one.o --recipe test.module.validator.v1 --initialize SparkTestAddOneInitialize --execute SparkTestAddOneExecute --destroy SparkTestAddOneDestroy --validator build/test_module_validator --validator-arg build/demo/validator_count.txt
	build/sparkpipe_module_publish --library build/demo/library --module spark.test.double.v1 --target host.cpu --link-unit build/test_modules/module_double.o --recipe test.module.validator.v1 --execute SparkTestDoubleExecute --validator build/test_module_validator --validator-arg build/demo/validator_count.txt
	build/sparkpipe_module_publish --library build/demo/library --module spark.test.double.v1 --target host.accelerator --link-unit build/test_modules/module_double.o --recipe test.module.validator.v1 --execute SparkTestDoubleExecute --validator build/test_module_validator --validator-arg build/demo/validator_count.txt
	test "$$(cat build/demo/validator_count.txt)" = "3"
	build/sparkpipe_model_compile --model examples/model_descriptions/firmware_demo.json --library build/demo/library --output build/demo/package --include include
	build/sparkpipe_driver_inspect build/demo/package/stages/stage_000/model_driver.so host.cpu
	build/sparkpipe_driver_inspect build/demo/package/stages/stage_001/model_driver.so host.accelerator

cuda_glm52_resident_sparse_mla:
	@if ! command -v $(NVCC) >/dev/null 2>&1; then \
		echo "cuda_glm52_resident_sparse_mla skipped: nvcc unavailable"; \
	else \
		$(MAKE) -C modules/glm52_resident_sparse_mla archive NVCC=$(NVCC) CUDA_ARCH=$(CUDA_ARCH); \
	fi

cuda_glm52_resident_sparse_mla_publish:
	@if ! command -v $(NVCC) >/dev/null 2>&1; then \
		echo "cuda_glm52_resident_sparse_mla_publish skipped: nvcc unavailable"; \
	else \
		$(MAKE) -C modules/glm52_resident_sparse_mla publish NVCC=$(NVCC) CUDA_ARCH=$(CUDA_ARCH) MAX_STAGE_MICROSECONDS=$(MAX_STAGE_MICROSECONDS); \
	fi

glm52_resident_sparse_mla_firmware_package:
	@if ! command -v $(NVCC) >/dev/null 2>&1; then \
		echo "glm52_resident_sparse_mla_firmware_package skipped: nvcc unavailable"; \
	else \
		$(MAKE) -C modules/glm52_resident_sparse_mla package NVCC=$(NVCC) CUDA_ARCH=$(CUDA_ARCH) MAX_STAGE_MICROSECONDS=$(MAX_STAGE_MICROSECONDS); \
	fi

cuda_glm52_resident_decode_stage:
	@if ! command -v $(NVCC) >/dev/null 2>&1; then \
		echo "cuda_glm52_resident_decode_stage skipped: nvcc unavailable"; \
	else \
		$(MAKE) -C modules/glm52_resident_decode_stage archive NVCC=$(NVCC) CUDA_ARCH=$(CUDA_ARCH); \
	fi

cuda_glm52_resident_decode_stage_publish:
	@if ! command -v $(NVCC) >/dev/null 2>&1; then \
		echo "cuda_glm52_resident_decode_stage_publish skipped: nvcc unavailable"; \
	else \
		$(MAKE) -C modules/glm52_resident_decode_stage publish NVCC=$(NVCC) CUDA_ARCH=$(CUDA_ARCH) MAX_STAGE_MICROSECONDS=$(MAX_STAGE_MICROSECONDS); \
	fi

glm52_resident_decode_stage_firmware_package:
	@if ! command -v $(NVCC) >/dev/null 2>&1; then \
		echo "glm52_resident_decode_stage_firmware_package skipped: nvcc unavailable"; \
	else \
		$(MAKE) -C modules/glm52_resident_decode_stage package NVCC=$(NVCC) CUDA_ARCH=$(CUDA_ARCH) MAX_STAGE_MICROSECONDS=$(MAX_STAGE_MICROSECONDS); \
	fi

cuda_dummy:
	@if ! command -v $(NVCC) >/dev/null 2>&1; then \
		echo "cuda_dummy skipped: nvcc unavailable"; \
	elif [ ! -f modules/cuda_candidates/source/device/spark_cuda_dummy_kernel.cu ]; then \
		echo "cuda_dummy skipped: preserved CUDA source library not present"; \
	else \
		mkdir -p build/cuda; \
		$(NVCC) $(NVCCFLAGS) -Imodules/cuda_candidates/include -c modules/cuda_candidates/source/device/spark_cuda_dummy_kernel.cu -o build/cuda/spark_cuda_dummy_kernel.o; \
	fi

tree_summary:
	@printf "public_headers="; find include/sparkpipe -type f | wc -l
	@printf "common_sources="; printf '%s\n' $(COMMON_SOURCES) | wc -l
	@printf "compiler_sources="; printf '%s\n' $(COMPILER_SOURCES) | wc -l
	@printf "runtime_sources="; printf '%s\n' $(RUNTIME_SOURCES) | wc -l
	@printf "test_executables="; printf '%s\n' $(TEST_NAMES) | wc -l

clean:
	rm -rf build

-include $(COMMON_OBJECTS:.o=.d) $(COMPILER_OBJECTS:.o=.d) $(RUNTIME_OBJECTS:.o=.d)
