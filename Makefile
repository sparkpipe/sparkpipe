CC ?= cc
AR ?= ar
NVCC ?= nvcc
CFLAGS ?= -std=c11 -Wall -Wextra -Werror -O3 -g
CPPFLAGS ?= -Iinclude -Isrc
LDFLAGS ?=
LDLIBS ?= -ldl
CUDA_ARCH ?= sm_121a
NVCCFLAGS ?= -O3 --use_fast_math -arch=$(CUDA_ARCH)
SPARKPIPE_B12X_AOT_ENV ?= $(HOME)/.config/sparkpipe/glm52_b12x_aot_env.sh
B12X_AOT_TOKENS ?= 1,2,4,8,16,32,64,96,128
B12X_AOT_WARMUP ?= 5
B12X_AOT_ITERATIONS ?= 20
B12X_AOT_OUTPUT_DIR ?= build/glm52_b12x_aot
B12X_AOT_BENCHMARK ?= --benchmark
B12X_MOE_PACK_OUTPUT_DIR ?= build/glm52_b12x_resident_moe
B12X_MOE_PACK_LAYERS ?= 3,4,5,6,7,8,9,10
B12X_MOE_PACK_REQUIRE_REUSE ?= 1
B12X_MOE_PACK_VERIFY_REUSED_SHA256 ?= 0
GLM52_VALIDATION_MODE ?= dense_to_layer3_routed
GLM52_VALIDATION_ACTIVE_SEQUENCE_COUNT ?= 1
GLM52_VALIDATION_FIRST_ROUTED_LAYER_INDEX ?= 3
GLM52_VALIDATION_ROUTED_CHAIN_LAYER_COUNT ?= 1
GLM52_PIPELINE_INPUT_HIDDEN_BF16 ?=
GLM52_PIPELINE_OUTPUT_HIDDEN_BF16 ?= build/glm52_pipeline_validation/output_hidden.bf16
GLM52_ENABLE_CUDA_GRAPH_REPLAY ?= 0
GLM52_STAGE_SWEEP_BUCKETS ?= 8,16,32,64
GLM52_STAGE_SWEEP_STAGE_ARGS ?=
GLM52_STAGE_SWEEP_MAX_STAGE_US ?= 1000000
GLM52_STAGE_SWEEP_OUTPUT_DIR ?= build/glm52_stage_bucket_sweep
B12X_ADAPTER_ARCHIVE := $(abspath build/modules/glm52_sm121_flashinfer_b12x_moe/libglm52_sm121_flashinfer_b12x_moe_adapter.a)
B12X_COMPILED_BACKEND_ARCHIVE := $(abspath build/modules/glm52_sm121_b12x_compiled_backend/libglm52_sm121_b12x_compiled_backend.a)
B12X_GENERATED_KERNEL_TABLE_ARCHIVE := $(abspath build/modules/glm52_sm121_b12x_compiled_backend/libglm52_sm121_b12x_generated_kernel_table.a)
B12X_RUNTIME_LINK_ARGS_FILE := $(abspath $(B12X_AOT_OUTPUT_DIR))/generated/runtime_link_args.txt

COMMON_SOURCES := \
    src/spark_status.c \
    src/spark_filesystem.c \
    src/spark_hidden_transport.c

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
    test_hidden_transport \
    test_model_description \
    test_module_library \
    test_driver_compiler \
    test_orchestrator \
    test_glm52_resident_decode_stage_firmware

TEST_BINARIES := $(addprefix build/,$(TEST_NAMES))
PYTHON_TESTS := \
    tests/test_b12x_scale_layout.py \
    tests/test_glm52_stage_bucket_sweep.py
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
TEST_MODULE_DEPENDENCIES := $(TEST_MODULE_OBJECTS:.o=.d)
TEST_VALIDATOR := build/test_module_validator
GLM52_RESIDENT_DECODE_STAGE_TEST_DIRECTORY := \
    build/glm52_resident_decode_stage_test
GLM52_RESIDENT_DECODE_STAGE_TEST_OBJECTS := \
    $(GLM52_RESIDENT_DECODE_STAGE_TEST_DIRECTORY)/spark_glm52_resident_decode_stage_module.o \
    $(GLM52_RESIDENT_DECODE_STAGE_TEST_DIRECTORY)/glm52_resident_decode_stage_fake_backend.o
GLM52_RESIDENT_DECODE_STAGE_TEST_DEPENDENCIES := \
    $(GLM52_RESIDENT_DECODE_STAGE_TEST_OBJECTS:.o=.d)
GLM52_RESIDENT_DECODE_STAGE_TEST_ARCHIVE := \
    $(GLM52_RESIDENT_DECODE_STAGE_TEST_DIRECTORY)/libglm52_resident_decode_stage_test.a

.PHONY: all clean test tools demo \
    cuda_glm52_resident_decode_stage \
    cuda_glm52_resident_decode_stage_publish \
    glm52_flashinfer_b12x_moe_adapter \
    glm52_b12x_prepare_spark_env \
    glm52_b12x_aot_compile \
    glm52_b12x_resident_pack \
    glm52_b12x_compiled_backend \
    glm52_required_cuda_link_args \
    glm52_stage_bucket_sweep \
    glm52_resident_decode_stage_firmware_package \
    tree_summary

all: $(LIBRARIES) tools

tools: $(TOOL_BINARIES)

build:
	mkdir -p build

build/test_modules:
	mkdir -p build/test_modules

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
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) -MMD -MP -c tests/test_support.c -o $@

build/test_modules/module_add_one.o: tests/fixtures/module_add_one.c | build/test_modules
	$(CC) -Iinclude $(CFLAGS) -fPIC -MMD -MP -c $< -o $@

build/test_modules/module_add_two.o: tests/fixtures/module_add_two.c | build/test_modules
	$(CC) -Iinclude $(CFLAGS) -fPIC -MMD -MP -c $< -o $@

build/test_modules/module_double.o: tests/fixtures/module_double.c | build/test_modules
	$(CC) -Iinclude $(CFLAGS) -fPIC -MMD -MP -c $< -o $@

build/test_modules/module_affine_entry.o: tests/fixtures/module_affine_entry.c | build/test_modules
	$(CC) -Iinclude $(CFLAGS) -fPIC -MMD -MP -c $< -o $@

build/test_modules/module_affine_helper.o: tests/fixtures/module_affine_helper.c | build/test_modules
	$(CC) -Iinclude $(CFLAGS) -fPIC -fvisibility=hidden -MMD -MP -c $< -o $@

build/test_modules/module_affine.a: build/test_modules/module_affine_entry.o build/test_modules/module_affine_helper.o
	$(AR) rcs $@ $^

$(TEST_VALIDATOR): tests/fixtures/module_validator.c | build
	$(CC) $(CFLAGS) $< -o $@


$(GLM52_RESIDENT_DECODE_STAGE_TEST_DIRECTORY)/spark_glm52_resident_decode_stage_module.o: modules/glm52_resident_decode_stage/source/spark_glm52_resident_decode_stage_module.c modules/glm52_resident_decode_stage/include/sparkpipe/spark_glm52_resident_decode_stage_firmware.h modules/glm52_resident_decode_stage/source/spark_glm52_resident_decode_stage_backend.h | $(GLM52_RESIDENT_DECODE_STAGE_TEST_DIRECTORY)
	$(CC) $(CPPFLAGS) -Imodules/glm52_resident_decode_stage/include -Imodules/glm52_resident_decode_stage/source $(CFLAGS) -fPIC -fvisibility=hidden -MMD -MP -c $< -o $@

$(GLM52_RESIDENT_DECODE_STAGE_TEST_DIRECTORY)/glm52_resident_decode_stage_fake_backend.o: tests/fixtures/glm52_resident_decode_stage_fake_backend.c tests/fixtures/glm52_resident_decode_stage_fake_backend.h modules/glm52_resident_decode_stage/include/sparkpipe/spark_glm52_resident_decode_stage_firmware.h modules/glm52_resident_decode_stage/source/spark_glm52_resident_decode_stage_backend.h | $(GLM52_RESIDENT_DECODE_STAGE_TEST_DIRECTORY)
	$(CC) $(CPPFLAGS) -Itests/fixtures -Imodules/glm52_resident_decode_stage/include -Imodules/glm52_resident_decode_stage/source $(CFLAGS) -fPIC -fvisibility=hidden -MMD -MP -c $< -o $@

$(GLM52_RESIDENT_DECODE_STAGE_TEST_ARCHIVE): $(GLM52_RESIDENT_DECODE_STAGE_TEST_OBJECTS)
	rm -f $@
	$(AR) rcs $@ $^

build/test_json: tests/test_json.c $(COMPILER_LIBRARY) $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(COMPILER_LIBRARY) $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_hidden_transport: tests/test_hidden_transport.c $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_model_description: tests/test_model_description.c $(COMPILER_LIBRARY) $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(COMPILER_LIBRARY) $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_module_library: tests/test_module_library.c $(TEST_SUPPORT_OBJECT) $(TEST_MODULE_LINK_UNITS) $(TEST_VALIDATOR) $(COMPILER_LIBRARY) $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(TEST_SUPPORT_OBJECT) $(COMPILER_LIBRARY) $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_driver_compiler: tests/test_driver_compiler.c $(TEST_SUPPORT_OBJECT) $(TEST_MODULE_LINK_UNITS) $(TEST_VALIDATOR) $(COMPILER_LIBRARY) $(RUNTIME_LIBRARY) $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(TEST_SUPPORT_OBJECT) $(COMPILER_LIBRARY) $(RUNTIME_LIBRARY) $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_orchestrator: tests/test_orchestrator.c $(TEST_SUPPORT_OBJECT) $(TEST_MODULE_LINK_UNITS) $(TEST_VALIDATOR) $(COMPILER_LIBRARY) $(RUNTIME_LIBRARY) $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) $< $(TEST_SUPPORT_OBJECT) $(COMPILER_LIBRARY) $(RUNTIME_LIBRARY) $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

build/test_glm52_resident_decode_stage_firmware: tests/test_glm52_resident_decode_stage_firmware.c modules/glm52_resident_decode_stage/include/sparkpipe/spark_glm52_resident_decode_stage_firmware.h tests/fixtures/glm52_resident_decode_stage_fake_backend.h $(GLM52_RESIDENT_DECODE_STAGE_TEST_ARCHIVE) $(TEST_SUPPORT_OBJECT) $(TEST_VALIDATOR) $(COMPILER_LIBRARY) $(RUNTIME_LIBRARY) $(COMMON_LIBRARY)
	$(CC) $(CPPFLAGS) -Itests -Itests/fixtures -Imodules/glm52_resident_decode_stage/include $(CFLAGS) $< $(TEST_SUPPORT_OBJECT) $(COMPILER_LIBRARY) $(RUNTIME_LIBRARY) $(COMMON_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@

test: $(TEST_BINARIES)
	@set -e; \
	for test_binary in $(TEST_BINARIES); do \
		echo "RUN $$test_binary"; \
		./$$test_binary; \
	done; \
	for python_test in $(PYTHON_TESTS); do \
		echo "RUN $$python_test"; \
		python3 $$python_test; \
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

cuda_glm52_resident_decode_stage:
	@if ! command -v $(NVCC) >/dev/null 2>&1; then \
		echo "cuda_glm52_resident_decode_stage skipped: nvcc unavailable"; \
	else \
		$(MAKE) -C modules/glm52_resident_decode_stage archive NVCC=$(NVCC) CUDA_ARCH=sm_121a; \
	fi

cuda_glm52_resident_decode_stage_publish:
	@if ! command -v $(NVCC) >/dev/null 2>&1; then \
		echo "cuda_glm52_resident_decode_stage_publish skipped: nvcc unavailable"; \
	else \
		$(MAKE) -C modules/glm52_resident_decode_stage publish NVCC=$(NVCC) CUDA_ARCH=sm_121a MAX_STAGE_MICROSECONDS=$(MAX_STAGE_MICROSECONDS) REQUIRED_CUDA_CC_ARGS='$(REQUIRED_CUDA_CC_ARGS)' GLM52_REQUIRED_CUDA_LINK_ARGS='$(GLM52_REQUIRED_CUDA_LINK_ARGS)'; \
	fi

glm52_flashinfer_b12x_moe_adapter:
	$(MAKE) -C modules/glm52_sm121_flashinfer_b12x_moe archive

glm52_b12x_prepare_spark_env:
	tools/glm52_b12x_prepare_spark_env.sh

glm52_b12x_aot_compile:
	@test -f "$(SPARKPIPE_B12X_AOT_ENV)" || \
		{ echo "missing $(SPARKPIPE_B12X_AOT_ENV); run make glm52_b12x_prepare_spark_env first" >&2; exit 2; }
	. "$(SPARKPIPE_B12X_AOT_ENV)" && \
		"$$SPARKPIPE_B12X_AOT_PYTHON" ./tools/glm52_b12x_aot_compile.py \
		--tokens "$(B12X_AOT_TOKENS)" \
		--warmup "$(B12X_AOT_WARMUP)" \
		--iterations "$(B12X_AOT_ITERATIONS)" \
		$(B12X_AOT_BENCHMARK) \
		--output-dir "$(B12X_AOT_OUTPUT_DIR)"

glm52_b12x_resident_pack:
	@test -n "$(GLM52_MODEL_DIR)" || \
		{ echo "set GLM52_MODEL_DIR to the live GLM artifact directory" >&2; exit 2; }
	@test -s "$(B12X_AOT_OUTPUT_DIR)/generated/aot_manifest.json" || \
		{ echo "missing $(B12X_AOT_OUTPUT_DIR)/generated/aot_manifest.json; run make glm52_b12x_aot_compile first" >&2; exit 2; }
	@test -f "$(SPARKPIPE_B12X_AOT_ENV)" || \
		{ echo "missing $(SPARKPIPE_B12X_AOT_ENV); run make glm52_b12x_prepare_spark_env first" >&2; exit 2; }
	. "$(SPARKPIPE_B12X_AOT_ENV)" && \
		"$$SPARKPIPE_B12X_AOT_PYTHON" ./tools/glm52_b12x_resident_pack.py \
			--model-dir "$(GLM52_MODEL_DIR)" \
				--aot-manifest "$(B12X_AOT_OUTPUT_DIR)/generated/aot_manifest.json" \
				--layers "$(B12X_MOE_PACK_LAYERS)" \
				--output-dir "$(B12X_MOE_PACK_OUTPUT_DIR)" \
				--reuse-valid \
				$(if $(filter 1,$(B12X_MOE_PACK_VERIFY_REUSED_SHA256)),--verify-reused-sha256,)

glm52_b12x_compiled_backend:
	$(MAKE) -C modules/glm52_sm121_b12x_compiled_backend archive NVCC=$(NVCC) CUDA_ARCH=sm_121a
	$(MAKE) -C modules/glm52_sm121_b12x_compiled_backend generated_archive NVCC=$(NVCC) CUDA_ARCH=sm_121a GENERATED_DIRECTORY=$(abspath build/glm52_b12x_aot/generated)

glm52_required_cuda_link_args: glm52_flashinfer_b12x_moe_adapter glm52_b12x_compiled_backend
	@test -s "$(B12X_RUNTIME_LINK_ARGS_FILE)" || \
		{ echo "missing $(B12X_RUNTIME_LINK_ARGS_FILE); run make glm52_b12x_aot_compile first" >&2; exit 2; }
	@printf "%s %s %s " "$(B12X_ADAPTER_ARCHIVE)" "$(B12X_COMPILED_BACKEND_ARCHIVE)" "$(B12X_GENERATED_KERNEL_TABLE_ARCHIVE)"
	@cat "$(B12X_RUNTIME_LINK_ARGS_FILE)"

glm52_stage_bucket_sweep:
	@test -n "$(GLM52_PIPELINE_INPUT_HIDDEN_BF16)" || \
		{ echo "set GLM52_PIPELINE_INPUT_HIDDEN_BF16 to a one-vector or B-vector hidden BF16 file" >&2; exit 2; }
	@test -s "$(GLM52_PIPELINE_INPUT_HIDDEN_BF16)" || \
		{ echo "missing GLM52_PIPELINE_INPUT_HIDDEN_BF16: $(GLM52_PIPELINE_INPUT_HIDDEN_BF16)" >&2; exit 2; }
	python3 ./tools/glm52_stage_bucket_sweep.py \
		--buckets "$(GLM52_STAGE_SWEEP_BUCKETS)" \
		--input-hidden "$(GLM52_PIPELINE_INPUT_HIDDEN_BF16)" \
		--output-dir "$(GLM52_STAGE_SWEEP_OUTPUT_DIR)" \
		--max-stage-us "$(GLM52_STAGE_SWEEP_MAX_STAGE_US)" \
		--model-dir "$(GLM52_MODEL_DIR)" \
		--nvcc "$(NVCC)" \
		--aot-env "$(SPARKPIPE_B12X_AOT_ENV)" \
		--aot-output-dir "$(B12X_AOT_OUTPUT_DIR)" \
		--b12x-moe-pack-dir "$(B12X_MOE_PACK_OUTPUT_DIR)" \
		--b12x-moe-pack-layers "$(B12X_MOE_PACK_LAYERS)" \
		$(if $(filter 1,$(B12X_MOE_PACK_REQUIRE_REUSE)),--require-pack-reuse,--allow-pack-build) \
		$(if $(filter 1,$(B12X_MOE_PACK_VERIFY_REUSED_SHA256)),--verify-reused-sha256,) \
		$(GLM52_STAGE_SWEEP_STAGE_ARGS) \
		$(if $(filter 1,$(GLM52_ENABLE_CUDA_GRAPH_REPLAY)),--graph,)

glm52_resident_decode_stage_firmware_package: glm52_flashinfer_b12x_moe_adapter glm52_b12x_compiled_backend
	@command -v $(NVCC) >/dev/null 2>&1 || \
		{ echo "missing nvcc for required GLM52 SM121 package build" >&2; exit 2; }
	@test -f "$(SPARKPIPE_B12X_AOT_ENV)" || \
		{ echo "missing $(SPARKPIPE_B12X_AOT_ENV); run make glm52_b12x_prepare_spark_env first" >&2; exit 2; }
	@test -s "$(B12X_RUNTIME_LINK_ARGS_FILE)" || \
		{ echo "missing $(B12X_RUNTIME_LINK_ARGS_FILE); run make glm52_b12x_aot_compile first" >&2; exit 2; }
	. "$(SPARKPIPE_B12X_AOT_ENV)" && \
		if [ -n "$(GLM52_REQUIRED_CUDA_LINK_ARGS)" ]; then \
			required_cuda_link_args='$(GLM52_REQUIRED_CUDA_LINK_ARGS)'; \
		else \
			required_cuda_link_args='$(B12X_ADAPTER_ARCHIVE) $(B12X_COMPILED_BACKEND_ARCHIVE) $(B12X_GENERATED_KERNEL_TABLE_ARCHIVE) '"$$(cat "$(B12X_RUNTIME_LINK_ARGS_FILE)")"; \
		fi; \
		$(MAKE) -C modules/glm52_resident_decode_stage package NVCC=$(NVCC) CUDA_ARCH=sm_121a MAX_STAGE_MICROSECONDS=$(MAX_STAGE_MICROSECONDS) REQUIRED_CUDA_CC_ARGS='$(REQUIRED_CUDA_CC_ARGS)' GLM52_REQUIRED_CUDA_LINK_ARGS="$$required_cuda_link_args" B12X_MOE_PACK_DIR='$(abspath $(B12X_MOE_PACK_OUTPUT_DIR))' B12X_MOE_PACK_LAYERS='$(B12X_MOE_PACK_LAYERS)' B12X_MOE_PACK_REQUIRE_REUSE='$(B12X_MOE_PACK_REQUIRE_REUSE)' B12X_MOE_PACK_VERIFY_REUSED_SHA256='$(B12X_MOE_PACK_VERIFY_REUSED_SHA256)' GLM52_VALIDATION_MODE='$(GLM52_VALIDATION_MODE)' GLM52_VALIDATION_ACTIVE_SEQUENCE_COUNT='$(GLM52_VALIDATION_ACTIVE_SEQUENCE_COUNT)' GLM52_VALIDATION_FIRST_ROUTED_LAYER_INDEX='$(GLM52_VALIDATION_FIRST_ROUTED_LAYER_INDEX)' GLM52_VALIDATION_ROUTED_CHAIN_LAYER_COUNT='$(GLM52_VALIDATION_ROUTED_CHAIN_LAYER_COUNT)' GLM52_PIPELINE_INPUT_HIDDEN_BF16='$(GLM52_PIPELINE_INPUT_HIDDEN_BF16)' GLM52_PIPELINE_OUTPUT_HIDDEN_BF16='$(GLM52_PIPELINE_OUTPUT_HIDDEN_BF16)' GLM52_ENABLE_CUDA_GRAPH_REPLAY='$(GLM52_ENABLE_CUDA_GRAPH_REPLAY)' B12X_PACK_PYTHON="$$SPARKPIPE_B12X_AOT_PYTHON" AOT_MANIFEST='$(abspath $(B12X_AOT_OUTPUT_DIR))/generated/aot_manifest.json'

tree_summary:
	@printf "public_headers="; find include/sparkpipe -type f | wc -l
	@printf "common_sources="; printf '%s\n' $(COMMON_SOURCES) | wc -l
	@printf "compiler_sources="; printf '%s\n' $(COMPILER_SOURCES) | wc -l
	@printf "runtime_sources="; printf '%s\n' $(RUNTIME_SOURCES) | wc -l
	@printf "test_executables="; printf '%s\n' $(TEST_NAMES) | wc -l

clean:
	rm -rf build

-include $(COMMON_OBJECTS:.o=.d) $(COMPILER_OBJECTS:.o=.d) \
    $(RUNTIME_OBJECTS:.o=.d) $(TEST_SUPPORT_OBJECT:.o=.d) \
    $(TEST_MODULE_DEPENDENCIES) \
    $(GLM52_RESIDENT_DECODE_STAGE_TEST_DEPENDENCIES)
