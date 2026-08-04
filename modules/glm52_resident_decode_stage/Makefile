# Keep target and prerequisite names relative to this module.  GNU Make treats
# spaces in absolute path names as target separators, so expanding abspath here
# breaks release builds from otherwise valid workspaces whose names contain a
# space.
REPOSITORY_ROOT := ../..
CC ?= cc
AR ?= ar
NVCC ?= nvcc
CUDA_ARCH ?= sm_121a
CUDA_COMPUTE_ARCH ?= $(subst sm_,compute_,$(CUDA_ARCH))
CFLAGS ?= -std=c11 -Wall -Wextra -Werror -O3
# -arch=$(CUDA_ARCH) silently drops the architecture-specific suffix (see
# tools/build.sh); spell the gencode pair out so sm_121a survives.
NVCCFLAGS ?= -std=c++17 -O3 --expt-relaxed-constexpr -lineinfo -gencode arch=$(CUDA_COMPUTE_ARCH),code=$(CUDA_ARCH)
MODULE_LIBRARY_ROOT ?= $(REPOSITORY_ROOT)/build/module_library
MAX_STAGE_MICROSECONDS ?= 0
ARTIFACT_CHECK_ARGS ?=
AOT_MANIFEST ?= $(REPOSITORY_ROOT)/build/glm52_b12x_aot/generated/aot_manifest.json
B12X_MOE_PACK_DIR ?= $(REPOSITORY_ROOT)/build/glm52_b12x_resident_moe
B12X_MOE_PACK_LAYERS ?= 3,4,5,6,7,8,9,10
B12X_MOE_PACK_REQUIRE_REUSE ?= 1
B12X_MOE_PACK_VERIFY_REUSED_SHA256 ?= 0
B12X_MOE_PACK_JOBS ?= 1
B12X_MOE_PACK_PACKAGE_MODE ?= hardlink
B12X_PACK_PYTHON ?= python3
GLM52_REQUIRE_B12X_RESIDENT_PACK ?= 1
GLM52_VALIDATION_INPUT_TOKEN_ID ?= 1000
GLM52_VALIDATION_MODE ?= dense_to_layer3_routed
GLM52_VALIDATION_ACTIVE_SEQUENCE_COUNT ?= 1
GLM52_VALIDATION_MAX_ACTIVE_SEQUENCE_COUNT ?= $(GLM52_VALIDATION_ACTIVE_SEQUENCE_COUNT)
GLM52_VALIDATION_EXACT_RING_BATCH_BUCKET ?= 0
GLM52_VALIDATION_EXPECTED_OUTPUT_HIDDEN_BF16 ?=
GLM52_EXACT_RING_ATTENTION_EXECUTION_MODE ?= 1
GLM52_VALIDATION_FIRST_ROUTED_LAYER_INDEX ?= 3
GLM52_VALIDATION_ROUTED_CHAIN_LAYER_COUNT ?= 1
GLM52_PIPELINE_INPUT_HIDDEN_BF16 ?=
GLM52_PIPELINE_OUTPUT_HIDDEN_BF16 ?= $(BUILD_DIRECTORY)/pipeline_hidden_output.bf16
GLM52_ENABLE_CUDA_GRAPH_REPLAY ?= 0
GLM52_ALLOW_REMOTE_MODEL_DIR ?= 0
GLM52_EXACT_RING_MODEL_QUANTIZATION ?= nvfp4
GLM52_FP8_MOE_PACK_DIR ?=
GLM52_W8LUT_MOE_PACK_DIR ?=
GLM52_STAGE_PACK_DIR ?=
ifeq ($(GLM52_REQUIRE_B12X_RESIDENT_PACK),1)
ifeq ($(GLM52_EXACT_RING_MODEL_QUANTIZATION),nvfp4)
GLM52_B12X_RESIDENT_PACK_PREREQ := b12x_resident_pack
endif
endif
BUILD_DIRECTORY := $(REPOSITORY_ROOT)/build/modules/glm52_resident_decode_stage
HOST_OBJECT := $(BUILD_DIRECTORY)/spark_glm52_resident_decode_stage_module.o
VALIDATION_OBJECT = $(BUILD_DIRECTORY)/spark_glm52_resident_decode_stage_glm52_validation.o
CUDA_OBJECT := $(BUILD_DIRECTORY)/spark_glm52_resident_decode_stage_cuda.o
FP8_MOE_PLAN_OBJECT := $(BUILD_DIRECTORY)/spark_glm52_resident_decode_stage_fp8_moe_plan.o
SERVING_ADAPTER_OBJECT := $(BUILD_DIRECTORY)/spark_glm52_resident_decode_stage_serving_adapter.o
MODULE_ARCHIVE := $(BUILD_DIRECTORY)/libglm52_resident_decode_stage.a
ARTIFACT_CHECK_TOOL := $(BUILD_DIRECTORY)/glm52_artifact_check
PACKAGE_DIRECTORY ?= $(REPOSITORY_ROOT)/build/packages/glm52_resident_decode_stage
CUDA_HOME ?= /usr/local/cuda
VALIDATION_RECIPE_VARIANT ?= $(GLM52_VALIDATION_MODE)
VALIDATION_RECIPE := glm52.resident_decode_stage.$(CUDA_ARCH).$(VALIDATION_RECIPE_VARIANT).b$(GLM52_VALIDATION_ACTIVE_SEQUENCE_COUNT).max_us_$(MAX_STAGE_MICROSECONDS).v1
REQUIRED_CUDA_CC_ARGS ?=
GLM52_REQUIRED_CUDA_LINK_ARGS ?=
B12X_ADAPTER_ARCHIVE := $(REPOSITORY_ROOT)/build/modules/glm52_sm121_flashinfer_b12x_moe/libglm52_sm121_flashinfer_b12x_moe_adapter.a
B12X_COMPILED_BACKEND_ARCHIVE := $(REPOSITORY_ROOT)/build/modules/glm52_sm121_b12x_compiled_backend/libglm52_sm121_b12x_compiled_backend.a
B12X_GENERATED_KERNEL_TABLE_ARCHIVE := $(REPOSITORY_ROOT)/build/modules/glm52_sm121_b12x_compiled_backend/libglm52_sm121_b12x_generated_kernel_table.a
GLM52_DSA_TOPK_LINK_ARGS := $(if $(GLM52_REQUIRED_CUDA_LINK_ARGS),$(GLM52_REQUIRED_CUDA_LINK_ARGS),$(B12X_ADAPTER_ARCHIVE) $(B12X_COMPILED_BACKEND_ARCHIVE) $(B12X_GENERATED_KERNEL_TABLE_ARCHIVE))
HOST_DEPENDENCY := $(HOST_OBJECT:.o=.d)
CUDA_DEPENDENCY := $(CUDA_OBJECT:.o=.d)
FP8_MOE_PLAN_DEPENDENCY := $(FP8_MOE_PLAN_OBJECT:.o=.d)
SERVING_ADAPTER_DEPENDENCY := $(SERVING_ADAPTER_OBJECT:.o=.d)

# BATCH VARIANTS. One source tree, N archives: every variant compiles the same
# five translation units as the default archive with exactly one extra flag,
# -DSPARK_BATCH_BUCKET=<n>, and every per-bucket constant that flag selects
# lives in spark_glm52_batch_tuning.h. DRY by construction: there is one
# recipe template and one bucket list, so a fix to a recipe reaches every
# variant and a source edit cannot land in one variant only.
#
# A bucket is a capacity ceiling - the b8 archive serves 1-8 rows - so the
# set {8, 64, 256, 1024} overlaps deliberately: B8 is the chat batch, B1024
# the stage planner's maximum and the winner of any B64-vs-B1024 tradeoff.
# Compile time is bounded by keeping the set at four and by emitting variants
# only for families with resident stage modules (glm52 here; k3 adopts the
# same contract when its module Makefile lands).
GLM52_BATCH_VARIANT_BUCKETS ?= 8 64 256 1024
GLM52_MODULE_ID_PREFIX := spark.glm52.resident_decode_stage.bf16.h6144.h64.d512.r64.k2048
GLM52_MODULE_ID_SUFFIX := rv256.mtp6.v1
GLM52_MODULE_ID := $(GLM52_MODULE_ID_PREFIX).b1024.$(GLM52_MODULE_ID_SUFFIX)
GLM52_BATCH_VARIANT_ARCHIVE = $(BUILD_DIRECTORY)/libglm52_resident_decode_stage_b$(1).a
GLM52_BATCH_VARIANT_ARCHIVES := $(foreach bucket,$(GLM52_BATCH_VARIANT_BUCKETS),$(call GLM52_BATCH_VARIANT_ARCHIVE,$(bucket)))

# all must be the FIRST target: the $(eval) below expands real object rules
# ahead of every other target in this file, and without an explicit default
# goal a bare `make` would build one b8 object and stop. all emits the whole
# variant set - the default archive plus one archive per bucket - because the
# variant IS the module now; an unbucketed-only build is a partial build.
all: archive variants

# $(1) is the bucket. The recipes mirror the default-archive rules flag for
# flag; the b-suffixed object names keep a variant build from ever linking a
# default-tuned object or vice versa.
define SPARK_GLM52_BATCH_VARIANT_RULES
$(BUILD_DIRECTORY)/spark_glm52_resident_decode_stage_module_b$(1).o: $(REPOSITORY_ROOT)/inference/stage/module.c | $(BUILD_DIRECTORY)
	$(CC) -I$(REPOSITORY_ROOT) -I$(REPOSITORY_ROOT)/include -I$(REPOSITORY_ROOT)/model-families/glm52/include -Iinclude -Isource $(CFLAGS) \
		-DNDEBUG -DSPARK_RESIDENT_DECODE_STAGE_REQUIRE_EXTERNAL_CUDA_MODULES -DSPARK_BATCH_BUCKET=$(1) \
		-fPIC -fvisibility=hidden -MMD -MP -c $$< -o $$@

$(BUILD_DIRECTORY)/spark_glm52_resident_decode_stage_glm52_validation_b$(1).o: source/spark_glm52_resident_decode_stage_validation.c | $(BUILD_DIRECTORY)
	$(CC) -I$(REPOSITORY_ROOT) -I$(REPOSITORY_ROOT)/include -I$(REPOSITORY_ROOT)/model-families/glm52/include -Iinclude -Isource $(CFLAGS) \
		-DNDEBUG -DSPARK_RESIDENT_DECODE_STAGE_REQUIRE_EXTERNAL_CUDA_MODULES -DSPARK_BATCH_BUCKET=$(1) \
		-fPIC -fvisibility=hidden -MMD -MP -c $$< -o $$@

$(BUILD_DIRECTORY)/spark_glm52_resident_decode_stage_cuda_b$(1).o: $(REPOSITORY_ROOT)/inference/stage/dispatch.cu | $(BUILD_DIRECTORY)
	$(NVCC) $(NVCCFLAGS) -I$(REPOSITORY_ROOT) -I$(REPOSITORY_ROOT)/include -I$(REPOSITORY_ROOT)/model-families/glm52/include -Iinclude -Isource \
		-DSPARK_BATCH_BUCKET=$(1) \
		-Xcompiler -fPIC -Xcompiler -fvisibility=hidden -MMD -MP -c $$< -o $$@

$(BUILD_DIRECTORY)/spark_glm52_resident_decode_stage_fp8_moe_plan_b$(1).o: source/spark_glm52_resident_decode_stage_fp8_moe_plan.cu | $(BUILD_DIRECTORY)
	$(NVCC) $(NVCCFLAGS) -I$(REPOSITORY_ROOT) -I$(REPOSITORY_ROOT)/include -I$(REPOSITORY_ROOT)/model-families/glm52/include -Iinclude -Isource \
		-DSPARK_BATCH_BUCKET=$(1) \
		-Xcompiler -fPIC -Xcompiler -fvisibility=hidden -MMD -MP -c $$< -o $$@

$(BUILD_DIRECTORY)/spark_glm52_resident_decode_stage_serving_adapter_b$(1).o: $(REPOSITORY_ROOT)/inference/stage/serving_adapter.cu | $(BUILD_DIRECTORY)
	$(NVCC) $(NVCCFLAGS) -I$(REPOSITORY_ROOT) -I$(REPOSITORY_ROOT)/include -I$(REPOSITORY_ROOT)/model-families/glm52/include -Iinclude -Isource \
		-DSPARK_BATCH_BUCKET=$(1) \
		-Xcompiler -fPIC -Xcompiler -fvisibility=hidden -MMD -MP -c $$< -o $$@

$(call GLM52_BATCH_VARIANT_ARCHIVE,$(1)): $(BUILD_DIRECTORY)/spark_glm52_resident_decode_stage_module_b$(1).o $(BUILD_DIRECTORY)/spark_glm52_resident_decode_stage_glm52_validation_b$(1).o $(BUILD_DIRECTORY)/spark_glm52_resident_decode_stage_cuda_b$(1).o $(BUILD_DIRECTORY)/spark_glm52_resident_decode_stage_fp8_moe_plan_b$(1).o $(BUILD_DIRECTORY)/spark_glm52_resident_decode_stage_serving_adapter_b$(1).o
	rm -f $$@
	$(AR) rcs $$@ $$^
endef
$(foreach bucket,$(GLM52_BATCH_VARIANT_BUCKETS),$(eval $(call SPARK_GLM52_BATCH_VARIANT_RULES,$(bucket))))
GLM52_BATCH_VARIANT_DEPENDENCIES := $(foreach bucket,$(GLM52_BATCH_VARIANT_BUCKETS),$(BUILD_DIRECTORY)/spark_glm52_resident_decode_stage_module_b$(bucket).d $(BUILD_DIRECTORY)/spark_glm52_resident_decode_stage_glm52_validation_b$(bucket).d $(BUILD_DIRECTORY)/spark_glm52_resident_decode_stage_cuda_b$(bucket).d $(BUILD_DIRECTORY)/spark_glm52_resident_decode_stage_fp8_moe_plan_b$(bucket).d $(BUILD_DIRECTORY)/spark_glm52_resident_decode_stage_serving_adapter_b$(bucket).d)

.PHONY: all archive cold_archive artifact_check b12x_resident_pack validate validate_dsa_topk_parity publish package clean require_sm121a require_latency_limit require_model_dir require_fast_model_dir require_validation_inputs prepare_validation_output variants cold_variants publish_variants

GLM52_VALIDATION_BASE_ENV := NVCC=$(NVCC) CUDA_ARCH=$(CUDA_ARCH) GLM52_REQUIRED_CUDA_LINK_ARGS="$(GLM52_REQUIRED_CUDA_LINK_ARGS)" GLM52_B12X_MOE_PACK_DIR="$(B12X_MOE_PACK_DIR)" GLM52_MODEL_DIR="$(GLM52_MODEL_DIR)" GLM52_ALLOW_REMOTE_MODEL_DIR="$(GLM52_ALLOW_REMOTE_MODEL_DIR)" GLM52_ROUTED_CHAIN_FIRST_LAYER_INDEX="$(GLM52_VALIDATION_FIRST_ROUTED_LAYER_INDEX)" GLM52_ROUTED_CHAIN_LAYER_COUNT="$(GLM52_VALIDATION_ROUTED_CHAIN_LAYER_COUNT)" GLM52_ENABLE_CUDA_GRAPH_REPLAY="$(GLM52_ENABLE_CUDA_GRAPH_REPLAY)" GLM52_VALIDATION_ACTIVE_SEQUENCE_COUNT="$(GLM52_VALIDATION_ACTIVE_SEQUENCE_COUNT)" GLM52_VALIDATION_MAX_ACTIVE_SEQUENCE_COUNT="$(GLM52_VALIDATION_MAX_ACTIVE_SEQUENCE_COUNT)" GLM52_VALIDATION_EXACT_RING_BATCH_BUCKET="$(GLM52_VALIDATION_EXACT_RING_BATCH_BUCKET)" GLM52_VALIDATION_EXPECTED_OUTPUT_HIDDEN_BF16="$(GLM52_VALIDATION_EXPECTED_OUTPUT_HIDDEN_BF16)" GLM52_EXACT_RING_ATTENTION_EXECUTION_MODE="$(GLM52_EXACT_RING_ATTENTION_EXECUTION_MODE)" GLM52_EXACT_RING_MODEL_QUANTIZATION="$(GLM52_EXACT_RING_MODEL_QUANTIZATION)" GLM52_STAGE_PACK_DIR="$(GLM52_STAGE_PACK_DIR)" GLM52_FP8_MOE_PACK_DIR="$(GLM52_FP8_MOE_PACK_DIR)" GLM52_W8LUT_MOE_PACK_DIR="$(GLM52_W8LUT_MOE_PACK_DIR)"

ifeq ($(GLM52_VALIDATION_MODE),dense_to_layer3_routed)
GLM52_VALIDATION_ENV := $(GLM52_VALIDATION_BASE_ENV) GLM52_INPUT_TOKEN_ID="$(GLM52_VALIDATION_INPUT_TOKEN_ID)" GLM52_CHAIN_DENSE_TO_LAYER3_ROUTED_EXPERT_NVFP4_TOPK=1 GLM52_PIPELINE_OUTPUT_HIDDEN_BF16="$(GLM52_PIPELINE_OUTPUT_HIDDEN_BF16)"
else ifeq ($(GLM52_VALIDATION_MODE),routed_from_hidden)
GLM52_VALIDATION_ENV := $(GLM52_VALIDATION_BASE_ENV) GLM52_CHAIN_ROUTED_FROM_HIDDEN_BF16=1 GLM52_PIPELINE_INPUT_HIDDEN_BF16="$(GLM52_PIPELINE_INPUT_HIDDEN_BF16)" GLM52_PIPELINE_OUTPUT_HIDDEN_BF16="$(GLM52_PIPELINE_OUTPUT_HIDDEN_BF16)"
else ifeq ($(GLM52_VALIDATION_MODE),routed_from_hidden_final)
GLM52_VALIDATION_ENV := $(GLM52_VALIDATION_BASE_ENV) GLM52_CHAIN_ROUTED_FROM_HIDDEN_FINAL_TOKEN=1 GLM52_PIPELINE_INPUT_HIDDEN_BF16="$(GLM52_PIPELINE_INPUT_HIDDEN_BF16)" GLM52_PIPELINE_OUTPUT_HIDDEN_BF16="$(GLM52_PIPELINE_OUTPUT_HIDDEN_BF16)"
else ifeq ($(GLM52_VALIDATION_MODE),exact_ring_stage_slice)
GLM52_VALIDATION_ENV := $(GLM52_VALIDATION_BASE_ENV) GLM52_EXACT_RING_STAGE_SLICE=1 GLM52_EXACT_RING_MODEL_QUANTIZATION="$(GLM52_EXACT_RING_MODEL_QUANTIZATION)" GLM52_STAGE_PACK_DIR="$(GLM52_STAGE_PACK_DIR)" GLM52_FP8_MOE_PACK_DIR="$(GLM52_FP8_MOE_PACK_DIR)" GLM52_PIPELINE_INPUT_HIDDEN_BF16="$(GLM52_PIPELINE_INPUT_HIDDEN_BF16)" GLM52_PIPELINE_OUTPUT_HIDDEN_BF16="$(GLM52_PIPELINE_OUTPUT_HIDDEN_BF16)"
else ifeq ($(GLM52_VALIDATION_MODE),exact_ring_stage_slice_final)
GLM52_VALIDATION_ENV := $(GLM52_VALIDATION_BASE_ENV) GLM52_EXACT_RING_STAGE_SLICE=1 GLM52_EXACT_RING_STAGE_SLICE_FINAL_TOKEN=1 GLM52_EXACT_RING_MODEL_QUANTIZATION="$(GLM52_EXACT_RING_MODEL_QUANTIZATION)" GLM52_STAGE_PACK_DIR="$(GLM52_STAGE_PACK_DIR)" GLM52_FP8_MOE_PACK_DIR="$(GLM52_FP8_MOE_PACK_DIR)" GLM52_PIPELINE_INPUT_HIDDEN_BF16="$(GLM52_PIPELINE_INPUT_HIDDEN_BF16)" GLM52_PIPELINE_OUTPUT_HIDDEN_BF16="$(GLM52_PIPELINE_OUTPUT_HIDDEN_BF16)"
else ifeq ($(GLM52_VALIDATION_MODE),exact_ring_stage_sequence)
GLM52_VALIDATION_ENV := $(GLM52_VALIDATION_BASE_ENV) GLM52_EXACT_RING_STAGE_SEQUENCE=1 GLM52_EXACT_RING_MODEL_QUANTIZATION="$(GLM52_EXACT_RING_MODEL_QUANTIZATION)" GLM52_STAGE_PACK_DIR="$(GLM52_STAGE_PACK_DIR)" GLM52_FP8_MOE_PACK_DIR="$(GLM52_FP8_MOE_PACK_DIR)" GLM52_PIPELINE_INPUT_HIDDEN_BF16="$(GLM52_PIPELINE_INPUT_HIDDEN_BF16)" GLM52_PIPELINE_OUTPUT_HIDDEN_BF16="$(GLM52_PIPELINE_OUTPUT_HIDDEN_BF16)"
else
$(error unsupported GLM52_VALIDATION_MODE '$(GLM52_VALIDATION_MODE)')
endif

archive: require_sm121a $(MODULE_ARCHIVE)

cold_archive:
	$(MAKE) clean
	$(MAKE) archive

# The variant set. variants is the compile-only half of the variant contract:
# every bucket's archive from the one template above. cold_variants mirrors
# cold_archive. publish_variants is the publish half: each bucket validated
# and published under its own module ID, so the module library holds four
# content-addressed identities the loader resolves between - the naming
# convention SPEC.md's contract needs and nothing more.
variants: require_sm121a $(GLM52_BATCH_VARIANT_ARCHIVES)

cold_variants:
	$(MAKE) clean
	$(MAKE) variants

publish_variants: require_sm121a require_latency_limit require_fast_model_dir require_validation_inputs prepare_validation_output $(GLM52_B12X_RESIDENT_PACK_PREREQ) cold_variants
	$(MAKE) -C $(REPOSITORY_ROOT) build/sparkpipe_module_publish
	@set -e; \
		for bucket in $(GLM52_BATCH_VARIANT_BUCKETS); do \
			$(GLM52_VALIDATION_ENV) \
				$(REPOSITORY_ROOT)/build/sparkpipe_module_publish \
				--library $(MODULE_LIBRARY_ROOT) \
				--module $(GLM52_MODULE_ID_PREFIX).b$$bucket.$(GLM52_MODULE_ID_SUFFIX) \
				--target cuda.sm121.glm52.resident_decode_stage.bf16 \
				--link-unit $(call GLM52_BATCH_VARIANT_ARCHIVE,$$bucket) \
				--recipe $(VALIDATION_RECIPE) \
				--initialize SparkGlm52ResidentDecodeStageInitialize \
				--execute SparkGlm52ResidentDecodeStageExecute \
				--admit SparkGlm52ResidentDecodeStageAdmit \
				--snapshot SparkGlm52ResidentDecodeStageSnapshot \
				--destroy SparkGlm52ResidentDecodeStageDestroy \
				--validator validation/validate_glm52_resident_decode_stage_cuda.sh \
				--validator-arg $(MAX_STAGE_MICROSECONDS); \
		done

require_sm121a:
	@test "$(CUDA_ARCH)" = "sm_121a" || \
		{ echo "GLM52 required CUDA module contract requires CUDA_ARCH=sm_121a" >&2; exit 2; }

require_latency_limit:
	@case "$(MAX_STAGE_MICROSECONDS)" in \
		''|0|*[!0-9]*) \
			echo "set MAX_STAGE_MICROSECONDS to a positive integer qualified stage limit" >&2; \
			exit 2 ;; \
	esac

require_model_dir:
	@test -n "$(GLM52_MODEL_DIR)" || \
		{ echo "set GLM52_MODEL_DIR to the live GLM artifact directory" >&2; exit 2; }

require_fast_model_dir: require_model_dir
	@if test "$(GLM52_ALLOW_REMOTE_MODEL_DIR)" != "1"; then \
		case "$(GLM52_MODEL_DIR)" in \
			/mnt/mac/*|/Volumes/*) \
				echo "GLM52 hardware validation requires local Spark NVMe artifacts, not remote model source: $(GLM52_MODEL_DIR)" >&2; \
				echo "set GLM52_ALLOW_REMOTE_MODEL_DIR=1 only for non-performance debugging" >&2; \
				exit 2; \
				;; \
		esac; \
	fi

require_validation_inputs:
ifneq (,$(filter routed_from_hidden routed_from_hidden_final exact_ring_stage_slice exact_ring_stage_slice_final exact_ring_stage_sequence,$(GLM52_VALIDATION_MODE)))
	@test -n "$(GLM52_PIPELINE_INPUT_HIDDEN_BF16)" || \
		{ echo "set GLM52_PIPELINE_INPUT_HIDDEN_BF16 for $(GLM52_VALIDATION_MODE) validation" >&2; exit 2; }
	@test -s "$(GLM52_PIPELINE_INPUT_HIDDEN_BF16)" || \
		{ echo "missing GLM52_PIPELINE_INPUT_HIDDEN_BF16: $(GLM52_PIPELINE_INPUT_HIDDEN_BF16)" >&2; exit 2; }
endif

prepare_validation_output:
	@if test -n "$(GLM52_PIPELINE_OUTPUT_HIDDEN_BF16)"; then \
		mkdir -p "$(dir $(GLM52_PIPELINE_OUTPUT_HIDDEN_BF16))"; \
	fi

$(BUILD_DIRECTORY):
	mkdir -p $(BUILD_DIRECTORY)

$(HOST_OBJECT): $(REPOSITORY_ROOT)/inference/stage/module.c | $(BUILD_DIRECTORY)
	$(CC) -I$(REPOSITORY_ROOT) -I$(REPOSITORY_ROOT)/include -I$(REPOSITORY_ROOT)/model-families/glm52/include -Iinclude -Isource $(CFLAGS) \
		-DNDEBUG -DSPARK_RESIDENT_DECODE_STAGE_REQUIRE_EXTERNAL_CUDA_MODULES \
		-fPIC -fvisibility=hidden -MMD -MP -c $< -o $@

$(VALIDATION_OBJECT): source/spark_glm52_resident_decode_stage_validation.c | $(BUILD_DIRECTORY)
	$(CC) -I$(REPOSITORY_ROOT) -I$(REPOSITORY_ROOT)/include -I$(REPOSITORY_ROOT)/model-families/glm52/include -Iinclude -Isource $(CFLAGS) \
		-DNDEBUG -DSPARK_RESIDENT_DECODE_STAGE_REQUIRE_EXTERNAL_CUDA_MODULES \
		-fPIC -fvisibility=hidden -MMD -MP -c $< -o $@

$(CUDA_OBJECT): $(REPOSITORY_ROOT)/inference/stage/dispatch.cu | $(BUILD_DIRECTORY)
	$(NVCC) $(NVCCFLAGS) -I$(REPOSITORY_ROOT) -I$(REPOSITORY_ROOT)/include -I$(REPOSITORY_ROOT)/model-families/glm52/include -Iinclude -Isource \
		-Xcompiler -fPIC -Xcompiler -fvisibility=hidden -MMD -MP -c $< -o $@

# The required SM121 stage symbols (SparkGlm52Sm121RequiredDecodeStage*) are
# an external seam: they arrive at package link time through
# GLM52_REQUIRED_CUDA_LINK_ARGS, never from this archive. See
# docs/GLM52_SM121_REQUIRED_CUDA_MODULE.md.

$(FP8_MOE_PLAN_OBJECT): source/spark_glm52_resident_decode_stage_fp8_moe_plan.cu | $(BUILD_DIRECTORY)
	$(NVCC) $(NVCCFLAGS) -I$(REPOSITORY_ROOT) -I$(REPOSITORY_ROOT)/include -I$(REPOSITORY_ROOT)/model-families/glm52/include -Iinclude -Isource \
		-Xcompiler -fPIC -Xcompiler -fvisibility=hidden -MMD -MP -c $< -o $@

$(SERVING_ADAPTER_OBJECT): $(REPOSITORY_ROOT)/inference/stage/serving_adapter.cu | $(BUILD_DIRECTORY)
	$(NVCC) $(NVCCFLAGS) -I$(REPOSITORY_ROOT) -I$(REPOSITORY_ROOT)/include -I$(REPOSITORY_ROOT)/model-families/glm52/include -Iinclude -Isource \
		-Xcompiler -fPIC -Xcompiler -fvisibility=hidden -MMD -MP -c $< -o $@

$(MODULE_ARCHIVE): $(HOST_OBJECT) $(VALIDATION_OBJECT) $(CUDA_OBJECT) $(FP8_MOE_PLAN_OBJECT) $(SERVING_ADAPTER_OBJECT)
	rm -f $@
	$(AR) rcs $@ $^

$(ARTIFACT_CHECK_TOOL): $(REPOSITORY_ROOT)/runtime/pack/artifact_check.c | $(BUILD_DIRECTORY)
	$(CC) -I$(REPOSITORY_ROOT) -I$(REPOSITORY_ROOT)/include -I$(REPOSITORY_ROOT)/model-families/glm52/include -Iinclude -Isource $(CFLAGS) \
		$< $(REPOSITORY_ROOT)/build/libsparkpipe_compiler.a $(REPOSITORY_ROOT)/build/libsparkpipe_model_common.a \
		-ldl -o $@

artifact_check: require_model_dir $(ARTIFACT_CHECK_TOOL)
	$(ARTIFACT_CHECK_TOOL) \
		--model-dir "$(GLM52_MODEL_DIR)" \
		--model "$(REPOSITORY_ROOT)/examples/model_descriptions/glm52_resident_decode_stage_firmware.json" \
		$(ARTIFACT_CHECK_ARGS)

b12x_resident_pack: require_model_dir
	@test -s "$(AOT_MANIFEST)" || \
		{ echo "missing B12x AOT manifest: $(AOT_MANIFEST)" >&2; exit 2; }
ifeq ($(B12X_MOE_PACK_REQUIRE_REUSE),1)
	@test -s "$(B12X_MOE_PACK_DIR)/resident_moe_pack_manifest.json" || \
		{ echo "missing B12x resident pack manifest: $(B12X_MOE_PACK_DIR)/resident_moe_pack_manifest.json; run setup pack generation first" >&2; exit 2; }
	@set -e; \
		old_ifs="$$IFS"; \
		IFS=,; \
		layers="$(B12X_MOE_PACK_LAYERS)"; \
		for layer_index in $$layers; do \
			pack_path=$$(printf "%s/glm52_layer_%04u_b12x_moe.spb12x" "$(B12X_MOE_PACK_DIR)" "$$layer_index"); \
			test -s "$$pack_path" || \
				{ echo "missing B12x resident pack: $$pack_path; run setup pack generation first" >&2; exit 2; }; \
		done; \
		IFS="$$old_ifs"
else
	$(B12X_PACK_PYTHON) $(REPOSITORY_ROOT)/tools/glm52_b12x_resident_pack.py \
		--model-dir "$(GLM52_MODEL_DIR)" \
		--aot-manifest "$(AOT_MANIFEST)" \
		--layers "$(B12X_MOE_PACK_LAYERS)" \
		--output-dir "$(B12X_MOE_PACK_DIR)" \
		--jobs "$(B12X_MOE_PACK_JOBS)" \
		--reuse-valid \
		$(if $(filter 1,$(B12X_MOE_PACK_VERIFY_REUSED_SHA256)),--verify-reused-sha256,)
endif

validate: require_sm121a require_latency_limit require_fast_model_dir require_validation_inputs prepare_validation_output $(GLM52_B12X_RESIDENT_PACK_PREREQ) cold_archive
	$(GLM52_VALIDATION_ENV) \
		validation/validate_glm52_resident_decode_stage_cuda.sh \
		$(MAX_STAGE_MICROSECONDS) $(MODULE_ARCHIVE)

validate_dsa_topk_parity: require_sm121a archive
	$(MAKE) -C $(REPOSITORY_ROOT) glm52_flashinfer_b12x_moe_adapter glm52_b12x_compiled_backend
	GLM52_REQUIRED_CUDA_LINK_ARGS="$(GLM52_DSA_TOPK_LINK_ARGS)" \
		bash validation/validate_glm52_dsa_topk_parity.sh \
		$(MODULE_ARCHIVE)

publish: require_sm121a require_latency_limit require_fast_model_dir require_validation_inputs prepare_validation_output $(GLM52_B12X_RESIDENT_PACK_PREREQ) cold_archive
	$(MAKE) -C $(REPOSITORY_ROOT) build/sparkpipe_module_publish
	$(GLM52_VALIDATION_ENV) \
		$(REPOSITORY_ROOT)/build/sparkpipe_module_publish \
		--library $(MODULE_LIBRARY_ROOT) \
		--module $(GLM52_MODULE_ID) \
		--target cuda.sm121.glm52.resident_decode_stage.bf16 \
		--link-unit $(MODULE_ARCHIVE) \
		--recipe $(VALIDATION_RECIPE) \
		--initialize SparkGlm52ResidentDecodeStageInitialize \
		--execute SparkGlm52ResidentDecodeStageExecute \
		--admit SparkGlm52ResidentDecodeStageAdmit \
		--snapshot SparkGlm52ResidentDecodeStageSnapshot \
		--destroy SparkGlm52ResidentDecodeStageDestroy \
		--validator validation/validate_glm52_resident_decode_stage_cuda.sh \
		--validator-arg $(MAX_STAGE_MICROSECONDS)

package: publish
	$(MAKE) -C $(REPOSITORY_ROOT) build/sparkpipe_model_compile
	rm -rf $(PACKAGE_DIRECTORY)
	$(REPOSITORY_ROOT)/build/sparkpipe_model_compile \
		--model $(REPOSITORY_ROOT)/examples/model_descriptions/glm52_resident_decode_stage_firmware.json \
		--library $(MODULE_LIBRARY_ROOT) \
		--output $(PACKAGE_DIRECTORY) \
		--include $(REPOSITORY_ROOT)/include \
		--cc cc \
		--cc-arg -L$(CUDA_HOME)/lib64 \
		--cc-arg -lcudart \
		--cc-arg -lcublasLt \
		--cc-arg -lcublas \
		--cc-arg -lstdc++ \
		$(foreach arg,$(GLM52_REQUIRED_CUDA_LINK_ARGS),--cc-arg $(arg)) \
		$(REQUIRED_CUDA_CC_ARGS)
ifeq ($(GLM52_REQUIRE_B12X_RESIDENT_PACK),1)
	mkdir -p $(PACKAGE_DIRECTORY)/artifacts/glm52_b12x_resident_moe
	@set -e; \
		case "$(B12X_MOE_PACK_PACKAGE_MODE)" in \
			hardlink) \
				for pack_path in $(B12X_MOE_PACK_DIR)/*.spb12x $(B12X_MOE_PACK_DIR)/resident_moe_pack_manifest.json; do \
					target_path="$(PACKAGE_DIRECTORY)/artifacts/glm52_b12x_resident_moe/$$(basename "$$pack_path")"; \
					ln -f "$$pack_path" "$$target_path"; \
				done; \
				;; \
			copy) \
				cp --sparse=always -f $(B12X_MOE_PACK_DIR)/*.spb12x $(PACKAGE_DIRECTORY)/artifacts/glm52_b12x_resident_moe/; \
				cp --sparse=always -f $(B12X_MOE_PACK_DIR)/resident_moe_pack_manifest.json $(PACKAGE_DIRECTORY)/artifacts/glm52_b12x_resident_moe/; \
				;; \
			*) \
				echo "unsupported B12X_MOE_PACK_PACKAGE_MODE: $(B12X_MOE_PACK_PACKAGE_MODE)" >&2; \
				exit 2; \
				;; \
		esac
endif
	$(GLM52_VALIDATION_ENV) \
		validation/validate_glm52_resident_decode_stage_cuda.sh \
		$(MAX_STAGE_MICROSECONDS) $(MODULE_ARCHIVE) \
		$(PACKAGE_DIRECTORY)/stages/stage_000/model_driver.so

clean:
	rm -rf $(BUILD_DIRECTORY)

-include $(HOST_DEPENDENCY) $(CUDA_DEPENDENCY) $(FP8_MOE_PLAN_DEPENDENCY) $(SERVING_ADAPTER_DEPENDENCY) $(GLM52_BATCH_VARIANT_DEPENDENCIES)
