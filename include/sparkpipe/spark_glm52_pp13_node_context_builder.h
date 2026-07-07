#pragma once

#include <stdint.h>

#include "sparkpipe/spark_glm52_pp13_runtime.h"
#include "sparkpipe/spark_glm52_pp13_work_control.h"
#include "sparkpipe/spark_glm52_serving_engine.h"
#include "sparkpipe/spark_model_driver.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_ABI_VERSION 2u
#define SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_MAX_PREFILL_TOKENS 256u
#define SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_INTERFACE_SYMBOL \
	"SparkGlm52Pp13NodeContextBuilderGetInterface"
#define SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_CONFIGURATION_BYTES \
	((uint32_t)sizeof(SparkGlm52Pp13NodeContextBuilderConfiguration))
#define SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_RESULT_BYTES \
	((uint32_t)sizeof(SparkGlm52Pp13NodeContextBuilderResult))
#define SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_INTERFACE_BYTES \
	((uint32_t)sizeof(SparkGlm52Pp13NodeContextBuilderInterface))

#define SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_CAP_FP8_PACKS 0x00000001u
#define SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_CAP_STAGE_SLICE 0x00000002u
#define SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_CAP_RANK0_TOKEN_INPUT 0x00000004u
#define SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_CAP_RANK_WORK_DISPATCH 0x00000008u
#define SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_REQUIRED_PRODUCTION_CAPS \
	(SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_CAP_FP8_PACKS | \
	 SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_CAP_STAGE_SLICE | \
	 SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_CAP_RANK0_TOKEN_INPUT | \
	 SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_CAP_RANK_WORK_DISPATCH)

typedef struct SparkGlm52Pp13NodeContextBuilderConfiguration
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t flags;
	uint32_t rank_index;
	uint32_t max_active_sequence_count;
	uint32_t port_base;
	uint32_t reserved0;
	uint32_t reserved1;
	const char *fp8_pack_root;
	const char *stagepack_root;
	const char *embedding_pack_path;
	const char *node_target;
	const SparkGlm52Pp13RuntimeRankPlan *rank_plan;
} SparkGlm52Pp13NodeContextBuilderConfiguration;

typedef struct SparkGlm52Pp13NodeContextBuilderResult
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t flags;
	uint32_t rank_index;
	uint32_t first_layer_index;
	uint32_t layer_count;
	uint32_t hidden_dimension;
	uint32_t vocabulary_size;
	void *node_context;
	const void *embedding_weight_bf16;
	void *private_state;
} SparkGlm52Pp13NodeContextBuilderResult;

typedef SparkStatus (*SparkGlm52Pp13NodeContextBuilderInitializeFunction)(
	const SparkGlm52Pp13NodeContextBuilderConfiguration *configuration,
	void **builder_state);
typedef void (*SparkGlm52Pp13NodeContextBuilderDestroyFunction)(
	void *builder_state);
typedef SparkStatus (*SparkGlm52Pp13NodeContextBuilderBuildFunction)(
	void *builder_state,
	SparkGlm52Pp13NodeContextBuilderResult *result);
typedef void (*SparkGlm52Pp13NodeContextBuilderDestroyResultFunction)(
	void *builder_state,
	SparkGlm52Pp13NodeContextBuilderResult *result);
typedef SparkStatus (*SparkGlm52Pp13NodeContextBuilderAttachDriverFunction)(
	void *builder_state,
	const SparkModelDriverInterface *driver_interface,
	void *driver_instance,
	const SparkModelDriverProgramDescriptor *program,
	SparkHiddenTransportSession *output_transport_session);
typedef SparkStatus (*SparkGlm52Pp13NodeContextBuilderIdlePumpFunction)(
	void *idle_pump_context);
typedef SparkStatus (*SparkGlm52Pp13NodeContextBuilderPrefillFunction)(
	void *builder_state,
	const SparkGlm52PromptPipelinePrefillDispatch *prefill_dispatch,
	SparkGlm52Pp13NodeContextBuilderIdlePumpFunction idle_pump_function,
	void *idle_pump_context);
typedef SparkStatus (*SparkGlm52Pp13NodeContextBuilderDecodeFunction)(
	void *builder_state,
	const SparkGlm52ServingDecodeDispatch *decode_dispatch,
	SparkGlm52ServingDecodeResult *decode_result);
typedef SparkStatus (*SparkGlm52Pp13NodeContextBuilderSubmitWorkFunction)(
	void *builder_state,
	const SparkGlm52Pp13WorkControlPacket *work_packet,
	SparkHiddenTransportSession *input_transport_session,
	SparkHiddenTransportSession *output_transport_session,
	SparkModelDriverCompletionFunction completion_function,
	void *completion_context);

typedef struct SparkGlm52Pp13NodeContextBuilderInterface
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t capability_flags;
	uint32_t reserved0;
	SparkGlm52Pp13NodeContextBuilderInitializeFunction initialize;
	SparkGlm52Pp13NodeContextBuilderDestroyFunction destroy;
	SparkGlm52Pp13NodeContextBuilderBuildFunction build;
	SparkGlm52Pp13NodeContextBuilderDestroyResultFunction destroy_result;
	SparkGlm52Pp13NodeContextBuilderAttachDriverFunction attach_driver;
	SparkGlm52Pp13NodeContextBuilderPrefillFunction prefill;
	SparkGlm52Pp13NodeContextBuilderDecodeFunction decode;
	SparkGlm52Pp13NodeContextBuilderSubmitWorkFunction submit_work;
} SparkGlm52Pp13NodeContextBuilderInterface;

typedef const SparkGlm52Pp13NodeContextBuilderInterface *(
	*SparkGlm52Pp13NodeContextBuilderGetInterfaceFunction)(void);

typedef struct SparkGlm52Pp13NodeContextBuilderDynamicLibrary
{
	void *dynamic_library;
	SparkGlm52Pp13NodeContextBuilderInterface builder_interface;
} SparkGlm52Pp13NodeContextBuilderDynamicLibrary;

SparkStatus SparkGlm52Pp13NodeContextBuilderValidateInterface(
	const SparkGlm52Pp13NodeContextBuilderInterface *builder_interface,
	uint32_t required_capability_flags);
SparkStatus SparkGlm52Pp13NodeContextBuilderValidateResult(
	const SparkGlm52Pp13NodeContextBuilderResult *result,
	const SparkGlm52Pp13RuntimeRankPlan *rank_plan);
SparkStatus SparkGlm52Pp13NodeContextBuilderLoadInterfaceFromSharedObject(
	const char *shared_object_path,
	uint32_t required_capability_flags,
	SparkGlm52Pp13NodeContextBuilderDynamicLibrary *library);
void SparkGlm52Pp13NodeContextBuilderUnloadInterface(
	SparkGlm52Pp13NodeContextBuilderDynamicLibrary *library);

#ifdef __cplusplus
}
#endif
