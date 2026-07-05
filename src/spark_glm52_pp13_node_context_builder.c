#include "sparkpipe/spark_glm52_pp13_node_context_builder.h"

#include <dlfcn.h>
#include <string.h>

SparkStatus SparkGlm52Pp13NodeContextBuilderValidateInterface(
	const SparkGlm52Pp13NodeContextBuilderInterface *builder_interface,
	uint32_t required_capability_flags)
{
	if (builder_interface == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (builder_interface->abi_version !=
		SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_ABI_VERSION)
		return SPARK_STATUS_ABI_MISMATCH;
	if (builder_interface->descriptor_bytes !=
		SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_INTERFACE_BYTES)
		return SPARK_STATUS_ABI_MISMATCH;
	if ((builder_interface->capability_flags & required_capability_flags) !=
		required_capability_flags)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (builder_interface->reserved0 != 0u ||
		builder_interface->initialize == 0 ||
		builder_interface->destroy == 0 ||
		builder_interface->build == 0 ||
		builder_interface->destroy_result == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if ((required_capability_flags &
			SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_CAP_RANK0_TOKEN_INPUT) !=
			0u &&
		(builder_interface->attach_driver == 0 ||
		 builder_interface->prefill == 0 ||
		 builder_interface->decode == 0))
		return SPARK_STATUS_INVALID_ARGUMENT;
	if ((required_capability_flags &
			SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_CAP_RANK_WORK_DISPATCH) !=
			0u &&
		builder_interface->submit_work == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52Pp13NodeContextBuilderValidateResult(
	const SparkGlm52Pp13NodeContextBuilderResult *result,
	const SparkGlm52Pp13RuntimeRankPlan *rank_plan)
{
	if (result == 0 || rank_plan == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (result->abi_version !=
		SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_ABI_VERSION ||
		result->descriptor_bytes !=
			SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_RESULT_BYTES)
		return SPARK_STATUS_ABI_MISMATCH;
	if (result->node_context == 0 ||
		result->rank_index != rank_plan->rank_index ||
		result->first_layer_index != rank_plan->first_layer_index ||
		result->layer_count != rank_plan->layer_count ||
		result->hidden_dimension != rank_plan->hidden_dimension)
		return SPARK_STATUS_INVALID_ARGUMENT;
	return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52Pp13NodeContextBuilderLoadInterfaceFromSharedObject(
	const char *shared_object_path,
	uint32_t required_capability_flags,
	SparkGlm52Pp13NodeContextBuilderDynamicLibrary *library)
{
	void *dynamic_library;
	SparkGlm52Pp13NodeContextBuilderGetInterfaceFunction get_interface;
	const SparkGlm52Pp13NodeContextBuilderInterface *builder_interface;
	SparkStatus status;

	if (shared_object_path == 0 || library == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	memset(library,0,sizeof(*library));
	dynamic_library = dlopen(shared_object_path,RTLD_NOW | RTLD_LOCAL);
	if (dynamic_library == 0)
		return SPARK_STATUS_NOT_FOUND;
	get_interface = (SparkGlm52Pp13NodeContextBuilderGetInterfaceFunction)dlsym(
		dynamic_library,
		SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_INTERFACE_SYMBOL);
	if (get_interface == 0)
	{
		dlclose(dynamic_library);
		return SPARK_STATUS_NOT_FOUND;
	}
	builder_interface = get_interface();
	status = SparkGlm52Pp13NodeContextBuilderValidateInterface(
		builder_interface,
		required_capability_flags);
	if (status != SPARK_STATUS_OK)
	{
		dlclose(dynamic_library);
		return status;
	}
	library->dynamic_library = dynamic_library;
	library->builder_interface = *builder_interface;
	return SPARK_STATUS_OK;
}

void SparkGlm52Pp13NodeContextBuilderUnloadInterface(
	SparkGlm52Pp13NodeContextBuilderDynamicLibrary *library)
{
	if (library == 0)
		return;
	if (library->dynamic_library != 0)
		dlclose(library->dynamic_library);
	memset(library,0,sizeof(*library));
}
