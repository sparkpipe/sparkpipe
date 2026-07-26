#include "sparkpipe/spark_glm52_service_backend.h"

#include <dlfcn.h>
#include <string.h>

SparkStatus SparkGlm52ServiceBackendValidateInterface(
	const SparkGlm52ServiceBackendInterface *backend_interface,
	uint32_t required_capability_flags)
{
	if (backend_interface == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (backend_interface->abi_version !=
		SPARK_GLM52_SERVICE_BACKEND_ABI_VERSION)
		return SPARK_STATUS_ABI_MISMATCH;
	if (backend_interface->descriptor_bytes !=
		SPARK_GLM52_SERVICE_BACKEND_INTERFACE_BYTES)
		return SPARK_STATUS_ABI_MISMATCH;
	if ((backend_interface->capability_flags & required_capability_flags) !=
		required_capability_flags)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (backend_interface->reserved0 != 0u ||
		backend_interface->initialize == 0 ||
		backend_interface->destroy == 0 ||
		backend_interface->get_view == 0 ||
		backend_interface->pump == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if ((backend_interface->capability_flags &
			SPARK_GLM52_SERVICE_BACKEND_CAPABILITY_POLL_DESCRIPTORS) != 0u &&
		backend_interface->get_poll_descriptors == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52ServiceBackendLoadInterfaceFromSharedObject(
	const char *shared_object_path,
	uint32_t required_capability_flags,
	SparkGlm52ServiceBackendDynamicLibrary *library)
{
	void *dynamic_library;
	SparkGlm52ServiceBackendGetInterfaceFunction get_interface;
	const SparkGlm52ServiceBackendInterface *backend_interface;
	SparkStatus status;

	if (shared_object_path == 0 || library == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	memset(library,0,sizeof(*library));
	dynamic_library = dlopen(shared_object_path,RTLD_NOW | RTLD_LOCAL);
	if (dynamic_library == 0)
		return SPARK_STATUS_NOT_FOUND;
	get_interface = (SparkGlm52ServiceBackendGetInterfaceFunction)dlsym(
		dynamic_library,
		SPARK_GLM52_SERVICE_BACKEND_INTERFACE_SYMBOL);
	if (get_interface == 0)
	{
		dlclose(dynamic_library);
		return SPARK_STATUS_NOT_FOUND;
	}
	backend_interface = get_interface();
	status = SparkGlm52ServiceBackendValidateInterface(
		backend_interface,
		required_capability_flags);
	if (status != SPARK_STATUS_OK)
	{
		dlclose(dynamic_library);
		return status;
	}
	library->dynamic_library = dynamic_library;
	library->backend_interface = *backend_interface;
	return SPARK_STATUS_OK;
}

void SparkGlm52ServiceBackendUnloadInterface(
	SparkGlm52ServiceBackendDynamicLibrary *library)
{
	if (library == 0)
		return;
	if (library->dynamic_library != 0)
		dlclose(library->dynamic_library);
	memset(library,0,sizeof(*library));
}
