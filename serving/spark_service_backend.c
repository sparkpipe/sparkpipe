#include "sparkpipe/spark_service_backend.h"

#include <dlfcn.h>
#include <string.h>

SparkStatus SparkServiceBackendValidateInterface(
	const SparkServiceBackendInterface *backend_interface,
	uint32_t required_capability_flags)
{
	if (backend_interface == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (backend_interface->abi_version !=
		SPARK_SERVICE_BACKEND_ABI_VERSION)
		return SPARK_STATUS_ABI_MISMATCH;
	if (backend_interface->descriptor_bytes !=
		SPARK_SERVICE_BACKEND_INTERFACE_BYTES)
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
			SPARK_SERVICE_BACKEND_CAPABILITY_POLL_DESCRIPTORS) != 0u &&
		backend_interface->get_poll_descriptors == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	return SPARK_STATUS_OK;
}

SparkStatus SparkServiceBackendLoadInterfaceFromSharedObject(
	const char *shared_object_path,
	uint32_t required_capability_flags,
	SparkServiceBackendDynamicLibrary *library)
{
	void *dynamic_library;
	SparkServiceBackendGetInterfaceFunction get_interface;
	const SparkServiceBackendInterface *backend_interface;
	SparkStatus status;

	if (shared_object_path == 0 || library == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	memset(library,0,sizeof(*library));
	dynamic_library = dlopen(shared_object_path,RTLD_NOW | RTLD_LOCAL);
	if (dynamic_library == 0)
		return SPARK_STATUS_NOT_FOUND;
	get_interface = (SparkServiceBackendGetInterfaceFunction)dlsym(
		dynamic_library,
		SPARK_SERVICE_BACKEND_INTERFACE_SYMBOL);
	if (get_interface == 0)
	{
		dlclose(dynamic_library);
		return SPARK_STATUS_NOT_FOUND;
	}
	backend_interface = get_interface();
	status = SparkServiceBackendValidateInterface(
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

void SparkServiceBackendUnloadInterface(
	SparkServiceBackendDynamicLibrary *library)
{
	if (library == 0)
		return;
	if (library->dynamic_library != 0)
		dlclose(library->dynamic_library);
	memset(library,0,sizeof(*library));
}
