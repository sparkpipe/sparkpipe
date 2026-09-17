#include <stdio.h>

#include "sparkpipe/spark_model_serving_adapter.h"

int main(int argc,char **argv)
{
	SparkModelServingAdapterDynamicLibrary library;
	const SparkModelServingAdapterDescriptor *descriptor;
	SparkStatus status;
	if ( argc != 2 )
	{
		fprintf(stderr,"usage: %s ADAPTER_SO\n",argv[0]);
		return(2);
	}
	status = SparkModelServingAdapterLoadInterfaceFromSharedObject(argv[1],0u,&library);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"adapter rejected: status=%u\n",(uint32_t)status);
		return(1);
	}
	descriptor = library.adapter_interface.descriptor;
	printf("abi_version=%u interface_bytes=%u\n",library.adapter_interface.abi_version,library.adapter_interface.interface_bytes);
	printf("stage_count=%u layer_count=%u capability_flags=0x%08x\n",descriptor->stage_count,descriptor->layer_count,descriptor->capability_flags);
	printf("adapter_id=%s\nmodel_id=%s\nprogram=%s\nartifact=%s\n",descriptor->adapter_id,descriptor->model_id,descriptor->driver_program_name,descriptor->artifact_sha256);
	printf("required_interface=valid behavioral_qualification=not_measured\n");
	SparkModelServingAdapterUnloadInterface(&library);
	return(0);
}
