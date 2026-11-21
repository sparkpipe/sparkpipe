#include <pthread.h>
#include <stdio.h>
#include <string.h>

#include "sparkpipe/spark_error_site.h"
#include "runtime/device/device_backend.h"
#include "runtime/device/nvrm/nvrm_register.h"
#include "runtime/device/nvrm/spark_gpu_nvrm.h"

#define GPU_OWNER "hosted launches"

static _Thread_local SparkGpuError gpu_last_error = SPARK_GPU_SUCCESS;
static pthread_mutex_t gpu_mutex = PTHREAD_MUTEX_INITIALIZER;
static SparkDeviceQueue *gpu_default_queue;

static SparkGpuError GpuFromStatus(SparkStatus status)
{
	switch ( status )
	{
	case SPARK_STATUS_OK:
		return(SPARK_GPU_SUCCESS);
	case SPARK_STATUS_INVALID_ARGUMENT:
	case SPARK_STATUS_NOT_FOUND:
	case SPARK_STATUS_DUPLICATE:
	case SPARK_STATUS_PARSE_ERROR:
	case SPARK_STATUS_TARGET_MISMATCH:
		return(SPARK_GPU_ERROR_INVALID_VALUE);
	case SPARK_STATUS_CAPACITY_EXCEEDED:
		return(SPARK_GPU_ERROR_MEMORY_ALLOCATION);
	case SPARK_STATUS_UNSUPPORTED:
		return(SPARK_GPU_ERROR_NOT_SUPPORTED);
	case SPARK_STATUS_BUSY:
		return(SPARK_GPU_ERROR_NOT_READY);
	default:
		return(SPARK_GPU_ERROR_UNKNOWN);
	}
}

static SparkGpuError GpuRecord(SparkStatus status)
{
	gpu_last_error = GpuFromStatus(status);
	return(gpu_last_error);
}

#define GPU_FAIL(status) \
	do { \
		const SparkGpuError gpu_error = GpuRecord(status); \
		SPARK_FAIL(gpu_error); \
	} while (0)

#define GPU_REFUSE(status, text) \
	do { \
		(void)SparkDeviceRecordError((status),"kernel_launch",GPU_OWNER,0,(text)); \
		GPU_FAIL(status); \
	} while (0)

SparkGpuError SparkGpuGetLastError(void)
{
	const SparkGpuError error = gpu_last_error;
	gpu_last_error = SPARK_GPU_SUCCESS;
	return(error);
}

SparkGpuError SparkGpuPeekAtLastError(void)
{
	return(gpu_last_error);
}

const char *SparkGpuGetErrorString(SparkGpuError error)
{
	switch ( error )
	{
	case SPARK_GPU_SUCCESS:
		return("no error");
	case SPARK_GPU_ERROR_INVALID_VALUE:
		return("invalid argument");
	case SPARK_GPU_ERROR_MEMORY_ALLOCATION:
		return("out of memory");
	case SPARK_GPU_ERROR_INVALID_DEVICE:
		return("invalid device");
	case SPARK_GPU_ERROR_NOT_READY:
		return("device not ready");
	case SPARK_GPU_ERROR_NOT_SUPPORTED:
		return("operation not supported");
	default:
		return("unknown error");
	}
}

SparkGpuError SparkNvrmGpuDefaultStream(SparkGpuStream *stream)
{
	SparkStatus status = SPARK_STATUS_OK;
	pthread_mutex_lock(&gpu_mutex);
	if ( gpu_default_queue == 0 )
		status = SparkDeviceQueueCreate(&gpu_default_queue,"default stream");
	*stream = gpu_default_queue;
	pthread_mutex_unlock(&gpu_mutex);
	if ( status != SPARK_STATUS_OK )
		GPU_FAIL(status);
	return(SPARK_GPU_SUCCESS);
}

SparkGpuError SparkNvrmGpuRelease(void)
{
	SparkStatus status = SPARK_STATUS_OK;
	pthread_mutex_lock(&gpu_mutex);
	if ( gpu_default_queue != 0 && (status = SparkDeviceQueueDestroy(gpu_default_queue)) == SPARK_STATUS_OK )
		gpu_default_queue = 0;
	pthread_mutex_unlock(&gpu_mutex);
	if ( status == SPARK_STATUS_OK )
		status = SparkNvrmRegisteredRelease();
	if ( status != SPARK_STATUS_OK )
		GPU_FAIL(status);
	return(SPARK_GPU_SUCCESS);
}

static SparkGpuError GpuLayoutAgrees(const SparkDeviceKernel *kernel, const SparkDeviceParameter *layout, uint32_t count, uint32_t bytes,
	const char *name)
{
	SparkDeviceKernelInfo info;
	char text[256];
	uint32_t index;
	if ( SparkDeviceKernelGetInfo(kernel,&info) != SPARK_STATUS_OK )
		GPU_FAIL(SparkDeviceLastError()->status);
	if ( info.parameter_count != count || info.parameter_bytes != bytes )
	{
		snprintf(text,sizeof(text),"%.48s: the host passes %u parameters in %u bytes, the kernel takes %u in %u",name,count,bytes,info.parameter_count,
			info.parameter_bytes);
		GPU_REFUSE(SPARK_STATUS_INVALID_ARGUMENT,text);
	}
	for (index = 0u; index < count; index++)
		if ( info.parameters[index].offset != layout[index].offset || info.parameters[index].bytes != layout[index].bytes )
		{
			snprintf(text,sizeof(text),"%.48s: parameter %u is %u bytes at %u on the host, %u bytes at %u in the kernel",name,index,layout[index].bytes,
				layout[index].offset,info.parameters[index].bytes,info.parameters[index].offset);
			GPU_REFUSE(SPARK_STATUS_INVALID_ARGUMENT,text);
		}
	return(SPARK_GPU_SUCCESS);
}

SparkGpuError SparkNvrmLaunchPacked(const void *function, const uint32_t grid[3], const uint32_t block[3], size_t shared_bytes, SparkGpuStream stream,
	const void *parameters, const SparkDeviceParameter *layout, uint32_t count, uint32_t bytes)
{
	SparkDeviceKernel *kernel = 0;
	SparkDeviceLaunch launch;
	const char *name = 0;
	SparkGpuError error;
	if ( grid == 0 || block == 0 || (count != 0u && (layout == 0 || parameters == 0)) || shared_bytes > UINT32_MAX )
		GPU_REFUSE(SPARK_STATUS_INVALID_ARGUMENT,"a hosted launch needs its grid, its block, its parameters and at most 4 GiB of shared memory");
	if ( SparkNvrmRegisteredKernel(function,&kernel) != SPARK_STATUS_OK )
		GPU_FAIL(SparkDeviceLastError()->status);
	(void)SparkNvrmRegisteredName(function,&name);
	if ( (error = GpuLayoutAgrees(kernel,layout,count,bytes,name)) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(error);
	if ( stream == 0 && (error = SparkNvrmGpuDefaultStream(&stream)) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(error);
	memset(&launch,0,sizeof(launch));
	launch.groups.x = grid[0];
	launch.groups.y = grid[1];
	launch.groups.z = grid[2];
	launch.threads.x = block[0];
	launch.threads.y = block[1];
	launch.threads.z = block[2];
	launch.shared_bytes = (uint32_t)shared_bytes;
	launch.constants = bytes != 0u ? parameters : 0;
	launch.constant_bytes = bytes;
	if ( SparkDeviceKernelLaunch(stream,kernel,&launch) != SPARK_STATUS_OK )
		GPU_FAIL(SparkDeviceLastError()->status);
	return(SPARK_GPU_SUCCESS);
}

SparkGpuError SparkNvrmLaunchUnsupported(const void *function, const char *what)
{
	const char *name = 0;
	char text[256];
	if ( SparkNvrmRegisteredName(function,&name) != SPARK_STATUS_OK )
		name = "an unregistered function";
	snprintf(text,sizeof(text),"%.64s: %.48s needs launch descriptor fields the backend does not set yet",name,what);
	GPU_REFUSE(SPARK_STATUS_UNSUPPORTED,text);
}

SparkGpuError SparkNvrmSharedMemoryOptIn(const void *function, uint32_t shared_bytes)
{
	const SparkDeviceProperties *properties = SparkDeviceGetProperties();
	const char *name = 0;
	if ( properties == 0 )
		GPU_REFUSE(SPARK_STATUS_INVALID_ARGUMENT,"shared memory is granted on an initialized device");
	if ( SparkNvrmRegisteredName(function,&name) != SPARK_STATUS_OK )
		GPU_FAIL(SparkDeviceLastError()->status);
	if ( shared_bytes > properties->max_shared_bytes )
		GPU_REFUSE(SPARK_STATUS_UNSUPPORTED,"the kernel asks for more group-shared memory than the device has");
	return(SPARK_GPU_SUCCESS);
}
