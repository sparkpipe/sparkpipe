#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>

#include "sparkpipe/spark_error_site.h"
#include "runtime/device/device_backend.h"
#include "runtime/device/nvrm/nvrm_gpu.h"
#include "ctrl/ctrl0000/ctrl0000unix.h"
#include "ctrl/ctrl0041.h"

#define SHARE_CONTROL_DEVICE "/dev/nvidiactl"

#define SHARE_TRY(call) \
	do { \
		const SparkStatus share_status = (call); \
		if ( share_status != SPARK_STATUS_OK ) \
			SPARK_FAIL(share_status); \
	} while (0)

SparkStatus SparkNvrmShareExport(SparkNvrmGpu *gpu, uint32_t handle, int *fd, const char *owner)
{
	NV0000_CTRL_OS_UNIX_EXPORT_OBJECT_TO_FD_PARAMS exported;
	const int opened = open(SHARE_CONTROL_DEVICE,O_RDWR | O_CLOEXEC);
	SparkStatus status;
	if ( opened < 0 )
	{
		status = SparkDeviceRecordError(SPARK_STATUS_IO_ERROR,"memory_export",owner,errno,"cannot open " SHARE_CONTROL_DEVICE " for an exported descriptor");
		SPARK_FAIL(status);
	}
	memset(&exported,0,sizeof(exported));
	exported.object.type = NV0000_CTRL_OS_UNIX_EXPORT_OBJECT_TYPE_RM;
	exported.object.data.rmObject.hDevice = gpu->device;
	exported.object.data.rmObject.hParent = gpu->device;
	exported.object.data.rmObject.hObject = handle;
	exported.fd = opened;
	status = SparkNvrmRmControl(&gpu->rm,gpu->rm.client,NV0000_CTRL_CMD_OS_UNIX_EXPORT_OBJECT_TO_FD,&exported,(uint32_t)sizeof(exported),owner);
	if ( status != SPARK_STATUS_OK )
	{
		close(opened);
		SPARK_FAIL(status);
	}
	*fd = opened;
	return(SPARK_STATUS_OK);
}

static SparkStatus ShareBytes(SparkNvrmGpu *gpu, uint32_t handle, uint64_t *bytes, const char *owner)
{
	NV0041_CTRL_SURFACE_INFO info[2];
	NV0041_CTRL_GET_SURFACE_INFO_PARAMS query;
	memset(info,0,sizeof(info));
	info[0].index = NV0041_CTRL_SURFACE_INFO_INDEX_PHYS_SIZE_LO;
	info[1].index = NV0041_CTRL_SURFACE_INFO_INDEX_PHYS_SIZE_HI;
	memset(&query,0,sizeof(query));
	query.surfaceInfoListSize = 2u;
	query.surfaceInfoList = (NvP64)(uintptr_t)info;
	SHARE_TRY(SparkNvrmRmControl(&gpu->rm,handle,NV0041_CTRL_CMD_GET_SURFACE_INFO,&query,(uint32_t)sizeof(query),owner));
	*bytes = ((uint64_t)info[1].data << 32) | (uint64_t)info[0].data;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkNvrmShareImport(SparkNvrmGpu *gpu, int fd, uint32_t *handle, uint64_t *bytes, const char *owner)
{
	NV0000_CTRL_OS_UNIX_IMPORT_OBJECT_FROM_FD_PARAMS imported;
	SparkStatus status;
	memset(&imported,0,sizeof(imported));
	imported.fd = fd;
	imported.object.type = NV0000_CTRL_OS_UNIX_EXPORT_OBJECT_TYPE_RM;
	imported.object.data.rmObject.hDevice = gpu->device;
	imported.object.data.rmObject.hParent = gpu->device;
	SHARE_TRY(SparkNvrmRmControl(&gpu->rm,gpu->rm.client,NV0000_CTRL_CMD_OS_UNIX_IMPORT_OBJECT_FROM_FD,&imported,(uint32_t)sizeof(imported),owner));
	status = imported.object.data.rmObject.hObject != 0u ? ShareBytes(gpu,imported.object.data.rmObject.hObject,bytes,owner) :
		SparkDeviceRecordError(SPARK_STATUS_IO_ERROR,"memory_import",owner,0,"the module imported the descriptor without naming a handle");
	if ( status != SPARK_STATUS_OK )
	{
		if ( imported.object.data.rmObject.hObject != 0u )
			(void)SparkNvrmRmFree(&gpu->rm,gpu->device,imported.object.data.rmObject.hObject,"free");
		SPARK_FAIL(status);
	}
	*handle = imported.object.data.rmObject.hObject;
	return(SPARK_STATUS_OK);
}
