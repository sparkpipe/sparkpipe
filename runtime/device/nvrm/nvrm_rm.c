#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include "sparkpipe/spark_error_site.h"
#include "runtime/device/device_backend.h"
#include "runtime/device/nvrm/nvrm_rm.h"
#include "runtime/device/nvrm/nvrm_version.h"
#include "nv-ioctl.h"
#include "nv_escape.h"
#include "nvos.h"
#include "class/cl0000.h"

#define RM_IOCTL_READ_WRITE 3ul
#define RM_IOCTL_SIZE_LIMIT 0x3fffu
#define RM_CONTROL_DEVICE "/dev/nvidiactl"
#define RM_UVM_DEVICE "/dev/nvidia-uvm"

static unsigned long RmNumber(uint32_t number, uint32_t bytes)
{
	return(RM_IOCTL_READ_WRITE << 30 | (unsigned long)bytes << 16 | (unsigned long)NV_IOCTL_MAGIC << 8 | number);
}

static int RmCall(int fd, unsigned long request, void *parameters)
{
	int result;
	do
		result = ioctl(fd,request,parameters);
	while ( result != 0 && errno == EINTR );
	return(result != 0 ? errno : 0);
}

SparkStatus SparkNvrmRmIoctl(int fd, uint32_t number, void *parameters, uint32_t bytes, const char *operation, const char *owner)
{
	nv_ioctl_xfer_t transfer;
	char text[96];
	int error;
	if ( bytes <= RM_IOCTL_SIZE_LIMIT )
		error = RmCall(fd,RmNumber(number,bytes),parameters);
	else
	{
		memset(&transfer,0,sizeof(transfer));
		transfer.cmd = number;
		transfer.size = bytes;
		transfer.ptr = (NvP64)(uintptr_t)parameters;
		error = RmCall(fd,RmNumber(NV_ESC_IOCTL_XFER_CMD,(uint32_t)sizeof(transfer)),&transfer);
	}
	if ( error == 0 )
		return(SPARK_STATUS_OK);
	snprintf(text,sizeof(text),"ioctl %#x of the open kernel module failed: %s",number,strerror(error));
	{
		const SparkStatus status = SparkDeviceRecordError(error == ENOMEM ? SPARK_STATUS_CAPACITY_EXCEEDED : SPARK_STATUS_IO_ERROR,operation,owner,error,text);
		SPARK_FAIL(status);
	}
}

SparkStatus SparkNvrmRmStatus(uint32_t rm_status, const char *operation, const char *owner)
{
	SparkStatus status = SPARK_STATUS_IO_ERROR;
	char text[128];
	if ( rm_status == NV_OK )
		return(SPARK_STATUS_OK);
	if ( rm_status == NV_ERR_NO_MEMORY || rm_status == NV_ERR_INSUFFICIENT_RESOURCES )
		status = SPARK_STATUS_CAPACITY_EXCEEDED;
	else if ( rm_status == NV_ERR_NOT_SUPPORTED || rm_status == NV_ERR_INVALID_CLASS )
		status = SPARK_STATUS_UNSUPPORTED;
	else if ( rm_status == NV_ERR_INVALID_ARGUMENT || rm_status == NV_ERR_INVALID_PARAMETER || rm_status == NV_ERR_INVALID_OBJECT_HANDLE ||
		rm_status == NV_ERR_INVALID_LIMIT || rm_status == NV_ERR_INVALID_OFFSET || rm_status == NV_ERR_INVALID_ADDRESS )
		status = SPARK_STATUS_INVALID_ARGUMENT;
	snprintf(text,sizeof(text),"the open kernel module answered %s (%#x)",SparkNvrmRmStatusText(rm_status),rm_status);
	status = SparkDeviceRecordError(status,operation,owner,(int32_t)rm_status,text);
	SPARK_FAIL(status);
}

static SparkStatus RmOpenNode(const char *path, int *fd, const char *owner)
{
	char text[96];
	SparkStatus status;
	*fd = open(path,O_RDWR | O_CLOEXEC);
	if ( *fd >= 0 )
		return(SPARK_STATUS_OK);
	snprintf(text,sizeof(text),"cannot open %s: %s",path,strerror(errno));
	status = SparkDeviceRecordError(errno == ENOENT ? SPARK_STATUS_NOT_FOUND : SPARK_STATUS_IO_ERROR,"initialize",owner,errno,text);
	SPARK_FAIL(status);
}

static SparkStatus RmCheckVersion(SparkNvrmRm *rm, const char *owner)
{
	nv_ioctl_rm_api_version_t check;
	char text[192];
	SparkStatus status;
	memset(&check,0,sizeof(check));
	check.cmd = NV_RM_API_VERSION_CMD_STRICT;
	snprintf(check.versionString,sizeof(check.versionString),"%s",SPARK_NVRM_MODULE_VERSION);
	if ( RmCall(rm->control,RmNumber(NV_ESC_CHECK_VERSION_STR,(uint32_t)sizeof(check)),&check) == 0 )
	{
		snprintf(rm->loaded_version,sizeof(rm->loaded_version),"%s",SPARK_NVRM_MODULE_VERSION);
		return(SPARK_STATUS_OK);
	}
	memset(&check,0,sizeof(check));
	check.cmd = NV_RM_API_VERSION_CMD_QUERY;
	if ( RmCall(rm->control,RmNumber(NV_ESC_CHECK_VERSION_STR,(uint32_t)sizeof(check)),&check) == 0 )
		snprintf(rm->loaded_version,sizeof(rm->loaded_version),"%.*s",(int)sizeof(check.versionString),check.versionString);
	else
		snprintf(rm->loaded_version,sizeof(rm->loaded_version),"%s","an unknown version");
	snprintf(text,sizeof(text),"the open kernel module is %s; this backend is built for %s only",rm->loaded_version,SPARK_NVRM_MODULE_VERSION);
	status = SparkDeviceRecordError(SPARK_STATUS_TARGET_MISMATCH,"initialize",owner,0,text);
	SPARK_FAIL(status);
}

SparkStatus SparkNvrmRmOpen(SparkNvrmRm *rm, const char *owner)
{
	NVOS21_PARAMETERS root;
	SparkStatus status;
	memset(rm,0,sizeof(*rm));
	rm->control = -1;
	rm->uvm = -1;
	status = RmOpenNode(RM_CONTROL_DEVICE,&rm->control,owner);
	if ( status == SPARK_STATUS_OK )
		status = RmCheckVersion(rm,owner);
	if ( status == SPARK_STATUS_OK )
		status = RmOpenNode(RM_UVM_DEVICE,&rm->uvm,owner);
	if ( status == SPARK_STATUS_OK )
	{
		memset(&root,0,sizeof(root));
		root.hClass = NV01_ROOT_CLIENT;
		status = SparkNvrmRmIoctl(rm->control,NV_ESC_RM_ALLOC,&root,(uint32_t)sizeof(root),"initialize",owner);
		if ( status == SPARK_STATUS_OK )
			status = SparkNvrmRmStatus(root.status,"initialize",owner);
		rm->client = root.hObjectNew;
	}
	if ( status != SPARK_STATUS_OK )
	{
		SparkNvrmRmClose(rm);
		SPARK_FAIL(status);
	}
	return(SPARK_STATUS_OK);
}

void SparkNvrmRmClose(SparkNvrmRm *rm)
{
	if ( rm->client != 0u )
	{
		NVOS00_PARAMETERS release;
		memset(&release,0,sizeof(release));
		release.hRoot = rm->client;
		release.hObjectOld = rm->client;
		(void)RmCall(rm->control,RmNumber(NV_ESC_RM_FREE,(uint32_t)sizeof(release)),&release);
	}
	if ( rm->uvm >= 0 )
		close(rm->uvm);
	if ( rm->control >= 0 )
		close(rm->control);
	rm->client = 0u;
	rm->uvm = -1;
	rm->control = -1;
}

SparkStatus SparkNvrmRmAlloc(SparkNvrmRm *rm, uint32_t parent, uint32_t class_number, void *parameters, uint32_t bytes,
	uint32_t *handle, const char *owner)
{
	NVOS21_PARAMETERS made;
	char operation[32];
	SparkStatus status;
	memset(&made,0,sizeof(made));
	made.hRoot = rm->client;
	made.hObjectParent = parent;
	made.hClass = class_number;
	made.pAllocParms = (NvP64)(uintptr_t)parameters;
	made.paramsSize = bytes;
	snprintf(operation,sizeof(operation),"alloc class %#x",class_number);
	status = SparkNvrmRmIoctl(rm->control,NV_ESC_RM_ALLOC,&made,(uint32_t)sizeof(made),operation,owner);
	if ( status == SPARK_STATUS_OK )
		status = SparkNvrmRmStatus(made.status,operation,owner);
	if ( status != SPARK_STATUS_OK )
		SPARK_FAIL(status);
	*handle = made.hObjectNew;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkNvrmRmFree(SparkNvrmRm *rm, uint32_t parent, uint32_t handle, const char *owner)
{
	NVOS00_PARAMETERS release;
	SparkStatus status;
	memset(&release,0,sizeof(release));
	release.hRoot = rm->client;
	release.hObjectParent = parent;
	release.hObjectOld = handle;
	status = SparkNvrmRmIoctl(rm->control,NV_ESC_RM_FREE,&release,(uint32_t)sizeof(release),"free",owner);
	if ( status == SPARK_STATUS_OK )
		status = SparkNvrmRmStatus(release.status,"free",owner);
	if ( status != SPARK_STATUS_OK )
		SPARK_FAIL(status);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkNvrmRmControl(SparkNvrmRm *rm, uint32_t object, uint32_t command, void *parameters, uint32_t bytes, const char *owner)
{
	NVOS54_PARAMETERS control;
	char operation[32];
	SparkStatus status;
	memset(&control,0,sizeof(control));
	control.hClient = rm->client;
	control.hObject = object;
	control.cmd = command;
	control.params = (NvP64)(uintptr_t)parameters;
	control.paramsSize = bytes;
	snprintf(operation,sizeof(operation),"control %#x",command);
	status = SparkNvrmRmIoctl(rm->control,NV_ESC_RM_CONTROL,&control,(uint32_t)sizeof(control),operation,owner);
	if ( status == SPARK_STATUS_OK )
		status = SparkNvrmRmStatus(control.status,operation,owner);
	if ( status != SPARK_STATUS_OK )
		SPARK_FAIL(status);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkNvrmRmUvm(SparkNvrmRm *rm, int fd, uint32_t command, void *parameters, uint32_t bytes, uint32_t *rm_status,
	const char *operation, const char *owner)
{
	char text[96];
	int error;
	(void)bytes;
	error = RmCall(fd >= 0 ? fd : rm->uvm,(unsigned long)command,parameters);
	if ( error != 0 )
	{
		snprintf(text,sizeof(text),"unified memory ioctl %u failed: %s",command,strerror(error));
		const SparkStatus status = SparkDeviceRecordError(SPARK_STATUS_IO_ERROR,operation,owner,error,text);
		SPARK_FAIL(status);
	}
	if ( *rm_status != NV_OK )
	{
		const SparkStatus status = SparkNvrmRmStatus(*rm_status,operation,owner);
		SPARK_FAIL(status);
	}
	return(SPARK_STATUS_OK);
}

SparkStatus SparkNvrmRmOpenDevice(SparkNvrmRm *rm, uint32_t minor, int *fd, const char *owner)
{
	nv_ioctl_register_fd_t registration;
	char path[32];
	SparkStatus status;
	snprintf(path,sizeof(path),"/dev/nvidia%u",minor);
	status = RmOpenNode(path,fd,owner);
	if ( status != SPARK_STATUS_OK )
		SPARK_FAIL(status);
	registration.ctl_fd = rm->control;
	status = SparkNvrmRmIoctl(*fd,NV_ESC_REGISTER_FD,&registration,(uint32_t)sizeof(registration),"initialize",owner);
	if ( status != SPARK_STATUS_OK )
	{
		close(*fd);
		*fd = -1;
		SPARK_FAIL(status);
	}
	return(SPARK_STATUS_OK);
}
