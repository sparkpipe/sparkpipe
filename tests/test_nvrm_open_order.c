#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "sparkpipe/spark_device.h"
#include "runtime/device/nvrm/nvrm_gpu.h"

#if defined(SPARK_NVRM_FAKE_MODULE)

#include "nv-ioctl.h"
#include "nv-ioctl-numbers.h"
#include "nv_escape.h"
#include "nvos.h"
#include "nvstatus.h"
#include "class/cl0000.h"
#include "class/cl0080.h"
#include "ctrl/ctrl0000/ctrl0000gpu.h"

#define FAKE_GPU_ID 0x100u
#define FAKE_CLIENT 0xc1d00001u
#define FAKE_NODES 3u

typedef enum FakeNode
{
	FAKE_CONTROL = 0,
	FAKE_UVM = 1,
	FAKE_DEVICE = 2
} FakeNode;

static const char *const fake_paths[FAKE_NODES] = {"/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia0"};
static int fake_fds[FAKE_NODES] = {-1, -1, -1};
static uint32_t fake_device_registered;
static uint32_t fake_identity_refused;
static uint32_t fake_identity_answered;

int __real_open(const char *path, int flags, ...);
int __real_ioctl(int fd, unsigned long request, ...);

int __wrap_open(const char *path, int flags, ...)
{
	uint32_t node;
	mode_t mode = 0;
	if ( (flags & O_CREAT) != 0 )
	{
		va_list arguments;
		va_start(arguments,flags);
		mode = (mode_t)va_arg(arguments,int);
		va_end(arguments);
	}
	for (node = 0u; node < FAKE_NODES; node++)
		if ( strcmp(path,fake_paths[node]) == 0 )
		{
			fake_fds[node] = __real_open("/dev/null",O_RDWR | O_CLOEXEC);
			return(fake_fds[node]);
		}
	return(__real_open(path,flags,mode));
}

static uint32_t FakeStatus(void *parameters, uint32_t number)
{
	if ( number == NV_ESC_RM_ALLOC )
	{
		NVOS21_PARAMETERS *made = (NVOS21_PARAMETERS *)parameters;
		if ( made->hClass == NV01_ROOT_CLIENT )
		{
			made->hObjectNew = FAKE_CLIENT;
			return(NV_OK);
		}
		return(made->hClass == NV01_DEVICE_0 ? NV_ERR_NOT_SUPPORTED : NV_ERR_INVALID_CLASS);
	}
	if ( number == NV_ESC_RM_CONTROL )
	{
		NVOS54_PARAMETERS *control = (NVOS54_PARAMETERS *)parameters;
		if ( control->cmd != NV0000_CTRL_CMD_GPU_GET_ID_INFO_V2 )
			return(NV_ERR_NOT_SUPPORTED);
		if ( fake_device_registered == 0u )
		{
			fake_identity_refused++;
			return(NV_ERR_INVALID_ARGUMENT);
		}
		fake_identity_answered++;
		((NV0000_CTRL_GPU_GET_ID_INFO_V2_PARAMS *)(uintptr_t)control->params)->deviceInstance = 0u;
		return(NV_OK);
	}
	return(NV_OK);
}

static int FakeEscape(int fd, uint32_t number, void *parameters)
{
	if ( number == NV_ESC_IOCTL_XFER_CMD )
	{
		const nv_ioctl_xfer_t *transfer = (const nv_ioctl_xfer_t *)parameters;
		return(FakeEscape(fd,transfer->cmd,(void *)(uintptr_t)transfer->ptr));
	}
	if ( number == NV_ESC_CHECK_VERSION_STR )
		return(0);
	if ( number == NV_ESC_CARD_INFO )
	{
		nv_ioctl_card_info_t *cards = (nv_ioctl_card_info_t *)parameters;
		cards[0].valid = 1u;
		cards[0].gpu_id = FAKE_GPU_ID;
		cards[0].minor_number = 0u;
		return(0);
	}
	if ( number == NV_ESC_REGISTER_FD )
	{
		fake_device_registered = fd == fake_fds[FAKE_DEVICE] ? 1u : fake_device_registered;
		return(0);
	}
	if ( number == NV_ESC_RM_ALLOC )
	{
		((NVOS21_PARAMETERS *)parameters)->status = FakeStatus(parameters,number);
		return(0);
	}
	if ( number == NV_ESC_RM_CONTROL )
	{
		((NVOS54_PARAMETERS *)parameters)->status = FakeStatus(parameters,number);
		return(0);
	}
	if ( number == NV_ESC_RM_FREE )
	{
		((NVOS00_PARAMETERS *)parameters)->status = NV_OK;
		return(0);
	}
	errno = ENOTTY;
	return(-1);
}

int __wrap_ioctl(int fd, unsigned long request, ...)
{
	va_list arguments;
	void *parameters;
	va_start(arguments,request);
	parameters = va_arg(arguments,void *);
	va_end(arguments);
	if ( fd >= 0 && (fd == fake_fds[FAKE_CONTROL] || fd == fake_fds[FAKE_UVM] || fd == fake_fds[FAKE_DEVICE]) )
		return(FakeEscape(fd,(uint32_t)(request & 0xffu),parameters));
	return(__real_ioctl(fd,request,parameters));
}

int main(void)
{
	static SparkNvrmGpu gpu;
	const SparkStatus status = SparkNvrmGpuOpen(&gpu,0u,SPARK_NVRM_MEMORY_HOST,"open order");
	const SparkDeviceError *error = SparkDeviceLastError();
	int failures = 0;
	if ( status == SPARK_STATUS_OK || fake_identity_refused != 0u || fake_identity_answered != 1u )
	{
		printf("FAIL the GPU's identity was asked for before its device file was opened and registered (refused %u, answered %u, last error: %s %s)\n",
			fake_identity_refused,fake_identity_answered,error->operation,error->text);
		failures++;
	}
	if ( strcmp(error->operation,"alloc class 0x80") != 0 || status != SPARK_STATUS_UNSUPPORTED )
	{
		printf("FAIL the open stopped before the device allocation the fake module refuses (last error: %s %s)\n",error->operation,error->text);
		failures++;
	}
	if ( failures != 0 )
		return(1);
	printf("PASS the backend opens and registers the GPU's device file, which initializes the GPU, before asking for its identity\n");
	return(0);
}

#else

int main(void)
{
	printf("SKIP test_nvrm_open_order (needs the GNU linker's --wrap to stand in for the open kernel module)\n");
	return(0);
}

#endif
