#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <cuda.h>
#include <infiniband/verbs.h>

#include "sparkpipe/spark_hidden_transport.h"

static SparkStatus SparkPreflightCudaResultToStatus(CUresult result)
{
    if (result == CUDA_SUCCESS)
    {
        return SPARK_STATUS_OK;
    }
    if (result == CUDA_ERROR_NOT_SUPPORTED)
    {
        return SPARK_STATUS_MODULE_NOT_VALIDATED;
    }
    if (result == CUDA_ERROR_INVALID_VALUE)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_DRIVER_LOAD_ERROR;
}

static SparkStatus SparkPreflightCheckCudaAttribute(
    CUdevice device,
    CUdevice_attribute attribute)
{
    CUresult result;
    int32_t value;

    value = 0;
    result = cuDeviceGetAttribute(&value, attribute, device);
    if (result != CUDA_SUCCESS)
    {
        return SparkPreflightCudaResultToStatus(result);
    }
    return value != 0 ? SPARK_STATUS_OK : SPARK_STATUS_MODULE_NOT_VALIDATED;
}

static SparkStatus SparkPreflightGetCudaDevice(CUdevice *device_out)
{
    CUresult result;
    int32_t count;

    if (device_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    result = cuInit(0);
    if (result != CUDA_SUCCESS)
    {
        return SparkPreflightCudaResultToStatus(result);
    }
    result = cuDeviceGetCount(&count);
    if (result != CUDA_SUCCESS)
    {
        return SparkPreflightCudaResultToStatus(result);
    }
    if (count <= 0)
    {
        return SPARK_STATUS_NOT_FOUND;
    }
    result = cuDeviceGet(device_out, 0);
    if (result != CUDA_SUCCESS)
    {
        return SparkPreflightCudaResultToStatus(result);
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkPreflightMakeCudaBuffer(
    CUdevice device,
    void **host_pointer_out,
    CUdeviceptr *device_pointer_out,
    size_t *bytes_out)
{
    CUcontext context;
    CUresult result;
    long page_size;

    if (host_pointer_out == 0 || device_pointer_out == 0 || bytes_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *host_pointer_out = 0;
    *device_pointer_out = 0;
    page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    *bytes_out = (size_t)page_size * 4u;
    result = cuDevicePrimaryCtxRetain(&context, device);
    if (result != CUDA_SUCCESS)
    {
        return SparkPreflightCudaResultToStatus(result);
    }
    result = cuCtxSetCurrent(context);
    if (result != CUDA_SUCCESS)
    {
        return SparkPreflightCudaResultToStatus(result);
    }
    result = cuMemHostAlloc(
        host_pointer_out,
        *bytes_out,
        CU_MEMHOSTALLOC_DEVICEMAP);
    if (result != CUDA_SUCCESS)
    {
        return SparkPreflightCudaResultToStatus(result);
    }
    result = cuMemHostGetDevicePointer(device_pointer_out, *host_pointer_out, 0);
    if (result != CUDA_SUCCESS)
    {
        cuMemFreeHost(*host_pointer_out);
        *host_pointer_out = 0;
        return SparkPreflightCudaResultToStatus(result);
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkPreflightExportDmabuf(
    CUdeviceptr device_pointer,
    size_t bytes,
    int32_t *fd_out)
{
    CUresult result;
    int32_t fd;

    if (fd_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    fd = -1;
    result = cuMemGetHandleForAddressRange(
        &fd,
        device_pointer,
        bytes,
        CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
        0);
    if (result != CUDA_SUCCESS)
    {
        return SparkPreflightCudaResultToStatus(result);
    }
    if (fd < 0)
    {
        return SPARK_STATUS_DRIVER_LOAD_ERROR;
    }
    *fd_out = fd;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkPreflightOpenVerbs(
    struct ibv_context **context_out,
    struct ibv_pd **protection_domain_out)
{
    struct ibv_context *context;
    struct ibv_device **devices;
    struct ibv_pd *protection_domain;
    int32_t count;
    int32_t index;

    if (context_out == 0 || protection_domain_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *context_out = 0;
    *protection_domain_out = 0;
    devices = ibv_get_device_list(&count);
    if (devices == 0 || count <= 0)
    {
        return SPARK_STATUS_ROUTE_NOT_FOUND;
    }
    context = 0;
    for (index = 0; index < count; index++)
    {
        context = ibv_open_device(devices[index]);
        if (context != 0)
        {
            break;
        }
    }
    ibv_free_device_list(devices);
    if (context == 0)
    {
        return SPARK_STATUS_ROUTE_NOT_FOUND;
    }
    protection_domain = ibv_alloc_pd(context);
    if (protection_domain == 0)
    {
        ibv_close_device(context);
        return SPARK_STATUS_DRIVER_LOAD_ERROR;
    }
    *context_out = context;
    *protection_domain_out = protection_domain;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkPreflightRegisterDmabuf(
    struct ibv_pd *protection_domain,
    int32_t fd,
    size_t bytes)
{
    struct ibv_mr *memory_region;
    int32_t access_flags;

    if (protection_domain == 0 || fd < 0 || bytes == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    access_flags =
        IBV_ACCESS_LOCAL_WRITE |
        IBV_ACCESS_REMOTE_WRITE |
        IBV_ACCESS_REMOTE_READ;
    errno = 0;
    memory_region = ibv_reg_dmabuf_mr(
        protection_domain,
        0,
        bytes,
        0,
        fd,
        access_flags);
    if (memory_region == 0)
    {
        return SPARK_STATUS_DRIVER_LOAD_ERROR;
    }
    ibv_dereg_mr(memory_region);
    return SPARK_STATUS_OK;
}

static SparkStatus SparkPreflightRun(void)
{
    struct ibv_context *verbs_context;
    struct ibv_pd *protection_domain;
    CUdevice device;
    CUdeviceptr device_pointer;
    SparkStatus status;
    void *host_pointer;
    int32_t fd;
    size_t bytes;

    verbs_context = 0;
    protection_domain = 0;
    host_pointer = 0;
    device_pointer = 0;
    fd = -1;
    bytes = 0u;
    status = SparkPreflightGetCudaDevice(&device);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkPreflightCheckCudaAttribute(
        device,
        CU_DEVICE_ATTRIBUTE_HOST_ALLOC_DMA_BUF_SUPPORTED);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkPreflightMakeCudaBuffer(
        device,
        &host_pointer,
        &device_pointer,
        &bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkPreflightExportDmabuf(device_pointer, bytes, &fd);
    if (status == SPARK_STATUS_OK)
    {
        status = SparkPreflightOpenVerbs(&verbs_context, &protection_domain);
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkPreflightRegisterDmabuf(protection_domain, fd, bytes);
    }
    if (protection_domain != 0)
    {
        ibv_dealloc_pd(protection_domain);
    }
    if (verbs_context != 0)
    {
        ibv_close_device(verbs_context);
    }
    if (fd >= 0)
    {
        close(fd);
    }
    if (host_pointer != 0)
    {
        cuMemFreeHost(host_pointer);
    }
    return status;
}

int main(void)
{
    SparkStatus status;

    status = SparkPreflightRun();
    printf("cuda_host_dmabuf_verbs_preflight=%s\n",
        SparkStatusToString(status));
    printf("module=%s\n",
        SPARK_HIDDEN_TRANSPORT_CUDA_HOST_DMABUF_VERBS_MODULE_ID);
    printf("hidden_dimension=6144\n");
    printf("max_active_sequence_count=1024\n");
    return status == SPARK_STATUS_OK ? 0 : 1;
}
