#pragma once

#include <stddef.h>
#include <stdint.h>

#include "sparkpipe/spark_device.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_NVRM_KERNEL_DIRECTORY_SUFFIX ".kernels"
#define SPARK_NVRM_REGISTER_FAILURE_BYTES 256u

typedef struct SparkNvrmModule SparkNvrmModule;

typedef struct SparkNvrmDim
{
	unsigned int x;
	unsigned int y;
	unsigned int z;
} SparkNvrmDim;

SparkStatus SparkNvrmRegisterModule(const char *path, SparkNvrmModule **module);
SparkStatus SparkNvrmModuleStatus(const SparkNvrmModule *module, const char **failure);
uint32_t SparkNvrmModuleKernelCount(const SparkNvrmModule *module);
SparkStatus SparkNvrmRegisterFunction(SparkNvrmModule *module, const void *host, const char *name);
SparkStatus SparkNvrmRegisterVariable(SparkNvrmModule *module, const void *host, const char *name, uint64_t bytes, uint32_t managed);
SparkStatus SparkNvrmRegisteredName(const void *host, const char **name);
SparkStatus SparkNvrmRegisteredKernel(const void *host, SparkDeviceKernel **kernel);
SparkStatus SparkNvrmRegisteredResolve(const void *host, const char **name, SparkDeviceKernel **kernel);
SparkStatus SparkNvrmRegisteredOverlay(const void *host, uint32_t *overlaid);
SparkStatus SparkNvrmOverlayReload(void);
uint32_t SparkNvrmOverlayStatsEnabled(void);
void SparkNvrmOverlayRecord(uint32_t overlay_index, uint64_t count, uint32_t in_program);
SparkStatus SparkNvrmOverlayStats(const char *name, uint64_t *direct_launches, uint64_t *program_launches);
void SparkNvrmOverlayStatsDump(void);
void SparkNvrmKernelSetOverlayIndex(SparkDeviceKernel *kernel, int32_t overlay_index);
int32_t SparkNvrmKernelOverlayIndex(const SparkDeviceKernel *kernel);
SparkStatus SparkNvrmRegisteredVariable(const void *host);
SparkStatus SparkNvrmRegisteredRelease(void);
SparkStatus SparkNvrmKernelResources(const SparkDeviceKernel *kernel, uint32_t *registers, uint32_t *stack_bytes, uint32_t *shared_bytes);

void **__cudaRegisterFatBinary(void *fat_binary);
void __cudaRegisterFatBinaryEnd(void **handle);
void __cudaUnregisterFatBinary(void **handle);
void __cudaRegisterFunction(void **handle, const char *host, char *device, const char *name, int thread_limit, void *thread, void *group,
	void *block, void *grid, int *warp);
void __cudaRegisterVar(void **handle, char *host, char *device, const char *name, int external, size_t bytes, int constant, int global);
void __cudaRegisterManagedVar(void **handle, void **host, char *device, const char *name, int external, size_t bytes, int constant, int global);
char __cudaInitModule(void **handle);
unsigned __cudaPushCallConfiguration(SparkNvrmDim grid, SparkNvrmDim block, size_t shared, void *stream);
int __cudaPopCallConfiguration(SparkNvrmDim *grid, SparkNvrmDim *block, size_t *shared, void *stream);
int __cudaGetKernel(void **kernel, const void *function);
int __cudaLaunchKernel(void *kernel, SparkNvrmDim grid, SparkNvrmDim block, void **arguments, size_t shared, void *stream);
int __cudaLaunchKernel_ptsz(void *kernel, SparkNvrmDim grid, SparkNvrmDim block, void **arguments, size_t shared, void *stream);
int cudaLaunchKernel(const void *function, SparkNvrmDim grid, SparkNvrmDim block, void **arguments, size_t shared, void *stream);

#ifdef __cplusplus
}
#endif
