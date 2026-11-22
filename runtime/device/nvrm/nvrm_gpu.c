#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include "sparkpipe/spark_error_site.h"
#include "runtime/device/device_backend.h"
#include "runtime/device/nvrm/nvrm_gpu.h"
#include "nvmisc.h"
#include "nv-ioctl.h"
#include "nv_escape.h"
#include "nvos.h"
#include "nv-unix-nvos-params-wrappers.h"
#include "uvm_linux_ioctl.h"
#include "class/cl003e.h"
#include "class/cl0070.h"
#include "class/cl0071.h"
#include "class/cl0080.h"
#include "class/cl2080.h"
#include "class/cl90f1.h"
#include "class/clc661.h"
#include "class/clc361.h"
#include "class/clc96f.h"
#include "ctrl/ctrl0000/ctrl0000gpu.h"
#include "ctrl/ctrl0080/ctrl0080gpu.h"
#include "ctrl/ctrl2080/ctrl2080gpu.h"
#include "ctrl/ctrl2080/ctrl2080gr.h"
#include "ctrl/ctrl2080/ctrl2080mc.h"
#include "classes/compute/clcec0.h"
#include "classes/dma-copy/clcab5.h"

#define GPU_CARDS 64u
#define GPU_LOW_BASE 0x1000000000ull
#define GPU_LOW_BYTES 0xf000000000ull
#define GPU_VASPACE_BASE 0x1000ull
#define GPU_VASPACE_BYTES 0x1fffffb000000ull
#define GPU_VIRTUAL_LIMIT 0x1ffffffffffffull
#define GPU_HOST_HANDLE_BASE 0x1000u
#define GPU_SHARED_WINDOW 0x729400000000ull
#define GPU_LOCAL_WINDOW 0x729300000000ull
#define GPU_MEMORY_FORMAT 6u

#define GPU_REFUSE(code, operation, owner, text) \
	do { \
		const SparkStatus gpu_status = SparkDeviceRecordError((code),(operation),(owner),0,(text)); \
		SPARK_FAIL(gpu_status); \
	} while (0)

#define GPU_TRY(call) \
	do { \
		const SparkStatus gpu_status = (call); \
		if ( gpu_status != SPARK_STATUS_OK ) \
			SPARK_FAIL(gpu_status); \
	} while (0)

static uint64_t GpuRound(uint64_t value, uint64_t unit)
{
	return((value + unit - 1u) / unit * unit);
}

static SparkStatus GpuSelectCard(SparkNvrmGpu *gpu, uint32_t ordinal, const char *owner)
{
	nv_ioctl_card_info_t *cards = (nv_ioctl_card_info_t *)calloc(GPU_CARDS,sizeof(nv_ioctl_card_info_t));
	uint32_t index,seen = 0u,found = 0u;
	SparkStatus status;
	if ( cards == 0 )
		GPU_REFUSE(SPARK_STATUS_CAPACITY_EXCEEDED,"initialize",owner,"out of host memory for the card table");
	status = SparkNvrmRmIoctl(gpu->rm.control,NV_ESC_CARD_INFO,cards,(uint32_t)(GPU_CARDS * sizeof(nv_ioctl_card_info_t)),"initialize",owner);
	for (index = 0u; status == SPARK_STATUS_OK && index < GPU_CARDS && found == 0u; index++)
	{
		if ( cards[index].valid == 0u )
			continue;
		if ( seen++ != ordinal )
			continue;
		gpu->gpu_id = cards[index].gpu_id;
		gpu->minor = cards[index].minor_number;
		found = 1u;
	}
	free(cards);
	if ( status != SPARK_STATUS_OK )
		SPARK_FAIL(status);
	if ( found == 0u )
		GPU_REFUSE(SPARK_STATUS_NOT_FOUND,"initialize",owner,"the open kernel module lists no GPU with that ordinal");
	return(SPARK_STATUS_OK);
}

uint32_t SparkNvrmGpuIntegrated(uint32_t gpu_flags)
{
	return(DRF_VAL(0000_CTRL_GPU,_ID_INFO,_SOC,gpu_flags) == NV0000_CTRL_GPU_ID_INFO_SOC_TRUE ? 1u : 0u);
}

static SparkStatus GpuIdentify(SparkNvrmGpu *gpu, const char *owner)
{
	NV0000_CTRL_GPU_GET_ID_INFO_V2_PARAMS identity;
	NV0080_ALLOC_PARAMETERS device;
	NV2080_ALLOC_PARAMETERS subdevice;
	GPU_TRY(SparkNvrmRmOpenDevice(&gpu->rm,gpu->minor,&gpu->device_fd,owner));
	memset(&identity,0,sizeof(identity));
	identity.gpuId = gpu->gpu_id;
	GPU_TRY(SparkNvrmRmControl(&gpu->rm,gpu->rm.client,NV0000_CTRL_CMD_GPU_GET_ID_INFO_V2,&identity,(uint32_t)sizeof(identity),owner));
	gpu->instance = identity.deviceInstance;
	gpu->integrated = SparkNvrmGpuIntegrated(identity.gpuFlags);
	memset(&device,0,sizeof(device));
	device.deviceId = gpu->instance;
	device.hClientShare = gpu->rm.client;
	device.vaMode = NV_DEVICE_ALLOCATION_VAMODE_OPTIONAL_MULTIPLE_VASPACES;
	GPU_TRY(SparkNvrmRmAlloc(&gpu->rm,gpu->rm.client,NV01_DEVICE_0,&device,(uint32_t)sizeof(device),&gpu->device,owner));
	memset(&subdevice,0,sizeof(subdevice));
	GPU_TRY(SparkNvrmRmAlloc(&gpu->rm,gpu->device,NV20_SUBDEVICE_0,&subdevice,(uint32_t)sizeof(subdevice),&gpu->subdevice,owner));
	return(SPARK_STATUS_OK);
}

static uint32_t GpuPick(const uint32_t *classes, uint32_t count, const uint32_t *wanted, uint32_t choices)
{
	uint32_t choice,index;
	for (choice = 0u; choice < choices; choice++)
		for (index = 0u; index < count; index++)
			if ( classes[index] == wanted[choice] )
				return(wanted[choice]);
	return(0u);
}

static SparkStatus GpuClasses(SparkNvrmGpu *gpu, const char *owner)
{
	static const uint32_t compute[] = {BLACKWELL_COMPUTE_B};
	static const uint32_t gpfifo[] = {BLACKWELL_CHANNEL_GPFIFO_A};
	static const uint32_t copy[] = {BLACKWELL_DMA_COPY_B};
	static const uint32_t usermode[] = {HOPPER_USERMODE_A};
	NV0080_CTRL_GPU_GET_CLASSLIST_PARAMS list;
	uint32_t *classes;
	SparkStatus status;
	memset(&list,0,sizeof(list));
	GPU_TRY(SparkNvrmRmControl(&gpu->rm,gpu->device,NV0080_CTRL_CMD_GPU_GET_CLASSLIST,&list,(uint32_t)sizeof(list),owner));
	classes = (uint32_t *)calloc(list.numClasses != 0u ? list.numClasses : 1u,sizeof(uint32_t));
	if ( classes == 0 )
		GPU_REFUSE(SPARK_STATUS_CAPACITY_EXCEEDED,"initialize",owner,"out of host memory for the class list");
	list.classList = (NvP64)(uintptr_t)classes;
	status = SparkNvrmRmControl(&gpu->rm,gpu->device,NV0080_CTRL_CMD_GPU_GET_CLASSLIST,&list,(uint32_t)sizeof(list),owner);
	if ( status == SPARK_STATUS_OK )
	{
		gpu->compute_class = GpuPick(classes,list.numClasses,compute,1u);
		gpu->gpfifo_class = GpuPick(classes,list.numClasses,gpfifo,1u);
		gpu->copy_class = GpuPick(classes,list.numClasses,copy,1u);
		gpu->usermode_class = GpuPick(classes,list.numClasses,usermode,1u);
	}
	free(classes);
	if ( status != SPARK_STATUS_OK )
		SPARK_FAIL(status);
	if ( gpu->compute_class == 0u || gpu->gpfifo_class == 0u || gpu->copy_class == 0u || gpu->usermode_class == 0u )
		GPU_REFUSE(SPARK_STATUS_TARGET_MISMATCH,"initialize",owner,"the GPU lacks the Blackwell compute, channel, copy or usermode class");
	return(SPARK_STATUS_OK);
}

static SparkStatus GpuGraphicsInfo(SparkNvrmGpu *gpu, const char *owner)
{
	NV2080_CTRL_GR_INFO info[5];
	NV2080_CTRL_GR_GET_INFO_PARAMS request;
	NV2080_CTRL_MC_GET_ARCH_INFO_PARAMS arch;
	memset(info,0,sizeof(info));
	info[0].index = NV2080_CTRL_GR_INFO_INDEX_LITTER_NUM_GPCS;
	info[1].index = NV2080_CTRL_GR_INFO_INDEX_LITTER_NUM_TPC_PER_GPC;
	info[2].index = NV2080_CTRL_GR_INFO_INDEX_LITTER_NUM_SM_PER_TPC;
	info[3].index = NV2080_CTRL_GR_INFO_INDEX_MAX_WARPS_PER_SM;
	info[4].index = NV2080_CTRL_GR_INFO_INDEX_SM_VERSION;
	memset(&request,0,sizeof(request));
	request.grInfoListSize = 5u;
	request.grInfoList = (NvP64)(uintptr_t)info;
	GPU_TRY(SparkNvrmRmControl(&gpu->rm,gpu->subdevice,NV2080_CTRL_CMD_GR_GET_INFO,&request,(uint32_t)sizeof(request),owner));
	gpu->gpcs = info[0].data;
	gpu->tpcs_per_gpc = info[1].data;
	gpu->sms_per_tpc = info[2].data;
	gpu->warps_per_sm = info[3].data;
	gpu->sm_version = info[4].data;
	gpu->sass_version = ((gpu->sm_version & 0xf00u) >> 4) | (gpu->sm_version & 0xfu);
	memset(&arch,0,sizeof(arch));
	GPU_TRY(SparkNvrmRmControl(&gpu->rm,gpu->subdevice,NV2080_CTRL_CMD_MC_GET_ARCH_INFO,&arch,(uint32_t)sizeof(arch),owner));
	gpu->architecture = arch.architecture;
	gpu->implementation = arch.implementation;
	if ( gpu->gpcs == 0u || gpu->tpcs_per_gpc == 0u || gpu->sms_per_tpc == 0u || gpu->warps_per_sm == 0u )
		GPU_REFUSE(SPARK_STATUS_TARGET_MISMATCH,"initialize",owner,"the GPU reported no graphics processing clusters");
	return(SPARK_STATUS_OK);
}

static SparkStatus GpuMapToCpu(SparkNvrmGpu *gpu, uint32_t handle, uint64_t offset, uint64_t bytes, uint32_t flags, uint32_t system, void *target,
	int *fd, uint8_t **cpu, const char *owner)
{
	nv_ioctl_nvos33_parameters_with_fd map;
	void *mapped;
	int map_fd = -1;
	SparkStatus status;
	if ( system != 0u )
	{
		map_fd = open("/dev/nvidiactl",O_RDWR | O_CLOEXEC);
		status = map_fd >= 0 ? SPARK_STATUS_OK : SparkDeviceRecordError(SPARK_STATUS_IO_ERROR,"map",owner,errno,"cannot open /dev/nvidiactl for a mapping");
	}
	else
		status = SparkNvrmRmOpenDevice(&gpu->rm,gpu->minor,&map_fd,owner);
	if ( status != SPARK_STATUS_OK )
		SPARK_FAIL(status);
	memset(&map,0,sizeof(map));
	map.fd = map_fd;
	map.params.hClient = gpu->rm.client;
	map.params.hDevice = gpu->device;
	map.params.hMemory = handle;
	map.params.offset = offset;
	map.params.length = bytes;
	map.params.flags = flags;
	status = SparkNvrmRmIoctl(gpu->rm.control,NV_ESC_RM_MAP_MEMORY,&map,(uint32_t)sizeof(map),"map",owner);
	if ( status == SPARK_STATUS_OK )
		status = SparkNvrmRmStatus(map.params.status,"map",owner);
	if ( status == SPARK_STATUS_OK )
	{
		mapped = mmap(target,(size_t)bytes,PROT_READ | (SPARK_NVRM_FIELD(flags,NVOS33_FLAGS_ACCESS) == NVOS33_FLAGS_ACCESS_READ_ONLY ? 0 : PROT_WRITE),
			MAP_SHARED | (target != 0 ? MAP_FIXED : 0),map_fd,0);
		if ( mapped == MAP_FAILED )
			status = SparkDeviceRecordError(SPARK_STATUS_IO_ERROR,"map",owner,errno,"cannot map device memory into the process");
		else
			*cpu = (uint8_t *)mapped;
	}
	if ( status != SPARK_STATUS_OK )
	{
		close(map_fd);
		SPARK_FAIL(status);
	}
	*fd = map_fd;
	return(SPARK_STATUS_OK);
}

static SparkStatus GpuUnifiedMemory(SparkNvrmGpu *gpu, const char *owner)
{
	UVM_INITIALIZE_PARAMS initialize;
	UVM_MM_INITIALIZE_PARAMS mm;
	NV2080_CTRL_GPU_GET_GID_INFO_PARAMS gid;
	UVM_REGISTER_GPU_PARAMS registration;
	UVM_REGISTER_GPU_VASPACE_PARAMS vaspace;
	memset(&initialize,0,sizeof(initialize));
	GPU_TRY(SparkNvrmRmUvm(&gpu->rm,-1,UVM_INITIALIZE,&initialize,(uint32_t)sizeof(initialize),&initialize.rmStatus,"initialize",owner));
	gpu->uvm_mm_fd = open("/dev/nvidia-uvm",O_RDWR | O_CLOEXEC);
	if ( gpu->uvm_mm_fd < 0 )
		GPU_REFUSE(SPARK_STATUS_IO_ERROR,"initialize",owner,"cannot open a second /dev/nvidia-uvm for the address space");
	memset(&mm,0,sizeof(mm));
	mm.uvmFd = gpu->rm.uvm;
	if ( ioctl(gpu->uvm_mm_fd,UVM_MM_INITIALIZE,&mm) != 0 )
		GPU_REFUSE(SPARK_STATUS_IO_ERROR,"initialize",owner,"the unified memory address space did not initialize");
	if ( mm.rmStatus == NV_WARN_NOTHING_TO_DO )
	{
		close(gpu->uvm_mm_fd);
		gpu->uvm_mm_fd = -1;
	}
	else if ( mm.rmStatus != NV_OK )
		GPU_TRY(SparkNvrmRmStatus(mm.rmStatus,"initialize",owner));
	memset(&gid,0,sizeof(gid));
	gid.flags = NV2080_GPU_CMD_GPU_GET_GID_FLAGS_FORMAT_BINARY;
	gid.length = 16u;
	GPU_TRY(SparkNvrmRmControl(&gpu->rm,gpu->subdevice,NV2080_CTRL_CMD_GPU_GET_GID_INFO,&gid,(uint32_t)sizeof(gid),owner));
	memcpy(gpu->uuid,gid.data,sizeof(gpu->uuid));
	memset(&registration,0,sizeof(registration));
	memcpy(registration.gpu_uuid.uuid,gpu->uuid,sizeof(gpu->uuid));
	registration.rmCtrlFd = -1;
	GPU_TRY(SparkNvrmRmUvm(&gpu->rm,-1,UVM_REGISTER_GPU,&registration,(uint32_t)sizeof(registration),&registration.rmStatus,"initialize",owner));
	gpu->uvm_registered = 1u;
	memset(&vaspace,0,sizeof(vaspace));
	memcpy(vaspace.gpuUuid.uuid,gpu->uuid,sizeof(gpu->uuid));
	vaspace.rmCtrlFd = gpu->rm.control;
	vaspace.hClient = gpu->rm.client;
	vaspace.hVaSpace = gpu->vaspace;
	GPU_TRY(SparkNvrmRmUvm(&gpu->rm,-1,UVM_REGISTER_GPU_VASPACE,&vaspace,(uint32_t)sizeof(vaspace),&vaspace.rmStatus,"initialize",owner));
	gpu->vaspace_registered = 1u;
	return(SPARK_STATUS_OK);
}

static SparkStatus GpuAddressSpace(SparkNvrmGpu *gpu, const char *owner)
{
	NV_MEMORY_VIRTUAL_ALLOCATION_PARAMS virtual_memory;
	NV_VASPACE_ALLOCATION_PARAMETERS vaspace;
	uint8_t *page = 0;
	memset(&virtual_memory,0,sizeof(virtual_memory));
	virtual_memory.limit = GPU_VIRTUAL_LIMIT;
	GPU_TRY(SparkNvrmRmAlloc(&gpu->rm,gpu->device,NV01_MEMORY_VIRTUAL,&virtual_memory,(uint32_t)sizeof(virtual_memory),&gpu->virtual_memory,owner));
	GPU_TRY(SparkNvrmRmAlloc(&gpu->rm,gpu->subdevice,gpu->usermode_class,0,0u,&gpu->usermode,owner));
	GPU_TRY(GpuMapToCpu(gpu,gpu->usermode,0u,NVC361_NV_USERMODE__SIZE,0u,0u,0,&gpu->usermode_fd,&page,owner));
	gpu->usermode_page = (volatile uint32_t *)(void *)page;
	memset(&vaspace,0,sizeof(vaspace));
	vaspace.vaBase = GPU_VASPACE_BASE;
	vaspace.vaSize = GPU_VASPACE_BYTES;
	vaspace.flags = NV_VASPACE_ALLOCATION_FLAGS_ENABLE_PAGE_FAULTING | NV_VASPACE_ALLOCATION_FLAGS_IS_EXTERNALLY_OWNED;
	GPU_TRY(SparkNvrmRmAlloc(&gpu->rm,gpu->device,FERMI_VASPACE_A,&vaspace,(uint32_t)sizeof(vaspace),&gpu->vaspace,owner));
	return(GpuUnifiedMemory(gpu,owner));
}

static uint32_t GpuReadOnly(const SparkNvrmMemory *memory)
{
	return(memory->kind == SPARK_NVRM_MEMORY_USER_READ_ONLY ? 1u : 0u);
}

static void GpuDmaUnmap(SparkNvrmGpu *gpu, uint64_t address, uint64_t bytes, uint32_t handle)
{
	NVOS47_PARAMETERS dma;
	memset(&dma,0,sizeof(dma));
	dma.hClient = gpu->rm.client;
	dma.hDevice = gpu->device;
	dma.hDma = gpu->virtual_memory;
	dma.hMemory = handle;
	dma.dmaOffset = address;
	dma.size = bytes;
	(void)SparkNvrmRmIoctl(gpu->rm.control,NV_ESC_RM_UNMAP_MEMORY_DMA,&dma,(uint32_t)sizeof(dma),"free","free");
}

static void GpuRangeFree(SparkNvrmGpu *gpu, uint64_t address, uint64_t bytes)
{
	UVM_FREE_PARAMS release;
	memset(&release,0,sizeof(release));
	release.base = address;
	release.length = bytes;
	(void)ioctl(gpu->rm.uvm,UVM_FREE,&release);
}

static SparkStatus GpuExternalMap(SparkNvrmGpu *gpu, uint64_t address, uint64_t bytes, uint32_t handle, uint64_t offset, uint32_t read_only, const char *owner)
{
	UVM_MAP_EXTERNAL_ALLOCATION_PARAMS *map = (UVM_MAP_EXTERNAL_ALLOCATION_PARAMS *)calloc(1u,sizeof(*map));
	SparkStatus status;
	if ( map == 0 )
		GPU_REFUSE(SPARK_STATUS_CAPACITY_EXCEEDED,"allocate",owner,"out of host memory for the mapping request");
	map->base = address;
	map->length = bytes;
	map->offset = offset;
	map->rmCtrlFd = gpu->rm.control;
	map->hClient = gpu->rm.client;
	map->hMemory = handle;
	map->gpuAttributesCount = 1u;
	memcpy(map->perGpuAttributes[0].gpuUuid.uuid,gpu->uuid,sizeof(gpu->uuid));
	map->perGpuAttributes[0].gpuMappingType = read_only != 0u ? UvmGpuMappingTypeReadOnly : UvmGpuMappingTypeReadWriteAtomic;
	status = SparkNvrmRmUvm(&gpu->rm,-1,UVM_MAP_EXTERNAL_ALLOCATION,map,(uint32_t)sizeof(*map),&map->rmStatus,"allocate",owner);
	free(map);
	if ( status != SPARK_STATUS_OK )
		SPARK_FAIL(status);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkNvrmGpuMap(SparkNvrmGpu *gpu, uint64_t address, uint64_t bytes, uint32_t handle, uint64_t offset, uint32_t read_only, const char *owner)
{
	UVM_CREATE_EXTERNAL_RANGE_PARAMS range;
	NVOS46_PARAMETERS dma;
	SparkStatus status;
	memset(&range,0,sizeof(range));
	range.base = address;
	range.length = bytes;
	GPU_TRY(SparkNvrmRmUvm(&gpu->rm,-1,UVM_CREATE_EXTERNAL_RANGE,&range,(uint32_t)sizeof(range),&range.rmStatus,"allocate",owner));
	memset(&dma,0,sizeof(dma));
	dma.hClient = gpu->rm.client;
	dma.hDevice = gpu->device;
	dma.hDma = gpu->virtual_memory;
	dma.hMemory = handle;
	dma.offset = offset;
	dma.length = bytes;
	dma.flags = (NvV32)(SPARK_NVRM_VALUE(NVOS46_FLAGS_PAGE_SIZE,NVOS46_FLAGS_PAGE_SIZE_4KB) |
		SPARK_NVRM_VALUE(NVOS46_FLAGS_CACHE_SNOOP,NVOS46_FLAGS_CACHE_SNOOP_ENABLE) |
		SPARK_NVRM_VALUE(NVOS46_FLAGS_DMA_OFFSET_FIXED,NVOS46_FLAGS_DMA_OFFSET_FIXED_TRUE) |
		SPARK_NVRM_VALUE(NVOS46_FLAGS_ACCESS,read_only != 0u ? NVOS46_FLAGS_ACCESS_READ_ONLY : NVOS46_FLAGS_ACCESS_READ_WRITE));
	dma.dmaOffset = address;
	status = SparkNvrmRmIoctl(gpu->rm.control,NV_ESC_RM_MAP_MEMORY_DMA,&dma,(uint32_t)sizeof(dma),"allocate",owner);
	if ( status == SPARK_STATUS_OK )
		status = SparkNvrmRmStatus(dma.status,"allocate",owner);
	if ( status == SPARK_STATUS_OK && dma.dmaOffset != address )
	{
		GpuDmaUnmap(gpu,dma.dmaOffset,bytes,handle);
		status = SparkDeviceRecordError(SPARK_STATUS_IO_ERROR,"allocate",owner,0,"the module mapped the memory at another address than the one asked for");
	}
	else if ( status == SPARK_STATUS_OK )
	{
		status = GpuExternalMap(gpu,address,bytes,handle,offset,read_only,owner);
		if ( status != SPARK_STATUS_OK )
			GpuDmaUnmap(gpu,address,bytes,handle);
	}
	if ( status != SPARK_STATUS_OK )
	{
		GpuRangeFree(gpu,address,bytes);
		SPARK_FAIL(status);
	}
	return(SPARK_STATUS_OK);
}

void SparkNvrmGpuUnmap(SparkNvrmGpu *gpu, uint64_t address, uint64_t bytes, uint32_t handle)
{
	GpuRangeFree(gpu,address,bytes);
	GpuDmaUnmap(gpu,address,bytes,handle);
}

static SparkStatus GpuUvmMap(SparkNvrmGpu *gpu, SparkNvrmMemory *memory, const char *owner)
{
	GPU_TRY(SparkNvrmGpuMap(gpu,memory->address,memory->bytes,memory->handle,0u,GpuReadOnly(memory),owner));
	memory->uvm_range = 1u;
	memory->dma_mapped = 1u;
	return(SPARK_STATUS_OK);
}

static SparkStatus GpuOsDescriptor(SparkNvrmGpu *gpu, SparkNvrmMemory *memory, const char *owner)
{
	nv_ioctl_nvos02_parameters_with_fd made;
	memset(&made,0,sizeof(made));
	made.fd = -1;
	made.params.hRoot = gpu->rm.client;
	made.params.hObjectParent = gpu->device;
	made.params.hObjectNew = gpu->host_handle_next++;
	made.params.hClass = NV01_MEMORY_SYSTEM_OS_DESCRIPTOR;
	made.params.flags = (NvV32)(SPARK_NVRM_VALUE(NVOS02_FLAGS_PHYSICALITY,NVOS02_FLAGS_PHYSICALITY_NONCONTIGUOUS) |
		SPARK_NVRM_VALUE(NVOS02_FLAGS_COHERENCY,NVOS02_FLAGS_COHERENCY_CACHED) | SPARK_NVRM_VALUE(NVOS02_FLAGS_MAPPING,NVOS02_FLAGS_MAPPING_NO_MAP) |
		SPARK_NVRM_VALUE(NVOS02_FLAGS_ALLOC_USER_READ_ONLY,GpuReadOnly(memory) != 0u ? NVOS02_FLAGS_ALLOC_USER_READ_ONLY_YES : NVOS02_FLAGS_ALLOC_USER_READ_ONLY_NO));
	made.params.pMemory = (NvP64)(uintptr_t)memory->cpu;
	made.params.limit = memory->bytes - 1u;
	GPU_TRY(SparkNvrmRmIoctl(gpu->device_fd,NV_ESC_RM_ALLOC_MEMORY,&made,(uint32_t)sizeof(made),"allocate",owner));
	GPU_TRY(SparkNvrmRmStatus(made.params.status,"allocate",owner));
	memory->handle = made.params.hObjectNew;
	return(SPARK_STATUS_OK);
}

static SparkStatus GpuHostMemory(SparkNvrmGpu *gpu, SparkNvrmMemory *memory, const char *owner)
{
	void *cpu = mmap((void *)(uintptr_t)memory->address,(size_t)memory->bytes,PROT_READ | PROT_WRITE,MAP_SHARED | MAP_ANONYMOUS | MAP_FIXED,-1,0);
	if ( cpu == MAP_FAILED )
		GPU_REFUSE(SPARK_STATUS_CAPACITY_EXCEEDED,"allocate",owner,"cannot back the device address range with host memory");
	memory->cpu = (uint8_t *)cpu;
	memory->host_mapped = 1u;
	return(GpuOsDescriptor(gpu,memory,owner));
}

int SparkNvrmModuleZeroesMemory(const char *parameters_path)
{
	char line[160];
	unsigned int value = 0u;
	int found = 0;
	FILE *parameters = parameters_path != 0 ? fopen(parameters_path,"r") : 0;
	if ( parameters == 0 )
		return(0);
	while ( fgets(line,sizeof(line),parameters) != 0 )
		if ( sscanf(line,"InitializeSystemMemoryAllocations: %u",&value) == 1 )
		{
			found = 1;
			break;
		}
	fclose(parameters);
	return(found != 0 && value == 1u ? 1 : 0);
}

uint32_t SparkNvrmSystemPageBytes(uint64_t bytes)
{
	if ( bytes >= SPARK_NVRM_HUGE_OBJECT_LIMIT )
		return(SPARK_NVRM_BIG_PAGE_BYTES);
	return(bytes >= 4u * SPARK_NVRM_HUGE_PAGE_BYTES ? SPARK_NVRM_HUGE_PAGE_BYTES : SPARK_NVRM_PAGE_BYTES);
}

SparkStatus SparkNvrmSystemAllocate(SparkNvrmGpu *gpu, uint64_t bytes, uint64_t page_bytes, SparkNvrmSystemCaching caching, uint32_t *handle, const char *owner)
{
	const uint32_t uncached = caching == SPARK_NVRM_SYSTEM_UNCACHED ? 1u : 0u,cpu_cached = caching == SPARK_NVRM_SYSTEM_CPU_CACHED ? 1u : 0u;
	NV_MEMORY_ALLOCATION_PARAMS made;
	const uint32_t huge = page_bytes == SPARK_NVRM_HUGE_PAGE_BYTES,big = page_bytes == SPARK_NVRM_BIG_PAGE_BYTES;
	if ( huge != 0u && bytes >= SPARK_NVRM_HUGE_OBJECT_LIMIT )
		GPU_REFUSE(SPARK_STATUS_INVALID_ARGUMENT,"allocate",owner,"an object of 2 MiB pages stays below 4 GiB: at 4 GiB or more its mapping misses the 64 KiB at 4 GiB - 64 KiB");
	memset(&made,0,sizeof(made));
	made.owner = gpu->rm.client;
	made.type = uncached != 0u ? NVOS32_TYPE_NOTIFIER : NVOS32_TYPE_IMAGE;
	made.flags = NVOS32_ALLOC_FLAGS_MAP_NOT_REQUIRED | NVOS32_ALLOC_FLAGS_MEMORY_HANDLE_PROVIDED | NVOS32_ALLOC_FLAGS_ALIGNMENT_FORCE |
		NVOS32_ALLOC_FLAGS_IGNORE_BANK_PLACEMENT;
	made.attr = (NvU32)(SPARK_NVRM_VALUE(NVOS32_ATTR_PHYSICALITY,NVOS32_ATTR_PHYSICALITY_ALLOW_NONCONTIGUOUS) |
		SPARK_NVRM_VALUE(NVOS32_ATTR_LOCATION,NVOS32_ATTR_LOCATION_PCI) |
		SPARK_NVRM_VALUE(NVOS32_ATTR_COHERENCY,cpu_cached != 0u ? NVOS32_ATTR_COHERENCY_CACHED : NVOS32_ATTR_COHERENCY_UNCACHED) |
		(huge != 0u ? SPARK_NVRM_VALUE(NVOS32_ATTR_PAGE_SIZE,NVOS32_ATTR_PAGE_SIZE_HUGE) : 0u) |
		(big != 0u ? SPARK_NVRM_VALUE(NVOS32_ATTR_PAGE_SIZE,NVOS32_ATTR_PAGE_SIZE_BIG) : 0u));
	made.attr2 = (NvU32)(SPARK_NVRM_VALUE(NVOS32_ATTR2_GPU_CACHEABLE,uncached != 0u || cpu_cached != 0u ? NVOS32_ATTR2_GPU_CACHEABLE_NO :
		NVOS32_ATTR2_GPU_CACHEABLE_YES) |
		SPARK_NVRM_VALUE(NVOS32_ATTR2_ZBC,NVOS32_ATTR2_ZBC_PREFER_NO_ZBC) |
		(huge != 0u ? SPARK_NVRM_VALUE(NVOS32_ATTR2_PAGE_SIZE_HUGE,NVOS32_ATTR2_PAGE_SIZE_HUGE_2MB) : 0u));
	made.format = GPU_MEMORY_FORMAT;
	made.size = bytes;
	made.alignment = page_bytes;
	made.limit = bytes - 1u;
	GPU_TRY(SparkNvrmRmAlloc(&gpu->rm,gpu->device,NV01_MEMORY_SYSTEM,&made,(uint32_t)sizeof(made),handle,owner));
	return(SPARK_STATUS_OK);
}

SparkStatus SparkNvrmSystemMapToCpu(SparkNvrmGpu *gpu, uint32_t handle, uint64_t offset, uint64_t bytes, uint32_t read_only, void *target, int *fd,
	const char *owner)
{
	uint8_t *cpu = 0;
	return(GpuMapToCpu(gpu,handle,offset,bytes,read_only != 0u ? SPARK_NVRM_VALUE(NVOS33_FLAGS_ACCESS,NVOS33_FLAGS_ACCESS_READ_ONLY) : 0u,1u,target,fd,&cpu,
		owner));
}

static SparkStatus GpuSystemMemory(SparkNvrmGpu *gpu, SparkNvrmMemory *memory, const char *owner)
{
	GPU_TRY(SparkNvrmSystemAllocate(gpu,memory->bytes,memory->page_bytes,memory->kind == SPARK_NVRM_MEMORY_SYSTEM_UNCACHED ? SPARK_NVRM_SYSTEM_UNCACHED :
		SPARK_NVRM_SYSTEM_GPU_CACHED,&memory->handle,owner));
	GPU_TRY(GpuMapToCpu(gpu,memory->handle,0u,memory->bytes,0u,1u,(void *)(uintptr_t)memory->address,&memory->map_fd,&memory->cpu,owner));
	memory->host_mapped = 1u;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkNvrmMemoryAllocate(SparkNvrmGpu *gpu, uint64_t bytes, SparkNvrmMemoryKind kind, SparkNvrmMemory *memory, const char *owner)
{
	SparkStatus status;
	memset(memory,0,sizeof(*memory));
	memory->kind = kind;
	memory->map_fd = -1;
	memory->page_bytes = kind != SPARK_NVRM_MEMORY_HOST ? SparkNvrmSystemPageBytes(bytes) : SPARK_NVRM_PAGE_BYTES;
	memory->bytes = GpuRound(bytes != 0u ? bytes : 1u,memory->page_bytes);
	if ( kind != SPARK_NVRM_MEMORY_HOST && kind != SPARK_NVRM_MEMORY_SYSTEM && kind != SPARK_NVRM_MEMORY_SYSTEM_UNCACHED )
		GPU_REFUSE(SPARK_STATUS_INVALID_ARGUMENT,"allocate",owner,"the module allocates host, system or uncached system memory");
	status = SparkNvrmVirtual(gpu,memory->bytes,memory->page_bytes,&memory->address,owner);
	memory->virtual_reserved = status == SPARK_STATUS_OK ? 1u : 0u;
	if ( status == SPARK_STATUS_OK )
		status = kind == SPARK_NVRM_MEMORY_HOST ? GpuHostMemory(gpu,memory,owner) : GpuSystemMemory(gpu,memory,owner);
	if ( status == SPARK_STATUS_OK )
		status = GpuUvmMap(gpu,memory,owner);
	if ( status != SPARK_STATUS_OK )
	{
		SparkNvrmMemoryRelease(gpu,memory);
		SPARK_FAIL(status);
	}
	return(SPARK_STATUS_OK);
}

SparkStatus SparkNvrmMemoryAttach(SparkNvrmGpu *gpu, void *host, uint64_t bytes, SparkNvrmMemoryKind kind, SparkNvrmMemory *memory, const char *owner)
{
	const uintptr_t first = (uintptr_t)host & ~(uintptr_t)(SPARK_NVRM_PAGE_BYTES - 1u);
	SparkStatus status;
	memset(memory,0,sizeof(*memory));
	memory->kind = kind;
	memory->map_fd = -1;
	if ( host == 0 || bytes == 0u || (kind != SPARK_NVRM_MEMORY_USER && kind != SPARK_NVRM_MEMORY_USER_READ_ONLY) )
		GPU_REFUSE(SPARK_STATUS_INVALID_ARGUMENT,"register",owner,"attaching needs host memory, a size and a user memory kind");
	memory->cpu = (uint8_t *)first;
	memory->page_bytes = SPARK_NVRM_PAGE_BYTES;
	memory->bytes = GpuRound((uintptr_t)host - first + bytes,SPARK_NVRM_PAGE_BYTES);
	status = SparkNvrmVirtual(gpu,memory->bytes,SPARK_NVRM_PAGE_BYTES,&memory->address,owner);
	memory->virtual_reserved = status == SPARK_STATUS_OK ? 1u : 0u;
	if ( status == SPARK_STATUS_OK )
		status = GpuOsDescriptor(gpu,memory,owner);
	if ( status == SPARK_STATUS_OK )
		status = GpuUvmMap(gpu,memory,owner);
	if ( status != SPARK_STATUS_OK )
	{
		SparkNvrmMemoryRelease(gpu,memory);
		SPARK_FAIL(status);
	}
	return(SPARK_STATUS_OK);
}

void SparkNvrmMemoryRelease(SparkNvrmGpu *gpu, SparkNvrmMemory *memory)
{
	if ( memory->uvm_range != 0u )
	{
		UVM_FREE_PARAMS release;
		memset(&release,0,sizeof(release));
		release.base = memory->address;
		release.length = memory->bytes;
		(void)ioctl(gpu->rm.uvm,UVM_FREE,&release);
	}
	if ( memory->host_mapped != 0u )
		SparkNvrmReserveCpu(gpu,(uint64_t)(uintptr_t)memory->cpu,memory->bytes);
	if ( memory->map_fd >= 0 )
		close(memory->map_fd);
	if ( memory->handle != 0u )
		(void)SparkNvrmRmFree(&gpu->rm,gpu->device,memory->handle,"free");
	if ( memory->virtual_reserved != 0u )
		SparkNvrmVirtualRelease(gpu,memory->address,memory->bytes);
	memset(memory,0,sizeof(*memory));
	memory->map_fd = -1;
}

SparkStatus SparkNvrmGpuOpen(SparkNvrmGpu *gpu, uint32_t ordinal, SparkNvrmMemoryKind control_kind, const char *owner)
{
	SparkStatus status;
	memset(gpu,0,sizeof(*gpu));
	gpu->device_fd = -1;
	gpu->uvm_mm_fd = -1;
	gpu->usermode_fd = -1;
	gpu->host_handle_next = GPU_HOST_HANDLE_BASE;
	gpu->shared_window = GPU_SHARED_WINDOW;
	gpu->local_window = GPU_LOCAL_WINDOW;
	gpu->compute.control.map_fd = -1;
	gpu->compute.notifier.map_fd = -1;
	gpu->compute.push.map_fd = -1;
	gpu->local_memory.map_fd = -1;
	status = SparkNvrmRmOpen(&gpu->rm,owner);
	if ( status == SPARK_STATUS_OK )
		status = SparkNvrmAddressOpen(gpu,GPU_LOW_BASE,GPU_LOW_BYTES,owner);
	if ( status == SPARK_STATUS_OK )
		status = GpuSelectCard(gpu,ordinal,owner);
	if ( status == SPARK_STATUS_OK )
		status = GpuIdentify(gpu,owner);
	if ( status == SPARK_STATUS_OK )
		status = GpuClasses(gpu,owner);
	if ( status == SPARK_STATUS_OK )
		status = GpuGraphicsInfo(gpu,owner);
	if ( status == SPARK_STATUS_OK )
		status = GpuAddressSpace(gpu,owner);
	if ( status == SPARK_STATUS_OK )
		status = SparkNvrmChannelOpen(gpu,control_kind,owner);
	if ( status != SPARK_STATUS_OK )
	{
		SparkNvrmGpuClose(gpu);
		SPARK_FAIL(status);
	}
	return(SPARK_STATUS_OK);
}

void SparkNvrmGpuClose(SparkNvrmGpu *gpu)
{
	uint32_t index;
	SparkNvrmChannelClose(gpu);
	if ( gpu->local_memory.bytes != 0u )
		SparkNvrmMemoryRelease(gpu,&gpu->local_memory);
	for (index = 0u; index < gpu->local_retired_count; index++)
		SparkNvrmMemoryRelease(gpu,&gpu->local_retired[index]);
	if ( gpu->usermode_page != 0 )
		munmap((void *)gpu->usermode_page,NVC361_NV_USERMODE__SIZE);
	if ( gpu->usermode_fd >= 0 )
		close(gpu->usermode_fd);
	if ( gpu->uvm_mm_fd >= 0 )
		close(gpu->uvm_mm_fd);
	if ( gpu->device_fd >= 0 )
		close(gpu->device_fd);
	SparkNvrmRmClose(&gpu->rm);
	SparkNvrmAddressClose(gpu);
	memset(gpu,0,sizeof(*gpu));
	gpu->device_fd = -1;
	gpu->uvm_mm_fd = -1;
	gpu->usermode_fd = -1;
}
