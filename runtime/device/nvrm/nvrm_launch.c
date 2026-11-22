#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_error_site.h"
#include "runtime/device/device_backend.h"
#include "runtime/device/nvrm/nvrm_launch.h"
#include "runtime/device/nvrm/nvrm_occupancy.h"
#include "classes/compute/clcec0.h"

#define LAUNCH_COMPUTE_SUBCHANNEL 1u
#define LAUNCH_BARRIERS 1u
#define LAUNCH_PATH_BYTES 4096u
#define LAUNCH_CUBIN_SUFFIX ".cubin"
#define LAUNCH_MANIFEST_SUFFIX ".manifest"

#define LAUNCH_TRY(call) \
	do { \
		const SparkStatus launch_status = (call); \
		if ( launch_status != SPARK_STATUS_OK ) \
			SPARK_FAIL(launch_status); \
	} while (0)

static SparkStatus LaunchManifest(SparkNvrmProgram *program, const uint8_t *blob, uint64_t bytes, const char *manifest, uint64_t manifest_bytes,
	const char *owner)
{
	uint32_t index;
	LAUNCH_TRY(SparkNvrmManifestParse(manifest,manifest_bytes,owner,&program->manifest));
	LAUNCH_TRY(SparkNvrmManifestCheck(&program->manifest,blob,bytes,&program->cubin,owner));
	for (index = 0u; index < program->cubin.kernel_count; index++)
	{
		const SparkNvrmManifestKernel *reported = SparkNvrmManifestKernelNamed(&program->manifest,program->cubin.kernels[index].name);
		program->cubin.kernels[index].barriers = reported->barriers;
		program->cubin.kernels[index].frame_bytes = reported->stack_bytes;
		program->cubin.kernels[index].clustered = reported->clustered;
		memcpy(program->cubin.kernels[index].cluster,reported->cluster,sizeof(reported->cluster));
		program->cubin.kernels[index].cluster_limit = reported->cluster_limit;
	}
	return(SPARK_STATUS_OK);
}

SparkStatus SparkNvrmProgramLoad(SparkNvrmGpu *gpu, const uint8_t *blob, uint64_t bytes, const char *manifest, uint64_t manifest_bytes,
	SparkNvrmProgram *program, const char *owner)
{
	SparkStatus status;
	memset(program,0,sizeof(*program));
	program->code.map_fd = -1;
	LAUNCH_TRY(SparkNvrmCubinParse(blob,bytes,owner,&program->cubin));
	status = LaunchManifest(program,blob,bytes,manifest,manifest_bytes,owner);
	if ( status == SPARK_STATUS_OK )
		status = SparkNvrmMemoryAllocate(gpu,program->cubin.image_bytes + SPARK_NVRM_PROGRAM_PAD,SPARK_NVRM_MEMORY_HOST,&program->code,owner);
	if ( status == SPARK_STATUS_OK )
	{
		uint32_t index,stack = 0u;
		memcpy(program->code.cpu,program->cubin.image,(size_t)program->cubin.image_bytes);
		status = SparkNvrmCubinRelocate(&program->cubin,program->code.address,program->code.cpu,owner);
		for (index = 0u; index < program->cubin.kernel_count; index++)
			stack = program->cubin.kernels[index].stack_bytes > stack ? program->cubin.kernels[index].stack_bytes : stack;
		if ( status == SPARK_STATUS_OK )
			status = SparkNvrmEnsureLocalMemory(gpu,stack + SPARK_NVRM_LOCAL_EXTRA_BYTES,owner);
	}
	if ( status != SPARK_STATUS_OK )
	{
		SparkNvrmProgramRelease(gpu,program);
		SPARK_FAIL(status);
	}
	return(SPARK_STATUS_OK);
}

uint8_t *SparkNvrmReadFile(const char *path, uint64_t *bytes)
{
	FILE *file = fopen(path,"rb");
	uint8_t *data = 0;
	long length = -1;
	if ( file == 0 )
		return(0);
	if ( fseek(file,0,SEEK_END) == 0 )
		length = ftell(file);
	if ( length >= 0 && fseek(file,0,SEEK_SET) == 0 )
		data = (uint8_t *)malloc((size_t)length + 1u);
	if ( data != 0 && fread(data,1u,(size_t)length,file) != (size_t)length )
	{
		free(data);
		data = 0;
	}
	fclose(file);
	if ( data != 0 )
	{
		data[length] = 0u;
		*bytes = (uint64_t)length;
	}
	return(data);
}

static SparkStatus LaunchMissing(const char *path, const char *what, const char *owner)
{
	char text[128];
	snprintf(text,sizeof(text),"cannot read %s %.80s",what,path);
	{
		const SparkStatus status = SparkDeviceRecordError(SPARK_STATUS_NOT_FOUND,"library_load",owner,0,text);
		SPARK_FAIL(status);
	}
}

SparkStatus SparkNvrmProgramLoadFile(SparkNvrmGpu *gpu, const char *path, SparkNvrmProgram *program, const char *owner)
{
	char manifest_path[LAUNCH_PATH_BYTES];
	const size_t length = strlen(path),suffix = sizeof(LAUNCH_CUBIN_SUFFIX) - 1u;
	uint64_t bytes = 0u,manifest_bytes = 0u;
	uint8_t *blob,*manifest;
	SparkStatus status;
	if ( length <= suffix || strcmp(path + length - suffix,LAUNCH_CUBIN_SUFFIX) != 0 || length - suffix + sizeof(LAUNCH_MANIFEST_SUFFIX) > sizeof(manifest_path) )
	{
		status = SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"library_load",owner,0,"a kernel image is a .cubin file with its .manifest beside it");
		SPARK_FAIL(status);
	}
	snprintf(manifest_path,sizeof(manifest_path),"%.*s%s",(int)(length - suffix),path,LAUNCH_MANIFEST_SUFFIX);
	blob = SparkNvrmReadFile(path,&bytes);
	if ( blob == 0 )
		return(LaunchMissing(path,"the kernel image",owner));
	manifest = SparkNvrmReadFile(manifest_path,&manifest_bytes);
	if ( manifest == 0 )
	{
		free(blob);
		return(LaunchMissing(manifest_path,"the kernel image's manifest",owner));
	}
	status = SparkNvrmProgramLoad(gpu,blob,bytes,(const char *)manifest,manifest_bytes,program,owner);
	free(manifest);
	free(blob);
	if ( status != SPARK_STATUS_OK )
		SPARK_FAIL(status);
	return(SPARK_STATUS_OK);
}

void SparkNvrmProgramRelease(SparkNvrmGpu *gpu, SparkNvrmProgram *program)
{
	if ( program->code.bytes != 0u )
		SparkNvrmMemoryRelease(gpu,&program->code);
	SparkNvrmManifestRelease(&program->manifest);
	SparkNvrmCubinRelease(&program->cubin);
	memset(program,0,sizeof(*program));
	program->code.map_fd = -1;
}

SparkStatus SparkNvrmKernelRunnable(const SparkNvrmCubinKernel *kernel, const char *owner)
{
	char text[160];
	if ( kernel->stack_bytes <= kernel->frame_bytes )
		return(SPARK_STATUS_OK);
	snprintf(text,sizeof(text),"%.40s: %u of its %u stack bytes are frames of functions it calls, which the backend cannot run yet",kernel->name,
		kernel->stack_bytes - kernel->frame_bytes,kernel->stack_bytes);
	{
		const SparkStatus status = SparkDeviceRecordError(SPARK_STATUS_UNSUPPORTED,"kernel_launch",owner,0,text);
		SPARK_FAIL(status);
	}
}

uint32_t SparkNvrmLaunchSlotBytes(const SparkNvrmCubinKernel *kernel)
{
	return(SPARK_NVRM_LAUNCH_QMD_BYTES + (uint32_t)((kernel->bank[0].bytes + SPARK_NVRM_BANK_ALIGN - 1u) / SPARK_NVRM_BANK_ALIGN * SPARK_NVRM_BANK_ALIGN));
}

static uint32_t LaunchResidentShared(const SparkNvrmGpu *gpu, const SparkNvrmCubinKernel *kernel, const SparkNvrmLaunchRequest *request, uint32_t shared)
{
	SparkNvrmOccupancyLimits limits;
	const uint64_t threads = (uint64_t)request->block[0] * request->block[1] * request->block[2];
	uint64_t resident;
	if ( shared == 0u || threads == 0u || threads > UINT32_MAX )
		return(shared);
	limits.warps_per_sm = gpu->warps_per_sm;
	limits.registers_per_sm = SPARK_NVRM_OCCUPANCY_REGISTERS_PER_SM;
	limits.blocks_per_sm = SPARK_NVRM_OCCUPANCY_BLOCKS_PER_SM;
	limits.shared_per_sm = SPARK_NVRM_OCCUPANCY_SHARED_PER_SM;
	limits.register_unit = SPARK_NVRM_OCCUPANCY_REGISTER_UNIT;
	limits.shared_unit = SPARK_NVRM_OCCUPANCY_SHARED_UNIT;
	resident = (uint64_t)SparkNvrmOccupancy(&limits,(uint32_t)threads,kernel->registers,0u) *
		((shared + SPARK_NVRM_OCCUPANCY_SHARED_UNIT - 1u) / SPARK_NVRM_OCCUPANCY_SHARED_UNIT * SPARK_NVRM_OCCUPANCY_SHARED_UNIT);
	return(resident < SPARK_NVRM_OCCUPANCY_SHARED_PER_SM ? (uint32_t)resident : SPARK_NVRM_OCCUPANCY_SHARED_PER_SM);
}

static void LaunchShape(const SparkNvrmGpu *gpu, const SparkNvrmProgram *program, const SparkNvrmLaunchRequest *request, SparkNvrmLaunchShape *shape)
{
	const SparkNvrmCubinKernel *kernel = request->kernel;
	uint32_t index;
	memset(shape,0,sizeof(*shape));
	shape->program_address = program->code.address + kernel->code_offset;
	shape->program_bytes = kernel->code_bytes;
	memcpy(shape->grid,request->grid,sizeof(shape->grid));
	memcpy(shape->block,request->block,sizeof(shape->block));
	shape->registers = kernel->registers;
	shape->barriers = kernel->barriers > LAUNCH_BARRIERS ? kernel->barriers : LAUNCH_BARRIERS;
	shape->shared_bytes = kernel->shared_bytes + request->dynamic_shared_bytes;
	shape->resident_shared_bytes = LaunchResidentShared(gpu,kernel,request,shape->shared_bytes);
	shape->local_bytes = gpu->local_bytes_per_thread;
	shape->sass_version = gpu->sass_version;
	shape->bank_address[0] = request->slot_address + SPARK_NVRM_LAUNCH_QMD_BYTES;
	shape->bank_bytes[0] = request->slot_bytes - SPARK_NVRM_LAUNCH_QMD_BYTES;
	shape->bank_mask = 1u;
	for (index = 1u; index < SPARK_NVRM_CUBIN_BANKS; index++)
		if ( kernel->bank[index].present != 0u )
		{
			shape->bank_address[index] = program->code.address + kernel->bank[index].image_offset;
			shape->bank_bytes[index] = (uint32_t)kernel->bank[index].bytes;
			shape->bank_mask |= 1u << index;
		}
	shape->release_address = request->release_address;
	shape->release_payload = request->release_payload;
}

static SparkStatus LaunchRefuse(const char *owner, const char *text)
{
	const SparkStatus status = SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"kernel_launch",owner,0,text);
	SPARK_FAIL(status);
}

SparkStatus SparkNvrmLaunchClusterShape(const SparkNvrmLaunchRequest *request, uint32_t cluster[3], const char *owner)
{
	const SparkNvrmCubinKernel *kernel = request->kernel;
	const uint32_t asked = request->cluster[0] != 0u || request->cluster[1] != 0u || request->cluster[2] != 0u ? 1u : 0u;
	const uint32_t limit = kernel->cluster_limit != 0u && kernel->cluster_limit < SPARK_NVRM_CLUSTER_GROUPS_MAX ? kernel->cluster_limit :
		SPARK_NVRM_CLUSTER_GROUPS_MAX;
	char text[192];
	uint64_t groups;
	uint32_t axis;
	memset(cluster,0,3u * sizeof(uint32_t));
	if ( asked != 0u && (request->cluster[0] == 0u || request->cluster[1] == 0u || request->cluster[2] == 0u) )
		return(LaunchRefuse(owner,"a cluster shape has three nonzero sizes"));
	if ( asked != 0u && kernel->cluster[0] != 0u && memcmp(request->cluster,kernel->cluster,sizeof(kernel->cluster)) != 0 )
	{
		snprintf(text,sizeof(text),"%.40s was compiled for clusters of %ux%ux%u blocks, not %ux%ux%u",kernel->name,kernel->cluster[0],kernel->cluster[1],
			kernel->cluster[2],request->cluster[0],request->cluster[1],request->cluster[2]);
		return(LaunchRefuse(owner,text));
	}
	if ( asked != 0u )
		memcpy(cluster,request->cluster,3u * sizeof(uint32_t));
	else if ( kernel->cluster[0] != 0u )
		memcpy(cluster,kernel->cluster,3u * sizeof(uint32_t));
	else if ( kernel->clustered != 0u )
	{
		snprintf(text,sizeof(text),"%.40s runs in clusters; launch it with a cluster shape",kernel->name);
		return(LaunchRefuse(owner,text));
	}
	else
		return(SPARK_STATUS_OK);
	for ( axis = 0u; axis < 3u; axis++ )
		if ( request->grid[axis] % cluster[axis] != 0u )
			return(LaunchRefuse(owner,"a cluster launch's grid is a whole number of clusters on every axis"));
	groups = (uint64_t)cluster[0] * cluster[1] * cluster[2];
	if ( groups > limit )
	{
		snprintf(text,sizeof(text),"a cluster of %llu blocks is over the limit of %u for %.40s",(unsigned long long)groups,limit,kernel->name);
		return(LaunchRefuse(owner,text));
	}
	return(SPARK_STATUS_OK);
}

SparkStatus SparkNvrmLaunchQueue(SparkNvrmGpu *gpu, SparkNvrmChannel *channel, const SparkNvrmProgram *program, const SparkNvrmLaunchRequest *request,
	const char *owner)
{
	SparkNvrmLaunchShape shape;
	SparkNvrmQmd qmd;
	uint32_t word,cluster[3];
	if ( request->kernel == 0 || request->slot == 0 || (request->slot_address & (SPARK_NVRM_QMD_ALIGN - 1u)) != 0u ||
		request->slot_bytes < SparkNvrmLaunchSlotBytes(request->kernel) )
	{
		const SparkStatus status = SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"kernel_launch",owner,0,
			"a launch needs a kernel and a 256-byte aligned slot for its descriptor and constant bank 0");
		SPARK_FAIL(status);
	}
	LAUNCH_TRY(SparkNvrmKernelRunnable(request->kernel,owner));
	LAUNCH_TRY(SparkNvrmLaunchClusterShape(request,cluster,owner));
	if ( gpu->local_bytes_per_thread < request->kernel->stack_bytes + SPARK_NVRM_LOCAL_EXTRA_BYTES )
	{
		const SparkStatus status = SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"kernel_launch",owner,0,
			"the kernel needs more local memory than its program provisioned");
		SPARK_FAIL(status);
	}
	LAUNCH_TRY(SparkNvrmBindLocalMemory(gpu,channel,owner));
	LAUNCH_TRY(SparkNvrmBank0Build(request->kernel,gpu->shared_window,gpu->local_window,request->grid,request->block,request->arguments,request->argument_bytes,
		request->slot + SPARK_NVRM_LAUNCH_QMD_BYTES,request->slot_address + SPARK_NVRM_LAUNCH_QMD_BYTES,request->slot_bytes - SPARK_NVRM_LAUNCH_QMD_BYTES,
		owner));
	LaunchShape(gpu,program,request,&shape);
	memcpy(shape.cluster,cluster,sizeof(shape.cluster));
	LAUNCH_TRY(SparkNvrmQmdBuild(&shape,&qmd,owner));
	memcpy(request->slot,qmd.word,sizeof(qmd.word));
	word = (uint32_t)(request->slot_address >> 8);
	LAUNCH_TRY(SparkNvrmPush(channel,LAUNCH_COMPUTE_SUBCHANNEL,NVCEC0_SEND_PCAS_A,&word,1u,owner));
	word = NVCEC0_SEND_SIGNALING_PCAS2_B_PCAS_ACTION_PREFETCH_SCHEDULE;
	return(SparkNvrmPush(channel,LAUNCH_COMPUTE_SUBCHANNEL,NVCEC0_SEND_SIGNALING_PCAS2_B,&word,1u,owner));
}

SparkStatus SparkNvrmLaunch(SparkNvrmGpu *gpu, SparkNvrmChannel *channel, const SparkNvrmProgram *program, const SparkNvrmLaunchRequest *request,
	uint64_t *value, const char *owner)
{
	SparkNvrmLaunchRequest ordered = *request;
	const uint64_t bound = channel->local_address;
	SparkStatus status;
	ordered.release_address = channel->fence.address;
	ordered.release_payload = channel->fence_value + 1u;
	LAUNCH_TRY(SparkNvrmOpBegin(channel,SPARK_NVRM_ENGINE_COMPUTE,owner));
	status = SparkNvrmLaunchQueue(gpu,channel,program,&ordered,owner);
	if ( status == SPARK_STATUS_OK )
		status = SparkNvrmOpEnd(gpu,channel,SPARK_NVRM_ENGINE_COMPUTE,value,owner);
	if ( status != SPARK_STATUS_OK )
	{
		SparkNvrmOpAbandon(channel);
		channel->local_address = bound;
		SPARK_FAIL(status);
	}
	return(SPARK_STATUS_OK);
}
