#include <string.h>

#include "sparkpipe/spark_error_site.h"
#include "runtime/device/device_backend.h"
#include "runtime/device/nvrm/nvrm_launch.h"
#include "classes/compute/clcec0.h"

#define LAUNCH_COMPUTE_SUBCHANNEL 1u
#define LAUNCH_BARRIERS 1u

#define LAUNCH_TRY(call) \
	do { \
		const SparkStatus launch_status = (call); \
		if ( launch_status != SPARK_STATUS_OK ) \
			SPARK_FAIL(launch_status); \
	} while (0)

SparkStatus SparkNvrmProgramLoad(SparkNvrmGpu *gpu, const uint8_t *blob, uint64_t bytes, SparkNvrmProgram *program, const char *owner)
{
	SparkStatus status;
	memset(program,0,sizeof(*program));
	program->code.map_fd = -1;
	LAUNCH_TRY(SparkNvrmCubinParse(blob,bytes,owner,&program->cubin));
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

void SparkNvrmProgramRelease(SparkNvrmGpu *gpu, SparkNvrmProgram *program)
{
	if ( program->code.bytes != 0u )
		SparkNvrmMemoryRelease(gpu,&program->code);
	SparkNvrmCubinRelease(&program->cubin);
	memset(program,0,sizeof(*program));
	program->code.map_fd = -1;
}

uint32_t SparkNvrmLaunchSlotBytes(const SparkNvrmCubinKernel *kernel)
{
	return(SPARK_NVRM_LAUNCH_QMD_BYTES + (uint32_t)((kernel->bank[0].bytes + SPARK_NVRM_BANK_ALIGN - 1u) / SPARK_NVRM_BANK_ALIGN * SPARK_NVRM_BANK_ALIGN));
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
	shape->barriers = LAUNCH_BARRIERS;
	shape->shared_bytes = kernel->shared_bytes + request->dynamic_shared_bytes;
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

SparkStatus SparkNvrmLaunchQueue(SparkNvrmGpu *gpu, const SparkNvrmProgram *program, const SparkNvrmLaunchRequest *request, const char *owner)
{
	SparkNvrmLaunchShape shape;
	SparkNvrmQmd qmd;
	uint32_t word;
	if ( request->kernel == 0 || request->slot == 0 || (request->slot_address & (SPARK_NVRM_QMD_ALIGN - 1u)) != 0u ||
		request->slot_bytes < SparkNvrmLaunchSlotBytes(request->kernel) )
	{
		const SparkStatus status = SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"kernel_launch",owner,0,
			"a launch needs a kernel and a 256-byte aligned slot for its descriptor and constant bank 0");
		SPARK_FAIL(status);
	}
	if ( gpu->local_bytes_per_thread < request->kernel->stack_bytes + SPARK_NVRM_LOCAL_EXTRA_BYTES )
	{
		const SparkStatus status = SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"kernel_launch",owner,0,
			"the kernel needs more local memory than its program provisioned");
		SPARK_FAIL(status);
	}
	LAUNCH_TRY(SparkNvrmBank0Build(request->kernel,gpu->shared_window,gpu->local_window,request->arguments,request->argument_bytes,
		request->slot + SPARK_NVRM_LAUNCH_QMD_BYTES,request->slot_bytes - SPARK_NVRM_LAUNCH_QMD_BYTES,owner));
	LaunchShape(gpu,program,request,&shape);
	LAUNCH_TRY(SparkNvrmQmdBuild(&shape,&qmd,owner));
	memcpy(request->slot,qmd.word,sizeof(qmd.word));
	LAUNCH_TRY(SparkNvrmPushComputeState(gpu,owner));
	word = (uint32_t)(request->slot_address >> 8);
	LAUNCH_TRY(SparkNvrmPush(&gpu->compute,LAUNCH_COMPUTE_SUBCHANNEL,NVCEC0_SEND_PCAS_A,&word,1u,owner));
	word = NVCEC0_SEND_SIGNALING_PCAS2_B_PCAS_ACTION_PREFETCH_SCHEDULE;
	return(SparkNvrmPush(&gpu->compute,LAUNCH_COMPUTE_SUBCHANNEL,NVCEC0_SEND_SIGNALING_PCAS2_B,&word,1u,owner));
}
