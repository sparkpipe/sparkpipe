#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_error_site.h"
#include "runtime/device/device_backend.h"

#define PROGRAM_GROWTH 64u
#define PROGRAM_DIGEST_SEED 1469598103934665603ull
#define PROGRAM_DIGEST_PRIME 1099511628211ull

uint64_t SparkDeviceTransformApply(const SparkDeviceTransform *transform, uint64_t value, uint32_t *fits)
{
	__int128 total;
	*fits = 1u;
	if ( transform == 0 || transform->divide == 0u )
		return(value);
	total = (__int128)value * transform->multiply + transform->add;
	if ( total < 0 )
	{
		*fits = 0u;
		return(0u);
	}
	total = (total + (__int128)transform->divide - 1) / (__int128)transform->divide;
	total = total * transform->scale + transform->offset;
	if ( total < 0 || total > (__int128)UINT64_MAX )
	{
		*fits = 0u;
		return(0u);
	}
	return((uint64_t)total);
}

static SparkStatus ProgramRoom(void **items, uint32_t *capacity, uint32_t count, size_t item_bytes, const char *operation, const char *owner)
{
	void *grown;
	if ( count < *capacity )
		return(SPARK_STATUS_OK);
	grown = realloc(*items,((size_t)*capacity + PROGRAM_GROWTH) * item_bytes);
	if ( grown == 0 )
		return(SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,operation,owner,0,"out of host memory for the program"));
	*items = grown;
	*capacity += PROGRAM_GROWTH;
	return(SPARK_STATUS_OK);
}

SparkDeviceRange SparkDeviceNodeRange(const SparkDeviceNode *node, uint32_t which)
{
	SparkDeviceRange range;
	range.buffer = &node->buffers[which];
	range.offset = node->offsets[which];
	return(range);
}

static SparkStatus ProgramCopyLaunch(SparkDeviceNode *node, const char *operation, const char *owner)
{
	const uint32_t count = node->launch.binding_count;
	void *constants = 0;
	SparkDeviceBinding *bindings = 0;
	SparkDeviceBuffer *buffers = 0;
	uint32_t index;
	if ( (node->launch.constant_bytes != 0u && (constants = malloc(node->launch.constant_bytes)) == 0) ||
		(count != 0u && ((bindings = (SparkDeviceBinding *)calloc(count,sizeof(*bindings))) == 0 ||
		(buffers = (SparkDeviceBuffer *)calloc(count,sizeof(*buffers))) == 0)) )
	{
		free(constants);
		free(bindings);
		free(buffers);
		return(SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,operation,owner,0,"out of host memory for a recorded launch"));
	}
	if ( constants != 0 )
		memcpy(constants,node->launch.constants,node->launch.constant_bytes);
	for ( index = 0u; index < count; index++ )
	{
		bindings[index] = node->launch.bindings[index];
		buffers[index] = *node->launch.bindings[index].range.buffer;
		bindings[index].range.buffer = &buffers[index];
	}
	node->launch.constants = constants;
	node->launch.bindings = bindings;
	node->binding_buffers = buffers;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDeviceProgramAppend(SparkDeviceProgram *program, const SparkDeviceNode *node, const char *operation)
{
	SparkDeviceNode *stored;
	SparkStatus status = ProgramRoom((void **)&program->nodes,&program->node_capacity,program->node_count,sizeof(*program->nodes),operation,program->name);
	if ( status == SPARK_STATUS_OK && node->kind == SPARK_DEVICE_NODE_RECORD )
		status = ProgramRoom((void **)&program->records,&program->record_capacity,program->record_count,sizeof(*program->records),operation,program->name);
	if ( status != SPARK_STATUS_OK )
	{
		program->broken = 1u;
		SPARK_FAIL(status);
	}
	if ( node->kind == SPARK_DEVICE_NODE_RECORD )
		program->records[program->record_count++] = program->node_count;
	stored = &program->nodes[program->node_count];
	*stored = *node;
	if ( node->kind == SPARK_DEVICE_NODE_LAUNCH )
	{
		status = ProgramCopyLaunch(stored,operation,program->name);
		if ( status != SPARK_STATUS_OK )
		{
			program->broken = 1u;
			SPARK_FAIL(status);
		}
		program->kernels++;
		if ( node->launch.constant_bytes > program->scratch_bytes )
			program->scratch_bytes = node->launch.constant_bytes;
	}
	program->node_count++;
	return(SPARK_STATUS_OK);
}

static SparkStatus ProgramRefuse(const char *owner, const char *text)
{
	const SparkStatus status = SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"program_patch",owner,0,text);
	SPARK_FAIL(status);
}

static uint32_t ProgramWidth(uint32_t bytes)
{
	return(bytes == 1u || bytes == 2u || bytes == 4u || bytes == 8u ? 1u : 0u);
}

SparkStatus SparkDeviceProgramPlace(SparkDeviceProgram *program, const SparkDevicePatch *patch, SparkDevicePlace *place, const char *owner)
{
	const SparkDeviceNode *node = program->node_count != 0u ? &program->nodes[program->node_count - 1u] : 0;
	if ( node == 0 || node->kind != SPARK_DEVICE_NODE_LAUNCH )
		return(ProgramRefuse(owner,"a patch follows the kernel launch it changes, and the queue's last recorded operation is no launch"));
	place->node = program->node_count - 1u;
	if ( patch->target == SPARK_DEVICE_PROGRAM_TARGET_ARGUMENT )
	{
		if ( node->kernel->info.parameters == 0 || patch->argument >= node->kernel->info.parameter_count )
			return(ProgramRefuse(node->kernel->name,"the patched argument is not one of the kernel's parameters"));
		place->offset = node->kernel->info.parameters[patch->argument].offset;
		place->bytes = node->kernel->info.parameters[patch->argument].bytes;
	}
	else if ( patch->target == SPARK_DEVICE_PROGRAM_TARGET_CONSTANT )
	{
		place->offset = patch->offset;
		place->bytes = patch->bytes;
	}
	else
	{
		place->offset = 0u;
		place->bytes = 4u;
	}
	if ( ProgramWidth(place->bytes) == 0u ||
		((patch->target == SPARK_DEVICE_PROGRAM_TARGET_ARGUMENT || patch->target == SPARK_DEVICE_PROGRAM_TARGET_CONSTANT) &&
		(uint64_t)place->offset + place->bytes > node->launch.constant_bytes) )
		return(ProgramRefuse(node->kernel->name,"a patch writes 1, 2, 4 or 8 bytes inside the launch's constants"));
	return(SPARK_STATUS_OK);
}

static uint64_t ProgramMix(uint64_t digest, const void *bytes, size_t count)
{
	size_t index;
	for ( index = 0u; index < count; index++ )
		digest = (digest ^ ((const uint8_t *)bytes)[index]) * PROGRAM_DIGEST_PRIME;
	return(digest);
}

static uint64_t ProgramMixRange(uint64_t digest, SparkDeviceRange range)
{
	digest = ProgramMix(digest,&range.buffer->handle,sizeof(range.buffer->handle));
	return(ProgramMix(digest,&range.offset,sizeof(range.offset)));
}

static uint64_t ProgramMixNode(uint64_t digest, const SparkDeviceNode *node)
{
	uint32_t index;
	digest = ProgramMix(digest,&node->kind,sizeof(node->kind));
	if ( node->kind == SPARK_DEVICE_NODE_COPY || node->kind == SPARK_DEVICE_NODE_FILL )
	{
		digest = ProgramMixRange(digest,SparkDeviceNodeRange(node,0u));
		if ( node->kind == SPARK_DEVICE_NODE_COPY )
			digest = ProgramMixRange(digest,SparkDeviceNodeRange(node,1u));
		else
			digest = ProgramMix(digest,&node->value,sizeof(node->value));
		return(ProgramMix(digest,&node->bytes,sizeof(node->bytes)));
	}
	if ( node->kind != SPARK_DEVICE_NODE_LAUNCH )
		return(digest);
	digest = ProgramMix(digest,node->kernel->name,strlen(node->kernel->name));
	digest = ProgramMix(digest,&node->launch.groups,sizeof(node->launch.groups));
	digest = ProgramMix(digest,&node->launch.threads,sizeof(node->launch.threads));
	digest = ProgramMix(digest,&node->launch.shared_bytes,sizeof(node->launch.shared_bytes));
	digest = ProgramMix(digest,&node->launch.constant_bytes,sizeof(node->launch.constant_bytes));
	if ( node->launch.constant_bytes != 0u )
		digest = ProgramMix(digest,node->launch.constants,node->launch.constant_bytes);
	for ( index = 0u; index < node->launch.binding_count; index++ )
	{
		digest = ProgramMixRange(digest,node->launch.bindings[index].range);
		digest = ProgramMix(digest,&node->launch.bindings[index].bytes,sizeof(node->launch.bindings[index].bytes));
		digest = ProgramMix(digest,&node->launch.bindings[index].access,sizeof(node->launch.bindings[index].access));
	}
	return(digest);
}

uint64_t SparkDeviceProgramDigest(const SparkDeviceProgram *program)
{
	uint64_t digest = PROGRAM_DIGEST_SEED;
	uint32_t index;
	for ( index = 0u; index < program->node_count; index++ )
		digest = ProgramMixNode(digest,&program->nodes[index]);
	for ( index = 0u; index < program->place_count; index++ )
	{
		const SparkDevicePlace *place = &program->places[index];
		digest = ProgramMix(digest,&place->node,sizeof(place->node));
		digest = ProgramMix(digest,&place->value,sizeof(place->value));
		digest = ProgramMix(digest,&place->target,sizeof(place->target));
		digest = ProgramMix(digest,&place->offset,sizeof(place->offset));
		digest = ProgramMix(digest,&place->bytes,sizeof(place->bytes));
		digest = ProgramMix(digest,&place->transform,sizeof(place->transform));
	}
	return(digest);
}

void SparkDeviceProgramFree(SparkDeviceProgram *program)
{
	uint32_t index;
	for ( index = 0u; index < program->node_count; index++ )
		if ( program->nodes[index].kind == SPARK_DEVICE_NODE_LAUNCH )
		{
			free((void *)program->nodes[index].launch.constants);
			free((void *)program->nodes[index].launch.bindings);
			free(program->nodes[index].binding_buffers);
		}
	free(program->nodes);
	free(program->records);
	free(program->places);
	free(program);
}

static uint64_t ProgramWidthLimit(uint32_t bytes)
{
	return(bytes >= 8u ? UINT64_MAX : ((uint64_t)1 << (8u * bytes)) - 1u);
}

SparkStatus SparkDeviceProgramValues(const SparkDeviceProgram *program, const uint64_t *values, uint64_t *placed, const char *owner)
{
	uint32_t index;
	for ( index = 0u; index < program->place_count; index++ )
	{
		const SparkDevicePlace *place = &program->places[index];
		uint32_t fits = 0u;
		const uint64_t value = SparkDeviceTransformApply(&place->transform,values[place->value],&fits);
		const uint32_t grid = place->target >= SPARK_DEVICE_PROGRAM_TARGET_GRID_X && place->target <= SPARK_DEVICE_PROGRAM_TARGET_GRID_Z;
		if ( fits == 0u || value > ProgramWidthLimit(place->bytes) || (grid != 0u && value == 0u) )
		{
			char text[128];
			snprintf(text,sizeof(text),"patch value %u gives %llu at place %u, which does not fit its %u-byte target",place->value,
				(unsigned long long)value,index,place->bytes);
			const SparkStatus status = SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"program_run",owner,0,text);
			SPARK_FAIL(status);
		}
		placed[index] = value;
	}
	return(SPARK_STATUS_OK);
}

void SparkDeviceProgramPatchLaunch(const SparkDeviceProgram *program, uint32_t node, const uint64_t *placed, SparkDeviceLaunch *launch, uint8_t *scratch)
{
	uint32_t index;
	for ( index = 0u; index < program->place_count; index++ )
	{
		const SparkDevicePlace *place = &program->places[index];
		if ( place->node != node )
			continue;
		if ( place->target == SPARK_DEVICE_PROGRAM_TARGET_GRID_X )
			launch->groups.x = (uint32_t)placed[index];
		else if ( place->target == SPARK_DEVICE_PROGRAM_TARGET_GRID_Y )
			launch->groups.y = (uint32_t)placed[index];
		else if ( place->target == SPARK_DEVICE_PROGRAM_TARGET_GRID_Z )
			launch->groups.z = (uint32_t)placed[index];
		else
		{
			if ( launch->constants != scratch )
			{
				memcpy(scratch,launch->constants,launch->constant_bytes);
				launch->constants = scratch;
			}
			memcpy(scratch + place->offset,&placed[index],place->bytes);
		}
	}
}

SparkStatus SparkDeviceProgramReplay(SparkDeviceProgram *program, SparkDeviceQueue *queue, const uint64_t *placed)
{
	uint8_t *scratch = program->scratch_bytes != 0u ? (uint8_t *)malloc(program->scratch_bytes) : 0;
	SparkStatus status = SPARK_STATUS_OK;
	uint32_t index;
	if ( program->scratch_bytes != 0u && scratch == 0 )
		return(SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"program_run",program->name,0,"out of host memory for patched constants"));
	for ( index = 0u; status == SPARK_STATUS_OK && index < program->node_count; index++ )
	{
		const SparkDeviceNode *node = &program->nodes[index];
		SparkDeviceLaunch launch;
		switch ( node->kind )
		{
		case SPARK_DEVICE_NODE_COPY:
			status = SparkDeviceCopy(queue,SparkDeviceNodeRange(node,0u),SparkDeviceNodeRange(node,1u),node->bytes);
			break;
		case SPARK_DEVICE_NODE_FILL:
			status = SparkDeviceFill(queue,SparkDeviceNodeRange(node,0u),node->value,node->bytes);
			break;
		case SPARK_DEVICE_NODE_LAUNCH:
			launch = node->launch;
			SparkDeviceProgramPatchLaunch(program,index,placed,&launch,scratch);
			status = SparkDeviceKernelLaunch(queue,node->kernel,&launch);
			break;
		case SPARK_DEVICE_NODE_HOST:
			status = SparkDeviceQueueHostCallback(queue,node->function,node->context);
			break;
		case SPARK_DEVICE_NODE_RECORD:
			status = SparkDeviceEventRecord(node->event,queue);
			break;
		case SPARK_DEVICE_NODE_AWAIT:
			status = SparkDeviceQueueWaitEvent(queue,node->event);
			break;
		default:
			status = SparkDeviceRecordError(SPARK_STATUS_INTERNAL_ERROR,"program_run",program->name,0,"the program holds an unknown node");
			break;
		}
	}
	free(scratch);
	if ( status != SPARK_STATUS_OK )
		SPARK_FAIL(status);
	return(SPARK_STATUS_OK);
}
