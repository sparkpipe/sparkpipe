#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "sparkpipe/spark_device.h"
#include "sparkpipe/spark_gpu.h"
#include "runtime/launch.h"

#define ARGUMENTS_BYTES 256u
#define ARGUMENTS_SLOTS 16u

typedef struct ArgumentBits
{
	uint32_t low : 5;
	uint32_t high : 27;
	uint16_t lanes : 9;
	int8_t sign : 4;
} ArgumentBits;

typedef struct ArgumentShape
{
	uint32_t rows;
	uint16_t lanes;
	uint8_t flag;
	double weight;
} ArgumentShape;

typedef struct ArgumentRecord
{
	const void *function;
	uint32_t grid[3];
	uint32_t block[3];
	size_t shared;
	uint32_t count;
	uint32_t bytes;
	uint8_t packed[ARGUMENTS_BYTES];
	SparkDeviceParameter layout[ARGUMENTS_SLOTS];
	uint32_t launches;
} ArgumentRecord;

static ArgumentRecord argument_record;
static uint32_t argument_failures;

__global__ void ArgumentConverted(uint32_t low, uint64_t high, uint16_t lanes, const float *table, double scale, uint8_t small, ArgumentShape shape,
	int8_t sign, int32_t nibble)
{
	(void)low;
	(void)high;
	(void)lanes;
	(void)table;
	(void)scale;
	(void)small;
	(void)shape;
	(void)sign;
	(void)nibble;
}

template<uint32_t Step>
__global__ void ArgumentStep(uint32_t *output, float base)
{
	(void)output;
	(void)base;
}

extern "C" SparkGpuError SparkNvrmLaunchPacked(const void *function, const uint32_t grid[3], const uint32_t block[3], size_t shared_bytes,
	SparkGpuStream stream, const void *parameters, const SparkDeviceParameter *layout, uint32_t count, uint32_t bytes)
{
	(void)stream;
	memset(&argument_record,0,sizeof(argument_record));
	argument_record.function = function;
	memcpy(argument_record.grid,grid,sizeof(argument_record.grid));
	memcpy(argument_record.block,block,sizeof(argument_record.block));
	argument_record.shared = shared_bytes;
	argument_record.count = count;
	argument_record.bytes = bytes;
	if ( count <= ARGUMENTS_SLOTS && bytes <= ARGUMENTS_BYTES )
	{
		memcpy(argument_record.packed,parameters,bytes);
		memcpy(argument_record.layout,layout,(size_t)count * sizeof(*layout));
	}
	argument_record.launches++;
	return(SPARK_GPU_SUCCESS);
}

static void Check(int condition, const char *what)
{
	if ( condition )
		return;
	printf("FAIL %s\n",what);
	argument_failures++;
}

template<typename Value>
static int PackedAt(uint32_t index, uint32_t offset, Value expected)
{
	Value seen;
	if ( index >= argument_record.count || argument_record.layout[index].offset != offset || argument_record.layout[index].bytes != sizeof(Value) ||
		argument_record.layout[index].align != alignof(Value) )
		return(0);
	memcpy(&seen,argument_record.packed + offset,sizeof(Value));
	return(memcmp(&seen,&expected,sizeof(Value)) == 0);
}

static int ShapeAt(uint32_t index, uint32_t offset, const ArgumentShape *expected)
{
	ArgumentShape seen;
	if ( index >= argument_record.count || argument_record.layout[index].offset != offset || argument_record.layout[index].bytes != sizeof(ArgumentShape) ||
		argument_record.layout[index].align != alignof(ArgumentShape) )
		return(0);
	memcpy(&seen,argument_record.packed + offset,sizeof(seen));
	return(seen.rows == expected->rows && seen.lanes == expected->lanes && seen.flag == expected->flag && seen.weight == expected->weight);
}

static void TestConversions(void)
{
	ArgumentBits bits;
	ArgumentShape shape;
	memset(&bits,0,sizeof(bits));
	memset(&shape,0,sizeof(shape));
	bits.low = 19u;
	bits.high = 0x5abcdefu;
	bits.lanes = 300u;
	bits.sign = -3;
	shape.rows = 7u;
	shape.lanes = 3u;
	shape.flag = 1u;
	shape.weight = 0.25;
	LM_LAUNCH((ArgumentConverted),dim3(2u,3u),dim3(32u,2u),128u,(SparkGpuStream)0,bits.low,bits.high,bits.lanes,0,3,'x',shape,-2,bits.sign);
	Check(argument_record.launches == 1u && argument_record.function == (const void *)ArgumentConverted,"the launch reaches the packed entry with its kernel");
	Check(argument_record.grid[0] == 2u && argument_record.grid[1] == 3u && argument_record.grid[2] == 1u && argument_record.block[0] == 32u &&
		argument_record.block[1] == 2u && argument_record.block[2] == 1u && argument_record.shared == 128u,"the launch keeps its grid, block and shared memory");
	Check(argument_record.count == 9u && argument_record.bytes == 72u,"nine parameters laid out in 72 bytes");
	Check(PackedAt<uint32_t>(0u,0u,19u),"a 5-bit field reaches a uint32_t parameter by value");
	Check(PackedAt<uint64_t>(1u,8u,0x5abcdefull),"a 27-bit field widens to a uint64_t parameter at its alignment");
	Check(PackedAt<uint16_t>(2u,16u,(uint16_t)300u),"a 9-bit uint16_t field reaches a uint16_t parameter");
	Check(PackedAt<const float *>(3u,24u,(const float *)0),"a literal 0 becomes a null pointer parameter");
	Check(PackedAt<double>(4u,32u,3.0),"an int argument converts to a double parameter");
	Check(PackedAt<uint8_t>(5u,40u,(uint8_t)'x'),"a char argument converts to a uint8_t parameter");
	Check(ShapeAt(6u,48u,&shape),"a struct passes by value at its own alignment");
	Check(PackedAt<int8_t>(7u,64u,(int8_t)-2),"a negative int converts to an int8_t parameter");
	Check(PackedAt<int32_t>(8u,68u,(int32_t)-3),"a signed 4-bit field sign-extends into an int32_t parameter");
}

static void TestTemplate(void)
{
	LM_LAUNCH((ArgumentStep<5u>),1,64,0,(SparkGpuStream)0,0,7);
	Check(argument_record.launches == 1u && argument_record.function == (const void *)ArgumentStep<5u>,"a template kernel launches by its template-id");
	Check(argument_record.count == 2u && argument_record.bytes == 12u,"two parameters laid out in 12 bytes");
	Check(PackedAt<uint32_t *>(0u,0u,(uint32_t *)0),"a literal 0 becomes a null output pointer");
	Check(PackedAt<float>(1u,8u,7.0f),"an int argument converts to a float parameter");
}

int main(void)
{
	TestConversions();
	TestTemplate();
	if ( argument_failures != 0u )
	{
		printf("FAIL (%u)\n",argument_failures);
		return(1);
	}
	printf("PASS LM_LAUNCH converts each argument to its kernel parameter's type as <<<>>> does: bit-fields, a literal 0 for a pointer, "
		"arithmetic conversions and structs by value, packed at each parameter's offset\n");
	return(0);
}
