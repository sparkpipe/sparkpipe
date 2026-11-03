#include <math.h>
#include <stdio.h>
#include <string.h>

#include "sparkpipe/spark_device.h"

static float ReferenceE4m3(uint32_t code)
{
	const uint32_t exponent = (code >> 3u) & 15u, mantissa = code & 7u;
	const float magnitude = (code & 0x7fu) == 0x7fu ? NAN : exponent == 0u ? ldexpf((float)mantissa,-9) : ldexpf(1.0f + (float)mantissa / 8.0f,(int)exponent - 7);
	return((code & 0x80u) != 0u ? -magnitude : magnitude);
}

static float ReferenceDecode(uint32_t slot, uint32_t code)
{
	static const float e2m1[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
	if ( slot == 0u )
		return(ReferenceE4m3(code));
	if ( slot == 1u )
		return((code & 0x7fu) == 0x7fu ? 448.0f : ReferenceE4m3(code & 0x7fu));
	if ( slot == 2u )
		return(code == 0xffu ? NAN : ldexpf(1.0f,(int)code - 127));
	return((code & 8u) != 0u ? -e2m1[code & 7u] : e2m1[code & 7u]);
}

static int Same(float got, float want)
{
	uint32_t a, b;
	memcpy(&a,&got,sizeof(a));
	memcpy(&b,&want,sizeof(b));
	return(isnan(want) ? isnan(got) : a == b);
}

static int CheckDecoders(SparkDeviceQueue *queue, SparkDeviceLibrary *library)
{
	static const char *names[4] = { "SparkMathE4m3ToFloat", "SparkMathUe4m3ToFloat", "SparkMathUe8m0ToFloat", "SparkMathE2m1ToFloat" };
	SparkDeviceKernel *kernel = 0;
	SparkDeviceBuffer out;
	SparkDeviceBinding binding;
	SparkDeviceLaunch launch;
	uint32_t slot, code, constant = 0u, bad = 0u;
	if ( SparkDeviceKernelCreate(&kernel,library,"spark_test_weight_decode") != SPARK_STATUS_OK ||
		SparkDeviceAllocate(&out,SPARK_DEVICE_SPACE_SHARED_COHERENT,1024u * sizeof(float),"weight decode") != SPARK_STATUS_OK )
	{
		printf("FAIL weight decode setup: %s\n",SparkDeviceLastError()->text);
		return(0);
	}
	binding.range.buffer = &out;
	binding.range.offset = 0u;
	binding.bytes = out.bytes;
	binding.access = SPARK_DEVICE_BIND_WRITE;
	memset(&launch,0,sizeof(launch));
	launch.groups.x = launch.groups.y = launch.groups.z = 1u;
	launch.threads.x = 256u;
	launch.threads.y = launch.threads.z = 1u;
	launch.constants = &constant;
	launch.constant_bytes = sizeof(constant);
	launch.bindings = &binding;
	launch.binding_count = 1u;
	if ( SparkDeviceKernelLaunch(queue,kernel,&launch) != SPARK_STATUS_OK || SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) != SPARK_STATUS_OK )
	{
		printf("FAIL weight decode launch: %s\n",SparkDeviceLastError()->text);
		return(0);
	}
	for ( slot = 0u; slot < 4u; slot++ )
		for ( code = 0u; code < (slot == 3u ? 16u : 256u); code++ )
			if ( !Same(((const float *)out.host)[slot * 256u + code],ReferenceDecode(slot,code)) )
			{
				if ( bad++ < 8u )
					printf("%s(0x%02x) = %.9g, the format gives %.9g\n",names[slot],code,((const float *)out.host)[slot * 256u + code],ReferenceDecode(slot,code));
			}
	SparkDeviceFree(&out);
	if ( bad != 0u )
	{
		printf("FAIL the shared FP8 and FP4 decoders differ from their formats on %u codes\n",bad);
		return(0);
	}
	printf("PASS the shared FP8 and FP4 decoders give every E4M3, UE4M3 and UE8M0 code and every E2M1 nibble its format value on Metal\n");
	return(1);
}

int main(void)
{
	static const char *labels[4] = { "SparkMathFloatToBf16", "hardware bfloat(), non-NaN inputs", "hardware bfloat(), NaN inputs",
		"hardware bfloat() with NaN -> 0x7fff" };
	SparkDeviceQueue *queue = 0;
	SparkDeviceLibrary *library = 0;
	SparkDeviceKernel *kernel = 0;
	SparkDeviceBuffer counts, firsts;
	SparkDeviceBinding bindings[2];
	SparkDeviceLaunch launch;
	uint32_t index, constant = 0u;
	if ( SparkDeviceInitialize(0) != SPARK_STATUS_OK || strcmp(SparkDeviceGetProperties()->target,"metal.apple10") != 0 )
	{
		printf("SKIP test_metal_bf16_convert (needs an Apple family 10 GPU)\n");
		return(0);
	}
	if ( SparkDeviceQueueCreate(&queue,"bf16 sweep") != SPARK_STATUS_OK || SparkDeviceLibraryLoad(&library,SPARK_BF16_TEST_LIBRARY,"bf16 sweep") != SPARK_STATUS_OK ||
		SparkDeviceKernelCreate(&kernel,library,"spark_test_bf16_sweep") != SPARK_STATUS_OK ||
		SparkDeviceAllocate(&counts,SPARK_DEVICE_SPACE_SHARED_COHERENT,4u * sizeof(uint32_t),"bf16 sweep counts") != SPARK_STATUS_OK ||
		SparkDeviceAllocate(&firsts,SPARK_DEVICE_SPACE_SHARED_COHERENT,4u * sizeof(uint32_t),"bf16 sweep firsts") != SPARK_STATUS_OK )
	{
		printf("FAIL bf16 sweep setup: %s\n",SparkDeviceLastError()->text);
		return(1);
	}
	memset(counts.host,0,4u * sizeof(uint32_t));
	memset(firsts.host,0xff,4u * sizeof(uint32_t));
	bindings[0].range.buffer = &counts;
	bindings[0].range.offset = 0u;
	bindings[0].bytes = counts.bytes;
	bindings[0].access = SPARK_DEVICE_BIND_READ | SPARK_DEVICE_BIND_WRITE;
	bindings[1].range.buffer = &firsts;
	bindings[1].range.offset = 0u;
	bindings[1].bytes = firsts.bytes;
	bindings[1].access = SPARK_DEVICE_BIND_READ | SPARK_DEVICE_BIND_WRITE;
	memset(&launch,0,sizeof(launch));
	launch.groups.x = 1u << 16u;
	launch.groups.y = launch.groups.z = 1u;
	launch.threads.x = 256u;
	launch.threads.y = launch.threads.z = 1u;
	launch.constants = &constant;
	launch.constant_bytes = sizeof(constant);
	launch.bindings = bindings;
	launch.binding_count = 2u;
	if ( SparkDeviceKernelLaunch(queue,kernel,&launch) != SPARK_STATUS_OK || SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) != SPARK_STATUS_OK )
	{
		printf("FAIL bf16 sweep launch: %s\n",SparkDeviceLastError()->text);
		return(1);
	}
	for ( index = 0u; index < 4u; index++ )
	{
		const uint32_t count = ((const uint32_t *)counts.host)[index], first = ((const uint32_t *)firsts.host)[index];
		if ( count == 0u )
			printf("%s: 0 of 4294967296 inputs differ from round-to-nearest-even with every NaN as 0x7fff\n",labels[index]);
		else
			printf("%s: %u of 4294967296 inputs differ from round-to-nearest-even with every NaN as 0x7fff (first 0x%08x)\n",labels[index],count,first);
	}
	if ( ((const uint32_t *)counts.host)[0] != 0u )
	{
		printf("FAIL the shared bf16 conversion breaks its contract on Metal\n");
		return(1);
	}
	printf("PASS the shared bf16 conversion matches its contract on all 2^32 FP32 inputs\n");
	return(CheckDecoders(queue,library) ? 0 : 1);
}
