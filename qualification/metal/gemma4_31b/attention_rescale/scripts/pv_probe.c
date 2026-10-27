#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "sparkpipe/spark_device.h"

static uint64_t state = 0x9e3779b97f4a7c15ull;
static double Uniform(void)
{
	state ^= state << 13; state ^= state >> 7; state ^= state << 17;
	return (double)(state >> 11) / 9007199254740992.0;
}

int main(int argc, char **argv)
{
	SparkDeviceQueue *queue = 0; SparkDeviceLibrary *library = 0; SparkDeviceKernel *kernel = 0;
	SparkDeviceBuffer p, v, out; SparkDeviceBinding b[3]; SparkDeviceLaunch launch;
	uint32_t i, r, d, k;
	if ( argc != 2 || SparkDeviceInitialize(0) != SPARK_STATUS_OK || SparkDeviceQueueCreate(&queue,"probe") != SPARK_STATUS_OK ||
		SparkDeviceLibraryLoad(&library,argv[1],"probe") != SPARK_STATUS_OK || SparkDeviceKernelCreate(&kernel,library,"probe_pv") != SPARK_STATUS_OK ||
		SparkDeviceAllocate(&p,SPARK_DEVICE_SPACE_SHARED_COHERENT,32u * 64u * 4u,"p") != SPARK_STATUS_OK ||
		SparkDeviceAllocate(&v,SPARK_DEVICE_SPACE_SHARED_COHERENT,64u * 256u * 2u,"v") != SPARK_STATUS_OK ||
		SparkDeviceAllocate(&out,SPARK_DEVICE_SPACE_SHARED_COHERENT,2u * 32u * 256u * 4u,"out") != SPARK_STATUS_OK )
	{
		printf("setup failed: %s\n",SparkDeviceLastError()->text);
		return(1);
	}
	float *P = (float *)p.host; uint16_t *V = (uint16_t *)v.host; float *O = (float *)out.host;
	for ( i = 0u; i < 32u * 64u; i++ )
		P[i] = (float)(0.001 + Uniform());
	for ( i = 0u; i < 64u * 256u; i++ )
	{
		float f = (float)(Uniform() * 2.0 - 1.0); uint32_t bits; memcpy(&bits,&f,4); V[i] = (uint16_t)(bits >> 16);
	}
	b[0].range.buffer = &p; b[0].range.offset = 0; b[0].bytes = p.bytes; b[0].access = SPARK_DEVICE_BIND_READ;
	b[1].range.buffer = &v; b[1].range.offset = 0; b[1].bytes = v.bytes; b[1].access = SPARK_DEVICE_BIND_READ;
	b[2].range.buffer = &out; b[2].range.offset = 0; b[2].bytes = out.bytes; b[2].access = SPARK_DEVICE_BIND_WRITE;
	memset(&launch,0,sizeof(launch));
	launch.groups.x = launch.groups.y = launch.groups.z = 1u;
	launch.threads.x = 128u; launch.threads.y = launch.threads.z = 1u;
	launch.bindings = b; launch.binding_count = 3u;
	if ( SparkDeviceKernelLaunch(queue,kernel,&launch) != SPARK_STATUS_OK || SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) != SPARK_STATUS_OK )
	{
		printf("launch failed: %s\n",SparkDeviceLastError()->text);
		return(1);
	}
	for ( k = 0u; k < 2u; k++ )
	{
		double worst = 0.0, mean = 0.0;
		for ( r = 0u; r < 32u; r++ )
			for ( d = 0u; d < 256u; d++ )
			{
				double exact = 0.0, magnitude = 0.0;
				for ( i = 0u; i < 64u; i++ )
				{
					uint32_t bits = (uint32_t)V[i * 256u + d] << 16; float vf; memcpy(&vf,&bits,4);
					exact += (double)P[r * 64u + i] * vf;
					magnitude += fabs((double)P[r * 64u + i] * vf);
				}
				const double e = fabs(O[k * 8192u + r * 256u + d] - exact) / magnitude / ldexp(1.0,-24);
				mean += e / 8192.0;
				if ( e > worst )
					worst = e;
			}
		printf("%s: error / (sum|p v| 2^-24): max %.1f mean %.2f\n",k == 0u ? "FP32 P x BF16 V in matmul2d" : "P as BF16 high + low, two runs",worst,mean);
	}
	return(0);
}
