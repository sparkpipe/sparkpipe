#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "sparkpipe/spark_device.h"
static uint64_t state = 0x2545F4914F6CDD1Dull;
static double Uniform(void) { state ^= state << 13; state ^= state >> 7; state ^= state << 17; return (double)(state >> 11) / 9007199254740992.0; }
static uint16_t ToBf16(float value) { uint32_t b; memcpy(&b, &value, sizeof(b)); return (uint16_t)((b + 0x7fffu + ((b >> 16) & 1u)) >> 16); }
static double Bf16(uint16_t x) { uint32_t w = (uint32_t)x << 16; float f; memcpy(&f, &w, sizeof(f)); return f; }
int main(int argc, char **argv)
{
	enum { ROWS = 16, D = 512, KEYS = 1024 };
	SparkDeviceQueue *queue = 0; SparkDeviceLibrary *library = 0; SparkDeviceKernel *kernel = 0;
	SparkDeviceBuffer p, v, out; SparkDeviceBinding b[3]; SparkDeviceLaunch launch;
	if ( SparkDeviceInitialize(0) != SPARK_STATUS_OK || SparkDeviceQueueCreate(&queue, "acc") != SPARK_STATUS_OK ||
		SparkDeviceLibraryLoad(&library, argv[1], "acc") != SPARK_STATUS_OK || SparkDeviceKernelCreate(&kernel, library, "probe_pv_tiles") != SPARK_STATUS_OK ||
		SparkDeviceAllocate(&p, SPARK_DEVICE_SPACE_SHARED_COHERENT, ROWS * KEYS * sizeof(float), "p") != SPARK_STATUS_OK ||
		SparkDeviceAllocate(&v, SPARK_DEVICE_SPACE_SHARED_COHERENT, (uint64_t)KEYS * D * sizeof(uint16_t), "v") != SPARK_STATUS_OK ||
		SparkDeviceAllocate(&out, SPARK_DEVICE_SPACE_SHARED_COHERENT, ROWS * D * sizeof(float), "o") != SPARK_STATUS_OK )
		return 1;
	float *P = (float *)p.host, *O = (float *)out.host; uint16_t *V = (uint16_t *)v.host;
	for ( int i = 0; i < ROWS * KEYS; i++ ) P[i] = (float)Uniform();
	for ( int i = 0; i < KEYS * D; i++ ) V[i] = ToBf16((float)(Uniform() * 2.0 - 1.0));
	b[0].range.buffer = &p; b[0].range.offset = 0; b[0].bytes = p.bytes; b[0].access = SPARK_DEVICE_BIND_READ;
	b[1].range.buffer = &v; b[1].range.offset = 0; b[1].bytes = v.bytes; b[1].access = SPARK_DEVICE_BIND_READ;
	b[2].range.buffer = &out; b[2].range.offset = 0; b[2].bytes = out.bytes; b[2].access = SPARK_DEVICE_BIND_WRITE;
	memset(&launch, 0, sizeof(launch));
	launch.groups.x = launch.groups.y = launch.groups.z = 1u; launch.threads.x = 128u; launch.threads.y = launch.threads.z = 1u;
	launch.bindings = b; launch.binding_count = 3u;
	if ( SparkDeviceKernelLaunch(queue, kernel, &launch) != SPARK_STATUS_OK || SparkDeviceQueueWait(queue, SPARK_DEVICE_WAIT_FOREVER) != SPARK_STATUS_OK )
		return 1;
	double worst = 0, mean = 0, worst_seq = 0, mean_seq = 0;
	for ( int r = 0; r < ROWS; r++ )
		for ( int d = 0; d < D; d++ )
		{
			double exact = 0, magnitude = 0; float sequential = 0.0f;
			for ( int j = 0; j < KEYS; j++ )
			{
				const double x = (double)P[r * KEYS + j] * Bf16(V[j * D + d]);
				exact += x; magnitude += fabs(x); sequential = fmaf(P[r * KEYS + j], (float)Bf16(V[j * D + d]), sequential);
			}
			const double e = fabs(O[r * D + d] - exact) / magnitude / ldexp(1.0, -24), e2 = fabs(sequential - exact) / magnitude / ldexp(1.0, -24);
			mean += e / (ROWS * D); worst = e > worst ? e : worst; mean_seq += e2 / (ROWS * D); worst_seq = e2 > worst_seq ? e2 : worst_seq;
		}
	printf("matmul2d P x V over 16 tiles of 64 keys, multiply_accumulate into one destination: error / (2^-24 sum |p v|) max %.1f mean %.2f; sequential FP32 fma: max %.1f mean %.2f\n",
		worst, mean, worst_seq, mean_seq);
	return 0;
}
