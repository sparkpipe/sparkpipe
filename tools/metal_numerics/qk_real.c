#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "sparkpipe/spark_device.h"
static uint16_t ToBf16(float value) { uint32_t b; memcpy(&b, &value, sizeof(b)); return (uint16_t)((b + 0x7fffu + ((b >> 16) & 1u)) >> 16); }
static double Bf16(uint16_t x) { uint32_t w = (uint32_t)x << 16; float f; memcpy(&f, &w, sizeof(f)); return f; }
int main(int argc, char **argv)
{
	const char *library_path = argv[1], *prefix = argv[2];
	const uint32_t heads = (uint32_t)atoi(argv[3]), kvh = (uint32_t)atoi(argv[4]), dim = (uint32_t)atoi(argv[5]), total = (uint32_t)atoi(argv[6]);
	const uint32_t rows = dim == 512u ? 16u : 32u;
	const char *kernel_name = getenv("QK_SPLIT") != 0 ? (dim == 512u ? "probe_qk512_split" : "probe_qk256_split") : (dim == 512u ? "probe_qk512" : "probe_qk256");
	char path[1024];
	float *q = (float *)malloc((size_t)total * heads * dim * sizeof(float)), *k = (float *)malloc((size_t)total * kvh * dim * sizeof(float));
	FILE *f;
	snprintf(path, sizeof(path), "%s.q", prefix); f = fopen(path, "rb"); fread(q, sizeof(float), (size_t)total * heads * dim, f); fclose(f);
	snprintf(path, sizeof(path), "%s.k", prefix); f = fopen(path, "rb"); fread(k, sizeof(float), (size_t)total * kvh * dim, f); fclose(f);
	SparkDeviceQueue *queue = 0; SparkDeviceLibrary *library = 0; SparkDeviceKernel *kernel = 0;
	SparkDeviceBuffer qb, kb, ob; SparkDeviceBinding b[3]; SparkDeviceLaunch launch;
	if ( SparkDeviceInitialize(0) != SPARK_STATUS_OK || SparkDeviceQueueCreate(&queue, "qk") != SPARK_STATUS_OK ||
		SparkDeviceLibraryLoad(&library, library_path, "qk") != SPARK_STATUS_OK || SparkDeviceKernelCreate(&kernel, library, kernel_name) != SPARK_STATUS_OK ||
		SparkDeviceAllocate(&qb, SPARK_DEVICE_SPACE_SHARED_COHERENT, rows * dim * sizeof(uint16_t), "q") != SPARK_STATUS_OK ||
		SparkDeviceAllocate(&kb, SPARK_DEVICE_SPACE_SHARED_COHERENT, 64u * dim * sizeof(uint16_t), "k") != SPARK_STATUS_OK ||
		SparkDeviceAllocate(&ob, SPARK_DEVICE_SPACE_SHARED_COHERENT, rows * 64u * sizeof(float), "o") != SPARK_STATUS_OK )
		return 1;
	double tensor_sq = 0, lane_sq = 0, n = 0, tensor_worst = 0, lane_worst = 0;
	for ( uint32_t head = 0; head < heads; head++ )
		for ( uint32_t block = 0; block + rows <= total && block < 16u * rows * 4u; block += rows * 4u )
			for ( uint32_t tile = 0; tile + 64u <= block + rows; tile += 256u )
			{
				const uint32_t kv = head / (heads / kvh);
				uint16_t *Q = (uint16_t *)qb.host, *K = (uint16_t *)kb.host; float *O = (float *)ob.host;
				for ( uint32_t r = 0; r < rows; r++ )
					for ( uint32_t d = 0; d < dim; d++ )
						Q[r * dim + d] = ToBf16(q[((uint64_t)(block + r) * heads + head) * dim + d]);
				for ( uint32_t j = 0; j < 64u; j++ )
					for ( uint32_t d = 0; d < dim; d++ )
						K[j * dim + d] = ToBf16(k[((uint64_t)(tile + j) * kvh + kv) * dim + d]);
				b[0].range.buffer = &qb; b[0].range.offset = 0; b[0].bytes = qb.bytes; b[0].access = SPARK_DEVICE_BIND_READ;
				b[1].range.buffer = &kb; b[1].range.offset = 0; b[1].bytes = kb.bytes; b[1].access = SPARK_DEVICE_BIND_READ;
				b[2].range.buffer = &ob; b[2].range.offset = 0; b[2].bytes = ob.bytes; b[2].access = SPARK_DEVICE_BIND_WRITE;
				memset(&launch, 0, sizeof(launch));
				launch.groups.x = launch.groups.y = launch.groups.z = 1u; launch.threads.x = 128u; launch.threads.y = launch.threads.z = 1u;
				launch.bindings = b; launch.binding_count = 3u;
				if ( SparkDeviceKernelLaunch(queue, kernel, &launch) != SPARK_STATUS_OK || SparkDeviceQueueWait(queue, SPARK_DEVICE_WAIT_FOREVER) != SPARK_STATUS_OK )
					return 1;
				for ( uint32_t r = 0; r < rows; r++ )
					for ( uint32_t j = 0; j < 64u; j++ )
					{
						double exact = 0; float lanes[32], lane = 0.0f;
						for ( uint32_t l = 0; l < 32u; l++ )
						{
							lanes[l] = 0.0f;
							for ( uint32_t c = 0; c < dim / 32u; c++ )
							{
								const uint32_t d = l * (dim / 32u) + c;
								lanes[l] += (float)Bf16(Q[r * dim + d]) * (float)Bf16(K[j * dim + d]);
							}
						}
						for ( uint32_t width = 16u; width > 0u; width /= 2u )
							for ( uint32_t l = 0; l < width; l++ )
								lanes[l] += lanes[l + width];
						lane = lanes[0];
						for ( uint32_t d = 0; d < dim; d++ )
							exact += Bf16(Q[r * dim + d]) * Bf16(K[j * dim + d]);
						const double et = O[r * 64u + j] - exact, el = lane - exact;
						tensor_sq += et * et; lane_sq += el * el; n++;
						tensor_worst = fabs(et) > tensor_worst ? fabs(et) : tensor_worst; lane_worst = fabs(el) > lane_worst ? fabs(el) : lane_worst;
					}
			}
	printf("%s: real attention scores (absolute error against float64, the error that becomes a weight's relative error): matmul2d rms %.3g max %.3g; the decode kernel's per-lane sum and SIMD tree rms %.3g max %.3g (%.0f scores)\n",
		prefix, sqrt(tensor_sq / n), tensor_worst, sqrt(lane_sq / n), lane_worst, n);
	return 0;
}
