#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include "sparkpipe/spark_drafter_module.h"

typedef SparkStatus (*ContextRows)(void *, uint64_t, uint32_t, uint64_t, uint32_t, uint16_t *);

static uint16_t Bf16(float value)
{
	uint32_t bits;
	memcpy(&bits, &value, 4u);
	return (uint16_t)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

int main(int argc, char **argv)
{
	const uint32_t hidden = 7168u, taps = 5u;
	SparkDrafterModuleLane lane;
	SparkDrafterModuleBind bind;
	SparkTapStreamReaderFrame frame;
	SparkTapStreamHello hello;
	uint32_t required[5] = {0u, 1u, 2u, 3u, 4u}, value = 4242u;
	char digest[80], path[600];
	struct stat info;
	void *state = 0;
	FILE *file;
	if ( argc != 6 )
	{
		fprintf(stderr, "usage: k31_context_probe MODULE PACK WEIGHTD_SOCKET OUT_DIR ROWS\n");
		return 2;
	}
	const uint32_t rows = (uint32_t)atoi(argv[5]);
	uint16_t *tap_rows = malloc((size_t)rows * taps * hidden * 2u), *out = malloc((size_t)rows * 576u * 2u);
	void *library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
	SparkDrafterModuleGetInterface get = library != 0 ? (SparkDrafterModuleGetInterface)dlsym(library, SPARK_DRAFTER_MODULE_INTERFACE_SYMBOL) : 0;
	ContextRows context_rows = library != 0 ? (ContextRows)dlsym(library, "spark_k31_dspark_context_rows") : 0;
	const SparkDrafterModuleInterface *module = get != 0 ? get() : 0;
	if ( module == 0 || context_rows == 0 )
	{
		fprintf(stderr, "module load failed: %s\n", dlerror());
		return 1;
	}
	snprintf(path, sizeof(path), "%s.sha256", argv[2]);
	file = fopen(path, "r");
	if ( file == 0 || fscanf(file, "%64s", digest) != 1 || stat(argv[2], &info) != 0 )
		return 1;
	fclose(file);
	memset(&lane, 0, sizeof(lane));
	memset(&bind, 0, sizeof(bind));
	lane.abi_version = SPARK_DRAFTER_MODULE_ABI_VERSION;
	lane.descriptor_bytes = sizeof(lane);
	lane.lane_name = "probe";
	lane.pack_path = argv[2];
	lane.model = "k31dspark";
	lane.revision = "6a2ef20a05ef";
	lane.pack_sha256 = digest;
	lane.weightd_socket = argv[3];
	lane.pack_bytes = (uint64_t)info.st_size;
	lane.context_bytes = UINT64_C(2) << 30;
	lane.timeout_ns = UINT64_C(600000000000);
	if ( module->open(&lane, &state, &bind) != SPARK_STATUS_OK )
	{
		fprintf(stderr, "open refused: %s\n", bind.text);
		return 1;
	}
	printf("bound: %s device_bytes=%llu bind_s=%.3f first_hold=%u\n", bind.text, (unsigned long long)bind.device_bytes, (double)bind.bind_ns / 1e9,
		bind.first_hold);
	for ( uint64_t index = 0u; index < (uint64_t)rows * taps * hidden; index++ )
	{
		float a, b;
		value = value * 1664525u + 1013904223u;
		a = (float)((value >> 8) + 1u) / 16777217.0f;
		value = value * 1664525u + 1013904223u;
		b = (float)(value >> 8) / 16777216.0f;
		tap_rows[index] = Bf16((a - 0.5f) * 3.0f + (b - 0.5f));
	}
	memset(&hello, 0, sizeof(hello));
	hello.hidden = hidden;
	memset(&frame, 0, sizeof(frame));
	frame.header.sequence_id = 42u;
	frame.header.first_position = 100u;
	frame.header.row_count = rows;
	frame.hello = &hello;
	frame.rows = tap_rows;
	frame.row_elements = (uint64_t)taps * hidden;
	frame.required_index = required;
	frame.required_count = taps;
	if ( module->ingest(state, &frame) != SPARK_STATUS_OK )
	{
		fprintf(stderr, "ingest failed\n");
		return 1;
	}
	snprintf(path, sizeof(path), "%s/taps.bin", argv[4]);
	file = fopen(path, "wb");
	fwrite(tap_rows, 2u, (size_t)rows * taps * hidden, file);
	fclose(file);
	for ( uint32_t layer = 0u; layer < 5u; layer++ )
	{
		if ( context_rows(state, 42u, layer, 100u, rows, out) != SPARK_STATUS_OK )
		{
			fprintf(stderr, "context rows layer %u failed\n", layer);
			return 1;
		}
		snprintf(path, sizeof(path), "%s/rows%u.bin", argv[4], layer);
		file = fopen(path, "wb");
		fwrite(out, 2u, (size_t)rows * 576u, file);
		fclose(file);
	}
	printf("wrote %u rows x 5 layers to %s\n", rows, argv[4]);
	return module->close(state) == SPARK_STATUS_OK ? 0 : 1;
}
