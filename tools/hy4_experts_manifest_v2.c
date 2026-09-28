#define _POSIX_C_SOURCE 200809L

#include "sparkpipe/spark_ck128.h"
#include "sparkpipe/spark_weightd.h"
#include "sparkpipe/spark_weightd_manifest.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define HY4_JSON_MAX (8ull << 20)
#define HY4_NAME_MAX 160u
#define HY4_LAYERS_MAX 128u
#define HY4_EXPERTS_MAX 256ull
#define HY4_CHUNK_BYTES (1ull << 20)

typedef struct
{
	const char *text;
	size_t bytes;
} Hy4Json;

static const char *json_find(const Hy4Json *json, const char *name)
{
	size_t name_bytes = strlen(name);
	size_t at;
	for (at = 0u; at + name_bytes + 2u < json->bytes; at++)
	{
		if (json->text[at] != '"')
			continue;
		if (memcmp(json->text + at + 1u, name, name_bytes) != 0)
			continue;
		if (json->text[at + 1u + name_bytes] != '"')
			continue;
		return json->text + at + 1u + name_bytes + 1u;
	}
	return 0;
}

static const char *json_key_after(const char *at, const char *key)
{
	size_t key_bytes = strlen(key);
	for (; at[0] != '\0'; at++)
	{
		if (at[0] != '"')
			continue;
		if (strncmp(at + 1u, key, key_bytes) != 0)
			continue;
		if (at[key_bytes + 1u] != '"')
			continue;
		return at + key_bytes + 2u;
	}
	return 0;
}

static int json_pair_after(const char *at, const char *key,
	unsigned long long *first, unsigned long long *second)
{
	int seen = 0;
	unsigned long long value = 0ull;
	int in_value = 0;
	at = json_key_after(at, key);
	if (at == 0)
		return 0;
	for (; at[0] != '\0' && seen < 2;)
	{
		if (at[0] == '"')
		{
			at++;
			while (at[0] != '\0' && at[0] != '"')
				at++;
			if (at[0] != '\0')
				at++;
			continue;
		}
		if (at[0] >= '0' && at[0] <= '9')
		{
			if (in_value == 0)
			{
				in_value = 1;
				value = 0ull;
			}
			value = value * 10ull +
			    (unsigned long long)(at[0] - '0');
		}
		else if (in_value != 0)
		{
			if (seen == 0)
				*first = value;
			else
				*second = value;
			seen++;
			in_value = 0;
		}
		at++;
	}
	return seen == 2;
}

typedef struct
{
	FILE *pack;
	FILE *out;
	uint8_t *buffer;
	uint32_t count;
	unsigned long long experts;
} Hy4Writer;

static int32_t range_write(Hy4Writer *writer, uint32_t layer,
	uint32_t expert, uint32_t kind, unsigned long long offset,
	unsigned long long bytes)
{
	SparkCk128Context ck;
	uint8_t record[48] = {0};
	unsigned long long remaining;
	unsigned long long piece;
	if (bytes == 0ull || bytes > SPARK_WEIGHTD_EXPERT_BYTES_MAX)
		return 1;
	if (fseeko(writer->pack, (off_t)offset, SEEK_SET) != 0)
		return 2;
	SparkCk128Initialize(&ck);
	remaining = bytes;
	while (remaining != 0ull)
	{
		piece = remaining < HY4_CHUNK_BYTES ? remaining :
		    HY4_CHUNK_BYTES;
		if (fread(writer->buffer, 1u, (size_t)piece, writer->pack)
			!= piece)
			return 3;
		SparkCk128Update(&ck, writer->buffer, (size_t)piece);
		remaining -= piece;
	}
	SparkCk128Finalize(&ck, record + 32u);
	memcpy(record, &layer, 4u);
	memcpy(record + 4u, &expert, 4u);
	memcpy(record + 8u, &kind, 4u);
	memcpy(record + 16u, &offset, 8u);
	memcpy(record + 24u, &bytes, 8u);
	if (fwrite(record, 1u, sizeof(record), writer->out) !=
	    sizeof(record))
		return 4;
	writer->count++;
	return 0;
}

static int32_t class_ranges(Hy4Writer *writer, const Hy4Json *json,
	unsigned long long data_base, uint32_t layer, uint32_t kind,
	const char *payload_name, const char *scale_name)
{
	const char *at;
	unsigned long long numbers[2];
	unsigned long long payload_start = 0ull;
	unsigned long long payload_end = 0ull;
	unsigned long long payload_experts = 0ull;
	unsigned long long scale_start = 0ull;
	unsigned long long scale_end = 0ull;
	unsigned long long payload_slab;
	unsigned long long scale_slab;
	unsigned long long expert;
	at = json_find(json, payload_name);
	if (at == 0)
		return 1;
	if (json_pair_after(at, "shape", &payload_experts, &numbers[0]) == 0)
		return 2;
	if (json_pair_after(at, "data_offsets", &payload_start,
		&payload_end) == 0)
		return 2;
	at = json_find(json, scale_name);
	if (at == 0)
		return 1;
	if (json_pair_after(at, "shape", &numbers[0], &numbers[1]) == 0)
		return 2;
	if (json_pair_after(at, "data_offsets", &scale_start,
		&scale_end) == 0)
		return 2;
	if (payload_experts != writer->experts)
	{
		fprintf(stderr, "%s: expert dim %llu\n", payload_name,
			payload_experts);
		return 5;
	}
	payload_slab = (payload_end - payload_start) / payload_experts;
	scale_slab = (scale_end - scale_start) / payload_experts;
	for (expert = 0ull; expert < writer->experts; expert++)
	{
		int32_t status = range_write(writer, layer,
			(uint32_t)expert, kind * 2u,
			data_base + payload_start + expert * payload_slab,
			payload_slab);
		if (status != 0)
			return status;
		status = range_write(writer, layer, (uint32_t)expert,
		    kind * 2u + 1u, data_base + scale_start +
		    expert * scale_slab, scale_slab);
		if (status != 0)
			return status;
	}
	return 0;
}

static int32_t layer_ranges(Hy4Writer *writer, const Hy4Json *json,
	unsigned long long data_base, uint32_t layer)
{
	char gate[HY4_NAME_MAX];
	char down[HY4_NAME_MAX];
	char gate_scale[HY4_NAME_MAX];
	char down_scale[HY4_NAME_MAX];
	int32_t status;
	snprintf(gate, sizeof(gate),
		"model.layers.%u.mlp.experts.gate_up_proj", layer);
	snprintf(gate_scale, sizeof(gate_scale),
		"model.layers.%u.mlp.experts.gate_up_proj_scale", layer);
	snprintf(down, sizeof(down),
		"model.layers.%u.mlp.experts.down_proj", layer);
	snprintf(down_scale, sizeof(down_scale),
		"model.layers.%u.mlp.experts.down_proj_scale", layer);
	status = class_ranges(writer, json, data_base, layer, 0u, gate,
		gate_scale);
	if (status != 0)
	{
		fprintf(stderr, "gate_up layer %u: status %d\n", layer,
			status);
		return status;
	}
	status = class_ranges(writer, json, data_base, layer, 1u, down,
		down_scale);
	if (status != 0)
		fprintf(stderr, "down layer %u: status %d\n", layer, status);
	return status;
}

static unsigned long long layer_experts(const Hy4Json *json, uint32_t layer)
{
	char name[HY4_NAME_MAX];
	const char *at;
	unsigned long long experts = 0ull;
	unsigned long long rows = 0ull;
	snprintf(name, sizeof(name),
		"model.layers.%u.mlp.experts.gate_up_proj", layer);
	at = json_find(json, name);
	if (at == 0 || json_pair_after(at, "shape", &experts, &rows) == 0)
		return 0ull;
	return experts;
}

static uint32_t layer_present(const Hy4Json *json, uint32_t layer)
{
	char name[HY4_NAME_MAX];
	snprintf(name, sizeof(name),
		"model.layers.%u.mlp.experts.gate_up_proj", layer);
	return json_find(json, name) != 0;
}

int main(int argc, char **argv)
{
	FILE *pack;
	FILE *out;
	uint8_t *json;
	Hy4Json view;
	Hy4Writer writer;
	uint8_t wire[16] = {0};
	uint32_t magic = SPARK_WEIGHTD_EXPERT_MANIFEST_MAGIC;
	uint32_t version = SPARK_WEIGHTD_RANGE_MANIFEST_VERSION;
	uint32_t layer;
	uint32_t layers_found = 0u;
	uint32_t layer_ids[HY4_LAYERS_MAX];
	uint32_t total;
	unsigned long long json_bytes;
	char path[4600];

	if (argc != 2)
	{
		fprintf(stderr, "usage: %s <rank-pack.safetensors>\n",
			argv[0]);
		return 2;
	}
	pack = fopen(argv[1], "rb");
	if (pack == 0)
	{
		perror("open pack");
		return 1;
	}
	{
		uint8_t prefix[8];
		if (fread(prefix, 1u, 8u, pack) != 8u)
		{
			fprintf(stderr, "short pack prefix\n");
			return 1;
		}
		memcpy(&json_bytes, prefix, 8u);
	}
	if (json_bytes == 0ull || json_bytes > HY4_JSON_MAX)
	{
		fprintf(stderr, "bad safetensors header size\n");
		return 1;
	}
	json = malloc((size_t)json_bytes + 1u);
	if (json == 0)
		return 1;
	if (fread(json, 1u, (size_t)json_bytes, pack) !=
	    (size_t)json_bytes)
	{
		fprintf(stderr, "short safetensors header\n");
		return 1;
	}
	json[json_bytes] = '\0';
	view.text = (const char *)json;
	view.bytes = (size_t)json_bytes;
	for (layer = 0u; layer < HY4_LAYERS_MAX; layer++)
	{
		if (layer_present(&view, layer) != 0u)
			layer_ids[layers_found++] = layer;
	}
	if (layers_found == 0u)
	{
		fprintf(stderr, "no routed-expert layers in header\n");
		return 1;
	}
	writer.experts = layer_experts(&view, layer_ids[0]);
	if (writer.experts == 0ull || writer.experts > HY4_EXPERTS_MAX ||
	    (unsigned long long)layers_found * 4ull * writer.experts >
	    SPARK_WEIGHTD_RANGE_COUNT_MAX)
	{
		fprintf(stderr, "bad routed-expert count %llu\n", writer.experts);
		return 1;
	}
	snprintf(path, sizeof(path), "%s.experts", argv[1]);
	out = fopen(path, "wb");
	if (out == 0)
	{
		perror("open manifest");
		return 1;
	}
	memcpy(wire, &magic, 4u);
	memcpy(wire + 4u, &version, 4u);
	total = layers_found * 4u * (uint32_t)writer.experts;
	memcpy(wire + 8u, &total, 4u);
	fwrite(wire, 1u, sizeof(wire), out);
	unsigned long long data_base = 8ull + json_bytes;
	writer.pack = pack;
	writer.out = out;
	writer.buffer = malloc(SPARK_WEIGHTD_EXPERT_BYTES_MAX);
	writer.count = 0u;
	if (writer.buffer == 0)
		return 1;
	for (layer = 0u; layer < layers_found; layer++)
	{
		if (layer_ranges(&writer, &view, data_base,
		    layer_ids[layer]) != 0)
			return 1;
	}
	free(writer.buffer);
	free(json);
	fclose(out);
	fclose(pack);
	if (writer.count != total)
	{
		fprintf(stderr, "%s: wrote %u ranges, header says %u\n", path,
			writer.count, total);
		return 1;
	}
	printf("%s: %u ranges (%u layers x 4 ranges x %llu experts)\n",
		path, writer.count, layers_found, writer.experts);
	return 0;
}
