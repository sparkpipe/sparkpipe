#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define RANS_M 4096u
#define RANS_L 12u
#define RANS_BOUND (1u << 23)
#define TILE_VALUES 8192u
#define SUBSTREAMS 128u
#define SUB_LEN 64u
#define PACK_HEADER_BYTES 120u
#define PACK_ENTRY_BYTES 56u
#define PACK_ALIGN 256u

static void die(const char *msg)
{
	fprintf(stderr, "qwen38_27b_stagepack_rans: %s\n", msg);
	exit(1);
}

typedef struct
{
	uint32_t ndirect;
	uint32_t id_bits;
	uint32_t *entries;
	uint16_t *id_table;
	int32_t *entry_of;
	int32_t *rid_of;
	uint32_t f_esc, c_esc;
} RansEnc;

static int cmp_u32(const void *a, const void *b)
{
	uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
	return x < y ? -1 : (x > y ? 1 : 0);
}

static void rans_build_table(RansEnc *e, const uint16_t *values, uint64_t n)
{
	uint32_t counts[65536];
	uint32_t present[65536];
	uint32_t direct[65536];
	uint32_t rest[65536];
	uint32_t npresent = 0, ndirect = 0, nrest = 0;
	uint32_t i;
	uint64_t fsum = 0;
	memset(counts, 0, sizeof(counts));
	for (i = 0; i < n; i++)
		counts[values[i]]++;
	for (i = 0; i < 65536; i++)
		if (counts[i])
			present[npresent++] = i;
	for (i = 0; i < npresent; i++)
	{
		uint32_t s = present[i];
		if ((uint64_t)counts[s] * RANS_M >= n / 2)
			direct[ndirect++] = s;
		else
			rest[nrest++] = s;
	}
	qsort(direct, ndirect, 4, cmp_u32);
	qsort(rest, nrest, 4, cmp_u32);
	e->id_bits = 1;
	while ((1u << e->id_bits) < nrest)
		e->id_bits++;
	if (e->id_bits > 16)
		die("id_bits > 16");
	e->ndirect = ndirect;
	e->entries = malloc((size_t)(ndirect + 1) * 12);
	e->id_table = calloc((size_t)1 << e->id_bits, 2);
	e->entry_of = malloc(65536 * 4);
	e->rid_of = malloc(65536 * 4);
	if (!e->entries || !e->id_table || !e->entry_of || !e->rid_of)
		die("oom table");
	for (i = 0; i < 65536; i++)
	{
		e->entry_of[i] = -1;
		e->rid_of[i] = -1;
	}
	for (i = 0; i < ndirect; i++)
	{
		uint32_t s = direct[i];
		uint64_t f = ((uint64_t)counts[s] * RANS_M + n / 2) / n;
		if (f < 1) f = 1;
		if (f > RANS_M - 1) f = RANS_M - 1;
		e->entries[3 * i] = s;
		e->entries[3 * i + 1] = (uint32_t)f;
		fsum += f;
		e->entry_of[s] = (int32_t)i;
	}
	for (i = 0; i < nrest; i++)
	{
		e->rid_of[rest[i]] = (int32_t)i;
		e->id_table[i] = (uint16_t)rest[i];
	}
	while (fsum >= RANS_M)
	{
		uint32_t bi = 0;
		uint64_t bf = 0;
		for (i = 0; i < ndirect; i++)
		{
			uint64_t f = e->entries[3 * i + 1];
			if (f > bf) { bf = f; bi = i; }
		}
		if (bf <= 1)
			die("freq normalization failed");
		e->entries[3 * bi + 1] -= 1u;
		fsum -= 1u;
	}
	e->f_esc = RANS_M - (uint32_t)fsum;
	e->entries[3 * ndirect] = 0xFFFFu;
	e->entries[3 * ndirect + 1] = e->f_esc;
	{
		uint32_t c = 0;
		for (i = 0; i <= ndirect; i++)
		{
			e->entries[3 * i + 2] = c;
			c += e->entries[3 * i + 1];
		}
		e->c_esc = e->entries[3 * ndirect + 2];
	}
}

typedef struct { uint8_t *data; uint64_t size; } Buffer;

static uint32_t rans_encode_lane(RansEnc *e, const uint16_t *sub, uint8_t *lane, uint32_t lane_cap, uint32_t *out_len)
{
	uint8_t *p = lane + lane_cap;
	uint64_t x = RANS_BOUND;
	uint32_t j;
	for (j = SUB_LEN; j-- > 0;)
	{
		uint32_t s = sub[j];
		int32_t ei = e->entry_of[s];
		uint32_t fs, cs;
		uint64_t x_max;
		if (ei >= 0)
		{
			fs = e->entries[3 * ei + 1];
			cs = e->entries[3 * ei + 2];
		}
		else
		{
			uint32_t rid = (uint32_t)e->rid_of[s];
			*--p = (uint8_t)((rid >> 8) & 0xFF);
			*--p = (uint8_t)(rid & 0xFF);
			fs = e->f_esc;
			cs = e->c_esc;
		}
		x_max = ((uint64_t)1 << 18) * fs;
		while (x >= x_max)
		{
			*--p = (uint8_t)(x & 0xFF);
			x >>= 8;
		}
		x = ((x / fs) << RANS_L) + (x % fs) + cs;
	}
	*out_len = (uint32_t)((lane + lane_cap) - p);
	memmove(lane, p, *out_len);
	return (uint32_t)x;
}

static Buffer rans_encode_tensor(RansEnc *e, const uint16_t *tiled, uint32_t chunk_count)
{
	Buffer b;
	uint32_t *offsets;
	uint64_t out_cap, out_len;
	uint32_t t;
	out_cap = (uint64_t)chunk_count * 24576 + (uint64_t)(e->ndirect + 1) * 12 +
	          (uint64_t)(1u << e->id_bits) * 2 + (uint64_t)chunk_count * 4 + 64;
	b.data = malloc((size_t)out_cap);
	if (!b.data)
		die("oom stream");
	out_len = 0;
	memcpy(b.data + out_len, &e->ndirect, 4); out_len += 4;
	memcpy(b.data + out_len, e->entries, (size_t)(e->ndirect + 1) * 12); out_len += (uint64_t)(e->ndirect + 1) * 12;
	memcpy(b.data + out_len, &e->id_bits, 4); out_len += 4;
	memcpy(b.data + out_len, e->id_table, (size_t)(1u << e->id_bits) * 2); out_len += (uint64_t)(1u << e->id_bits) * 2;
	memcpy(b.data + out_len, &chunk_count, 4); out_len += 4;
	offsets = (uint32_t *)(b.data + out_len);
	out_len += (uint64_t)chunk_count * 4;
	for (t = 0; t < chunk_count; t++)
	{
		const uint16_t *tile = tiled + (size_t)t * TILE_VALUES;
		uint8_t *chunk = b.data + out_len;
		uint32_t states[SUBSTREAMS];
		uint16_t lens[SUBSTREAMS];
		uint64_t data_start = SUBSTREAMS * 4 + SUBSTREAMS * 2;
		uint32_t i, j;
		offsets[t] = (uint32_t)out_len;
		for (i = 0; i < SUBSTREAMS; i++)
		{
			uint16_t sub[SUB_LEN];
			uint8_t lane[SUB_LEN * 3 + 16];
			uint32_t llen = 0;
			for (j = 0; j < SUB_LEN; j++)
				sub[j] = tile[j * SUBSTREAMS + i];
			states[i] = rans_encode_lane(e, sub, lane, sizeof(lane), &llen);
			lens[i] = llen;
			memcpy(chunk + data_start, lane, llen);
			data_start += llen;
		}
		memcpy(chunk, states, sizeof(states));
		memcpy(chunk + SUBSTREAMS * 4, lens, sizeof(lens));
		while (data_start & 3u)
			chunk[data_start++] = 0;
		out_len += data_start;
	}
	b.size = out_len;
	return b;
}

typedef struct
{
	uint32_t tensor_kind;
	uint32_t layer_index;
	uint32_t weight_format;
	uint32_t rows;
	uint32_t columns;
	uint32_t scale_group_size;
	uint64_t payload_offset;
	uint64_t payload_bytes;
	uint64_t scale_offset;
	uint64_t scale_bytes;
} PackEntry;

static int compressible_kind(uint32_t kind)
{
	switch (kind)
	{
	case 7:
	case 8:
	case 9:
	case 12:
	case 17:
	case 18:
	case 19:
	case 20:
	case 23:
		return 1;
	default:
		return 0;
	}
}

static Buffer read_file(const char *path)
{
	FILE *f = fopen(path, "rb");
	Buffer b;
	long size;
	if (!f)
		die("cannot open pack");
	fseek(f, 0, SEEK_END);
	size = ftell(f);
	fseek(f, 0, SEEK_SET);
	b.data = malloc((size_t)size);
	b.size = (uint64_t)size;
	if (!b.data || fread(b.data, 1, (size_t)size, f) != (size_t)size)
		die("short read");
	fclose(f);
	return b;
}

int main(int argc, char **argv)
{
	Buffer pack;
	uint32_t tensor_count, header_bytes, dir_entry_bytes;
	uint64_t directory_offset, directory_bytes;
	PackEntry *entries;
	uint64_t i;
	uint32_t compressed_count = 0;
	uint64_t compressed_before = 0, compressed_after = 0;
	FILE *of;
	uint64_t out_size;
	uint8_t pad[PACK_ALIGN];
	if (argc < 3)
	{
		fprintf(stderr, "usage: %s in.qwen38_27bsp out.qwen38_27bsp\n", argv[0]);
		return 1;
	}
	memset(pad, 0, sizeof(pad));
	pack = read_file(argv[1]);
	if (pack.size < PACK_HEADER_BYTES)
		die("pack too small");
	{
		uint32_t magic = *(uint32_t *)(pack.data + 0);
		if (magic != 0x50533651u)
			die("bad magic");
		header_bytes = *(uint32_t *)(pack.data + 8);
		dir_entry_bytes = *(uint32_t *)(pack.data + 12);
		tensor_count = *(uint32_t *)(pack.data + 16);
		directory_offset = *(uint64_t *)(pack.data + 104);
		if (header_bytes != PACK_HEADER_BYTES || dir_entry_bytes != PACK_ENTRY_BYTES)
			die("unexpected pack geometry");
	}
	directory_bytes = (uint64_t)tensor_count * PACK_ENTRY_BYTES;
	if (directory_offset + directory_bytes > pack.size)
		die("directory out of range");
	entries = malloc((size_t)tensor_count * sizeof(PackEntry));
	if (!entries)
		die("oom directory");
	for (i = 0; i < tensor_count; i++)
	{
		uint8_t *e = pack.data + directory_offset + i * PACK_ENTRY_BYTES;
		entries[i].tensor_kind = *(uint32_t *)(e + 0);
		entries[i].layer_index = *(uint32_t *)(e + 4);
		entries[i].weight_format = *(uint32_t *)(e + 8);
		entries[i].rows = *(uint32_t *)(e + 12);
		entries[i].columns = *(uint32_t *)(e + 16);
		entries[i].scale_group_size = *(uint32_t *)(e + 20);
		entries[i].payload_offset = *(uint64_t *)(e + 24);
		entries[i].payload_bytes = *(uint64_t *)(e + 32);
		entries[i].scale_offset = *(uint64_t *)(e + 40);
		entries[i].scale_bytes = *(uint64_t *)(e + 48);
	}
	of = fopen(argv[2], "wb");
	if (!of)
		die("cannot create output");
	out_size = (directory_offset + directory_bytes + PACK_ALIGN - 1) & ~(uint64_t)(PACK_ALIGN - 1);
	if (fwrite(pack.data, 1, (size_t)(directory_offset + directory_bytes), of) != directory_offset + directory_bytes)
		die("short header write");
	if (fwrite(pad, 1, (size_t)(out_size - directory_offset - directory_bytes), of) != out_size - directory_offset - directory_bytes)
		die("short pad write");
	for (i = 0; i < tensor_count; i++)
	{
		PackEntry *en = &entries[i];
		uint64_t next;
		if (compressible_kind(en->tensor_kind) && en->weight_format == 0u &&
		    (en->rows % 64u) == 0u && (en->columns % 128u) == 0u && en->scale_bytes == 0u)
		{
			uint64_t nvals = (uint64_t)en->rows * en->columns;
			uint32_t chunks = (en->rows / 64u) * (en->columns / 128u);
			const uint16_t *raw = (const uint16_t *)(pack.data + en->payload_offset);
			uint16_t *tiled = malloc((size_t)nvals * 2);
			RansEnc e;
			Buffer comp;
			uint32_t r, c;
			if (en->payload_offset + en->payload_bytes > pack.size)
				die("payload out of range");
			if (!tiled)
				die("oom tiled");
			for (r = 0; r < en->rows; r++)
			{
				uint32_t tr = (r >> 6) * (en->columns >> 7);
				uint32_t rr = r & 63u;
				for (c = 0; c < en->columns; c++)
				{
					uint32_t tc = c >> 7, cc = c & 127u;
					tiled[((size_t)(tr + tc) * 64u + rr) * 128u + cc] = raw[(size_t)r * en->columns + c];
				}
			}
			rans_build_table(&e, tiled, nvals);
			comp = rans_encode_tensor(&e, tiled, chunks);
			compressed_before += en->payload_bytes;
			compressed_after += comp.size;
			next = (out_size + PACK_ALIGN - 1) & ~(uint64_t)(PACK_ALIGN - 1);
			if (fwrite(pad, 1, (size_t)(next - out_size), of) != next - out_size)
				die("short pad write (compressed)");
			if (fwrite(comp.data, 1, (size_t)comp.size, of) != comp.size)
				die("short payload write");
			out_size = next + comp.size;
			en->payload_offset = next;
			en->payload_bytes = comp.size;
			en->weight_format = 4u;
			compressed_count++;
			free(tiled);
			free(comp.data);
			free(e.entries);
			free(e.id_table);
			free(e.entry_of);
			free(e.rid_of);
			continue;
		}
		next = (out_size + PACK_ALIGN - 1) & ~(uint64_t)(PACK_ALIGN - 1);
		if (en->payload_offset + en->payload_bytes > pack.size)
			die("payload out of range (copy)");
		if (fwrite(pad, 1, (size_t)(next - out_size), of) != next - out_size)
			die("short pad write (copy)");
		if (fwrite(pack.data + en->payload_offset, 1, (size_t)en->payload_bytes, of) != en->payload_bytes)
			die("short payload write (copy)");
		out_size = next + en->payload_bytes;
		en->payload_offset = next;
	}
	if (fseek(of, (long)directory_offset, SEEK_SET) != 0)
		die("seek directory failed");
	for (i = 0; i < tensor_count; i++)
	{
		uint8_t ebuf[PACK_ENTRY_BYTES];
		uint8_t *e = ebuf;
		*(uint32_t *)(e + 0) = entries[i].tensor_kind;
		*(uint32_t *)(e + 4) = entries[i].layer_index;
		*(uint32_t *)(e + 8) = entries[i].weight_format;
		*(uint32_t *)(e + 12) = entries[i].rows;
		*(uint32_t *)(e + 16) = entries[i].columns;
		*(uint32_t *)(e + 20) = entries[i].scale_group_size;
		*(uint64_t *)(e + 24) = entries[i].payload_offset;
		*(uint64_t *)(e + 32) = entries[i].payload_bytes;
		*(uint64_t *)(e + 40) = entries[i].scale_offset;
		*(uint64_t *)(e + 48) = entries[i].scale_bytes;
		if (fwrite(ebuf, 1, PACK_ENTRY_BYTES, of) != PACK_ENTRY_BYTES)
			die("short directory write");
	}
	if (fseek(of, 112, SEEK_SET) != 0)
		die("seek header failed");
	if (fwrite(&out_size, 8, 1, of) != 1)
		die("short header patch");
	fclose(of);
	fprintf(stderr, "compressed %u tensors: %llu -> %llu bytes (%.3fx); pack %llu -> %llu bytes\n",
	        compressed_count,
	        (unsigned long long)compressed_before, (unsigned long long)compressed_after,
	        compressed_before ? (double)compressed_before / compressed_after : 0.0,
	        (unsigned long long)pack.size, (unsigned long long)out_size);
	return 0;
}
