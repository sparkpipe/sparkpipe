#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_device.h"
#include "inference/runner/device_ops.h"

enum { SLOTS = 64 };
static uint32_t HEADS = 24, KV = 4, DIM = 256;
static double SCALE = 1.0 / 16.0;

static SparkDeviceOps ops;
static SparkDeviceQueue *queue;

static uint16_t ToBf16(float value)
{
	uint32_t bits;
	memcpy(&bits, &value, sizeof(bits));
	return (uint16_t)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

static double Bf16(uint16_t bits)
{
	uint32_t word = (uint32_t)bits << 16;
	float value;
	memcpy(&value, &word, sizeof(value));
	return value;
}

static void *Shared(SparkDeviceBuffer *buffer, uint64_t bytes, const char *name)
{
	if ( SparkDeviceAllocate(buffer, SPARK_DEVICE_SPACE_SHARED_COHERENT, bytes, name) != SPARK_STATUS_OK )
		exit(1);
	memset(buffer->host, 0, bytes);
	return buffer->host;
}

static SparkTensor Whole(SparkDeviceBuffer *buffer)
{
	return SparkTensorAt(buffer, 0u, buffer->bytes);
}

static float *Load(const char *path, uint64_t count)
{
	float *data = (float *)malloc(count * sizeof(float));
	FILE *file = fopen(path, "rb");
	if ( file == 0 || data == 0 || fread(data, sizeof(float), count, file) != count )
	{
		fprintf(stderr, "cannot read %s\n", path);
		exit(1);
	}
	fclose(file);
	return data;
}

int main(int argc, char **argv)
{
	const char *prefix = argv[1], *label = argv[2], *outpath = argv[3];
	const uint32_t total = (uint32_t)atoi(argv[4]), split = (uint32_t)atoi(argv[5]);
	if ( argc > 9 )
	{
		HEADS = (uint32_t)atoi(argv[6]); KV = (uint32_t)atoi(argv[7]); DIM = (uint32_t)atoi(argv[8]); SCALE = atof(argv[9]);
	}
	const uint32_t pages = (total + SLOTS - 1u) / SLOTS;
	char path[1024];
	float *q, *k, *v;
	SparkDeviceBuffer pool, table, seq, pos, keys, values, query, gate, partial, output, one_seq, one_pos;
	SparkOpsPaged paged;
	SparkKvParams store;
	SparkAttentionParams params;
	uint32_t row, head, d, j, band;
	FILE *dump = fopen(outpath, "wb");
	if ( SparkDeviceInitialize(0) != SPARK_STATUS_OK || SparkDeviceQueueCreate(&queue, "real attention") != SPARK_STATUS_OK ||
		SparkDeviceOpsOpen(&ops, SPARK_KERNEL_LIBRARY) != SPARK_STATUS_OK )
		return 1;
	snprintf(path, sizeof(path), "%s.q", prefix); q = Load(path, (uint64_t)total * HEADS * DIM);
	snprintf(path, sizeof(path), "%s.k", prefix); k = Load(path, (uint64_t)total * KV * DIM);
	snprintf(path, sizeof(path), "%s.v", prefix); v = Load(path, (uint64_t)total * KV * DIM);
	Shared(&pool, (uint64_t)pages * SLOTS * 2u * KV * DIM * 2u, "pool");
	{
		uint32_t *t = (uint32_t *)Shared(&table, pages * 4u, "table");
		for ( j = 0u; j < pages; j++ )
			t[j] = j;
	}
	{
		uint16_t *kk = (uint16_t *)Shared(&keys, (uint64_t)total * KV * DIM * 2u, "keys");
		uint16_t *vv = (uint16_t *)Shared(&values, (uint64_t)total * KV * DIM * 2u, "values");
		uint32_t *ss = (uint32_t *)Shared(&one_seq, total * 4u, "store seq");
		uint32_t *pp = (uint32_t *)Shared(&one_pos, total * 4u, "store pos");
		for ( j = 0u; j < (uint64_t)total * KV * DIM; j++ )
		{
			kk[j] = ToBf16(k[j]);
			vv[j] = ToBf16(v[j]);
		}
		for ( j = 0u; j < total; j++ )
		{
			ss[j] = 0u;
			pp[j] = j;
		}
	}
	memset(&paged, 0, sizeof(paged));
	paged.pool = Whole(&pool); paged.page_table = Whole(&table); paged.sequence_of_row = Whole(&one_seq); paged.positions = Whole(&one_pos);
	paged.page_slots = SLOTS; paged.pages_per_sequence = pages; paged.pool_pages = pages; paged.kv_heads = KV;
	memset(&store, 0, sizeof(store));
	store.rows = total; store.kv_heads = KV; store.head_dimension = DIM; store.page_slots = SLOTS; store.pages_per_sequence = pages;
	store.key_stride = KV * DIM; store.value_stride = KV * DIM; store.value_offset = 0u; store.pool_pages = pages;
	if ( SparkOpsKvStore(&ops, queue, &paged, Whole(&keys), Whole(&values), &store) != SPARK_STATUS_OK ||
		SparkDeviceQueueWait(queue, SPARK_DEVICE_WAIT_FOREVER) != SPARK_STATUS_OK )
		return 1;
	for ( band = 0u; band < 2u; band++ )
	{
		const uint32_t first = band == 0u ? 0u : split, count = band == 0u ? split : total - split;
		const uint32_t keys_per_partition = band == 0u && getenv("REAL_ATTENTION_DECODE_ONLY") == 0 ? 1024u : 128u;
		double sq = 0.0, worst = 0.0, signed_sum = 0.0, acc_sq = 0.0, sum_sq = 0.0, den_sq = 0.0;
		uint64_t split_n = 0u, den_n = 0u;
		uint64_t n = 0u, mismatch = 0u;
		uint16_t *qq = (uint16_t *)Shared(&query, (uint64_t)count * HEADS * DIM * 2u, "query");
		uint32_t *ss = (uint32_t *)Shared(&seq, count * 4u, "seq"), *pp = (uint32_t *)Shared(&pos, count * 4u, "pos");
		Shared(&gate, (uint64_t)count * HEADS * DIM * 4u, "gate");
		uint16_t *out = (uint16_t *)Shared(&output, (uint64_t)count * HEADS * DIM * 2u, "out");
		Shared(&partial, SparkOpsAttentionScratchBytes(count, HEADS, DIM, total), "partial");
		for ( row = 0u; row < count; row++ )
		{
			ss[row] = 0u;
			pp[row] = first + row;
			for ( j = 0u; j < HEADS * DIM; j++ )
				qq[(uint64_t)row * HEADS * DIM + j] = ToBf16(q[(uint64_t)(first + row) * HEADS * DIM + j]);
		}
		paged.sequence_of_row = Whole(&seq);
		paged.positions = Whole(&pos);
		memset(&params, 0, sizeof(params));
		params.rows = count; params.heads = HEADS; params.kv_heads = KV; params.head_dimension = DIM; params.page_slots = SLOTS;
		params.pages_per_sequence = pages; params.pool_pages = pages; params.query_stride = HEADS * DIM; params.gate_stride = 2u * HEADS * DIM;
		params.output_stride = HEADS * DIM; params.scale = (float)SCALE; params.output_gate = 0u; params.gate_offset = DIM; params.gate_head_stride = 2u * DIM;
		params.row_exact = band == 0u && getenv("REAL_ATTENTION_DECODE_ONLY") == 0 ? 0u : 1u;
		if ( SparkOpsAttention(&ops, queue, &paged, Whole(&query), Whole(&gate), Whole(&partial), Whole(&output), &params, pp, ss) != SPARK_STATUS_OK ||
			SparkDeviceQueueWait(queue, SPARK_DEVICE_WAIT_FOREVER) != SPARK_STATUS_OK )
			return 1;
		{
			const float *records = (const float *)partial.host;
			const uint32_t partitions = (total - 0u + keys_per_partition - 1u) / keys_per_partition;
			const uint32_t used_partitions = ((first + count) + keys_per_partition - 1u) / keys_per_partition;
			double *score = (double *)malloc(total * sizeof(double));
			for ( row = 0u; row < count; row++ )
				for ( head = 0u; head < HEADS; head++ )
				{
					const uint32_t kv = head / (HEADS / KV), position = first + row, used = (position + keys_per_partition) / keys_per_partition;
					const uint32_t parts = band == 0u ? used_partitions : partitions;
					const float *rec = records + ((uint64_t)row * HEADS + head) * parts * (DIM + 2u);
					double best = -INFINITY, sum = 0.0, top = -INFINITY, den = 0.0;
					(void)parts;
					for ( j = 0u; j <= position; j++ )
					{
						double dot = 0.0;
						for ( d = 0u; d < DIM; d++ )
							dot += Bf16(qq[((uint64_t)row * HEADS + head) * DIM + d]) * Bf16(ToBf16(k[((uint64_t)j * KV + kv) * DIM + d]));
						score[j] = dot * (double)(float)SCALE;
						best = score[j] > best ? score[j] : best;
					}
					for ( j = 0u; j <= position; j++ )
						sum += (score[j] = exp(score[j] - best));
					for ( j = 0u; j < used; j++ )
						top = rec[j * (DIM + 2u)] > top ? rec[j * (DIM + 2u)] : top;
					for ( j = 0u; j < used; j++ )
						if ( rec[j * (DIM + 2u)] != -INFINITY )
							den += exp((double)rec[j * (DIM + 2u)] - top) * rec[j * (DIM + 2u) + 1u];
					{
						double exact_den = 0.0;
						for ( j = 0u; j <= position; j++ )
							exact_den += exp(log(score[j]) + best - top);
						den_sq += pow((den - exact_den) / exact_den * 16777216.0, 2.0);
						den_n++;
					}
					for ( d = 0u; d < DIM; d++ )
					{
						double exact = 0.0, magnitude = 0.0, num = 0.0;
						float value;
						for ( j = 0u; j <= position; j++ )
						{
							const double vv = Bf16(ToBf16(v[((uint64_t)j * KV + kv) * DIM + d]));
							exact += score[j] / sum * vv;
							magnitude += score[j] / sum * fabs(vv);
						}
						for ( j = 0u; j < used; j++ )
							if ( rec[j * (DIM + 2u)] != -INFINITY )
								num += exp((double)rec[j * (DIM + 2u)] - top) * rec[j * (DIM + 2u) + 2u + d];
						const double e = (num / den - exact) / magnitude * 16777216.0;
						sq += e * e; signed_sum += e; n++;
						worst = fabs(e) > worst ? fabs(e) : worst;
						mismatch += out[((uint64_t)row * HEADS + head) * DIM + d] != ToBf16((float)exact) ? 1u : 0u;
						if ( used == 1u )
						{
							double acc_exact = 0.0, acc_magnitude = 0.0, sum_exact = 0.0;
							for ( j = 0u; j <= position; j++ )
							{
								const double w = exp(log(score[j]) + best - (double)rec[0]), vv = Bf16(ToBf16(v[((uint64_t)j * KV + kv) * DIM + d]));
								acc_exact += w * vv; acc_magnitude += w * fabs(vv); sum_exact += w;
							}
							acc_sq += pow((rec[2u + d] - acc_exact) / acc_magnitude * 16777216.0, 2.0);
							sum_sq += pow((rec[1] - sum_exact) / sum_exact * 16777216.0, 2.0);
							split_n++;
						}
						value = (float)(num / den);
						fwrite(&value, sizeof(value), 1u, dump);
						value = (float)exact;
						fwrite(&value, sizeof(value), 1u, dump);
						value = (float)magnitude;
						fwrite(&value, sizeof(value), 1u, dump);
					}
				}
			free(score);
		}
		printf("%-8s %s combined weight sum against float64 / (2^-24 sum): rms %.2f\n", label, band == 0u ? "prefill" : "decode ", sqrt(den_sq / den_n));
		if ( split_n != 0u )
			printf("%-8s %s single-partition rows: accumulator error / (2^-24 sum w|v|) rms %.2f, weight-sum error / (2^-24 sum) rms %.2f\n", label,
				band == 0u ? "prefill" : "decode ", sqrt(acc_sq / split_n), sqrt(sum_sq / split_n));
		printf("%-8s %s rows %u (positions %u..%u): record error / (2^-24 sum p|v|) rms %.2f max %.1f mean %+.3f; bf16 outputs not the correctly rounded value: %.4f%%\n",
			label, band == 0u ? "prefill" : "decode ", count, first, first + count - 1u, sqrt(sq / n), worst, signed_sum / n, 100.0 * mismatch / n);
		fflush(stdout);
		SparkDeviceFree(&query); SparkDeviceFree(&seq); SparkDeviceFree(&pos); SparkDeviceFree(&gate); SparkDeviceFree(&output); SparkDeviceFree(&partial);
	}
	fclose(dump);
	return 0;
}
