#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <pthread.h>
#include <unistd.h>

#include "runtime/launch.h"
#include "inference/kernels/dtype.cuh"
#include "sparkpipe/spark_pack_window.h"
#include "sparkpipe/spark_pass_plan.h"
#include "sparkpipe/spark_stage_runner.h"
#include "sparkpipe/spark_stage_runner_model.h"
#include "sparkpipe/spark_tp_device_collective.h"
#include "sparkpipe/spark_weight_stream.h"
#include "sparkpipe/spark_weightd_lazy_pack.h"
#include "sparkpipe/spark_weightd_map.h"

#define LM_PASS_RANKS SPARK_PASS_PLAN_MAX_RANKS
#define LM_PASS_MAX_LAYERS 128u
#define LM_PASS_RANGE_GAP 4096u
#define LM_PASS_HELP_MIN_SHARE 0.005
#define LM_PASS_HELP_LAYERS 6u
#define LM_PASS_STREAM_CHUNK (4u << 20)
#define LM_PASS_STREAM_DEPTH 32u
#define LM_PASS_MAX_ALLOCATIONS 160u
#define LM_PASS_GATHER_BYTES 4096u
#define LM_PASS_ROUND_BLOCK_BYTES (2u << 20)
#define LM_PASS_ROUND_RING 4u
#define LM_PASS_DIRECT_ALIGN 4096u
#define LM_PASS_DIRECT_CHUNK (64ull << 20)
#define LM_PASS_SPENT 7u
#define LM_PASS_THREADS 256u

static_assert(LM_PASS_RANKS <= SPARK_WEIGHT_STREAM_MAX_RANKS && LM_PASS_RANKS <= SPARK_TP_DEVICE_COLLECTIVE_MAX_DEGREE, "every pass rank is a stream and collective rank");

__global__ static void LmPassSumKernel(const uint8_t *blocks, uint64_t block_bytes, uint64_t pitch, uint64_t *sums)
{
	__shared__ unsigned long long partial[256];
	const uint32_t *words = (const uint32_t *)(blocks + (uint64_t)blockIdx.x * pitch);
	unsigned long long sum = 0ull;
	for (uint64_t index = threadIdx.x; index < block_bytes / 4u; index += blockDim.x)
		sum += (unsigned long long)words[index] * (2u * (uint32_t)index + 1u);
	partial[threadIdx.x] = sum;
	__syncthreads();
	for (uint32_t half = blockDim.x / 2u; half != 0u; half /= 2u)
	{
		if ( threadIdx.x < half )
			partial[threadIdx.x] += partial[threadIdx.x + half];
		__syncthreads();
	}
	if ( threadIdx.x == 0u )
		sums[blockIdx.x] = partial[0];
}

__global__ static void LmPassHealthKernel(const uint16_t *hidden, uint64_t elements, uint32_t *health)
{
	uint32_t nonfinite = 0u, top = 0u;
	for (uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < elements; index += (uint64_t)gridDim.x * blockDim.x)
	{
		const uint32_t bits = (uint32_t)hidden[index] & 0x7fffu;
		if ( bits >= 0x7f80u )
			nonfinite++;
		else
			top = bits > top ? bits : top;
	}
	if ( nonfinite != 0u )
		atomicAdd(&health[0], nonfinite);
	atomicMax(&health[1], top);
}

struct LmPassVocab
{
	const uint16_t *slice[LM_PASS_RANKS];
	uint32_t slice_rows;
	uint32_t slices;
	uint32_t width;
};

__global__ static void LmPassEmbedKernel(LmPassVocab vocab, const uint32_t *tokens, uint16_t *hidden, uint32_t rows)
{
	const uint32_t row = blockIdx.x, token = row < rows ? tokens[row] : 0u, slice = token / vocab.slice_rows;
	const uint4 *source = slice < vocab.slices ? (const uint4 *)(vocab.slice[slice] + (uint64_t)(token % vocab.slice_rows) * vocab.width) : 0;
	uint4 *target = (uint4 *)(hidden + (uint64_t)row * vocab.width), zero = {0u,0u,0u,0u};
	if ( row >= rows )
		return;
	for (uint32_t index = threadIdx.x; index < vocab.width * sizeof(uint16_t) / 16u; index += blockDim.x)
		target[index] = source != 0 ? source[index] : zero;
}

static int32_t LmPassEmbed(const LmPassVocab *vocab, const uint32_t *tokens, uint16_t *hidden, uint32_t rows, cudaStream_t stream)
{
	if ( vocab == 0 || vocab->slice_rows == 0u || vocab->slices == 0u || vocab->slices > LM_PASS_RANKS || vocab->width == 0u || (vocab->width * sizeof(uint16_t)) % 16u != 0u ||
		tokens == 0 || hidden == 0 || rows == 0u )
		return(LM_LAUNCH_ERR_SHAPE);
	LM_LAUNCH((LmPassEmbedKernel), rows, LM_PASS_THREADS, 0, stream, *vocab,tokens,hidden,rows);
	return(LM_LAUNCH_OK);
}

typedef struct LmPassRoundContext
{
	const SparkStageRunnerServices *services;
	const char *tag;
	SparkStatus status;
	uint8_t *send_stage[LM_PASS_ROUND_RING];
	uint8_t *receive_stage[LM_PASS_ROUND_RING];
	uint32_t next_stage;
	uint32_t rank;
	uint32_t row_elements;
	uint64_t *sums;
	uint64_t *sums_host;
	uint64_t mismatches;
	uint64_t late;
} LmPassRoundContext;

static void LmPassRound(void *context, void *stream, uint32_t operation, uint32_t rows, uint32_t row_elements, const void *send, void *receive)
{
	LmPassRoundContext *round = (LmPassRoundContext *)context;
	uint32_t split = 1u;
	SparkStatus status;
	if ( round == 0 || round->services == 0 || round->services->round == 0 )
		return;
	while ( row_elements / split > round->row_elements || row_elements % split != 0u )
		split++;
	if ( (uint64_t)rows * split > UINT32_MAX )
	{
		if ( round->status == SPARK_STATUS_OK )
			round->status = SPARK_STATUS_CAPACITY_EXCEEDED;
		return;
	}
	rows *= split;
	row_elements /= split;
	{
		const uint64_t row_bytes = (uint64_t)row_elements * sizeof(uint16_t);
		const uint32_t block = (uint32_t)(LM_PASS_ROUND_BLOCK_BYTES / row_bytes != 0u ? LM_PASS_ROUND_BLOCK_BYTES / row_bytes : 1u);
		const uint32_t peers = operation == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL ? LM_PASS_RANKS : 1u;
		uint32_t first;
		if ( round->send_stage[LM_PASS_ROUND_RING - 1u] == 0 || round->receive_stage[LM_PASS_ROUND_RING - 1u] == 0 || round->sums_host == 0 ||
			row_bytes > LM_PASS_ROUND_BLOCK_BYTES )
		{
			if ( round->status == SPARK_STATUS_OK )
				round->status = SPARK_STATUS_CAPACITY_EXCEEDED;
			return;
		}
		for (first = 0u; first < rows && round->status == SPARK_STATUS_OK; first += block)
		{
			const uint32_t count = rows - first < block ? rows - first : block;
			const uint64_t width = (uint64_t)count * row_bytes, pitch = (uint64_t)rows * row_bytes;
			const uint32_t ring = round->next_stage++ % LM_PASS_ROUND_RING;
			uint8_t *stage = round->send_stage[ring], *landing = round->receive_stage[ring];
			if ( cudaMemcpy2DAsync(stage, (size_t)width, (const uint8_t *)send + (uint64_t)first * row_bytes, (size_t)pitch, (size_t)width, peers,
					cudaMemcpyDeviceToDevice, (cudaStream_t)stream) != cudaSuccess )
			{
				round->status = SPARK_STATUS_IO_ERROR;
				return;
			}
			LmPassSumKernel<<<peers, 256u, 0, (cudaStream_t)stream>>>(stage, width, width, round->sums);
			status = round->services->round(round->services->context, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, SPARK_STAGE_RUNNER_ROUND_ALL, 1u, 4u * LM_PASS_RANKS,
				round->sums, round->sums + LM_PASS_RANKS);
			if ( status == SPARK_STATUS_OK )
				status = round->services->round(round->services->context, stream, operation, SPARK_STAGE_RUNNER_ROUND_ALL, count, row_elements, stage, landing);
			for (uint32_t attempt = 0u; status == SPARK_STATUS_OK; attempt++)
			{
				uint32_t peer, bad = UINT32_MAX;
				LmPassSumKernel<<<LM_PASS_RANKS, 256u, 0, (cudaStream_t)stream>>>(landing, width, width, round->sums + LM_PASS_RANKS + LM_PASS_RANKS * LM_PASS_RANKS);
				if ( cudaMemcpyAsync(round->sums_host, round->sums, (size_t)(2u * LM_PASS_RANKS + LM_PASS_RANKS * LM_PASS_RANKS) * sizeof(uint64_t),
						cudaMemcpyDeviceToHost, (cudaStream_t)stream) != cudaSuccess || cudaStreamSynchronize((cudaStream_t)stream) != cudaSuccess )
				{
					status = SPARK_STATUS_IO_ERROR;
					break;
				}
				for (peer = 0u; peer < LM_PASS_RANKS && bad == UINT32_MAX; peer++)
					if ( round->sums_host[LM_PASS_RANKS + LM_PASS_RANKS * LM_PASS_RANKS + peer] !=
						round->sums_host[LM_PASS_RANKS + (uint64_t)peer * LM_PASS_RANKS + (peers == 1u ? 0u : round->rank)] )
						bad = peer;
				if ( bad == UINT32_MAX )
				{
					round->late += attempt != 0u ? 1u : 0u;
					break;
				}
				if ( attempt == 0u )
					round->mismatches++;
				if ( attempt >= 2000u )
				{
					fprintf(stderr, "%s: streamed exchange op=%u rows=%u: the block from rank %u does not match its sender's checksum after 2 s\n", round->tag, operation, count, bad);
					status = SPARK_STATUS_VALIDATION_FAILED;
					break;
				}
				usleep(1000u);
			}
			if ( status == SPARK_STATUS_OK && cudaMemcpy2DAsync((uint8_t *)receive + (uint64_t)first * row_bytes, (size_t)pitch, landing, (size_t)width,
					(size_t)width, LM_PASS_RANKS, cudaMemcpyDeviceToDevice, (cudaStream_t)stream) != cudaSuccess )
				status = SPARK_STATUS_IO_ERROR;
			if ( status != SPARK_STATUS_OK && round->status == SPARK_STATUS_OK )
				round->status = status;
		}
	}
}

struct LmPassTensor
{
	uint64_t start;
	uint64_t bytes;
	uint32_t expert;
	uint32_t reserved0;
};

static int32_t LmPassLayerRanges(const LmPassTensor *tensors, uint32_t tensor_count, SparkPackWindowLayer *ranges)
{
	uint64_t *start, *end, expert_start = UINT64_MAX, expert_end = 0u;
	uint32_t count = 0u, index, other, dense;
	int32_t result = 0;
	if ( tensors == 0 || ranges == 0 )
		return -1;
	memset(ranges, 0, sizeof(*ranges));
	start = (uint64_t *)calloc((size_t)tensor_count + 1u, sizeof(uint64_t));
	end = (uint64_t *)calloc((size_t)tensor_count + 1u, sizeof(uint64_t));
	if ( start == 0 || end == 0 )
	{
		free(start);
		free(end);
		return -1;
	}
	for ( index = 0u; index < tensor_count; index++ )
	{
		const uint64_t first = tensors[index].start, last = first + tensors[index].bytes;
		if ( tensors[index].expert != 0u )
		{
			expert_start = first < expert_start ? first : expert_start;
			expert_end = last > expert_end ? last : expert_end;
			continue;
		}
		start[count] = first;
		end[count] = last;
		count++;
	}
	for ( index = 1u; index < count; index++ )
		for ( other = index; other > 0u && start[other - 1u] > start[other]; other-- )
		{
			const uint64_t swap_start = start[other], swap_end = end[other];
			start[other] = start[other - 1u];
			end[other] = end[other - 1u];
			start[other - 1u] = swap_start;
			end[other - 1u] = swap_end;
		}
	if ( expert_end != 0u )
	{
		ranges->experts.offset = expert_start;
		ranges->experts.bytes = expert_end - expert_start;
	}
	for ( index = 0u, dense = 0u; index < count && result == 0; index++ )
	{
		const uint32_t crosses = expert_end != 0u && dense != 0u &&
			ranges->dense[dense - 1u].offset < expert_start && start[index] >= expert_end ? 1u : 0u;
		if ( expert_end != 0u && start[index] < expert_end && end[index] > expert_start )
		{
			result = -1;
			break;
		}
		if ( dense != 0u && crosses == 0u && start[index] <= ranges->dense[dense - 1u].offset + ranges->dense[dense - 1u].bytes + LM_PASS_RANGE_GAP )
		{
			const uint64_t range_end = ranges->dense[dense - 1u].offset + ranges->dense[dense - 1u].bytes;
			if ( end[index] > range_end )
				ranges->dense[dense - 1u].bytes = end[index] - ranges->dense[dense - 1u].offset;
			continue;
		}
		if ( dense == SPARK_PACK_WINDOW_MAX_DENSE_RANGES )
		{
			result = -1;
			break;
		}
		ranges->dense[dense].offset = start[index];
		ranges->dense[dense].bytes = end[index] - start[index];
		dense++;
	}
	free(start);
	free(end);
	ranges->dense_count = dense;
	return result == 0 && dense != 0u ? 0 : -1;
}

struct LmPassStep
{
	uint32_t pass;
	uint32_t passes;
	uint64_t base;
	uint64_t pass_tokens;
	uint32_t rows;
	uint32_t pass_rows;
	uint64_t node_first;
	uint64_t node_tokens;
	uint32_t sequence_slot;
	uint32_t reserved0;
};

struct LmPass
{
	SparkStageRunnerStreamedPrefillConfig config;
	const char *tag;
	SparkWeightdLazyPack *lazy_pack;
	int device;
	uint32_t rank;
	uint32_t layers;
	uint32_t node_rows;
	uint32_t row_align;
	uint32_t hidden;
	uint32_t opened;
	uint32_t tag_counter;
	uint64_t pack_file_bytes;
	uint64_t pass_rows_max;
	int partner_fd;
	int own_fd;
	int partner_direct_fd;
	int own_direct_fd;
	uint8_t *bounce;
	const uint8_t *pool;
	uint8_t *region;
	uint64_t region_bytes;
	uint64_t engine_bytes;
	SparkWeightStream *stream;
	SparkWeightStream *rows_stream;
	uint8_t *rows_region;
	uint64_t rows_region_bytes;
	uint64_t rows_block_bytes;
	uint32_t rows_tag;
	uint32_t streams_experts;
	SparkPackWindow window;
	SparkPackWindowLayer ranges[LM_PASS_MAX_LAYERS];
	uint32_t experts_per_layer[LM_PASS_MAX_LAYERS];
	uint32_t balances[LM_PASS_MAX_LAYERS];
	LmPassTensor *tensors;
	uint32_t tensor_first[LM_PASS_MAX_LAYERS + 1u];
	SparkPackWindowLayer vocab;
	uint32_t vocab_rows;
	uint32_t *tokens;
	uint32_t *positions;
	uint32_t *pass_positions;
	uint8_t *gather_device;
	uint32_t *host_staging;
	uint32_t *health;
	cudaStream_t gather_stream;
	cudaStream_t copy_stream;
	cudaEvent_t dense_done[SPARK_PACK_WINDOW_MAX_DENSE_BUFFERS];
	cudaEvent_t slot_done[SPARK_PACK_WINDOW_MAX_EXPERT_SLOTS];
	cudaEvent_t marks[LM_PASS_MAX_LAYERS][3];
	cudaEvent_t own_marks[LM_PASS_MAX_LAYERS][2];
	float *balance_device;
	SparkPassBalance balance;
	double pass_attention[LM_PASS_RANKS];
	uint64_t pass_helped;
	pthread_mutex_t lock;
	pthread_cond_t changed;
	uint32_t synchronization;
	uint32_t stream_passes;
	SparkStatus stream_status;
	uint64_t dense_published;
	uint64_t dense_released;
	uint64_t expert_published;
	uint64_t expert_released;
	uint8_t *dense_base[SPARK_PACK_WINDOW_MAX_DENSE_BUFFERS][LM_PASS_RANKS];
	uint32_t slot_source[SPARK_PACK_WINDOW_MAX_EXPERT_SLOTS];
	const uint8_t *slot_block[SPARK_PACK_WINDOW_MAX_EXPERT_SLOTS];
	LmPassRoundContext round;
	uint64_t spent[LM_PASS_SPENT];
	void *allocations[LM_PASS_MAX_ALLOCATIONS];
	uint32_t allocation_count;
};

struct LmPassModelLayout
{
	const char *tag;
	SparkWeightdLazyPack *lazy_pack;
	const char *own_pack_path;
	uint64_t pack_file_bytes;
	int device;
	uint32_t rank;
	uint32_t layers;
	uint32_t row_align;
	uint32_t hidden;
	uint32_t row_elements;
	const LmPassTensor *tensors;
	const uint32_t *tensor_first;
	const uint32_t *experts_per_layer;
	const uint32_t *balances;
	uint64_t vocab_offset;
	uint64_t vocab_bytes;
	uint32_t vocab_rows;
	uint64_t rows_block_bytes;
};

template<class T>
static T *LmPassAlloc(LmPass *p, uint64_t count)
{
	void *pointer = 0;
	const uint64_t bytes = count * sizeof(T) + 256u;
	if ( p->allocation_count >= LM_PASS_MAX_ALLOCATIONS || cudaMalloc(&pointer, bytes) != cudaSuccess )
		return 0;
	if ( cudaMemset(pointer, 0, bytes) != cudaSuccess )
	{
		cudaFree(pointer);
		return 0;
	}
	p->allocations[p->allocation_count++] = pointer;
	p->engine_bytes += bytes;
	return (T *)pointer;
}

static uint64_t LmPassNowNs(void)
{
	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC, &now);
	return (uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec;
}

static SparkStatus LmPassCopyTensor(const LmPass *p, uint64_t start, uint64_t bytes, uint64_t offset, uint64_t length, uint8_t *destination)
{
	const uint64_t first = start > offset ? start : offset, last = start + bytes < offset + length ? start + bytes : offset + length;
	const void *slice = 0;
	if ( first >= last )
		return SPARK_STATUS_OK;
	if ( SparkWeightdLazyPackSlice(p->lazy_pack, start, bytes, &slice) != SPARK_STATUS_OK || slice == 0 ||
		cudaMemcpyAsync(destination + (first - offset), (const uint8_t *)slice + (first - start), (size_t)(last - first), cudaMemcpyDefault, p->copy_stream) != cudaSuccess ||
		cudaStreamSynchronize(p->copy_stream) != cudaSuccess )
	{
		fprintf(stderr, "%s: streamed prefill cannot read the own tensor at %llu (%llu bytes) from the spine\n", p->tag, (unsigned long long)start,
			(unsigned long long)bytes);
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	}
	return SPARK_STATUS_OK;
}

static SparkStatus LmPassPreadSpan(int fd, uint64_t pack_offset, uint64_t bytes, uint8_t *destination)
{
	uint64_t done = 0u;
	while ( done < bytes )
	{
		const ssize_t got = pread(fd, destination + done, (size_t)(bytes - done < (1ull << 30) ? bytes - done : (1ull << 30)), (off_t)(pack_offset + done));
		if ( got <= 0 )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		done += (uint64_t)got;
	}
	return SPARK_STATUS_OK;
}

static SparkStatus LmPassPread(int fd, int direct_fd, uint8_t *bounce, uint64_t pack_offset, uint64_t bytes, uint8_t *destination)
{
	uint64_t done = 0u;
	if ( direct_fd < 0 || bounce == 0 )
	{
		const SparkStatus status = LmPassPreadSpan(fd, pack_offset, bytes, destination);
		(void)posix_fadvise(fd, (off_t)pack_offset, (off_t)bytes, POSIX_FADV_DONTNEED);
		return status;
	}
	while ( done < bytes )
	{
		const uint64_t start = (pack_offset + done) / LM_PASS_DIRECT_ALIGN * LM_PASS_DIRECT_ALIGN, skip = pack_offset + done - start;
		const uint64_t want = bytes - done < LM_PASS_DIRECT_CHUNK - skip ? bytes - done : LM_PASS_DIRECT_CHUNK - skip;
		const uint64_t span = (skip + want + LM_PASS_DIRECT_ALIGN - 1u) / LM_PASS_DIRECT_ALIGN * LM_PASS_DIRECT_ALIGN;
		uint64_t got = 0u;
		while ( got < skip + want )
		{
			const ssize_t part = pread(direct_fd, bounce + got, (size_t)(span - got), (off_t)(start + got));
			if ( part <= 0 )
				SPARK_FAIL(SPARK_STATUS_IO_ERROR);
			got += (uint64_t)part;
		}
		memcpy(destination + done, bounce + skip, (size_t)want);
		done += want;
	}
	return SPARK_STATUS_OK;
}

static SparkStatus LmPassReadOwn(void *context, uint64_t pack_offset, uint64_t bytes, uint8_t *destination)
{
	LmPass *p = (LmPass *)context;
	uint32_t layer, index;
	SparkStatus status = SPARK_STATUS_OK;
	if ( pack_offset + bytes > p->pack_file_bytes )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	for (layer = 0u; layer < p->layers; layer++)
		if ( p->ranges[layer].experts.bytes != 0u && pack_offset >= p->ranges[layer].experts.offset &&
			pack_offset + bytes <= p->ranges[layer].experts.offset + p->ranges[layer].experts.bytes )
		{
			if ( p->pool == 0 )
			{
				const uint64_t mark = LmPassNowNs();
				status = LmPassPread(p->own_fd, p->own_direct_fd, p->bounce, pack_offset, bytes, destination);
				p->spent[6] += LmPassNowNs() - mark;
				return status;
			}
			if ( cudaMemcpyAsync(destination, p->pool + pack_offset, (size_t)bytes, cudaMemcpyDefault, p->copy_stream) != cudaSuccess ||
				cudaStreamSynchronize(p->copy_stream) != cudaSuccess )
				SPARK_FAIL(SPARK_STATUS_IO_ERROR);
			return SPARK_STATUS_OK;
		}
	if ( pack_offset == p->vocab.dense[0].offset && bytes == p->vocab.dense[0].bytes )
		return LmPassCopyTensor(p, pack_offset, bytes, pack_offset, bytes, destination);
	for (layer = 0u; layer < p->layers && status == SPARK_STATUS_OK; layer++)
		for (index = p->tensor_first[layer]; index < p->tensor_first[layer + 1u] && status == SPARK_STATUS_OK; index++)
		{
			const LmPassTensor *tensor = &p->tensors[index];
			if ( tensor->start < pack_offset + bytes && tensor->start + tensor->bytes > pack_offset &&
				!(p->ranges[layer].experts.bytes != 0u && tensor->start >= p->ranges[layer].experts.offset &&
				  tensor->start < p->ranges[layer].experts.offset + p->ranges[layer].experts.bytes) )
				status = LmPassCopyTensor(p, tensor->start, tensor->bytes, pack_offset, bytes, destination);
		}
	return status;
}

static SparkStatus LmPassReadPartner(void *context, uint64_t pack_offset, uint64_t bytes, uint8_t *destination)
{
	LmPass *p = (LmPass *)context;
	const uint64_t mark = LmPassNowNs();
	const SparkStatus status = LmPassPread(p->partner_fd, p->partner_direct_fd, p->bounce, pack_offset, bytes, destination);
	p->spent[6] += LmPassNowNs() - mark;
	return status;
}

static SparkStatus LmPassGather(void *context, const void *mine, void *all, uint64_t bytes_per_rank)
{
	LmPass *p = (LmPass *)context;
	uint8_t *send = p->gather_device, *receive = p->gather_device + LM_PASS_GATHER_BYTES;
	if ( bytes_per_rank == 0u || bytes_per_rank > LM_PASS_GATHER_BYTES || (bytes_per_rank % 2u) != 0u )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( cudaMemcpyAsync(send, mine, (size_t)bytes_per_rank, cudaMemcpyHostToDevice, p->gather_stream) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	LmPassRound(&p->round, (void *)p->gather_stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, 1u, (uint32_t)(bytes_per_rank / 2u), send, receive);
	if ( p->round.status != SPARK_STATUS_OK ||
		cudaMemcpyAsync(all, receive, (size_t)(bytes_per_rank * LM_PASS_RANKS), cudaMemcpyDeviceToHost, p->gather_stream) != cudaSuccess ||
		cudaStreamSynchronize(p->gather_stream) != cudaSuccess )
		SPARK_FAIL(p->round.status != SPARK_STATUS_OK ? p->round.status : SPARK_STATUS_IO_ERROR);
	return SPARK_STATUS_OK;
}

static void LmPassDestroy(LmPass *p)
{
	uint32_t index;
	if ( p == 0 )
		return;
	if ( p->stream != 0 )
		SparkWeightStreamClose(p->stream);
	if ( p->rows_stream != 0 )
		SparkWeightStreamClose(p->rows_stream);
	if ( p->region != 0 )
	{
		(void)cudaHostUnregister(p->region);
		free(p->region);
	}
	if ( p->rows_region != 0 )
	{
		(void)cudaHostUnregister(p->rows_region);
		free(p->rows_region);
	}
	if ( p->partner_fd >= 0 )
		close(p->partner_fd);
	if ( p->own_fd >= 0 )
		close(p->own_fd);
	if ( p->partner_direct_fd >= 0 )
		close(p->partner_direct_fd);
	if ( p->own_direct_fd >= 0 )
		close(p->own_direct_fd);
	free(p->bounce);
	free(p->tensors);
	for (index = 0u; index < SPARK_PACK_WINDOW_MAX_EXPERT_SLOTS; index++)
		if ( p->slot_done[index] != 0 )
			(void)cudaEventDestroy(p->slot_done[index]);
	for (index = 0u; index < SPARK_PACK_WINDOW_MAX_DENSE_BUFFERS; index++)
		if ( p->dense_done[index] != 0 )
			(void)cudaEventDestroy(p->dense_done[index]);
	for (index = 0u; index < LM_PASS_MAX_LAYERS * 3u; index++)
		if ( p->marks[index / 3u][index % 3u] != 0 )
			(void)cudaEventDestroy(p->marks[index / 3u][index % 3u]);
	for (index = 0u; index < LM_PASS_MAX_LAYERS * 2u; index++)
		if ( p->own_marks[index / 2u][index % 2u] != 0 )
			(void)cudaEventDestroy(p->own_marks[index / 2u][index % 2u]);
	if ( p->copy_stream != 0 )
		(void)cudaStreamDestroy(p->copy_stream);
	if ( p->synchronization != 0u )
	{
		(void)pthread_cond_destroy(&p->changed);
		(void)pthread_mutex_destroy(&p->lock);
	}
	if ( p->gather_stream != 0 )
		(void)cudaStreamDestroy(p->gather_stream);
	cudaFreeHost(p->host_staging);
	cudaFreeHost(p->health);
	cudaFreeHost(p->round.sums_host);
	for (index = 0u; index < p->allocation_count; index++)
		cudaFree(p->allocations[index]);
	p->allocation_count = 0u;
}

static SparkStatus LmPassBuffers(LmPass *p)
{
	const uint64_t pass = (uint64_t)LM_PASS_RANKS * p->node_rows;
	p->tokens = LmPassAlloc<uint32_t>(p, p->node_rows);
	p->positions = LmPassAlloc<uint32_t>(p, p->node_rows);
	p->pass_positions = LmPassAlloc<uint32_t>(p, pass);
	p->gather_device = LmPassAlloc<uint8_t>(p, (uint64_t)(LM_PASS_RANKS + 1u) * LM_PASS_GATHER_BYTES);
	for (uint32_t stage = 0u; stage < LM_PASS_ROUND_RING; stage++)
	{
		p->round.send_stage[stage] = LmPassAlloc<uint8_t>(p, (uint64_t)LM_PASS_RANKS * LM_PASS_ROUND_BLOCK_BYTES);
		p->round.receive_stage[stage] = LmPassAlloc<uint8_t>(p, (uint64_t)LM_PASS_RANKS * LM_PASS_ROUND_BLOCK_BYTES);
	}
	p->round.rank = p->rank;
	p->round.tag = p->tag;
	p->round.sums = LmPassAlloc<uint64_t>(p, 2u * LM_PASS_RANKS + LM_PASS_RANKS * LM_PASS_RANKS);
	if ( cudaMallocHost((void **)&p->round.sums_host, (size_t)(2u * LM_PASS_RANKS + LM_PASS_RANKS * LM_PASS_RANKS) * sizeof(uint64_t)) != cudaSuccess )
		p->round.sums_host = 0;
	p->balance_device = LmPassAlloc<float>(p, 4u + 4u * LM_PASS_RANKS);
	p->pass_rows_max = pass;
	if ( cudaMallocHost((void **)&p->host_staging, (size_t)(pass + 64u) * sizeof(uint32_t)) != cudaSuccess )
		p->host_staging = 0;
	if ( cudaHostAlloc((void **)&p->health, (size_t)(p->layers + 1u) * 2u * sizeof(uint32_t), cudaHostAllocMapped) != cudaSuccess )
		p->health = 0;
	if ( p->tokens == 0 || p->positions == 0 || p->pass_positions == 0 || p->gather_device == 0 || p->round.sums == 0 || p->round.sums_host == 0 ||
		p->balance_device == 0 || p->host_staging == 0 || p->health == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	for (uint32_t stage = 0u; stage < LM_PASS_ROUND_RING; stage++)
		if ( p->round.send_stage[stage] == 0 || p->round.receive_stage[stage] == 0 )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	return SPARK_STATUS_OK;
}

static SparkStatus LmPassCreate(LmPass *p, const SparkStageRunnerStreamedPrefillConfig *config, const LmPassModelLayout *layout)
{
	uint64_t dense_capacity, expert_capacity = 0u;
	uint32_t layer, index, tensor_count;
	int usable = 0;
	SparkStatus status;
	memset(p, 0, sizeof(*p));
	p->partner_fd = -1;
	p->own_fd = -1;
	p->partner_direct_fd = -1;
	p->own_direct_fd = -1;
	p->tag = layout->tag;
	if ( layout->layers == 0u || layout->layers > LM_PASS_MAX_LAYERS || layout->row_align == 0u || config->node_rows == 0u || (config->node_rows % layout->row_align) != 0u ||
		layout->hidden == 0u || layout->row_elements == 0u || layout->vocab_bytes == 0u || layout->vocab_rows == 0u || layout->tensors == 0 || layout->tensor_first == 0 )
	{
		fprintf(stderr, "%s: streamed prefill needs node rows in multiples of %u, at most %u layers and a vocab slice per rank\n", layout->tag, layout->row_align,
			LM_PASS_MAX_LAYERS);
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	if ( config->dense_buffers == 0u || config->dense_buffers > SPARK_PACK_WINDOW_MAX_DENSE_BUFFERS || config->expert_slots > SPARK_PACK_WINDOW_MAX_EXPERT_SLOTS )
	{
		fprintf(stderr, "%s: streamed prefill takes 1..%u dense buffers and up to %u expert slots (got %u and %u)\n", layout->tag, SPARK_PACK_WINDOW_MAX_DENSE_BUFFERS,
			SPARK_PACK_WINDOW_MAX_EXPERT_SLOTS, config->dense_buffers, config->expert_slots);
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	if ( cudaDeviceGetAttribute(&usable, cudaDevAttrCanUseHostPointerForRegisteredMem, layout->device) != cudaSuccess || usable == 0 )
	{
		fprintf(stderr, "%s: streamed prefill needs a device that reads registered host memory at the host address\n", layout->tag);
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	}
	p->config = *config;
	p->lazy_pack = layout->lazy_pack;
	p->pack_file_bytes = layout->pack_file_bytes;
	p->device = layout->device;
	p->rank = layout->rank;
	p->layers = layout->layers;
	p->node_rows = config->node_rows;
	p->row_align = layout->row_align;
	p->hidden = layout->hidden;
	p->round.row_elements = layout->row_elements;
	p->vocab_rows = layout->vocab_rows;
	p->vocab.dense_count = 1u;
	p->vocab.dense[0].offset = layout->vocab_offset;
	p->vocab.dense[0].bytes = layout->vocab_bytes;
	p->balance.ranks = LM_PASS_RANKS;
	dense_capacity = layout->vocab_bytes;
	tensor_count = layout->tensor_first[layout->layers];
	p->tensors = (LmPassTensor *)calloc((size_t)tensor_count + 1u, sizeof(LmPassTensor));
	if ( p->tensors == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	memcpy(p->tensors, layout->tensors, (size_t)tensor_count * sizeof(LmPassTensor));
	memcpy(p->tensor_first, layout->tensor_first, (size_t)(layout->layers + 1u) * sizeof(uint32_t));
	for (layer = 0u; layer < p->layers; layer++)
	{
		p->experts_per_layer[layer] = layout->experts_per_layer[layer];
		p->balances[layer] = layout->balances[layer];
		if ( LmPassLayerRanges(p->tensors + p->tensor_first[layer], p->tensor_first[layer + 1u] - p->tensor_first[layer], &p->ranges[layer]) != 0 ||
			(p->experts_per_layer[layer] != 0u && (p->ranges[layer].experts.bytes == 0u || (p->ranges[layer].experts.bytes % 2u) != 0u)) )
		{
			fprintf(stderr, "%s: streamed prefill cannot place layer %u of the pack in window ranges\n", p->tag, layer);
			LmPassDestroy(p);
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		}
		dense_capacity = SparkPackWindowDenseBytes(&p->ranges[layer]) > dense_capacity ? SparkPackWindowDenseBytes(&p->ranges[layer]) : dense_capacity;
		if ( p->experts_per_layer[layer] != 0u )
		{
			expert_capacity = p->ranges[layer].experts.bytes > expert_capacity ? p->ranges[layer].experts.bytes : expert_capacity;
			p->streams_experts = 1u;
		}
	}
	if ( p->streams_experts != 0u && (config->partner_pack_path == 0 || config->expert_slots < 2u) )
	{
		fprintf(stderr, "%s: streamed prefill streams expert blocks, so it needs the partner's pack and at least 2 expert slots\n", p->tag);
		LmPassDestroy(p);
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	if ( layout->rows_block_bytes != 0u )
	{
		p->rows_block_bytes = (layout->rows_block_bytes + (2ull << 20) - 1u) & ~((2ull << 20) - 1u);
		p->rows_region_bytes = 2u * (uint64_t)LM_PASS_RANKS * p->rows_block_bytes;
		p->rows_region = (uint8_t *)aligned_alloc(2ull << 20, (size_t)p->rows_region_bytes);
		if ( p->rows_region == 0 || cudaHostRegister(p->rows_region, (size_t)p->rows_region_bytes, cudaHostRegisterDefault) != cudaSuccess )
		{
			fprintf(stderr, "%s: the %llu-byte streamed row exchange region could not be allocated and registered\n", p->tag, (unsigned long long)p->rows_region_bytes);
			if ( p->rows_region != 0 )
			{
				free(p->rows_region);
				p->rows_region = 0;
			}
			LmPassDestroy(p);
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		}
	}
	p->region_bytes = SparkPackWindowRegionBytes(LM_PASS_RANKS, p->config.dense_buffers, dense_capacity, p->streams_experts != 0u ? p->config.expert_slots : 0u, expert_capacity);
	p->region_bytes = (p->region_bytes + (2ull << 20) - 1u) & ~((2ull << 20) - 1u);
	p->region = (uint8_t *)aligned_alloc(2ull << 20, (size_t)p->region_bytes);
	if ( p->region == 0 || cudaHostRegister(p->region, (size_t)p->region_bytes, cudaHostRegisterDefault) != cudaSuccess )
	{
		fprintf(stderr, "%s: the %llu-byte streamed weight window could not be allocated and registered\n", p->tag, (unsigned long long)p->region_bytes);
		if ( p->region != 0 )
		{
			free(p->region);
			p->region = 0;
		}
		LmPassDestroy(p);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	p->window.config.dense_capacity = dense_capacity;
	p->window.config.expert_capacity = expert_capacity;
	p->partner_fd = config->partner_pack_path != 0 ? open(config->partner_pack_path, O_RDONLY | O_CLOEXEC) : -1;
	p->own_fd = layout->own_pack_path != 0 ? open(layout->own_pack_path, O_RDONLY | O_CLOEXEC) : -1;
	p->partner_direct_fd = config->partner_pack_path != 0 ? open(config->partner_pack_path, O_RDONLY | O_CLOEXEC | O_DIRECT) : -1;
	p->own_direct_fd = layout->own_pack_path != 0 ? open(layout->own_pack_path, O_RDONLY | O_CLOEXEC | O_DIRECT) : -1;
	p->bounce = p->own_direct_fd >= 0 && (p->partner_direct_fd >= 0 || p->streams_experts == 0u) ?
		(uint8_t *)aligned_alloc(LM_PASS_DIRECT_ALIGN, (size_t)(LM_PASS_DIRECT_CHUNK + LM_PASS_DIRECT_ALIGN)) : 0;
	status = (p->partner_fd >= 0 || p->streams_experts == 0u) && p->own_fd >= 0 ? LmPassBuffers(p) : SPARK_STATUS_IO_ERROR;
	if ( status == SPARK_STATUS_OK && (cudaStreamCreateWithFlags(&p->gather_stream, cudaStreamNonBlocking) != cudaSuccess ||
		cudaStreamCreateWithFlags(&p->copy_stream, cudaStreamNonBlocking) != cudaSuccess) )
		status = SPARK_STATUS_IO_ERROR;
	for (index = 0u; status == SPARK_STATUS_OK && index < p->config.dense_buffers; index++)
		if ( cudaEventCreateWithFlags(&p->dense_done[index], cudaEventDisableTiming) != cudaSuccess )
			status = SPARK_STATUS_IO_ERROR;
	for (index = 0u; status == SPARK_STATUS_OK && index < p->config.expert_slots; index++)
		if ( cudaEventCreateWithFlags(&p->slot_done[index], cudaEventDisableTiming) != cudaSuccess )
			status = SPARK_STATUS_IO_ERROR;
	for (index = 0u; status == SPARK_STATUS_OK && index < p->layers * 3u; index++)
		if ( cudaEventCreate(&p->marks[index / 3u][index % 3u]) != cudaSuccess )
			status = SPARK_STATUS_IO_ERROR;
	for (index = 0u; status == SPARK_STATUS_OK && index < p->layers * 2u; index++)
		if ( cudaEventCreate(&p->own_marks[index / 2u][index % 2u]) != cudaSuccess )
			status = SPARK_STATUS_IO_ERROR;
	if ( status == SPARK_STATUS_OK )
	{
		if ( pthread_mutex_init(&p->lock, 0) != 0 )
			status = SPARK_STATUS_INTERNAL_ERROR;
		else if ( pthread_cond_init(&p->changed, 0) != 0 )
		{
			(void)pthread_mutex_destroy(&p->lock);
			status = SPARK_STATUS_INTERNAL_ERROR;
		}
		else
			p->synchronization = 1u;
	}
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr, "%s: streamed prefill resources failed status=%d (own pack %s, partner pack %s)\n", p->tag, (int)status,
			layout->own_pack_path != 0 ? layout->own_pack_path : "(none)", config->partner_pack_path != 0 ? config->partner_pack_path : "(none)");
		LmPassDestroy(p);
		return status;
	}
	return SPARK_STATUS_OK;
}

static SparkStatus LmPassOpen(LmPass *p, const SparkStageRunnerServices *services)
{
	SparkWeightStreamConfig stream_config;
	SparkPackWindowConfig window_config;
	const void *pool = 0;
	uint32_t resident = 0u;
	SparkStatus status;
	p->round.services = services;
	p->round.status = SPARK_STATUS_OK;
	if ( p->opened != 0u )
		return SPARK_STATUS_OK;
	status = p->lazy_pack != 0 && p->lazy_pack->map != 0 ? SparkWeightdMapResident(p->lazy_pack->map, SPARK_WEIGHTD_ATTACH_TIMEOUT_DEFAULT_NS, &resident) : SPARK_STATUS_INVALID_ARGUMENT;
	if ( status == SPARK_STATUS_OK && resident != 0u )
		status = SparkWeightdMapPool(p->lazy_pack->map, &pool);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr, "%s: streamed prefill cannot ask weightd whether the own shard is resident (status=%d)\n", p->tag, (int)status);
		return status;
	}
	p->pool = resident != 0u ? (const uint8_t *)pool : 0;
	fprintf(stderr, "%s: streamed prefill reads the own shard's experts from %s\n", p->tag, p->pool != 0 ? "the resident pool" : "the own pack on local NVMe");
	memset(&stream_config, 0, sizeof(stream_config));
	stream_config.rank = p->rank;
	stream_config.ranks = LM_PASS_RANKS;
	stream_config.chunk_bytes = LM_PASS_STREAM_CHUNK;
	stream_config.depth = LM_PASS_STREAM_DEPTH;
	stream_config.fabric_gid_index = p->config.gid_index;
	stream_config.pair_gid_index = p->config.pair_gid_index;
	stream_config.traffic_class = p->config.traffic_class;
	stream_config.fabric_device = p->config.fabric_device;
	stream_config.pair_device = p->config.pair_device;
	stream_config.region = p->region;
	stream_config.region_bytes = p->region_bytes;
	stream_config.gather = LmPassGather;
	stream_config.gather_context = p;
	status = SparkWeightStreamOpen(&stream_config, &p->stream);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr, "%s: the streamed weight stream did not open status=%d (fabric %s, pair %s)\n", p->tag, (int)status, p->config.fabric_device, p->config.pair_device);
		p->stream = 0;
		return status;
	}
	memset(&window_config, 0, sizeof(window_config));
	window_config.stream = p->stream;
	window_config.region = p->region;
	window_config.region_bytes = p->region_bytes;
	window_config.rank = p->rank;
	window_config.ranks = LM_PASS_RANKS;
	window_config.dense_buffers = p->config.dense_buffers;
	window_config.expert_slots = p->streams_experts != 0u ? p->config.expert_slots : 0u;
	window_config.dense_capacity = p->window.config.dense_capacity;
	window_config.expert_capacity = p->window.config.expert_capacity;
	window_config.read_own = LmPassReadOwn;
	window_config.own_context = p;
	window_config.read_partner = p->streams_experts != 0u ? LmPassReadPartner : 0;
	window_config.partner_context = p;
	status = SparkPackWindowInitialize(&p->window, &window_config);
	if ( status != SPARK_STATUS_OK )
		return status;
	if ( p->rows_region != 0 )
	{
		stream_config.region = p->rows_region;
		stream_config.region_bytes = p->rows_region_bytes;
		status = SparkWeightStreamOpen(&stream_config, &p->rows_stream);
		if ( status != SPARK_STATUS_OK )
		{
			fprintf(stderr, "%s: the streamed row exchange did not open status=%d (fabric %s, pair %s)\n", p->tag, (int)status, p->config.fabric_device, p->config.pair_device);
			p->rows_stream = 0;
			return status;
		}
	}
	p->opened = 1u;
	return SPARK_STATUS_OK;
}

static uint8_t *LmPassRowsSend(const LmPass *p, uint32_t destination)
{
	return p->rows_region != 0 ? p->rows_region + (uint64_t)destination * p->rows_block_bytes : 0;
}

static uint8_t *LmPassRowsLanding(const LmPass *p, uint32_t source)
{
	return p->rows_region != 0 ? p->rows_region + ((uint64_t)LM_PASS_RANKS + source) * p->rows_block_bytes : 0;
}

static SparkStatus LmPassRowsAllGather(LmPass *p, uint64_t bytes, cudaStream_t stream)
{
	const uint64_t landing = (uint64_t)LM_PASS_RANKS * p->rows_block_bytes;
	const uint64_t mark = LmPassNowNs();
	SparkStatus status;
	if ( p->rows_stream == 0 || bytes == 0u || bytes > p->rows_block_bytes )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( cudaStreamSynchronize(stream) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	status = SparkWeightStreamBroadcast(p->rows_stream, landing + (uint64_t)p->rank * bytes, bytes, landing, p->rows_tag++);
	p->spent[2] += LmPassNowNs() - mark;
	if ( status != SPARK_STATUS_OK )
		fprintf(stderr, "%s: streamed row all-gather of %llu bytes per rank failed status=%d\n", p->tag, (unsigned long long)bytes, (int)status);
	return status;
}

static SparkStatus LmPassRowsExchange(LmPass *p, uint64_t bytes, cudaStream_t stream)
{
	const uint64_t mark = LmPassNowNs();
	SparkStatus status;
	if ( p->rows_stream == 0 || bytes == 0u || bytes > p->rows_block_bytes )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( cudaStreamSynchronize(stream) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	status = SparkWeightStreamExchange(p->rows_stream, 0u, p->rows_block_bytes, bytes, (uint64_t)LM_PASS_RANKS * p->rows_block_bytes, p->rows_tag++);
	p->spent[2] += LmPassNowNs() - mark;
	if ( status != SPARK_STATUS_OK )
		fprintf(stderr, "%s: streamed row exchange of %llu bytes per rank failed status=%d\n", p->tag, (unsigned long long)bytes, (int)status);
	return status;
}

__global__ static void LmPassSumBlocksKernel(const uint16_t *blocks, uint64_t block_elements, uint64_t elements, uint16_t *out_bf16, float *out_f32, uint32_t accumulate)
{
	for (uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < elements; index += (uint64_t)gridDim.x * blockDim.x)
	{
		float sum = out_f32 != 0 && accumulate != 0u ? out_f32[index] : 0.0f;
		for (uint32_t block = 0u; block < LM_PASS_RANKS; block++)
			sum += LmBf16ToFloat(blocks[(uint64_t)block * block_elements + index]);
		if ( out_f32 != 0 )
			out_f32[index] = sum;
		if ( out_bf16 != 0 )
			out_bf16[index] = LmFloatToBf16(sum);
	}
}

static int32_t LmPassSumBlocks(const LmPass *p, uint64_t elements, uint16_t *out_bf16, float *out_f32, uint32_t accumulate, cudaStream_t stream)
{
	const uint64_t block_elements = p->rows_block_bytes / sizeof(uint16_t);
	if ( p->rows_region == 0 || elements == 0u || elements > block_elements || (out_bf16 == 0 && out_f32 == 0) )
		return(LM_LAUNCH_ERR_SHAPE);
	LmPassSumBlocksKernel<<<512u, LM_PASS_THREADS, 0, stream>>>((const uint16_t *)LmPassRowsLanding(p, 0u), block_elements, elements, out_bf16, out_f32, accumulate);
	return(cudaPeekAtLastError() == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}

static SparkStatus LmPassCheck(LmPass *p, int32_t launched, const char *what, uint32_t layer)
{
	if ( launched == LM_LAUNCH_OK && p->round.status == SPARK_STATUS_OK )
		return SPARK_STATUS_OK;
	fprintf(stderr, "%s: streamed prefill %s failed at layer %u launch=%d collective=%d\n", p->tag, what, layer, (int)launched, (int)p->round.status);
	return p->round.status != SPARK_STATUS_OK ? p->round.status : SPARK_STATUS_INTERNAL_ERROR;
}

static void LmPassStreamFail(LmPass *p, SparkStatus status)
{
	pthread_mutex_lock(&p->lock);
	if ( p->stream_status == SPARK_STATUS_OK )
		p->stream_status = status;
	pthread_cond_broadcast(&p->changed);
	pthread_mutex_unlock(&p->lock);
}

static SparkStatus LmPassAwaitRelease(LmPass *p, const uint64_t *released, uint64_t item, uint32_t depth, cudaEvent_t event)
{
	SparkStatus status;
	const uint64_t mark = LmPassNowNs();
	if ( item < depth )
		return SPARK_STATUS_OK;
	pthread_mutex_lock(&p->lock);
	while ( *released < item - depth + 1u && p->stream_status == SPARK_STATUS_OK )
		pthread_cond_wait(&p->changed, &p->lock);
	status = p->stream_status;
	pthread_mutex_unlock(&p->lock);
	if ( status == SPARK_STATUS_OK && cudaEventSynchronize(event) != cudaSuccess )
		status = SPARK_STATUS_IO_ERROR;
	p->spent[0] += LmPassNowNs() - mark;
	return status;
}

static SparkStatus LmPassAwaitPublished(LmPass *p, const uint64_t *published, uint64_t item, uint32_t which)
{
	SparkStatus status;
	const uint64_t mark = LmPassNowNs();
	pthread_mutex_lock(&p->lock);
	while ( *published <= item && p->stream_status == SPARK_STATUS_OK )
		pthread_cond_wait(&p->changed, &p->lock);
	status = *published > item ? SPARK_STATUS_OK : p->stream_status;
	pthread_mutex_unlock(&p->lock);
	p->spent[which] += LmPassNowNs() - mark;
	return status;
}

static SparkStatus LmPassRelease(LmPass *p, uint64_t *released, cudaEvent_t event, cudaStream_t stream)
{
	if ( cudaEventRecord(event, stream) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	pthread_mutex_lock(&p->lock);
	(*released)++;
	pthread_cond_broadcast(&p->changed);
	pthread_mutex_unlock(&p->lock);
	return SPARK_STATUS_OK;
}

static SparkStatus LmPassStreamDense(LmPass *p, const SparkPackWindowLayer *range, uint64_t item)
{
	const uint32_t buffer = (uint32_t)(item % p->config.dense_buffers);
	uint8_t *base[LM_PASS_RANKS];
	SparkStatus status = LmPassAwaitRelease(p, &p->dense_released, item, p->config.dense_buffers, p->dense_done[buffer]);
	const uint64_t mark = LmPassNowNs();
	if ( status == SPARK_STATUS_OK )
		status = SparkPackWindowDense(&p->window, range, buffer, p->tag_counter++, base);
	p->spent[4] += LmPassNowNs() - mark;
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr, "%s: streamed prefill dense broadcast %llu failed status=%d\n", p->tag, (unsigned long long)item, (int)status);
		return status;
	}
	pthread_mutex_lock(&p->lock);
	memcpy(p->dense_base[buffer], base, sizeof(base));
	p->dense_published = item + 1u;
	pthread_cond_broadcast(&p->changed);
	pthread_mutex_unlock(&p->lock);
	return SPARK_STATUS_OK;
}

static SparkStatus LmPassStreamStep(LmPass *p, uint32_t layer, uint32_t step, uint64_t item)
{
	const uint32_t slot = (uint32_t)(item % p->config.expert_slots);
	const uint64_t half = p->ranges[layer].experts.bytes / 2u;
	const uint8_t *block = 0;
	uint32_t source = UINT32_MAX;
	SparkStatus status = LmPassAwaitRelease(p, &p->expert_released, item, p->config.expert_slots, p->slot_done[slot]);
	const uint64_t mark = LmPassNowNs();
	if ( status == SPARK_STATUS_OK )
		status = SparkPackWindowExpertStep(&p->window, &p->ranges[layer], step, slot, p->tag_counter++, &source, &block);
	p->spent[5] += LmPassNowNs() - mark;
	if ( status == SPARK_STATUS_OK && (source >= LM_PASS_RANKS || (source != p->rank && block == 0)) )
		status = SPARK_STATUS_INTERNAL_ERROR;
	if ( status == SPARK_STATUS_OK && source == p->rank && p->pool == 0 )
	{
		uint8_t *landing = p->region + p->window.expert_slot[slot];
		const uint64_t own = LmPassNowNs();
		memcpy(landing, p->region + p->window.own_staging, (size_t)half);
		status = LmPassPread(p->own_fd, p->own_direct_fd, p->bounce, p->ranges[layer].experts.offset + half, half, landing + half);
		block = landing;
		p->spent[3] += LmPassNowNs() - own;
	}
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr, "%s: streamed prefill expert step %u failed at layer %u status=%d\n", p->tag, step, layer, (int)status);
		return status;
	}
	pthread_mutex_lock(&p->lock);
	p->slot_source[slot] = source;
	p->slot_block[slot] = block;
	p->expert_published = item + 1u;
	pthread_cond_broadcast(&p->changed);
	pthread_mutex_unlock(&p->lock);
	return SPARK_STATUS_OK;
}

static void *LmPassStreamer(void *argument)
{
	LmPass *p = (LmPass *)argument;
	uint64_t dense = 0u, experts = 0u;
	uint32_t pass, layer, step;
	SparkStatus status = cudaSetDevice(p->device) == cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR;
	for (pass = 0u; pass < p->stream_passes && status == SPARK_STATUS_OK; pass++)
	{
		status = LmPassStreamDense(p, &p->vocab, dense++);
		for (layer = 0u; layer < p->layers && status == SPARK_STATUS_OK; layer++)
		{
			status = LmPassStreamDense(p, &p->ranges[layer], dense++);
			for (step = 0u; step < p->experts_per_layer[layer] && status == SPARK_STATUS_OK; step++)
				status = LmPassStreamStep(p, layer, step, experts++);
		}
	}
	if ( status != SPARK_STATUS_OK )
		LmPassStreamFail(p, status);
	return 0;
}

static SparkStatus LmPassDense(LmPass *p, uint64_t dense_item, uint8_t **rank_base)
{
	const SparkStatus status = LmPassAwaitPublished(p, &p->dense_published, dense_item, 1u);
	if ( status == SPARK_STATUS_OK )
		memcpy(rank_base, p->dense_base[dense_item % p->config.dense_buffers], sizeof(uint8_t *) * LM_PASS_RANKS);
	return status;
}

static SparkStatus LmPassReleaseDense(LmPass *p, uint64_t dense_item, cudaStream_t stream)
{
	return LmPassRelease(p, &p->dense_released, p->dense_done[dense_item % p->config.dense_buffers], stream);
}

static SparkStatus LmPassExpert(LmPass *p, uint64_t item, uint32_t *source, const uint8_t **block)
{
	const uint32_t slot = (uint32_t)(item % p->config.expert_slots);
	const SparkStatus status = LmPassAwaitPublished(p, &p->expert_published, item, 2u);
	if ( status != SPARK_STATUS_OK )
		return status;
	*source = p->slot_source[slot];
	*block = p->slot_block[slot];
	return SPARK_STATUS_OK;
}

static SparkStatus LmPassReleaseExpert(LmPass *p, uint64_t item, cudaStream_t stream)
{
	return LmPassRelease(p, &p->expert_released, p->slot_done[item % p->config.expert_slots], stream);
}

static float LmPassBf16(uint32_t bits)
{
	const uint32_t word = bits << 16u;
	float value;
	memcpy(&value, &word, sizeof(value));
	return value;
}

static SparkStatus LmPassHealth(LmPass *p, uint32_t pass)
{
	char line[4096];
	uint32_t layer, length = 0u, first_bad = UINT32_MAX;
	for (layer = 0u; layer <= p->layers; layer++)
	{
		if ( p->health[2u * layer] != 0u && first_bad == UINT32_MAX )
			first_bad = layer;
		if ( length + 16u < sizeof(line) && (layer < 8u || layer % 8u == 0u || layer == p->layers || layer == first_bad) )
			length += (uint32_t)snprintf(line + length, sizeof(line) - length, " %d:%.3g", (int)layer - 1, LmPassBf16(p->health[2u * layer + 1u]));
	}
	fprintf(stderr, "%s: streamed prefill pass %u rank %u hidden absmax by layer (-1 = embedding):%s\n", p->tag, pass, p->rank, line);
	if ( first_bad == UINT32_MAX )
		return SPARK_STATUS_OK;
	fprintf(stderr, "%s: streamed prefill pass %u rank %u: %u non-finite hidden values after layer %d; the prefill fails before its KV reaches decode\n",
		p->tag, pass, p->rank, p->health[2u * first_bad], (int)first_bad - 1);
	SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
}

static SparkStatus LmPassUpload(void *device, const void *host, uint64_t bytes, cudaStream_t stream)
{
	return cudaMemcpyAsync(device, host, (size_t)bytes, cudaMemcpyHostToDevice, stream) == cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR;
}

static SparkStatus LmPassMeasureHelp(LmPass *p, uint32_t first_layer, uint32_t end_layer, uint64_t first_key, uint32_t rows, uint64_t pass_tokens, cudaStream_t stream)
{
	float sent[4] = {0.0f, 0.0f, 0.0f, 0.0f}, received[4u * LM_PASS_RANKS], part;
	uint32_t layer, node, measured = 0u;
	for (layer = first_layer; layer < end_layer; layer++)
		if ( p->balances[layer] != 0u )
		{
			sent[0] = cudaEventElapsedTime(&part, p->own_marks[layer][0], p->own_marks[layer][1]) == cudaSuccess && sent[0] >= 0.0f ? sent[0] + part : -1.0f;
			measured++;
		}
	if ( LmPassUpload(p->balance_device, sent, sizeof(sent), stream) != SPARK_STATUS_OK )
		return SPARK_STATUS_IO_ERROR;
	LmPassRound(&p->round, (void *)stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, 1u, (uint32_t)(sizeof(sent) / 2u), p->balance_device, p->balance_device + 4u);
	if ( cudaMemcpyAsync(received, p->balance_device + 4u, sizeof(received), cudaMemcpyDeviceToHost, stream) != cudaSuccess || cudaStreamSynchronize(stream) != cudaSuccess )
		return SPARK_STATUS_IO_ERROR;
	if ( p->round.status != SPARK_STATUS_OK )
		return p->round.status;
	for (node = 0u; node < LM_PASS_RANKS; node++)
	{
		const double done = (double)measured * SparkPassOwnAttention(&p->balance, node, first_key, rows, pass_tokens);
		p->balance.node_speed[node] = received[4u * node] > 0.0f && measured != 0u ? done / (double)received[4u * node] : 0.0;
		p->pass_attention[node] += received[4u * node] > 0.0f ? (double)received[4u * node] : 0.0;
	}
	for (node = 0u; node < LM_PASS_RANKS * LM_PASS_RANKS; node++)
		p->pass_helped += p->balance.help_active != 0u ? (uint64_t)p->balance.help[node] * measured : 0u;
	return SPARK_STATUS_OK;
}

static void LmPassReportHelp(LmPass *p, uint32_t pass)
{
	uint32_t node, slowest = 0u, fastest = 0u;
	for (node = 1u; node < LM_PASS_RANKS; node++)
	{
		slowest = p->pass_attention[node] > p->pass_attention[slowest] ? node : slowest;
		fastest = p->pass_attention[node] < p->pass_attention[fastest] ? node : fastest;
	}
	if ( p->rank == 0u )
		fprintf(stderr, "%s: streamed prefill pass %u attention per node: slowest rank %u %.1f s, fastest rank %u %.1f s; helpers attended %llu head-group rows over the pass's layers\n",
			p->tag, pass + 1u, slowest, p->pass_attention[slowest] / 1e3, fastest, p->pass_attention[fastest] / 1e3, (unsigned long long)p->pass_helped);
	memset(p->pass_attention, 0, sizeof(p->pass_attention));
	p->pass_helped = 0u;
}

template<class Model>
static SparkStatus LmPassRun(LmPass *p, Model *model, const SparkStageRunnerStreamedPrefill *request, const SparkStageRunnerServices *services, cudaStream_t stream)
{
	const uint64_t keys = request != 0 ? (uint64_t)request->first_position + request->token_count : 0u;
	const uint64_t started = LmPassNowNs();
	uint32_t pass, layer, row, rows, balanced, measured_from = 0u;
	uint64_t dense_item = 0u, expert_item = 0u;
	SparkPassPlan plan;
	pthread_t streamer;
	LmPassVocab vocab;
	SparkStatus status;
	if ( request == 0 || request->token_ids == 0 || request->token_count == 0u || keys > UINT32_MAX ||
		SparkPassPlanTokens(request->first_position, request->token_count, LM_PASS_RANKS, p->node_rows, p->row_align, &plan) != 0 )
	{
		fprintf(stderr, "%s: streamed prefill takes at least one token for a known slot (got %u tokens, slot %u, state %u)\n", p->tag,
			request != 0 ? request->token_count : 0u, request != 0 ? request->sequence_slot : 0u, request != 0 ? request->state_target : 0u);
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	status = LmPassOpen(p, services);
	if ( status == SPARK_STATUS_OK )
		status = model->Begin(p, request, plan.padded_keys, stream);
	if ( status == SPARK_STATUS_OK && cudaStreamSynchronize(stream) != cudaSuccess )
		status = SPARK_STATUS_IO_ERROR;
	if ( status != SPARK_STATUS_OK )
		return status;
	p->stream_passes = plan.passes;
	p->stream_status = SPARK_STATUS_OK;
	p->dense_published = p->dense_released = p->expert_published = p->expert_released = 0u;
	memset(p->spent, 0, sizeof(p->spent));
	if ( pthread_create(&streamer, 0, LmPassStreamer, p) != 0 )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	for (pass = 0u; pass < plan.passes && status == SPARK_STATUS_OK; pass++)
	{
		LmPassStep step;
		uint32_t *host = p->host_staging;
		memset(&step, 0, sizeof(step));
		step.pass = pass;
		step.passes = plan.passes;
		step.base = (uint64_t)request->first_position + (uint64_t)pass * plan.pass_share;
		step.pass_tokens = pass + 1u < plan.passes ? plan.pass_share : plan.last_tokens;
		step.rows = rows = SparkPassNodeRows(step.pass_tokens, LM_PASS_RANKS, p->row_align);
		step.pass_rows = LM_PASS_RANKS * rows;
		step.node_first = step.base + (uint64_t)p->rank * rows;
		step.node_tokens = step.pass_tokens > (uint64_t)p->rank * rows ? (step.pass_tokens - (uint64_t)p->rank * rows < rows ? step.pass_tokens - (uint64_t)p->rank * rows : rows) : 0u;
		step.sequence_slot = request->sequence_slot;
		if ( cudaStreamSynchronize(stream) != cudaSuccess )
		{
			status = SPARK_STATUS_IO_ERROR;
			break;
		}
		for (row = 0u; row < step.pass_rows; row++)
			host[row] = (uint32_t)(step.base + row);
		status = LmPassUpload(p->pass_positions, host, (uint64_t)step.pass_rows * sizeof(uint32_t), stream);
		if ( status == SPARK_STATUS_OK )
			status = LmPassUpload(p->positions, host + (uint64_t)p->rank * rows, (uint64_t)rows * sizeof(uint32_t), stream);
		if ( status == SPARK_STATUS_OK && step.node_tokens != 0u )
			status = LmPassUpload(p->tokens, request->token_ids + (step.node_first - request->first_position), step.node_tokens * sizeof(uint32_t), stream);
		if ( status == SPARK_STATUS_OK && step.node_tokens < rows &&
			cudaMemsetAsync(p->tokens + step.node_tokens, 0, (size_t)(rows - step.node_tokens) * sizeof(uint32_t), stream) != cudaSuccess )
			status = SPARK_STATUS_IO_ERROR;
		if ( status == SPARK_STATUS_OK && cudaStreamSynchronize(stream) != cudaSuccess )
			status = SPARK_STATUS_IO_ERROR;
		if ( status == SPARK_STATUS_OK )
			status = model->PassBegin(p, &step, stream);
		if ( status != SPARK_STATUS_OK )
			break;
		p->balance.help_capacity = model->HelpCapacity();
		SparkPassPlanHelp(&p->balance, step.base, rows, step.pass_tokens, LM_PASS_HELP_MIN_SHARE);
		model->HelpChanged(p);
		status = LmPassAwaitPublished(p, &p->dense_published, dense_item, 1u);
		if ( status != SPARK_STATUS_OK )
			break;
		memset(&vocab, 0, sizeof(vocab));
		for (row = 0u; row < LM_PASS_RANKS; row++)
			vocab.slice[row] = (const uint16_t *)p->dense_base[dense_item % p->config.dense_buffers][row];
		vocab.slice_rows = p->vocab_rows;
		vocab.slices = LM_PASS_RANKS;
		vocab.width = p->hidden;
		memset(p->health, 0, (size_t)(p->layers + 1u) * 2u * sizeof(uint32_t));
		status = LmPassCheck(p, LmPassEmbed(&vocab, p->tokens, model->Hidden(), rows, stream), "embedding", 0u);
		if ( status == SPARK_STATUS_OK )
			LmPassHealthKernel<<<256u, 256u, 0, stream>>>(model->Hidden(), (uint64_t)rows * p->hidden, p->health);
		if ( status == SPARK_STATUS_OK )
			status = LmPassReleaseDense(p, dense_item, stream);
		dense_item++;
		for (layer = 0u, balanced = 0u, measured_from = 0u; layer < p->layers && status == SPARK_STATUS_OK; layer++)
		{
			if ( p->balances[layer] != 0u && balanced != 0u && balanced % LM_PASS_HELP_LAYERS == 0u )
			{
				status = cudaStreamSynchronize(stream) == cudaSuccess ? LmPassMeasureHelp(p, measured_from, layer, step.base, rows, step.pass_tokens, stream) : SPARK_STATUS_IO_ERROR;
				SparkPassPlanHelp(&p->balance, step.base, rows, step.pass_tokens, LM_PASS_HELP_MIN_SHARE);
				model->HelpChanged(p);
				measured_from = layer;
				if ( status != SPARK_STATUS_OK )
					break;
			}
			status = model->Layer(p, layer, dense_item++, &expert_item, rows, request->multiprocessors, stream);
			if ( status == SPARK_STATUS_OK )
				LmPassHealthKernel<<<256u, 256u, 0, stream>>>(model->Hidden(), (uint64_t)rows * p->hidden, p->health + 2u * (layer + 1u));
			if ( status == SPARK_STATUS_OK )
				status = LmPassReleaseDense(p, dense_item - 1u, stream);
			balanced += p->balances[layer] != 0u ? 1u : 0u;
		}
		if ( status == SPARK_STATUS_OK )
			status = cudaStreamSynchronize(stream) == cudaSuccess ? LmPassHealth(p, pass) : SPARK_STATUS_IO_ERROR;
		if ( status == SPARK_STATUS_OK )
		{
			model->PassReport(p, pass);
			status = LmPassMeasureHelp(p, measured_from, p->layers, step.base, rows, step.pass_tokens, stream);
			LmPassReportHelp(p, pass);
		}
		fprintf(stderr, "%s: streamed prefill pass %u/%u rows %u x %u keys to %llu status=%d at %.1f s; streamer waited %.1f s for the GPU to release buffers, "
			"compute waited %.1f s for dense and %.1f s for expert blocks, own blocks took %.1f s; streamer spent %.1f s in dense broadcasts and %.1f s in expert steps, "
			"%.1f s of them reading the packs\n", p->tag, pass + 1u, plan.passes, rows, LM_PASS_RANKS,
			(unsigned long long)(step.base + step.pass_tokens), (int)status, (double)(LmPassNowNs() - started) / 1e9, (double)p->spent[0] / 1e9, (double)p->spent[1] / 1e9,
			(double)p->spent[2] / 1e9, (double)p->spent[3] / 1e9, (double)p->spent[4] / 1e9, (double)p->spent[5] / 1e9, (double)p->spent[6] / 1e9);
		memset(p->spent, 0, sizeof(p->spent));
	}
	if ( status != SPARK_STATUS_OK )
		LmPassStreamFail(p, status);
	(void)pthread_join(streamer, 0);
	if ( status == SPARK_STATUS_OK && p->stream_status != SPARK_STATUS_OK )
		status = p->stream_status;
	if ( status == SPARK_STATUS_OK )
		status = cudaStreamSynchronize(stream) == cudaSuccess ? p->round.status : SPARK_STATUS_IO_ERROR;
	if ( status == SPARK_STATUS_OK )
		status = model->End(p, request, stream);
	if ( status == SPARK_STATUS_OK && cudaStreamSynchronize(stream) != cudaSuccess )
		status = SPARK_STATUS_IO_ERROR;
	fprintf(stderr, "%s: streamed prefill sequence %llu positions %u..%llu in %u passes status=%d %.1f s; exchanges with a block late or wrong at first check %llu, settled after waiting %llu\n",
		p->tag, (unsigned long long)request->sequence_id, request->first_position, (unsigned long long)keys, plan.passes, (int)status, (double)(LmPassNowNs() - started) / 1e9,
		(unsigned long long)p->round.mismatches, (unsigned long long)p->round.late);
	return status;
}
