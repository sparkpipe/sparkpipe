#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "runtime/launch.h"
#include "inference/kernels/index_score.cuh"

#define PROBE_HEADS 32u
#define PROBE_DIM 128u

using ProbeIndexKv = LmKvLatent<16u, PROBE_DIM, 0u, 64u>;

static int probe_failures;

#define PROBE_CUDA(call) do { cudaError_t probe_error = (call); if ( probe_error != cudaSuccess ) { printf("FAIL cuda %s: %s\n",#call,cudaGetErrorString(probe_error)); exit(1); } } while (0)

template<class Geometry, class Pages, uint32_t INDEX_HEADS, uint32_t INDEX_DIM, uint32_t ROWS_PER_BLOCK>
__global__ __launch_bounds__(LM_INDEX_SCORE_THREADS)
void ProbeOneTileScoreKernel(const uint16_t *__restrict__ index_query_bf16, const uint16_t *__restrict__ head_weight_bf16, Pages index_view, const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ context_length, const uint32_t *__restrict__ row_position, uint32_t rows, uint32_t score_stride, float qk_scale, float *__restrict__ scores)
{
	const LmKvView &index_cache = LmIndexScorePages(index_view);
	constexpr uint32_t HEAD_TILES = INDEX_HEADS / 16u;
	const uint32_t steps = INDEX_DIM / 16u;
	const uint32_t lane = threadIdx.x % LM_WARP_LANES,warp = threadIdx.x / LM_WARP_LANES;
	const uint32_t group = lane / 4u,pair = lane % 4u;
	const uint32_t row = blockIdx.y * ROWS_PER_BLOCK + warp % ROWS_PER_BLOCK;
	const uint32_t tile_stride = LM_INDEX_SCORE_WARPS / ROWS_PER_BLOCK;
	uint32_t a[HEAD_TILES][INDEX_DIM / 16u][4];
	float weight[2u * HEAD_TILES];
	uint32_t sequence,bound,tile,step,head_tile,entry,valid;
	if ( scores == 0 || row >= rows )
		return;
	valid = sequence_of_row != 0 && context_length != 0 && index_query_bf16 != 0 && head_weight_bf16 != 0 && LmKvViewIsConfigured(index_cache) ? 1u : 0u;
	sequence = valid != 0u ? sequence_of_row[row] : 0xffffffffu;
	bound = 0u;
	if ( valid == 0u || sequence >= index_cache.sequence_count )
		valid = 0u;
	else
	{
		const uint16_t *query = index_query_bf16 + (uint64_t)row * INDEX_HEADS * INDEX_DIM;
		bound = context_length[sequence];
		if ( row_position != 0 && row_position[row] + 1u < bound )
			bound = row_position[row] + 1u;
		bound = LmIndexScoreKeys(index_view, bound);
		for (head_tile = 0u; head_tile < HEAD_TILES; head_tile++)
			for (step = 0u; step < steps; step++)
			{
				const uint16_t *top = query + (uint64_t)(head_tile * 16u + group) * INDEX_DIM + step * 16u + pair * 2u;
				const uint16_t *bottom = top + 8u * INDEX_DIM;
				a[head_tile][step][0] = *(const uint32_t *)top;
				a[head_tile][step][1] = *(const uint32_t *)bottom;
				a[head_tile][step][2] = *(const uint32_t *)(top + 8u);
				a[head_tile][step][3] = *(const uint32_t *)(bottom + 8u);
			}
		for (head_tile = 0u; head_tile < HEAD_TILES; head_tile++)
		{
			weight[2u * head_tile] = LmBf16ToFloat(head_weight_bf16[(uint64_t)row * INDEX_HEADS + head_tile * 16u + group]);
			weight[2u * head_tile + 1u] = LmBf16ToFloat(head_weight_bf16[(uint64_t)row * INDEX_HEADS + head_tile * 16u + group + 8u]);
		}
	}
	for (tile = warp / ROWS_PER_BLOCK; tile < LM_INDEX_SCORE_BLOCK_TILES; tile += tile_stride)
	{
		const uint32_t first = blockIdx.x * LM_INDEX_SCORE_BLOCK_POSITIONS + tile * LM_INDEX_SCORE_TILE_POSITIONS;
		const uint32_t mine = first + group;
		const uint32_t written = first + pair * 2u;
		float accumulator[HEAD_TILES][4] = {};
		float low,high;
		if ( first >= score_stride )
			break;
		if ( valid != 0u && first < bound )
		{
			const uint8_t *slot = mine < bound ? LmIndexScoreSlot<Geometry>(index_view,sequence,mine,row) : 0;
			for (step = 0u; step < steps; step++)
			{
				uint32_t b[2];
				b[0] = slot != 0 ? *(const uint32_t *)(slot + (step * 16u + pair * 2u) * sizeof(uint16_t)) : 0u;
				b[1] = slot != 0 ? *(const uint32_t *)(slot + (step * 16u + pair * 2u + 8u) * sizeof(uint16_t)) : 0u;
				for (head_tile = 0u; head_tile < HEAD_TILES; head_tile++)
					LmMmaBf16(accumulator[head_tile],a[head_tile][step],b);
			}
		}
		for (head_tile = 0u; head_tile < HEAD_TILES; head_tile++)
			for (entry = 0u; entry < 4u; entry++)
				accumulator[head_tile][entry] = fmaxf(accumulator[head_tile][entry], 0.0f);
		low = __fmul_rn(weight[0], accumulator[0][0]);
		high = __fmul_rn(weight[0], accumulator[0][1]);
		low = __fmaf_rn(weight[1], accumulator[0][2], low);
		high = __fmaf_rn(weight[1], accumulator[0][3], high);
		for (head_tile = 1u; head_tile < HEAD_TILES; head_tile++)
		{
			low = __fmaf_rn(weight[2u * head_tile], accumulator[head_tile][0], low);
			high = __fmaf_rn(weight[2u * head_tile], accumulator[head_tile][1], high);
			low = __fmaf_rn(weight[2u * head_tile + 1u], accumulator[head_tile][2], low);
			high = __fmaf_rn(weight[2u * head_tile + 1u], accumulator[head_tile][3], high);
		}
		for (entry = 4u; entry < LM_WARP_LANES; entry <<= 1u)
		{
			low += __shfl_xor_sync(0xffffffffu,low,(int)entry);
			high += __shfl_xor_sync(0xffffffffu,high,(int)entry);
		}
		if ( group == 0u )
		{
			if ( written < score_stride )
				scores[(uint64_t)row * score_stride + written] = valid != 0u && written < bound ? low * qk_scale : -INFINITY;
			if ( written + 1u < score_stride )
				scores[(uint64_t)row * score_stride + written + 1u] = valid != 0u && written + 1u < bound ? high * qk_scale : -INFINITY;
		}
	}
}

template<class Pages>
static void ProbeReferenceLaunch(const uint16_t *query, const uint16_t *heads, Pages view, const uint32_t *sequence, const uint32_t *context, const uint32_t *position,
	uint32_t rows, uint32_t stride, float scale, float *scores)
{
	if ( LmIndexScoreRowsPerBlock(rows) == LM_INDEX_SCORE_SHARED_ROWS )
		ProbeOneTileScoreKernel<ProbeIndexKv,Pages,PROBE_HEADS,PROBE_DIM,LM_INDEX_SCORE_SHARED_ROWS><<<dim3(LmIndexScoreBlocks(stride),(rows + LM_INDEX_SCORE_SHARED_ROWS - 1u) / LM_INDEX_SCORE_SHARED_ROWS),
			LM_INDEX_SCORE_THREADS>>>(query,heads,view,sequence,context,position,rows,stride,scale,scores);
	else
		ProbeOneTileScoreKernel<ProbeIndexKv,Pages,PROBE_HEADS,PROBE_DIM,1u><<<dim3(LmIndexScoreBlocks(stride),rows),LM_INDEX_SCORE_THREADS>>>(
			query,heads,view,sequence,context,position,rows,stride,scale,scores);
	PROBE_CUDA(cudaPeekAtLastError());
}

static uint16_t ProbeBf16(uint32_t *state)
{
	float value;
	uint32_t bits;
	*state = *state * 1664525u + 1013904223u;
	value = ((float)((*state >> 8) & 0xffffu) / 32768.0f - 1.0f) * 0.75f;
	memcpy(&bits,&value,sizeof(bits));
	return (uint16_t)(bits >> 16);
}

static void ProbeFill(std::vector<uint16_t> &values, uint32_t seed)
{
	for (size_t index = 0; index < values.size(); ++index)
		values[index] = ProbeBf16(&seed);
}

struct ProbeShape
{
	const char *name;
	uint32_t rows;
	uint32_t keys;
	uint32_t paged;
};

static void ProbeCase(const ProbeShape &shape)
{
	const uint32_t rows = shape.rows,keys = shape.keys,pages = (keys + ProbeIndexKv::kPageSlots - 1u) / ProbeIndexKv::kPageSlots;
	const uint64_t score_count = (uint64_t)rows * keys;
	std::vector<uint16_t> query((size_t)rows * PROBE_HEADS * PROBE_DIM),heads((size_t)rows * PROBE_HEADS),key_values((size_t)pages * ProbeIndexKv::kPageSlots * PROBE_DIM);
	std::vector<uint32_t> positions(rows),page_table(pages);
	std::vector<float> reference(score_count),candidate(score_count);
	uint32_t index;
	uint16_t *device_query,*device_heads;
	uint8_t *device_keys;
	uint32_t *device_sequence,*device_context,*device_position,*device_page_table;
	float *device_reference,*device_candidate,best_reference = 1e30f,best_candidate = 1e30f,elapsed;
	LmKvAccessError *device_error;
	cudaEvent_t start,stop;
	double flops = 0.0;
	ProbeFill(query,rows * 7919u + keys);
	ProbeFill(heads,rows * 104729u + 17u);
	ProbeFill(key_values,keys * 31u + 5u);
	for (index = 0u; index < rows; ++index)
	{
		positions[index] = index % 5u == 0u ? keys - 1u : (uint32_t)(((uint64_t)index * 2654435761u) % keys);
		if ( index % 11u == 3u )
			positions[index] = index % 300u;
		flops += 2.0 * PROBE_HEADS * PROBE_DIM * (double)(positions[index] + 1u < keys ? positions[index] + 1u : keys);
	}
	for (index = 0u; index < pages; ++index)
		page_table[index] = shape.paged != 0u ? pages - 1u - index : index;
	PROBE_CUDA(cudaMalloc((void **)&device_query,query.size() * sizeof(uint16_t)));
	PROBE_CUDA(cudaMalloc((void **)&device_heads,heads.size() * sizeof(uint16_t)));
	PROBE_CUDA(cudaMalloc((void **)&device_keys,(size_t)pages * ProbeIndexKv::kPageBytes));
	PROBE_CUDA(cudaMalloc((void **)&device_sequence,rows * sizeof(uint32_t)));
	PROBE_CUDA(cudaMalloc((void **)&device_context,sizeof(uint32_t)));
	PROBE_CUDA(cudaMalloc((void **)&device_position,rows * sizeof(uint32_t)));
	PROBE_CUDA(cudaMalloc((void **)&device_page_table,pages * sizeof(uint32_t)));
	PROBE_CUDA(cudaMalloc((void **)&device_reference,score_count * sizeof(float)));
	PROBE_CUDA(cudaMalloc((void **)&device_candidate,score_count * sizeof(float)));
	PROBE_CUDA(cudaMalloc((void **)&device_error,sizeof(LmKvAccessError)));
	PROBE_CUDA(cudaMemcpy(device_query,query.data(),query.size() * sizeof(uint16_t),cudaMemcpyHostToDevice));
	PROBE_CUDA(cudaMemcpy(device_heads,heads.data(),heads.size() * sizeof(uint16_t),cudaMemcpyHostToDevice));
	if ( shape.paged != 0u )
		for (index = 0u; index < pages; ++index)
			PROBE_CUDA(cudaMemcpy(device_keys + (uint64_t)page_table[index] * ProbeIndexKv::kPageBytes,key_values.data() + (size_t)index * ProbeIndexKv::kPageSlots * PROBE_DIM,
				ProbeIndexKv::kPageBytes,cudaMemcpyHostToDevice));
	else
		PROBE_CUDA(cudaMemcpy(device_keys,key_values.data(),key_values.size() * sizeof(uint16_t),cudaMemcpyHostToDevice));
	PROBE_CUDA(cudaMemset(device_sequence,0,rows * sizeof(uint32_t)));
	PROBE_CUDA(cudaMemcpy(device_context,&keys,sizeof(uint32_t),cudaMemcpyHostToDevice));
	PROBE_CUDA(cudaMemcpy(device_position,positions.data(),rows * sizeof(uint32_t),cudaMemcpyHostToDevice));
	PROBE_CUDA(cudaMemcpy(device_page_table,page_table.data(),pages * sizeof(uint32_t),cudaMemcpyHostToDevice));
	PROBE_CUDA(cudaMemset(device_error,0,sizeof(LmKvAccessError)));
	PROBE_CUDA(cudaMemset(device_reference,0x7f,score_count * sizeof(float)));
	PROBE_CUDA(cudaMemset(device_candidate,0x3c,score_count * sizeof(float)));
	PROBE_CUDA(cudaEventCreate(&start));
	PROBE_CUDA(cudaEventCreate(&stop));
	LmKvView pages_view;
	memset(&pages_view,0,sizeof(pages_view));
	pages_view.pool = device_keys;
	pages_view.page_table = device_page_table;
	pages_view.page_table_stride = pages;
	pages_view.sequence_count = 1u;
	pages_view.pool_page_count = pages;
	pages_view.access_error = device_error;
	for (uint32_t pass = 0u; pass < 3u; ++pass)
	{
		if ( shape.paged != 0u )
		{
			LmKvShardView view;
			view.pages = pages_view;
			view.shard.degree = 1u;
			view.shard.rank = 0u;
			view.shard.grain = 1u;
			PROBE_CUDA(cudaEventRecord(start));
			ProbeReferenceLaunch(device_query,device_heads,view,device_sequence,device_context,device_position,rows,keys,0.0156f,device_reference);
			PROBE_CUDA(cudaEventRecord(stop));
			PROBE_CUDA(cudaEventSynchronize(stop));
			PROBE_CUDA(cudaEventElapsedTime(&elapsed,start,stop));
			best_reference = elapsed < best_reference ? elapsed : best_reference;
			PROBE_CUDA(cudaEventRecord(start));
			PROBE_CUDA((LmWeightedSparseScoreLaunch<ProbeIndexKv,PROBE_HEADS,PROBE_DIM,LmKvShardView>(device_query,device_heads,view,device_sequence,device_context,
				device_position,rows,keys,0.0156f,device_candidate,0)));
		}
		else
		{
			LmIndexScorePackedView view;
			view.pages = pages_view;
			view.shard.degree = 1u;
			view.shard.rank = 0u;
			view.shard.grain = 1u;
			view.keys = device_keys;
			PROBE_CUDA(cudaEventRecord(start));
			ProbeReferenceLaunch(device_query,device_heads,view,device_sequence,device_context,device_position,rows,keys,0.0156f,device_reference);
			PROBE_CUDA(cudaEventRecord(stop));
			PROBE_CUDA(cudaEventSynchronize(stop));
			PROBE_CUDA(cudaEventElapsedTime(&elapsed,start,stop));
			best_reference = elapsed < best_reference ? elapsed : best_reference;
			PROBE_CUDA(cudaEventRecord(start));
			PROBE_CUDA((LmWeightedSparseScoreLaunch<ProbeIndexKv,PROBE_HEADS,PROBE_DIM,LmIndexScorePackedView>(device_query,device_heads,view,device_sequence,device_context,
				device_position,rows,keys,0.0156f,device_candidate,0)));
		}
		PROBE_CUDA(cudaEventRecord(stop));
		PROBE_CUDA(cudaEventSynchronize(stop));
		PROBE_CUDA(cudaEventElapsedTime(&elapsed,start,stop));
		best_candidate = elapsed < best_candidate ? elapsed : best_candidate;
	}
	PROBE_CUDA(cudaDeviceSynchronize());
	PROBE_CUDA(cudaMemcpy(reference.data(),device_reference,score_count * sizeof(float),cudaMemcpyDeviceToHost));
	PROBE_CUDA(cudaMemcpy(candidate.data(),device_candidate,score_count * sizeof(float),cudaMemcpyDeviceToHost));
	if ( memcmp(reference.data(),candidate.data(),score_count * sizeof(float)) != 0 )
	{
		uint64_t first = 0u;
		while ( first < score_count && memcmp(&reference[first],&candidate[first],sizeof(float)) == 0 )
			first++;
		printf("FAIL %s rows=%u keys=%u: score %llu (row %llu key %llu) differs: %.9g against %.9g\n",shape.name,rows,keys,(unsigned long long)first,
			(unsigned long long)(first / keys),(unsigned long long)(first % keys),reference[first],candidate[first]);
		probe_failures++;
	}
	else
		printf("PASS %s rows=%u keys=%u: identical bits; one-tile form %.3f ms (%.1f TFLOPS), current kernel %.3f ms (%.1f TFLOPS), %.2fx\n",shape.name,rows,keys,
			best_reference,flops / best_reference / 1e9,best_candidate,flops / best_candidate / 1e9,best_reference / best_candidate);
	PROBE_CUDA(cudaEventDestroy(start));
	PROBE_CUDA(cudaEventDestroy(stop));
	PROBE_CUDA(cudaFree(device_query));
	PROBE_CUDA(cudaFree(device_heads));
	PROBE_CUDA(cudaFree(device_keys));
	PROBE_CUDA(cudaFree(device_sequence));
	PROBE_CUDA(cudaFree(device_context));
	PROBE_CUDA(cudaFree(device_position));
	PROBE_CUDA(cudaFree(device_page_table));
	PROBE_CUDA(cudaFree(device_reference));
	PROBE_CUDA(cudaFree(device_candidate));
	PROBE_CUDA(cudaFree(device_error));
}

int main(void)
{
	static const ProbeShape shapes[] = {
		{"pass chunk, packed keys", 128u, 65536u, 0u},
		{"pass chunk, packed keys, long", 128u, 262144u, 0u},
		{"pass chunk, packed keys, 1M", 128u, 1048576u, 0u},
		{"pass chunk, uneven rows", 100u, 70001u, 0u},
		{"pass tail chunk, 1M", 32u, 1048576u, 0u},
		{"wave, paged keys", 2048u, 16384u, 1u},
		{"few rows, ragged keys", 40u, 5003u, 0u},
		{"few rows, paged ragged keys", 7u, 777u, 1u},
		{"one row", 1u, 300u, 0u},
	};
	for (size_t index = 0; index < sizeof(shapes) / sizeof(shapes[0]); ++index)
		ProbeCase(shapes[index]);
	if ( probe_failures != 0 )
	{
		printf("FAIL %d index score cases\n",probe_failures);
		return 1;
	}
	printf("PASS index score kernel against its one-tile form\n");
	return 0;
}
