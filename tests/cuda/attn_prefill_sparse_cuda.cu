#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <vector>

#include "runtime/launch.h"
#include "inference/kernels/dtype.cuh"
#include "inference/kernels/attn.cuh"
#include "inference/kernels/attn_prefill.cuh"

#define PROBE_PAGE 64u
#define PROBE_LATENT 512u
#define PROBE_ROPE 64u
#define PROBE_WIDTH (PROBE_LATENT + PROBE_ROPE)
#define PROBE_HEADS 4u
#define PROBE_SELECTED 2048u
#define PROBE_SEQUENCES 3u
#define PROBE_ABSENT 0xffffffffu

struct ProbeKv
{
	static constexpr uint32_t kSlotBytes = PROBE_WIDTH * 2u;
	static constexpr uint32_t kPageSlots = PROBE_PAGE;
	static constexpr uint32_t kPageBytes = kSlotBytes * PROBE_PAGE;
	static constexpr bool kGrows = true;
	static __host__ __device__ constexpr uint32_t PageOf(uint32_t position) { return position / PROBE_PAGE; }
	static __host__ __device__ constexpr uint32_t SlotInPage(uint32_t position) { return position % PROBE_PAGE; }
	static __host__ __device__ constexpr uint64_t PagesForTokens(uint64_t tokens) { return (tokens + PROBE_PAGE - 1u) / PROBE_PAGE; }
};

typedef LmKvGeometry<ProbeKv::kSlotBytes,1u,true> ProbeRemap;

enum ProbePattern
{
	PROBE_RANDOM,
	PROBE_GROUPS,
	PROBE_RESIDUES,
	PROBE_RECENT,
	PROBE_EVERY
};

typedef struct ProbeSparseCase
{
	const char *label;
	uint32_t sequences;
	uint32_t rows[PROBE_SEQUENCES];
	uint32_t prefix[PROBE_SEQUENCES];
	ProbePattern pattern;
	uint32_t group;
	uint32_t shared_percent;
	uint32_t future;
	uint32_t table;
	uint32_t remap;
	uint32_t equal_unions;
} ProbeSparseCase;

static int probe_failures;

static __host__ __device__ uint16_t ProbeValue(uint64_t key)
{
	uint64_t z = key * 0x9e3779b97f4a7c15ull + 0x632be59bd9b4e019ull;
	z = (z ^ (z >> 30u)) * 0xbf58476d1ce4e5b9ull;
	z = (z ^ (z >> 27u)) * 0x94d049bb133111ebull;
	z ^= z >> 31u;
	float value = (float)((int32_t)((z >> 40u) & 0xfffu) - 2048) / 2048.0f;
	uint32_t bits;
	memcpy(&bits,&value,4u);
	return (uint16_t)((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
}

static double ProbeFloat(uint16_t value)
{
	uint32_t bits = (uint32_t)value << 16u;
	float result;
	memcpy(&result,&bits,4u);
	return result;
}

static double ProbeRoundBf16(double value)
{
	float narrow = (float)value;
	uint32_t bits;
	memcpy(&bits,&narrow,4u);
	return ProbeFloat((uint16_t)((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u));
}

static double ProbeUlp(double value)
{
	int exponent;
	frexp(fabs(value) > 1.0e-30 ? value : 1.0e-30,&exponent);
	return(ldexp(1.0,exponent - 8));
}

static uint64_t ProbeMix(uint64_t z)
{
	z = z * 0x9e3779b97f4a7c15ull + 1u;
	z = (z ^ (z >> 30u)) * 0xbf58476d1ce4e5b9ull;
	return(z ^ (z >> 31u));
}

__global__ void ProbeFillKernel(uint16_t *values,uint64_t count,uint64_t salt)
{
	uint64_t index;
	for (index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < count; index += (uint64_t)gridDim.x * blockDim.x)
		values[index] = ProbeValue(index ^ salt);
}

#define PROBE_CUDA(call) do { cudaError_t probe_error = (call); if ( probe_error != cudaSuccess ) { printf("FAIL cuda %s: %s\n",#call,cudaGetErrorString(probe_error)); exit(1); } } while (0)

template<class T>
static T *ProbeDevice(const std::vector<T> &host)
{
	T *device;
	PROBE_CUDA(cudaMalloc((void **)&device,(host.size() > 0u ? host.size() : 1u) * sizeof(T)));
	if ( !host.empty() )
		PROBE_CUDA(cudaMemcpy(device,host.data(),host.size() * sizeof(T),cudaMemcpyHostToDevice));
	return device;
}

static void ProbePick(std::vector<uint32_t> &pick,std::vector<char> &used,uint32_t low,uint32_t high,uint32_t count,uint64_t seed,uint32_t residue)
{
	uint32_t span = high >= low ? high - low + 1u : 0u,available = 0u,position;
	uint64_t draw;
	for (position = low; span != 0u && position <= high; position++)
		available += used[position] == 0 && (residue == PROBE_ABSENT || position % 8u == residue) ? 1u : 0u;
	count = count < available ? count : available;
	for (draw = 0u; count != 0u; draw++)
	{
		position = low + (uint32_t)(ProbeMix(seed * 1000003u + draw) % span);
		if ( used[position] != 0 || (residue != PROBE_ABSENT && position % 8u != residue) )
			continue;
		used[position] = 1;
		pick.push_back(position);
		count--;
	}
}

static std::vector<uint32_t> ProbeSelection(const ProbeSparseCase &probe,uint32_t sequence,uint32_t local,uint32_t position,uint32_t context)
{
	std::vector<uint32_t> pick;
	std::vector<char> used(context,0);
	const uint32_t future = probe.future != 0u && position + 1u < context ? std::min(48u,context - 1u - position) : 0u;
	const uint64_t seed = ((uint64_t)sequence << 40u) + ((uint64_t)local << 8u) + 17u;
	if ( probe.pattern == PROBE_EVERY )
		for (uint32_t key = 0u; key <= position && pick.size() < PROBE_SELECTED - future; key++)
			pick.push_back(key);
	else if ( probe.pattern == PROBE_RECENT )
		for (uint32_t key = position >= PROBE_SELECTED - 1u - future ? position - (PROBE_SELECTED - 1u - future) : 0u; key <= position; key++)
			pick.push_back(key);
	else if ( probe.pattern == PROBE_RESIDUES )
		ProbePick(pick,used,0u,position,PROBE_SELECTED - future,seed,local % 8u);
	else
	{
		uint32_t shared = 0u;
		if ( probe.pattern == PROBE_GROUPS )
		{
			const uint32_t first = local / probe.group * probe.group;
			const uint32_t floor = probe.prefix[sequence] + first;
			shared = (PROBE_SELECTED - future) * probe.shared_percent / 100u;
			ProbePick(pick,used,0u,floor,shared,((uint64_t)sequence << 40u) + ((uint64_t)first << 8u) + 5u,PROBE_ABSENT);
		}
		ProbePick(pick,used,0u,position,PROBE_SELECTED - future - (uint32_t)pick.size(),seed,PROBE_ABSENT);
	}
	for (uint32_t key = position + 1u; pick.size() < PROBE_SELECTED && key < position + 1u + future; key++)
		pick.push_back(key);
	std::sort(pick.begin(),pick.end());
	while (pick.size() < PROBE_SELECTED)
		pick.push_back(PROBE_ABSENT);
	return pick;
}

static std::vector<uint32_t> ProbeBlocks(const std::vector<uint32_t> &sequence_of_row,uint32_t rows_per_block,uint32_t table)
{
	std::vector<uint32_t> first_rows;
	uint32_t row = 0u,taken;
	while (row < sequence_of_row.size())
	{
		for (taken = 1u; row + taken < sequence_of_row.size() && taken < rows_per_block && (table == 0u || sequence_of_row[row + taken] == sequence_of_row[row]); taken++)
		{
		}
		first_rows.push_back(row);
		row += taken;
	}
	first_rows.push_back((uint32_t)sequence_of_row.size());
	return first_rows;
}

static void ProbeFlash(const std::vector<double> &scores,const std::vector<uint32_t> &order,const std::vector<char> &member,const std::vector<const float *> &values,std::vector<double> &out)
{
	double running = -INFINITY,total = 0.0;
	std::fill(out.begin(),out.end(),0.0);
	for (uint64_t first = 0u; first < order.size(); first += LM_PREFILL_ATTN_TILE)
	{
		const uint64_t last = std::min<uint64_t>(first + LM_PREFILL_ATTN_TILE,order.size());
		double peak = running;
		for (uint64_t key = first; key < last; key++)
			if ( member[key] != 0 )
				peak = scores[key] > peak ? scores[key] : peak;
		if ( peak == -INFINITY )
			continue;
		if ( running != -INFINITY && peak != running )
		{
			const double factor = exp(running - peak);
			total *= factor;
			for (double &element : out)
				element *= factor;
		}
		running = peak;
		for (uint64_t key = first; key < last; key++)
		{
			if ( member[key] == 0 )
				continue;
			const double weight = exp(scores[key] - running),rounded = ProbeRoundBf16(weight);
			total += weight;
			for (uint32_t element = 0u; element < PROBE_LATENT; element++)
				out[element] += rounded * values[key][element];
		}
	}
	for (double &element : out)
		element /= total;
}

template<uint32_t QUERIES>
static void ProbeLaunch(const ProbeSparseCase &probe,LmKvView view,LmKvView remap_view,const uint16_t *latent,const uint16_t *rope,const uint32_t *sequences,const uint32_t *positions,const uint32_t *selected,uint32_t rows,uint32_t remap,uint16_t *out)
{
	constexpr uint32_t ROWS_PER_BLOCK = QUERIES / PROBE_HEADS;
	const uint32_t table_blocks = probe.table != 0u ? (rows + ROWS_PER_BLOCK - 1u) / ROWS_PER_BLOCK + probe.sequences : (rows + ROWS_PER_BLOCK - 1u) / ROWS_PER_BLOCK;
	const uint64_t entries = LmPrefillSparseUnionEntries<QUERIES>(table_blocks,PROBE_HEADS,PROBE_SELECTED);
	const float scale = 1.0f / sqrtf((float)PROBE_WIDTH);
	uint32_t *table = 0,*union_positions,*union_counts;
	uint8_t *union_masks;
	PROBE_CUDA(cudaMalloc((void **)&union_positions,entries * sizeof(uint32_t)));
	PROBE_CUDA(cudaMalloc((void **)&union_masks,entries));
	PROBE_CUDA(cudaMalloc((void **)&union_counts,table_blocks * sizeof(uint32_t)));
	if ( probe.table != 0u )
	{
		PROBE_CUDA(cudaMalloc((void **)&table,table_blocks * sizeof(uint32_t)));
		PROBE_CUDA((LmPrefillBlockTableLaunch<PROBE_HEADS,QUERIES>(view,sequences,rows,table,table_blocks,0,1u,0)));
	}
	PROBE_CUDA(cudaMemset(out,0xff,(uint64_t)rows * PROBE_HEADS * PROBE_LATENT * 2u));
	if ( remap != 0u )
		PROBE_CUDA((LmLatentAttentionSparsePrefillLaunch<ProbeRemap,PROBE_LATENT,PROBE_ROPE,PROBE_SELECTED,QUERIES>(latent,rope,remap_view,sequences,positions,PROBE_HEADS,scale,out,rows,table,probe.table != 0u ? table_blocks : 0u,
			selected,PROBE_SELECTED,union_positions,union_masks,union_counts,entries,0)));
	else
		PROBE_CUDA((LmLatentAttentionSparsePrefillLaunch<ProbeKv,PROBE_LATENT,PROBE_ROPE,PROBE_SELECTED,QUERIES>(latent,rope,view,sequences,positions,PROBE_HEADS,scale,out,rows,table,probe.table != 0u ? table_blocks : 0u,
			selected,PROBE_SELECTED,union_positions,union_masks,union_counts,entries,0)));
	PROBE_CUDA(cudaDeviceSynchronize());
	cudaFree(union_positions);
	cudaFree(union_masks);
	cudaFree(union_counts);
	if ( table != 0 )
		cudaFree(table);
}

static void ProbeCase(const ProbeSparseCase &probe)
{
	uint32_t context = 0u,rows = 0u,sequence,local,row,page,index,head,element;
	for (sequence = 0u; sequence < probe.sequences; sequence++)
	{
		context = std::max(context,probe.prefix[sequence] + probe.rows[sequence]);
		rows += probe.rows[sequence];
	}
	const uint32_t sequence_pages = (context + PROBE_PAGE - 1u) / PROBE_PAGE,pages = sequence_pages * probe.sequences;
	const uint64_t tokens = (uint64_t)context * probe.sequences,query_count = (uint64_t)rows * PROBE_HEADS;
	const uint32_t spread = probe.sequences * ((context + 6u) / 7u);
	const double scale = 1.0 / sqrt((double)(float)PROBE_WIDTH);
	std::vector<uint32_t> table(pages),token_sequence(tokens),token_position(tokens),sequence_of_row,row_position,selected,remap(tokens);
	std::vector<uint16_t> out32(query_count * PROBE_LATENT),out16(out32.size()),again(out32.size()),remapped(out32.size());
	std::vector<float> keys(tokens * PROBE_WIDTH);
	uint32_t *table_device,*token_sequence_device,*token_position_device,*sequences_device,*positions_device,*selected_device,*remap_device;
	uint16_t *values_device,*latent_device,*rope_device,*out_device;
	uint8_t *cache_device,*pool_device;
	LmKvAccessError *error_device,error_host;
	LmKvView view,remap_view;
	double worst_model32 = 0.0,worst_model16 = 0.0,worst_exact = 0.0,worst_allowance = 0.0,model_exact = 0.0,worst_between = 0.0,worst_steps = 0.0;
	uint64_t differing = 0u,union_keys32 = 0u,union_keys16 = 0u;
	int failed = probe_failures;

	for (page = 0u; page < pages; page++)
		table[page] = pages - 1u - page;
	for (index = 0u; index < tokens; index++)
	{
		token_sequence[index] = index / context;
		token_position[index] = index % context;
		remap[index] = index % context % 7u * spread + index / context * ((context + 6u) / 7u) + index % context / 7u;
	}
	for (sequence = 0u; sequence < probe.sequences; sequence++)
		for (local = 0u; local < probe.rows[sequence]; local++)
		{
			const uint32_t position = probe.prefix[sequence] + local;
			std::vector<uint32_t> pick = ProbeSelection(probe,sequence,local,position,probe.prefix[sequence] + probe.rows[sequence]);
			sequence_of_row.push_back(sequence);
			row_position.push_back(position);
			selected.insert(selected.end(),pick.begin(),pick.end());
		}
	table_device = ProbeDevice(table);
	token_sequence_device = ProbeDevice(token_sequence);
	token_position_device = ProbeDevice(token_position);
	sequences_device = ProbeDevice(sequence_of_row);
	positions_device = ProbeDevice(row_position);
	selected_device = ProbeDevice(selected);
	remap_device = ProbeDevice(remap);
	PROBE_CUDA(cudaMalloc((void **)&values_device,tokens * PROBE_WIDTH * 2u));
	PROBE_CUDA(cudaMalloc((void **)&latent_device,query_count * PROBE_LATENT * 2u));
	PROBE_CUDA(cudaMalloc((void **)&rope_device,query_count * PROBE_ROPE * 2u));
	PROBE_CUDA(cudaMalloc((void **)&cache_device,(uint64_t)pages * ProbeKv::kPageBytes));
	PROBE_CUDA(cudaMalloc((void **)&pool_device,(uint64_t)spread * 7u * ProbeKv::kSlotBytes));
	PROBE_CUDA(cudaMalloc((void **)&out_device,query_count * PROBE_LATENT * 2u));
	PROBE_CUDA(cudaMalloc((void **)&error_device,sizeof(LmKvAccessError)));
	LmKvAccessErrorReset(&error_host);
	PROBE_CUDA(cudaMemcpy(error_device,&error_host,sizeof(error_host),cudaMemcpyHostToDevice));
	ProbeFillKernel<<<1024,256>>>(values_device,tokens * PROBE_WIDTH,0u);
	ProbeFillKernel<<<256,256>>>(latent_device,query_count * PROBE_LATENT,1ull << 60u);
	ProbeFillKernel<<<256,256>>>(rope_device,query_count * PROBE_ROPE,1ull << 61u);
	PROBE_CUDA(cudaMemset(cache_device,0,(uint64_t)pages * ProbeKv::kPageBytes));
	PROBE_CUDA(cudaMemset(pool_device,0,(uint64_t)spread * 7u * ProbeKv::kSlotBytes));
	if ( LmKvViewInitialize(&view,cache_device,table_device,sequence_pages,probe.sequences,pages,error_device) != 0 ||
		LmKvViewInitialize(&remap_view,pool_device,remap_device,context,probe.sequences,spread * 7u,error_device) != 0 )
	{
		printf("FAIL %s: view\n",probe.label);
		probe_failures++;
		return;
	}
	LmKvStoreKernel<ProbeKv,256u><<<(uint32_t)tokens,256>>>(view,values_device,token_sequence_device,token_position_device,(uint32_t)tokens,PROBE_WIDTH);
	LmKvStoreKernel<ProbeRemap,256u><<<(uint32_t)tokens,256>>>(remap_view,values_device,token_sequence_device,token_position_device,(uint32_t)tokens,PROBE_WIDTH);
	PROBE_CUDA(cudaDeviceSynchronize());
	for (index = 0u; index < tokens * PROBE_WIDTH; index++)
		keys[index] = (float)ProbeFloat(ProbeValue(index));

	ProbeLaunch<32u>(probe,view,remap_view,latent_device,rope_device,sequences_device,positions_device,selected_device,rows,0u,out_device);
	PROBE_CUDA(cudaMemcpy(out32.data(),out_device,out32.size() * 2u,cudaMemcpyDeviceToHost));
	ProbeLaunch<32u>(probe,view,remap_view,latent_device,rope_device,sequences_device,positions_device,selected_device,rows,0u,out_device);
	PROBE_CUDA(cudaMemcpy(again.data(),out_device,again.size() * 2u,cudaMemcpyDeviceToHost));
	ProbeLaunch<16u>(probe,view,remap_view,latent_device,rope_device,sequences_device,positions_device,selected_device,rows,0u,out_device);
	PROBE_CUDA(cudaMemcpy(out16.data(),out_device,out16.size() * 2u,cudaMemcpyDeviceToHost));
	if ( probe.remap != 0u )
	{
		ProbeLaunch<32u>(probe,view,remap_view,latent_device,rope_device,sequences_device,positions_device,selected_device,rows,1u,out_device);
		PROBE_CUDA(cudaMemcpy(remapped.data(),out_device,remapped.size() * 2u,cudaMemcpyDeviceToHost));
		if ( memcmp(remapped.data(),out32.data(),out32.size() * 2u) != 0 )
		{
			printf("FAIL %s: keys read through a per-position remap give different bits than through pages\n",probe.label);
			probe_failures++;
		}
	}
	PROBE_CUDA(cudaMemcpy(&error_host,error_device,sizeof(error_host),cudaMemcpyDeviceToHost));
	if ( error_host.error_code != LM_FRAME_ERROR_NONE )
	{
		printf("FAIL %s: KV access error %u\n",probe.label,(unsigned)error_host.error_code);
		probe_failures++;
	}
	if ( memcmp(again.data(),out32.data(),out32.size() * 2u) != 0 )
	{
		printf("FAIL %s: two identical launches give different bits\n",probe.label);
		probe_failures++;
	}

	for (uint32_t variant = 0u; variant < 2u; variant++)
	{
		const uint32_t rows_per_block = variant == 0u ? 32u / PROBE_HEADS : 16u / PROBE_HEADS;
		const std::vector<uint32_t> blocks = ProbeBlocks(sequence_of_row,rows_per_block,probe.table);
		const std::vector<uint16_t> &kernel = variant == 0u ? out32 : out16;
		double &worst_model = variant == 0u ? worst_model32 : worst_model16;
		for (uint32_t block = 0u; block + 1u < blocks.size(); block++)
		{
			std::vector<uint32_t> order;
			for (row = blocks[block]; row < blocks[block + 1u]; row++)
				for (index = 0u; index < PROBE_SELECTED; index++)
					if ( selected[(uint64_t)row * PROBE_SELECTED + index] <= row_position[row] )
						order.push_back(selected[(uint64_t)row * PROBE_SELECTED + index]);
			std::sort(order.begin(),order.end());
			order.erase(std::unique(order.begin(),order.end()),order.end());
			(variant == 0u ? union_keys32 : union_keys16) += order.size();
			for (row = blocks[block]; row < blocks[block + 1u]; row++)
			{
				const uint32_t owner = sequence_of_row[row];
				const uint32_t *list = &selected[(uint64_t)row * PROBE_SELECTED];
				std::vector<char> member(order.size(),0);
				std::vector<const float *> values(order.size());
				std::vector<double> scores(order.size()),flash(PROBE_LATENT),exact(PROBE_LATENT),magnitude(PROBE_LATENT);
				for (index = 0u; index < order.size(); index++)
				{
					member[index] = order[index] <= row_position[row] && std::binary_search(list,list + PROBE_SELECTED,order[index]) ? 1 : 0;
					values[index] = &keys[((uint64_t)owner * context + order[index]) * PROBE_WIDTH];
				}
				for (head = 0u; head < PROBE_HEADS; head++)
				{
					const uint64_t q = (uint64_t)row * PROBE_HEADS + head;
					std::vector<double> query(PROBE_WIDTH);
					double top = -INFINITY,total = 0.0,peak = 0.0;
					for (element = 0u; element < PROBE_LATENT; element++)
						query[element] = ProbeFloat(ProbeValue((q * PROBE_LATENT + element) ^ (1ull << 60u)));
					for (element = 0u; element < PROBE_ROPE; element++)
						query[PROBE_LATENT + element] = ProbeFloat(ProbeValue((q * PROBE_ROPE + element) ^ (1ull << 61u)));
					for (index = 0u; index < order.size(); index++)
					{
						double dot = 0.0;
						if ( member[index] == 0 )
							continue;
						for (element = 0u; element < PROBE_WIDTH; element++)
							dot += query[element] * values[index][element];
						scores[index] = dot * scale;
						top = scores[index] > top ? scores[index] : top;
					}
					ProbeFlash(scores,order,member,values,flash);
					if ( variant == 0u )
					{
						std::fill(exact.begin(),exact.end(),0.0);
						std::fill(magnitude.begin(),magnitude.end(),0.0);
						for (index = 0u; index < order.size(); index++)
						{
							if ( member[index] == 0 )
								continue;
							const double weight = exp(scores[index] - top);
							total += weight;
							for (element = 0u; element < PROBE_LATENT; element++)
							{
								exact[element] += weight * values[index][element];
								magnitude[element] += weight * fabs(values[index][element]);
							}
						}
					}
					for (element = 0u; element < PROBE_LATENT; element++)
						peak = std::max(peak,fabs(flash[element]));
					for (element = 0u; element < PROBE_LATENT; element++)
					{
						const double got = ProbeFloat(kernel[q * PROBE_LATENT + element]);
						const double error = fabs(got - flash[element]) / (ProbeUlp(flash[element]) + ldexp(peak,-14));
						worst_model = error > worst_model ? error : worst_model;
						if ( variant == 0u )
						{
							const double reference = exact[element] / total;
							worst_exact = std::max(worst_exact,fabs(got - reference));
							model_exact = std::max(model_exact,fabs(flash[element] - reference));
							worst_allowance = std::max(worst_allowance,fabs(got - reference) / (ldexp(magnitude[element] / total,-8) + ProbeUlp(reference) + ldexp(peak,-14)));
						}
					}
				}
			}
		}
	}
	for (uint64_t q = 0u; q < query_count; q++)
	{
		double peak = 0.0;
		for (element = 0u; element < PROBE_LATENT; element++)
			peak = std::max(peak,fabs(ProbeFloat(out16[q * PROBE_LATENT + element])));
		for (element = 0u; element < PROBE_LATENT; element++)
		{
			const double a = ProbeFloat(out32[q * PROBE_LATENT + element]),b = ProbeFloat(out16[q * PROBE_LATENT + element]);
			differing += out32[q * PROBE_LATENT + element] != out16[q * PROBE_LATENT + element] ? 1u : 0u;
			worst_between = std::max(worst_between,fabs(a - b));
			worst_steps = std::max(worst_steps,fabs(a - b) / ProbeUlp(peak));
		}
	}
	if ( !(worst_model32 <= 2.0) || !(worst_model16 <= 2.0) )
	{
		printf("FAIL %s: the kernel departs from the f64 flash form of its own union tiles by %.2f (32 queries) and %.2f (16 queries) of one BF16 step plus 2^-14 of the head's peak\n",probe.label,worst_model32,worst_model16);
		probe_failures++;
	}
	if ( !(worst_allowance <= 1.0) )
	{
		printf("FAIL %s: the kernel lies %.2e from exact sparse attention, %.2f of the allowance for BF16 tile weights (2^-8 of the weighted |V|) and one BF16 output step\n",probe.label,worst_exact,worst_allowance);
		probe_failures++;
	}
	if ( probe.equal_unions != 0u && differing != 0u )
	{
		printf("FAIL %s: blocks of 8 and of 4 rows hold the same union here, yet %llu outputs differ\n",probe.label,(unsigned long long)differing);
		probe_failures++;
	}
	printf("%s %s: %u rows, union %.0f keys a block of 8 rows and %.0f of 4, flash-form error %.2f / %.2f steps (32 / 16 queries), |out-exact| %.2e (%.2f of its allowance; f64 flash form %.2e), 32 vs 16 queries: %.2f%% of outputs differ, worst %.2e (%.2f BF16 steps of the head's largest output)\n",
		failed == probe_failures ? "PASS" : "FAIL",probe.label,rows,(double)union_keys32 / (ProbeBlocks(sequence_of_row,8u,probe.table).size() - 1u),(double)union_keys16 / (ProbeBlocks(sequence_of_row,4u,probe.table).size() - 1u),
		worst_model32,worst_model16,worst_exact,worst_allowance,model_exact,100.0 * differing / out32.size(),worst_between,worst_steps);
	cudaFree(table_device); cudaFree(token_sequence_device); cudaFree(token_position_device); cudaFree(sequences_device); cudaFree(positions_device);
	cudaFree(selected_device); cudaFree(remap_device); cudaFree(values_device); cudaFree(latent_device); cudaFree(rope_device);
	cudaFree(cache_device); cudaFree(pool_device); cudaFree(out_device); cudaFree(error_device);
}

int main(void)
{
	static const ProbeSparseCase cases[] = {
		{ "random 61 rows after 3000", 1u, { 61u }, { 3000u }, PROBE_RANDOM, 0u, 0u, 1u, 0u, 1u, 0u },
		{ "every key from the cache start", 1u, { 45u }, { 0u }, PROBE_EVERY, 0u, 0u, 1u, 0u, 0u, 0u },
		{ "90% shared in groups of 8", 1u, { 64u }, { 9000u }, PROBE_GROUPS, 8u, 90u, 0u, 0u, 0u, 0u },
		{ "75% shared in groups of 3", 1u, { 70u }, { 9000u }, PROBE_GROUPS, 3u, 75u, 1u, 0u, 0u, 0u },
		{ "identical in groups of 8", 1u, { 40u }, { 5000u }, PROBE_GROUPS, 8u, 100u, 0u, 0u, 0u, 1u },
		{ "disjoint residues", 1u, { 43u }, { 17000u }, PROBE_RESIDUES, 0u, 0u, 0u, 0u, 0u, 0u },
		{ "recent window", 1u, { 37u }, { 2500u }, PROBE_RECENT, 0u, 0u, 1u, 0u, 0u, 0u },
		{ "three sequences through a block table", 3u, { 13u, 22u, 9u }, { 2100u, 40u, 4000u }, PROBE_RANDOM, 0u, 0u, 1u, 1u, 1u, 0u },
		{ "random 19 rows after 140000, two union windows", 1u, { 19u }, { 140000u }, PROBE_RANDOM, 0u, 0u, 1u, 0u, 0u, 0u },
	};
	for (uint32_t index = 0u; index < sizeof(cases) / sizeof(cases[0]); index++)
		ProbeCase(cases[index]);
	if ( probe_failures != 0 )
	{
		printf("FAIL %d checks\n",probe_failures);
		return 1;
	}
	printf("PASS sparse latent prefill attention on the device: blocks of 32 and of 16 queries (8 and 4 rows of 4 heads) match the f64 flash form of their own union tiles (each 32-key tile's weights rounded to BF16 against the running maximum) to within two BF16 output steps, lie within that rounding (2^-8 of the weighted |V|) plus one output step of exact sparse attention over each row's selected keys at or below its position, read the same bits through pages and a per-position remap, and give the same bits when both block sizes hold the same union\n");
	return 0;
}
