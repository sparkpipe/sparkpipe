#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "runtime/launch.h"
#include "inference/kernels/dtype.cuh"
#include "inference/kernels/project.cuh"
#include "inference/kernels/attn_expanded_prefill.cuh"

#define EXPAND_PAGE 64u
#define EXPAND_LATENT 512u
#define EXPAND_ROPE 64u
#define EXPAND_VALUE 128u
#define EXPAND_WIDTH (EXPAND_LATENT + EXPAND_ROPE)
#define EXPAND_LATENT_RECORD LM_LATENT_SHARD_RECORD_FLOATS(EXPAND_LATENT)
#define EXPAND_VALUE_RECORD LM_LATENT_SHARD_RECORD_FLOATS(EXPAND_VALUE)

struct ExpandKv
{
	static constexpr uint32_t kSlotBytes = EXPAND_WIDTH * 2u;
	static constexpr uint32_t kPageSlots = EXPAND_PAGE;
	static constexpr uint32_t kPageBytes = kSlotBytes * EXPAND_PAGE;
	static constexpr bool kGrows = true;
	static __host__ __device__ constexpr uint32_t PageOf(uint32_t position) { return position / EXPAND_PAGE; }
	static __host__ __device__ constexpr uint32_t SlotInPage(uint32_t position) { return position % EXPAND_PAGE; }
	static __host__ __device__ constexpr uint64_t PagesForTokens(uint64_t tokens) { return (tokens + EXPAND_PAGE - 1u) / EXPAND_PAGE; }
};

static int expand_failures;

#define EXPAND_CUDA(call) do { cudaError_t expand_error = (call); if ( expand_error != cudaSuccess ) { printf("FAIL cuda %s: %s\n",#call,cudaGetErrorString(expand_error)); exit(1); } } while (0)

static void ExpandCheck(int condition,const char *what,const char *label)
{
	if ( condition )
		return;
	printf("FAIL %s: %s\n",label,what);
	expand_failures++;
}

static uint64_t ExpandMix(uint64_t z)
{
	z = z * 0x9e3779b97f4a7c15ull + 0x632be59bd9b4e019ull;
	z = (z ^ (z >> 30u)) * 0xbf58476d1ce4e5b9ull;
	z = (z ^ (z >> 27u)) * 0x94d049bb133111ebull;
	return z ^ (z >> 31u);
}

static uint16_t ExpandHostBf16(float value)
{
	uint32_t bits;
	memcpy(&bits,&value,4u);
	bits += 0x7fffu + ((bits >> 16u) & 1u);
	return (uint16_t)(bits >> 16u);
}

static double ExpandHostFloat(uint16_t value)
{
	uint32_t bits = (uint32_t)value << 16u;
	float result;
	memcpy(&result,&bits,4u);
	return result;
}

static std::vector<uint16_t> ExpandRandom(uint64_t count,uint64_t salt,float scale)
{
	std::vector<uint16_t> values(count);
	for (uint64_t index = 0u; index < count; index++)
		values[index] = ExpandHostBf16((float)((int32_t)((ExpandMix(index ^ salt) >> 40u) & 0xfffu) - 2048) / 2048.0f * scale);
	return values;
}

template<class T>
static T *ExpandDevice(const std::vector<T> &host)
{
	T *device;
	EXPAND_CUDA(cudaMalloc((void **)&device,host.size() * sizeof(T) + 16u));
	EXPAND_CUDA(cudaMemcpy(device,host.data(),host.size() * sizeof(T),cudaMemcpyHostToDevice));
	return device;
}

static double ExpandWorst(const std::vector<uint16_t> &value,const std::vector<double> &reference,uint32_t rows,uint32_t heads)
{
	double worst = 0.0;
	for (uint32_t row = 0u; row < rows; row++)
		for (uint32_t head = 0u; head < heads; head++)
		{
			const uint64_t base = ((uint64_t)row * heads + head) * EXPAND_VALUE;
			double peak = 0.0;
			for (uint32_t element = 0u; element < EXPAND_VALUE; element++)
				peak = fmax(peak,fabs(reference[base + element]));
			for (uint32_t element = 0u; element < EXPAND_VALUE; element++)
				worst = fmax(worst,fabs(ExpandHostFloat(value[base + element]) - reference[base + element]) / fmax(peak,1e-6));
		}
	return worst;
}

static void ExpandCase(uint32_t rows,uint32_t context,uint32_t degree,uint32_t heads_per_rank,uint32_t window_keys)
{
	const uint32_t pages = (context + EXPAND_PAGE - 1u) / EXPAND_PAGE;
	const uint64_t query_count = (uint64_t)rows * heads_per_rank * EXPAND_WIDTH;
	const uint64_t latent_stride = (uint64_t)rows * heads_per_rank * EXPAND_LATENT_RECORD,value_stride_records = (uint64_t)rows * heads_per_rank * EXPAND_VALUE_RECORD;
	const uint64_t out_count = (uint64_t)rows * heads_per_rank * EXPAND_VALUE;
	const float scale = 0.07216878365f;
	SparkKvShard probe = {degree,0u,1u};
	const uint32_t most = SparkKvShardGatherKeys(probe,context);
	const uint64_t key_stride = ((uint64_t)most * ExpandKv::kSlotBytes + 4095u) / 4096u * 4096u;
	const uint32_t windows = window_keys != 0u ? (most + window_keys - 1u) / window_keys : 1u;
	const uint32_t pass_keys = window_keys != 0u ? window_keys : most;
	const uint64_t value_stride = (uint64_t)pass_keys * heads_per_rank * EXPAND_VALUE * 2u;
	std::vector<uint32_t> table(pages),positions(rows),sequences(rows,0u),contexts(1,context),token_sequence(context,0u),token_position(context),plan(3u,0u);
	std::vector<uint16_t> latent_host = ExpandRandom((uint64_t)context * EXPAND_WIDTH,3u,2.0f);
	std::vector<uint16_t> query_host = ExpandRandom(query_count,5u,2.0f);
	std::vector<uint16_t> weight_host = ExpandRandom((uint64_t)heads_per_rank * EXPAND_VALUE * EXPAND_LATENT,7u,0.1f);
	std::vector<uint8_t *> pools(degree);
	std::vector<uint16_t> expanded(out_count),absorbed(out_count);
	std::vector<double> reference(out_count,0.0);
	uint32_t *table_device,*positions_device,*sequences_device,*contexts_device,*token_sequence_device,*token_position_device,*plan_device;
	uint16_t *latent_device,*query_device,*weight_device,*expanded_device,*absorbed_device,*latent_out_device;
	uint8_t *keys_device,*values_device;
	float *latent_partials,*value_partials,*running_device;
	LmKvAccessError *error_device,error_host;
	uint32_t page,row,rank,head,window,index;
	char label[160];

	snprintf(label,sizeof(label),"rows%u ctx%u degree%u heads%u windows%u",rows,context,degree,heads_per_rank,windows);
	for (page = 0u; page < pages; page++)
		table[page] = (page * 7u + 3u) % pages;
	for (row = 0u; row < rows; row++)
		positions[row] = context - rows + row;
	for (index = 0u; index < context; index++)
		token_position[index] = index;
	plan[2] = context;
	table_device = ExpandDevice(table);
	positions_device = ExpandDevice(positions);
	sequences_device = ExpandDevice(sequences);
	contexts_device = ExpandDevice(contexts);
	token_sequence_device = ExpandDevice(token_sequence);
	token_position_device = ExpandDevice(token_position);
	plan_device = ExpandDevice(plan);
	latent_device = ExpandDevice(latent_host);
	query_device = ExpandDevice(query_host);
	weight_device = ExpandDevice(weight_host);
	EXPAND_CUDA(cudaMalloc((void **)&keys_device,(uint64_t)degree * key_stride));
	EXPAND_CUDA(cudaMalloc((void **)&values_device,(uint64_t)degree * value_stride));
	EXPAND_CUDA(cudaMalloc((void **)&latent_partials,(uint64_t)degree * latent_stride * sizeof(float)));
	EXPAND_CUDA(cudaMalloc((void **)&value_partials,(uint64_t)degree * value_stride_records * sizeof(float)));
	EXPAND_CUDA(cudaMalloc((void **)&running_device,value_stride_records * sizeof(float)));
	EXPAND_CUDA(cudaMalloc((void **)&expanded_device,out_count * 2u));
	EXPAND_CUDA(cudaMalloc((void **)&absorbed_device,out_count * 2u));
	EXPAND_CUDA(cudaMalloc((void **)&latent_out_device,(uint64_t)rows * heads_per_rank * EXPAND_LATENT * 2u));
	EXPAND_CUDA(cudaMalloc((void **)&error_device,sizeof(LmKvAccessError)));
	EXPAND_CUDA(cudaMemset(keys_device,0xff,(uint64_t)degree * key_stride));
	EXPAND_CUDA(cudaMemset(values_device,0xff,(uint64_t)degree * value_stride));
	LmKvAccessErrorReset(&error_host);
	EXPAND_CUDA(cudaMemcpy(error_device,&error_host,sizeof(error_host),cudaMemcpyHostToDevice));
	for (rank = 0u; rank < degree; rank++)
	{
		SparkKvShard shard = {degree,rank,1u};
		LmKvShardView local;
		const uint64_t bytes = SparkKvShardPoolBytes(shard,EXPAND_PAGE,ExpandKv::kSlotBytes,pages);
		EXPAND_CUDA(cudaMalloc((void **)&pools[rank],bytes));
		EXPAND_CUDA(cudaMemset(pools[rank],0,bytes));
		ExpandCheck(LmKvShardViewInitialize<ExpandKv>(&local,pools[rank],table_device,pages,1u,pages,error_device,shard) == 0,"shard view",label);
		LmKvShardStoreKernel<ExpandKv,256u><<<context,256>>>(local,latent_device,token_sequence_device,token_position_device,context,EXPAND_WIDTH);
		EXPAND_CUDA(cudaGetLastError());
		ExpandCheck(LmKvShardGatherPackLaunch<ExpandKv,256u>(local,plan_device,plan_device + 1u,plan_device + 2u,1u,most,
			keys_device + (uint64_t)rank * key_stride,0) == cudaSuccess,"gather pack launch",label);
	}
	auto view_of = [&](uint32_t source,uint32_t begin,uint32_t keys,LmKvShardGatherView *view) -> int
	{
		SparkKvShard shard = {degree,source,1u};
		LmKvShardView pages_view;
		if ( LmKvShardViewInitialize<ExpandKv>(&pages_view,pools[source],table_device,pages,1u,pages,error_device,shard) != 0 )
			return -1;
		if ( LmKvShardGatherViewInitialize<ExpandKv>(view,pages_view.pages,shard,keys_device + (uint64_t)source * key_stride + (uint64_t)begin * ExpandKv::kSlotBytes,
				plan_device + 1u,plan_device + 2u) != 0 )
			return -1;
		return window_keys != 0u ? LmKvShardGatherViewWindow(view,begin,begin + keys) : 0;
	};
	for (rank = 0u; rank < degree; rank++)
	{
		LmKvShardGatherView view;
		ExpandCheck(view_of(rank,0u,most,&view) == 0,"absorbed view",label);
		view.key_begin = 0u;
		view.key_end = LM_KV_SHARD_KEYS_ALL;
		ExpandCheck(LmLatentGatherPrefillLaunch<ExpandKv,LmKvShardGatherView,EXPAND_LATENT,EXPAND_ROPE>(view,query_device,heads_per_rank,sequences_device,
			contexts_device,positions_device,scale,latent_partials + (uint64_t)rank * latent_stride,rows,0) == cudaSuccess,"absorbed launch",label);
	}
	ExpandCheck(LmLatentShardMergeLaunch<EXPAND_LATENT>(latent_partials,latent_stride,degree,heads_per_rank,latent_out_device,rows,0) == cudaSuccess,"absorbed merge",label);
	LmPerHeadProjectSplitKernel<256u,EXPAND_LATENT,EXPAND_VALUE><<<dim3(rows,heads_per_rank,EXPAND_VALUE / LM_PER_HEAD_SPLIT_OUTPUTS),256u>>>(latent_out_device,weight_device,absorbed_device,heads_per_rank,rows);
	EXPAND_CUDA(cudaGetLastError());
	for (window = 0u; window < windows; window++)
	{
		const uint32_t begin = window * pass_keys,keys = most - begin < pass_keys ? most - begin : pass_keys;
		ExpandCheck(LmLatentValueExpandLaunch<EXPAND_LATENT,EXPAND_WIDTH,EXPAND_VALUE>(keys_device + (uint64_t)begin * ExpandKv::kSlotBytes,key_stride,degree,keys,
			weight_device,heads_per_rank,values_device,value_stride,0) == cudaSuccess,"value expansion launch",label);
		for (rank = 0u; rank < degree; rank++)
		{
			LmKvShardGatherView view;
			ExpandCheck(view_of(rank,begin,keys,&view) == 0,"expanded view",label);
			ExpandCheck(LmExpandedGatherPrefillLaunch<ExpandKv,EXPAND_LATENT,EXPAND_ROPE,EXPAND_VALUE>(view,values_device + (uint64_t)rank * value_stride,query_device,
				heads_per_rank,sequences_device,contexts_device,positions_device,scale,value_partials + (uint64_t)rank * value_stride_records,rows,0) == cudaSuccess,
				"expanded launch",label);
		}
		if ( windows > 1u )
			ExpandCheck(LmLatentShardFoldLaunch<EXPAND_VALUE>(value_partials,value_stride_records,degree,heads_per_rank,running_device,rows,window == 0u ? 1u : 0u,0) == cudaSuccess,
				"expanded fold",label);
	}
	if ( windows > 1u )
		ExpandCheck(LmLatentShardMergeLaunch<EXPAND_VALUE>(running_device,value_stride_records,1u,heads_per_rank,expanded_device,rows,0) == cudaSuccess,"expanded finish",label);
	else
		ExpandCheck(LmLatentShardMergeLaunch<EXPAND_VALUE>(value_partials,value_stride_records,degree,heads_per_rank,expanded_device,rows,0) == cudaSuccess,"expanded merge",label);
	EXPAND_CUDA(cudaDeviceSynchronize());
	EXPAND_CUDA(cudaMemcpy(&error_host,error_device,sizeof(error_host),cudaMemcpyDeviceToHost));
	ExpandCheck(error_host.error_code == LM_FRAME_ERROR_NONE,"no KV access error on a single-sequence wave",label);
	EXPAND_CUDA(cudaMemcpy(expanded.data(),expanded_device,out_count * 2u,cudaMemcpyDeviceToHost));
	EXPAND_CUDA(cudaMemcpy(absorbed.data(),absorbed_device,out_count * 2u,cudaMemcpyDeviceToHost));
	for (row = 0u; row < rows; row++)
		for (head = 0u; head < heads_per_rank; head++)
		{
			const uint16_t *query = &query_host[((uint64_t)row * heads_per_rank + head) * EXPAND_WIDTH];
			const uint32_t visible = positions[row] + 1u < context ? positions[row] + 1u : context;
			std::vector<double> scores(visible),latent(EXPAND_LATENT,0.0);
			double peak = -1e300,total = 0.0;
			uint32_t position,element,output;
			for (position = 0u; position < visible; position++)
			{
				double score = 0.0;
				for (element = 0u; element < EXPAND_WIDTH; element++)
					score += ExpandHostFloat(query[element]) * ExpandHostFloat(latent_host[(uint64_t)position * EXPAND_WIDTH + element]);
				scores[position] = score * scale;
				peak = fmax(peak,scores[position]);
			}
			for (position = 0u; position < visible; position++)
			{
				const double weight = exp(scores[position] - peak);
				total += weight;
				for (element = 0u; element < EXPAND_LATENT; element++)
					latent[element] += weight * ExpandHostFloat(latent_host[(uint64_t)position * EXPAND_WIDTH + element]);
			}
			for (output = 0u; output < EXPAND_VALUE; output++)
			{
				double value = 0.0;
				for (element = 0u; element < EXPAND_LATENT; element++)
					value += ExpandHostFloat(weight_host[((uint64_t)head * EXPAND_VALUE + output) * EXPAND_LATENT + element]) * latent[element] / total;
				reference[((uint64_t)row * heads_per_rank + head) * EXPAND_VALUE + output] = value;
			}
		}
	{
		const double expanded_worst = ExpandWorst(expanded,reference,rows,heads_per_rank),absorbed_worst = ExpandWorst(absorbed,reference,rows,heads_per_rank);
		ExpandCheck(window_keys == 0u || windows >= 2u,"a windowed case runs more than one pass, so the fold and finish are exercised",label);
		ExpandCheck(expanded_worst <= 1.0 / 64.0,"expanded attention matches the double-precision attention and value projection within 1/64 of each head's peak",label);
		ExpandCheck(expanded_worst <= 2.0 * absorbed_worst + 1.0 / 256.0,"expanded attention is no less accurate than the absorbed path it replaces",label);
		printf("%s %s: worst error against double precision %.5f (absorbed path %.5f)\n",expanded_worst <= 1.0 / 64.0 ? "ok  " : "FAIL",label,expanded_worst,absorbed_worst);
	}
	if ( rows > 8u )
	{
		std::vector<uint32_t> mixed(sequences);
		uint32_t *mixed_device;
		LmKvShardGatherView view;
		SparkKvShard shard = {degree,0u,1u};
		LmKvShardView pages_view;
		mixed[1] = 1u;
		mixed_device = ExpandDevice(mixed);
		LmKvAccessErrorReset(&error_host);
		EXPAND_CUDA(cudaMemcpy(error_device,&error_host,sizeof(error_host),cudaMemcpyHostToDevice));
		ExpandCheck(LmKvShardViewInitialize<ExpandKv>(&pages_view,pools[0],table_device,pages,2u,pages,error_device,shard) == 0,"two-sequence pages",label);
		ExpandCheck(LmKvShardGatherViewInitialize<ExpandKv>(&view,pages_view.pages,shard,keys_device,plan_device + 1u,plan_device + 2u) == 0,"two-sequence view",label);
		ExpandCheck(LmExpandedGatherPrefillLaunch<ExpandKv,EXPAND_LATENT,EXPAND_ROPE,EXPAND_VALUE>(view,values_device,query_device,heads_per_rank,mixed_device,contexts_device,
			positions_device,scale,value_partials,rows,0) == cudaSuccess,"two-sequence launch",label);
		EXPAND_CUDA(cudaDeviceSynchronize());
		EXPAND_CUDA(cudaMemcpy(&error_host,error_device,sizeof(error_host),cudaMemcpyDeviceToHost));
		ExpandCheck(error_host.error_code != LM_FRAME_ERROR_NONE,"a block whose rows span two sequences reports a KV access error instead of attending",label);
		cudaFree(mixed_device);
	}
	{
		std::vector<uint32_t> short_plan(plan);
		uint32_t *short_device;
		LmKvShardGatherView view;
		short_plan[2] = context / 2u;
		short_device = ExpandDevice(short_plan);
		LmKvAccessErrorReset(&error_host);
		EXPAND_CUDA(cudaMemcpy(error_device,&error_host,sizeof(error_host),cudaMemcpyDeviceToHost));
		ExpandCheck(view_of(0u,0u,most,&view) == 0,"short view",label);
		view.key_context = short_device + 2u;
		view.key_begin = 0u;
		view.key_end = LM_KV_SHARD_KEYS_ALL;
		ExpandCheck(LmExpandedGatherPrefillLaunch<ExpandKv,EXPAND_LATENT,EXPAND_ROPE,EXPAND_VALUE>(view,values_device,query_device,heads_per_rank,sequences_device,contexts_device,
			positions_device,scale,value_partials,rows,0) == cudaSuccess,"short-gather launch",label);
		EXPAND_CUDA(cudaDeviceSynchronize());
		EXPAND_CUDA(cudaMemcpy(&error_host,error_device,sizeof(error_host),cudaMemcpyDeviceToHost));
		ExpandCheck(error_host.error_code != LM_FRAME_ERROR_NONE,"rows that see keys the gather did not carry report a KV access error instead of attending",label);
		cudaFree(short_device);
	}
	{
		LmKvShardGatherView view;
		ExpandCheck(view_of(0u,0u,most,&view) == 0,"unaligned view",label);
		view.key_begin = 8u;
		view.key_end = most > 8u ? most : 9u;
		ExpandCheck(LmExpandedGatherPrefillLaunch<ExpandKv,EXPAND_LATENT,EXPAND_ROPE,EXPAND_VALUE>(view,values_device,query_device,heads_per_rank,sequences_device,contexts_device,
			positions_device,scale,value_partials,rows,0) == cudaErrorInvalidValue,"a key window that does not start on a key tile is refused",label);
		view.key_begin = 0u;
		view.key_end = LM_KV_SHARD_KEYS_ALL;
		ExpandCheck(LmExpandedGatherPrefillLaunch<ExpandKv,EXPAND_LATENT,EXPAND_ROPE,EXPAND_VALUE>(view,values_device,query_device,heads_per_rank,sequences_device,contexts_device,
			positions_device,scale,value_partials,rows,0) == cudaSuccess,"the same view with an aligned window and an even head count launches",label);
		ExpandCheck(LmExpandedGatherPrefillLaunch<ExpandKv,EXPAND_LATENT,EXPAND_ROPE,EXPAND_VALUE>(view,values_device,query_device,heads_per_rank | 1u,sequences_device,contexts_device,
			positions_device,scale,value_partials,rows,0) == cudaErrorInvalidValue,"an odd head count per rank is refused",label);
	}
	for (rank = 0u; rank < degree; rank++)
		cudaFree(pools[rank]);
	cudaFree(table_device); cudaFree(positions_device); cudaFree(sequences_device); cudaFree(contexts_device);
	cudaFree(token_sequence_device); cudaFree(token_position_device); cudaFree(plan_device);
	cudaFree(latent_device); cudaFree(query_device); cudaFree(weight_device); cudaFree(keys_device); cudaFree(values_device);
	cudaFree(latent_partials); cudaFree(value_partials); cudaFree(running_device);
	cudaFree(expanded_device); cudaFree(absorbed_device); cudaFree(latent_out_device); cudaFree(error_device);
}

int main(void)
{
	ExpandCase(17u,640u,16u,6u,0u);
	ExpandCase(100u,3000u,16u,6u,0u);
	ExpandCase(203u,4099u,16u,6u,96u);
	ExpandCase(64u,2048u,8u,12u,64u);
	ExpandCase(512u,9000u,16u,6u,224u);
	if ( expand_failures != 0 )
	{
		printf("expanded gather prefill: %d failure(s)\n",expand_failures);
		return 1;
	}
	printf("expanded gather prefill: PASS\n");
	return 0;
}
