#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "runtime/launch.h"
#include "inference/kernels/dtype.cuh"
#include "inference/kernels/project.cuh"
#include "inference/kernels/attn_expanded_prefill.cuh"
#include "inference/kernels/gqa_shard.cuh"

#define MHA_PAGE 64u
#define MHA_LATENT 512u
#define MHA_ROPE 64u
#define MHA_NOPE 128u
#define MHA_VALUE 128u
#define MHA_QK (MHA_NOPE + MHA_ROPE)
#define MHA_WIDTH (MHA_LATENT + MHA_ROPE)

struct MhaLatentKv
{
	static constexpr uint32_t kSlotBytes = MHA_WIDTH * 2u;
	static constexpr uint32_t kPageSlots = MHA_PAGE;
	static constexpr uint32_t kPageBytes = kSlotBytes * MHA_PAGE;
	static constexpr bool kGrows = true;
	static __host__ __device__ constexpr uint32_t PageOf(uint32_t position) { return position / MHA_PAGE; }
	static __host__ __device__ constexpr uint32_t SlotInPage(uint32_t position) { return position % MHA_PAGE; }
	static __host__ __device__ constexpr uint64_t PagesForTokens(uint64_t tokens) { return (tokens + MHA_PAGE - 1u) / MHA_PAGE; }
};

template<uint32_t HEADS>
struct MhaExpandedKv
{
	static constexpr uint32_t kSlotBytes = HEADS * (MHA_QK + MHA_VALUE) * 2u;
	static constexpr uint32_t kPageSlots = MHA_PAGE;
	static constexpr uint32_t kPageBytes = kSlotBytes * MHA_PAGE;
	static constexpr bool kGrows = true;
	static __host__ __device__ constexpr uint32_t PageOf(uint32_t position) { return position / MHA_PAGE; }
	static __host__ __device__ constexpr uint32_t SlotInPage(uint32_t position) { return position % MHA_PAGE; }
	static __host__ __device__ constexpr uint64_t PagesForTokens(uint64_t tokens) { return (tokens + MHA_PAGE - 1u) / MHA_PAGE; }
};

static int mha_failures;

#define MHA_CUDA(call) do { cudaError_t mha_error = (call); if ( mha_error != cudaSuccess ) { printf("FAIL cuda %s: %s\n",#call,cudaGetErrorString(mha_error)); exit(1); } } while (0)

static void MhaCheck(int condition,const char *what,const char *label)
{
	if ( condition )
		return;
	printf("FAIL %s: %s\n",label,what);
	mha_failures++;
}

static uint64_t MhaMix(uint64_t z)
{
	z = z * 0x9e3779b97f4a7c15ull + 0x632be59bd9b4e019ull;
	z = (z ^ (z >> 30u)) * 0xbf58476d1ce4e5b9ull;
	z = (z ^ (z >> 27u)) * 0x94d049bb133111ebull;
	return z ^ (z >> 31u);
}

static uint16_t MhaHostBf16(double value)
{
	float single = (float)value;
	uint32_t bits;
	memcpy(&bits,&single,4u);
	bits += 0x7fffu + ((bits >> 16u) & 1u);
	return (uint16_t)(bits >> 16u);
}

static double MhaHostFloat(uint16_t value)
{
	uint32_t bits = (uint32_t)value << 16u;
	float result;
	memcpy(&result,&bits,4u);
	return result;
}

static std::vector<uint16_t> MhaRandom(uint64_t count,uint64_t salt,float scale)
{
	std::vector<uint16_t> values(count);
	for (uint64_t index = 0u; index < count; index++)
		values[index] = MhaHostBf16((double)((int32_t)((MhaMix(index ^ (salt << 40u)) >> 40u) & 0xfffu) - 2048) / 2048.0 * scale);
	return values;
}

template<class T>
static T *MhaDevice(const std::vector<T> &host)
{
	T *device;
	MHA_CUDA(cudaMalloc((void **)&device,host.size() * sizeof(T) + 16u));
	MHA_CUDA(cudaMemcpy(device,host.data(),host.size() * sizeof(T),cudaMemcpyHostToDevice));
	return device;
}

template<uint32_t HEADS>
static void MhaCase(uint32_t rows,uint32_t context,uint32_t degree)
{
	typedef MhaExpandedKv<HEADS> Expanded;
	const uint32_t pages = (context + MHA_PAGE - 1u) / MHA_PAGE;
	const float scale = 1.0f / sqrtf((float)MHA_QK);
	const uint64_t key_pitch = (uint64_t)HEADS * (MHA_QK + MHA_VALUE),value_offset = (uint64_t)HEADS * MHA_QK;
	SparkKvShard probe = {degree,0u,1u};
	const uint32_t most = SparkKvShardGatherKeys(probe,context);
	const uint64_t latent_stride = ((uint64_t)most * MhaLatentKv::kSlotBytes + 4095u) / 4096u * 4096u;
	const uint64_t expanded_stride = ((uint64_t)most * Expanded::kSlotBytes + 4095u) / 4096u * 4096u;
	const uint64_t record_stride = (uint64_t)rows * LmGqaShardRecordFloats(HEADS,MHA_VALUE,degree);
	const uint32_t width = LmGqaShardWidth(HEADS,MHA_VALUE,degree);
	std::vector<uint32_t> table(pages),positions(rows),sequences(rows,0u),contexts(1,context),token_sequence(context,0u),token_position(context),plan(3u,0u);
	std::vector<uint16_t> latent = MhaRandom((uint64_t)context * MHA_WIDTH,3u,1.0f);
	std::vector<uint16_t> query = MhaRandom((uint64_t)rows * HEADS * MHA_QK,5u,1.0f);
	std::vector<uint16_t> key_weight = MhaRandom((uint64_t)HEADS * MHA_NOPE * MHA_LATENT,7u,0.08f);
	std::vector<uint16_t> value_weight = MhaRandom((uint64_t)HEADS * MHA_VALUE * MHA_LATENT,11u,0.08f);
	std::vector<double> reference((uint64_t)rows * HEADS * MHA_VALUE,0.0);
	std::vector<uint16_t> keys_nope((uint64_t)context * HEADS * MHA_NOPE),values((uint64_t)context * HEADS * MHA_VALUE),merged((uint64_t)rows * width);
	std::vector<uint8_t *> pools(degree);
	uint32_t *table_device,*positions_device,*sequences_device,*contexts_device,*token_sequence_device,*token_position_device,*plan_device;
	uint16_t *latent_device,*query_device,*key_weight_device,*value_weight_device,*merged_device;
	uint8_t *latent_keys_device,*expanded_device;
	float *send_device,*received_device;
	LmKvAccessError *error_device,error_host;
	uint32_t page,row,rank,head,element,index,destination;
	double worst = 0.0;
	char label[160];

	snprintf(label,sizeof(label),"rows%u ctx%u degree%u heads%u",rows,context,degree,HEADS);
	for (page = 0u; page < pages; page++)
		table[page] = (page * 7u + 3u) % pages;
	for (row = 0u; row < rows; row++)
		positions[row] = context - rows + row;
	for (index = 0u; index < context; index++)
		token_position[index] = index;
	plan[2] = context;
	for (index = 0u; index < context; index++)
		for (head = 0u; head < HEADS; head++)
			for (element = 0u; element < MHA_NOPE; element++)
			{
				double key = 0.0,value = 0.0;
				for (uint32_t inner = 0u; inner < MHA_LATENT; inner++)
				{
					const double c = MhaHostFloat(latent[(uint64_t)index * MHA_WIDTH + inner]);
					key += c * MhaHostFloat(key_weight[((uint64_t)head * MHA_NOPE + element) * MHA_LATENT + inner]);
					value += c * MhaHostFloat(value_weight[((uint64_t)head * MHA_VALUE + element) * MHA_LATENT + inner]);
				}
				keys_nope[((uint64_t)index * HEADS + head) * MHA_NOPE + element] = MhaHostBf16(key);
				values[((uint64_t)index * HEADS + head) * MHA_VALUE + element] = MhaHostBf16(value);
			}
	for (row = 0u; row < rows; row++)
		for (head = 0u; head < HEADS; head++)
		{
			std::vector<double> score(positions[row] + 1u);
			double peak = -1e300,total = 0.0;
			const uint16_t *q = &query[((uint64_t)row * HEADS + head) * MHA_QK];
			for (index = 0u; index <= positions[row]; index++)
			{
				double dot = 0.0;
				for (element = 0u; element < MHA_NOPE; element++)
					dot += MhaHostFloat(q[element]) * MhaHostFloat(keys_nope[((uint64_t)index * HEADS + head) * MHA_NOPE + element]);
				for (element = 0u; element < MHA_ROPE; element++)
					dot += MhaHostFloat(q[MHA_NOPE + element]) * MhaHostFloat(latent[(uint64_t)index * MHA_WIDTH + MHA_LATENT + element]);
				score[index] = dot * scale;
				peak = fmax(peak,score[index]);
			}
			for (index = 0u; index <= positions[row]; index++)
			{
				score[index] = exp(score[index] - peak);
				total += score[index];
			}
			for (index = 0u; index <= positions[row]; index++)
				for (element = 0u; element < MHA_VALUE; element++)
					reference[((uint64_t)row * HEADS + head) * MHA_VALUE + element] += score[index] / total *
						MhaHostFloat(values[((uint64_t)index * HEADS + head) * MHA_VALUE + element]);
		}
	table_device = MhaDevice(table);
	positions_device = MhaDevice(positions);
	sequences_device = MhaDevice(sequences);
	contexts_device = MhaDevice(contexts);
	token_sequence_device = MhaDevice(token_sequence);
	token_position_device = MhaDevice(token_position);
	plan_device = MhaDevice(plan);
	latent_device = MhaDevice(latent);
	query_device = MhaDevice(query);
	key_weight_device = MhaDevice(key_weight);
	value_weight_device = MhaDevice(value_weight);
	MHA_CUDA(cudaMalloc((void **)&latent_keys_device,(uint64_t)degree * latent_stride));
	MHA_CUDA(cudaMalloc((void **)&expanded_device,(uint64_t)degree * expanded_stride));
	MHA_CUDA(cudaMalloc((void **)&send_device,(uint64_t)degree * degree * record_stride * sizeof(float)));
	MHA_CUDA(cudaMalloc((void **)&received_device,(uint64_t)degree * record_stride * sizeof(float)));
	MHA_CUDA(cudaMalloc((void **)&merged_device,(uint64_t)rows * width * 2u));
	MHA_CUDA(cudaMalloc((void **)&error_device,sizeof(LmKvAccessError)));
	MHA_CUDA(cudaMemset(expanded_device,0xff,(uint64_t)degree * expanded_stride));
	LmKvAccessErrorReset(&error_host);
	MHA_CUDA(cudaMemcpy(error_device,&error_host,sizeof(error_host),cudaMemcpyHostToDevice));
	for (rank = 0u; rank < degree; rank++)
	{
		SparkKvShard shard = {degree,rank,1u};
		LmKvShardView local;
		LmKvShardGatherView view;
		uint8_t *expanded = expanded_device + (uint64_t)rank * expanded_stride;
		const uint64_t bytes = SparkKvShardPoolBytes(shard,MHA_PAGE,MhaLatentKv::kSlotBytes,pages);
		const uint32_t local_keys = SparkKvShardLocalKeys(shard,context);
		MHA_CUDA(cudaMalloc((void **)&pools[rank],bytes));
		MHA_CUDA(cudaMemset(pools[rank],0,bytes));
		MhaCheck(LmKvShardViewInitialize<MhaLatentKv>(&local,pools[rank],table_device,pages,1u,pages,error_device,shard) == 0,"latent shard view",label);
		LmKvShardStoreKernel<MhaLatentKv,256u><<<context,256>>>(local,latent_device,token_sequence_device,token_position_device,context,MHA_WIDTH);
		MHA_CUDA(cudaGetLastError());
		MhaCheck(LmKvShardGatherPackLaunch<MhaLatentKv,256u>(local,plan_device,plan_device + 1u,plan_device + 2u,1u,most,
			latent_keys_device + (uint64_t)rank * latent_stride,0) == cudaSuccess,"pack the local latent keys",label);
		MhaCheck(LmLatentExpandPitchedLaunch<MHA_LATENT,MHA_WIDTH,MHA_NOPE>(latent_keys_device + (uint64_t)rank * latent_stride,local_keys,key_weight_device,HEADS,
			expanded,key_pitch,MHA_QK,0) == cudaSuccess,"expand the keys",label);
		MhaCheck(LmLatentRopeBroadcastLaunch<MHA_LATENT,MHA_WIDTH>(latent_keys_device + (uint64_t)rank * latent_stride,local_keys,HEADS,expanded,key_pitch,MHA_QK,MHA_NOPE,0) == cudaSuccess,
			"broadcast the rope part",label);
		MhaCheck(LmLatentExpandPitchedLaunch<MHA_LATENT,MHA_WIDTH,MHA_VALUE>(latent_keys_device + (uint64_t)rank * latent_stride,local_keys,value_weight_device,HEADS,
			expanded + value_offset * 2u,key_pitch,MHA_VALUE,0) == cudaSuccess,"expand the values",label);
		MhaCheck(LmKvShardGatherViewInitialize<Expanded>(&view,local.pages,shard,expanded,plan_device + 1u,plan_device + 2u) == 0,"expanded view",label);
		MhaCheck(LmGqaShardPartialLaunch<Expanded,LmKvShardGatherView,HEADS,MHA_QK,MHA_VALUE>(view,query_device,HEADS,sequences_device,contexts_device,positions_device,
			scale,send_device + (uint64_t)rank * degree * record_stride,record_stride,rows,0) == cudaSuccess,"tiled shard partials",label);
	}
	MHA_CUDA(cudaDeviceSynchronize());
	for (destination = 0u; destination < degree; destination++)
	{
		for (rank = 0u; rank < degree; rank++)
			MHA_CUDA(cudaMemcpy(received_device + (uint64_t)rank * record_stride,send_device + ((uint64_t)rank * degree + destination) * record_stride,
				record_stride * sizeof(float),cudaMemcpyDeviceToDevice));
		MhaCheck(LmGqaShardMergeLaunch<MHA_VALUE>(received_device,record_stride,degree,HEADS,destination,merged_device,rows,0) == cudaSuccess,"merge",label);
		MHA_CUDA(cudaDeviceSynchronize());
		MHA_CUDA(cudaMemcpy(merged.data(),merged_device,merged.size() * 2u,cudaMemcpyDeviceToHost));
		for (row = 0u; row < rows; row++)
			for (element = 0u; element < width; element++)
			{
				const uint64_t column = (uint64_t)destination * width + element;
				const uint64_t base = ((uint64_t)row * HEADS + column / MHA_VALUE) * MHA_VALUE;
				double peak = 0.0;
				for (index = 0u; index < MHA_VALUE; index++)
					peak = fmax(peak,fabs(reference[base + index]));
				worst = fmax(worst,fabs(MhaHostFloat(merged[(uint64_t)row * width + element]) - reference[base + column % MHA_VALUE]) / fmax(peak,1e-6));
			}
	}
	MHA_CUDA(cudaMemcpy(&error_host,error_device,sizeof(error_host),cudaMemcpyDeviceToHost));
	MhaCheck(error_host.error_code == LM_FRAME_ERROR_NONE,"no KV access errors",label);
	MhaCheck(worst <= 0.02,"expanded-MHA shard attention matches the double reference within 2% of each head's peak",label);
	printf("%s %s: worst error %.4f of the head peak\n",worst <= 0.02 ? "PASS" : "FAIL",label,worst);
	for (rank = 0u; rank < degree; rank++)
		cudaFree(pools[rank]);
	cudaFree(table_device); cudaFree(positions_device); cudaFree(sequences_device); cudaFree(contexts_device); cudaFree(token_sequence_device);
	cudaFree(token_position_device); cudaFree(plan_device); cudaFree(latent_device); cudaFree(query_device); cudaFree(key_weight_device);
	cudaFree(value_weight_device); cudaFree(latent_keys_device); cudaFree(expanded_device); cudaFree(send_device); cudaFree(received_device);
	cudaFree(merged_device); cudaFree(error_device);
}

int main(void)
{
	MhaCase<16u>(64u,2048u,4u);
	MhaCase<16u>(130u,4093u,16u);
	MhaCase<96u>(32u,1024u,16u);
	if ( mha_failures != 0 )
	{
		printf("FAIL expanded-MHA shard attention: %d check(s)\n",mha_failures);
		return 1;
	}
	printf("PASS expanded-MHA shard attention: latent keys expanded per head into the tiled kernel's slots reproduce the reference MLA attention\n");
	return 0;
}
