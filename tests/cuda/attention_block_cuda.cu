#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "runtime/launch.h"
#include "inference/kernels/attention_block.cuh"
#include "inference/kernels/gqa_shard.cuh"

#define AB_HEADS 6u
#define AB_HEAD 192u
#define AB_VALUE 128u
#define AB_PAGE 64u
#define AB_PITCH (AB_HEADS * (AB_HEAD + AB_VALUE))
#define AB_SCALE 0.0721687836f

#define AB_CUDA(call) do { cudaError_t ab_error = (call); if ( ab_error != cudaSuccess ) { printf("FAIL cuda %s: %s\n",#call,cudaGetErrorString(ab_error)); exit(1); } } while (0)

struct AbLatentKv
{
	static constexpr uint32_t kSlotBytes = 576u * 2u;
	static constexpr uint32_t kPageSlots = AB_PAGE;
	static constexpr uint32_t kPageBytes = kSlotBytes * AB_PAGE;
	static constexpr bool kGrows = true;
	static __host__ __device__ constexpr uint32_t PageOf(uint32_t position) { return position / AB_PAGE; }
	static __host__ __device__ constexpr uint32_t SlotInPage(uint32_t position) { return position % AB_PAGE; }
	static __host__ __device__ constexpr uint64_t PagesForTokens(uint64_t tokens) { return (tokens + AB_PAGE - 1u) / AB_PAGE; }
};

struct AbExpandedKv
{
	static constexpr uint32_t kSlotBytes = AB_PITCH * 2u;
	static constexpr uint32_t kPageSlots = AB_PAGE;
	static constexpr uint32_t kPageBytes = kSlotBytes * AB_PAGE;
	static constexpr bool kGrows = true;
	static __host__ __device__ constexpr uint32_t PageOf(uint32_t position) { return position / AB_PAGE; }
	static __host__ __device__ constexpr uint32_t SlotInPage(uint32_t position) { return position % AB_PAGE; }
	static __host__ __device__ constexpr uint64_t PagesForTokens(uint64_t tokens) { return (tokens + AB_PAGE - 1u) / AB_PAGE; }
};

static int ab_failures;

static void AbCheck(int condition, const char *what)
{
	if ( condition )
		return;
	printf("FAIL %s\n",what);
	ab_failures++;
}

static uint16_t AbBf16(float value)
{
	uint32_t bits;
	memcpy(&bits,&value,4u);
	return((uint16_t)((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u));
}

static float AbUniform(uint64_t z)
{
	z = z * 0x9e3779b97f4a7c15ull + 0x632be59bd9b4e019ull;
	z = (z ^ (z >> 30u)) * 0xbf58476d1ce4e5b9ull;
	z = (z ^ (z >> 27u)) * 0x94d049bb133111ebull;
	return((float)(((z ^ (z >> 31u)) >> 11u) * (1.0 / 9007199254740992.0)) * 2.0f - 1.0f);
}

template<class T>
static T *AbDevice(const std::vector<T> &host)
{
	T *device;
	AB_CUDA(cudaMalloc((void **)&device,host.size() * sizeof(T) + 16u));
	AB_CUDA(cudaMemcpy(device,host.data(),host.size() * sizeof(T),cudaMemcpyHostToDevice));
	return(device);
}

static void AbCase(const char *label, uint32_t rows, uint32_t first_position, uint32_t key_first, uint32_t keys, int exact)
{
	const uint32_t context = key_first + keys, pages = (context + AB_PAGE - 1u) / AB_PAGE, record = LmGqaShardRecordFloats(AB_HEADS,AB_VALUE,1u);
	const SparkKvShard shard = {1u,0u,1u};
	std::vector<uint16_t> query((size_t)rows * AB_HEADS * AB_HEAD), slots((size_t)keys * AB_PITCH);
	std::vector<uint32_t> positions(rows),sequences(rows,0u),table(pages),offset(1,0u),contexts(1,context);
	std::vector<float> want((size_t)rows * record),got(want.size());
	LmKvAccessError error_host,*error_device;
	LmKvShardView local;
	LmKvShardGatherView view;
	uint8_t *pool;
	float *send_want,*send_got;
	uint64_t differ = 0u;
	double numerator = 0.0,denominator = 0.0;
	uint32_t empty = 0u,empty_want = 0u;
	for (uint32_t row = 0u; row < rows; ++row)
	{
		positions[row] = first_position + row;
		for (uint32_t index = 0u; index < AB_HEADS * AB_HEAD; ++index)
			query[(size_t)row * AB_HEADS * AB_HEAD + index] = AbBf16(1.5f * AbUniform(17ull * row + 1000003ull * index + 5ull) + (index % AB_HEAD < 8u ? 1.0f : 0.0f));
	}
	for (uint32_t key = 0u; key < keys; ++key)
		for (uint32_t index = 0u; index < AB_PITCH; ++index)
			slots[(size_t)key * AB_PITCH + index] = AbBf16((index < AB_HEADS * AB_HEAD ? 1.5f : 1.0f) * AbUniform(31ull * key + 7919ull * index + 11ull) +
				(index < AB_HEADS * AB_HEAD && index % AB_HEAD < 8u && key % 53u == 0u ? 3.0f : 0.0f));
	for (uint32_t page = 0u; page < pages; ++page)
		table[page] = page;
	uint16_t *query_device = AbDevice(query),*slots_device = AbDevice(slots);
	uint32_t *positions_device = AbDevice(positions),*sequences_device = AbDevice(sequences),*table_device = AbDevice(table);
	uint32_t *offset_device = AbDevice(offset),*context_device = AbDevice(contexts);
	AB_CUDA(cudaMalloc((void **)&pool,SparkKvShardPoolBytes(shard,AB_PAGE,AbLatentKv::kSlotBytes,pages)));
	AB_CUDA(cudaMalloc((void **)&error_device,sizeof(LmKvAccessError)));
	LmKvAccessErrorReset(&error_host);
	AB_CUDA(cudaMemcpy(error_device,&error_host,sizeof(error_host),cudaMemcpyHostToDevice));
	AB_CUDA(cudaMalloc((void **)&send_want,want.size() * sizeof(float)));
	AB_CUDA(cudaMalloc((void **)&send_got,got.size() * sizeof(float)));
	AB_CUDA(cudaMemset(send_want,0,want.size() * sizeof(float)));
	AB_CUDA(cudaMemset(send_got,0,got.size() * sizeof(float)));
	if ( LmKvShardViewInitialize<AbLatentKv>(&local,pool,table_device,pages,1u,pages,error_device,shard) != 0 ||
		LmKvShardGatherViewInitialize<AbExpandedKv>(&view,local.pages,shard,(const uint8_t *)slots_device,offset_device,context_device) != 0 ||
		LmKvShardGatherViewWindow(&view,key_first,key_first + keys) != 0 )
	{
		printf("FAIL %s: views\n",label);
		exit(1);
	}
	AB_CUDA((LmGqaShardPartialLaunch<AbExpandedKv,LmKvShardGatherView,AB_HEADS,AB_HEAD,AB_VALUE>(view,query_device,AB_HEADS,sequences_device,context_device,positions_device,
		AB_SCALE,send_want,(uint64_t)rows * record,rows,0)));
	AB_CUDA((LmAttentionBlockPartialLaunch<AB_HEAD,AB_VALUE>(query_device,AB_HEADS,positions_device,rows,slots_device,AB_PITCH,key_first,keys,AB_SCALE,send_got,
		(uint64_t)rows * record,0)));
	AB_CUDA(cudaDeviceSynchronize());
	AB_CUDA(cudaMemcpy(want.data(),send_want,want.size() * sizeof(float),cudaMemcpyDeviceToHost));
	AB_CUDA(cudaMemcpy(got.data(),send_got,got.size() * sizeof(float),cudaMemcpyDeviceToHost));
	for (uint32_t row = 0u; row < rows; ++row)
		for (uint32_t head = 0u; head < AB_HEADS; ++head)
		{
			const float *w = &want[(size_t)row * record], *g = &got[(size_t)row * record];
			const float wmax = w[AB_HEADS * AB_VALUE + 2u * head], gmax = g[AB_HEADS * AB_VALUE + 2u * head];
			const float wsum = w[AB_HEADS * AB_VALUE + 2u * head + 1u], gsum = g[AB_HEADS * AB_VALUE + 2u * head + 1u];
			empty += isinf(gmax) && gmax < 0.0f ? 1u : 0u;
			empty_want += isinf(wmax) && wmax < 0.0f ? 1u : 0u;
			differ += memcmp(&wmax,&gmax,4u) != 0 ? 1u : 0u;
			differ += memcmp(&wsum,&gsum,4u) != 0 ? 1u : 0u;
			for (uint32_t d = 0u; d < AB_VALUE; ++d)
			{
				const float a = w[head * AB_VALUE + d], b = g[head * AB_VALUE + d];
				differ += memcmp(&a,&b,4u) != 0 ? 1u : 0u;
				if ( wsum > 0.0f && gsum > 0.0f )
				{
					numerator += ((double)a / wsum - (double)b / gsum) * ((double)a / wsum - (double)b / gsum);
					denominator += ((double)a / wsum) * ((double)a / wsum);
				}
			}
		}
	printf("%s: %llu record floats differ from the paged-view kernel, normalized rel rms %.2e, rows with an empty softmax %u (paged view %u)\n",label,
		(unsigned long long)differ,sqrt(numerator / fmax(denominator,1e-30)),empty / AB_HEADS,empty_want / AB_HEADS);
	AbCheck(empty == empty_want,"rows that see no key of the block record an empty softmax, as the paged-view kernel does");
	if ( exact )
		AbCheck(differ == 0u,"a key block that starts on a 64-key tile gives records bit-identical to the paged-view kernel");
	else
		AbCheck(sqrt(numerator / fmax(denominator,1e-30)) < 5e-3,"a key block off the 64-key grid matches the paged-view kernel within bf16 rounding of the softmax weights");
	cudaFree(query_device); cudaFree(slots_device); cudaFree(positions_device); cudaFree(sequences_device); cudaFree(table_device);
	cudaFree(offset_device); cudaFree(context_device); cudaFree(pool); cudaFree(error_device); cudaFree(send_want); cudaFree(send_got);
}

int main(void)
{
	AbCase("block from key 0, 700 keys, 300 rows at positions 400..699",300u,400u,0u,700u,1);
	AbCase("block from key 256, 300 keys, 200 rows at positions 200..399",200u,200u,256u,300u,1);
	AbCase("block from key 128, 1000 keys, 1129 rows at positions 0..1128",1129u,0u,128u,1000u,1);
	AbCase("block from key 48, 333 keys, 257 rows at positions 30..286",257u,30u,48u,333u,0);
	if ( ab_failures != 0 )
	{
		printf("attention_block_cuda: %d failures\n",ab_failures);
		return(1);
	}
	printf("attention_block_cuda: PASS\n");
	return(0);
}
