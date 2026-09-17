#include <cuda_runtime.h>

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define SPARK_DSV4_TP4_BITWISE_COUNT 262144u
#define SPARK_DSV4_TP4_BITWISE_RANK_COUNT 4u

extern "C" cudaError_t SparkDsv4LaunchAccumAdd(
	cudaStream_t stream,void *destination_bf16,const void *source_bf16,
	uint32_t row_count,uint32_t width);
extern "C" cudaError_t SparkDsv4LaunchAccumAddRelay(
	cudaStream_t stream,void *destination_bf16,const void *source_bf16,
	void *relay_bf16,uint32_t row_count,uint32_t width);
extern "C" cudaError_t SparkDsv4LaunchAccumAddTp4Tree(
	cudaStream_t stream,void *destination_bf16,
	const void *const rank_devices[SPARK_DSV4_TP4_BITWISE_RANK_COUNT],
	uint32_t tp_rank,uint32_t row_count,uint32_t width);

static uint16_t SparkDsv4Tp4BitwiseRanks[SPARK_DSV4_TP4_BITWISE_RANK_COUNT]
	[SPARK_DSV4_TP4_BITWISE_COUNT];
static uint16_t SparkDsv4Tp4BitwiseReference[SPARK_DSV4_TP4_BITWISE_COUNT];
static uint16_t SparkDsv4Tp4BitwiseFused[SPARK_DSV4_TP4_BITWISE_COUNT];

static const uint16_t SparkDsv4Tp4BitwiseAdversarial[][4] =
{
	{0x0000u,0x8000u,0x0000u,0x8000u},
	{0x3f80u,0xbf80u,0x3f80u,0xbf80u},
	{0x7f7fu,0xff7fu,0x0080u,0x8080u},
	{0x0001u,0x8001u,0x007fu,0x807fu},
	{0x3f80u,0x3380u,0xbf80u,0xb380u},
	{0x4000u,0xbf80u,0xc000u,0x3f80u},
	{0x7e00u,0xfe00u,0x3f00u,0xbf00u},
	{0x0080u,0x0080u,0x8080u,0x8080u},
	{0x3f00u,0x3f00u,0xbf00u,0xbf00u},
	{0x3f7fu,0x3380u,0xbf7fu,0xb380u},
	{0x0001u,0x0001u,0x8001u,0x8001u},
	{0x7f7fu,0x0080u,0xff7fu,0x8080u},
	{0x3e80u,0x3e80u,0xbe80u,0xbe80u},
	{0x4b00u,0xcb00u,0x3f80u,0xbf80u},
	{0x0100u,0x8100u,0x0001u,0x8001u},
	{0x7f00u,0xff00u,0x7e80u,0xfe80u}
};

static uint32_t SparkDsv4Tp4BitwiseRandom(uint32_t *state)
{
	uint32_t value;
	value = *state;
	value ^= value << 13u;
	value ^= value >> 17u;
	value ^= value << 5u;
	*state = value;
	return(value);
}

static uint16_t SparkDsv4Tp4BitwiseFinite(uint32_t *state)
{
	uint16_t value;
	value = (uint16_t)SparkDsv4Tp4BitwiseRandom(state);
	if ( (value & 0x7f80u) == 0x7f80u )
		value &= 0xff7fu;
	return(value);
}

static uint64_t SparkDsv4Tp4BitwiseHash(const uint16_t *values)
{
	uint64_t hash;
	uint32_t index;
	hash = UINT64_C(1469598103934665603);
	for (index=0u; index<SPARK_DSV4_TP4_BITWISE_COUNT; index++)
	{
		hash ^= values[index] & 0xffu;
		hash *= UINT64_C(1099511628211);
		hash ^= values[index] >> 8u;
		hash *= UINT64_C(1099511628211);
	}
	return(hash);
}

static void SparkDsv4Tp4BitwiseInitialize(uint32_t seed)
{
	uint32_t adversarial_count,index,rank;
	adversarial_count = (uint32_t)(sizeof(SparkDsv4Tp4BitwiseAdversarial) /
		sizeof(SparkDsv4Tp4BitwiseAdversarial[0]));
	for (index=0u; index<SPARK_DSV4_TP4_BITWISE_COUNT; index++)
		for (rank=0u; rank<SPARK_DSV4_TP4_BITWISE_RANK_COUNT; rank++)
			SparkDsv4Tp4BitwiseRanks[rank][index] = index < adversarial_count ?
				SparkDsv4Tp4BitwiseAdversarial[index][rank] :
				SparkDsv4Tp4BitwiseFinite(&seed);
}

static uint32_t SparkDsv4Tp4BitwiseCompare(
	uint32_t rank,uint32_t *first_mismatch)
{
	uint32_t index,mismatch_count;
	mismatch_count = 0u;
	*first_mismatch = UINT32_MAX;
	for (index=0u; index<SPARK_DSV4_TP4_BITWISE_COUNT; index++)
		if ( SparkDsv4Tp4BitwiseReference[index] !=
			SparkDsv4Tp4BitwiseFused[index] )
		{
			if ( mismatch_count == 0u )
				*first_mismatch = index;
			mismatch_count++;
		}
	if ( mismatch_count != 0u )
		fprintf(stderr,
			"rank=%u first_mismatch=%u reference=0x%04x fused=0x%04x\n",
			rank,*first_mismatch,SparkDsv4Tp4BitwiseReference[*first_mismatch],
			SparkDsv4Tp4BitwiseFused[*first_mismatch]);
	return(mismatch_count);
}

int main(void)
{
	const void *fused_rank_devices[SPARK_DSV4_TP4_BITWISE_RANK_COUNT];
	void *device_fused,*device_ranks[SPARK_DSV4_TP4_BITWISE_RANK_COUNT];
	void *device_reference,*device_relay,*device_remote;
	uint64_t fused_hashes[4],reference_hashes[4];
	uint32_t first_mismatch,mismatches[4],rank,source_rank;
	cudaStream_t stream;
	cudaError_t error;
	SparkDsv4Tp4BitwiseInitialize(UINT32_C(0x6d5a56e9));
	error = cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking);
	for (rank=0u; error == cudaSuccess && rank<4u; rank++)
	{
		error = cudaMalloc(&device_ranks[rank],
			sizeof(SparkDsv4Tp4BitwiseRanks[rank]));
		if ( error == cudaSuccess )
			error = cudaMemcpyAsync(device_ranks[rank],
				SparkDsv4Tp4BitwiseRanks[rank],
				sizeof(SparkDsv4Tp4BitwiseRanks[rank]),cudaMemcpyHostToDevice,stream);
	}
	if ( error == cudaSuccess )
		error = cudaMalloc(&device_reference,
			sizeof(SparkDsv4Tp4BitwiseReference));
	if ( error == cudaSuccess )
		error = cudaMalloc(&device_remote,sizeof(SparkDsv4Tp4BitwiseReference));
	if ( error == cudaSuccess )
		error = cudaMalloc(&device_relay,sizeof(SparkDsv4Tp4BitwiseReference));
	if ( error == cudaSuccess )
		error = cudaMalloc(&device_fused,sizeof(SparkDsv4Tp4BitwiseFused));
	for (rank=0u; error == cudaSuccess && rank<4u; rank++)
	{
		error = cudaMemcpyAsync(device_reference,device_ranks[rank],
			sizeof(SparkDsv4Tp4BitwiseReference),cudaMemcpyDeviceToDevice,stream);
		if ( error == cudaSuccess )
			error = SparkDsv4LaunchAccumAddRelay(stream,device_reference,
				device_ranks[rank ^ 1u],device_relay,1u,
				SPARK_DSV4_TP4_BITWISE_COUNT);
		source_rank = rank ^ 2u;
		if ( error == cudaSuccess )
			error = cudaMemcpyAsync(device_remote,device_ranks[source_rank],
				sizeof(SparkDsv4Tp4BitwiseReference),cudaMemcpyDeviceToDevice,stream);
		if ( error == cudaSuccess )
			error = SparkDsv4LaunchAccumAddRelay(stream,device_remote,
				device_ranks[rank ^ 3u],device_relay,1u,
				SPARK_DSV4_TP4_BITWISE_COUNT);
		if ( error == cudaSuccess )
			error = SparkDsv4LaunchAccumAdd(stream,device_reference,device_remote,
				1u,SPARK_DSV4_TP4_BITWISE_COUNT);
		if ( error == cudaSuccess )
			error = cudaMemcpyAsync(device_fused,device_ranks[rank],
				sizeof(SparkDsv4Tp4BitwiseFused),cudaMemcpyDeviceToDevice,stream);
		for (source_rank=0u; source_rank<4u; source_rank++)
			fused_rank_devices[source_rank] = source_rank == rank ?
				device_fused : device_ranks[source_rank];
		if ( error == cudaSuccess )
			error = SparkDsv4LaunchAccumAddTp4Tree(stream,device_fused,
				fused_rank_devices,rank,1u,SPARK_DSV4_TP4_BITWISE_COUNT);
		if ( error == cudaSuccess )
			error = cudaMemcpyAsync(SparkDsv4Tp4BitwiseReference,device_reference,
				sizeof(SparkDsv4Tp4BitwiseReference),cudaMemcpyDeviceToHost,stream);
		if ( error == cudaSuccess )
			error = cudaMemcpyAsync(SparkDsv4Tp4BitwiseFused,device_fused,
				sizeof(SparkDsv4Tp4BitwiseFused),cudaMemcpyDeviceToHost,stream);
		if ( error == cudaSuccess )
			error = cudaStreamSynchronize(stream);
		if ( error == cudaSuccess )
		{
			mismatches[rank] = SparkDsv4Tp4BitwiseCompare(rank,&first_mismatch);
			reference_hashes[rank] =
				SparkDsv4Tp4BitwiseHash(SparkDsv4Tp4BitwiseReference);
			fused_hashes[rank] =
				SparkDsv4Tp4BitwiseHash(SparkDsv4Tp4BitwiseFused);
		}
	}
	if ( error != cudaSuccess )
	{
		fprintf(stderr,"cuda_error=%s\n",cudaGetErrorString(error));
		return(2);
	}
	printf("{\"seed\":\"0x6d5a56e9\",\"elements_per_rank\":%u,"
		"\"finite_input_values\":%u,\"adversarial_tuples\":%u,"
		"\"rank_mismatches\":[%u,%u,%u,%u],"
		"\"reference_hashes\":[\"%016llx\",\"%016llx\",\"%016llx\",\"%016llx\"],"
		"\"fused_hashes\":[\"%016llx\",\"%016llx\",\"%016llx\",\"%016llx\"],"
		"\"status\":\"%s\"}\n",SPARK_DSV4_TP4_BITWISE_COUNT,
		SPARK_DSV4_TP4_BITWISE_COUNT * 4u,
		(uint32_t)(sizeof(SparkDsv4Tp4BitwiseAdversarial) /
			sizeof(SparkDsv4Tp4BitwiseAdversarial[0])),
		mismatches[0],mismatches[1],mismatches[2],mismatches[3],
		(unsigned long long)reference_hashes[0],
		(unsigned long long)reference_hashes[1],
		(unsigned long long)reference_hashes[2],
		(unsigned long long)reference_hashes[3],
		(unsigned long long)fused_hashes[0],
		(unsigned long long)fused_hashes[1],
		(unsigned long long)fused_hashes[2],
		(unsigned long long)fused_hashes[3],
		(mismatches[0] | mismatches[1] | mismatches[2] | mismatches[3]) == 0u ?
			"pass" : "fail");
	return((mismatches[0] | mismatches[1] | mismatches[2] | mismatches[3]) ==
		0u ? 0 : 1);
}
