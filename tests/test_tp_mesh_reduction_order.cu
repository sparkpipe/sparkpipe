#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include "sparkpipe/spark_tp_mesh_kernels.cuh"

#define CUDA(call) do { cudaError_t e=(call); if(e!=cudaSuccess) { fprintf(stderr,"FAIL line=%d cuda=%s call=%s\n",__LINE__,cudaGetErrorString(e),#call); exit(1); } } while(0)
#define REQUIRE(test) do { if(!(test)) { fprintf(stderr,"FAIL line=%d test=%s\n",__LINE__,#test); exit(1); } } while(0)

static uint32_t random_state = 20260929u;

static uint32_t Random()
{
	random_state ^= random_state << 13u;
	random_state ^= random_state >> 17u;
	random_state ^= random_state << 5u;
	return random_state;
}

static uint16_t RandomBf16()
{
	float value = ldexpf((float)(Random() % 2000001u) / 1000000.0f - 1.0f,(int)(Random() % 41u) - 20);
	uint32_t bits;
	memcpy(&bits,&value,sizeof(bits));
	return (uint16_t)(bits >> 16u);
}

static float Bf16(uint16_t half)
{
	uint32_t bits = (uint32_t)half << 16u;
	float value;
	memcpy(&value,&bits,sizeof(value));
	return value;
}

static uint16_t Truncate(float value)
{
	uint32_t bits;
	memcpy(&bits,&value,sizeof(bits));
	return (uint16_t)(bits >> 16u);
}

struct Mesh
{
	uint8_t *band;
	SparkTpMeshRoundControl *control;
	uint64_t slot_bytes;
	uint32_t degree;
};

static Mesh MeshCreate(uint32_t degree,uint64_t elements)
{
	Mesh mesh;
	SparkTpMeshRoundControl control;
	mesh.degree = degree;
	mesh.slot_bytes = (elements * sizeof(uint16_t) + 16u + 255u) & ~(uint64_t)255u;
	CUDA(cudaMalloc(&mesh.band,(size_t)degree * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK * mesh.slot_bytes));
	CUDA(cudaMemset(mesh.band,0,(size_t)degree * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK * mesh.slot_bytes));
	CUDA(cudaMalloc(&mesh.control,sizeof(SparkTpMeshRoundControl)));
	memset(&control,0,sizeof(control));
	control.round_seq = 1u;
	CUDA(cudaMemcpy(mesh.control,&control,sizeof(control),cudaMemcpyHostToDevice));
	return mesh;
}

static void MeshFree(Mesh *mesh)
{
	CUDA(cudaFree(mesh->band));
	CUDA(cudaFree(mesh->control));
}

static void MeshPlace(const Mesh *mesh,uint32_t peer,uint64_t offset,const uint16_t *values,uint64_t count)
{
	CUDA(cudaMemcpy(mesh->band + (uint64_t)peer * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK * mesh->slot_bytes + offset * sizeof(uint16_t),values,count * sizeof(uint16_t),cudaMemcpyHostToDevice));
}

static void MeshCombine(const Mesh *mesh,uint16_t *output,uint64_t local_elements,uint64_t count,uint32_t operation,uint64_t first,uint64_t span)
{
	uint32_t blocks = (uint32_t)((span + SPARK_TP_MESH_THREADS - 1u) / SPARK_TP_MESH_THREADS);
	SparkTpMeshHardwareDirectKernel<<<blocks != 0u ? blocks : 1u,SPARK_TP_MESH_THREADS>>>(mesh->band,mesh->slot_bytes,SPARK_WEIGHTD_MESH_SLOTS_PER_RANK,mesh->control,mesh->degree,output,local_elements,0u,count,operation,first,span,0u);
	CUDA(cudaGetLastError());
	CUDA(cudaDeviceSynchronize());
}

static std::vector<uint16_t> Download(const uint16_t *device,uint64_t count)
{
	std::vector<uint16_t> host(count);
	CUDA(cudaMemcpy(host.data(),device,count * sizeof(uint16_t),cudaMemcpyDeviceToHost));
	return host;
}

static std::vector<uint16_t> Direct(uint32_t degree,const std::vector<std::vector<uint16_t> > &contribution,uint64_t count)
{
	Mesh mesh = MeshCreate(degree,count);
	uint16_t *output;
	CUDA(cudaMalloc(&output,count * sizeof(uint16_t)));
	for (uint32_t peer = 0u; peer < degree; peer++)
		MeshPlace(&mesh,peer,0u,contribution[peer].data(),count);
	MeshCombine(&mesh,output,count,count,1u,0u,count);
	std::vector<uint16_t> result = Download(output,count);
	CUDA(cudaFree(output));
	MeshFree(&mesh);
	return result;
}

static std::vector<uint16_t> ReduceScatterAllGather(uint32_t degree,const std::vector<std::vector<uint16_t> > &contribution,uint64_t count)
{
	const uint64_t slice = SparkTpMeshRsagSlice(count,degree);
	Mesh scatter = MeshCreate(degree,count),gather = MeshCreate(degree,count);
	std::vector<std::vector<uint16_t> > owned(degree);
	uint16_t *reduced,*output;
	CUDA(cudaMalloc(&reduced,count * sizeof(uint16_t)));
	CUDA(cudaMalloc(&output,count * sizeof(uint16_t)));
	for (uint32_t rank = 0u; rank < degree; rank++)
	{
		const uint64_t first = (uint64_t)rank * slice < count ? (uint64_t)rank * slice : count;
		const uint64_t span = count - first < slice ? count - first : slice;
		CUDA(cudaMemset(scatter.band,0xff,(size_t)degree * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK * scatter.slot_bytes));
		for (uint32_t peer = 0u; peer < degree; peer++)
			if (span != 0u)
				MeshPlace(&scatter,peer,first,contribution[peer].data() + first,span);
		CUDA(cudaMemset(reduced,0,count * sizeof(uint16_t)));
		if (span != 0u)
			MeshCombine(&scatter,reduced,count,count,1u,first,span);
		std::vector<uint16_t> mine = Download(reduced,count);
		owned[rank].assign(mine.begin() + first,mine.begin() + first + span);
		if (span != 0u)
			MeshPlace(&gather,rank,first,owned[rank].data(),span);
	}
	MeshCombine(&gather,output,slice,count,SPARK_TP_MESH_OPERATION_SLICE_GATHER,0u,count);
	std::vector<uint16_t> result = Download(output,count);
	CUDA(cudaFree(reduced));
	CUDA(cudaFree(output));
	MeshFree(&scatter);
	MeshFree(&gather);
	return result;
}

static std::vector<uint16_t> RankOrder(uint32_t degree,const std::vector<std::vector<uint16_t> > &contribution,uint64_t count,bool reversed)
{
	std::vector<uint16_t> result(count);
	for (uint64_t element = 0u; element < count; element++)
	{
		float sum = 0.0f;
		for (uint32_t step = 0u; step < degree; step++)
			sum += Bf16(contribution[reversed ? degree - 1u - step : step][element]);
		result[element] = Truncate(sum);
	}
	return result;
}

static void Case(uint32_t degree,uint32_t hidden,uint32_t rows)
{
	const uint64_t count = (uint64_t)hidden * rows;
	std::vector<std::vector<uint16_t> > contribution(degree,std::vector<uint16_t>(count));
	for (uint32_t peer = 0u; peer < degree; peer++)
		for (uint64_t element = 0u; element < count; element++)
			contribution[peer][element] = RandomBf16();
	for (uint32_t peer = 0u; peer < degree; peer++)
		contribution[peer][count / 2u] = peer == 0u ? 0x4e80u : peer == 1u ? 0xce80u : peer == 2u ? 0x3f80u : 0u;
	const std::vector<uint16_t> want = RankOrder(degree,contribution,count,false);
	const std::vector<uint16_t> reversed = RankOrder(degree,contribution,count,true);
	const std::vector<uint16_t> direct = Direct(degree,contribution,count);
	const std::vector<uint16_t> rsag = ReduceScatterAllGather(degree,contribution,count);
	uint64_t order_sensitive = 0u;
	for (uint64_t element = 0u; element < count; element++)
		order_sensitive += want[element] != reversed[element] ? 1u : 0u;
	REQUIRE(order_sensitive != 0u);
	REQUIRE(direct == want);
	REQUIRE(rsag == want);
	for (uint32_t row = 0u; row < rows; row++)
	{
		std::vector<std::vector<uint16_t> > alone(degree);
		for (uint32_t peer = 0u; peer < degree; peer++)
			alone[peer].assign(contribution[peer].begin() + (uint64_t)row * hidden,contribution[peer].begin() + (uint64_t)(row + 1u) * hidden);
		const std::vector<uint16_t> single = Direct(degree,alone,hidden);
		REQUIRE(memcmp(single.data(),direct.data() + (uint64_t)row * hidden,hidden * sizeof(uint16_t)) == 0);
	}
	printf("PASS degree=%u hidden=%u rows=%u elements=%llu direct=rank-order rs+ag=rank-order each-row-alone=equal reversed-order-differs=%llu\n",
		degree,hidden,rows,(unsigned long long)count,(unsigned long long)order_sensitive);
}

int main(int argc,char **argv)
{
	if (argc < 2 || strcmp(argv[1],"--run") != 0)
	{
		fprintf(stderr,"usage: test_tp_mesh_reduction_order --run\n");
		return 2;
	}
	Case(16u,4096u,1u);
	Case(16u,4096u,12u);
	Case(16u,4096u,17u);
	Case(8u,4096u,13u);
	Case(4u,2048u,24u);
	Case(16u,1000u,49u);
	printf("test_tp_mesh_reduction_order PASS\n");
	return 0;
}
