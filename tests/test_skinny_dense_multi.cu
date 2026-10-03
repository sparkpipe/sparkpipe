#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "inference/kernels/skinny.cuh"

#define CUDA(call) do { cudaError_t e=(call); if(e!=cudaSuccess) { fprintf(stderr,"FAIL line=%d cuda=%s call=%s\n",__LINE__,cudaGetErrorString(e),#call); exit(1); } } while(0)
#define REQUIRE(test) do { if(!(test)) { fprintf(stderr,"FAIL line=%d test=%s\n",__LINE__,#test); exit(1); } } while(0)

static uint32_t random_state = 20260928u;

static uint32_t Random()
{
	random_state ^= random_state << 13u;
	random_state ^= random_state >> 17u;
	random_state ^= random_state << 5u;
	return random_state;
}

static uint16_t Bf16(float value)
{
	uint32_t bits;
	memcpy(&bits,&value,sizeof(bits));
	return (uint16_t)((bits+0x7fffu+((bits>>16u)&1u))>>16u);
}

static void *UploadRandom(size_t count,float amplitude)
{
	std::vector<uint16_t> host(count);
	void *device;
	for (auto &value : host) value=Bf16(amplitude*(((int32_t)(Random()%2049u)-1024)/1024.0f));
	CUDA(cudaMalloc(&device,count*2u));
	CUDA(cudaMemcpy(device,host.data(),count*2u,cudaMemcpyHostToDevice));
	return device;
}

typedef struct MultiPart
{
	uint32_t output,stride,offset,f32;
}
MultiPart;

static void Case(uint32_t input,uint32_t rows,const MultiPart *parts,uint32_t count,cudaStream_t stream)
{
	LmSkinnyDenseTarget targets[LM_SKINNY_MULTI_MAX];
	void *weight[LM_SKINNY_MULTI_MAX],*single[LM_SKINNY_MULTI_MAX],*fused[LM_SKINNY_MULTI_MAX];
	size_t bytes[LM_SKINNY_MULTI_MAX];
	uint16_t *activation=(uint16_t *)UploadRandom((size_t)rows*input,4.0f);
	uint32_t part;
	for (part=0u; part<count; part++)
	{
		const uint32_t stride=parts[part].stride != 0u ? parts[part].stride : parts[part].output;
		weight[part]=UploadRandom((size_t)parts[part].output*input,0.0625f);
		bytes[part]=(size_t)rows*stride*(parts[part].f32 ? 4u : 2u);
		CUDA(cudaMalloc(&single[part],bytes[part])); CUDA(cudaMalloc(&fused[part],bytes[part]));
		CUDA(cudaMemset(single[part],0x5a,bytes[part])); CUDA(cudaMemset(fused[part],0x5a,bytes[part]));
		REQUIRE(LmSkinnyDense<LmBf16Format>(weight[part],activation,parts[part].f32 ? 0 : (uint16_t *)single[part],parts[part].f32 ? (float *)single[part] : 0,rows,input,parts[part].output,parts[part].stride,parts[part].offset,stream) == LM_LAUNCH_OK);
		targets[part].weight=weight[part];
		targets[part].output_bf16=parts[part].f32 ? 0 : (uint16_t *)fused[part];
		targets[part].output_f32=parts[part].f32 ? (float *)fused[part] : 0;
		targets[part].output_dimension=parts[part].output;
		targets[part].output_row_stride=parts[part].stride;
		targets[part].output_column_offset=parts[part].offset;
	}
	REQUIRE(LmSkinnyDenseMulti<LmBf16Format>(targets,count,activation,rows,input,stream) == LM_LAUNCH_OK);
	CUDA(cudaStreamSynchronize(stream));
	for (part=0u; part<count; part++)
	{
		std::vector<uint8_t> a(bytes[part]),b(bytes[part]);
		CUDA(cudaMemcpy(a.data(),single[part],bytes[part],cudaMemcpyDeviceToHost));
		CUDA(cudaMemcpy(b.data(),fused[part],bytes[part],cudaMemcpyDeviceToHost));
		if (a != b)
		{
			fprintf(stderr,"MISMATCH input=%u rows=%u part=%u output=%u\n",input,rows,part,parts[part].output);
			exit(1);
		}
		CUDA(cudaFree(weight[part])); CUDA(cudaFree(single[part])); CUDA(cudaFree(fused[part]));
	}
	CUDA(cudaFree(activation));
	printf("PASS input=%u rows=%u parts=%u\n",input,rows,count);
}

int main(int argc,char **argv)
{
	static const MultiPart kda[2]={{1540u,0u,0u,0u},{256u,0u,0u,0u}};
	static const MultiPart mixed[4]={{1536u,0u,0u,0u},{128u,0u,0u,0u},{32u,40u,8u,0u},{288u,0u,0u,1u}};
	static const MultiPart narrow[3]={{512u,0u,0u,0u},{37u,64u,3u,0u},{4097u,0u,0u,1u}};
	static const MultiPart tiny[2]={{37u,0u,0u,1u},{5u,0u,0u,0u}};
	static const MultiPart mid[3]={{1024u,0u,0u,0u},{7u,0u,0u,0u},{4096u,0u,0u,0u}};
	cudaStream_t stream;
	uint32_t rows;
	LmSkinnyDenseTarget bad[1];
	if (argc < 2 || strcmp(argv[1],"--run") != 0)
	{
		fprintf(stderr,"usage: test_skinny_dense_multi --run\n");
		return 2;
	}
	CUDA(cudaStreamCreate(&stream));
	for (rows=1u; rows<=LM_SKINNY_ROWS_WIDE; rows++)
	{
		Case(4096u,rows,kda,2u,stream);
		Case(4096u,rows,mixed,4u,stream);
		Case(128u,rows,narrow,3u,stream);
		Case(96u,rows,tiny,2u,stream);
		Case(1536u,rows,mid,3u,stream);
	}
	memset(bad,0,sizeof(bad));
	REQUIRE(LmSkinnyDenseMulti<LmBf16Format>(bad,0u,(const uint16_t *)16,1u,4096u,stream) == LM_LAUNCH_ERR_SHAPE);
	REQUIRE(LmSkinnyDenseMulti<LmBf16Format>(bad,LM_SKINNY_MULTI_MAX + 1u,(const uint16_t *)16,1u,4096u,stream) == LM_LAUNCH_ERR_SHAPE);
	bad[0].weight=(const void *)16; bad[0].output_bf16=(uint16_t *)16; bad[0].output_dimension=16u;
	REQUIRE(LmSkinnyDenseMulti<LmBf16Format>(bad,1u,(const uint16_t *)16,LM_SKINNY_ROWS_WIDE + 1u,4096u,stream) == LM_LAUNCH_ERR_SHAPE);
	printf("test_skinny_dense_multi PASS\n");
	return 0;
}
