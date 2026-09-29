#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "inference/kernels/route.cuh"

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

template<class T> static T *Upload(const std::vector<T> &host)
{
	T *device;
	CUDA(cudaMalloc(&device,host.size()*sizeof(T)));
	CUDA(cudaMemcpy(device,host.data(),host.size()*sizeof(T),cudaMemcpyHostToDevice));
	return device;
}

template<class T> static std::vector<T> Download(const T *device,size_t count)
{
	std::vector<T> host(count);
	CUDA(cudaMemcpy(host.data(),device,count*sizeof(T),cudaMemcpyDeviceToHost));
	return host;
}

template<uint32_t THREADS, uint32_t EXPERTS>
static void Case(uint32_t rows,uint32_t top_k,uint32_t skew,uint32_t up_dimension,uint32_t down_dimension,uint32_t tile_n,cudaStream_t stream)
{
	const uint32_t packed=rows*top_k, tile_m=LmLaunchGroupedTileM(rows,top_k,EXPERTS), up_tiles=(up_dimension+tile_n-1u)/tile_n, down_tiles=(down_dimension+tile_n-1u)/tile_n;
	std::vector<uint32_t> route(packed),offset(EXPERTS+1u,0u),up(EXPERTS+1u,0u),down(EXPERTS+1u,0u),seen(packed,0u);
	for (uint32_t row=0u; row<rows; row++)
		for (uint32_t k=0u; k<top_k; k++)
		{
			uint32_t expert;
			bool fresh;
			do
			{
				expert=skew != 0u ? Random()%(top_k+skew) : Random()%EXPERTS;
				fresh=true;
				for (uint32_t other=0u; other<k; other++) fresh=fresh && route[row*top_k+other] != expert;
			} while (!fresh);
			route[row*top_k+k]=expert;
		}
	for (uint32_t index=0u; index<packed; index++) offset[route[index]+1u]++;
	for (uint32_t expert=0u; expert<EXPERTS; expert++)
	{
		uint32_t held=offset[expert+1u], tiles=(held+tile_m-1u)/tile_m;
		offset[expert+1u]=offset[expert]+held;
		up[expert+1u]=up[expert]+tiles*up_tiles;
		down[expert+1u]=down[expert]+tiles*down_tiles;
	}
	uint32_t *device_route=Upload(route),*device_offset,*device_packed,*device_source,*device_up,*device_down;
	CUDA(cudaMalloc(&device_offset,(EXPERTS+1u)*4u)); CUDA(cudaMalloc(&device_up,(EXPERTS+1u)*4u)); CUDA(cudaMalloc(&device_down,(EXPERTS+1u)*4u));
	CUDA(cudaMalloc(&device_packed,packed*4u)); CUDA(cudaMalloc(&device_source,packed*4u));
	REQUIRE((LmRouteBuild<THREADS,EXPERTS>(device_route,rows,packed,top_k,device_offset,device_packed,device_source,up_dimension,down_dimension,tile_n,device_up,device_down,stream)) == LM_LAUNCH_OK);
	CUDA(cudaStreamSynchronize(stream));
	std::vector<uint32_t> got_offset=Download(device_offset,EXPERTS+1u),got_up=Download(device_up,EXPERTS+1u),got_down=Download(device_down,EXPERTS+1u),got_packed=Download(device_packed,packed),got_source=Download(device_source,packed);
	REQUIRE(got_offset == offset);
	REQUIRE(got_up == up);
	REQUIRE(got_down == down);
	std::vector<uint32_t> next(offset.begin(),offset.end()-1);
	for (uint32_t index=0u; index<packed; index++)
	{
		uint32_t slot=got_packed[index], expert=route[index];
		REQUIRE(slot >= offset[expert] && slot < offset[expert+1u]);
		REQUIRE(slot == next[expert]++);
		REQUIRE(seen[slot] == 0u);
		seen[slot]=1u;
		REQUIRE(got_source[slot] == index/top_k);
	}
	printf("PASS threads=%u experts=%u rows=%u top_k=%u skew=%u tile_m=%u\n",THREADS,EXPERTS,rows,top_k,skew,tile_m);
	CUDA(cudaFree(device_route)); CUDA(cudaFree(device_offset)); CUDA(cudaFree(device_up)); CUDA(cudaFree(device_down)); CUDA(cudaFree(device_packed)); CUDA(cudaFree(device_source));
}

int main(int argc,char **argv)
{
	cudaStream_t stream;
	if (argc < 2 || strcmp(argv[1],"--run") != 0)
	{
		fprintf(stderr,"usage: test_route_build_scan --run\n");
		return 2;
	}
	CUDA(cudaStreamCreate(&stream));
	Case<256u,288u>(1u,8u,0u,256u,4096u,64u,stream);
	Case<256u,288u>(8u,8u,0u,256u,4096u,64u,stream);
	Case<256u,288u>(8u,8u,3u,256u,4096u,64u,stream);
	Case<256u,288u>(64u,8u,0u,256u,4096u,128u,stream);
	Case<256u,288u>(128u,8u,5u,512u,4096u,64u,stream);
	Case<256u,256u>(17u,8u,0u,256u,2048u,64u,stream);
	Case<128u,384u>(9u,6u,0u,768u,2048u,64u,stream);
	Case<256u,64u>(33u,4u,2u,256u,1024u,32u,stream);
	Case<16u,64u>(40u,4u,2u,256u,1024u,32u,stream);
	printf("test_route_build_scan PASS\n");
	return 0;
}
