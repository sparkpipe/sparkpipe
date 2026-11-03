#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "inference/kernels/moe_experts.cuh"
#include "inference/kernels/route.cuh"

#define CUDA(call) do { cudaError_t e=(call); if(e!=cudaSuccess) { fprintf(stderr,"FAIL line=%d cuda=%s call=%s\n",__LINE__,cudaGetErrorString(e),#call); exit(1); } } while(0)
#define REQUIRE(test) do { if(!(test)) { fprintf(stderr,"FAIL line=%d test=%s\n",__LINE__,#test); exit(1); } } while(0)

#define ET_LOCAL 8u
#define ET_TOP_K 4u
#define ET_HIDDEN 512u
#define ET_INTER 80u
#define ET_PADDED 128u
#define ET_THREADS 256u
#define ET_MAX_ROWS 256u

static uint32_t random_state = 20261026u;
static uint64_t compared = 0u, failures = 0u, padded_differences = 0u, tail_sensitive = 0u, tail_outputs = 0u, skinny_cases = 0u, gemm_cases = 0u;

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

static double Bf16Value(uint16_t value)
{
	uint32_t bits=(uint32_t)value<<16u;
	float result;
	memcpy(&result,&bits,sizeof(result));
	return result;
}

static uint16_t *Device(const std::vector<uint16_t> &host)
{
	uint16_t *device;
	CUDA(cudaMalloc(&device,host.size()*2u));
	CUDA(cudaMemcpy(device,host.data(),host.size()*2u,cudaMemcpyHostToDevice));
	return device;
}

static std::vector<uint16_t> Download(const uint16_t *device, size_t count)
{
	std::vector<uint16_t> host(count);
	CUDA(cudaMemcpy(host.data(),device,count*2u,cudaMemcpyDeviceToHost));
	return host;
}

static void Fill(std::vector<uint16_t> &host, size_t count, float amplitude)
{
	host.resize(count);
	for (size_t i=0u; i<count; i++) host[i]=Bf16(amplitude*(((int32_t)(Random()%2049u)-1024)/1024.0f));
}

static int32_t Up(const LmLocalExpertRoutes &routes, const uint16_t *weight, const uint16_t *activation, uint16_t *output, uint32_t width, bool gemm, cudaStream_t stream)
{
	if ( !gemm )
		return LmSkinnyGroupedExperts<LmBf16Format>(weight,LmScaleTensorNone(),activation,output,routes.group_row_offset,routes.route_source_token,
			routes.local_experts,routes.routes,0u,ET_HIDDEN,width,stream);
	LmGemmArguments args;
	memset(&args,0,sizeof(args));
	args.scale_a = LmScaleTensorNone();
	args.scale_b = LmScaleTensorNone();
	args.prefix_built = 1u;
	args.group_row_offset = routes.group_row_offset;
	args.group_tile_prefix = routes.tile_prefix_up;
	args.source_row_map = routes.route_source_token;
	args.source_row_count = routes.rows;
	args.output_bf16 = output;
	return LmGemmWeightOnlyIndirectLaunch<LmBf16Format,LM_MOE_GEMM_TILE_N,LM_MOE_GEMM_STAGES,LM_MOE_GEMM_WARPS>(&args,activation,weight,
		routes.routes,routes.rows,routes.top_k,routes.local_experts,ET_HIDDEN,width,routes.multiprocessors,stream);
}

static int32_t Down(const LmLocalExpertRoutes &routes, const uint16_t *weight, const uint16_t *activation, uint16_t *output, uint32_t depth, bool gemm, cudaStream_t stream)
{
	if ( !gemm )
		return LmSkinnyGroupedExperts<LmBf16Format>(weight,LmScaleTensorNone(),activation,output,routes.group_row_offset,routes.route_source_token,
			routes.local_experts,routes.routes,1u,depth,ET_HIDDEN,stream);
	LmGemmArguments args;
	memset(&args,0,sizeof(args));
	args.scale_a = LmScaleTensorNone();
	args.scale_b = LmScaleTensorNone();
	args.prefix_built = 1u;
	args.group_row_offset = routes.group_row_offset;
	args.group_tile_prefix = routes.tile_prefix_down;
	args.output_bf16 = output;
	return LmGemmWeightOnlyLaunch<LmBf16Format,LM_MOE_GEMM_TILE_N,LM_MOE_GEMM_STAGES,LM_MOE_GEMM_WARPS>(&args,activation,weight,routes.routes,
		routes.rows,routes.top_k,routes.local_experts,depth,ET_HIDDEN,routes.multiprocessors,true,stream);
}

static void Compare(const char *path, uint32_t rows, uint32_t local_rows, uint32_t width, uint32_t stride, const std::vector<uint16_t> &got,
	const std::vector<double> &expect, const std::vector<double> &bound)
{
	uint32_t printed=0u;
	for (uint32_t p=0u; p<local_rows; p++)
		for (uint32_t n=0u; n<width; n++)
		{
			const size_t at=(size_t)p*width+n;
			const double value=Bf16Value(got[(size_t)p*stride+n]), tolerance=fabs(expect[at])/128.0+1e-5*bound[at]+1e-30;
			compared++;
			if (!(fabs(value-expect[at])<=tolerance))
			{
				failures++;
				if (printed++<3u) fprintf(stderr,"MISMATCH %s rows=%u packed=%u col=%u got=%.6g ref=%.6g tol=%.3g\n",path,rows,p,n,value,expect[at],tolerance);
			}
		}
}

static void Identical(const char *path, uint32_t rows, uint32_t local_rows, uint32_t width, const std::vector<uint16_t> &tail, uint32_t tail_stride,
	const std::vector<uint16_t> &padded, uint32_t padded_stride)
{
	uint32_t printed=0u;
	for (uint32_t p=0u; p<local_rows; p++)
		for (uint32_t n=0u; n<width; n++)
			if (tail[(size_t)p*tail_stride+n]!=padded[(size_t)p*padded_stride+n])
			{
				padded_differences++;
				if (printed++<3u) fprintf(stderr,"PADDED %s rows=%u packed=%u col=%u tail=%04x padded=%04x\n",path,rows,p,n,tail[(size_t)p*tail_stride+n],padded[(size_t)p*padded_stride+n]);
			}
}

static void Case(uint32_t rows, const std::vector<uint16_t> &w_up, const std::vector<uint16_t> &w_down, const std::vector<uint16_t> &normed, cudaStream_t stream)
{
	const uint32_t routes=rows*ET_TOP_K;
	std::vector<uint32_t> local((size_t)routes), offset(ET_LOCAL+2u), source(routes);
	std::vector<uint16_t> up_padded((size_t)ET_LOCAL*ET_PADDED*ET_HIDDEN,0u), down_padded((size_t)ET_LOCAL*ET_HIDDEN*ET_PADDED,0u), act_host, act_padded;
	uint32_t *d_route, *d_offset, *d_packed, *d_source, *d_tile_up, *d_tile_down, e, p, n, k;
	int device_multiprocessors;
	for (uint32_t t=0u; t<rows; t++)
		for (uint32_t j=0u; j<ET_TOP_K; j++)
			local[(size_t)t*ET_TOP_K+j]=Random()%(ET_LOCAL+2u)<ET_LOCAL ? (t+j*3u+Random()%2u)%ET_LOCAL : ET_LOCAL;
	CUDA(cudaDeviceGetAttribute(&device_multiprocessors,cudaDevAttrMultiProcessorCount,0));
	CUDA(cudaMalloc(&d_route,routes*4u)); CUDA(cudaMalloc(&d_offset,(ET_LOCAL+2u)*4u)); CUDA(cudaMalloc(&d_packed,routes*4u)); CUDA(cudaMalloc(&d_source,routes*4u));
	CUDA(cudaMalloc(&d_tile_up,(ET_LOCAL+2u)*4u)); CUDA(cudaMalloc(&d_tile_down,(ET_LOCAL+2u)*4u));
	CUDA(cudaMemcpy(d_route,local.data(),routes*4u,cudaMemcpyHostToDevice));
	REQUIRE((LmRouteBuild<ET_THREADS,ET_LOCAL+1u>(d_route,rows,routes,ET_TOP_K,d_offset,d_packed,d_source,ET_INTER,ET_HIDDEN,LM_MOE_GEMM_TILE_N,d_tile_up,d_tile_down,stream))==LM_LAUNCH_OK);
	CUDA(cudaStreamSynchronize(stream));
	CUDA(cudaMemcpy(offset.data(),d_offset,(ET_LOCAL+2u)*4u,cudaMemcpyDeviceToHost));
	CUDA(cudaMemcpy(source.data(),d_source,routes*4u,cudaMemcpyDeviceToHost));
	const uint32_t local_rows=offset[ET_LOCAL];
	Fill(act_host,(size_t)routes*ET_INTER,4.0f);
	act_padded.assign((size_t)routes*ET_PADDED,0u);
	for (p=0u; p<routes; p++) memcpy(&act_padded[(size_t)p*ET_PADDED],&act_host[(size_t)p*ET_INTER],ET_INTER*2u);
	for (e=0u; e<ET_LOCAL; e++)
	{
		memcpy(&up_padded[(size_t)e*ET_PADDED*ET_HIDDEN],&w_up[(size_t)e*ET_INTER*ET_HIDDEN],(size_t)ET_INTER*ET_HIDDEN*2u);
		for (n=0u; n<ET_HIDDEN; n++) memcpy(&down_padded[((size_t)e*ET_HIDDEN+n)*ET_PADDED],&w_down[((size_t)e*ET_HIDDEN+n)*ET_INTER],ET_INTER*2u);
	}
	std::vector<double> expect_up((size_t)local_rows*ET_INTER), bound_up((size_t)local_rows*ET_INTER), expect_down((size_t)local_rows*ET_HIDDEN), bound_down((size_t)local_rows*ET_HIDDEN);
	for (e=0u; e<ET_LOCAL; e++)
		for (p=offset[e]; p<offset[e+1u]; p++)
		{
			for (n=0u; n<ET_INTER; n++)
			{
				double s=0.0, m=0.0;
				for (k=0u; k<ET_HIDDEN; k++)
				{
					const double term=Bf16Value(w_up[((size_t)e*ET_INTER+n)*ET_HIDDEN+k])*Bf16Value(normed[(size_t)source[p]*ET_HIDDEN+k]);
					s+=term; m+=fabs(term);
				}
				expect_up[(size_t)p*ET_INTER+n]=s; bound_up[(size_t)p*ET_INTER+n]=m;
			}
			for (n=0u; n<ET_HIDDEN; n++)
			{
				double s=0.0, m=0.0, head=0.0;
				for (k=0u; k<ET_INTER; k++)
				{
					const double term=Bf16Value(w_down[((size_t)e*ET_HIDDEN+n)*ET_INTER+k])*Bf16Value(act_host[(size_t)p*ET_INTER+k]);
					s+=term; m+=fabs(term);
					if (k<64u) head+=term;
				}
				expect_down[(size_t)p*ET_HIDDEN+n]=s; bound_down[(size_t)p*ET_HIDDEN+n]=m;
				tail_outputs++;
				if (fabs(head-s)>fabs(s)/128.0+1e-5*m) tail_sensitive++;
			}
		}
	uint16_t *d_up=Device(w_up), *d_down=Device(w_down), *d_up_padded=Device(up_padded), *d_down_padded=Device(down_padded), *d_normed=Device(normed);
	uint16_t *d_act=Device(act_host), *d_act_padded=Device(act_padded), *d_out, *d_out_padded;
	CUDA(cudaMalloc(&d_out,(size_t)routes*ET_HIDDEN*2u)); CUDA(cudaMalloc(&d_out_padded,(size_t)routes*ET_HIDDEN*2u));
	LmLocalExpertRoutes expert_routes;
	memset(&expert_routes,0,sizeof(expert_routes));
	expert_routes.group_row_offset=d_offset;
	expert_routes.route_source_token=d_source;
	expert_routes.tile_prefix_up=d_tile_up;
	expert_routes.tile_prefix_down=d_tile_down;
	expert_routes.rows=rows;
	expert_routes.routes=routes;
	expert_routes.top_k=ET_TOP_K;
	expert_routes.local_experts=ET_LOCAL;
	expert_routes.multiprocessors=(uint32_t)device_multiprocessors;
	for (uint32_t path=0u; path<2u; path++)
	{
		const bool gemm=path==1u;
		const char *up_name=gemm ? "gemm-up" : "skinny-up", *down_name=gemm ? "gemm-down" : "skinny-down";
		int32_t status=Up(expert_routes,d_up,d_normed,d_out,ET_INTER,gemm,stream);
		if (!gemm && status==LM_LAUNCH_ERR_SHAPE) continue;
		REQUIRE(status==LM_LAUNCH_OK);
		CUDA(cudaStreamSynchronize(stream));
		std::vector<uint16_t> got_up=Download(d_out,(size_t)routes*ET_INTER);
		REQUIRE(Up(expert_routes,d_up_padded,d_normed,d_out_padded,ET_PADDED,gemm,stream)==LM_LAUNCH_OK);
		CUDA(cudaStreamSynchronize(stream));
		std::vector<uint16_t> got_up_padded=Download(d_out_padded,(size_t)routes*ET_PADDED);
		REQUIRE(Down(expert_routes,d_down,d_act,d_out,ET_INTER,gemm,stream)==LM_LAUNCH_OK);
		CUDA(cudaStreamSynchronize(stream));
		std::vector<uint16_t> got_down=Download(d_out,(size_t)routes*ET_HIDDEN);
		REQUIRE(Down(expert_routes,d_down_padded,d_act_padded,d_out_padded,ET_PADDED,gemm,stream)==LM_LAUNCH_OK);
		CUDA(cudaStreamSynchronize(stream));
		std::vector<uint16_t> got_down_padded=Download(d_out_padded,(size_t)routes*ET_HIDDEN);
		Compare(up_name,rows,local_rows,ET_INTER,ET_INTER,got_up,expect_up,bound_up);
		Compare(down_name,rows,local_rows,ET_HIDDEN,ET_HIDDEN,got_down,expect_down,bound_down);
		if (gemm)
		{
			Identical(up_name,rows,local_rows,ET_INTER,got_up,ET_INTER,got_up_padded,ET_PADDED);
			Identical(down_name,rows,local_rows,ET_HIDDEN,got_down,ET_HIDDEN,got_down_padded,ET_HIDDEN);
			gemm_cases++;
		}
		else
			skinny_cases++;
	}
	CUDA(cudaFree(d_route)); CUDA(cudaFree(d_offset)); CUDA(cudaFree(d_packed)); CUDA(cudaFree(d_source)); CUDA(cudaFree(d_tile_up)); CUDA(cudaFree(d_tile_down));
	CUDA(cudaFree(d_up)); CUDA(cudaFree(d_down)); CUDA(cudaFree(d_up_padded)); CUDA(cudaFree(d_down_padded)); CUDA(cudaFree(d_normed));
	CUDA(cudaFree(d_act)); CUDA(cudaFree(d_act_padded)); CUDA(cudaFree(d_out)); CUDA(cudaFree(d_out_padded));
}

int main(int argc, char **argv)
{
	static const uint32_t row_counts[] = { 1u, 2u, 5u, 16u, 33u, 96u, 160u, ET_MAX_ROWS };
	std::vector<uint16_t> w_up, w_down, normed;
	cudaStream_t stream;
	if (argc<2 || strcmp(argv[1],"--run")!=0) { printf("SKIP test_expert_tail_reference needs --run on a CUDA host\n"); return 0; }
	CUDA(cudaStreamCreate(&stream));
	Fill(w_up,(size_t)ET_LOCAL*ET_INTER*ET_HIDDEN,0.0625f);
	Fill(w_down,(size_t)ET_LOCAL*ET_HIDDEN*ET_INTER,0.0625f);
	Fill(normed,(size_t)ET_MAX_ROWS*ET_HIDDEN,4.0f);
	for (uint32_t i=0u; i<sizeof(row_counts)/sizeof(row_counts[0]); i++)
		Case(row_counts[i],w_up,w_down,normed,stream);
	CUDA(cudaStreamDestroy(stream));
	fprintf(stderr,"compared=%llu failures=%llu padded_differences=%llu skinny_cases=%llu gemm_cases=%llu tail_sensitive=%llu/%llu\n",(unsigned long long)compared,
		(unsigned long long)failures,(unsigned long long)padded_differences,(unsigned long long)skinny_cases,(unsigned long long)gemm_cases,
		(unsigned long long)tail_sensitive,(unsigned long long)tail_outputs);
	if (failures!=0u || padded_differences!=0u || skinny_cases==0u || gemm_cases==0u || tail_sensitive*2u<tail_outputs)
	{
		fprintf(stderr,"FAIL expert tail reference\n");
		return 1;
	}
	printf("OK expert tail reference: %u-row experts match the fp64 reference on the skinny and GEMM paths, and the GEMM output equals the %u-row zero-padded layout bit for bit\n",
		ET_INTER,ET_PADDED);
	return 0;
}
