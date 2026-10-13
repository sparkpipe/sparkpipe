#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "inference/kernels/skinny.cuh"
#include "inference/kernels/route.cuh"

#define CUDA(call) do { cudaError_t e=(call); if(e!=cudaSuccess) { fprintf(stderr,"FAIL line=%d cuda=%s call=%s\n",__LINE__,cudaGetErrorString(e),#call); exit(1); } } while(0)
#define REQUIRE(test) do { if(!(test)) { fprintf(stderr,"FAIL line=%d test=%s\n",__LINE__,#test); exit(1); } } while(0)

#define GR_LOCAL 16u
#define GR_GLOBAL 256u
#define GR_TOP_K 10u
#define GR_MAX_ROWS 25u
#define GR_HIDDEN 3072u
#define GR_UP 128u
#define GR_DOWN_IN 1024u
#define GR_DOWN_OUT 512u
#define GR_THREADS 256u
#define GR_SENTINEL 0x5a5au

static uint32_t random_state = 20261013u;
static uint64_t compared = 0u, failures = 0u, sentinel_failures = 0u, multi_pass_groups = 0u;
static uint64_t failures_by_rows[GR_MAX_ROWS + 1u];

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

static uint16_t *Upload(std::vector<uint16_t> &host, std::vector<double> &value, size_t count, float amplitude)
{
	uint16_t *device;
	host.resize(count); value.resize(count);
	for (size_t i=0u; i<count; i++) { host[i]=Bf16(amplitude*(((int32_t)(Random()%2049u)-1024)/1024.0f)); value[i]=Bf16Value(host[i]); }
	CUDA(cudaMalloc(&device,count*2u));
	CUDA(cudaMemcpy(device,host.data(),count*2u,cudaMemcpyHostToDevice));
	return device;
}

static void Routes(uint32_t rows, uint32_t skew, std::vector<uint32_t> *local)
{
	local->assign((size_t)rows*GR_TOP_K,0u);
	for (uint32_t t=0u; t<rows; t++)
	{
		uint32_t chosen[GR_TOP_K], count=0u, e, j;
		bool seen;
		if (skew!=0u)
			for (e=0u; e<3u && count<GR_TOP_K; e++)
				if (Random()%10u!=0u) chosen[count++]=e*5u;
		while (count<GR_TOP_K)
		{
			e=Random()%GR_GLOBAL;
			seen=false;
			for (j=0u; j<count; j++) seen=seen || chosen[j]==e;
			if (!seen) chosen[count++]=e;
		}
		for (j=0u; j<GR_TOP_K; j++) (*local)[(size_t)t*GR_TOP_K+j]=chosen[j]<GR_LOCAL ? chosen[j] : GR_LOCAL;
	}
}

static void Compare(const char *path, uint32_t rows, uint32_t skew, uint32_t packed_rows, uint32_t local_rows, uint32_t output, const std::vector<uint16_t> &got,
	const std::vector<double> &expect, const std::vector<double> &bound)
{
	uint32_t printed=0u;
	for (uint32_t p=0u; p<packed_rows; p++)
		for (uint32_t n=0u; n<output; n++)
		{
			const size_t index=(size_t)p*output+n;
			if (p>=local_rows)
			{
				if (got[index]!=GR_SENTINEL) { sentinel_failures++; if (sentinel_failures<=8u) fprintf(stderr,"SENTINEL %s rows=%u skew=%u packed=%u col=%u\n",path,rows,skew,p,n); }
				continue;
			}
			const double value=Bf16Value(got[index]), tolerance=fabs(expect[index])/128.0+1e-5*bound[index]+1e-30;
			compared++;
			if (!(fabs(value-expect[index])<=tolerance))
			{
				failures++; failures_by_rows[rows]++;
				if (printed<3u && failures<=40u) fprintf(stderr,"MISMATCH %s rows=%u skew=%u packed=%u col=%u got=%.6g ref=%.6g tol=%.3g\n",path,rows,skew,p,n,value,expect[index],tolerance);
				printed++;
			}
		}
}

static void Case(uint32_t rows, uint32_t skew, const uint16_t *w_up, const std::vector<double> &up, const uint16_t *w_down, const std::vector<double> &down,
	const uint16_t *normed, const std::vector<double> &normed_value, cudaStream_t stream)
{
	const uint32_t routes=rows*GR_TOP_K;
	std::vector<uint32_t> local, offset(GR_LOCAL+2u), packed(routes), source(routes), tile_up(GR_LOCAL+2u), tile_down(GR_LOCAL+2u);
	uint32_t *d_route, *d_offset, *d_packed, *d_source, *d_tile_up, *d_tile_down, e, r, p, n, k;
	uint16_t *d_act, *d_up_out, *d_down_out;
	Routes(rows,skew,&local);
	CUDA(cudaMalloc(&d_route,routes*4u)); CUDA(cudaMalloc(&d_offset,(GR_LOCAL+2u)*4u)); CUDA(cudaMalloc(&d_packed,routes*4u)); CUDA(cudaMalloc(&d_source,routes*4u));
	CUDA(cudaMalloc(&d_tile_up,(GR_LOCAL+2u)*4u)); CUDA(cudaMalloc(&d_tile_down,(GR_LOCAL+2u)*4u));
	CUDA(cudaMemcpy(d_route,local.data(),routes*4u,cudaMemcpyHostToDevice));
	REQUIRE((LmRouteBuild<GR_THREADS,GR_LOCAL+1u>(d_route,rows,routes,GR_TOP_K,d_offset,d_packed,d_source,GR_UP,GR_HIDDEN,128u,d_tile_up,d_tile_down,stream))==LM_LAUNCH_OK);
	CUDA(cudaStreamSynchronize(stream));
	CUDA(cudaMemcpy(offset.data(),d_offset,(GR_LOCAL+2u)*4u,cudaMemcpyDeviceToHost));
	CUDA(cudaMemcpy(packed.data(),d_packed,routes*4u,cudaMemcpyDeviceToHost));
	CUDA(cudaMemcpy(source.data(),d_source,routes*4u,cudaMemcpyDeviceToHost));
	{
		std::vector<uint32_t> count(GR_LOCAL+1u,0u), used(routes,0u);
		for (r=0u; r<routes; r++) count[local[r]]++;
		REQUIRE(offset[0]==0u);
		for (e=0u; e<=GR_LOCAL; e++) { REQUIRE(offset[e+1u]-offset[e]==count[e]); if (e<GR_LOCAL && count[e]>LM_SKINNY_ROWS) multi_pass_groups++; }
		REQUIRE(offset[GR_LOCAL+1u]==routes);
		for (r=0u; r<routes; r++)
		{
			REQUIRE(packed[r]>=offset[local[r]] && packed[r]<offset[local[r]+1u]);
			REQUIRE(used[packed[r]]==0u); used[packed[r]]=1u;
			REQUIRE(source[packed[r]]==r/GR_TOP_K);
		}
	}
	const uint32_t local_rows=offset[GR_LOCAL];
	std::vector<uint16_t> act_host((size_t)routes*GR_DOWN_IN), got_up((size_t)routes*GR_UP), got_down((size_t)routes*GR_DOWN_OUT);
	std::vector<double> act((size_t)routes*GR_DOWN_IN), expect_up((size_t)routes*GR_UP,0.0), bound_up((size_t)routes*GR_UP,0.0), expect_down((size_t)routes*GR_DOWN_OUT,0.0), bound_down((size_t)routes*GR_DOWN_OUT,0.0);
	for (size_t i=0u; i<act_host.size(); i++) { act_host[i]=Bf16(4.0f*(((int32_t)(Random()%2049u)-1024)/1024.0f)); act[i]=Bf16Value(act_host[i]); }
	for (e=0u; e<GR_LOCAL; e++)
		for (p=offset[e]; p<offset[e+1u]; p++)
		{
			const double *a=&normed_value[(size_t)source[p]*GR_HIDDEN], *b=&act[(size_t)p*GR_DOWN_IN];
			for (n=0u; n<GR_UP; n++)
			{
				const double *w=&up[((size_t)e*GR_UP+n)*GR_HIDDEN];
				double s=0.0, m=0.0;
				for (k=0u; k<GR_HIDDEN; k++) { s+=w[k]*a[k]; m+=fabs(w[k]*a[k]); }
				expect_up[(size_t)p*GR_UP+n]=s; bound_up[(size_t)p*GR_UP+n]=m;
			}
			for (n=0u; n<GR_DOWN_OUT; n++)
			{
				const double *w=&down[((size_t)e*GR_DOWN_OUT+n)*GR_DOWN_IN];
				double s=0.0, m=0.0;
				for (k=0u; k<GR_DOWN_IN; k++) { s+=w[k]*b[k]; m+=fabs(w[k]*b[k]); }
				expect_down[(size_t)p*GR_DOWN_OUT+n]=s; bound_down[(size_t)p*GR_DOWN_OUT+n]=m;
			}
		}
	CUDA(cudaMalloc(&d_act,act_host.size()*2u)); CUDA(cudaMalloc(&d_up_out,got_up.size()*2u)); CUDA(cudaMalloc(&d_down_out,got_down.size()*2u));
	CUDA(cudaMemcpy(d_act,act_host.data(),act_host.size()*2u,cudaMemcpyHostToDevice));
	CUDA(cudaMemset(d_up_out,0x5a,got_up.size()*2u)); CUDA(cudaMemset(d_down_out,0x5a,got_down.size()*2u));
	REQUIRE(LmSkinnyGroupedExperts<LmBf16Format>(w_up,LmScaleTensorNone(),normed,d_up_out,d_offset,d_source,GR_LOCAL,routes,0u,GR_HIDDEN,GR_UP,stream)==LM_LAUNCH_OK);
	REQUIRE(LmSkinnyGroupedExperts<LmBf16Format>(w_down,LmScaleTensorNone(),d_act,d_down_out,d_offset,d_source,GR_LOCAL,routes,1u,GR_DOWN_IN,GR_DOWN_OUT,stream)==LM_LAUNCH_OK);
	CUDA(cudaStreamSynchronize(stream));
	CUDA(cudaMemcpy(got_up.data(),d_up_out,got_up.size()*2u,cudaMemcpyDeviceToHost));
	CUDA(cudaMemcpy(got_down.data(),d_down_out,got_down.size()*2u,cudaMemcpyDeviceToHost));
	Compare("grouped-up",rows,skew,routes,local_rows,GR_UP,got_up,expect_up,bound_up);
	Compare("grouped-down",rows,skew,routes,local_rows,GR_DOWN_OUT,got_down,expect_down,bound_down);
	CUDA(cudaFree(d_route)); CUDA(cudaFree(d_offset)); CUDA(cudaFree(d_packed)); CUDA(cudaFree(d_source)); CUDA(cudaFree(d_tile_up)); CUDA(cudaFree(d_tile_down));
	CUDA(cudaFree(d_act)); CUDA(cudaFree(d_up_out)); CUDA(cudaFree(d_down_out));
}

int main(int argc, char **argv)
{
	std::vector<uint16_t> host;
	std::vector<double> up, down, normed;
	cudaStream_t stream;
	uint32_t rows, skew;
	if (argc<2 || strcmp(argv[1],"--run")!=0) { printf("SKIP test_skinny_grouped_reference needs --run on a CUDA host\n"); return 0; }
	CUDA(cudaStreamCreate(&stream));
	uint16_t *w_up=Upload(host,up,(size_t)GR_LOCAL*GR_UP*GR_HIDDEN,0.0625f);
	uint16_t *w_down=Upload(host,down,(size_t)GR_LOCAL*GR_DOWN_OUT*GR_DOWN_IN,0.0625f);
	uint16_t *d_normed=Upload(host,normed,(size_t)GR_MAX_ROWS*GR_HIDDEN,4.0f);
	for (rows=1u; rows<=GR_MAX_ROWS; rows++)
		for (skew=0u; skew<2u; skew++)
			Case(rows,skew,w_up,up,w_down,down,d_normed,normed,stream);
	for (rows=1u; rows<=GR_MAX_ROWS; rows++)
		if (failures_by_rows[rows]!=0u) fprintf(stderr,"rows=%u failures=%llu\n",rows,(unsigned long long)failures_by_rows[rows]);
	CUDA(cudaFree(w_up)); CUDA(cudaFree(w_down)); CUDA(cudaFree(d_normed));
	CUDA(cudaStreamDestroy(stream));
	fprintf(stderr,"compared=%llu failures=%llu sentinel=%llu groups_over_%u_rows=%llu\n",(unsigned long long)compared,(unsigned long long)failures,(unsigned long long)sentinel_failures,LM_SKINNY_ROWS,(unsigned long long)multi_pass_groups);
	if (failures!=0u || sentinel_failures!=0u || multi_pass_groups==0u) { fprintf(stderr,"FAIL skinny grouped reference\n"); return 1; }
	printf("OK skinny grouped reference compared=%llu\n",(unsigned long long)compared);
	return 0;
}
