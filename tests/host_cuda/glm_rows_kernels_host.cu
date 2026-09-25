#include "tests/host_cuda/lm_host_cuda.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "inference/kernels/dtype.cuh"
#define __CUDACC__ 1
#include "inference/kernels/kv.cuh"
#include "inference/kernels/gemm.cuh"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/topk.cuh"
#include "runtime/launch.h"
#include "tests/host_cuda/lm_host_threads.cuh"
LmHostDim3 blockDim,gridDim;
#include "inference/kernels/attn.cuh"
#include "inference/kernels/head.cuh"

#define HOST_LATENT 512u
#define HOST_PAGE 64u
#define HOST_HIDDEN 256u
#define HOST_VOCABULARY 1000u
#define HOST_TILE 128u
#define HOST_TILES ((HOST_VOCABULARY + HOST_TILE - 1u) / HOST_TILE)
#define HOST_MAX_ROWS 40u
#define HOST_MAX_CONTEXT 3000u
#define HOST_MAX_POOL_ROWS 5u

struct HostKv
{
	static constexpr uint32_t kSlotBytes = HOST_LATENT * 2u;
	static constexpr uint32_t kPageSlots = HOST_PAGE;
	static constexpr uint32_t kPageBytes = kSlotBytes * HOST_PAGE;
	static constexpr bool kGrows = true;
	static constexpr uint32_t PageOf(uint32_t position) { return position / HOST_PAGE; }
	static constexpr uint32_t SlotInPage(uint32_t position) { return position % HOST_PAGE; }
	static constexpr uint64_t PagesForTokens(uint64_t tokens) { return (tokens + HOST_PAGE - 1u) / HOST_PAGE; }
};

typedef struct HostAttention
{
	uint32_t rows,heads,pages,selected,contexts[HOST_MAX_POOL_ROWS],positions[HOST_MAX_POOL_ROWS],sequences[HOST_MAX_POOL_ROWS];
	uint32_t table[HOST_MAX_POOL_ROWS * (HOST_MAX_CONTEXT / HOST_PAGE + 1u)],selection[HOST_MAX_POOL_ROWS * 64u];
	uint16_t pool[HOST_MAX_POOL_ROWS * (HOST_MAX_CONTEXT / HOST_PAGE + 1u) * HOST_PAGE * HOST_LATENT],query[HOST_MAX_POOL_ROWS * 4u * HOST_LATENT],output[HOST_MAX_POOL_ROWS * 4u * HOST_LATENT];
	float partials[HOST_MAX_POOL_ROWS * 4u * LM_LATENT_ATTN_SPLIT_MAX_PARTITIONS * (HOST_LATENT + 2u)];
}
HostAttention;

static uint32_t host_state = 424242u;

static float HostSigned(void)
{
	host_state = host_state * 1664525u + 1013904223u;
	return (float)((int32_t)(host_state >> 20u) - 2048) / 2048.0f;
}

static void HostHeadRun(uint32_t rows_kernel,const uint16_t *normed,const uint16_t *weight,uint32_t rows,uint32_t *token,float *score)
{
	static float candidate_score[HOST_MAX_ROWS * HOST_TILES];
	static uint32_t candidate_token[HOST_MAX_ROWS * HOST_TILES];
	if (rows_kernel != 0u)
		LM_LAUNCH((LmHeadCandidateRowsKernel<256u,HOST_TILE,16u>),dim3(HOST_TILES,(rows + 15u) / 16u),256u,0,0,normed,weight,(const uint32_t *)0,candidate_score,candidate_token,rows,HOST_HIDDEN,HOST_VOCABULARY);
	else
		LM_LAUNCH((LmHeadCandidateKernel<256u,HOST_TILE>),dim3(HOST_TILES,rows),256u,0,0,normed,weight,(const uint32_t *)0,candidate_score,candidate_token,rows,HOST_HIDDEN,HOST_VOCABULARY);
	LM_LAUNCH((LmHeadCommitKernel<256u>),dim3(rows),256u,0,0,candidate_score,candidate_token,HOST_TILES,token,score,rows);
}

static void HostHeadCase(uint32_t rows)
{
	static uint16_t normed[HOST_MAX_ROWS * HOST_HIDDEN],weight[HOST_VOCABULARY * HOST_HIDDEN];
	uint32_t token_a[HOST_MAX_ROWS],token_b[HOST_MAX_ROWS],index,row;
	float score_a[HOST_MAX_ROWS],score_b[HOST_MAX_ROWS];
	for (index=0u; index<rows * HOST_HIDDEN; index++) normed[index] = LmFloatToBf16(HostSigned());
	for (index=0u; index<HOST_VOCABULARY * HOST_HIDDEN; index++) weight[index] = LmFloatToBf16(HostSigned() * 0.05f);
	for (index=0u; index<HOST_HIDDEN; index++) weight[901u * HOST_HIDDEN + index] = weight[77u * HOST_HIDDEN + index] = normed[index];
	HostHeadRun(0u,normed,weight,rows,token_a,score_a);
	HostHeadRun(1u,normed,weight,rows,token_b,score_b);
	assert(token_b[0] == 77u);
	for (row=0u; row<rows; row++)
		if (token_a[row] != token_b[row] || memcmp(&score_a[row],&score_b[row],sizeof(float)) != 0)
		{
			fprintf(stderr,"FAIL head rows=%u row=%u token=%u/%u score=%.9g/%.9g\n",rows,row,token_a[row],token_b[row],score_a[row],score_b[row]);
			exit(1);
		}
}

static void HostAttentionReference(const HostAttention *item,uint32_t row,uint32_t head,double *out)
{
	uint32_t count = item->selected != 0u ? item->selected : item->contexts[row],step,position,element,pass;
	double maximum = -INFINITY,sum = 0.0,score,weight;
	const uint16_t *slot;
	for (element=0u; element<HOST_LATENT; element++) out[element] = 0.0;
	for (pass=0u; pass<2u; pass++)
		for (step=0u; step<count; step++)
		{
			position = item->selected != 0u ? item->selection[row * item->selected + step] : step;
			if (position > item->positions[row])
				continue;
			slot = &item->pool[((uint64_t)item->table[row * item->pages + position / HOST_PAGE] * HOST_PAGE + position % HOST_PAGE) * HOST_LATENT];
			score = 0.0;
			for (element=0u; element<HOST_LATENT; element++) score += (double)LmBf16ToFloat(item->query[(row * item->heads + head) * HOST_LATENT + element]) * LmBf16ToFloat(slot[element]);
			score *= 0.0625;
			if (pass == 0u) { maximum = score > maximum ? score : maximum; continue; }
			weight = exp(score - maximum);
			sum += weight;
			for (element=0u; element<HOST_LATENT; element++) out[element] += weight * LmBf16ToFloat(slot[element]);
		}
	for (element=0u; element<HOST_LATENT; element++) out[element] /= sum;
}

static void HostAttentionCase(uint32_t rows,uint32_t heads,uint32_t context,uint32_t selected,uint32_t multiprocessors)
{
	static HostAttention item;
	static double reference[HOST_LATENT];
	LmKvAccessError error = {};
	LmKvView view;
	uint32_t row,step,index,head;
	double worst = 0.0,difference;
	memset(&item,0,sizeof(item));
	item.rows = rows; item.heads = heads; item.selected = selected; item.pages = (context + HOST_PAGE - 1u) / HOST_PAGE;
	for (index=0u; index<rows * item.pages; index++) item.table[index] = rows * item.pages - 1u - index;
	for (index=0u; index<rows * item.pages * HOST_PAGE * HOST_LATENT; index++) item.pool[index] = LmFloatToBf16(HostSigned());
	for (index=0u; index<rows * heads * HOST_LATENT; index++) item.query[index] = LmFloatToBf16(HostSigned() * 0.25f);
	for (row=0u; row<rows; row++)
	{
		item.sequences[row] = row;
		item.contexts[row] = row == 0u ? context : 1u + (host_state = host_state * 1664525u + 1013904223u) % context;
		item.positions[row] = item.contexts[row] - 1u;
		for (step=0u; step<selected; step++)
			item.selection[row * selected + step] = step % 13u == 5u ? 0xffffffffu : (host_state = host_state * 1664525u + 1013904223u) % item.contexts[row];
	}
	assert(LmKvViewInitialize(&view,(uint8_t *)item.pool,item.table,item.pages,rows,rows * item.pages,&error) == 0);
	assert((LmLatentAttentionHeadsLaunch<HostKv,HOST_LATENT>(item.query,view,item.sequences,item.contexts,selected != 0u ? item.selection : 0,selected,heads,0.0625f,item.output,item.positions,rows,selected != 0u ? selected : context,64u,item.partials,rows * heads * LM_LATENT_ATTN_SPLIT_MAX_PARTITIONS,multiprocessors,0)) == cudaSuccess);
	assert(error.error_code == LM_KV_ACCESS_ERROR_NONE);
	for (row=0u; row<rows; row++)
		for (head=0u; head<heads; head++)
		{
			HostAttentionReference(&item,row,head,reference);
			for (index=0u; index<HOST_LATENT; index++)
			{
				difference = fabs((double)LmBf16ToFloat(item.output[(row * heads + head) * HOST_LATENT + index]) - reference[index]);
				worst = difference > worst ? difference : worst;
			}
		}
	if (worst > 1.0e-2)
	{
		fprintf(stderr,"FAIL attention rows=%u heads=%u context=%u selected=%u multiprocessors=%u worst=%.6f\n",rows,heads,context,selected,multiprocessors,worst);
		exit(1);
	}
	printf("attention rows=%u heads=%u context=%u selected=%u multiprocessors=%u worst_abs=%.6f\n",rows,heads,context,selected,multiprocessors,worst);
}

int main(void)
{
	uint32_t rows;
	for (rows=2u; rows<=HOST_MAX_ROWS; rows = rows == 2u ? 15u : rows == 17u ? 40u : rows + 1u)
		HostHeadCase(rows);
	HostAttentionCase(3u,1u,130u,0u,48u);
	HostAttentionCase(2u,2u,3000u,40u,48u);
	HostAttentionCase(3u,4u,200u,0u,48u);
	HostAttentionCase(5u,4u,60u,0u,1u);
	HostAttentionCase(2u,4u,1000u,64u,48u);
	puts("PASS GLM row kernels on host threads: head rows kernel equals the per-row kernel bitwise for 2-40 rows; all-heads latent attention matches an f64 reference, split and unsplit, with selected positions");
	return(0);
}
