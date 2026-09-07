#include "runtime/workspace.h"
#include <stdio.h>
#include <string.h>
static int32_t fails=0;
static void ck(int c,const char*l){printf(c?"  ok   %s\n":"  FAIL %s\n",l); if(!c)fails++;}
int main(void){
	LmWorkspaceshape_t shape; LmWorkspacelayout_t layout;
	uint32_t a,b; int32_t st;
	memset(&shape,0,sizeof shape);
	shape.tokens=128; shape.top_k=8; shape.expert_count=256;
	shape.hidden_dimension=6144; shape.intermediate_dimension=2048;
	shape.tile_m=16; shape.tile_n=128;
	printf("GLM 5.2 routed MoE, B128, NVFP4 workspace\n\n");
	st=LmWorkspacelayout_build(&shape,&layout);
	ck(st==LM_WS_OK,"builds");
	ck(layout.packed_rows==1024,"128 tokens x top-8 = 1024 packed rows");
	ck(layout.bytes[LM_WS_REGION_PACKED_HIDDEN]==1024ULL*3072ULL,
	   "packed hidden is rows x hidden/2 bytes (NVFP4), not rows x hidden");
	ck(layout.bytes[LM_WS_REGION_PACKED_HIDDEN_SCALE]==1024ULL*384ULL,
	   "hidden scales are one UE4M3 byte per 16 elements");
	ck(layout.bytes[LM_WS_REGION_GATE_UP_BF16]==1024ULL*2048ULL*2ULL*2ULL,
	   "gate+up bf16 carries both components");
	ck(layout.bytes[LM_WS_REGION_INTERMEDIATE]==1024ULL*1024ULL,
	   "intermediate is rows x intermediate/2 bytes");
	printf("\nno region overlaps its neighbour, all aligned\n");
	for(a=0;a<LM_WS_REGION_COUNT;a++){
		if((layout.offset[a]%LM_WS_ALIGNMENT)!=0){
			printf("  FAIL region %u misaligned\n",a); fails++; }
		for(b=a+1;b<LM_WS_REGION_COUNT;b++){
			uint64_t ae=layout.offset[a]+layout.bytes[a], be=layout.offset[b]+layout.bytes[b];
			if(layout.offset[a]<be && layout.offset[b]<ae){
				printf("  FAIL regions %u and %u overlap\n",a,b); fails++; }}}
	ck(1,"9 regions pairwise disjoint and 256-byte aligned");
	ck(layout.total_bytes>=layout.offset[LM_WS_REGION_ROUTE_OUTPUT_BF16]
	   +layout.bytes[LM_WS_REGION_ROUTE_OUTPUT_BF16],
	   "total covers the last region");
	printf("\ntile count reflects that decode groups are short\n");
	printf("  packed rows %llu over %u experts -> %llu tiles\n",
	   (unsigned long long)layout.packed_rows,shape.expert_count,
	   (unsigned long long)layout.total_tiles);
	ck(layout.total_tiles==256ULL*1ULL*32ULL,
	   "4 rows/expert fits one M tile; 4096 N over TILE_N=128 is 32");
	printf("\nTILE_M selection across token buckets\n");
	printf("  the failure this prevents: a tile shorter than an expert's row count\n");
	printf("  splits every expert in two, and each M tile re-reads the same weight\n");
	printf("  tile, doubling the stream that is 96%% of all traffic.\n");
	{
		static const uint32_t buckets[] = { 1u, 8u, 64u, 128u, 512u, 1024u };
		uint32_t i;
		for (i = 0; i < 6u; ++i)
		{
			uint64_t peak; uint64_t m_tiles;
			memset(&shape,0,sizeof shape);
			shape.tokens=buckets[i]; shape.top_k=8; shape.expert_count=256;
			shape.hidden_dimension=6144; shape.intermediate_dimension=2048;
			shape.tile_m=0; shape.tile_n=128;
			if (LmWorkspacelayout_build(&shape,&layout)!=LM_WS_OK){
				printf("  FAIL bucket %u did not build\n",buckets[i]); fails++; continue; }
			peak = LmWorkspacepeak_rows_per_expert(&shape);
			m_tiles = (peak + layout.tile_m - 1u) / layout.tile_m;
			printf("    B%-5u peak rows/expert %-4llu -> TILE_M %-4u  M tiles %llu %s\n",
				buckets[i],(unsigned long long)peak,layout.tile_m,
				(unsigned long long)m_tiles, m_tiles<=1?"":"  <-- WEIGHT RE-READ");
			if (m_tiles > 1u){ fails++; }
		}
	}
	ck(1,"every bucket resolves to a single M tile, so no weight tile is read twice");
	printf("\n  stage depth selected with the tile, bounded by 128 KB shared\n");
	{
		static const uint32_t buckets[] = { 1u, 128u, 512u, 1024u };
		uint32_t i;
		for (i = 0; i < 4u; ++i)
		{
			memset(&shape,0,sizeof shape);
			shape.tokens=buckets[i]; shape.top_k=8; shape.expert_count=256;
			shape.hidden_dimension=6144; shape.intermediate_dimension=2048;
			shape.tile_m=0; shape.tile_n=128;
			if (LmWorkspacelayout_build(&shape,&layout)!=LM_WS_OK){
				printf("    FAIL B%u did not build\n",buckets[i]); fails++; continue; }
			printf("    B%-5u TILE_M %-4u stages %-2u shared %6llu B  CTAs/SM %u %s\n",
				buckets[i],layout.tile_m,layout.stages,
				(unsigned long long)layout.shared_bytes,layout.ctas_per_sm,
				layout.shared_bytes<=131072ULL?"":"  <-- OVER LIMIT");
			if (layout.shared_bytes > 131072ULL) fails++;
			if (layout.ctas_per_sm < 2u) { printf("      only %u CTA/SM\n",layout.ctas_per_sm); fails++; }
		}
	}
	ck(1,"every selected (TILE_M, stages) pair fits in shared memory");
	{
		ck(LmWorkspaceselect_tile_m(64u)==64u,
		   "64 is the tile ceiling; B1024 is the supported maximum");
		ck(LmWorkspaceselect_tile_m(4096u)==64u,
		   "no bucket selects a 128-row tile, so none is instantiated");
		ck(LmWorkspaceselect_stages(64u,128u,256u,4u,131072ULL)==2u,
		   "lookahead of one; deeper satisfies a requirement already met 20x over");
		ck(LmWorkspacectas_per_sm(
		       LmWorkspaceshared_bytes(64u,128u,256u,2u,4u),131072ULL)>=2u,
		   "the freed shared memory buys at least a second CTA per SM");
	}
	{
		memset(&shape,0,sizeof shape);
		shape.tokens=1024; shape.top_k=8; shape.expert_count=256;
		shape.hidden_dimension=6144; shape.intermediate_dimension=2048;
		shape.tile_m=16; shape.tile_n=128;
		LmWorkspacelayout_build(&shape,&layout);
		ck(layout.tile_m==16,"an explicit TILE_M is honoured, so a sweep can pin it");
		shape.tile_m=0;
		LmWorkspacelayout_build(&shape,&layout);
		ck(layout.tile_m==64,"B1024 auto-selects 64, not the 16 that would double the stream");
		ck(layout.stages==2,"B1024 runs at lookahead 1");
		ck(layout.ctas_per_sm>=2,"B1024 fits at least two CTAs per SM");
	}
	printf("\nworkspace requirement versus what B12x reserves\n");
	{
		uint64_t need = LmWorkspacebytes_for_max_batch(1024u,8u,256u,6144u,2048u);
		uint64_t b12x = 8u*4u + 8u*4u + (uint64_t)8u*6144u*2u;
		printf("    first-party unfused, B1024 max : %8.2f MB\n", need/1048576.0);
		printf("    B12x fused                     : %8.2f KB\n", b12x/1024.0);
		printf("    ratio                          : %8.0fx\n", (double)need/(double)b12x);
		ck(need > 0, "max-batch requirement resolves");
		ck(need > b12x * 1000u,
		   "the unfused path needs three orders of magnitude more workspace, so it "
		   "must own its allocation and cannot borrow the B12x plan's");
	}
	printf("\nrejections\n");
	memset(&shape,0,sizeof shape);
	shape.tokens=128; shape.top_k=8; shape.expert_count=256;
	shape.intermediate_dimension=2048; shape.tile_m=16; shape.tile_n=128;
	shape.hidden_dimension=6150;
	ck(LmWorkspacelayout_build(&shape,&layout)==LM_WS_ERR_GROUP,
	   "hidden not divisible by the NVFP4 group is rejected");
	shape.hidden_dimension=6144; shape.intermediate_dimension=0;
	ck(LmWorkspacelayout_build(&shape,&layout)==LM_WS_ERR_SHAPE,
	   "zero intermediate rejected");
	printf("\n%s (%d failing)\n",fails?"FAIL":"PASS",fails); return fails?1:0;}
