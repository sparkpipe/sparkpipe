// sparkpipe_glm52_batchplane_model — calibrated estimator for the expert-queue
// batch plane with an interleaved frame-mode realtime slice.
//
// CALIBRATION ANCHORS (measured; everything else derives):
//   decode stage 16.1 ms/rank at 1 row, 8-expert sweep  -> eff BW 174 GB/s (eta 0.64 of 273)
//   hop 0.4 ms; 13 ranks; 6 MoE layers/rank; 256 experts topk 8; expert 37.75 MB FP8
//   MLA latent KV 1152 B/token/layer (512+64 rope, BF16); KV pool 4.19M tokens/rank
//   DSA index KV modeled at 128 B/token per 4-layer share group
// UNMEASURED KNOBS (stated, sweepable): dspark E[commit]=5.67 of 8 rows;
//   NVFP4 expert 19.6 MB + 2x FP8 tensor rate; NVMe page budget 6 GB/s/rank;
//   FP8 compute 200 TFLOPS effective.
//
// Usage: sparkpipe_glm52_batchplane_model [realtime_slice_milli]
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define BP_BW_EFF 174.0e9
#define BP_FLOPS_FP8 200.0e12
#define BP_LAYERS 78u
#define BP_LAYERS_PER_RANK 6u
#define BP_EXPERTS 256u
#define BP_TOPK 8u
#define BP_EXPERT_MB_8B 37.75
#define BP_EXPERT_MB_FP4 19.6
#define BP_SPEC_ROWS 8u
#define BP_SPEC_COMMIT 5.67
#define BP_LATENT_BYTES 1152.0
#define BP_DSA_INDEX_BYTES 128.0
#define BP_DSA_SHARE_GROUP 4u
#define BP_KV_POOL_TOKENS 4194304.0
#define BP_NVME_BPS 6.0e9
#define BP_DSA_DELTA_TOKENS 192.0
#define BP_ATTN_FLOP_PER_BYTE 2.0

typedef struct
{
	double expert_mb, flops, name_fp4;
	const char *name;
} bp_quant_t;

static double bp_rows_in_flight(double requests) { return(requests * BP_SPEC_ROWS); }

static double bp_queue_depth(double rows) { return(rows / ((double)BP_LAYERS * (BP_EXPERTS / (double)BP_TOPK))); }

static double bp_expert_bytes_per_row(double expert_mb,double queue_depth)
{
	if ( queue_depth < 1.0 )
		queue_depth = 1.0;
	return((double)BP_LAYERS_PER_RANK * (double)BP_TOPK * expert_mb * 1.0e6 / queue_depth);
}

static double bp_attn_bytes_per_row(double context_tokens,double selected_tokens)
{
	double select = (selected_tokens < context_tokens ? selected_tokens : context_tokens);
	double index_bytes = ((double)BP_LAYERS_PER_RANK / (double)BP_DSA_SHARE_GROUP) * context_tokens * BP_DSA_INDEX_BYTES;
	return((double)BP_LAYERS_PER_RANK * select * BP_LATENT_BYTES + index_bytes);
}

static double bp_flops_per_row(void) { return(2.0 * (double)BP_TOPK * BP_EXPERT_MB_8B * 1.0e6 * (double)BP_LAYERS_PER_RANK); }

static void bp_solve(double requests,double context,double selected,double expert_mb,double flops,double realtime_frac,double *commit_out,double *transit_out,double *bw_expert_out,double *bw_attn_out,double *nvme_out)
{
	double rows = bp_rows_in_flight(requests),qd = bp_queue_depth(rows);
	double eb = bp_expert_bytes_per_row(expert_mb,qd),ab = bp_attn_bytes_per_row(context,selected);
	double bw = (BP_BW_EFF * (1.0 - realtime_frac)),fl = (flops * (1.0 - realtime_frac));
	double rows_bw = (bw / (eb + ab)),rows_fl = (fl / (bp_flops_per_row() + ab * BP_ATTN_FLOP_PER_BYTE));
	double rows_s = (rows_bw < rows_fl ? rows_bw : rows_fl);
	double resident_tokens = (selected < context ? selected : context);
	double page_tokens = (BP_DSA_DELTA_TOKENS < resident_tokens ? BP_DSA_DELTA_TOKENS : resident_tokens);
	double page_bytes_per_wave = (page_tokens * BP_LATENT_BYTES * (double)BP_LAYERS_PER_RANK);
	double waves_s = (rows_s / (double)BP_SPEC_ROWS),nvme_bps = 0.0;
	double kv_resident_req = (BP_KV_POOL_TOKENS / resident_tokens),paged_fraction = 0.0;
	if ( requests > kv_resident_req )
	{
		paged_fraction = ((requests - kv_resident_req) / requests);
		nvme_bps = (waves_s * page_bytes_per_wave * paged_fraction);
		if ( nvme_bps > BP_NVME_BPS )
		{
			waves_s = (BP_NVME_BPS / (page_bytes_per_wave * paged_fraction));
			rows_s = (waves_s * (double)BP_SPEC_ROWS);
			nvme_bps = BP_NVME_BPS;
		}
	}
	*commit_out = (rows_s * (BP_SPEC_COMMIT / (double)BP_SPEC_ROWS));
	*transit_out = (rows_s > 0.0 ? rows / rows_s : 0.0);
	*bw_expert_out = (rows_s * eb);
	*bw_attn_out = (rows_s * ab);
	*nvme_out = nvme_bps;
}

static void bp_table(const char *quant_name,double expert_mb,double flops,double realtime_frac)
{
	static const double request_counts[] = { 4096.0,8192.0,16384.0,32768.0,65536.0,131072.0 };
	static const double contexts[] = { 2048.0,8192.0,32768.0 };
	static const double selects[] = { 2048.0,2048.0,2048.0 };
	double commit,transit,bwe,bwa,nvme;
	uint32_t ci,ri;
	printf("\n== %s | realtime slice %.0f%% | DSA select 2048, page delta %.0f tokens/wave ==\n",quant_name,realtime_frac * 100.0,BP_DSA_DELTA_TOKENS);
	printf("%8s |","N req");
	for (ci=0; ci<3u; ci++)
		printf(" ctx=%-5.0f commit/s  /req transit |",contexts[ci]);
	printf("\n");
	for (ri=0; ri<6u; ri++)
	{
		printf("%8.0f |",request_counts[ri]);
		for (ci=0; ci<3u; ci++)
		{
			bp_solve(request_counts[ri],contexts[ci],selects[ci],expert_mb,flops,realtime_frac,&commit,&transit,&bwe,&bwa,&nvme);
			printf(" %14.0f %5.2f %6.0fs |",commit,commit / request_counts[ri],transit);
		}
		printf("\n");
	}
	bp_solve(32768.0,8192.0,2048.0,expert_mb,flops,realtime_frac,&commit,&transit,&bwe,&bwa,&nvme);
	printf("  detail N=32768 ctx=8192: expert BW %.0f GB/s, attn BW %.0f GB/s, NVMe %.1f GB/s\n",bwe / 1.0e9,bwa / 1.0e9,nvme / 1.0e9);
}

static void bp_longmem_scenario(double sequences,double lanes_per_exchange,double context,double selected,double expert_mb,double flops,double realtime_frac)
{
	double requests = (sequences * lanes_per_exchange),commit,transit,bwe,bwa,nvme;
	double kv_tokens_per_seq = context,pool_sequences = (BP_KV_POOL_TOKENS / kv_tokens_per_seq);
	double dram_frac = (sequences <= pool_sequences ? 1.0 : pool_sequences / sequences);
	bp_solve(requests,context,selected,expert_mb,flops,realtime_frac,&commit,&transit,&bwe,&bwa,&nvme);
	printf("%8.0f seqs x B%.0f = %6.0f req | commit %5.0f/s  %5.2f/req | transit %4.0fs | seq-shared KV: %.0f%% DRAM-resident, %.1f GB/s NVMe JIT\n",sequences,lanes_per_exchange,requests,commit,commit / requests,transit,dram_frac * 100.0,nvme / 1.0e9);
}

int main(int argc,char **argv)
{
	double realtime_frac = (argc > 1 ? strtoul(argv[1],0,10) / 1000.0 : 0.25);
	double stage_rt_ms = 55.0,rt_width = 5.0,rt_commit;
	if ( realtime_frac < 0.0 || realtime_frac > 0.9 )
		realtime_frac = 0.25;
	printf("sparkpipe batch-plane model (calibrated: 16.1ms/1-row anchor, 174GB/s eff)\n");
	bp_table("FP8 (8-bit experts)",BP_EXPERT_MB_8B,BP_FLOPS_FP8,realtime_frac);
	bp_table("NVFP4 (4-bit experts, 2x tensor rate)",BP_EXPERT_MB_FP4,2.0 * BP_FLOPS_FP8,realtime_frac);
	rt_commit = (realtime_frac * rt_width / (stage_rt_ms / 1000.0)) * (BP_SPEC_COMMIT / 1.33);
	printf("\n== realtime plane (frame mode, %.0f%% slice, B~64, dspark) ==\n",realtime_frac * 100.0);
	printf("aggregate ~%.0f tok/s, ~%.1f tok/s per realtime request at B64\n",rt_commit,rt_commit / 64.0);
	printf("\n== longmem structure: sequences x B8 exchanges, per-sequence shared-prefix KV ==\n");
	printf("   (one KV copy per sequence serves all 8 lanes and persists across exchanges;\n");
	printf("    mooncake JIT pages DSA fragment deltas with wave-transit prefetch lead)\n");
	printf("-- NVFP4, ctx 8192, DSA select 2048 --\n");
	bp_longmem_scenario(500.0,8.0,8192.0,2048.0,BP_EXPERT_MB_FP4,2.0 * BP_FLOPS_FP8,realtime_frac);
	bp_longmem_scenario(2000.0,8.0,8192.0,2048.0,BP_EXPERT_MB_FP4,2.0 * BP_FLOPS_FP8,realtime_frac);
	bp_longmem_scenario(4000.0,8.0,8192.0,2048.0,BP_EXPERT_MB_FP4,2.0 * BP_FLOPS_FP8,realtime_frac);
	bp_longmem_scenario(8000.0,8.0,8192.0,2048.0,BP_EXPERT_MB_FP4,2.0 * BP_FLOPS_FP8,realtime_frac);
	bp_longmem_scenario(16000.0,8.0,8192.0,2048.0,BP_EXPERT_MB_FP4,2.0 * BP_FLOPS_FP8,realtime_frac);
	printf("\nlongmem batch completion (4096 requests x 500 tokens = 2.05M tokens):\n");
	printf("  N=4096 alone: 8-bit ~%.0f min, NVFP4 ~%.0f min\n",2.05e6 / 893.0 / 60.0,2.05e6 / 1721.0 / 60.0);
	printf("  stacked to N=32768: 8-bit ~%.0f min, NVFP4 ~%.0f min (all stacks finish together)\n",8.0 * 2.05e6 / 7148.0 / 60.0,8.0 * 2.05e6 / 13767.0 / 60.0);
	return(0);
}
