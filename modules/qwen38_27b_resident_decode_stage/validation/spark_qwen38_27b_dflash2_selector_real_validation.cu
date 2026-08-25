#include <cuda_runtime.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_qwen38_27b_model.h"
#include "sparkpipe/spark_qwen38_27b_resident_decode_stage_firmware.h"
#include "spark_qwen38_27b_dspark_format.h"
#include "spark_qwen38_27b_dspark_selector_host.h"

/*
 * W7 end-to-end parity on the REAL DFlash2 weights.
 *
 * tools/qwen36_dflash2_selector_real_case.py runs the landed reference forward
 * (the 5-layer conv-wrapped block on the real drafter checkpoint plus the
 * target's shared lm_head and embed_tokens) and dumps the final-normed block
 * hidden without the anchor row - the exact tensor the module hands the
 * selector - together with the reference's own top-16 ids, unary logits,
 * context gate, lattice and walked draft ids, plus raw BF16 blobs of the
 * weights the selector reads. This validator loads them and runs
 * SparkQwen38_27bDsparkSelectorEmit, the module's own inline sequence, on the real
 * 248320 x 5120 head.
 *
 * READ THE PASS CRITERION. Unlike the integer harness, real weights do NOT make
 * the fp32 dot product order-independent: numpy's BLAS order and the kernel's
 * warp tree differ in the last ulps, and a logit that lands within an ulp of a
 * BF16 rounding boundary can therefore round the other way, which can also move
 * a top-16 boundary id. So this test gates on the DRAFT IDS - the contract's
 * output - and reports id/score agreement with any disagreement classified as
 * either an adjacent-BF16 boundary effect (fp32 accumulation order, expected)
 * or a real divergence (more than one BF16 step apart, a bug).
 */

#define SPARK_DFLASH2_REAL_MAGIC "Q6DF2RW1"

static uint32_t spark_dflash2_failures = 0u;

static float SparkDflash2Bf16ToFloat(uint16_t value)
{
	uint32_t bits = (uint32_t)value << 16u;
	float output;
	memcpy(&output,&bits,sizeof(output));
	return(output);
}

static uint16_t SparkDflash2FloatToBf16Bits(float value)
{
	uint32_t bits,lsb;
	memcpy(&bits,&value,sizeof(bits));
	lsb = (bits >> 16u) & 1u;
	bits += 0x7fffu + lsb;
	return((uint16_t)(bits >> 16u));
}

static int SparkDflash2LoadBlob(const char *path, void *destination_device, uint64_t bytes)
{
	FILE *file;
	uint8_t *staging;
	uint64_t chunk = 64ull << 20u,offset,count;
	cudaError_t error;
	file = fopen(path,"rb");
	if ( file == 0 )
	{
		fprintf(stderr,"cannot open blob %s\n",path);
		return(-1);
	}
	staging = (uint8_t *)malloc((size_t)chunk);
	if ( staging == 0 )
	{
		fclose(file);
		return(-1);
	}
	for (offset = 0ull; offset < bytes; offset += chunk)
	{
		count = bytes - offset < chunk ? bytes - offset : chunk;
		if ( fread(staging,1u,(size_t)count,file) != count )
		{
			fprintf(stderr,"blob %s is short at %llu\n",path,(unsigned long long)offset);
			free(staging);
			fclose(file);
			return(-1);
		}
		error = cudaMemcpy((uint8_t *)destination_device + offset,staging,(size_t)count,cudaMemcpyHostToDevice);
		if ( error != cudaSuccess )
		{
			fprintf(stderr,"blob %s upload failed: %s\n",path,cudaGetErrorString(error));
			free(staging);
			fclose(file);
			return(-1);
		}
	}
	free(staging);
	fclose(file);
	return(0);
}

#define SPARK_DFLASH2_CUDA(expression) \
	do { \
		cudaError_t status = (expression); \
		if ( status != cudaSuccess ) \
		{ \
			fprintf(stderr,"%s failed: %s\n",#expression,cudaGetErrorString(status)); \
			return(1); \
		} \
	} while (0)

int main(int argc, char **argv)
{
	const char *directory = argc > 1 ? argv[1] : "build/dflash2/real";
	char path[1024];
	FILE *file;
	char magic[8];
	uint32_t header[7],slots,vocab_count,hidden_dimension,rank,top_k,anchor_token_id,base_position;
	uint32_t slot_index,candidate,mismatched_ids = 0u,mismatched_scores = 0u,boundary = 0u,divergent = 0u,mismatched_drafts = 0u;
	uint16_t *hidden_host;
	uint32_t *reference_ids,*reference_drafts,*host_ids,*host_drafts;
	float *reference_unary,*host_scores;
	uint16_t *reference_gate;
	float *reference_edges;
	double worst_score_delta = 0.0;
	SparkQwen38_27bLinearView head_view;
	SparkQwen38_27bDsparkSelectorWorkspace workspace;
	void *hidden_device = 0,*head_device = 0,*projection_device = 0,*predecessor_device = 0,*successor_device = 0;
	uint32_t *anchor_device = 0;

	snprintf(path,sizeof(path),"%s/real_case.bin",directory);
	file = fopen(path,"rb");
	if ( file == 0 )
	{
		fprintf(stderr,"cannot open %s\n",path);
		return(2);
	}
	if ( fread(magic,1u,sizeof(magic),file) != sizeof(magic) || memcmp(magic,SPARK_DFLASH2_REAL_MAGIC,sizeof(magic)) != 0 ||
	     fread(header,sizeof(uint32_t),7u,file) != 7u )
	{
		fprintf(stderr,"case header is not %s\n",SPARK_DFLASH2_REAL_MAGIC);
		fclose(file);
		return(2);
	}
	slots = header[0];
	vocab_count = header[1];
	hidden_dimension = header[2];
	rank = header[3];
	top_k = header[4];
	anchor_token_id = header[5];
	base_position = header[6];
	hidden_host = (uint16_t *)malloc((size_t)slots * hidden_dimension * sizeof(uint16_t));
	reference_ids = (uint32_t *)malloc((size_t)slots * top_k * sizeof(uint32_t));
	reference_unary = (float *)malloc((size_t)slots * top_k * sizeof(float));
	reference_gate = (uint16_t *)malloc((size_t)slots * rank * sizeof(uint16_t));
	reference_edges = (float *)malloc((size_t)slots * top_k * top_k * sizeof(float));
	reference_drafts = (uint32_t *)malloc((size_t)slots * sizeof(uint32_t));
	host_ids = (uint32_t *)malloc((size_t)slots * top_k * sizeof(uint32_t));
	host_scores = (float *)malloc((size_t)slots * top_k * sizeof(float));
	host_drafts = (uint32_t *)malloc((size_t)slots * sizeof(uint32_t));
	if ( hidden_host == 0 || reference_ids == 0 || reference_unary == 0 || reference_gate == 0 ||
	     reference_edges == 0 || reference_drafts == 0 || host_ids == 0 || host_scores == 0 || host_drafts == 0 ||
	     fread(hidden_host,sizeof(uint16_t),(size_t)slots * hidden_dimension,file) != (size_t)slots * hidden_dimension ||
	     fread(reference_ids,sizeof(uint32_t),(size_t)slots * top_k,file) != (size_t)slots * top_k ||
	     fread(reference_unary,sizeof(float),(size_t)slots * top_k,file) != (size_t)slots * top_k ||
	     fread(reference_gate,sizeof(uint16_t),(size_t)slots * rank,file) != (size_t)slots * rank ||
	     fread(reference_edges,sizeof(float),(size_t)slots * top_k * top_k,file) != (size_t)slots * top_k * top_k ||
	     fread(reference_drafts,sizeof(uint32_t),slots,file) != slots )
	{
		fprintf(stderr,"case payload is short\n");
		fclose(file);
		return(2);
	}
	fclose(file);

	printf("case            = %s/real_case.bin\n",directory);
	printf("geometry        = slots=%u vocab=%u hidden=%u rank=%u top_k=%u anchor=%u base_position=%u\n",
		slots,vocab_count,hidden_dimension,rank,top_k,anchor_token_id,base_position);
	printf("weights         = the real DFlash2 drafter codebooks + the real target lm_head (%.3f GiB BF16)\n",
		(double)((uint64_t)vocab_count * hidden_dimension * 2ull) / 1073741824.0);
	printf("host path       = SparkQwen38_27bDsparkSelectorEmit (the module's own inline sequence)\n");

	SPARK_DFLASH2_CUDA(cudaMalloc(&hidden_device,(size_t)slots * hidden_dimension * 2ull));
	SPARK_DFLASH2_CUDA(cudaMalloc(&head_device,(size_t)vocab_count * hidden_dimension * 2ull));
	SPARK_DFLASH2_CUDA(cudaMalloc(&projection_device,(size_t)rank * hidden_dimension * 2ull));
	SPARK_DFLASH2_CUDA(cudaMalloc(&predecessor_device,(size_t)vocab_count * rank * 2ull));
	SPARK_DFLASH2_CUDA(cudaMalloc(&successor_device,(size_t)vocab_count * rank * 2ull));
	SPARK_DFLASH2_CUDA(cudaMalloc((void **)&anchor_device,sizeof(uint32_t)));
	SPARK_DFLASH2_CUDA(cudaMalloc((void **)&workspace.chunk_keys,(size_t)SparkQwen38_27bDsparkHeadTopKChunkKeyCount(slots,top_k) * sizeof(uint64_t)));
	SPARK_DFLASH2_CUDA(cudaMalloc((void **)&workspace.candidate_ids,(size_t)slots * top_k * sizeof(uint32_t)));
	SPARK_DFLASH2_CUDA(cudaMalloc((void **)&workspace.candidate_scores,(size_t)slots * top_k * sizeof(float)));
	SPARK_DFLASH2_CUDA(cudaMalloc(&workspace.context_gate_bf16,(size_t)slots * rank * 2ull));
	SPARK_DFLASH2_CUDA(cudaMalloc((void **)&workspace.edges_f32,(size_t)slots * top_k * top_k * sizeof(float)));
	SPARK_DFLASH2_CUDA(cudaMalloc((void **)&workspace.draft_token_ids,(size_t)slots * sizeof(uint32_t)));
	SPARK_DFLASH2_CUDA(cudaMemcpy(hidden_device,hidden_host,(size_t)slots * hidden_dimension * 2ull,cudaMemcpyHostToDevice));
	SPARK_DFLASH2_CUDA(cudaMemcpy(anchor_device,&anchor_token_id,sizeof(uint32_t),cudaMemcpyHostToDevice));
	snprintf(path,sizeof(path),"%s/lm_head.bf16",directory);
	if ( SparkDflash2LoadBlob(path,head_device,(uint64_t)vocab_count * hidden_dimension * 2ull) != 0 )
		return(3);
	snprintf(path,sizeof(path),"%s/hidden_projection.bf16",directory);
	if ( SparkDflash2LoadBlob(path,projection_device,(uint64_t)rank * hidden_dimension * 2ull) != 0 )
		return(3);
	snprintf(path,sizeof(path),"%s/predecessor.bf16",directory);
	if ( SparkDflash2LoadBlob(path,predecessor_device,(uint64_t)vocab_count * rank * 2ull) != 0 )
		return(3);
	snprintf(path,sizeof(path),"%s/successor.bf16",directory);
	if ( SparkDflash2LoadBlob(path,successor_device,(uint64_t)vocab_count * rank * 2ull) != 0 )
		return(3);

	memset(&head_view,0,sizeof(head_view));
	head_view.abi_version = SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_LINEAR_VIEW_ABI_VERSION;
	head_view.weight_format = SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16;
	head_view.input_dimension = hidden_dimension;
	head_view.output_dimension = vocab_count;
	head_view.weight_payload = head_device;
	head_view.weight_payload_bytes = (uint64_t)vocab_count * hidden_dimension * 2ull;

	SPARK_DFLASH2_CUDA(SparkQwen38_27bDsparkSelectorEmit(0,&head_view,hidden_device,projection_device,
		predecessor_device,successor_device,anchor_device,&workspace,slots,top_k,rank,hidden_dimension,host_drafts));
	SPARK_DFLASH2_CUDA(cudaDeviceSynchronize());
	SPARK_DFLASH2_CUDA(cudaMemcpy(host_ids,workspace.candidate_ids,(size_t)slots * top_k * sizeof(uint32_t),cudaMemcpyDeviceToHost));
	SPARK_DFLASH2_CUDA(cudaMemcpy(host_scores,workspace.candidate_scores,(size_t)slots * top_k * sizeof(float),cudaMemcpyDeviceToHost));

	for (slot_index = 0u; slot_index < slots; slot_index++)
		for (candidate = 0u; candidate < top_k; candidate++)
		{
			uint32_t index = (slot_index * top_k) + candidate;
			double delta = fabs((double)host_scores[index] - (double)reference_unary[index]);
			if ( host_ids[index] != reference_ids[index] )
				mismatched_ids++;
			if ( delta > worst_score_delta )
				worst_score_delta = delta;
			if ( host_scores[index] != reference_unary[index] )
			{
				uint16_t actual = SparkDflash2FloatToBf16Bits(host_scores[index]);
				uint16_t expected = SparkDflash2FloatToBf16Bits(reference_unary[index]);
				mismatched_scores++;
				if ( (uint32_t)(actual > expected ? actual - expected : expected - actual) <= 1u )
					boundary++;
				else
					divergent++;
			}
		}
	for (slot_index = 0u; slot_index < slots; slot_index++)
		if ( host_drafts[slot_index] != reference_drafts[slot_index] )
			mismatched_drafts++;

	printf("\n-- W4 top-%u on the real head (%u slots)\n",top_k,slots);
	printf("top ids           %8u / %8u equal\n",(slots * top_k) - mismatched_ids,slots * top_k);
	printf("unary logits      %8u / %8u bitwise equal  max|delta|=%.6g\n",(slots * top_k) - mismatched_scores,slots * top_k,worst_score_delta);
	printf("score disagreement class: %u adjacent-BF16 (fp32 accumulation order), %u divergent (> 1 BF16 step)\n",boundary,divergent);
	printf("\n-- W7 end-to-end: the module's emitted drafts vs the reference walk\n");
	printf("reference drafts  =");
	for (slot_index = 0u; slot_index < slots; slot_index++)
		printf(" %u",reference_drafts[slot_index]);
	printf("\nmodule drafts     =");
	for (slot_index = 0u; slot_index < slots; slot_index++)
		printf(" %u",host_drafts[slot_index]);
	printf("\ndraft ids         %8u / %8u equal\n",slots - mismatched_drafts,slots);
	if ( mismatched_drafts != 0u )
		spark_dflash2_failures++;
	if ( divergent != 0u )
		spark_dflash2_failures++;
	printf("\nresult            = %s (%u failing gates)\n",spark_dflash2_failures == 0u ? "PARITY" : "MISMATCH",spark_dflash2_failures);
	(void)reference_gate;
	(void)reference_edges;
	return(spark_dflash2_failures == 0u ? 0 : 1);
}
