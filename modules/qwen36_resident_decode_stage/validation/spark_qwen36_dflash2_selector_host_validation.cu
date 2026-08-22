#include <cuda_runtime.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_qwen36_model.h"
#include "sparkpipe/spark_qwen36_resident_decode_stage_firmware.h"
#include "spark_qwen36_dspark_format.h"
#include "spark_qwen36_dspark_selector_host.h"

/*
 * DFlash2 selector HOST-PATH validator (adoption item W7).
 *
 * The kernel-level harness proves the kernels. This one proves the WIRING: it
 * calls SparkQwen36DsparkSelectorEmit - the exact inline sequence the module's
 * block forward now calls, from the same header - and checks that the draft ids
 * that come back on the host equal the numpy oracle's walk for the same inputs.
 * Inputs come from the same case file and the same counter-based splitmix64 the
 * kernel harness uses, so the two harnesses cannot disagree about the contract.
 *
 * Three things only a host-path test can catch, all asserted here:
 *   1. THE ANCHOR ROW OFFSET. The module hands the selector norm + H, skipping
 *      block position 0 (the committed token's own row, the reference's
 *      hidden[1:]). The block hidden here is allocated with an extra leading row
 *      filled with a POISON constant far larger than any real logit, so if the
 *      offset were dropped the top-16 of every slot would come from the poison
 *      row and the drafts would not match. The test cannot pass with the wrong
 *      offset.
 *   2. THE EMITTED WIDTH. The drafter proposes B-1 tokens, and
 *      SPARK_QWEN36_RESIDENT_DECODE_STAGE_DSPARK_BLOCK_SIZE is what the serving
 *      adapter asks for; the DSpark host loop wrote B of them into that B-1
 *      buffer. The test pins width == B-1 == the adapter's count and checks the
 *      word past the end is untouched.
 *   3. THE DEVICE ANCHOR. The walk's starting predecessor is read from device
 *      memory (the token the target head just committed), never from a host
 *      round trip; the anchor here is uploaded as a device word exactly as
 *      slot->output_token_ids is.
 */

#define SPARK_DFLASH2_CASE_MAGIC "Q6DF2CS2"
#define SPARK_DFLASH2_HEAD_CHUNK_ROWS 4096u
#define SPARK_DFLASH2_POISON_VALUE 4096.0f
#define SPARK_DFLASH2_GUARD_WORD 0xdeadbeefu

#define SPARK_DFLASH2_STREAM_HIDDEN 0x0000000011111111ull
#define SPARK_DFLASH2_STREAM_HEAD 0x0000000022222222ull
#define SPARK_DFLASH2_STREAM_PROJECTION 0x0000000033333333ull
#define SPARK_DFLASH2_STREAM_PREDECESSOR 0x0000000044444444ull
#define SPARK_DFLASH2_STREAM_SUCCESSOR 0x0000000055555555ull

static uint32_t spark_dflash2_failures = 0u;

static uint64_t SparkDflash2CaseValue(uint64_t index, uint64_t stream_seed, uint64_t seed)
{
	uint64_t z;
	z = ((index + 1ull) * 0x9E3779B97F4A7C15ull) + (stream_seed + seed);
	z = (z ^ (z >> 30u)) * 0xBF58476D1CE4E5B9ull;
	z = (z ^ (z >> 27u)) * 0x94D049BB133111EBull;
	return(z ^ (z >> 31u));
}

static float SparkDflash2SmallInteger(uint64_t index, uint64_t stream_seed, uint64_t seed, uint32_t span)
{
	uint64_t raw;
	if ( span == 0u )
		return(0.0f);
	raw = SparkDflash2CaseValue(index,stream_seed,seed) >> 11u;
	return((float)((int64_t)(raw % (uint64_t)((2u * span) + 1u)) - (int64_t)span));
}

static uint16_t SparkDflash2FloatToBf16(float value)
{
	uint32_t bits,lsb;
	memcpy(&bits,&value,sizeof(bits));
	lsb = (bits >> 16u) & 1u;
	bits += 0x7fffu + lsb;
	return((uint16_t)(bits >> 16u));
}

/* One device BF16 tensor from the shared PRNG, chunked so the 2.37 GiB head
 * never needs a host mirror. leading_rows are filled with the poison value
 * instead, standing in for the block's anchor row. */
static int SparkDflash2FillDeviceBf16(void *device, uint64_t element_count, uint64_t stream_seed, uint64_t seed, uint32_t span, uint64_t chunk_elements, uint64_t row_stride, uint32_t pairing, uint64_t poison_elements)
{
	uint16_t *staging;
	uint64_t first,index,count,flat,row,column;
	cudaError_t error;
	staging = (uint16_t *)malloc((size_t)chunk_elements * sizeof(uint16_t));
	if ( staging == 0 )
		return(-1);
	for (first = 0ull; first < element_count; first += chunk_elements)
	{
		count = element_count - first < chunk_elements ? element_count - first : chunk_elements;
		for (index = 0ull; index < count; index++)
		{
			flat = first + index;
			if ( flat < poison_elements )
			{
				staging[index] = SparkDflash2FloatToBf16(SPARK_DFLASH2_POISON_VALUE);
				continue;
			}
			flat -= poison_elements;
			if ( pairing != 0u && row_stride != 0ull )
			{
				row = flat / row_stride;
				column = flat - (row * row_stride);
				flat = ((row & ~1ull) * row_stride) + column;
			}
			staging[index] = SparkDflash2FloatToBf16(SparkDflash2SmallInteger(flat,stream_seed,seed,span));
		}
		error = cudaMemcpy((uint16_t *)device + first,staging,(size_t)count * sizeof(uint16_t),cudaMemcpyHostToDevice);
		if ( error != cudaSuccess )
		{
			free(staging);
			return(-1);
		}
	}
	free(staging);
	return(0);
}

static void SparkDflash2Check(const char *label, uint32_t condition, const char *detail)
{
	printf("%-38s %s%s%s\n",label,condition != 0u ? "PASS" : "FAIL",detail != 0 ? "  " : "",detail != 0 ? detail : "");
	if ( condition == 0u )
		spark_dflash2_failures++;
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
	const char *case_path = argc > 1 ? argv[1] : "build/dflash2/dflash2_selector_case.bin";
	FILE *file;
	char magic[8];
	uint32_t header[13],batch,slot_index,mismatches,batch_count,slot_count,vocab_count,hidden_dimension,rank,top_k;
	uint32_t hidden_span,weight_span,codebook_span,head_row_pairing;
	uint64_t seed,rows,candidates,index;
	uint32_t *anchor_ids,*expected_top_ids,*untruncated,*expected_drafts,*expected_slots,*lastmax_ids,*lastmax_slots;
	float *expected_unary,*expected_edges;
	uint16_t *gate_expected;
	SparkQwen36LinearView head_view;
	SparkQwen36DsparkSelectorWorkspace workspace;
	void *block_hidden = 0,*head_device = 0,*projection_device = 0,*predecessor_device = 0,*successor_device = 0;
	uint32_t *anchor_device = 0,*host_drafts = 0;
	uint64_t block_elements;

	file = fopen(case_path,"rb");
	if ( file == 0 )
	{
		fprintf(stderr,"cannot open case %s\n",case_path);
		return(2);
	}
	if ( fread(magic,1u,sizeof(magic),file) != sizeof(magic) || memcmp(magic,SPARK_DFLASH2_CASE_MAGIC,sizeof(magic)) != 0 ||
	     fread(header,sizeof(uint32_t),13u,file) != 13u || fread(&seed,sizeof(uint64_t),1u,file) != 1u )
	{
		fprintf(stderr,"case header is not %s\n",SPARK_DFLASH2_CASE_MAGIC);
		fclose(file);
		return(2);
	}
	batch_count = header[0];
	slot_count = header[1];
	vocab_count = header[2];
	hidden_dimension = header[3];
	rank = header[4];
	top_k = header[5];
	hidden_span = header[6];
	weight_span = header[7];
	codebook_span = header[8];
	head_row_pairing = header[9];
	rows = (uint64_t)batch_count * slot_count;
	candidates = rows * top_k;
	anchor_ids = (uint32_t *)malloc(batch_count * sizeof(uint32_t));
	expected_top_ids = (uint32_t *)malloc(candidates * sizeof(uint32_t));
	untruncated = (uint32_t *)malloc(candidates * sizeof(uint32_t));
	expected_unary = (float *)malloc(candidates * sizeof(float));
	gate_expected = (uint16_t *)malloc(rows * rank * sizeof(uint16_t));
	expected_edges = (float *)malloc(candidates * top_k * sizeof(float));
	expected_drafts = (uint32_t *)malloc(rows * sizeof(uint32_t));
	expected_slots = (uint32_t *)malloc(rows * sizeof(uint32_t));
	lastmax_ids = (uint32_t *)malloc(rows * sizeof(uint32_t));
	lastmax_slots = (uint32_t *)malloc(rows * sizeof(uint32_t));
	if ( anchor_ids == 0 || expected_top_ids == 0 || untruncated == 0 || expected_unary == 0 || gate_expected == 0 ||
	     expected_edges == 0 || expected_drafts == 0 || expected_slots == 0 || lastmax_ids == 0 || lastmax_slots == 0 ||
	     fread(anchor_ids,sizeof(uint32_t),batch_count,file) != batch_count ||
	     fread(expected_top_ids,sizeof(uint32_t),candidates,file) != candidates ||
	     fread(untruncated,sizeof(uint32_t),candidates,file) != candidates ||
	     fread(expected_unary,sizeof(float),candidates,file) != candidates ||
	     fread(gate_expected,sizeof(uint16_t),rows * rank,file) != rows * rank ||
	     fread(expected_edges,sizeof(float),candidates * top_k,file) != candidates * top_k ||
	     fread(expected_drafts,sizeof(uint32_t),rows,file) != rows ||
	     fread(expected_slots,sizeof(uint32_t),rows,file) != rows ||
	     fread(lastmax_ids,sizeof(uint32_t),rows,file) != rows ||
	     fread(lastmax_slots,sizeof(uint32_t),rows,file) != rows )
	{
		fprintf(stderr,"case payload is short\n");
		fclose(file);
		return(2);
	}
	fclose(file);

	printf("case            = %s\n",case_path);
	printf("geometry        = batch=%u slots=%u vocab=%u hidden=%u rank=%u top_k=%u\n",
		batch_count,slot_count,vocab_count,hidden_dimension,rank,top_k);
	printf("host path       = SparkQwen36DsparkSelectorEmit (the module's own inline sequence)\n");
	printf("anchor row      = 1 poison row of %.0f prepended per batch; a dropped +H offset cannot pass\n",
		(double)SPARK_DFLASH2_POISON_VALUE);
	printf("workspace bytes = %llu at slots=%u K=%u rank=%u\n",
		(unsigned long long)SparkQwen36DsparkSelectorWorkspaceBytes(slot_count,top_k,rank),slot_count,top_k,rank);

	/* The block hidden the module hands over: ONE anchor row, then the case's
	 * mask rows for this batch. */
	block_elements = (uint64_t)(slot_count + 1u) * hidden_dimension;
	SPARK_DFLASH2_CUDA(cudaMalloc(&block_hidden,(size_t)block_elements * 2ull));
	SPARK_DFLASH2_CUDA(cudaMalloc(&head_device,(size_t)vocab_count * hidden_dimension * 2ull));
	SPARK_DFLASH2_CUDA(cudaMalloc(&projection_device,(size_t)rank * hidden_dimension * 2ull));
	SPARK_DFLASH2_CUDA(cudaMalloc(&predecessor_device,(size_t)vocab_count * rank * 2ull));
	SPARK_DFLASH2_CUDA(cudaMalloc(&successor_device,(size_t)vocab_count * rank * 2ull));
	SPARK_DFLASH2_CUDA(cudaMalloc((void **)&anchor_device,sizeof(uint32_t)));
	SPARK_DFLASH2_CUDA(cudaMalloc((void **)&workspace.chunk_keys,(size_t)SparkQwen36DsparkHeadTopKChunkKeyCount(slot_count,top_k) * sizeof(uint64_t)));
	SPARK_DFLASH2_CUDA(cudaMalloc((void **)&workspace.candidate_ids,(size_t)slot_count * top_k * sizeof(uint32_t)));
	SPARK_DFLASH2_CUDA(cudaMalloc((void **)&workspace.candidate_scores,(size_t)slot_count * top_k * sizeof(float)));
	SPARK_DFLASH2_CUDA(cudaMalloc(&workspace.context_gate_bf16,(size_t)slot_count * rank * 2ull));
	SPARK_DFLASH2_CUDA(cudaMalloc((void **)&workspace.edges_f32,(size_t)slot_count * top_k * top_k * sizeof(float)));
	SPARK_DFLASH2_CUDA(cudaMalloc((void **)&workspace.draft_token_ids,(size_t)slot_count * sizeof(uint32_t)));
	if ( SparkDflash2FillDeviceBf16(head_device,(uint64_t)vocab_count * hidden_dimension,SPARK_DFLASH2_STREAM_HEAD,seed,weight_span,(uint64_t)SPARK_DFLASH2_HEAD_CHUNK_ROWS * hidden_dimension,hidden_dimension,head_row_pairing,0ull) != 0 ||
	     SparkDflash2FillDeviceBf16(projection_device,(uint64_t)rank * hidden_dimension,SPARK_DFLASH2_STREAM_PROJECTION,seed,weight_span,1ull << 20u,0ull,0u,0ull) != 0 ||
	     SparkDflash2FillDeviceBf16(predecessor_device,(uint64_t)vocab_count * rank,SPARK_DFLASH2_STREAM_PREDECESSOR,seed,codebook_span,1ull << 22u,0ull,0u,0ull) != 0 ||
	     SparkDflash2FillDeviceBf16(successor_device,(uint64_t)vocab_count * rank,SPARK_DFLASH2_STREAM_SUCCESSOR,seed,codebook_span,1ull << 22u,0ull,0u,0ull) != 0 )
	{
		fprintf(stderr,"input upload failed\n");
		return(3);
	}
	memset(&head_view,0,sizeof(head_view));
	head_view.abi_version = SPARK_QWEN36_RESIDENT_DECODE_STAGE_LINEAR_VIEW_ABI_VERSION;
	head_view.weight_format = SPARK_QWEN36_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16;
	head_view.input_dimension = hidden_dimension;
	head_view.output_dimension = vocab_count;
	head_view.weight_payload = head_device;
	head_view.weight_payload_bytes = (uint64_t)vocab_count * hidden_dimension * 2ull;

	/* One extra guard word past the draft buffer, as the adapter's array would
	 * have: the DSpark loop's B-th write landed exactly there. */
	host_drafts = (uint32_t *)malloc((size_t)(slot_count + 1u) * sizeof(uint32_t));
	if ( host_drafts == 0 )
		return(3);

	printf("\n-- host emit per batch (anchor from device, drafts D2H)\n");
	for (batch = 0u; batch < batch_count; batch++)
	{
		/* Poison the anchor row, then this batch's mask rows from the PRNG. */
		if ( SparkDflash2FillDeviceBf16(block_hidden,block_elements,SPARK_DFLASH2_STREAM_HIDDEN,seed,hidden_span,1ull << 20u,0ull,0u,(uint64_t)hidden_dimension) != 0 )
			return(3);
		if ( batch != 0u )
		{
			/* rows for batch b start at b * slot_count * hidden in the PRNG
			 * stream; refill the mask rows with that offset. */
			uint16_t *staging = (uint16_t *)malloc((size_t)slot_count * hidden_dimension * sizeof(uint16_t));
			if ( staging == 0 )
				return(3);
			for (index = 0ull; index < (uint64_t)slot_count * hidden_dimension; index++)
				staging[index] = SparkDflash2FloatToBf16(SparkDflash2SmallInteger(((uint64_t)batch * slot_count * hidden_dimension) + index,SPARK_DFLASH2_STREAM_HIDDEN,seed,hidden_span));
			SPARK_DFLASH2_CUDA(cudaMemcpy((uint16_t *)block_hidden + hidden_dimension,staging,(size_t)slot_count * hidden_dimension * sizeof(uint16_t),cudaMemcpyHostToDevice));
			free(staging);
		}
		SPARK_DFLASH2_CUDA(cudaMemcpy(anchor_device,&anchor_ids[batch],sizeof(uint32_t),cudaMemcpyHostToDevice));
		for (slot_index = 0u; slot_index <= slot_count; slot_index++)
			host_drafts[slot_index] = SPARK_DFLASH2_GUARD_WORD;
		SPARK_DFLASH2_CUDA(SparkQwen36DsparkSelectorEmit(0,&head_view,
			(const void *)((const uint16_t *)block_hidden + hidden_dimension),
			projection_device,predecessor_device,successor_device,anchor_device,&workspace,
			slot_count,top_k,rank,hidden_dimension,host_drafts));
		SPARK_DFLASH2_CUDA(cudaDeviceSynchronize());
		mismatches = 0u;
		for (slot_index = 0u; slot_index < slot_count; slot_index++)
			if ( host_drafts[slot_index] != expected_drafts[((uint64_t)batch * slot_count) + slot_index] )
				mismatches++;
		printf("batch %u draft ids                     %u / %u equal",batch,slot_count - mismatches,slot_count);
		if ( mismatches != 0u )
		{
			printf("  FAIL actual[0]=%u expected[0]=%u",host_drafts[0],expected_drafts[(uint64_t)batch * slot_count]);
			spark_dflash2_failures++;
		}
		printf("\n");
		SparkDflash2Check("guard word past the draft buffer",host_drafts[slot_count] == SPARK_DFLASH2_GUARD_WORD ? 1u : 0u,
			"the DSpark loop wrote B ids into a B-1 buffer");
	}
	SparkDflash2Check("emitted width == adapter's count",
		slot_count == SPARK_QWEN36_RESIDENT_DECODE_STAGE_DSPARK_BLOCK_SIZE ? 1u : 0u,
		slot_count == SPARK_QWEN36_RESIDENT_DECODE_STAGE_DSPARK_BLOCK_SIZE ? "(7 = block 8 minus the anchor)" : "case slots != the adapter's draft count");

	printf("\nresult          = %s (%u failing checks)\n",spark_dflash2_failures == 0u ? "PARITY" : "MISMATCH",spark_dflash2_failures);
	return(spark_dflash2_failures == 0u ? 0 : 1);
}
