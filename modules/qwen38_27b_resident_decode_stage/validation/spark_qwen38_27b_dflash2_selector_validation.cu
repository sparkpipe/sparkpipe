#include <cuda_runtime.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_qwen38_27b_model.h"
#include "sparkpipe/spark_qwen38_27b_resident_decode_stage_firmware.h"
#include "spark_qwen38_27b_dspark_format.h"

/*
 * DFlash2 candidate-selector parity validator (adoption items W4 and W3).
 *
 * Runs the two new kernel paths through their PRODUCTION launchers on exactly
 * the inputs tools/qwen36_dflash2_selector_case.py generated, and diffs against
 * the expected outputs that script dumped from the numpy oracles in
 * tools/qwen36_dspark_reference.py (_score_edges and _greedy_walk, the exact
 * ports of vLLM PR #52816, plus the reference's stable / id-ascending top-K).
 *
 * Inputs are never transferred: the case file carries geometry, seed and the
 * EXPECTED outputs, and both sides regenerate every input tensor from the same
 * counter-based splitmix64. That keeps a 2.37 GiB BF16 head out of the file
 * while leaving the comparison exact.
 *
 * Every generated value is a small integer (hidden in [-span,span], weights in
 * [-1,0,1]) so each dot product is an exact fp32 integer regardless of
 * summation order: numpy's BLAS order and the kernels' warp-tree / ascending
 * orders MUST produce identical bits, and every comparison below is therefore
 * bit-for-bit (==), not a tolerance. The BF16 truncation points still round,
 * including exact ties-to-even, so the contract's rounding is exercised.
 *
 * FIVE CHECK GROUPS. Three compare against the oracle:
 *   W4          - kernel top-k ids and scores vs the oracle's top-k
 *   W3 isolated - the oracle's ids/unary in, kernel gate/lattice/walk out
 *   fused       - the kernel's OWN top-k feeding the selector, drafts vs oracle
 * and two PROVE the rules a comment cannot:
 *   truncation-before-selection - the case also carries the top-k that
 *     selecting on UNtruncated fp32 logits would give (the oracle truncates the
 *     whole row first). The kernel must differ from that alternative in exactly
 *     the positions the case recorded, so a kernel that truncated AFTER
 *     selecting fails here even though its scores would still look right.
 *   strict-greater first-max - the case also carries the walk a non-strict (>=,
 *     last-max) implementation would take; the kernel must differ from that
 *     alternative in exactly the positions the case recorded.
 * A case with zero recorded differences cannot prove its rule; that is reported
 * as NOT DISCRIMINATING, and on the tie scale (which exists to discriminate) it
 * is a failure - the harness refuses to pass vacuously.
 */

#define SPARK_DFLASH2_CASE_MAGIC "Q6DF2CS2"
#define SPARK_DFLASH2_HEAD_CHUNK_ROWS 4096u

extern "C" uint64_t SparkQwen38_27bDsparkHeadTopKChunkKeyCount(uint32_t row_count, uint32_t top_k);
extern "C" cudaError_t SparkQwen38_27bLaunchDsparkHeadTopK(cudaStream_t stream, const SparkQwen38_27bLinearView *head, const void *hidden_bf16, uint64_t *chunk_keys, uint32_t *top_candidate_ids, float *top_scores_f32, void *top_scores_bf16, uint32_t row_count, uint32_t candidate_offset, uint32_t top_k);
extern "C" cudaError_t SparkQwen38_27bLaunchDsparkSelectorProject(cudaStream_t stream, const void *hidden_bf16, const void *hidden_projection_bf16, void *context_gate_bf16, uint32_t row_count, uint32_t rank, uint32_t hidden_dimension);
extern "C" cudaError_t SparkQwen38_27bLaunchDsparkSelectorEdges(cudaStream_t stream, const void *predecessor_bf16, const void *successor_bf16, const uint32_t *candidate_ids, const uint32_t *anchor_token_ids, const float *unary_f32, const void *context_gate_bf16, float *edges_f32, uint32_t batch_count, uint32_t slot_count, uint32_t top_k, uint32_t rank);
extern "C" cudaError_t SparkQwen38_27bLaunchDsparkSelectorWalk(cudaStream_t stream, const float *edges_f32, const uint32_t *candidate_ids, uint32_t *draft_token_ids, uint32_t *draft_candidate_slots, uint32_t batch_count, uint32_t slot_count, uint32_t top_k);
extern "C" cudaError_t SparkQwen38_27bLaunchDsparkSelector(cudaStream_t stream, const void *hidden_bf16, const void *hidden_projection_bf16, const void *predecessor_bf16, const void *successor_bf16, const uint32_t *candidate_ids, const uint32_t *anchor_token_ids, const float *unary_f32, void *context_gate_bf16, float *edges_f32, uint32_t *draft_token_ids, uint32_t *draft_candidate_slots, uint32_t batch_count, uint32_t slot_count, uint32_t top_k, uint32_t rank, uint32_t hidden_dimension);

typedef struct SparkDflash2Case
{
	uint32_t batch_count;
	uint32_t slot_count;
	uint32_t vocab_count;
	uint32_t hidden_dimension;
	uint32_t rank;
	uint32_t top_k;
	uint32_t hidden_span;
	uint32_t weight_span;
	uint32_t codebook_span;
	uint32_t head_row_pairing;
	uint32_t truncation_discriminator;
	uint32_t walk_tie_rows;
	uint32_t walk_lastmax_diffs;
	uint64_t seed;
	uint32_t *anchor_token_ids;
	uint32_t *top_candidate_ids;
	uint32_t *untruncated_top_ids;
	float *unary_f32;
	uint16_t *context_gate_bf16;
	float *edges_f32;
	uint32_t *draft_token_ids;
	uint32_t *draft_candidate_slots;
	uint32_t *lastmax_token_ids;
	uint32_t *lastmax_candidate_slots;
} SparkDflash2Case;

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

/* Numpy f32_to_bf16 in the reference: round to nearest even. */
static uint16_t SparkDflash2FloatToBf16(float value)
{
	uint32_t bits,lsb;
	memcpy(&bits,&value,sizeof(bits));
	lsb = (bits >> 16u) & 1u;
	bits += 0x7fffu + lsb;
	return((uint16_t)(bits >> 16u));
}

static float SparkDflash2Bf16ToFloat(uint16_t value)
{
	uint32_t bits = (uint32_t)value << 16u;
	float output;
	memcpy(&output,&bits,sizeof(output));
	return(output);
}

static int SparkDflash2ReadCase(const char *path, SparkDflash2Case *output)
{
	FILE *file;
	char magic[8];
	uint32_t header[13];
	uint64_t rows,lattice,gate_elements,candidates;
	memset(output,0,sizeof(*output));
	file = fopen(path,"rb");
	if ( file == 0 )
	{
		fprintf(stderr,"cannot open case %s\n",path);
		return(-1);
	}
	if ( fread(magic,1u,sizeof(magic),file) != sizeof(magic) || memcmp(magic,SPARK_DFLASH2_CASE_MAGIC,sizeof(magic)) != 0 )
	{
		fprintf(stderr,"case magic mismatch (regenerate with tools/qwen36_dflash2_selector_case.py)\n");
		fclose(file);
		return(-1);
	}
	if ( fread(header,sizeof(uint32_t),13u,file) != 13u || fread(&output->seed,sizeof(uint64_t),1u,file) != 1u )
	{
		fclose(file);
		return(-1);
	}
	output->batch_count = header[0];
	output->slot_count = header[1];
	output->vocab_count = header[2];
	output->hidden_dimension = header[3];
	output->rank = header[4];
	output->top_k = header[5];
	output->hidden_span = header[6];
	output->weight_span = header[7];
	output->codebook_span = header[8];
	output->head_row_pairing = header[9];
	output->truncation_discriminator = header[10];
	output->walk_tie_rows = header[11];
	output->walk_lastmax_diffs = header[12];
	rows = (uint64_t)output->batch_count * output->slot_count;
	candidates = rows * output->top_k;
	lattice = candidates * output->top_k;
	gate_elements = rows * output->rank;
	output->anchor_token_ids = (uint32_t *)malloc(output->batch_count * sizeof(uint32_t));
	output->top_candidate_ids = (uint32_t *)malloc(candidates * sizeof(uint32_t));
	output->untruncated_top_ids = (uint32_t *)malloc(candidates * sizeof(uint32_t));
	output->unary_f32 = (float *)malloc(candidates * sizeof(float));
	output->context_gate_bf16 = (uint16_t *)malloc(gate_elements * sizeof(uint16_t));
	output->edges_f32 = (float *)malloc(lattice * sizeof(float));
	output->draft_token_ids = (uint32_t *)malloc(rows * sizeof(uint32_t));
	output->draft_candidate_slots = (uint32_t *)malloc(rows * sizeof(uint32_t));
	output->lastmax_token_ids = (uint32_t *)malloc(rows * sizeof(uint32_t));
	output->lastmax_candidate_slots = (uint32_t *)malloc(rows * sizeof(uint32_t));
	if ( output->anchor_token_ids == 0 || output->top_candidate_ids == 0 || output->untruncated_top_ids == 0 ||
	     output->unary_f32 == 0 || output->context_gate_bf16 == 0 || output->edges_f32 == 0 ||
	     output->draft_token_ids == 0 || output->draft_candidate_slots == 0 ||
	     output->lastmax_token_ids == 0 || output->lastmax_candidate_slots == 0 )
	{
		fclose(file);
		return(-1);
	}
	if ( fread(output->anchor_token_ids,sizeof(uint32_t),output->batch_count,file) != output->batch_count ||
	     fread(output->top_candidate_ids,sizeof(uint32_t),candidates,file) != candidates ||
	     fread(output->untruncated_top_ids,sizeof(uint32_t),candidates,file) != candidates ||
	     fread(output->unary_f32,sizeof(float),candidates,file) != candidates ||
	     fread(output->context_gate_bf16,sizeof(uint16_t),gate_elements,file) != gate_elements ||
	     fread(output->edges_f32,sizeof(float),lattice,file) != lattice ||
	     fread(output->draft_token_ids,sizeof(uint32_t),rows,file) != rows ||
	     fread(output->draft_candidate_slots,sizeof(uint32_t),rows,file) != rows ||
	     fread(output->lastmax_token_ids,sizeof(uint32_t),rows,file) != rows ||
	     fread(output->lastmax_candidate_slots,sizeof(uint32_t),rows,file) != rows )
	{
		fprintf(stderr,"case payload is short\n");
		fclose(file);
		return(-1);
	}
	if ( fgetc(file) != EOF )
	{
		fprintf(stderr,"case payload has trailing bytes\n");
		fclose(file);
		return(-1);
	}
	fclose(file);
	return(0);
}

/* Fill a device BF16 tensor from the shared PRNG, one host chunk at a time so
 * a 248320 x 5120 head never needs a full host mirror. row_stride != 0 with
 * pairing folds every odd row onto the even row below it, exactly as the
 * generator's head_row_index does, which makes adjacent candidates tie. */
static int SparkDflash2FillDeviceBf16(void *device, uint64_t element_count, uint64_t stream_seed, uint64_t seed, uint32_t span, uint64_t chunk_elements, uint64_t row_stride, uint32_t pairing, const char *label)
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
			fprintf(stderr,"%s upload failed: %s\n",label,cudaGetErrorString(error));
			free(staging);
			return(-1);
		}
	}
	free(staging);
	return(0);
}

static uint64_t SparkDflash2CountDifferences(const uint32_t *left, const uint32_t *right, uint64_t count)
{
	uint64_t index,differences = 0ull;
	for (index = 0ull; index < count; index++)
		if ( left[index] != right[index] )
			differences++;
	return(differences);
}

static void SparkDflash2ReportU32(const char *label, const uint32_t *actual, const uint32_t *expected, uint64_t count)
{
	uint64_t index,mismatches = 0ull,first_mismatch = 0ull;
	for (index = 0ull; index < count; index++)
		if ( actual[index] != expected[index] )
		{
			if ( mismatches == 0ull )
				first_mismatch = index;
			mismatches++;
		}
	printf("%-34s %8llu / %8llu equal",label,(unsigned long long)(count - mismatches),(unsigned long long)count);
	if ( mismatches != 0ull )
		printf("  FAIL first=%llu actual=%u expected=%u",(unsigned long long)first_mismatch,actual[first_mismatch],expected[first_mismatch]);
	printf("\n");
	if ( mismatches != 0ull )
		spark_dflash2_failures++;
}

static void SparkDflash2ReportF32(const char *label, const float *actual, const float *expected, uint64_t count)
{
	uint64_t index,mismatches = 0ull,first_mismatch = 0ull;
	double worst = 0.0,delta;
	for (index = 0ull; index < count; index++)
	{
		delta = fabs((double)actual[index] - (double)expected[index]);
		if ( delta > worst )
			worst = delta;
		if ( memcmp(&actual[index],&expected[index],sizeof(float)) != 0 )
		{
			if ( mismatches == 0ull )
				first_mismatch = index;
			mismatches++;
		}
	}
	printf("%-34s %8llu / %8llu bitwise equal  max|delta|=%.9g",label,(unsigned long long)(count - mismatches),(unsigned long long)count,worst);
	if ( mismatches != 0ull )
		printf("  FAIL first=%llu actual=%.9g expected=%.9g",(unsigned long long)first_mismatch,(double)actual[first_mismatch],(double)expected[first_mismatch]);
	printf("\n");
	if ( mismatches != 0ull )
		spark_dflash2_failures++;
}

static void SparkDflash2ReportBf16(const char *label, const uint16_t *actual, const uint16_t *expected, uint64_t count)
{
	uint64_t index,mismatches = 0ull,first_mismatch = 0ull;
	for (index = 0ull; index < count; index++)
		if ( actual[index] != expected[index] )
		{
			if ( mismatches == 0ull )
				first_mismatch = index;
			mismatches++;
		}
	printf("%-34s %8llu / %8llu bitwise equal",label,(unsigned long long)(count - mismatches),(unsigned long long)count);
	if ( mismatches != 0ull )
		printf("  FAIL first=%llu actual=%.9g expected=%.9g",(unsigned long long)first_mismatch,
			(double)SparkDflash2Bf16ToFloat(actual[first_mismatch]),(double)SparkDflash2Bf16ToFloat(expected[first_mismatch]));
	printf("\n");
	if ( mismatches != 0ull )
		spark_dflash2_failures++;
}

/* A rule is PROVEN when the kernel differs from the wrong-rule alternative in
 * exactly the positions the oracle-side case recorded. Zero recorded
 * differences means this case cannot prove the rule; require_discriminating
 * turns that into a failure for the scale whose purpose is to discriminate. */
static void SparkDflash2ReportRule(const char *label, const uint32_t *actual, const uint32_t *alternative, uint64_t count, uint32_t expected_differences, uint32_t require_discriminating)
{
	uint64_t observed = SparkDflash2CountDifferences(actual,alternative,count);
	if ( expected_differences == 0u )
	{
		printf("%-34s NOT DISCRIMINATING at this scale (case records 0 differences)%s\n",label,
			require_discriminating != 0u ? "  FAIL" : "");
		if ( require_discriminating != 0u )
			spark_dflash2_failures++;
		return;
	}
	printf("%-34s PROVEN: kernel differs from the wrong rule in %llu / %llu positions (case says %u)",
		label,(unsigned long long)observed,(unsigned long long)count,expected_differences);
	if ( observed != (uint64_t)expected_differences )
	{
		printf("  FAIL");
		spark_dflash2_failures++;
	}
	printf("\n");
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
	SparkDflash2Case testcase;
	SparkQwen38_27bLinearView head_view;
	const char *case_path = argc > 1 ? argv[1] : "build/dflash2/dflash2_selector_case.bin";
	uint32_t require_discriminating = argc > 2 && strcmp(argv[2],"--require-discriminating") == 0 ? 1u : 0u;
	uint64_t rows,lattice,gate_elements,candidates,chunk_keys_count;
	void *hidden_device = 0,*head_device = 0,*projection_device = 0,*predecessor_device = 0,*successor_device = 0,*context_gate_device = 0;
	uint64_t *chunk_keys_device = 0;
	uint32_t *top_ids_device = 0,*oracle_ids_device = 0,*anchor_device = 0,*draft_ids_device = 0,*draft_slots_device = 0;
	float *top_scores_device = 0,*oracle_unary_device = 0,*edges_device = 0;
	uint32_t *host_ids,*host_drafts,*host_slots;
	uint16_t *host_gate;
	float *host_scores,*host_edges;
	if ( SparkDflash2ReadCase(case_path,&testcase) != 0 )
		return(2);
	rows = (uint64_t)testcase.batch_count * testcase.slot_count;
	candidates = rows * testcase.top_k;
	lattice = candidates * testcase.top_k;
	gate_elements = rows * testcase.rank;
	chunk_keys_count = SparkQwen38_27bDsparkHeadTopKChunkKeyCount((uint32_t)rows,testcase.top_k);
	printf("case            = %s\n",case_path);
	printf("geometry        = batch=%u slots=%u vocab=%u hidden=%u rank=%u top_k=%u seed=0x%llx\n",
		testcase.batch_count,testcase.slot_count,testcase.vocab_count,testcase.hidden_dimension,
		testcase.rank,testcase.top_k,(unsigned long long)testcase.seed);
	printf("spans           = hidden=%u weight=%u codebook=%u head_row_pairing=%u\n",
		testcase.hidden_span,testcase.weight_span,testcase.codebook_span,testcase.head_row_pairing);
	printf("discriminators  = truncation %u ids, walk %u columns (%u walked rows with a tied maximum)\n",
		testcase.truncation_discriminator,testcase.walk_lastmax_diffs,testcase.walk_tie_rows);
	printf("head bytes      = %.3f GiB (BF16, dense, read through SparkQwen38_27bLinearView)\n",
		(double)((uint64_t)testcase.vocab_count * testcase.hidden_dimension * 2ull) / 1073741824.0);

	SPARK_DFLASH2_CUDA(cudaMalloc(&hidden_device,(size_t)rows * testcase.hidden_dimension * 2ull));
	SPARK_DFLASH2_CUDA(cudaMalloc(&head_device,(size_t)testcase.vocab_count * testcase.hidden_dimension * 2ull));
	SPARK_DFLASH2_CUDA(cudaMalloc(&projection_device,(size_t)testcase.rank * testcase.hidden_dimension * 2ull));
	SPARK_DFLASH2_CUDA(cudaMalloc(&predecessor_device,(size_t)testcase.vocab_count * testcase.rank * 2ull));
	SPARK_DFLASH2_CUDA(cudaMalloc(&successor_device,(size_t)testcase.vocab_count * testcase.rank * 2ull));
	SPARK_DFLASH2_CUDA(cudaMalloc(&context_gate_device,(size_t)gate_elements * 2ull));
	SPARK_DFLASH2_CUDA(cudaMalloc((void **)&chunk_keys_device,(size_t)chunk_keys_count * sizeof(uint64_t)));
	SPARK_DFLASH2_CUDA(cudaMalloc((void **)&top_ids_device,(size_t)candidates * sizeof(uint32_t)));
	SPARK_DFLASH2_CUDA(cudaMalloc((void **)&oracle_ids_device,(size_t)candidates * sizeof(uint32_t)));
	SPARK_DFLASH2_CUDA(cudaMalloc((void **)&top_scores_device,(size_t)candidates * sizeof(float)));
	SPARK_DFLASH2_CUDA(cudaMalloc((void **)&oracle_unary_device,(size_t)candidates * sizeof(float)));
	SPARK_DFLASH2_CUDA(cudaMalloc((void **)&anchor_device,(size_t)testcase.batch_count * sizeof(uint32_t)));
	SPARK_DFLASH2_CUDA(cudaMalloc((void **)&edges_device,(size_t)lattice * sizeof(float)));
	SPARK_DFLASH2_CUDA(cudaMalloc((void **)&draft_ids_device,(size_t)rows * sizeof(uint32_t)));
	SPARK_DFLASH2_CUDA(cudaMalloc((void **)&draft_slots_device,(size_t)rows * sizeof(uint32_t)));

	if ( SparkDflash2FillDeviceBf16(hidden_device,rows * testcase.hidden_dimension,SPARK_DFLASH2_STREAM_HIDDEN,testcase.seed,testcase.hidden_span,1ull << 20u,0ull,0u,"hidden") != 0 )
		return(3);
	if ( SparkDflash2FillDeviceBf16(head_device,(uint64_t)testcase.vocab_count * testcase.hidden_dimension,SPARK_DFLASH2_STREAM_HEAD,testcase.seed,testcase.weight_span,(uint64_t)SPARK_DFLASH2_HEAD_CHUNK_ROWS * testcase.hidden_dimension,testcase.hidden_dimension,testcase.head_row_pairing,"head") != 0 )
		return(3);
	if ( SparkDflash2FillDeviceBf16(projection_device,(uint64_t)testcase.rank * testcase.hidden_dimension,SPARK_DFLASH2_STREAM_PROJECTION,testcase.seed,testcase.weight_span,1ull << 20u,0ull,0u,"projection") != 0 )
		return(3);
	if ( SparkDflash2FillDeviceBf16(predecessor_device,(uint64_t)testcase.vocab_count * testcase.rank,SPARK_DFLASH2_STREAM_PREDECESSOR,testcase.seed,testcase.codebook_span,1ull << 22u,0ull,0u,"predecessor") != 0 )
		return(3);
	if ( SparkDflash2FillDeviceBf16(successor_device,(uint64_t)testcase.vocab_count * testcase.rank,SPARK_DFLASH2_STREAM_SUCCESSOR,testcase.seed,testcase.codebook_span,1ull << 22u,0ull,0u,"successor") != 0 )
		return(3);
	SPARK_DFLASH2_CUDA(cudaMemcpy(anchor_device,testcase.anchor_token_ids,(size_t)testcase.batch_count * sizeof(uint32_t),cudaMemcpyHostToDevice));
	SPARK_DFLASH2_CUDA(cudaMemcpy(oracle_ids_device,testcase.top_candidate_ids,(size_t)candidates * sizeof(uint32_t),cudaMemcpyHostToDevice));
	SPARK_DFLASH2_CUDA(cudaMemcpy(oracle_unary_device,testcase.unary_f32,(size_t)candidates * sizeof(float),cudaMemcpyHostToDevice));

	/* The module's own head view: dense BF16 lm_head, exactly as the drafter
	 * host path builds it around state->lm_head_weight_bf16. */
	memset(&head_view,0,sizeof(head_view));
	head_view.abi_version = SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_LINEAR_VIEW_ABI_VERSION;
	head_view.weight_format = SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16;
	head_view.input_dimension = testcase.hidden_dimension;
	head_view.output_dimension = testcase.vocab_count;
	head_view.weight_payload = head_device;
	head_view.weight_payload_bytes = (uint64_t)testcase.vocab_count * testcase.hidden_dimension * 2ull;

	host_ids = (uint32_t *)malloc((size_t)candidates * sizeof(uint32_t));
	host_scores = (float *)malloc((size_t)candidates * sizeof(float));
	host_gate = (uint16_t *)malloc((size_t)gate_elements * sizeof(uint16_t));
	host_edges = (float *)malloc((size_t)lattice * sizeof(float));
	host_drafts = (uint32_t *)malloc((size_t)rows * sizeof(uint32_t));
	host_slots = (uint32_t *)malloc((size_t)rows * sizeof(uint32_t));
	if ( host_ids == 0 || host_scores == 0 || host_gate == 0 || host_edges == 0 || host_drafts == 0 || host_slots == 0 )
		return(3);

	/* ---- W4: top-k over the vocabulary ---- */
	SPARK_DFLASH2_CUDA(SparkQwen38_27bLaunchDsparkHeadTopK(0,&head_view,hidden_device,chunk_keys_device,top_ids_device,top_scores_device,0,(uint32_t)rows,0u,testcase.top_k));
	SPARK_DFLASH2_CUDA(cudaDeviceSynchronize());
	SPARK_DFLASH2_CUDA(cudaMemcpy(host_ids,top_ids_device,(size_t)candidates * sizeof(uint32_t),cudaMemcpyDeviceToHost));
	SPARK_DFLASH2_CUDA(cudaMemcpy(host_scores,top_scores_device,(size_t)candidates * sizeof(float),cudaMemcpyDeviceToHost));
	printf("\n-- W4 top-%u over %u candidates, %llu slots\n",testcase.top_k,testcase.vocab_count,(unsigned long long)rows);
	SparkDflash2ReportU32("w4 top ids",host_ids,testcase.top_candidate_ids,candidates);
	SparkDflash2ReportF32("w4 top scores (unary)",host_scores,testcase.unary_f32,candidates);
	SparkDflash2ReportRule("w4 truncation-before-select",host_ids,testcase.untruncated_top_ids,candidates,testcase.truncation_discriminator,require_discriminating);

	/* ---- W3 isolated: the oracle's candidates in, gate/lattice/walk out ---- */
	SPARK_DFLASH2_CUDA(SparkQwen38_27bLaunchDsparkSelectorProject(0,hidden_device,projection_device,context_gate_device,(uint32_t)rows,testcase.rank,testcase.hidden_dimension));
	SPARK_DFLASH2_CUDA(SparkQwen38_27bLaunchDsparkSelectorEdges(0,predecessor_device,successor_device,oracle_ids_device,anchor_device,oracle_unary_device,context_gate_device,edges_device,testcase.batch_count,testcase.slot_count,testcase.top_k,testcase.rank));
	SPARK_DFLASH2_CUDA(SparkQwen38_27bLaunchDsparkSelectorWalk(0,edges_device,oracle_ids_device,draft_ids_device,draft_slots_device,testcase.batch_count,testcase.slot_count,testcase.top_k));
	SPARK_DFLASH2_CUDA(cudaDeviceSynchronize());
	SPARK_DFLASH2_CUDA(cudaMemcpy(host_gate,context_gate_device,(size_t)gate_elements * sizeof(uint16_t),cudaMemcpyDeviceToHost));
	SPARK_DFLASH2_CUDA(cudaMemcpy(host_edges,edges_device,(size_t)lattice * sizeof(float),cudaMemcpyDeviceToHost));
	SPARK_DFLASH2_CUDA(cudaMemcpy(host_drafts,draft_ids_device,(size_t)rows * sizeof(uint32_t),cudaMemcpyDeviceToHost));
	SPARK_DFLASH2_CUDA(cudaMemcpy(host_slots,draft_slots_device,(size_t)rows * sizeof(uint32_t),cudaMemcpyDeviceToHost));
	printf("\n-- W3 selector on the ORACLE's candidates (rank=%u, K=%u)\n",testcase.rank,testcase.top_k);
	SparkDflash2ReportBf16("w3 context gate H",host_gate,testcase.context_gate_bf16,gate_elements);
	SparkDflash2ReportF32("w3 edge lattice",host_edges,testcase.edges_f32,lattice);
	SparkDflash2ReportU32("w3 walk draft ids",host_drafts,testcase.draft_token_ids,rows);
	SparkDflash2ReportU32("w3 walk lattice columns",host_slots,testcase.draft_candidate_slots,rows);
	SparkDflash2ReportRule("w3 walk strict-greater first-max",host_slots,testcase.lastmax_candidate_slots,rows,testcase.walk_lastmax_diffs,require_discriminating);

	/* ---- fused: W4's own top-k feeds W3 ---- */
	SPARK_DFLASH2_CUDA(cudaMemset(edges_device,0,(size_t)lattice * sizeof(float)));
	SPARK_DFLASH2_CUDA(cudaMemset(draft_ids_device,0,(size_t)rows * sizeof(uint32_t)));
	SPARK_DFLASH2_CUDA(SparkQwen38_27bLaunchDsparkSelector(0,hidden_device,projection_device,predecessor_device,successor_device,top_ids_device,anchor_device,top_scores_device,context_gate_device,edges_device,draft_ids_device,draft_slots_device,testcase.batch_count,testcase.slot_count,testcase.top_k,testcase.rank,testcase.hidden_dimension));
	SPARK_DFLASH2_CUDA(cudaDeviceSynchronize());
	SPARK_DFLASH2_CUDA(cudaMemcpy(host_edges,edges_device,(size_t)lattice * sizeof(float),cudaMemcpyDeviceToHost));
	SPARK_DFLASH2_CUDA(cudaMemcpy(host_drafts,draft_ids_device,(size_t)rows * sizeof(uint32_t),cudaMemcpyDeviceToHost));
	SPARK_DFLASH2_CUDA(cudaMemcpy(host_slots,draft_slots_device,(size_t)rows * sizeof(uint32_t),cudaMemcpyDeviceToHost));
	printf("\n-- fused W4 -> W3 (the drafter's actual path)\n");
	SparkDflash2ReportF32("fused edge lattice",host_edges,testcase.edges_f32,lattice);
	SparkDflash2ReportU32("fused draft ids",host_drafts,testcase.draft_token_ids,rows);
	SparkDflash2ReportU32("fused lattice columns",host_slots,testcase.draft_candidate_slots,rows);

	printf("\nresult          = %s (%u failing comparisons)\n",spark_dflash2_failures == 0u ? "PARITY" : "MISMATCH",spark_dflash2_failures);
	return(spark_dflash2_failures == 0u ? 0 : 1);
}
