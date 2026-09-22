// The KDA recurrence's run contract, executed on CPU: "a run of T is
// bit-identical to T decode calls" - LmDeltaRuleKernel states that as a
// gate, and this harness executes it for the shapes the K3 prefill port
// (the adapter's run-prefix derivation) produces:
//
//   S1  one 8-row run vs 8 sequential one-row waves (outputs + state slot)
//   S2  an explicit runs-of-one prefix vs the NULL-prefix decode reading
//   S3  a mixed wave (prefill span + lone rows, three slots) vs sequential
//   S4  the causal conv window carry over a run vs sequential
//
// THREADS=1 is the blessed host regime: the strided partial loops ARE the
// serial walk, so both legs of every comparison run the same summation
// order and only the RUN STRUCTURE differs. The GPU tier3/tier4 cell
// through the live layer chain remains the serving gate; this pins the
// kernel contract the wiring is built on.

#include "tests/host_cuda/lm_host_cuda.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

LmHostDim3 blockIdx, threadIdx, blockDim, gridDim;

/* The shim backs `extern __shared__` with these fixed buffers; the delta
 * rule's state slab is the one this harness's kernels need. */
float state_s[LM_HOST_SHARED_BYTES / sizeof(float)];

#include "inference/kernels/mma.cuh"
#undef LM_WARP_LANES
#define LM_WARP_LANES LM_HOST_WARP_LANES
#include "inference/kernels/dtype.cuh"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/linear_attn.cuh"

#define K_THREADS 1u
#define K_KEY_DIM 8u
#define K_VALUE_DIM 8u
#define K_KEY_HEADS 2u
#define K_VPK 2u
#define K_CHANNELS 6u
#define K_KERNEL 4u
#define K_ROWS 12u
#define K_SLOTS 8u
typedef float KState;
/* slot_bytes is the WHOLE slot: the kernel slices it per key head
 * internally (head offset = KEY_DIM * VALUE_DIM * sizeof(State)). */
#define K_SLOT_BYTES (K_KEY_HEADS * K_KEY_DIM * K_VALUE_DIM * sizeof(KState))

static uint32_t lcg_state = 0x12345678u;

static float lcg_float(void)
{
	lcg_state = lcg_state * 1664525u + 1013904223u;
	return (((lcg_state >> 8) & 0xffffu) - 32768.0f) / 32768.0f;
}

static uint16_t lcg_bf16(void)
{
	return LmFloatToBf16(lcg_float());
}

static void *xmalloc(size_t bytes)
{
	void *block = malloc(bytes);
	if ( block == 0 )
	{
		fprintf(stderr, "oom\n");
		exit(2);
	}
	return block;
}

static void init_inputs(uint32_t rows, uint16_t *q, uint16_t *k, uint16_t *v,
	float *forget, float *beta, uint16_t *conv_in)
{
	uint32_t i;
	lcg_state = 0x12345678u;
	for ( i = 0; i < rows * K_KEY_HEADS * K_KEY_DIM; ++i )
	{
		q[i] = lcg_bf16();
		k[i] = lcg_bf16();
	}
	for ( i = 0; i < rows * K_KEY_HEADS * K_VPK * K_VALUE_DIM; ++i )
		v[i] = lcg_bf16();
	for ( i = 0; i < rows * K_KEY_HEADS * K_KEY_DIM; ++i )
		forget[i] = 0.75f + 0.25f * lcg_float();
	for ( i = 0; i < rows * K_KEY_HEADS; ++i )
		beta[i] = 0.5f + 0.25f * lcg_float();
	if ( conv_in != 0 )
		for ( i = 0; i < rows * K_CHANNELS; ++i )
			conv_in[i] = lcg_bf16();
}

/* One wave call of the delta rule: grid (sequences, key_heads). */
static void delta_wave(uint8_t *pool, const uint32_t *seqslots,
	const uint32_t *prefix, const uint16_t *q, const uint16_t *k,
	const uint16_t *v, const float *forget, const float *beta,
	uint16_t *out, uint32_t sequences)
{
	LM_LAUNCH((LmDeltaRuleKernel<K_THREADS,K_KEY_DIM,K_VALUE_DIM,KState>),
		dim3(sequences,K_KEY_HEADS), K_THREADS, 0, 0,
		pool,(uint32_t)K_SLOT_BYTES,seqslots,prefix,(const uint32_t *)0,
		q,k,v,forget,beta,out,K_KEY_HEADS,K_VPK,sequences,1u);
}

static void delta_sequential(uint8_t *pool, const uint32_t *row_slot,
	const uint16_t *q, const uint16_t *k, const uint16_t *v,
	const float *forget, const float *beta, uint16_t *out, uint32_t rows)
{
	uint32_t i;
	uint32_t slots[1];
	for ( i = 0; i < rows; ++i )
	{
		slots[0] = row_slot[i];
		delta_wave(pool, slots, 0,
			q + (size_t)i * K_KEY_HEADS * K_KEY_DIM,
			k + (size_t)i * K_KEY_HEADS * K_KEY_DIM,
			v + (size_t)i * K_KEY_HEADS * K_VPK * K_VALUE_DIM,
			forget + (size_t)i * K_KEY_HEADS * K_KEY_DIM,
			beta + (size_t)i * K_KEY_HEADS,
			out + (size_t)i * K_KEY_HEADS * K_VALUE_DIM, 1u);
	}
}

static int compare_outputs(const uint16_t *a, const uint16_t *b, size_t count,
	const char *label)
{
	size_t i;
	for ( i = 0; i < count; ++i )
		if ( a[i] != b[i] )
		{
			printf("  %s MISMATCH at word %zu: %04x vs %04x\n",
				label, i, a[i], b[i]);
			return 1;
		}
	return 0;
}

static int compare_pool(const uint8_t *a, const uint8_t *b, size_t bytes,
	const char *label)
{
	size_t i;
	for ( i = 0; i < bytes; ++i )
		if ( a[i] != b[i] )
		{
			printf("  %s MISMATCH at state byte %zu: %02x vs %02x\n",
				label, i, a[i], b[i]);
			return 1;
		}
	return 0;
}

static int scenario_delta_run8(void)
{
	uint16_t *q = (uint16_t *)xmalloc((size_t)8 * K_KEY_HEADS * K_KEY_DIM * 2);
	uint16_t *k = (uint16_t *)xmalloc((size_t)8 * K_KEY_HEADS * K_KEY_DIM * 2);
	uint16_t *v = (uint16_t *)xmalloc((size_t)8 * K_KEY_HEADS * K_VPK * K_VALUE_DIM * 2);
	float *forget = (float *)xmalloc((size_t)8 * K_KEY_HEADS * K_KEY_DIM * sizeof(float));
	float *beta = (float *)xmalloc((size_t)8 * K_KEY_HEADS * sizeof(float));
	uint16_t *out_a = (uint16_t *)xmalloc((size_t)8 * K_KEY_HEADS * K_VALUE_DIM * 2);
	uint16_t *out_b = (uint16_t *)xmalloc((size_t)8 * K_KEY_HEADS * K_VALUE_DIM * 2);
	uint8_t *pool_a = (uint8_t *)xmalloc(K_SLOT_BYTES);
	uint8_t *pool_b = (uint8_t *)xmalloc(K_SLOT_BYTES);
	uint32_t prefix[2] = { 0u, 8u };
	uint32_t slots[1] = { 0u };
	uint32_t row_slot[8];
	int failures;
	uint32_t i;
	init_inputs(8u, q, k, v, forget, beta, 0);
	for ( i = 0; i < K_SLOT_BYTES; ++i )
	{
		pool_a[i] = (uint8_t)(i * 7u + 3u);
		pool_b[i] = (uint8_t)(i * 7u + 3u);
	}
	memset(out_a, 0, (size_t)8 * K_KEY_HEADS * K_VALUE_DIM * 2);
	memset(out_b, 0, (size_t)8 * K_KEY_HEADS * K_VALUE_DIM * 2);
	for ( i = 0; i < 8u; ++i )
		row_slot[i] = 0u;
	delta_wave(pool_a, slots, prefix, q, k, v, forget, beta, out_a, 1u);
	delta_sequential(pool_b, row_slot, q, k, v, forget, beta, out_b, 8u);
	failures = compare_outputs(out_a, out_b, (size_t)8 * K_KEY_HEADS * K_VALUE_DIM,
		"S1 outputs");
	failures += compare_pool(pool_a, pool_b, K_SLOT_BYTES, "S1 state");
	free(q); free(k); free(v); free(forget); free(beta);
	free(out_a); free(out_b); free(pool_a); free(pool_b);
	return failures;
}

static int scenario_delta_runs_of_one(void)
{
	uint16_t *q = (uint16_t *)xmalloc((size_t)8 * K_KEY_HEADS * K_KEY_DIM * 2);
	uint16_t *k = (uint16_t *)xmalloc((size_t)8 * K_KEY_HEADS * K_KEY_DIM * 2);
	uint16_t *v = (uint16_t *)xmalloc((size_t)8 * K_KEY_HEADS * K_VPK * K_VALUE_DIM * 2);
	float *forget = (float *)xmalloc((size_t)8 * K_KEY_HEADS * K_KEY_DIM * sizeof(float));
	float *beta = (float *)xmalloc((size_t)8 * K_KEY_HEADS * sizeof(float));
	uint16_t *out_a = (uint16_t *)xmalloc((size_t)8 * K_KEY_HEADS * K_VALUE_DIM * 2);
	uint16_t *out_b = (uint16_t *)xmalloc((size_t)8 * K_KEY_HEADS * K_VALUE_DIM * 2);
	uint8_t *pool_a = (uint8_t *)xmalloc((size_t)8 * K_SLOT_BYTES);
	uint8_t *pool_b = (uint8_t *)xmalloc((size_t)8 * K_SLOT_BYTES);
	uint32_t prefix[9];
	uint32_t slots[8];
	uint32_t row_slot[8];
	int failures;
	uint32_t i;
	init_inputs(8u, q, k, v, forget, beta, 0);
	for ( i = 0; i < (size_t)8 * K_SLOT_BYTES; ++i )
	{
		pool_a[i] = (uint8_t)(i * 13u + 5u);
		pool_b[i] = (uint8_t)(i * 13u + 5u);
	}
	for ( i = 0; i < 8u; ++i )
	{
		prefix[i] = i;
		slots[i] = i;
		row_slot[i] = i;
	}
	prefix[8] = 8u;
	delta_wave(pool_a, slots, prefix, q, k, v, forget, beta, out_a, 8u);
	/* the decode form: no prefix, row i IS sequence i */
	delta_wave(pool_b, slots, 0, q, k, v, forget, beta, out_b, 8u);
	failures = compare_outputs(out_a, out_b, (size_t)8 * K_KEY_HEADS * K_VALUE_DIM,
		"S2 outputs");
	failures += compare_pool(pool_a, pool_b, (size_t)8 * K_SLOT_BYTES, "S2 state");
	free(q); free(k); free(v); free(forget); free(beta);
	free(out_a); free(out_b); free(pool_a); free(pool_b);
	return failures;
}

static int scenario_delta_mixed(void)
{
	uint16_t *q = (uint16_t *)xmalloc((size_t)K_ROWS * K_KEY_HEADS * K_KEY_DIM * 2);
	uint16_t *k = (uint16_t *)xmalloc((size_t)K_ROWS * K_KEY_HEADS * K_KEY_DIM * 2);
	uint16_t *v = (uint16_t *)xmalloc((size_t)K_ROWS * K_KEY_HEADS * K_VPK * K_VALUE_DIM * 2);
	float *forget = (float *)xmalloc((size_t)K_ROWS * K_KEY_HEADS * K_KEY_DIM * sizeof(float));
	float *beta = (float *)xmalloc((size_t)K_ROWS * K_KEY_HEADS * sizeof(float));
	uint16_t *out_a = (uint16_t *)xmalloc((size_t)K_ROWS * K_KEY_HEADS * K_VALUE_DIM * 2);
	uint16_t *out_b = (uint16_t *)xmalloc((size_t)K_ROWS * K_KEY_HEADS * K_VALUE_DIM * 2);
	uint8_t *pool_a = (uint8_t *)xmalloc((size_t)K_SLOTS * K_SLOT_BYTES);
	uint8_t *pool_b = (uint8_t *)xmalloc((size_t)K_SLOTS * K_SLOT_BYTES);
	/* rows 0..4 = one prefill span (slot 3), row 5 = a lone decode row
	 * (slot 1), rows 6..11 = a second span (slot 0) */
	const uint32_t prefix[4] = { 0u, 5u, 6u, 12u };
	const uint32_t slots[3] = { 3u, 1u, 0u };
	uint32_t row_slot[K_ROWS] = { 3u, 3u, 3u, 3u, 3u, 1u, 0u, 0u, 0u, 0u, 0u, 0u };
	int failures;
	uint32_t i;
	init_inputs(K_ROWS, q, k, v, forget, beta, 0);
	for ( i = 0; i < (size_t)K_SLOTS * K_SLOT_BYTES; ++i )
	{
		pool_a[i] = (uint8_t)(i * 29u + 11u);
		pool_b[i] = (uint8_t)(i * 29u + 11u);
	}
	delta_wave(pool_a, slots, prefix, q, k, v, forget, beta, out_a, 3u);
	delta_sequential(pool_b, row_slot, q, k, v, forget, beta, out_b, K_ROWS);
	failures = compare_outputs(out_a, out_b, (size_t)K_ROWS * K_KEY_HEADS * K_VALUE_DIM,
		"S3 outputs");
	failures += compare_pool(pool_a, pool_b, (size_t)K_SLOTS * K_SLOT_BYTES,
		"S3 state");
	free(q); free(k); free(v); free(forget); free(beta);
	free(out_a); free(out_b); free(pool_a); free(pool_b);
	return failures;
}

static int scenario_conv_run8(void)
{
	uint16_t *conv_in = (uint16_t *)xmalloc((size_t)8 * K_CHANNELS * 2);
	uint16_t *out_a = (uint16_t *)xmalloc((size_t)8 * K_CHANNELS * 2);
	uint16_t *out_b = (uint16_t *)xmalloc((size_t)8 * K_CHANNELS * 2);
	uint16_t *window_a = (uint16_t *)xmalloc((size_t)K_CHANNELS * K_KERNEL * 2);
	uint16_t *window_b = (uint16_t *)xmalloc((size_t)K_CHANNELS * K_KERNEL * 2);
	float *weight = (float *)xmalloc((size_t)K_CHANNELS * K_KERNEL * sizeof(float));
	uint32_t prefix[2] = { 0u, 8u };
	uint32_t slots[1] = { 0u };
	uint32_t i;
	lcg_state = 0xdeadbeefu;
	for ( i = 0; i < 8u * K_CHANNELS; ++i )
		conv_in[i] = lcg_bf16();
	for ( i = 0; i < K_CHANNELS * K_KERNEL; ++i )
		weight[i] = lcg_float();
	for ( i = 0; i < K_CHANNELS * K_KERNEL; ++i )
	{
		window_a[i] = (uint16_t)(i * 3u + 1u);
		window_b[i] = (uint16_t)(i * 3u + 1u);
	}
	memset(out_a, 0, (size_t)8 * K_CHANNELS * 2);
	memset(out_b, 0, (size_t)8 * K_CHANNELS * 2);
	LM_LAUNCH((LmCausalConvKernel<K_THREADS,K_KERNEL,LM_CONV_SWISH,float>),
		dim3(1u,(K_CHANNELS + K_THREADS - 1u) / K_THREADS), K_THREADS, 0, 0,
		window_a,slots,prefix,(const uint32_t *)0,conv_in,weight,out_a,
		K_CHANNELS,1u,1u);
	for ( i = 0; i < 8u; ++i )
		LM_LAUNCH((LmCausalConvKernel<K_THREADS,K_KERNEL,LM_CONV_SWISH,float>),
			dim3(1u,(K_CHANNELS + K_THREADS - 1u) / K_THREADS), K_THREADS, 0, 0,
			window_b,slots,0,(const uint32_t *)0,
			conv_in + (size_t)i * K_CHANNELS,weight,
			out_b + (size_t)i * K_CHANNELS,K_CHANNELS,1u,1u);
	{
		int failures = compare_outputs(out_a, out_b, (size_t)8 * K_CHANNELS,
			"S4 outputs");
		failures += compare_pool((const uint8_t *)window_a,
			(const uint8_t *)window_b, (size_t)K_CHANNELS * K_KERNEL * 2,
			"S4 window");
		free(conv_in); free(out_a); free(out_b); free(window_a);
		free(window_b); free(weight);
		return failures;
	}
}

static int scenario_indexed(uint32_t lanes)
{
	uint16_t q[K_ROWS*K_KEY_HEADS*K_KEY_DIM],k[K_ROWS*K_KEY_HEADS*K_KEY_DIM],v[K_ROWS*K_KEY_HEADS*K_VPK*K_VALUE_DIM];
	uint16_t a[K_ROWS*K_KEY_HEADS*K_VALUE_DIM],b[K_ROWS*K_KEY_HEADS*K_VALUE_DIM];
	uint16_t input[K_ROWS*K_CHANNELS],ca[K_ROWS*K_CHANNELS],cb[K_ROWS*K_CHANNELS];
	uint16_t wa[K_SLOTS*K_CHANNELS*K_KERNEL] = {},wb[K_SLOTS*K_CHANNELS*K_KERNEL] = {};
	float forget[K_ROWS*K_KEY_HEADS*K_KEY_DIM],beta[K_ROWS*K_KEY_HEADS],weights[K_CHANNELS*K_KERNEL];
	uint8_t pa[K_SLOTS*K_SLOT_BYTES] = {},pb[K_SLOTS*K_SLOT_BYTES] = {};
	uint32_t counts[5] = {4u,1u,3u,2u,2u},slots[5] = {7u,2u,4u,1u,6u},row_slots[K_ROWS],begin[6] = {},indices[K_ROWS];
	uint32_t row=0u,lane,step,index=0u;
	if (lanes == 1u) counts[0] = 12u;
	if (lanes == 3u) { counts[0] = 5u; counts[1] = 1u; counts[2] = 6u; }
	for (step=0u; step<K_ROWS; step++)
		for (lane=0u; lane<lanes; lane++)
			if (step < counts[lane]) row_slots[row++] = slots[lane];
	if (row != K_ROWS) return 1;
	for (lane=0u; lane<lanes; lane++)
	{
		for (row=0u; row<K_ROWS; row++)
			if (row_slots[row] == slots[lane]) indices[index++] = row;
		begin[lane + 1u] = index;
	}
	init_inputs(K_ROWS,q,k,v,forget,beta,input);
	for (index=0u; index<K_CHANNELS*K_KERNEL; index++) weights[index] = lcg_float();
	LM_LAUNCH((LmDeltaRuleKernel<K_THREADS,K_KEY_DIM,K_VALUE_DIM,KState>),
		dim3(lanes,K_KEY_HEADS),K_THREADS,0,0,pa,(uint32_t)K_SLOT_BYTES,slots,begin,(const uint32_t *)0,
		q,k,v,forget,beta,a,K_KEY_HEADS,K_VPK,lanes,0u,indices);
	int failures = compare_pool(pa,pb,sizeof(pa),"indexed uncommitted state");
	LM_LAUNCH((LmDeltaRuleKernel<K_THREADS,K_KEY_DIM,K_VALUE_DIM,KState>),
		dim3(lanes,K_KEY_HEADS),K_THREADS,0,0,pa,(uint32_t)K_SLOT_BYTES,slots,begin,(const uint32_t *)0,
		q,k,v,forget,beta,a,K_KEY_HEADS,K_VPK,lanes,1u,indices);
	delta_sequential(pb,row_slots,q,k,v,forget,beta,b,K_ROWS);
	uint32_t nonzero = 0u;
	for (index=0u; index<K_ROWS*K_KEY_HEADS*K_VALUE_DIM; index++) nonzero += b[index] != 0u;
	if (nonzero == 0u) failures++;
	failures += compare_outputs(a,b,K_ROWS*K_KEY_HEADS*K_VALUE_DIM,"indexed delta outputs");
	failures += compare_pool(pa,pb,sizeof(pa),"indexed delta state");
	LM_LAUNCH((LmCausalConvKernel<K_THREADS,K_KERNEL,LM_CONV_SWISH,float>),
		dim3(lanes,K_CHANNELS),K_THREADS,0,0,wa,slots,begin,(const uint32_t *)0,input,weights,ca,K_CHANNELS,lanes,0u,indices);
	failures += compare_pool((const uint8_t *)wa,(const uint8_t *)wb,sizeof(wa),"indexed uncommitted window");
	LM_LAUNCH((LmCausalConvKernel<K_THREADS,K_KERNEL,LM_CONV_SWISH,float>),
		dim3(lanes,K_CHANNELS),K_THREADS,0,0,wa,slots,begin,(const uint32_t *)0,input,weights,ca,K_CHANNELS,lanes,1u,indices);
	for (row=0u; row<K_ROWS; row++)
		LM_LAUNCH((LmCausalConvKernel<K_THREADS,K_KERNEL,LM_CONV_SWISH,float>),
			dim3(1u,K_CHANNELS),K_THREADS,0,0,wb,row_slots+row,(const uint32_t *)0,(const uint32_t *)0,
			input+row*K_CHANNELS,weights,cb+row*K_CHANNELS,K_CHANNELS,1u,1u);
	failures += compare_outputs(ca,cb,K_ROWS*K_CHANNELS,"indexed conv outputs");
	failures += compare_pool((const uint8_t *)wa,(const uint8_t *)wb,sizeof(wa),"indexed conv windows");
	printf("indexed B%u unequal round-major rows: %s\n",lanes,failures == 0 ? "PASS" : "FAIL");
	return failures;
}

int main(void)
{
	int failures = 0;
	failures += scenario_indexed(1u);
	failures += scenario_indexed(3u);
	failures += scenario_indexed(5u);
	failures += scenario_delta_run8();
	printf("S1 delta run-of-8 vs sequential: %s\n",
		failures == 0 ? "PASS" : "FAIL");
	{
		int before = failures;
		failures += scenario_delta_runs_of_one();
		printf("S2 delta runs-of-one vs NULL prefix: %s\n",
			failures == before ? "PASS" : "FAIL");
	}
	{
		int before = failures;
		failures += scenario_delta_mixed();
		printf("S3 delta mixed 3-run wave vs sequential: %s\n",
			failures == before ? "PASS" : "FAIL");
	}
	{
		int before = failures;
		failures += scenario_conv_run8();
		printf("S4 conv run-of-8 vs sequential: %s\n",
			failures == before ? "PASS" : "FAIL");
	}
	return failures != 0;
}
