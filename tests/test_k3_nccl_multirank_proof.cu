// F1 owed-evidence closer: THE LIVE MULTI-RANK NCCL EXCHANGE PROOF.
//
// The single-rank gates (tests/test_k3_device_only_init.cu,
// tests/test_k3_head_key_probe.cu) prove the init admission and the key
// contract, but their exchanges are either unreachable-peers or ONE process
// emulating the reduce. This program closes the gap at window boot: WORLD
// real OS processes bootstrap the production NCCL backend
// (ring/transport/tp_device_collective_nccl.c -> ncclCommInitRank) through
// SparkTpDeviceCollectiveCreate and drive the exact two ops the K3 runner
// issues - SparkTpDeviceCollectiveSubmitBf16 (the hidden sums) and
// SparkTpDeviceCollectiveSubmitU64Max (the head argmax) - on device buffers
// over a real CUDA stream.
//
// Legs (strict submission-ordinal order, one sync point per leg):
//   A  bf16 hidden sum at the K3 width (7168): every rank's post-reduce
//      buffer is bit-exact vs the host expectation (values chosen so every
//      partial sum is exactly representable in bf16, so any NCCL reduction
//      order must produce identical bits), plus a reserved0-NARROWED submit
//      whose frame tail must stay untouched (the phase-hook override).
//   B  head argmax over engineered cross-rank cases: clear winner, all-tie,
//      partial tie, NaN-on-rank-0, +/-inf, denormal-vs-zero, negatives,
//      vocab-scale tokens. Every rank packs ITS column with the runner's
//      verbatim kernels, submits ONE u64-max, unpacks from the stream-
//      ordered completion callback (the K3RunnerHeadCompletion pattern),
//      and checks the winner (score bits, token) against the packed-key
//      oracle computed locally. Finite cases additionally check parity with
//      the host tier's first-max scan rule. Row 3 (NaN on rank 0) is
//      deliberately NOT rule-parity-checked: the device tier's documented
//      improvement is that a NaN never wins, while the host scan's
//      false-compare would keep the NaN-holding first rank.
//   C  four bf16 reductions enqueued back-to-back BEFORE any sync - the
//      strict per-process ordinal chain and stream ordering, live.
//
// Every rank prints "ok/FAIL" lines and a final
// "K3_NCCL_MULTIRANK_RANK_RESULT rank=R world=W failures=F"; the driver
// (tools/k3_nccl_multirank_gate.sh) requires failures==0 from ALL ranks.
//
// One GPU, WORLD processes, one free TCP port for the unique-id bootstrap.
// Needs nvcc + libnccl.so.2 + a CUDA device: everything else SKIPs.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

#include <cuda_runtime.h>

#include "sparkpipe/spark_tp_device_collective.h"

// ---- extracted verbatim from spark_k3_resident_decode_stage_runner.cu ----
// (same pin as tests/test_k3_head_key_probe.cu; drift is caught there)
__global__ static void K3RunnerHeadPackKernel(const uint32_t *tokens,
	const float *scores, unsigned long long *keys, uint32_t tp_rank,
	uint32_t rows)
{
	uint32_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if ( i >= rows )
		return;
	float s = scores[i];
	if ( s != s )
		{ keys[i] = 0ull; return; }
	uint32_t bits = __float_as_uint(s);
	uint32_t ordered = (bits & 0x80000000u) != 0u ? ~bits :
		(bits | 0x80000000u);
	keys[i] = ((unsigned long long)ordered << 32) |
		((unsigned long long)(uint32_t)(~tp_rank & 0xFFu) << 24) |
		(unsigned long long)(tokens[i] & 0xFFFFFFu);
}

__global__ static void K3RunnerHeadUnpackKernel(
	const unsigned long long *keys, uint32_t *tokens, float *scores,
	uint32_t rows)
{
	uint32_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if ( i >= rows )
		return;
	unsigned long long k = keys[i];
	uint32_t bits = (uint32_t)(k >> 32);
	bits = (bits & 0x80000000u) != 0u ? (bits ^ 0x80000000u) : ~bits;
	scores[i] = __uint_as_float(bits);
	tokens[i] = (uint32_t)(k & 0xFFFFFFull);
}
// -------------------------------------------------------------------------

#define PROOF_HIDDEN 7168u          /* the K3 hidden width */
#define PROOF_NARROW 2048u          /* the narrowed-submit element count */
#define PROOF_ROWS 10u              /* head cases, one row per case */
#define PROOF_MAX_WORLD 16u

static uint32_t g_failures = 0;
static uint32_t g_rank;
static uint32_t g_world;

static void Check(int ok, const char *what)
{
	if ( ok )
		printf("ok world=%u rank=%u %s\n", g_world, g_rank, what);
	else
	{
		printf("FAIL world=%u rank=%u %s\n", g_world, g_rank, what);
		g_failures++;
	}
}

/* fp32 -> bf16, round-to-nearest-even (expected values are exact small
 * multiples of four, so this is identity for them - kept general anyway). */
static uint16_t HostF32ToBf16(float value)
{
	uint32_t bits, rounded;
	memcpy(&bits, &value, sizeof(bits));
	rounded = 0x7FFFu + ((bits >> 16u) & 1u);
	return (uint16_t)((bits + rounded) >> 16u);
}

static void HostPackKey(float score, uint32_t token, uint32_t rank,
	unsigned long long *key_out)
{
	if ( score != score )
		{ *key_out = 0ull; return; }
	uint32_t bits;
	memcpy(&bits, &score, sizeof(bits));
	uint32_t ordered = (bits & 0x80000000u) != 0u ? ~bits :
		(bits | 0x80000000u);
	*key_out = ((unsigned long long)ordered << 32) |
		((unsigned long long)(uint32_t)(~rank & 0xFFu) << 24) |
		(unsigned long long)(token & 0xFFFFFFu);
}

static void HostUnpackKey(unsigned long long key, uint32_t *token_out,
	uint32_t *bits_out)
{
	uint32_t bits = (uint32_t)(key >> 32);
	bits = (bits & 0x80000000u) != 0u ? (bits ^ 0x80000000u) : ~bits;
	*bits_out = bits;
	*token_out = (uint32_t)(key & 0xFFFFFFull);
}

typedef struct ProofCase
{
	float scores[PROOF_MAX_WORLD];
	uint32_t tokens[PROOF_MAX_WORLD];
	int rule_comparable;           /* finite everywhere -> host-rule parity */
} ProofCase;

/* The deterministic cross-rank case table (identical on every rank). */
static void BuildCases(uint32_t world, ProofCase *cases)
{
	uint32_t r, w1 = world - 1u;
	uint32_t tie_partner = w1 < 3u ? w1 : 3u;
	uint32_t denormal_rank = w1 < 2u ? w1 : 2u;
	uint32_t flt_max_rank = w1 < 1u ? w1 : 1u;
	uint32_t half_rank = world / 2u;
	memset(cases, 0, sizeof(*cases) * PROOF_ROWS);
	for ( r = 0u; r < world; ++r )
	{
		/* 0: clear winner on the LAST rank */
		cases[0].scores[r] = r == w1 ? 9.0f : 1.0f + 0.5f * (float)r;
		cases[0].tokens[r] = 1000u + r;
		cases[0].rule_comparable = 1;
		/* 1: ALL-tie -> lowest rank's token wins */
		cases[1].scores[r] = 7.0f;
		cases[1].tokens[r] = 262143u - r;
		cases[1].rule_comparable = 1;
		/* 2: partial tie {0, tie_partner} */
		cases[2].scores[r] =
			(r == 0u || r == tie_partner) ? 7.0f : 3.0f;
		cases[2].tokens[r] = 500u + r;
		cases[2].rule_comparable = 1;
		/* 3: NaN on rank 0 - key-oracle ONLY (documented divergence:
		 * the device tier refuses NaN winners, the host scan keeps
		 * the first rank on false compares) */
		cases[3].scores[r] = r == 0u ?
			__builtin_nanf("") : 2.0f + (float)r;
		cases[3].tokens[r] = 60u + r;
		/* 4: +inf beats large finites */
		cases[4].scores[r] = r == w1 ?
			__builtin_inff() : -3.0f * (float)r;
		cases[4].tokens[r] = 70u + r;
		cases[4].rule_comparable = 1;
		/* 5: -inf loses to every finite */
		cases[5].scores[r] = r == 0u ?
			-__builtin_inff() : 1.5f * (float)(r + 1u);
		cases[5].tokens[r] = 80u + r;
		cases[5].rule_comparable = 1;
		/* 6: denormal beats exact zero */
		cases[6].scores[r] = r == denormal_rank ? 1e-45f : 0.0f;
		cases[6].tokens[r] = 90u + r;
		cases[6].rule_comparable = 1;
		/* 7: negatives */
		cases[7].scores[r] = r == 0u ? -5.0f : -1.0f;
		cases[7].tokens[r] = 110u + r;
		cases[7].rule_comparable = 1;
		/* 8: FLT_MAX edge */
		cases[8].scores[r] = r == flt_max_rank ?
			3.4028235e38f : 0.0f;
		cases[8].tokens[r] = 120u + r;
		cases[8].rule_comparable = 1;
		/* 9: vocab-scale token carried intact on the winner */
		cases[9].scores[r] = r == half_rank ? 4.0f : 3.0f;
		cases[9].tokens[r] = r == half_rank ? 262143u : 130u + r;
		cases[9].rule_comparable = 1;
	}
}

/* The ground truth of the wire: max over the HOST-packed keys. */
static void KeyOracle(const ProofCase *cases, uint32_t world, uint32_t row,
	uint32_t *token_out, uint32_t *bits_out)
{
	unsigned long long best = 0ull, key;
	uint32_t r;
	for ( r = 0u; r < world; ++r )
	{
		HostPackKey(cases[row].scores[r], cases[row].tokens[r], r, &key);
		if ( key > best )
			best = key;
	}
	HostUnpackKey(best, token_out, bits_out);
}

/* The runner host tier's first-max scan (strict > keeps lowest rank). */
static uint32_t RuleOracle(const ProofCase *cases, uint32_t world,
	uint32_t row)
{
	uint32_t best = 0u, r;
	for ( r = 1u; r < world; ++r )
		if ( cases[row].scores[r] > cases[row].scores[best] )
			best = r;
	return best;
}

static uint16_t LegValue(uint32_t rank, uint32_t i, uint32_t iter)
{
	/* every partial sum stays a multiple of four below 64 -> bit-exact
	 * under ANY reduction order in bf16 */
	return (i % 7u) == ((rank + iter) % 7u) ?
		HostF32ToBf16(4.0f) : (uint16_t)0u;
}

typedef struct CountingCompletion
{
	uint32_t seen;
	unsigned long long ordinals[8];
} CountingCompletion;

static void CountingComplete(void *context,
	const SparkTpDeviceCollectiveCompletion *completion)
{
	CountingCompletion *state = (CountingCompletion *)context;
	state->ordinals[state->seen % 8u] = completion->ordinal;
	state->seen++;
}

typedef struct HeadCompletion
{
	unsigned long long *keys;
	uint32_t *tokens;
	float *scores;
	cudaStream_t stream;
	uint32_t rows;
} HeadCompletion;

/* Stream-order continuation: mirrors K3RunnerHeadCompletion in the runner. */
static void HeadComplete(void *context,
	const SparkTpDeviceCollectiveCompletion *completion)
{
	HeadCompletion *head = (HeadCompletion *)context;
	(void)completion;
	K3RunnerHeadUnpackKernel<<<(head->rows + 255u) / 256u, 256u, 0,
		head->stream>>>(head->keys, head->tokens, head->scores,
		head->rows);
	delete head;
}

int main(int argc, char **argv)
{
	uint32_t rank = 0u, world = 0u;
	long port = 0L;
	const char *module = "libnccl.so.2";
	int i;
	for ( i = 1; i + 1 < argc; i += 2 )
	{
		if ( strcmp(argv[i], "--rank") == 0 )
			rank = (uint32_t)strtoul(argv[i + 1], 0, 10);
		else if ( strcmp(argv[i], "--world") == 0 )
			world = (uint32_t)strtoul(argv[i + 1], 0, 10);
		else if ( strcmp(argv[i], "--port") == 0 )
			port = strtol(argv[i + 1], 0, 10);
		else if ( strcmp(argv[i], "--module") == 0 )
			module = argv[i + 1];
	}
	if ( world < 2u || world > PROOF_MAX_WORLD || port <= 0L ||
		port > 65535L || rank >= world )
	{
		fprintf(stderr, "usage: %s --rank R --world N --port P "
			"[--module libnccl.so.2]\n", argv[0]);
		return 2;
	}
	g_rank = rank;
	g_world = world;

	cudaError_t error = cudaSetDevice(0);
	if ( error != cudaSuccess )
	{
		printf("FAIL world=%u rank=%u cudaSetDevice %s\n", world, rank,
			cudaGetErrorString(error));
		return 1;
	}
	cudaStream_t stream;
	error = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
	if ( error != cudaSuccess )
	{
		printf("FAIL world=%u rank=%u cudaStreamCreate %s\n", world, rank,
			cudaGetErrorString(error));
		return 1;
	}

	SparkTpDeviceCollectiveConfig config;
	static const char *hosts[PROOF_MAX_WORLD];
	memset(&config, 0, sizeof(config));
	config.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	config.backend_kind = SPARK_TP_DEVICE_COLLECTIVE_BACKEND_NCCL;
	config.tp_degree = world;
	config.tp_rank = rank;
	config.operation_kind =
		SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16;
	config.credit_count = 8u;
	config.local_hidden_dimension = PROOF_HIDDEN;
	config.max_active_sequence_count = 16u;
	config.connect_timeout_milli = 20000u;
	config.operation_timeout_milli = 20000u;
	config.control_port_base = (uint16_t)port;
	config.collective_identifier =
		UINT64_C(0x4B330000) | (uint64_t)(uint16_t)port;
	config.backend_module_path = module;
	config.local_host = "127.0.0.1";
	for ( i = 0; i < (int)world; ++i )
		hosts[i] = "127.0.0.1";
	memcpy(config.rank_hosts, hosts, sizeof(hosts));

	SparkTpDeviceCollective collective;
	SparkStatus status = SparkTpDeviceCollectiveCreate(&config, &collective);
	Check(status == SPARK_STATUS_OK,
		"live nccl create (bootstrap + comm_init_rank)");
	if ( status != SPARK_STATUS_OK )
	{
		printf("K3_NCCL_MULTIRANK_RANK_RESULT rank=%u world=%u "
			"failures=1\n", rank, world);
		return 1;
	}
	Check(collective.backend_kind ==
		SPARK_TP_DEVICE_COLLECTIVE_BACKEND_NCCL &&
		collective.tp_degree == world && collective.tp_rank == rank,
		"collective descriptor matches request");

	CountingCompletion counting;
	memset(&counting, 0, sizeof(counting));
	SparkTpDeviceCollectiveSubmission submission;
	uint64_t ordinal = 0u;

	/* ---- LEG A: bf16 hidden sum at the K3 width ---- */
	{
		static uint16_t local[PROOF_HIDDEN];
		static uint16_t expected[PROOF_HIDDEN];
		static uint16_t after[PROOF_HIDDEN];
		uint16_t *device = 0;
		error = cudaMalloc(&device, (uint64_t)PROOF_HIDDEN * 2u);
		Check(error == cudaSuccess && device != 0, "legA cudaMalloc");
		for ( uint32_t iter = 0u; iter < 2u && device != 0; ++iter )
		{
			uint32_t mismatch;
			char what[96];
			for ( uint32_t j = 0u; j < PROOF_HIDDEN; ++j )
				local[j] = LegValue(rank, j, iter);
			for ( uint32_t j = 0u; j < PROOF_HIDDEN; ++j )
			{
				uint32_t hits = 0u;
				for ( uint32_t r = 0u; r < world; ++r )
					if ( (j % 7u) == ((r + iter) % 7u) )
						hits++;
				expected[j] = HostF32ToBf16(4.0f * (float)hits);
			}
			error = cudaMemcpy(device, local,
				(uint64_t)PROOF_HIDDEN * 2u, cudaMemcpyHostToDevice);
			Check(error == cudaSuccess, "legA H2D");
			memset(&submission, 0, sizeof(submission));
			submission.abi_version =
				SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
			submission.descriptor_bytes = sizeof(submission);
			submission.slot_index = rank;
			submission.active_sequence_count = 1u;
			submission.reserved0 = PROOF_HIDDEN; /* element override */
			submission.ordinal = ordinal++;
			submission.local_device = device;
			submission.full_device = device;
			submission.cuda_stream = stream;
			submission.completion_function = CountingComplete;
			submission.completion_context = &counting;
			status = SparkTpDeviceCollectiveSubmitBf16(&collective,
				&submission);
			snprintf(what, sizeof(what),
				"legA iter%u submit (ordinal %llu)", iter,
				(unsigned long long)submission.ordinal);
			Check(status == SPARK_STATUS_OK, what);
			if ( status != SPARK_STATUS_OK )
				break;
			error = cudaStreamSynchronize(stream);
			Check(error == cudaSuccess, "legA sync");
			error = cudaMemcpy(after, device,
				(uint64_t)PROOF_HIDDEN * 2u, cudaMemcpyDeviceToHost);
			Check(error == cudaSuccess, "legA D2H");
			mismatch = 0u;
			for ( uint32_t j = 0u; j < PROOF_HIDDEN; ++j )
				if ( after[j] != expected[j] )
					mismatch++;
			snprintf(what, sizeof(what),
				"legA iter%u bf16 sum bit-exact (%u mismatches)",
				iter, mismatch);
			Check(mismatch == 0u, what);
		}

		/* narrowed override: the frame tail must survive untouched */
		if ( device != 0 )
		{
			uint32_t bad;
			for ( uint32_t j = 0u; j < PROOF_HIDDEN; ++j )
				local[j] = (uint16_t)(0x3800u |
					(0x007Fu & ((rank * 13u + j) ^ (j >> 3u))));
			for ( uint32_t j = 0u; j < PROOF_NARROW; ++j )
				local[j] = LegValue(rank, j, 9u);
			for ( uint32_t j = 0u; j < PROOF_NARROW; ++j )
			{
				uint32_t hits = 0u;
				for ( uint32_t r = 0u; r < world; ++r )
					if ( (j % 7u) == ((r + 9u) % 7u) )
						hits++;
				expected[j] = HostF32ToBf16(4.0f * (float)hits);
			}
			error = cudaMemcpy(device, local,
				(uint64_t)PROOF_HIDDEN * 2u, cudaMemcpyHostToDevice);
			Check(error == cudaSuccess, "legAnarrow H2D");
			memset(&submission, 0, sizeof(submission));
			submission.abi_version =
				SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
			submission.descriptor_bytes = sizeof(submission);
			submission.slot_index = rank;
			submission.active_sequence_count = 1u;
			submission.reserved0 = PROOF_NARROW;
			submission.ordinal = ordinal++;
			submission.local_device = device;
			submission.full_device = device;
			submission.cuda_stream = stream;
			submission.completion_function = CountingComplete;
			submission.completion_context = &counting;
			status = SparkTpDeviceCollectiveSubmitBf16(&collective,
				&submission);
			Check(status == SPARK_STATUS_OK, "legAnarrow submit");
			error = cudaStreamSynchronize(stream);
			Check(error == cudaSuccess, "legAnarrow sync");
			error = cudaMemcpy(after, device,
				(uint64_t)PROOF_HIDDEN * 2u, cudaMemcpyDeviceToHost);
			Check(error == cudaSuccess, "legAnarrow D2H");
			bad = 0u;
			for ( uint32_t j = 0u; j < PROOF_NARROW; ++j )
				if ( after[j] != expected[j] )
					bad++;
			for ( uint32_t j = PROOF_NARROW; j < PROOF_HIDDEN; ++j )
				if ( after[j] != local[j] )
					bad++;
			Check(bad == 0u,
				"legAnarrow narrowed sum exact + tail untouched");
		}
		if ( device != 0 )
			cudaFree(device);
	}

	/* ---- LEG B: the head's u64-max argmax, live ---- */
	{
		ProofCase cases[PROOF_ROWS];
		BuildCases(world, cases);
		static float host_scores[PROOF_ROWS];
		static uint32_t host_tokens[PROOF_ROWS];
		static uint32_t out_tokens[PROOF_ROWS];
		static float out_scores[PROOF_ROWS];
		uint32_t leg_b_failures_before = g_failures;
		for ( uint32_t j = 0u; j < PROOF_ROWS; ++j )
		{
			host_scores[j] = cases[j].scores[rank];
			host_tokens[j] = cases[j].tokens[rank];
		}
		float *d_scores = 0;
		uint32_t *d_tokens = 0;
		unsigned long long *d_keys = 0;
		error = cudaMalloc(&d_scores, (uint64_t)PROOF_ROWS * 4u);
		error = cudaMalloc(&d_tokens, (uint64_t)PROOF_ROWS * 4u);
		if ( error == cudaSuccess )
			error = cudaMalloc(&d_keys, (uint64_t)PROOF_ROWS * 8u);
		Check(error == cudaSuccess && d_scores != 0 &&
			d_tokens != 0 && d_keys != 0, "legB cudaMalloc");
		if ( error == cudaSuccess )
		{
			error = cudaMemcpy(d_scores, host_scores,
				(uint64_t)PROOF_ROWS * 4u, cudaMemcpyHostToDevice);
			error = cudaMemcpy(d_tokens, host_tokens,
				(uint64_t)PROOF_ROWS * 4u, cudaMemcpyHostToDevice);
			Check(error == cudaSuccess, "legB H2D");
			K3RunnerHeadPackKernel<<<(PROOF_ROWS + 255u) / 256u, 256u,
				0, stream>>>(d_tokens, d_scores, d_keys, rank,
				PROOF_ROWS);
			memset(&submission, 0, sizeof(submission));
			submission.abi_version =
				SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
			submission.descriptor_bytes = sizeof(submission);
			submission.slot_index = rank;
			submission.active_sequence_count = PROOF_ROWS;
			submission.flags =
				SPARK_TP_DEVICE_COLLECTIVE_SUBMISSION_STREAM_ORDERED_COMPLETION;
			submission.ordinal = ordinal++;
			submission.local_device = d_keys;
			submission.full_device = d_keys;
			submission.cuda_stream = stream;
			submission.completion_function = HeadComplete;
			HeadCompletion *context = new HeadCompletion;
			context->keys = d_keys;
			context->tokens = d_tokens;
			context->scores = d_scores;
			context->stream = stream;
			context->rows = PROOF_ROWS;
			submission.completion_context = context;
			status = SparkTpDeviceCollectiveSubmitU64Max(&collective,
				&submission);
			Check(status == SPARK_STATUS_OK, "legB u64-max submit");
			if ( status == SPARK_STATUS_OK )
			{
				uint32_t j;
				error = cudaStreamSynchronize(stream);
				Check(error == cudaSuccess, "legB sync");
				error = cudaMemcpy(out_tokens, d_tokens,
					(uint64_t)PROOF_ROWS * 4u,
					cudaMemcpyDeviceToHost);
				if ( error == cudaSuccess )
					error = cudaMemcpy(out_scores, d_scores,
						(uint64_t)PROOF_ROWS * 4u,
						cudaMemcpyDeviceToHost);
				Check(error == cudaSuccess, "legB D2H");
				for ( j = 0u; j < PROOF_ROWS; ++j )
				{
					uint32_t want_token, want_bits, got_bits;
					KeyOracle(cases, world, j, &want_token,
						&want_bits);
					memcpy(&got_bits, &out_scores[j],
						sizeof(got_bits));
					if ( got_bits != want_bits ||
						out_tokens[j] != want_token )
					{
						printf("FAIL world=%u rank=%u legB row %u: "
							"want(bits %08x tok %u) got(bits %08x tok"
							" %u)\n", world, rank, j, want_bits,
							want_token, got_bits, out_tokens[j]);
						g_failures++;
					}
					if ( cases[j].rule_comparable )
					{
						uint32_t rule_rank = RuleOracle(cases, world,
							j);
						uint32_t rule_bits;
						memcpy(&rule_bits,
							&cases[j].scores[rule_rank],
							sizeof(rule_bits));
						if ( got_bits != rule_bits ||
							out_tokens[j] !=
								cases[j].tokens[rule_rank] )
						{
							printf("FAIL world=%u rank=%u legB row %u "
								"host-rule parity: want(bits %08x tok %"
								"u) got(bits %08x tok %u)\n", world,
								rank, j, rule_bits,
								cases[j].tokens[rule_rank], got_bits,
								out_tokens[j]);
							g_failures++;
						}
					}
				}
				Check(g_failures == leg_b_failures_before,
					"legB head argmax == key oracle (+ host rule on "
					"finite rows) on ALL cases");
			}
			cudaFree(d_scores);
			cudaFree(d_tokens);
			cudaFree(d_keys);
		}
	}

	/* ---- LEG C: four reduces enqueued BEFORE any sync ---- */
	{
		/* Four INDEPENDENT buffer pairs, all prefilled up front, one
		 * reduce per pair: back-to-back submission with zero host syncs
		 * between ops, while every expected value stays an exact multiple
		 * of four below 256 regardless of reduction order or world size.
		 * (An accumulate-in-place variant would scale by the world size
		 * per op and lose bf16 exactness at w16 - the per-op form is the
		 * honest drain test.) */
		static uint16_t local[PROOF_HIDDEN];
		static uint16_t after[PROOF_HIDDEN];
		uint16_t *bufs[4];
		uint32_t step, submits_ok = 0u, bad;
		unsigned long long first_ordinal = ordinal;
		for ( step = 0u; step < 4u; ++step )
			bufs[step] = 0;
		uint32_t alloc_bad = 0u;
		for ( step = 0u; step < 4u; ++step )
			if ( cudaMalloc(&bufs[step],
				(uint64_t)PROOF_HIDDEN * 2u) != cudaSuccess )
					alloc_bad++;
		Check(alloc_bad == 0u, "legC cudaMalloc x4");
		if ( alloc_bad == 0u )
		{
			uint32_t j;
			for ( step = 0u; step < 4u; ++step )
			{
				for ( j = 0u; j < PROOF_HIDDEN; ++j )
					local[j] = LegValue(rank, j, 3u + step);
				error = cudaMemcpy(bufs[step], local,
					(uint64_t)PROOF_HIDDEN * 2u,
					cudaMemcpyHostToDevice);
			}
			Check(error == cudaSuccess, "legC H2D x4");
			memset(&submission, 0, sizeof(submission));
			submission.abi_version =
				SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
			submission.descriptor_bytes = sizeof(submission);
			submission.slot_index = rank;
			submission.active_sequence_count = 1u;
			submission.reserved0 = PROOF_HIDDEN;
			submission.cuda_stream = stream;
			submission.completion_function = CountingComplete;
			submission.completion_context = &counting;
			for ( step = 0u; step < 4u; ++step )
			{
				submission.ordinal = ordinal++;
				submission.local_device = bufs[step];
				submission.full_device = bufs[step];
				status = SparkTpDeviceCollectiveSubmitBf16(&collective,
					&submission);
				if ( status == SPARK_STATUS_OK )
					submits_ok++;
			}
			Check(submits_ok == 4u, "legC four back-to-back submits");
			error = cudaStreamSynchronize(stream);
			Check(error == cudaSuccess, "legC single trailing sync");
			bad = 0u;
			for ( step = 0u; step < 4u; ++step )
			{
				error = cudaMemcpy(after, bufs[step],
					(uint64_t)PROOF_HIDDEN * 2u,
					cudaMemcpyDeviceToHost);
				if ( error != cudaSuccess )
					{ bad++; continue; }
				for ( j = 0u; j < PROOF_HIDDEN; ++j )
				{
					uint32_t hits = 0u, r;
					for ( r = 0u; r < world; ++r )
						if ( (j % 7u) ==
							((r + 3u + step) % 7u) )
							hits++;
					if ( after[j] !=
						HostF32ToBf16(4.0f *
							(float)hits) )
						bad++;
				}
			}
			Check(bad == 0u, "legC per-op sums bit-exact");
			/* CountingComplete sees SEVEN completions, not eight:
			   leg B unpacks through its own HeadComplete context.
			   The last four recorded ordinals are leg C's strict chain. */
			Check(counting.seen == 7u &&
				counting.ordinals[(counting.seen - 4u) % 8u] ==
					first_ordinal &&
				counting.ordinals[(counting.seen - 1u) % 8u] ==
					first_ordinal + 3u,
				"legC completions carry the strict ordinal chain");
			for ( step = 0u; step < 4u; ++step )
				cudaFree(bufs[step]);
		}
	}

	/* THE LEG C DRAIN: every submission is stream-ordered, and ANY
	 * early-exit path above (a failed check, a refused submit) can reach
	 * this point with an enqueued collective still in flight - destroying
	 * the communicator with outstanding work aborts it and hangs the
	 * peers. Drain the stream unconditionally before teardown. */
	error = cudaStreamSynchronize(stream);
	if ( error != cudaSuccess )
		Check(error == cudaSuccess, "pre-destroy drain sync");
	SparkTpDeviceCollectiveDestroy(&collective);
	printf("K3_NCCL_MULTIRANK_RANK_RESULT rank=%u world=%u failures=%u\n",
		rank, world, g_failures);
	return g_failures == 0u ? 0 : 1;
}
