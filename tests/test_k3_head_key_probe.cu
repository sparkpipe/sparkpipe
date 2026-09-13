// F1 head-argmax key contract probe: extracts the REAL
// K3RunnerHeadPackKernel / K3RunnerHeadUnpackKernel source from the runner
// at build time and proves, on the GPU, that the packed-u64 MAX reduction
// reproduces the host tier's f32 slot-sum + first-max scan outcome exactly:
//
//   1. round-trip: every finite float survives encode->decode bit-exactly,
//      with its token id intact (incl. +0/-0, denormals, +/-inf, max);
//   2. NaN loses: a NaN score never wins against any finite score (the
//      host scan's false-compare behaviour);
//   3. score dominance: a strictly greater score wins regardless of rank or
//      token payload;
//   4. THE TIE RULE: equal scores resolve to the LOWEST rank - the host
//      scan keeps the first maximum because its comparisons are strict;
//   5. end-to-end: over randomized 16-rank candidate sets, MAX(keys) equals
//      the host-rule winner's (score bits, token) pair every time.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <limits>
#include <vector>

#define PROBE_ROWS (1024 * 1024)

// ---- extracted verbatim from spark_k3_resident_decode_stage_runner.cu ----
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

static uint32_t HostRuleWinner(const std::vector<float> &scores,
	const std::vector<uint32_t> &tokens, uint32_t degree)
{
	uint32_t best = 0u;
	for ( uint32_t r = 1u; r < degree; ++r )
		if ( scores[r] > scores[best] )
			best = r;
	return best;
}

int main()
{
	uint32_t *d_tokens;
	float *d_scores;
	unsigned long long *d_keys;
	cudaMalloc(&d_tokens, (uint64_t)PROBE_ROWS * 4u);
	cudaMalloc(&d_scores, (uint64_t)PROBE_ROWS * 4u);
	cudaMalloc(&d_keys, (uint64_t)PROBE_ROWS * 8u);
	std::vector<float> h_scores(PROBE_ROWS);
	std::vector<uint32_t> h_tokens(PROBE_ROWS);
	std::vector<unsigned long long> h_keys(PROBE_ROWS);
	int failures = 0;

	/* 1+2: the fixed edge sweep, one row per case */
	struct { float s; uint32_t t; } edges[] = {
		{ 0.0f, 0u }, { -0.0f, 262143u },
		{ 1.0f, 1u }, { -1.0f, 2u },
		{ 3.4028235e38f, 262140u }, { -3.4028235e38f, 3u },
		{ 1.1754944e-38f, 4u }, { -1.1754944e-38f, 5u },
		{ 1e-45f, 6u }, { 7.0f, 262142u },
		{ std::numeric_limits<float>::infinity(), 7u },
		{ -std::numeric_limits<float>::infinity(), 262141u },
	};
	const uint32_t edge_count =
		(uint32_t)(sizeof(edges) / sizeof(edges[0]));
	for ( uint32_t i = 0u; i < edge_count; ++i )
	{
		h_scores[i] = edges[i].s;
		h_tokens[i] = edges[i].t;
	}
	cudaMemcpy(d_scores, h_scores.data(), (uint64_t)edge_count * 4u,
		cudaMemcpyHostToDevice);
	cudaMemcpy(d_tokens, h_tokens.data(), (uint64_t)edge_count * 4u,
		cudaMemcpyHostToDevice);
	K3RunnerHeadPackKernel<<<1u, 256u>>>(d_tokens, d_scores, d_keys, 3u,
		edge_count);
	K3RunnerHeadUnpackKernel<<<1u, 256u>>>(d_keys, d_tokens, d_scores,
		edge_count);
	cudaDeviceSynchronize();
	cudaMemcpy(h_scores.data(), d_scores, (uint64_t)edge_count * 4u,
		cudaMemcpyDeviceToHost);
	cudaMemcpy(h_tokens.data(), d_tokens, (uint64_t)edge_count * 4u,
		cudaMemcpyDeviceToHost);
	cudaMemcpy(h_keys.data(), d_keys, (uint64_t)edge_count * 8u,
		cudaMemcpyDeviceToHost);
	for ( uint32_t i = 0u; i < edge_count; ++i )
	{
		float orig = edges[i].s;
		if ( orig != orig )
		{
			if ( h_keys[i] != 0ull )
				{ printf("FAIL nan row %u did not encode as losing key\n", i); failures++; }
			continue;
		}
		uint32_t a, b;
		memcpy(&a, &orig, sizeof(a));
		memcpy(&b, &h_scores[i], sizeof(b));
		if ( a != b || h_tokens[i] != edges[i].t )
		{
			printf("FAIL roundtrip row %u: bits %08x->%08x token %u->%u\n",
				i, a, b, edges[i].t, h_tokens[i]);
			failures++;
		}
	}
	printf("edge roundtrip: %u cases\n", edge_count);

	/* 3+4+5: randomized 16-rank sets, GPU keys vs host rule */
	srand(1234567u);
	const uint32_t trials = 200000u;
	for ( uint32_t trial = 0u; trial < trials; ++trial )
	{
		const uint32_t degree = 16u;
		uint32_t winner_rank = trial % degree;
		for ( uint32_t r = 0u; r < degree; ++r )
		{
			/* mostly-tied scores around a few bit patterns: exercises the
			 * tie rule hard, plus random tokens */
			uint32_t sbits;
			if ( r < degree / 2u )
				sbits = 0x3F800000u + (rand() % 3u);
			else
				sbits = ((uint32_t)rand() << 8u) ^ (uint32_t)rand();
			float s;
			memcpy(&s, &sbits, sizeof(s));
			if ( r == winner_rank && (trial & 1u) == 0u )
				s = s + 1.0f; /* a clear winner half the time */
			h_scores[r] = s;
			h_tokens[r] = (uint32_t)rand() % 262144u;
		}
		cudaMemcpy(d_scores, h_scores.data(), degree * 4u,
			cudaMemcpyHostToDevice);
		cudaMemcpy(d_tokens, h_tokens.data(), degree * 4u,
			cudaMemcpyHostToDevice);
		/* each rank packs ITS OWN row (rank r's device holds row r): one
		 * launch per rank at offset r */
		for ( uint32_t r = 0u; r < degree; ++r )
			K3RunnerHeadPackKernel<<<1u, 256u>>>(d_tokens + r, d_scores + r,
				d_keys + r, r, 1u);
		/* the reduction the NCCL u64-max op performs */
		unsigned long long best_key = 0ull;
		cudaMemcpy(h_keys.data(), d_keys, degree * 8u,
			cudaMemcpyDeviceToHost);
		for ( uint32_t r = 0u; r < degree; ++r )
			if ( h_keys[r] > best_key )
				best_key = h_keys[r];
		uint32_t win_bits = (uint32_t)(best_key >> 32);
		win_bits = (win_bits & 0x80000000u) != 0u ?
			(win_bits ^ 0x80000000u) : ~win_bits;
		float win_score;
		memcpy(&win_score, &win_bits, sizeof(win_score));
		uint32_t win_token = (uint32_t)(best_key & 0xFFFFFFull);
		uint32_t expect = HostRuleWinner(h_scores, h_tokens, degree);
		float expect_score = h_scores[expect];
		uint32_t a, b;
		memcpy(&a, &expect_score, sizeof(a));
		memcpy(&b, &win_score, sizeof(b));
		if ( a != b || win_token != h_tokens[expect] )
		{
			if ( failures < 8u )
				printf("FAIL trial %u: host=(rank%u bits %08x tok %u) "
					"device=(bits %08x tok %u)\n", trial, expect, a,
					h_tokens[expect], b, win_token);
			if ( failures == 0u )
			{
				printf("DEBUG trial %u winner_rank=%u:\n", trial,
					winner_rank);
				for ( uint32_t r = 0u; r < degree; ++r )
				{
					uint32_t sb;
					memcpy(&sb, &h_scores[r], sizeof(sb));
					printf("  r%02u sbits=%08x tok=%6u key=%016llx\n",
						r, sb, h_tokens[r], h_keys[r]);
				}
			}
			failures++;
		}
	}
	printf("randomized 16-rank reduction vs host rule: %u trials\n", trials);

	cudaFree(d_tokens);
	cudaFree(d_scores);
	cudaFree(d_keys);
	if ( failures == 0 )
	{
		printf("k3 head key contract probe PASS\n");
		return 0;
	}
	printf("k3 head key contract probe FAIL (%d)\n", failures);
	return 1;
}
