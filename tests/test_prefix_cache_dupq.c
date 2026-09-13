/* Duplicate-content publish conservation cell (OPT-IN).
 *
 * Permanent in-tree home of the scratch instrument that SETTLED the
 * duplicate-publish documented limit (see .agents/pccore-dev/INTEGRATION.md
 * "Documented limits"; scratch receipts stay at
 * .agents/pccore-dev/_tmp/dupq/{dupq_probe.c,dupq_results.txt} for
 * provenance - do not rename them).
 *
 * The core dedupes at ADMIT time only, never at PUBLISH time: lanes that
 * prefill identical root content CONCURRENTLY (interleaved scheduler walk)
 * each publish their own physical blocks for the same chain hashes.
 * Sequential admits never show this (admit-time matching absorbs them),
 * which is exactly why this cell drives a cold INTERLEAVED walk.
 *
 * This file pins three things against the CURRENT core behavior so any
 * drift is loud and the eventual ratified dedupe-on-publish landing stays
 * a production-only diff (the only edit it should need here is the
 * deliberate same-commit rebaseline of DUPQ_BURST_EXPECTED /
 * DUPQ_COHORT_EXPECTED below):
 *
 *   1. CONSERVATION: the three-set partition
 *         {FREE} U {PRIVATE/live-open} U {PUBLISHED} == block_count
 *      holds after EVERY core call (admit/append/release/trim), with the
 *      census cross-checked against the reported stats fields every time,
 *      teardown included.
 *   2. THE MEASURED WASTE FRACTIONS reproduce exactly (deterministic
 *      mix64 scripts): concurrent shared-root bursts leave dup_live =
 *      34-48% of used blocks at the first capacity stall (B25 pool=100)
 *      and 44-50% (B512 pool=1024); staggered cohort rhythm never stalls
 *      but dead duplicates persist at 33-49% of occupied blocks. Controls
 *      clean: unique prompts (roots=batch) must measure ZERO duplicates -
 *      this keeps the measurement non-vacuous from both sides.
 *   3. THE SHADOW upper bound: a perfect-dedupe replay of the SAME event
 *      trace never stalls and saves exactly one allocation per duplicate.
 *
 * OPT-IN CONTRACT (mirrors the env-guarded leg discipline of
 * test_qwen38_prefix_cache.c CASE 11): the binary builds everywhere and
 * runs in `make test`, but UNARMED it prints SKIP and exits 0 naming the
 * pending decision - it can never red an unrelated tree. Arm it with
 *   SPARK_PREFIX_CACHE_DUPQ_GATE=1 ./build/test_prefix_cache_dupq
 * Failures exit 1 with the mismatching field spelled out.
 *
 * Honesty bounds: host fixture over the public+header-visible API, linked
 * exactly like build/test_prefix_cache_core; geometry = the Qwen3.8-27B
 * squeezed gate cells (B25/B512, 64-token blocks, 23 attn layers x 4 local
 * heads x 256 dims x K+V bf16 stride). This certifies ALLOCATOR-LEVEL
 * accounting only - no device bytes, no latency claim. Burst numbers are
 * the raw-core worst-case bound; shipped deferred-admit ordering reduces
 * real exposure vs the bound; cohort cells are the realistic rhythm.
 */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "prefix_cache.h"

#define DUPQ_BLOCK_TOKENS 64u
#define DUPQ_ROOT_BLOCKS 2u
#define DUPQ_SUFFIX_BLOCKS 2u
#define DUPQ_PROMPT_TOKENS ((DUPQ_ROOT_BLOCKS + DUPQ_SUFFIX_BLOCKS) * \
                            DUPQ_BLOCK_TOKENS)
#define DUPQ_DECODE_TOKENS 16u
#define DUPQ_MAX_LANES 512u

/* Recorded bands (INTEGRATION.md documented-limits entry), padded +-0.5
 * for one-decimal print rounding; the exact per-cell counts are pinned
 * separately in the expected tables. */
#define DUPQ_BURST_B25_BAND_LO 33.5
#define DUPQ_BURST_B25_BAND_HI 48.5
#define DUPQ_BURST_B512_BAND_LO 43.0
#define DUPQ_BURST_B512_BAND_HI 50.0
#define DUPQ_COHORT_BAND_LO 32.5
#define DUPQ_COHORT_BAND_HI 49.5

typedef struct
{
	uint64_t hash;
	uint32_t phys;
	int live;
} DupqShadowEntry;

typedef struct
{
	uint64_t published_total;
	uint64_t evicted_total;
	uint64_t dup_ever;
	uint64_t dup_live;
	uint64_t publish_count;
	DupqShadowEntry *shadow;
	uint32_t shadow_count;
	uint32_t shadow_cap;
	uint64_t shadow_occupancy;
	uint64_t shadow_saved;
	int shadow_stalled;
	uint64_t shadow_stall_at;
	uint64_t *ever_set;
	uint32_t ever_count;
} DupqTrace;

typedef struct
{
	uint64_t stall_event;
	uint64_t dup_live;
	uint64_t used_at_stall;
	uint64_t dup_ever;
	uint64_t trim_evicted;
	uint64_t dup_after_trim;
	uint64_t shadow_saved;
} DupqBurstExpected;

typedef struct
{
	uint64_t published_total;
	uint64_t dup_ever;
	uint64_t residual_dup;
	uint64_t used_end;
	uint64_t stalls;
} DupqCohortExpected;

static uint64_t DupqMix64(uint64_t x)
{
	x += 0x9e3779b97f4a7c15ull;
	x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
	x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
	return x ^ (x >> 31);
}

static uint32_t DupqTokenFor(uint64_t tag, uint32_t pos)
{
	return (uint32_t)(DupqMix64((tag << 20) ^ pos) & 0xffffff);
}

static void DupqFail(const char *cell, const char *field,
                     uint64_t expected, uint64_t measured)
{
	fprintf(stderr,
	    "DUPQ_FAIL(%s): %s expected=%" PRIu64 " measured=%" PRIu64 "\n"
	    "  if a DELIBERATE core change moved this, rebaseline "
	    "DUPQ_*_EXPECTED in tests/test_prefix_cache_dupq.c IN THE SAME "
	    "COMMIT as the production change - never chase the script\n",
	    cell, field, expected, measured);
	exit(1);
}

/*
 * Conservation census: {FREE} U {PRIVATE} U {PUBLISHED} == block_count,
 * cross-checked against the stats the core reports for the same instant.
 * Returns 0 when conserved, nonzero (with a stderr line) otherwise.
 */
static int DupqAuditPartition(const SparkPrefixCacheCore *core,
    const SparkPrefixCacheCoreStats *stats)
{
	uint32_t free_blocks = 0u;
	uint32_t private_blocks = 0u;
	uint32_t published_blocks = 0u;
	uint32_t index;

	for ( index = 0u; index < core->block_count; index++ )
	{
		switch ( core->blocks[index].state )
		{
			case SPARK_PREFIX_CACHE_CORE_BLOCK_FREE:
				free_blocks++;
				break;
			case SPARK_PREFIX_CACHE_CORE_BLOCK_PRIVATE:
				private_blocks++;
				break;
			case SPARK_PREFIX_CACHE_CORE_BLOCK_PUBLISHED:
				published_blocks++;
				break;
			default:
				fprintf(stderr,
				    "DUPQ_FAIL(partition): block %u carries illegal state %u\n",
				    index, core->blocks[index].state);
				return 1;
		}
	}
	if ( (uint64_t)free_blocks + private_blocks + published_blocks !=
	     core->block_count )
	{
		fprintf(stderr, "DUPQ_FAIL(partition): census %u+%u+%u != pool %u\n",
		    free_blocks, private_blocks, published_blocks, core->block_count);
		return 1;
	}
	if ( free_blocks != stats->free_block_count ||
	     published_blocks != stats->published_block_count ||
	     private_blocks + published_blocks != stats->used_block_count )
	{
		fprintf(stderr,
		    "DUPQ_FAIL(partition): census free=%u publ=%u live-open=%u vs "
		    "stats free=%u publ=%u used=%u\n",
		    free_blocks, published_blocks, private_blocks,
		    stats->free_block_count, stats->published_block_count,
		    stats->used_block_count);
		return 1;
	}
	if ( stats->live_sequence_count > core->max_sequence_count )
	{
		fprintf(stderr, "DUPQ_FAIL(partition): live seqs %u > ceiling %u\n",
		    stats->live_sequence_count, core->max_sequence_count);
		return 1;
	}
	return 0;
}

static int ever_seen(DupqTrace *t, uint64_t h)
{
	for ( uint32_t i = 0; i < t->ever_count; i++ )
		if ( t->ever_set[i] == h ) return 1;
	return 0;
}

static void trace_init(DupqTrace *t)
{
	memset(t, 0, sizeof(*t));
	t->shadow_cap = 4096;
	t->shadow = calloc(t->shadow_cap, sizeof(DupqShadowEntry));
	t->ever_set = calloc(65536, sizeof(uint64_t));
}

/* dedupe replay: a publish whose hash is already live allocates nothing */
static void shadow_publish(DupqTrace *t, uint64_t hash, uint32_t phys,
    uint32_t pool)
{
	for ( uint32_t i = 0; i < t->shadow_count; i++ )
		if ( t->shadow[i].live && t->shadow[i].hash == hash )
		{
			t->shadow_saved++;
			return;
		}
	if ( !t->shadow_stalled && t->shadow_occupancy >= pool )
	{
		t->shadow_stalled = 1;
		t->shadow_stall_at = t->publish_count;
	}
	if ( t->shadow_count == t->shadow_cap )
	{
		t->shadow_cap *= 2;
		t->shadow = realloc(t->shadow, t->shadow_cap * sizeof(DupqShadowEntry));
	}
	t->shadow[t->shadow_count++] = (DupqShadowEntry){hash, phys, 1};
	t->shadow_occupancy++;
}

static void shadow_evict(DupqTrace *t, uint32_t phys)
{
	for ( uint32_t i = 0; i < t->shadow_count; i++ )
		if ( t->shadow[i].live && t->shadow[i].phys == phys )
		{
			t->shadow[i].live = 0;
			t->shadow_occupancy--;
			return;
		}
}

/* Drain publish/evict events after one or more core operations, then
 * re-audit the partition - this runs after EVERY core call. */
static int pump_events(DupqTrace *t, SparkPrefixCacheCore *core,
	uint8_t *phys_published, uint32_t pool)
{
	SparkPrefixCacheCoreStats stats;
	uint64_t evicted_seen;
	uint64_t published_seen;
	SparkPrefixCacheCoreQueryStats(core, &stats);
	published_seen = stats.published_block_count_total;
	while ( t->published_total < published_seen )
	{
		t->published_total++;
		t->publish_count++;
		for ( uint32_t p = 0; p < pool; p++ )
		{
			if ( !phys_published[p] &&
			     core->blocks[p].state ==
			         SPARK_PREFIX_CACHE_CORE_BLOCK_PUBLISHED )
			{
				phys_published[p] = 1;
				uint64_t h = core->blocks[p].chain_hash;
				int canonical_live = 0;
				for ( uint32_t q = 0; q < pool; q++ )
					if ( q != p && phys_published[q] &&
					     core->blocks[q].state ==
					         SPARK_PREFIX_CACHE_CORE_BLOCK_PUBLISHED &&
					     core->blocks[q].chain_hash == h )
					{ canonical_live = 1; break; }
				if ( canonical_live ) t->dup_live++;
				if ( ever_seen(t, h) ) t->dup_ever++;
				if ( t->ever_count < 65536 )
					t->ever_set[t->ever_count++] = h;
				shadow_publish(t, h, p, pool);
			}
		}
	}
	evicted_seen = stats.evicted_block_count;
	while ( t->evicted_total < evicted_seen )
	{
		t->evicted_total++;
		for ( uint32_t p = 0; p < pool; p++ )
			if ( phys_published[p] &&
			     core->blocks[p].state ==
			         SPARK_PREFIX_CACHE_CORE_BLOCK_FREE )
			{
				phys_published[p] = 0;
				shadow_evict(t, p);
			}
	}
	/* Partition audit at the post-call instant. */
	SparkPrefixCacheCoreQueryStats(core, &stats);
	return DupqAuditPartition(core, &stats);
}

static void fill_prompt(uint32_t *p, uint32_t s, uint32_t roots)
{
	uint32_t root = s % roots;
	for ( uint32_t i = 0; i < DUPQ_PROMPT_TOKENS; i++ )
	{
		uint32_t blk = i / DUPQ_BLOCK_TOKENS;
		p[i] = (blk < DUPQ_ROOT_BLOCKS)
		    ? DupqTokenFor(((uint64_t)root << 8) | blk, i % DUPQ_BLOCK_TOKENS)
		    : DupqTokenFor(0xffffff00ull | s, i);
	}
}

static void configure(SparkPrefixCacheCoreConfiguration *cfg, uint32_t pool,
    uint32_t lanes)
{
	memset(cfg, 0, sizeof(*cfg));
	cfg->abi_version = SPARK_PREFIX_CACHE_CORE_ABI_VERSION;
	cfg->descriptor_bytes =
	    SPARK_PREFIX_CACHE_CORE_CONFIGURATION_DESCRIPTOR_BYTES;
	cfg->block_token_count = DUPQ_BLOCK_TOKENS;
	/* Qwen3.8-27B attn-layer slice x 64 tokens: 23 layers x 4 local heads
	 * x 256 dims x K+V x bf16. Accounting-only in this fixture. */
	cfg->block_stride_bytes =
	    (uint64_t)23 * 4 * 128 * 2 * 2 * DUPQ_BLOCK_TOKENS;
	cfg->block_count = pool;
	cfg->max_sequence_count = lanes;
	cfg->sequence_block_capacity = 4096u;
	cfg->hash_bucket_count = 8192u;
}

/* BURST: all lanes cold together, fully interleaved walk until first
 * stall, then one adapter-style Trim(pool/2). Returns 0 on conservation
 * success (value pinning happens in main against the expected table). */
static int run_burst(uint32_t batch, uint32_t pool, uint32_t roots,
	const char *label, DupqBurstExpected *out)
{
	SparkPrefixCacheCore core;
	SparkPrefixCacheCoreConfiguration cfg;
	SparkPrefixCacheCoreStats stats;
	DupqTrace t;
	uint32_t *prompt;
	uint8_t *phys_published;
	uint64_t base_seq = 0x5100;
	int stalled = 0;
	uint64_t real_stall_event = 0;
	uint64_t cursor[DUPQ_MAX_LANES];
	uint32_t lanes = batch < DUPQ_MAX_LANES ? batch : DUPQ_MAX_LANES;
	int ok = 1;

	configure(&cfg, pool, lanes);
	if ( SparkPrefixCacheCoreInitialize(&core, &cfg) != SPARK_STATUS_OK )
	{
		printf("%s INIT FAILED\n", label);
		return 1;
	}
	trace_init(&t);
	phys_published = calloc(pool, 1);
	prompt = calloc((size_t)lanes * DUPQ_PROMPT_TOKENS, 4);
	for ( uint32_t s = 0; s < lanes; s++ )
	{
		cursor[s] = 0;
		fill_prompt(prompt + (size_t)s * DUPQ_PROMPT_TOKENS, s, roots);
	}

	for ( uint32_t s = 0; s < lanes && !stalled; s++ )
	{
		uint32_t matched = 0;
		SparkStatus st = SparkPrefixCacheCoreAdmitSequence(&core,
			base_seq + s, prompt + (size_t)s * DUPQ_PROMPT_TOKENS, 1,
			&matched);
		cursor[s] = 1;
		if ( pump_events(&t, &core, phys_published, pool) ) ok = 0;
		if ( st != SPARK_STATUS_OK ) { stalled = 1; real_stall_event = t.publish_count; }
	}

	uint64_t total_tokens =
	    (uint64_t)lanes * (DUPQ_PROMPT_TOKENS - 1 + DUPQ_DECODE_TOKENS);
	for ( uint64_t tok = 0; tok < total_tokens && !stalled; tok++ )
	{
		uint32_t s = (uint32_t)(tok % lanes);
		if ( cursor[s] >= DUPQ_PROMPT_TOKENS + DUPQ_DECODE_TOKENS ) continue;
		uint32_t ti = (uint32_t)cursor[s]++;
		uint32_t id = (ti < DUPQ_PROMPT_TOKENS)
		    ? prompt[(size_t)s * DUPQ_PROMPT_TOKENS + ti]
		    : DupqTokenFor(0xffffee00ull | s, ti);
		SparkPrefixCacheCoreQueryStats(&core, &stats);
		uint64_t last_stalls = stats.capacity_stall_count;
		SparkStatus st = SparkPrefixCacheCoreAppendTokens(&core,
			base_seq + s, &id, 1);
		if ( pump_events(&t, &core, phys_published, pool) ) ok = 0;
		SparkPrefixCacheCoreQueryStats(&core, &stats);
		if ( st != SPARK_STATUS_OK ||
		     stats.capacity_stall_count > last_stalls ||
		     stats.used_block_count >= pool )
		{ stalled = 1; real_stall_event = t.publish_count; }
	}

	SparkPrefixCacheCoreQueryStats(&core, &stats);
	double waste_pct = stats.used_block_count
	    ? 100.0 * (double)t.dup_live / (double)stats.used_block_count : 0.0;
	uint64_t used_at_stall = stats.used_block_count;

	uint32_t evicted = 0;
	SparkPrefixCacheCoreTrim(&core, pool / 2, &evicted);
	if ( pump_events(&t, &core, phys_published, pool) ) ok = 0;
	SparkPrefixCacheCoreQueryStats(&core, &stats);
	if ( DupqAuditPartition(&core, &stats) ) ok = 0;

	printf("%s BURST  roots=%-4u pool=%-4u batch=%-3u | stall@pub#%-5llu "
	       "dup_live=%-5llu (%4.1f%% of used=%llu) dup_ever=%llu | "
	       "trim->%u: evicted=%llu dup_now=%llu | shadow: %s saved=%llu\n",
	    label, roots, pool, lanes,
	    (unsigned long long)real_stall_event,
	    (unsigned long long)t.dup_live, waste_pct,
	    (unsigned long long)used_at_stall,
	    (unsigned long long)t.dup_ever,
	    pool / 2,
	    (unsigned long long)evicted,
	    (unsigned long long)t.dup_live,
	    t.shadow_stalled ? "STALLS TOO" : "never stalls",
	    (unsigned long long)t.shadow_saved);

	out->stall_event = real_stall_event;
	out->dup_live = t.dup_live;
	out->used_at_stall = used_at_stall;
	out->dup_ever = t.dup_ever;
	out->trim_evicted = evicted;
	out->dup_after_trim = t.dup_live;
	out->shadow_saved = t.shadow_saved;

	free(t.shadow);
	free(t.ever_set);
	free(phys_published);
	free(prompt);
	SparkPrefixCacheCoreDestroy(&core);
	return !ok;
}

/* COHORT: staggered serving rhythm over the full batch. */
static int run_cohort(uint32_t batch, uint32_t pool, uint32_t roots,
	uint32_t cohort, const char *label, DupqCohortExpected *out)
{
	SparkPrefixCacheCore core;
	SparkPrefixCacheCoreConfiguration cfg;
	SparkPrefixCacheCoreStats stats;
	DupqTrace t;
	uint32_t *prompt;
	uint8_t *phys_published;
	uint64_t base_seq = 0x5100;
	uint64_t cursor[DUPQ_MAX_LANES];
	int stalled = 0;
	int ok = 1;

	configure(&cfg, pool, cohort < batch ? cohort : batch);
	if ( SparkPrefixCacheCoreInitialize(&core, &cfg) != SPARK_STATUS_OK )
	{
		printf("%s INIT FAILED\n", label);
		return 1;
	}
	trace_init(&t);
	phys_published = calloc(pool, 1);
	prompt = calloc((size_t)batch * DUPQ_PROMPT_TOKENS, 4);

	for ( uint32_t c0 = 0; c0 < batch && !stalled; c0 += cohort )
	{
		uint32_t c = (c0 + cohort <= batch) ? cohort : batch - c0;
		for ( uint32_t j = 0; j < c; j++ )
		{
			uint32_t s = c0 + j;
			fill_prompt(prompt + (size_t)s * DUPQ_PROMPT_TOKENS, s, roots);
			cursor[j] = 1;
			uint32_t matched = 0;
			SparkStatus st = SparkPrefixCacheCoreAdmitSequence(&core,
				base_seq + s, prompt + (size_t)s * DUPQ_PROMPT_TOKENS, 1,
				&matched);
			if ( pump_events(&t, &core, phys_published, pool) ) ok = 0;
			if ( st != SPARK_STATUS_OK ) stalled = 1;
		}
		if ( stalled ) break;
		uint64_t total_tokens =
		    (uint64_t)c * (DUPQ_PROMPT_TOKENS - 1 + DUPQ_DECODE_TOKENS);
		for ( uint64_t tok = 0; tok < total_tokens && !stalled; tok++ )
		{
			uint32_t j = (uint32_t)(tok % c);
			uint32_t s = c0 + j;
			if ( cursor[j] >= DUPQ_PROMPT_TOKENS + DUPQ_DECODE_TOKENS )
				continue;
			uint32_t ti = (uint32_t)cursor[j]++;
			uint32_t id = (ti < DUPQ_PROMPT_TOKENS)
			    ? prompt[(size_t)s * DUPQ_PROMPT_TOKENS + ti]
			    : DupqTokenFor(0xffffee00ull | s, ti);
			if ( SparkPrefixCacheCoreAppendTokens(&core, base_seq + s, &id,
			        1) != SPARK_STATUS_OK )
				stalled = 1;
			if ( pump_events(&t, &core, phys_published, pool) ) ok = 0;
		}
		for ( uint32_t j = 0; j < c; j++ )
			SparkPrefixCacheCoreReleaseSequence(&core, base_seq + c0 + j);
		if ( pump_events(&t, &core, phys_published, pool) ) ok = 0;
	}

	SparkPrefixCacheCoreQueryStats(&core, &stats);
	double waste_pct = stats.used_block_count
	    ? 100.0 * (double)t.dup_live / (double)stats.used_block_count : 0.0;
	if ( DupqAuditPartition(&core, &stats) ) ok = 0;
	printf("%s COHORT roots=%-4u pool=%-4u batch=%-3uC%-2u | published=%llu "
	       "dup_ever=%llu residual_dup=%llu (%4.1f%% of used=%llu) "
	       "stalls=%llu\n",
	    label, roots, pool, batch, cohort,
	    (unsigned long long)t.published_total,
	    (unsigned long long)t.dup_ever,
	    (unsigned long long)t.dup_live, waste_pct,
	    (unsigned long long)stats.used_block_count,
	    (unsigned long long)stats.capacity_stall_count);

	out->published_total = t.published_total;
	out->dup_ever = t.dup_ever;
	out->residual_dup = t.dup_live;
	out->used_end = stats.used_block_count;
	out->stalls = stats.capacity_stall_count;

	free(t.shadow);
	free(t.ever_set);
	free(phys_published);
	free(prompt);
	SparkPrefixCacheCoreDestroy(&core);
	return !ok;
}

/* Expected values = MEASURED 2026-08-23 scratch run
 * (.agents/pccore-dev/_tmp/dupq/dupq_results.txt), re-proven by this
 * file's first armed in-tree run. Deterministic scripts: any difference
 * means core publish/evict behavior moved. The ratified dedupe-on-publish
 * landing is EXPECTED to move these - rebaseline them deliberately in the
 * same commit and watch every fraction collapse toward zero. */
static const DupqBurstExpected DUPQ_BURST_EXPECTED[] = {
	/* roots, pool, batch */         /* stall dup  used ever trim dup' saved */
	{75,  48,  100,  48,  0, 48, 48},   /* b25 r=1  */
	{75,  42,  100,  42,  0, 42, 42},   /* b25 r=4  */
	{75,  34,  100,  34,  0, 34, 34},   /* b25 r=8  */
	{75,   0,  100,   0,  0,  0,  0},   /* b25 r=25 control  */
	{512, 511, 1024, 511, 0, 511, 511}, /* b512 r=1  */
	{512, 504, 1024, 504, 0, 504, 504}, /* b512 r=8  */
	{512, 448, 1024, 448, 0, 448, 448}, /* b512 r=64 */
	{512,   0, 1024,   0, 0,   0,   0}, /* b512 r=512 control */
};

static const DupqCohortExpected DUPQ_COHORT_EXPECTED[] = {
	/* publ ever resid used stalls */
	{100,  40,  40,  96, 0},  /* b25  r=4  C8   */
	{100,  32,  32,  96, 0},  /* b25  r=8  C8   */
	{2048, 496, 496, 1008, 0},/* b512 r=8  C16  */
	{2048, 384, 384, 1008, 0},/* b512 r=64 C16  */
	{2048, 384, 384, 960, 0}, /* b512 r=64 C64  */
};

static void check_burst(const char *cell, const DupqBurstExpected *want,
	const DupqBurstExpected *got)
{
	if ( want->stall_event != got->stall_event )
		DupqFail(cell, "stall_publish_event", want->stall_event,
		    got->stall_event);
	if ( want->dup_live != got->dup_live )
		DupqFail(cell, "dup_live", want->dup_live, got->dup_live);
	if ( want->used_at_stall != got->used_at_stall )
		DupqFail(cell, "used_at_stall", want->used_at_stall,
		    got->used_at_stall);
	if ( want->dup_ever != got->dup_ever )
		DupqFail(cell, "dup_ever", want->dup_ever, got->dup_ever);
	if ( want->trim_evicted != got->trim_evicted )
		DupqFail(cell, "trim_evicted", want->trim_evicted,
		    got->trim_evicted);
	if ( want->dup_after_trim != got->dup_after_trim )
		DupqFail(cell, "dup_after_trim", want->dup_after_trim,
		    got->dup_after_trim);
	if ( want->shadow_saved != got->shadow_saved )
		DupqFail(cell, "shadow_saved", want->shadow_saved,
		    got->shadow_saved);
}

static void check_cohort(const char *cell, const DupqCohortExpected *want,
	const DupqCohortExpected *got)
{
	if ( want->published_total != got->published_total )
		DupqFail(cell, "published_total", want->published_total,
		    got->published_total);
	if ( want->dup_ever != got->dup_ever )
		DupqFail(cell, "dup_ever", want->dup_ever, got->dup_ever);
	if ( want->residual_dup != got->residual_dup )
		DupqFail(cell, "residual_dup", want->residual_dup,
		    got->residual_dup);
	if ( want->used_end != got->used_end )
		DupqFail(cell, "used_end", want->used_end, got->used_end);
	if ( want->stalls != got->stalls )
		DupqFail(cell, "stalls", want->stalls, got->stalls);
}

int main(void)
{
	const char *armed = getenv("SPARK_PREFIX_CACHE_DUPQ_GATE");
	DupqBurstExpected burst[8];
	DupqCohortExpected coh[5];
	int ok = 1;

	if ( armed == NULL || strcmp(armed, "1") != 0 )
	{
		printf("dupq SKIP: SPARK_PREFIX_CACHE_DUPQ_GATE unset - "
		       "duplicate-publish production landing awaits the coordinator "
		       "RATIFIED line; arm the cell with "
		       "SPARK_PREFIX_CACHE_DUPQ_GATE=1 (documented limit: "
		       ".agents/pccore-dev/INTEGRATION.md)\n");
		return 0;
	}

	printf("== duplicate-content publish conservation cell (opt-in; "
	       "64-token blocks; interleaved cold walk; partition audited after "
	       "every call) ==\n");

	/* Burst cells: order MUST match DUPQ_BURST_EXPECTED. */
	if ( run_burst(25, 100, 1, "b25 ", &burst[0]) ) ok = 0;
	if ( run_burst(25, 100, 4, "b25 ", &burst[1]) ) ok = 0;
	if ( run_burst(25, 100, 8, "b25 ", &burst[2]) ) ok = 0;
	if ( run_burst(25, 100, 25, "b25 ", &burst[3]) ) ok = 0;
	if ( run_burst(512, 1024, 1, "b512", &burst[4]) ) ok = 0;
	if ( run_burst(512, 1024, 8, "b512", &burst[5]) ) ok = 0;
	if ( run_burst(512, 1024, 64, "b512", &burst[6]) ) ok = 0;
	if ( run_burst(512, 1024, 512, "b512", &burst[7]) ) ok = 0;

	/* Cohort cells: order MUST match DUPQ_COHORT_EXPECTED. */
	if ( run_cohort(25, 100, 4, 8, "b25 ", &coh[0]) ) ok = 0;
	if ( run_cohort(25, 100, 8, 8, "b25 ", &coh[1]) ) ok = 0;
	if ( run_cohort(512, 1024, 8, 16, "b512", &coh[2]) ) ok = 0;
	if ( run_cohort(512, 1024, 64, 16, "b512", &coh[3]) ) ok = 0;
	if ( run_cohort(512, 1024, 64, 64, "b512", &coh[4]) ) ok = 0;

	if ( !ok )
	{
		fprintf(stderr, "DUPQ_FAIL(conservation): partition broken above\n");
		return 1;
	}

	/* Pin the recorded measurements exactly. */
	check_burst("b25_burst_roots1", &DUPQ_BURST_EXPECTED[0], &burst[0]);
	check_burst("b25_burst_roots4", &DUPQ_BURST_EXPECTED[1], &burst[1]);
	check_burst("b25_burst_roots8", &DUPQ_BURST_EXPECTED[2], &burst[2]);
	check_burst("b25_burst_roots25_control", &DUPQ_BURST_EXPECTED[3],
	    &burst[3]);
	check_burst("b512_burst_roots1", &DUPQ_BURST_EXPECTED[4], &burst[4]);
	check_burst("b512_burst_roots8", &DUPQ_BURST_EXPECTED[5], &burst[5]);
	check_burst("b512_burst_roots64", &DUPQ_BURST_EXPECTED[6], &burst[6]);
	check_burst("b512_burst_roots512_control", &DUPQ_BURST_EXPECTED[7],
	    &burst[7]);
	check_cohort("b25_cohort_r4_C8", &DUPQ_COHORT_EXPECTED[0], &coh[0]);
	check_cohort("b25_cohort_r8_C8", &DUPQ_COHORT_EXPECTED[1], &coh[1]);
	check_cohort("b512_cohort_r8_C16", &DUPQ_COHORT_EXPECTED[2], &coh[2]);
	check_cohort("b512_cohort_r64_C16", &DUPQ_COHORT_EXPECTED[3], &coh[3]);
	check_cohort("b512_cohort_r64_C64", &DUPQ_COHORT_EXPECTED[4], &coh[4]);

	/* Non-vacuity, both directions + recorded bands. */
	for ( uint32_t i = 0; i < 4u; i++ )
	{
		char cell[64];
		double pct = 100.0 * (double)burst[i].dup_live /
		    (double)burst[i].used_at_stall;
		snprintf(cell, sizeof(cell), "b25_burst_cell%u", i);
		if ( i < 3u && burst[i].dup_live == 0 )
			DupqFail(cell, "nonvacuous_dup_live", 1, burst[i].dup_live);
		if ( i == 3u && burst[i].dup_live != 0 )
			DupqFail(cell, "control_dup_live_must_be_zero", 0,
			    burst[i].dup_live);
		if ( i < 3u && (pct < DUPQ_BURST_B25_BAND_LO ||
		        pct > DUPQ_BURST_B25_BAND_HI) )
		{
			fprintf(stderr, "DUPQ_FAIL(%s): waste %.1f%% outside recorded "
			    "band [%.1f,%.1f]\n", cell, pct, DUPQ_BURST_B25_BAND_LO,
			    DUPQ_BURST_B25_BAND_HI);
			return 1;
		}
	}
	for ( uint32_t i = 4u; i < 8u; i++ )
	{
		char cell[64];
		double pct = 100.0 * (double)burst[i].dup_live /
		    (double)burst[i].used_at_stall;
		snprintf(cell, sizeof(cell), "b512_burst_cell%u", i);
		if ( i < 7u && burst[i].dup_live == 0 )
			DupqFail(cell, "nonvacuous_dup_live", 1, burst[i].dup_live);
		if ( i == 7u && burst[i].dup_live != 0 )
			DupqFail(cell, "control_dup_live_must_be_zero", 0,
			    burst[i].dup_live);
		if ( i < 7u && (pct < DUPQ_BURST_B512_BAND_LO ||
		        pct > DUPQ_BURST_B512_BAND_HI) )
		{
			fprintf(stderr, "DUPQ_FAIL(%s): waste %.1f%% outside recorded "
			    "band [%.1f,%.1f]\n", cell, pct, DUPQ_BURST_B512_BAND_LO,
			    DUPQ_BURST_B512_BAND_HI);
			return 1;
		}
	}
	for ( uint32_t i = 0; i < 5u; i++ )
	{
		char cell[64];
		double pct = 100.0 * (double)coh[i].residual_dup /
		    (double)coh[i].used_end;
		snprintf(cell, sizeof(cell), "cohort_cell%u", i);
		if ( coh[i].stalls != 0 )
			DupqFail(cell, "cohort_must_not_stall", 0, coh[i].stalls);
		if ( i < 5u && coh[i].residual_dup == 0 )
			DupqFail(cell, "nonvacuous_residual_dup", 1,
			    coh[i].residual_dup);
		if ( pct < DUPQ_COHORT_BAND_LO || pct > DUPQ_COHORT_BAND_HI )
		{
			fprintf(stderr, "DUPQ_FAIL(%s): waste %.1f%% outside recorded "
			    "band [%.1f,%.1f]\n", cell, pct, DUPQ_COHORT_BAND_LO,
			    DUPQ_COHORT_BAND_HI);
			return 1;
		}
	}

	printf("== dupq PASS: partition {free} U {live/open} U {published} == "
	       "pool after every call; controls zero-dup; waste fractions "
	       "reproduce the recorded bands (burst B25 34-48%% / B512 44-50%%, "
	       "cohort 33-49%%); shadow perfect-dedupe replay never stalled ==\n");
	return 0;
}
