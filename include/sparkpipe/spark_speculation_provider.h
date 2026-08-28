#ifndef SPARKPIPE_SPARK_SPECULATION_PROVIDER_H
#define SPARKPIPE_SPARK_SPECULATION_PROVIDER_H

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Speculation-provider slot (the design in
 * docs/SPECULATION_PROVIDER_DESIGN.md, folded into the serving-adapter
 * template per the W2 sequencing). A speculation method is a capability
 * unit BEHIND the adapter: the interface covers LIFECYCLE + CONTRACT and
 * never the inner loop - draft generation and verification kernels stay
 * provider-owned, so the abstraction costs zero hot-path indirection
 * (the module-ABI principle applied to speculation).
 *
 * Two binding shapes must fit this one interface (both are proven by
 * tests/test_speculation_provider_slot.c):
 *   - the MODULE-PROVIDER shape: the provider is its own firmware-module
 *     like unit; a resolve function hands the adapter its ops table
 *     (the dispatch-policy backend shape).
 *   - the EMBEDDED-PROVIDER shape: the provider is a static ops table
 *     inside the family adapter, compiled into it (the block-drafter
 *     shape).
 * Family migration onto the slot is a later, cell-unchanged cutover; the
 * env contract of the block-diffusion drafter (the most receipted) moves
 * last.
 */

typedef enum SparkSpeculationProviderKind
{
	SPARK_SPECULATION_PROVIDER_MTP = 1,
	SPARK_SPECULATION_PROVIDER_DFLASH = 2,
	SPARK_SPECULATION_PROVIDER_DSPARK = 3,
	SPARK_SPECULATION_PROVIDER_DFLASH2 = 4,
	SPARK_SPECULATION_PROVIDER_DSPARK2 = 5
} SparkSpeculationProviderKind;

#define SPARK_SPECULATION_PROVIDER_ABI_VERSION 1u
#define SPARK_SPECULATION_PROVIDER_DESCRIPTOR_BYTES \
	((uint32_t)sizeof(SparkSpeculationProviderDescriptor))

/* The provider's published face: identity, depth envelope, and the ONE
 * canonical launch contract (the per-family SPEC_METHOD/DRAFT_COUNT
 * sprawl converges here; the schema names are the canonical keys). */
#define SPARK_SPECULATION_PROVIDER_MAX_ENV_SCHEMA 8u

typedef struct SparkSpeculationProviderDescriptor
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	SparkSpeculationProviderKind kind;
	const char *provider_id;
	uint32_t max_draft_token_count;
	uint32_t default_draft_token_count;
	/* Canonical launch contract, key names in order; 0 entries end. */
	const char *const *environment_schema;
	uint32_t environment_schema_count;
} SparkSpeculationProviderDescriptor;

/* Capability query input: the geometry a provider must fit. A refusal
 * carries the reason (the supports() -> yes/no/WHY of the design). */
typedef struct SparkSpeculationGeometryQuery
{
	uint32_t hidden_dimension;
	uint32_t layer_count;
	uint32_t attention_head_count;  /* 0 when not attention-shaped */
	uint32_t ffn_dimension;
} SparkSpeculationGeometryQuery;

/*
 * The verify contract - where the lease-advance bug class dies. ONE
 * accounting: the provider reports the accepted prefix length, the chain
 * width it verified, and the tokens-per-sequence the serving completion
 * must carry; the adapter copies these and nothing else.
 */
typedef struct SparkSpeculationVerifyContract
{
	uint32_t chain_width;                 /* drafts verified this round */
	uint32_t accepted_token_count;        /* the accepted prefix length */
	uint32_t tokens_per_sequence;         /* completion accounting */
	uint32_t chain_live;                  /* 0 = speculation dead this lane */
} SparkSpeculationVerifyContract;

/* KV interaction: which frames the provider reads/writes and the shape
 * of its block history, so the adapter can prove coverage before launch
 * without knowing the method. */
#define SPARK_SPECULATION_KV_FLAG_SCRATCH_FRAME UINT32_C(0x1)
#define SPARK_SPECULATION_KV_FLAG_TAIL_FRAME UINT32_C(0x2)
#define SPARK_SPECULATION_KV_FLAG_BLOCK_HISTORY UINT32_C(0x4)

typedef struct SparkSpeculationKvContract
{
	uint32_t frame_flags;
	uint32_t block_history_depth; /* 0 = no multi-block history */
} SparkSpeculationKvContract;

typedef struct SparkSpeculationDraftRequest
{
	const uint32_t *committed_ids;   /* the committed context tokens */
	uint32_t committed_count;
	uint32_t draft_token_count;      /* this round's requested depth */
} SparkSpeculationDraftRequest;

typedef struct SparkSpeculationDraft
{
	uint32_t *ids;                   /* provider-written draft tokens */
	uint32_t count;
	uint32_t first_draft_is_committed; /* providers emit the anchor first */
} SparkSpeculationDraft;

typedef struct SparkSpeculationProviderOps
{
	/* capability query: SPARK_STATUS_OK, or UNSUPPORTED with the reason */
	SparkStatus (*capability_query)(
		const SparkSpeculationGeometryQuery *geometry,
		char *refusal_buffer,
		uint32_t refusal_buffer_bytes);
	/* draft lifecycle: begin/next/cancel; the provider owns batching,
	 * its KV taps, and its graphs behind these calls */
	SparkStatus (*draft_begin)(void *provider_state,
		const SparkSpeculationDraftRequest *request);
	SparkStatus (*draft_next)(void *provider_state,
		SparkSpeculationDraft *draft);
	void (*draft_cancel)(void *provider_state);
	/* verify accounting: the ONE implementation of accepted-prefix
	 * semantics; the adapter never re-derives it */
	SparkStatus (*verify_account)(void *provider_state,
		uint32_t verified_count,
		SparkSpeculationVerifyContract *contract_out);
	/* KV interaction (static per provider; 0 = the descriptor's default) */
	const SparkSpeculationKvContract *(*kv_contract)(void *provider_state);
} SparkSpeculationProviderOps;

typedef struct SparkSpeculationProvider
{
	const SparkSpeculationProviderDescriptor *descriptor;
	const SparkSpeculationProviderOps *ops;
	void *provider_state;
} SparkSpeculationProvider;

SparkStatus SparkSpeculationProviderValidate(
	const SparkSpeculationProvider *provider);

#ifdef __cplusplus
}
#endif

#endif
