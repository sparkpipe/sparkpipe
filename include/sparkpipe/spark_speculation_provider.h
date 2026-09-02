#ifndef SPARKPIPE_SPARK_SPECULATION_PROVIDER_H
#define SPARKPIPE_SPARK_SPECULATION_PROVIDER_H

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif


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

#define SPARK_SPECULATION_PROVIDER_MAX_ENV_SCHEMA 8u

typedef struct SparkSpeculationProviderDescriptor
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	SparkSpeculationProviderKind kind;
	const char *provider_id;
	uint32_t max_draft_token_count;
	uint32_t default_draft_token_count;
	const char *const *environment_schema;
	uint32_t environment_schema_count;
} SparkSpeculationProviderDescriptor;

typedef struct SparkSpeculationGeometryQuery
{
	uint32_t hidden_dimension;
	uint32_t layer_count;
	uint32_t attention_head_count;
	uint32_t ffn_dimension;
} SparkSpeculationGeometryQuery;

typedef struct SparkSpeculationVerifyContract
{
	uint32_t chain_width;
	uint32_t accepted_token_count;
	uint32_t tokens_per_sequence;
	uint32_t chain_live;
} SparkSpeculationVerifyContract;

#define SPARK_SPECULATION_KV_FLAG_SCRATCH_FRAME UINT32_C(0x1)
#define SPARK_SPECULATION_KV_FLAG_TAIL_FRAME UINT32_C(0x2)
#define SPARK_SPECULATION_KV_FLAG_BLOCK_HISTORY UINT32_C(0x4)

typedef struct SparkSpeculationKvContract
{
	uint32_t frame_flags;
	uint32_t block_history_depth;
} SparkSpeculationKvContract;

typedef struct SparkSpeculationDraftRequest
{
	const uint32_t *committed_ids;
	uint32_t committed_count;
	uint32_t draft_token_count;
} SparkSpeculationDraftRequest;

typedef struct SparkSpeculationDraft
{
	uint32_t *ids;
	uint32_t count;
	uint32_t first_draft_is_committed;
} SparkSpeculationDraft;

typedef struct SparkSpeculationProviderOps
{
	SparkStatus (*capability_query)(
		const SparkSpeculationGeometryQuery *geometry,
		char *refusal_buffer,
		uint32_t refusal_buffer_bytes);
	SparkStatus (*draft_begin)(void *provider_state,
		const SparkSpeculationDraftRequest *request);
	SparkStatus (*draft_next)(void *provider_state,
		SparkSpeculationDraft *draft);
	void (*draft_cancel)(void *provider_state);
	SparkStatus (*verify_account)(void *provider_state,
		uint32_t verified_count,
		SparkSpeculationVerifyContract *contract_out);
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
