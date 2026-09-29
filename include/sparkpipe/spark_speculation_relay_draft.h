#pragma once

#include <stdint.h>

#include "sparkpipe/spark_speculation_policy.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_SPECULATION_RELAY_MAGIC 0x31525053u
#define SPARK_SPECULATION_RELAY_VERSION 1u
#define SPARK_SPECULATION_RELAY_KIND_REQUEST 1u
#define SPARK_SPECULATION_RELAY_KIND_DRAFT 2u
#define SPARK_SPECULATION_RELAY_MAX_TOKENS 16u
#define SPARK_SPECULATION_RELAY_HEADER_BYTES 48u
#define SPARK_SPECULATION_RELAY_FRAME_BYTES (SPARK_SPECULATION_RELAY_HEADER_BYTES + 4u * SPARK_SPECULATION_RELAY_MAX_TOKENS)

typedef struct SparkSpeculationRelayFrame
{
	uint32_t kind;
	uint32_t token_count;
	uint64_t engine_generation;
	uint64_t round_id;
	uint64_t sequence_id;
	uint64_t anchor_position;
	uint32_t anchor_token;
	uint32_t requested_token_count;
	uint32_t token_ids[SPARK_SPECULATION_RELAY_MAX_TOKENS];
} SparkSpeculationRelayFrame;

typedef struct SparkSpeculationRelayDraft
{
	uint32_t vocab_size;
	uint64_t engine_generation;
	uint64_t round_id;
	SparkSpeculationRelayFrame pending;
	SparkSpeculationRelayFrame ready;
	uint32_t ready_valid;
	uint64_t issued;
	uint64_t delivered;
	uint64_t stale;
	uint64_t misses;
	uint64_t used;
} SparkSpeculationRelayDraft;

SparkStatus SparkSpeculationRelayEncode(const SparkSpeculationRelayFrame *frame,uint32_t vocab_size,uint8_t bytes[SPARK_SPECULATION_RELAY_FRAME_BYTES]);
SparkStatus SparkSpeculationRelayDecode(const uint8_t *bytes,uint32_t length,uint32_t vocab_size,SparkSpeculationRelayFrame *frame);
SparkStatus SparkSpeculationRelayDraftInitialize(SparkSpeculationRelayDraft *relay,uint64_t engine_generation,uint32_t vocab_size);
SparkStatus SparkSpeculationRelayDraftIssue(SparkSpeculationRelayDraft *relay,uint64_t sequence_id,uint64_t anchor_position,uint32_t anchor_token,const uint32_t *committed_token_ids,uint32_t committed_token_count,uint32_t requested_token_count,uint8_t bytes[SPARK_SPECULATION_RELAY_FRAME_BYTES]);
SparkStatus SparkSpeculationRelayDraftDeliver(SparkSpeculationRelayDraft *relay,const uint8_t *bytes,uint32_t length);
SparkStatus SparkSpeculationRelayDraftTokens(void *context,const SparkSpeculationPolicyDraftRequest *request,SparkSpeculationPolicyDraftResult *result);

#ifdef __cplusplus
}
#endif
