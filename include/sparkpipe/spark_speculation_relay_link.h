#pragma once

#include <stdint.h>

#include "sparkpipe/spark_speculation_policy.h"
#include "sparkpipe/spark_speculation_relay_draft.h"
#include "sparkpipe/spark_speculation_tap.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_SPECULATION_RELAY_RTT_BUCKETS 1024u
#define SPARK_SPECULATION_RELAY_RTT_BUCKET_NS 2000u
#define SPARK_SPECULATION_RELAY_HISTORY_TOKENS 64u
#define SPARK_SPECULATION_RELAY_SHADOW_SLOTS 32u
#define SPARK_SPECULATION_RELAY_SOCKET_BUFFER_BYTES (8u << 20)

typedef struct SparkSpeculationRelayLink
{
	int socket_fd;
	uint32_t peer_length;
	uint8_t peer[128];
	uint32_t send_buffer_bytes;
	uint32_t receive_buffer_bytes;
	uint64_t datagrams_sent;
	uint64_t bytes_sent;
	uint64_t send_errors;
	uint64_t datagrams_received;
	uint64_t datagrams_foreign;
	uint64_t tap_records_sent;
	uint64_t tap_records_unsent;
} SparkSpeculationRelayLink;

typedef struct SparkSpeculationRelayShadowSlot
{
	uint64_t sequence_id;
	uint64_t anchor_position;
	uint32_t token_count;
	uint32_t open;
	uint32_t token_ids[SPARK_SPECULATION_RELAY_MAX_TOKENS];
} SparkSpeculationRelayShadowSlot;

typedef struct SparkSpeculationRelayRemote
{
	SparkSpeculationRelayLink *link;
	SparkSpeculationRelayDraft mailbox;
	uint64_t await_ns;
	uint32_t shadow;
	uint32_t round_open;
	uint64_t request_sent_ns;
	uint64_t history_sequence;
	uint64_t history_first_position;
	uint64_t history_next_position;
	uint64_t history_sent_position;
	uint32_t history_tokens[SPARK_SPECULATION_RELAY_HISTORY_TOKENS];
	SparkSpeculationRelayShadowSlot shadow_slots[SPARK_SPECULATION_RELAY_SHADOW_SLOTS];
	uint32_t shadow_next_slot;
	uint64_t requests;
	uint64_t unsynced;
	uint64_t unsent;
	uint64_t answered_in_time;
	uint64_t answered_late;
	uint64_t deadline_misses;
	uint64_t shadow_rounds;
	uint64_t shadow_proposed;
	uint64_t shadow_accepted;
	uint64_t shadow_evicted;
	uint64_t shadow_accepted_at[SPARK_SPECULATION_RELAY_MAX_TOKENS + 1u];
	uint64_t rtt_count;
	uint64_t rtt_max_ns;
	uint64_t rtt_buckets[SPARK_SPECULATION_RELAY_RTT_BUCKETS];
} SparkSpeculationRelayRemote;

uint64_t SparkSpeculationRelayNowNs(void);
SparkStatus SparkSpeculationRelayLinkOpen(SparkSpeculationRelayLink *link,const char *local_endpoint,const char *peer_endpoint);
void SparkSpeculationRelayLinkClose(SparkSpeculationRelayLink *link);
SparkStatus SparkSpeculationRelayLinkSend(SparkSpeculationRelayLink *link,const uint8_t *bytes,uint32_t length);
SparkStatus SparkSpeculationRelayLinkReceive(SparkSpeculationRelayLink *link,uint8_t *buffer,uint32_t capacity,uint32_t *length_out);
SparkStatus SparkSpeculationRelayLinkSendTap(SparkSpeculationRelayLink *link,uint64_t fingerprint,const SparkSpeculationTapRecord *record,const uint8_t *payload,uint32_t record_bytes,uint32_t payload_max);
SparkStatus SparkSpeculationRelayRemoteInitialize(SparkSpeculationRelayRemote *remote,SparkSpeculationRelayLink *link,uint64_t engine_generation,uint32_t vocab_size,uint64_t await_ns,uint32_t shadow);
SparkStatus SparkSpeculationRelayRemoteObserve(SparkSpeculationRelayRemote *remote,uint64_t sequence_id,uint64_t first_position,const uint32_t *token_ids,uint32_t token_count);
SparkStatus SparkSpeculationRelayRemotePoll(SparkSpeculationRelayRemote *remote);
SparkStatus SparkSpeculationRelayRemoteRound(SparkSpeculationRelayRemote *remote,uint64_t sequence_id,uint64_t anchor_position,uint32_t requested_token_count,SparkSpeculationPolicyDraftResult *result);
SparkStatus SparkSpeculationRelayRemoteDraftTokens(void *context,const SparkSpeculationPolicyDraftRequest *request,SparkSpeculationPolicyDraftResult *result);
uint64_t SparkSpeculationRelayRemoteRttPercentileNs(const SparkSpeculationRelayRemote *remote,uint32_t per_mille);

#ifdef __cplusplus
}
#endif
