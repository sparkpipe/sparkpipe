#pragma once

#include <pthread.h>
#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_KV_LEDGER_MAGIC UINT64_C(0x31524744454c564b)
#define SPARK_KV_LEDGER_VERSION 1u
#define SPARK_KV_LEDGER_SLOTS 128u
#define SPARK_KV_LEDGER_DIRECTORY_BYTES 512u
#define SPARK_KV_LEDGER_NAME ".sparkpipe-kv-ledger"
#define SPARK_KV_NODE_QUOTA_BYTES UINT64_C(500000000000)
#define SPARK_KV_LEDGER_KIND_SNAPSHOT 1u
#define SPARK_KV_LEDGER_KIND_BACKING 2u
#define SPARK_KV_LEDGER_NO_SLOT UINT32_MAX
#define SPARK_KV_LEDGER_NO_OLDEST UINT64_MAX
#define SPARK_KV_LEDGER_POLL_SECONDS 1u
#define SPARK_KV_LEDGER_WAIT_NS UINT64_C(3000000000)

typedef struct SparkKvLedgerSlot
{
	uint32_t kind;
	uint32_t pid;
	uint64_t pid_start;
	uint64_t bytes;
	uint64_t oldest_used_seconds;
	uint64_t evict_request_bytes;
	char directory[SPARK_KV_LEDGER_DIRECTORY_BYTES];
} SparkKvLedgerSlot;

typedef struct SparkKvLedgerFile
{
	uint64_t magic;
	uint32_t version;
	uint32_t slot_count;
	uint64_t quota_bytes;
	uint64_t reserved[5];
	SparkKvLedgerSlot slots[SPARK_KV_LEDGER_SLOTS];
} SparkKvLedgerFile;

typedef struct SparkKvLedger
{
	int descriptor;
	uint32_t open;
	SparkKvLedgerFile *file;
	pthread_mutex_t mutex;
	char path[SPARK_KV_LEDGER_DIRECTORY_BYTES];
} SparkKvLedger;

typedef struct SparkKvLedgerUsage
{
	uint64_t quota_bytes;
	uint64_t used_bytes;
	uint64_t snapshot_bytes;
	uint64_t backing_bytes;
	uint32_t snapshot_slots;
	uint32_t backing_slots;
} SparkKvLedgerUsage;

typedef struct SparkKvLedgerVictim
{
	uint32_t slot;
	uint32_t live;
	uint64_t shortfall_bytes;
	uint64_t oldest_used_seconds;
	char directory[SPARK_KV_LEDGER_DIRECTORY_BYTES];
} SparkKvLedgerVictim;

typedef uint64_t (*SparkKvLedgerEvictOwn)(void *context,uint64_t older_than_seconds);

SparkStatus SparkKvLedgerDefaultPath(char *path,uint32_t capacity);
SparkStatus SparkKvLedgerOpen(SparkKvLedger *ledger,const char *path,uint64_t quota_bytes);
void SparkKvLedgerClose(SparkKvLedger *ledger);
SparkStatus SparkKvLedgerJoin(SparkKvLedger *ledger,uint32_t kind,const char *directory,uint32_t *slot_out);
void SparkKvLedgerLeave(SparkKvLedger *ledger,uint32_t slot);
void SparkKvLedgerPublish(SparkKvLedger *ledger,uint32_t slot,uint64_t bytes,uint64_t oldest_used_seconds);
SparkStatus SparkKvLedgerRecord(SparkKvLedger *ledger,const char *directory,uint64_t bytes,uint64_t oldest_used_seconds);
SparkStatus SparkKvLedgerReserve(SparkKvLedger *ledger,uint32_t slot,uint64_t own_bytes,uint64_t incoming_bytes,uint64_t own_oldest_seconds,SparkKvLedgerVictim *victim);
void SparkKvLedgerRequest(SparkKvLedger *ledger,const SparkKvLedgerVictim *victim);
uint64_t SparkKvLedgerTakeRequest(SparkKvLedger *ledger,uint32_t slot,uint64_t *others_oldest_seconds);
SparkStatus SparkKvLedgerSample(SparkKvLedger *ledger,SparkKvLedgerUsage *usage);
void SparkKvLedgerForgetMissing(SparkKvLedger *ledger);
SparkStatus SparkKvLedgerMakeRoom(SparkKvLedger *ledger,uint32_t slot,uint64_t incoming_bytes,uint64_t wait_ns,SparkKvLedgerEvictOwn evict_own,void *context,
	uint64_t (*own_bytes)(void *context),uint64_t (*own_oldest)(void *context));
uint64_t SparkKvLedgerProcessStart(uint32_t pid);

#ifdef __cplusplus
}
#endif
