#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <errno.h>
#include "sparkpipe/spark_clock.h"
#include <fcntl.h>
#include <pwd.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "sparkpipe/spark_kv_ledger.h"
#include "sparkpipe/spark_kv_snapshot.h"
#include "sparkpipe/spark_error_site.h"

uint64_t SparkKvLedgerProcessStart(uint32_t pid)
{
#if defined(__linux__)
	char path[64],line[1024],*cursor;
	unsigned long long start = 0u;
	uint32_t field;
	FILE *file;
	(void)snprintf(path,sizeof(path),"/proc/%u/stat",pid);
	file = fopen(path,"r");
	if ( file == 0 )
		return(0u);
	cursor = fgets(line,sizeof(line),file);
	(void)fclose(file);
	if ( cursor == 0 || (cursor = strrchr(line,')')) == 0 )
		return(0u);
	for (field=2u; field<22u && cursor != 0; field++)
		cursor = strchr(cursor + 1,' ');
	if ( cursor == 0 || sscanf(cursor + 1,"%llu",&start) != 1 )
		return(0u);
	return((uint64_t)start + 1u);
#else
	return(pid != 0u && kill((pid_t)pid,0) == 0 ? 1u : 0u);
#endif
}

static uint32_t SparkKvLedgerSlotLive(const SparkKvLedgerSlot *slot)
{
	return(slot->pid != 0u && slot->pid_start != 0u && SparkKvLedgerProcessStart(slot->pid) == slot->pid_start ? 1u : 0u);
}

static uint32_t SparkKvLedgerSlotCounted(const SparkKvLedgerSlot *slot)
{
	return(slot->kind == SPARK_KV_LEDGER_KIND_SNAPSHOT || (slot->kind == SPARK_KV_LEDGER_KIND_BACKING && SparkKvLedgerSlotLive(slot) != 0u) ? 1u : 0u);
}

static void SparkKvLedgerLock(SparkKvLedger *ledger)
{
	(void)pthread_mutex_lock(&ledger->mutex);
	while ( flock(ledger->descriptor,LOCK_EX) != 0 && errno == EINTR )
		;
}

static void SparkKvLedgerUnlock(SparkKvLedger *ledger)
{
	(void)flock(ledger->descriptor,LOCK_UN);
	(void)pthread_mutex_unlock(&ledger->mutex);
}

static uint32_t SparkKvLedgerValid(const SparkKvLedger *ledger,uint32_t slot)
{
	return(ledger != 0 && ledger->open != 0u && slot < SPARK_KV_LEDGER_SLOTS ? 1u : 0u);
}

SparkStatus SparkKvLedgerDefaultPath(char *path,uint32_t capacity)
{
	const struct passwd *owner = getpwuid(geteuid());
	if ( path == 0 || owner == 0 || owner->pw_dir == 0 || owner->pw_dir[0] != '/' )
		SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
	if ( (size_t)snprintf(path,capacity,"%s/%s",owner->pw_dir,SPARK_KV_LEDGER_NAME) >= capacity )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkKvLedgerOpen(SparkKvLedger *ledger,const char *path,uint64_t quota_bytes)
{
	struct stat info;
	SparkKvLedgerFile *file;
	SparkStatus status = SPARK_STATUS_OK;
	uint32_t fresh;
	if ( ledger == 0 || path == 0 || path[0] != '/' || strlen(path) >= sizeof(ledger->path) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memset(ledger,0,sizeof(*ledger));
	ledger->descriptor = open(path,O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW,0600);
	if ( ledger->descriptor < 0 )
	{
		fprintf(stderr,"kv ledger %s cannot be opened: %s\n",path,strerror(errno));
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	}
	while ( flock(ledger->descriptor,LOCK_EX) != 0 && errno == EINTR )
		;
	fresh = fstat(ledger->descriptor,&info) == 0 && info.st_size == 0 ? 1u : 0u;
	if ( fresh != 0u && ftruncate(ledger->descriptor,(off_t)sizeof(SparkKvLedgerFile)) != 0 )
		status = SPARK_STATUS_IO_ERROR;
	else if ( fresh == 0u && (fstat(ledger->descriptor,&info) != 0 || (uint64_t)info.st_size != sizeof(SparkKvLedgerFile)) )
		status = SPARK_STATUS_SCHEMA_ERROR;
	file = status == SPARK_STATUS_OK ? (SparkKvLedgerFile *)mmap(0,sizeof(SparkKvLedgerFile),PROT_READ | PROT_WRITE,MAP_SHARED,ledger->descriptor,0) : 0;
	if ( status == SPARK_STATUS_OK && file == MAP_FAILED )
	{
		file = 0;
		status = SPARK_STATUS_IO_ERROR;
	}
	if ( status == SPARK_STATUS_OK && fresh != 0u )
	{
		memset(file,0,sizeof(*file));
		file->magic = SPARK_KV_LEDGER_MAGIC;
		file->version = SPARK_KV_LEDGER_VERSION;
		file->slot_count = SPARK_KV_LEDGER_SLOTS;
		file->quota_bytes = SPARK_KV_NODE_QUOTA_BYTES;
	}
	if ( status == SPARK_STATUS_OK && (file->magic != SPARK_KV_LEDGER_MAGIC || file->version != SPARK_KV_LEDGER_VERSION || file->slot_count != SPARK_KV_LEDGER_SLOTS) )
		status = SPARK_STATUS_SCHEMA_ERROR;
	if ( status == SPARK_STATUS_OK && quota_bytes != 0u )
		file->quota_bytes = quota_bytes;
	(void)flock(ledger->descriptor,LOCK_UN);
	if ( status == SPARK_STATUS_OK && pthread_mutex_init(&ledger->mutex,0) != 0 )
		status = SPARK_STATUS_INTERNAL_ERROR;
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"kv ledger %s refused: status=%s (a ledger is %llu bytes of version %u)\n",path,SparkStatusToString(status),
			(unsigned long long)sizeof(SparkKvLedgerFile),SPARK_KV_LEDGER_VERSION);
		if ( file != 0 )
			(void)munmap(file,sizeof(*file));
		(void)close(ledger->descriptor);
		memset(ledger,0,sizeof(*ledger));
		ledger->descriptor = -1;
		SPARK_RETURN(status);
	}
	ledger->file = file;
	ledger->open = 1u;
	memcpy(ledger->path,path,strlen(path) + 1u);
	return(SPARK_STATUS_OK);
}

void SparkKvLedgerClose(SparkKvLedger *ledger)
{
	if ( ledger == 0 || ledger->open == 0u )
		return;
	(void)munmap(ledger->file,sizeof(*ledger->file));
	(void)close(ledger->descriptor);
	(void)pthread_mutex_destroy(&ledger->mutex);
	memset(ledger,0,sizeof(*ledger));
	ledger->descriptor = -1;
}

static uint32_t SparkKvLedgerReusable(const SparkKvLedgerSlot *slot)
{
	struct stat info;
	if ( slot->kind == 0u )
		return(1u);
	if ( SparkKvLedgerSlotLive(slot) != 0u )
		return(0u);
	return(slot->kind == SPARK_KV_LEDGER_KIND_BACKING || stat(slot->directory,&info) != 0 ? 1u : 0u);
}

static uint32_t SparkKvLedgerFind(const SparkKvLedgerFile *file,uint32_t kind,const char *directory)
{
	uint32_t index;
	for (index=0u; index<SPARK_KV_LEDGER_SLOTS; index++)
		if ( file->slots[index].kind == kind && strcmp(file->slots[index].directory,directory) == 0 )
			return(index);
	return(SPARK_KV_LEDGER_NO_SLOT);
}

static uint32_t SparkKvLedgerAllocate(SparkKvLedgerFile *file,uint32_t kind,const char *directory)
{
	uint32_t index;
	for (index=0u; index<SPARK_KV_LEDGER_SLOTS; index++)
		if ( SparkKvLedgerReusable(&file->slots[index]) != 0u )
		{
			memset(&file->slots[index],0,sizeof(file->slots[index]));
			file->slots[index].kind = kind;
			file->slots[index].oldest_used_seconds = SPARK_KV_LEDGER_NO_OLDEST;
			memcpy(file->slots[index].directory,directory,strlen(directory) + 1u);
			return(index);
		}
	return(SPARK_KV_LEDGER_NO_SLOT);
}

SparkStatus SparkKvLedgerJoin(SparkKvLedger *ledger,uint32_t kind,const char *directory,uint32_t *slot_out)
{
	SparkKvLedgerSlot *slot;
	uint32_t index;
	if ( ledger == 0 || ledger->open == 0u || slot_out == 0 || directory == 0 || directory[0] != '/' || strlen(directory) >= SPARK_KV_LEDGER_DIRECTORY_BYTES ||
		(kind != SPARK_KV_LEDGER_KIND_SNAPSHOT && kind != SPARK_KV_LEDGER_KIND_BACKING) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	SparkKvLedgerLock(ledger);
	index = SparkKvLedgerFind(ledger->file,kind,directory);
	if ( index != SPARK_KV_LEDGER_NO_SLOT && SparkKvLedgerSlotLive(&ledger->file->slots[index]) != 0u )
	{
		SparkKvLedgerUnlock(ledger);
		fprintf(stderr,"kv ledger %s: %s is held by live process %u\n",ledger->path,directory,ledger->file->slots[index].pid);
		SPARK_FAIL(SPARK_STATUS_BUSY);
	}
	if ( index == SPARK_KV_LEDGER_NO_SLOT )
		index = SparkKvLedgerAllocate(ledger->file,kind,directory);
	if ( index == SPARK_KV_LEDGER_NO_SLOT )
	{
		SparkKvLedgerUnlock(ledger);
		fprintf(stderr,"kv ledger %s: all %u slots hold live or present stores; %s cannot join\n",ledger->path,SPARK_KV_LEDGER_SLOTS,directory);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	slot = &ledger->file->slots[index];
	slot->pid = (uint32_t)getpid();
	slot->pid_start = SparkKvLedgerProcessStart(slot->pid);
	slot->evict_request_bytes = 0u;
	if ( kind == SPARK_KV_LEDGER_KIND_BACKING )
	{
		slot->bytes = 0u;
		slot->oldest_used_seconds = SPARK_KV_LEDGER_NO_OLDEST;
	}
	SparkKvLedgerUnlock(ledger);
	*slot_out = index;
	return(SPARK_STATUS_OK);
}

void SparkKvLedgerLeave(SparkKvLedger *ledger,uint32_t slot)
{
	if ( SparkKvLedgerValid(ledger,slot) == 0u )
		return;
	SparkKvLedgerLock(ledger);
	if ( ledger->file->slots[slot].kind == SPARK_KV_LEDGER_KIND_BACKING )
		memset(&ledger->file->slots[slot],0,sizeof(ledger->file->slots[slot]));
	else
	{
		ledger->file->slots[slot].pid = 0u;
		ledger->file->slots[slot].pid_start = 0u;
	}
	SparkKvLedgerUnlock(ledger);
}

void SparkKvLedgerPublish(SparkKvLedger *ledger,uint32_t slot,uint64_t bytes,uint64_t oldest_used_seconds)
{
	if ( SparkKvLedgerValid(ledger,slot) == 0u )
		return;
	SparkKvLedgerLock(ledger);
	ledger->file->slots[slot].bytes = bytes;
	ledger->file->slots[slot].oldest_used_seconds = oldest_used_seconds;
	SparkKvLedgerUnlock(ledger);
}

SparkStatus SparkKvLedgerRecord(SparkKvLedger *ledger,const char *directory,uint64_t bytes,uint64_t oldest_used_seconds)
{
	uint32_t index;
	if ( ledger == 0 || ledger->open == 0u || directory == 0 || directory[0] != '/' || strlen(directory) >= SPARK_KV_LEDGER_DIRECTORY_BYTES )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	SparkKvLedgerLock(ledger);
	index = SparkKvLedgerFind(ledger->file,SPARK_KV_LEDGER_KIND_SNAPSHOT,directory);
	if ( index == SPARK_KV_LEDGER_NO_SLOT && bytes != 0u )
		index = SparkKvLedgerAllocate(ledger->file,SPARK_KV_LEDGER_KIND_SNAPSHOT,directory);
	if ( index != SPARK_KV_LEDGER_NO_SLOT && SparkKvLedgerSlotLive(&ledger->file->slots[index]) == 0u )
	{
		ledger->file->slots[index].pid = 0u;
		ledger->file->slots[index].pid_start = 0u;
		ledger->file->slots[index].bytes = bytes;
		ledger->file->slots[index].oldest_used_seconds = bytes != 0u ? oldest_used_seconds : SPARK_KV_LEDGER_NO_OLDEST;
		ledger->file->slots[index].evict_request_bytes = 0u;
	}
	SparkKvLedgerUnlock(ledger);
	if ( index == SPARK_KV_LEDGER_NO_SLOT && bytes != 0u )
	{
		fprintf(stderr,"kv ledger %s: all %u slots hold live or present stores; %s holds %llu uncounted bytes\n",ledger->path,SPARK_KV_LEDGER_SLOTS,directory,
			(unsigned long long)bytes);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	return(SPARK_STATUS_OK);
}

static uint64_t SparkKvLedgerUsedLocked(const SparkKvLedgerFile *file,SparkKvLedgerUsage *usage)
{
	uint32_t index;
	memset(usage,0,sizeof(*usage));
	usage->quota_bytes = file->quota_bytes;
	for (index=0u; index<SPARK_KV_LEDGER_SLOTS; index++)
	{
		if ( SparkKvLedgerSlotCounted(&file->slots[index]) == 0u )
			continue;
		if ( file->slots[index].kind == SPARK_KV_LEDGER_KIND_SNAPSHOT )
		{
			usage->snapshot_bytes += file->slots[index].bytes;
			usage->snapshot_slots++;
		}
		else
		{
			usage->backing_bytes += file->slots[index].bytes;
			usage->backing_slots++;
		}
	}
	usage->used_bytes = usage->snapshot_bytes + usage->backing_bytes;
	return(usage->used_bytes);
}

SparkStatus SparkKvLedgerReserve(SparkKvLedger *ledger,uint32_t slot,uint64_t own_bytes,uint64_t incoming_bytes,uint64_t own_oldest_seconds,SparkKvLedgerVictim *victim)
{
	SparkKvLedgerUsage usage;
	SparkKvLedgerSlot *candidate;
	uint32_t index,chosen = SPARK_KV_LEDGER_NO_SLOT;
	if ( SparkKvLedgerValid(ledger,slot) == 0u || victim == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memset(victim,0,sizeof(*victim));
	victim->slot = SPARK_KV_LEDGER_NO_SLOT;
	SparkKvLedgerLock(ledger);
	ledger->file->slots[slot].bytes = own_bytes;
	ledger->file->slots[slot].oldest_used_seconds = own_oldest_seconds;
	(void)SparkKvLedgerUsedLocked(ledger->file,&usage);
	if ( incoming_bytes <= usage.quota_bytes && usage.used_bytes <= usage.quota_bytes - incoming_bytes )
	{
		ledger->file->slots[slot].bytes = own_bytes + incoming_bytes;
		SparkKvLedgerUnlock(ledger);
		return(SPARK_STATUS_OK);
	}
	for (index=0u; index<SPARK_KV_LEDGER_SLOTS; index++)
	{
		candidate = &ledger->file->slots[index];
		if ( candidate->kind != SPARK_KV_LEDGER_KIND_SNAPSHOT || candidate->bytes == 0u || candidate->oldest_used_seconds == SPARK_KV_LEDGER_NO_OLDEST )
			continue;
		if ( chosen == SPARK_KV_LEDGER_NO_SLOT || candidate->oldest_used_seconds < ledger->file->slots[chosen].oldest_used_seconds )
			chosen = index;
	}
	victim->shortfall_bytes = incoming_bytes > usage.quota_bytes ? incoming_bytes : usage.used_bytes + incoming_bytes - usage.quota_bytes;
	if ( incoming_bytes <= usage.quota_bytes && chosen != SPARK_KV_LEDGER_NO_SLOT )
	{
		victim->slot = chosen;
		victim->live = chosen == slot ? 1u : SparkKvLedgerSlotLive(&ledger->file->slots[chosen]);
		victim->oldest_used_seconds = ledger->file->slots[chosen].oldest_used_seconds;
		memcpy(victim->directory,ledger->file->slots[chosen].directory,sizeof(victim->directory));
	}
	SparkKvLedgerUnlock(ledger);
	return(SPARK_STATUS_CAPACITY_EXCEEDED);
}

void SparkKvLedgerRequest(SparkKvLedger *ledger,const SparkKvLedgerVictim *victim)
{
	SparkKvLedgerSlot *slot;
	if ( victim == 0 || SparkKvLedgerValid(ledger,victim->slot) == 0u )
		return;
	SparkKvLedgerLock(ledger);
	slot = &ledger->file->slots[victim->slot];
	if ( slot->kind == SPARK_KV_LEDGER_KIND_SNAPSHOT && strcmp(slot->directory,victim->directory) == 0 && slot->evict_request_bytes < victim->shortfall_bytes )
		slot->evict_request_bytes = victim->shortfall_bytes;
	SparkKvLedgerUnlock(ledger);
}

uint64_t SparkKvLedgerTakeRequest(SparkKvLedger *ledger,uint32_t slot,uint64_t *others_oldest_seconds)
{
	uint64_t request,oldest = SPARK_KV_LEDGER_NO_OLDEST;
	uint32_t index;
	if ( SparkKvLedgerValid(ledger,slot) == 0u )
		return(0u);
	SparkKvLedgerLock(ledger);
	request = ledger->file->slots[slot].evict_request_bytes;
	ledger->file->slots[slot].evict_request_bytes = 0u;
	for (index=0u; index<SPARK_KV_LEDGER_SLOTS; index++)
		if ( index != slot && ledger->file->slots[index].kind == SPARK_KV_LEDGER_KIND_SNAPSHOT && ledger->file->slots[index].bytes != 0u &&
			ledger->file->slots[index].oldest_used_seconds < oldest )
			oldest = ledger->file->slots[index].oldest_used_seconds;
	SparkKvLedgerUnlock(ledger);
	if ( others_oldest_seconds != 0 )
		*others_oldest_seconds = oldest;
	return(request);
}

SparkStatus SparkKvLedgerSample(SparkKvLedger *ledger,SparkKvLedgerUsage *usage)
{
	if ( ledger == 0 || ledger->open == 0u || usage == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	SparkKvLedgerLock(ledger);
	(void)SparkKvLedgerUsedLocked(ledger->file,usage);
	SparkKvLedgerUnlock(ledger);
	return(SPARK_STATUS_OK);
}

void SparkKvLedgerForgetMissing(SparkKvLedger *ledger)
{
	struct stat info;
	uint32_t index;
	if ( ledger == 0 || ledger->open == 0u )
		return;
	SparkKvLedgerLock(ledger);
	for (index=0u; index<SPARK_KV_LEDGER_SLOTS; index++)
		if ( ledger->file->slots[index].kind != 0u && SparkKvLedgerSlotLive(&ledger->file->slots[index]) == 0u &&
			(ledger->file->slots[index].kind == SPARK_KV_LEDGER_KIND_BACKING || stat(ledger->file->slots[index].directory,&info) != 0) )
			memset(&ledger->file->slots[index],0,sizeof(ledger->file->slots[index]));
	SparkKvLedgerUnlock(ledger);
}

SparkStatus SparkKvLedgerMakeRoom(SparkKvLedger *ledger,uint32_t slot,uint64_t incoming_bytes,uint64_t wait_ns,SparkKvLedgerEvictOwn evict_own,void *context,
	uint64_t (*own_bytes)(void *context),uint64_t (*own_oldest)(void *context))
{
	const struct timespec pause = {0,20000000L};
	SparkKvLedgerVictim victim;
	uint64_t deadline = SparkMonotonicNs() + wait_ns,freed;
	SparkStatus status;
	for (;;)
	{
		status = SparkKvLedgerReserve(ledger,slot,own_bytes(context),incoming_bytes,own_oldest(context),&victim);
		if ( status != SPARK_STATUS_CAPACITY_EXCEEDED || victim.slot == SPARK_KV_LEDGER_NO_SLOT )
			SPARK_RETURN(status);
		if ( victim.slot == slot )
		{
			if ( evict_own == 0 || evict_own(context,SPARK_KV_LEDGER_NO_OLDEST) == 0u )
				SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
			continue;
		}
		if ( victim.live == 0u )
		{
			freed = 0u;
			status = SparkKvSnapshotPurgeOldest(ledger,victim.directory,victim.shortfall_bytes,&freed);
			if ( status == SPARK_STATUS_OK || status == SPARK_STATUS_NOT_FOUND )
				continue;
			if ( status != SPARK_STATUS_BUSY )
				SPARK_RETURN(status);
		}
		SparkKvLedgerRequest(ledger,&victim);
		if ( SparkMonotonicNs() >= deadline )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		(void)nanosleep(&pause,0);
	}
}
