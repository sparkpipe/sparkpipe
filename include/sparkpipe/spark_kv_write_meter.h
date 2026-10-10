#pragma once

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_KV_WRITE_METER_WINDOW_NS (3600ull * 1000000000ull)
#define SPARK_KV_WRITE_ALERT_BYTES_PER_WINDOW (20ull * 1024ull * 1024ull * 1024ull)
#define SPARK_KV_WRITE_METER_LABEL_BYTES 32u

typedef struct SparkKvWriteMeter
{
	uint64_t written_bytes;
	uint64_t write_count;
	uint64_t window_started_ns;
	uint64_t window_bytes;
	uint64_t previous_window_bytes;
	uint64_t alert_count;
	uint64_t alerted_window_ns;
	char label[SPARK_KV_WRITE_METER_LABEL_BYTES];
} SparkKvWriteMeter;

static inline void SparkKvWriteMeterInitialize(SparkKvWriteMeter *meter,const char *label,uint64_t now_ns)
{
	memset(meter,0,sizeof(*meter));
	meter->window_started_ns = now_ns;
	(void)snprintf(meter->label,sizeof(meter->label),"%s",label != 0 ? label : "kv");
}

static inline void SparkKvWriteMeterRoll(SparkKvWriteMeter *meter,uint64_t now_ns)
{
	uint64_t started = __atomic_load_n(&meter->window_started_ns,__ATOMIC_ACQUIRE);
	if ( now_ns < started || now_ns - started < SPARK_KV_WRITE_METER_WINDOW_NS )
		return;
	if ( __atomic_compare_exchange_n(&meter->window_started_ns,&started,now_ns,0,__ATOMIC_ACQ_REL,__ATOMIC_ACQUIRE) != 0 )
		__atomic_store_n(&meter->previous_window_bytes,__atomic_exchange_n(&meter->window_bytes,0u,__ATOMIC_ACQ_REL),__ATOMIC_RELEASE);
}

static inline void SparkKvWriteMeterRecord(SparkKvWriteMeter *meter,uint64_t bytes,uint64_t now_ns)
{
	uint64_t window,started,alerted;
	SparkKvWriteMeterRoll(meter,now_ns);
	(void)__atomic_add_fetch(&meter->written_bytes,bytes,__ATOMIC_ACQ_REL);
	(void)__atomic_add_fetch(&meter->write_count,1u,__ATOMIC_ACQ_REL);
	window = __atomic_add_fetch(&meter->window_bytes,bytes,__ATOMIC_ACQ_REL);
	if ( window <= SPARK_KV_WRITE_ALERT_BYTES_PER_WINDOW )
		return;
	started = __atomic_load_n(&meter->window_started_ns,__ATOMIC_ACQUIRE);
	alerted = __atomic_load_n(&meter->alerted_window_ns,__ATOMIC_ACQUIRE);
	if ( alerted == started + 1u || __atomic_compare_exchange_n(&meter->alerted_window_ns,&alerted,started + 1u,0,__ATOMIC_ACQ_REL,__ATOMIC_ACQUIRE) == 0 )
		return;
	(void)__atomic_add_fetch(&meter->alert_count,1u,__ATOMIC_ACQ_REL);
	fprintf(stderr,"KV-WRITE-ALERT %s wrote %llu bytes to NVMe within the hour (alert above %llu; %.2f TB/day at this rate); total %llu bytes in %llu writes\n",
		meter->label,(unsigned long long)window,(unsigned long long)SPARK_KV_WRITE_ALERT_BYTES_PER_WINDOW,
		(double)window * 24.0 / 1.0e12,(unsigned long long)__atomic_load_n(&meter->written_bytes,__ATOMIC_ACQUIRE),
		(unsigned long long)__atomic_load_n(&meter->write_count,__ATOMIC_ACQUIRE));
}

static inline uint32_t SparkKvWriteMeterAlerting(SparkKvWriteMeter *meter,uint64_t now_ns)
{
	SparkKvWriteMeterRoll(meter,now_ns);
	return(__atomic_load_n(&meter->window_bytes,__ATOMIC_ACQUIRE) > SPARK_KV_WRITE_ALERT_BYTES_PER_WINDOW ? 1u : 0u);
}

#ifdef __cplusplus
}
#endif
