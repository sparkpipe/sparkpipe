#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include <cuda.h>
#include <cupti.h>
#include <cupti_activity.h>

typedef struct
{
	uint64_t count;
	uint64_t total_ns;
	uint64_t maximum_ns;
} spark_cupti_kernel_stats_t;

typedef struct
{
	uint64_t start_ns;
	uint64_t end_ns;
	uint32_t stream;
	std::string name;
} spark_cupti_interval_t;

typedef struct
{
	uint64_t duration_ns;
	uint32_t token_index;
	std::string before;
	std::string after;
} spark_cupti_gap_t;

typedef struct
{
	uint64_t total_ns;
	std::vector<uint64_t> durations_ns;
} spark_cupti_gap_stats_t;

static std::atomic<uint32_t> SparkCuptiInitialized(0u);
static std::mutex SparkCuptiMutex;
static std::map<std::string,spark_cupti_kernel_stats_t> SparkCuptiKernels;
static std::map<uint32_t,spark_cupti_kernel_stats_t> SparkCuptiStreams;
static std::vector<spark_cupti_interval_t> SparkCuptiIntervals;
static uint64_t SparkCuptiDropped;

static void SparkCuptiRecord(const CUpti_ActivityKernel10 *kernel)
{
	spark_cupti_kernel_stats_t *stats;
	char shape[512];
	uint64_t duration;
	std::lock_guard<std::mutex> lock(SparkCuptiMutex);
	if ( kernel == 0 || kernel->name == 0 || kernel->end <= kernel->start )
		return;
	duration = kernel->end - kernel->start;
	snprintf(shape,sizeof(shape),"%s grid=%d,%d,%d block=%d,%d,%d shared=%d registers=%u",
		kernel->name,kernel->gridX,kernel->gridY,kernel->gridZ,kernel->blockX,
		kernel->blockY,kernel->blockZ,kernel->dynamicSharedMemory,
		(uint32_t)kernel->registersPerThread);
	stats = &SparkCuptiKernels[std::string(shape)];
	stats->count++;
	stats->total_ns += duration;
	stats->maximum_ns = std::max(stats->maximum_ns,duration);
	stats = &SparkCuptiStreams[kernel->streamId];
	stats->count++;
	stats->total_ns += duration;
	stats->maximum_ns = std::max(stats->maximum_ns,duration);
	SparkCuptiIntervals.push_back({kernel->start,kernel->end,kernel->streamId,
		std::string(shape)});
}

static void CUPTIAPI SparkCuptiRequestBuffer(uint8_t **buffer,size_t *size,
	size_t *maximum_records)
{
	void *allocation;
	allocation = 0;
	if ( posix_memalign(&allocation,8u,8u * 1024u * 1024u) != 0 )
		allocation = 0;
	*buffer = (uint8_t *)allocation;
	*size = allocation != 0 ? 8u * 1024u * 1024u : 0u;
	*maximum_records = 0u;
}

static void CUPTIAPI SparkCuptiCompleteBuffer(CUcontext context,uint32_t stream,
	uint8_t *buffer,size_t size,size_t valid)
{
	CUpti_Activity *record;
	CUptiResult result;
	size_t dropped;
	(void)size;
	record = 0;
	result = CUPTI_SUCCESS;
	while ( valid != 0u && result == CUPTI_SUCCESS )
	{
		result = cuptiActivityGetNextRecord(buffer,valid,&record);
		if ( result == CUPTI_SUCCESS && record->kind ==
			CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL )
			SparkCuptiRecord((const CUpti_ActivityKernel10 *)record);
	}
	dropped = 0u;
	if ( cuptiActivityGetNumDroppedRecords(context,stream,&dropped) == CUPTI_SUCCESS )
	{
		std::lock_guard<std::mutex> lock(SparkCuptiMutex);
		SparkCuptiDropped += dropped;
	}
	free(buffer);
}

static uint64_t SparkCuptiBusyUnion(void)

{
	uint64_t busy,begin,end;
	uint32_t index;
	if ( SparkCuptiIntervals.empty() )
		return(0u);
	std::sort(SparkCuptiIntervals.begin(),SparkCuptiIntervals.end(),
		[](const spark_cupti_interval_t &left,const spark_cupti_interval_t &right)
		{ return(left.start_ns < right.start_ns); });
	busy = 0u;
	begin = SparkCuptiIntervals[0].start_ns;
	end = SparkCuptiIntervals[0].end_ns;
	for (index=1u; index<SparkCuptiIntervals.size(); index++)
	{
		if ( SparkCuptiIntervals[index].start_ns > end )
		{
			busy += end - begin;
			begin = SparkCuptiIntervals[index].start_ns;
			end = SparkCuptiIntervals[index].end_ns;
		}
		else
			end = std::max(end,SparkCuptiIntervals[index].end_ns);
	}
	return(busy + end - begin);
}

static void SparkCuptiRecordGap(std::vector<spark_cupti_gap_t> *gaps,
	uint32_t token_index,uint64_t begin,uint64_t end,
	const std::string &before,const std::string &after)
{
	if ( gaps == 0 || end <= begin )
		return;
	gaps->push_back({end - begin,token_index,before,after});
}

static uint64_t SparkCuptiDecodeWindowBusy(uint64_t begin,uint64_t end,
	uint32_t token_index,const std::string &next_name,
	std::vector<spark_cupti_gap_t> *gaps)
{
	uint64_t busy,cursor,interval_begin,interval_end;
	std::string active_name;
	busy = 0u;
	cursor = begin;
	for (const auto &interval : SparkCuptiIntervals)
	{
		if ( interval.end_ns <= begin )
			continue;
		if ( interval.start_ns >= end )
			break;
		interval_begin = std::max(interval.start_ns,begin);
		interval_end = std::min(interval.end_ns,end);
		if ( interval_begin > cursor )
			SparkCuptiRecordGap(gaps,token_index,cursor,interval_begin,
				active_name,interval.name);
		if ( interval_end > cursor )
		{
			busy += interval_end - std::max(cursor,interval_begin);
			cursor = interval_end;
			active_name = interval.name;
		}
	}
	if ( cursor < end )
		SparkCuptiRecordGap(gaps,token_index,cursor,end,active_name,next_name);
	return(busy);
}

static void SparkCuptiPrintDecodeWindows(void)
{
	std::vector<const spark_cupti_interval_t *> boundaries;
	std::vector<spark_cupti_gap_t> gaps;
	std::map<std::string,spark_cupti_gap_stats_t> transitions;
	std::vector<std::pair<std::string,spark_cupti_gap_stats_t>> transition_rows;
	uint64_t begin,end,busy,period;
	uint32_t index,limit,p95;
	for (const auto &interval : SparkCuptiIntervals)
		if ( interval.name.find("SparkLmEmbeddingGatherKernel") !=
			std::string::npos )
			boundaries.push_back(&interval);
	fprintf(stderr,"spark_cupti_decode_boundaries count=%zu\n",boundaries.size());
	for (index=0u; index + 1u<boundaries.size(); index++)
	{
		begin = boundaries[index]->start_ns;
		end = boundaries[index + 1u]->start_ns;
		period = end - begin;
		busy = SparkCuptiDecodeWindowBusy(begin,end,index,
			boundaries[index + 1u]->name,&gaps);
		fprintf(stderr,"spark_cupti_decode_window token=%u period_ns=%llu busy_ns=%llu idle_ns=%llu busy_percent=%.3f\n",
			index,(unsigned long long)period,(unsigned long long)busy,
			(unsigned long long)(period - busy),period != 0u ?
			100.0 * (double)busy / (double)period : 0.0);
	}
	for (const auto &gap : gaps)
	{
		std::string before = gap.before.substr(0,gap.before.find(" grid="));
		std::string after = gap.after.substr(0,gap.after.find(" grid="));
		std::string key = before + " -> " + after;
		transitions[key].total_ns += gap.duration_ns;
		transitions[key].durations_ns.push_back(gap.duration_ns);
	}
	for (auto &row : transitions)
	{
		std::sort(row.second.durations_ns.begin(),row.second.durations_ns.end());
		transition_rows.push_back(row);
	}
	std::sort(transition_rows.begin(),transition_rows.end(),
		[](const auto &left,const auto &right)
		{ return(left.second.total_ns > right.second.total_ns); });
	limit = std::min((uint32_t)transition_rows.size(),64u);
	for (index=0u; index<limit; index++)
	{
		const auto &stats = transition_rows[index].second;
		p95 = (uint32_t)((stats.durations_ns.size() * 95u + 99u) / 100u);
		if ( p95 != 0u )
			p95--;
		fprintf(stderr,"spark_cupti_decode_transition rank=%u total_ns=%llu count=%zu average_ns=%llu median_ns=%llu p95_ns=%llu maximum_ns=%llu path=%s\n",
			index,(unsigned long long)stats.total_ns,stats.durations_ns.size(),
			(unsigned long long)(stats.total_ns / stats.durations_ns.size()),
			(unsigned long long)stats.durations_ns[stats.durations_ns.size() / 2u],
			(unsigned long long)stats.durations_ns[p95],
			(unsigned long long)stats.durations_ns.back(),
			transition_rows[index].first.c_str());
	}
	std::sort(gaps.begin(),gaps.end(),
		[](const spark_cupti_gap_t &left,const spark_cupti_gap_t &right)
		{ return(left.duration_ns > right.duration_ns); });
	limit = std::min((uint32_t)gaps.size(),64u);
	for (index=0u; index<limit; index++)
		fprintf(stderr,"spark_cupti_decode_gap rank=%u token=%u duration_ns=%llu before=%s after=%s\n",
			index,gaps[index].token_index,
			(unsigned long long)gaps[index].duration_ns,
			gaps[index].before.c_str(),gaps[index].after.c_str());
}

static void SparkCuptiPrint(void)
{
	typedef std::pair<std::string,spark_cupti_kernel_stats_t> row_t;
	std::vector<row_t> rows;
	uint64_t total,busy,span;
	uint32_t index;
	for (const auto &row : SparkCuptiKernels)
		rows.push_back(row);
	std::sort(rows.begin(),rows.end(),[](const row_t &left,const row_t &right)
		{ return(left.second.total_ns > right.second.total_ns); });
	total = 0u;
	for (const auto &row : rows)
		total += row.second.total_ns;
	busy = SparkCuptiBusyUnion();
	span = SparkCuptiIntervals.empty() ? 0u :
		SparkCuptiIntervals.back().end_ns - SparkCuptiIntervals.front().start_ns;
	fprintf(stderr,"spark_cupti_summary kernels=%zu names=%zu total_ns=%llu busy_ns=%llu span_ns=%llu dropped=%llu\n",SparkCuptiIntervals.size(),rows.size(),(unsigned long long)total,(unsigned long long)busy,(unsigned long long)span,(unsigned long long)SparkCuptiDropped);
	SparkCuptiPrintDecodeWindows();
	for (index=0u; index<rows.size(); index++)
		fprintf(stderr,"spark_cupti_kernel rank=%u total_ns=%llu count=%llu average_ns=%llu maximum_ns=%llu name=%s\n",index,(unsigned long long)rows[index].second.total_ns,(unsigned long long)rows[index].second.count,(unsigned long long)(rows[index].second.total_ns / rows[index].second.count),(unsigned long long)rows[index].second.maximum_ns,rows[index].first.c_str());
	for (const auto &row : SparkCuptiStreams)
		fprintf(stderr,"spark_cupti_stream stream=%u total_ns=%llu count=%llu maximum_ns=%llu\n",row.first,(unsigned long long)row.second.total_ns,(unsigned long long)row.second.count,(unsigned long long)row.second.maximum_ns);
}

static void SparkCuptiExit(void)
{
	(void)cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
	(void)cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
	SparkCuptiPrint();
}

extern "C" int InitializeInjection(void)
{
	CUptiResult result;
	uint32_t expected;
	expected = 0u;
	if ( !SparkCuptiInitialized.compare_exchange_strong(expected,1u) )
		return(1);
	result = cuptiActivityRegisterCallbacks(SparkCuptiRequestBuffer,
		SparkCuptiCompleteBuffer);
	if ( result == CUPTI_SUCCESS )
		result = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
	if ( result != CUPTI_SUCCESS )
	{
		fprintf(stderr,"spark_cupti_init_failed result=%u\n",(uint32_t)result);
		return(0);
	}
	atexit(SparkCuptiExit);
	fprintf(stderr,"spark_cupti_ready\n");
	return(1);
}
