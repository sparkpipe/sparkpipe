#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include "sparkpipe/spark_device.h"
#include "runtime/device/nvrm/nvrm_image.h"
#include "runtime/device/nvrm/nvrm_launch.h"
#include "runtime/device/nvrm/nvrm_rm.h"
#include "nvstatus.h"
#include "ctrl/ctrl0000/ctrl0000gpu.h"
#include "nvmisc.h"

#define MW(range) range
#include "classes/compute/clcec0qmd.h"

#ifndef SPARK_NVRM_FIXTURE_CUBIN
#define SPARK_NVRM_FIXTURE_CUBIN "tests/fixtures/nvrm/fixture_kernels.sm_121a.cubin"
#endif

#define PUSH_ADDRESS 0x1000200000ull
#define FENCE_ADDRESS 0x1000900000ull
#define FENCE_DONE 0xffffffffffffull
#define RELEASE_BYTES 24u
#define CONTROL_ENTRIES 8u
#define DOORBELL_WORD (0x90u / 4u)
#define GPPUT_OFFSET 0x8cu

static int failures;

#define CHECK(condition, what) \
	do { \
		if ( !(condition) ) \
		{ \
			printf("FAIL %s\n",what); \
			failures++; \
		} \
	} while (0)

typedef struct Fake
{
	SparkNvrmGpu gpu;
	uint8_t push[4096];
	uint8_t control[CONTROL_ENTRIES * 8u + 4096u];
	uint64_t fence[512];
	uint32_t usermode[0x10000 / 4];
} Fake;

static void FakeOpen(Fake *fake)
{
	memset(fake,0,sizeof(*fake));
	fake->gpu.compute.push.cpu = fake->push;
	fake->gpu.compute.push.address = PUSH_ADDRESS;
	fake->gpu.compute.push.bytes = sizeof(fake->push);
	fake->gpu.compute.control.cpu = fake->control;
	fake->gpu.compute.entries = CONTROL_ENTRIES;
	fake->gpu.compute.token = 0x1234abcdu;
	fake->gpu.compute.fence.cpu = (uint8_t *)fake->fence;
	fake->gpu.compute.fence.address = FENCE_ADDRESS;
	fake->gpu.compute.fence.bytes = sizeof(fake->fence);
	fake->fence[0] = FENCE_DONE;
	fake->gpu.usermode_page = fake->usermode;
	fake->gpu.local_bytes_per_thread = 0x240u;
	fake->gpu.sass_version = 0x81u;
	fake->gpu.shared_window = 0x729400000000ull;
	fake->gpu.local_window = 0x729300000000ull;
}

static uint32_t PushWord(const Fake *fake, uint32_t index)
{
	uint32_t word;
	memcpy(&word,fake->push + 4u * index,4u);
	return(word);
}

static uint64_t ControlWord64(const Fake *fake, uint32_t index)
{
	uint64_t word;
	memcpy(&word,fake->control + 8u * index,8u);
	return(word);
}

static uint32_t GpPut(const Fake *fake)
{
	uint32_t word;
	memcpy(&word,fake->control + CONTROL_ENTRIES * 8u + GPPUT_OFFSET,4u);
	return(word);
}

static void TestPush(void)
{
	static Fake fake;
	const uint32_t values[3] = {0x11111111u, 0x22222222u, 0x33333333u};
	SparkNvrmChannel *channel;
	FakeOpen(&fake);
	channel = &fake.gpu.compute;
	CHECK(SparkNvrmPush(channel,1u,0x02b4u,values,1u,"push") == SPARK_STATUS_OK && channel->push_used == 8u,"a one-word method takes two words");
	CHECK(PushWord(&fake,0) == ((1u << 29) | (1u << 16) | (1u << 13) | (0x02b4u >> 2)) && PushWord(&fake,1) == 0x11111111u,
		"the header is an incrementing method: operation 1 at 31:29, count at 28:16, subchannel at 15:13, address/4 at 11:0");
	CHECK(SparkNvrmPush(channel,7u,0x0ffcu,values,3u,"push") == SPARK_STATUS_OK && PushWord(&fake,2) == ((1u << 29) | (3u << 16) | (7u << 13) | 0x3ffu) &&
		PushWord(&fake,5) == 0x33333333u,"a three-word method on subchannel 7 at the last address");
	CHECK(SparkNvrmPush(channel,1u,0x02b4u,values,0u,"push") == SPARK_STATUS_INVALID_ARGUMENT,"an empty method is refused");
	CHECK(SparkNvrmPush(channel,1u,0x02b6u,values,1u,"push") == SPARK_STATUS_INVALID_ARGUMENT,"a method address off a word is refused");
	CHECK(SparkNvrmPush(channel,8u,0x02b4u,values,1u,"push") == SPARK_STATUS_INVALID_ARGUMENT,"subchannel 8 is refused");
	CHECK(SparkNvrmPush(channel,1u,0x02b4u,values,8192u,"push") == SPARK_STATUS_INVALID_ARGUMENT,"8192 words in one method are refused");
	channel->push_submitted = 8u;
	channel->push_used = sizeof(fake.push) - 4u;
	CHECK(SparkNvrmPush(channel,1u,0x02b4u,values,3u,"push") == SPARK_STATUS_CAPACITY_EXCEEDED && channel->push_used == sizeof(fake.push) - 4u,
		"methods that would not fit the push buffer even after wrapping are refused and nothing is written");
	channel->push_submitted = channel->push_used = sizeof(fake.push) - 12u;
	memcpy(fake.push + channel->push_used,values,8u);
	channel->push_used += 8u;
	CHECK(SparkNvrmPush(channel,1u,0x02b4u,values,1u,"push") == SPARK_STATUS_OK && channel->push_submitted == 0u && channel->push_used == 16u &&
		PushWord(&fake,0) == 0x11111111u && PushWord(&fake,1) == 0x22222222u && PushWord(&fake,3) == 0x11111111u,
		"a method that does not fit before the end moves the unsubmitted methods to the start and follows them");
}

static void TestSubmit(void)
{
	static Fake fake;
	const uint32_t value = 7u;
	SparkNvrmChannel *channel;
	uint32_t round;
	FakeOpen(&fake);
	channel = &fake.gpu.compute;
	CHECK(SparkNvrmSubmit(&fake.gpu,channel,"submit") == SPARK_STATUS_INVALID_ARGUMENT,"an empty submission is refused");
	(void)SparkNvrmPush(channel,1u,0x02b4u,&value,1u,"submit");
	(void)SparkNvrmPush(channel,1u,0x02c0u,&value,1u,"submit");
	CHECK(SparkNvrmSubmit(&fake.gpu,channel,"submit") == SPARK_STATUS_OK,"a submission of two methods succeeds");
	CHECK(ControlWord64(&fake,0) == (PUSH_ADDRESS | ((16ull + RELEASE_BYTES) / 4u << 42) | (1ull << 41)),
		"the ring entry holds the address, the word count at bit 42 and bit 41, and the segment ends with a release of the channel's fence");
	CHECK(PushWord(&fake,4) == ((1u << 29) | (5u << 16) | (0x5cu >> 2)) && PushWord(&fake,5) == (uint32_t)FENCE_ADDRESS && PushWord(&fake,7) == 1u &&
		channel->fence_value == 1u && channel->segment_count == 1u && channel->segments[0].value == 1u,
		"a submission without its own fence release gets the next fence value, which marks when its methods are consumed");
	CHECK(GpPut(&fake) == 1u && fake.usermode[DOORBELL_WORD] == 0x1234abcdu,"GPPut advances and the doorbell carries the channel's token");
	for (round = 1u; round < CONTROL_ENTRIES + 1u; round++)
	{
		(void)SparkNvrmPush(channel,1u,0x02b4u,&value,1u,"submit");
		(void)SparkNvrmSubmit(&fake.gpu,channel,"submit");
	}
	CHECK(GpPut(&fake) == 1u && ControlWord64(&fake,0) == ((PUSH_ADDRESS + 40u + 32u * (CONTROL_ENTRIES - 1u)) | ((8ull + RELEASE_BYTES) / 4u << 42) | (1ull << 41)),
		"the ring wraps: GPPut returns to 1 and entry 0 holds the newest submission");
	CHECK(channel->segment_count < CONTROL_ENTRIES,"finished submissions are retired so the ring never holds more than it has entries");
	CHECK(SparkNvrmSubmit(&fake.gpu,channel,"submit") == SPARK_STATUS_INVALID_ARGUMENT,"a submission with nothing new is refused");
	(void)SparkNvrmPush(channel,1u,0x02b4u,&value,1u,"submit");
	(void)SparkNvrmPush(channel,1u,0x02c0u,&value,1u,"submit");
	(void)SparkNvrmPush(channel,1u,0x02b4u,&value,1u,"submit");
	CHECK(SparkNvrmSubmit(&fake.gpu,channel,"submit") == SPARK_STATUS_OK &&
		ControlWord64(&fake,1) == ((PUSH_ADDRESS + 40u + 32u * CONTROL_ENTRIES) | ((24ull + RELEASE_BYTES) / 4u << 42) | (1ull << 41)),
		"a submission sends every method pushed since the last one");
	channel->push.address = 1ull << 40;
	(void)SparkNvrmPush(channel,1u,0x02b4u,&value,1u,"submit");
	CHECK(SparkNvrmSubmit(&fake.gpu,channel,"submit") == SPARK_STATUS_INVALID_ARGUMENT,"a push buffer at or above 2^40 is refused");
}

typedef struct Finisher
{
	volatile uint64_t *fence;
	uint64_t value;
	long delay_ns;
} Finisher;

static void *FinishLater(void *argument)
{
	const Finisher *finisher = (const Finisher *)argument;
	struct timespec pause = {0, finisher->delay_ns};
	nanosleep(&pause,0);
	__atomic_store_n(finisher->fence,finisher->value,__ATOMIC_RELEASE);
	return(0);
}

static void TestReclaim(void)
{
	static Fake fake;
	const uint32_t value = 7u;
	SparkNvrmChannel *channel;
	pthread_t thread;
	Finisher finisher;
	uint32_t round;
	FakeOpen(&fake);
	channel = &fake.gpu.compute;
	fake.fence[0] = 0u;
	for (round = 0u; round < 3u; round++)
	{
		(void)SparkNvrmPush(channel,1u,0x02b4u,&value,1u,"reclaim");
		(void)SparkNvrmSubmit(&fake.gpu,channel,"reclaim");
	}
	CHECK(channel->segment_count == 3u && channel->push_used == 96u,"three unfinished submissions hold the start of the push buffer");
	channel->push_submitted = channel->push_used = sizeof(fake.push) - 4u;
	finisher.fence = (volatile uint64_t *)fake.fence;
	finisher.value = 2u;
	finisher.delay_ns = 30000000L;
	pthread_create(&thread,0,FinishLater,&finisher);
	CHECK(SparkNvrmPush(channel,1u,0x02b4u,&value,1u,"reclaim") == SPARK_STATUS_OK && fake.fence[0] == 2u && channel->segment_count == 2u &&
		channel->push_used == 8u,"wrapping waits until the GPU has finished the submissions it would overwrite, and no longer");
	pthread_join(thread,0);
}

static uint32_t Field(uint32_t high, uint32_t low, uint32_t value)
{
	return((value & (uint32_t)((2ull << (high - low)) - 1u)) << low);
}

static void TestOrdering(void)
{
	static Fake fake;
	SparkNvrmChannel *channel;
	uint64_t value = 0u;
	const uint32_t copy_launch = Field(1u,0u,2u) | Field(7u,7u,1u) | Field(8u,8u,1u);
	const uint32_t copy_release = Field(1u,0u,0u) | Field(2u,2u,1u) | Field(4u,3u,1u) | Field(27u,27u,1u);
	FakeOpen(&fake);
	channel = &fake.gpu.compute;
	channel->fence_value = 0xfffffffeull;
	CHECK(SparkNvrmCopy(&fake.gpu,channel,0x2000001000ull,0x3000002000ull,4096u,&value,"order") == SPARK_STATUS_OK && value == 0xffffffffull,
		"a copy takes the channel's next fence value");
	CHECK(PushWord(&fake,0) == ((1u << 29) | (4u << 16) | (4u << 13) | (0x400u >> 2)) && PushWord(&fake,1) == 0x30u && PushWord(&fake,2) == 0x2000u &&
		PushWord(&fake,3) == 0x20u && PushWord(&fake,4) == 0x1000u,"the copy sends source then destination, upper word first, on subchannel 4");
	CHECK(PushWord(&fake,5) == ((1u << 29) | (1u << 16) | (4u << 13) | (0x418u >> 2)) && PushWord(&fake,6) == 4096u &&
		PushWord(&fake,8) == copy_launch,"one line of 4096 bytes, non-pipelined, pitch to pitch");
	CHECK(PushWord(&fake,9) == ((1u << 29) | (4u << 16) | (4u << 13) | (0x240u >> 2)) && PushWord(&fake,10) == (uint32_t)(FENCE_ADDRESS >> 32) &&
		PushWord(&fake,11) == (uint32_t)FENCE_ADDRESS && PushWord(&fake,12) == 0xffffffffu && PushWord(&fake,13) == 0u && PushWord(&fake,15) == copy_release,
		"the copy engine itself releases the 64-bit fence value after the copy, with a flush");
	CHECK(channel->push_used == 64u && channel->last_engine == SPARK_NVRM_ENGINE_COPY,"the copy is one submission of its own");
	CHECK(SparkNvrmFill(&fake.gpu,channel,0x2000001000ull,0xabu,64u,&value,"order") == SPARK_STATUS_OK && value == 0x100000000ull &&
		PushWord(&fake,16) == ((1u << 29) | (3u << 16) | (4u << 13) | (0x700u >> 2)),
		"a copy after a copy needs no wait: the copy engine runs non-pipelined transfers in order, and the payload's upper word carries");
	CHECK(PushWord(&fake,17) == 0xababababu && PushWord(&fake,19) == (Field(2u,0u,4u) | Field(17u,16u,3u)),
		"an aligned fill writes the byte four times per element from constant A");
	CHECK(SparkNvrmSemaphoreRelease(channel,FENCE_ADDRESS + 64u,1u,0u,"order") == SPARK_STATUS_OK &&
		SparkNvrmOpBegin(channel,SPARK_NVRM_ENGINE_COPY,"order") == SPARK_STATUS_BUSY,"an ordered operation refuses to start on another caller's unsubmitted methods");
	SparkNvrmOpAbandon(channel);
	channel->last_engine = SPARK_NVRM_ENGINE_COMPUTE;
	{
		const uint64_t start = channel->push_used;
		CHECK(SparkNvrmCopy(&fake.gpu,channel,0x2000001000ull,0x3000002000ull,4096u,&value,"order") == SPARK_STATUS_OK,"a copy after a launch");
		CHECK(PushWord(&fake,(uint32_t)(start / 4u)) == ((1u << 29) | (5u << 16) | (0x5cu >> 2)) && PushWord(&fake,(uint32_t)(start / 4u) + 3u) == 0u &&
			PushWord(&fake,(uint32_t)(start / 4u) + 4u) == 1u && PushWord(&fake,(uint32_t)(start / 4u) + 5u) == (3u | (1u << 24)),
			"a copy after a launch first makes the channel wait until the fence reaches the launch's value");
	}
}

static void TestFillSplit(void)
{
	static Fake fake;
	SparkNvrmChannel *channel;
	uint64_t value = 0u;
	FakeOpen(&fake);
	channel = &fake.gpu.compute;
	CHECK(SparkNvrmFill(&fake.gpu,channel,0x2000001003ull,0x5au,14u,&value,"fill") == SPARK_STATUS_OK,"an unaligned 14-byte fill");
	CHECK(PushWord(&fake,1) == 0x5au && PushWord(&fake,3) == Field(2u,0u,4u) && PushWord(&fake,5) == 0x20u && PushWord(&fake,6) == 0x1003u &&
		PushWord(&fake,8) == 1u,"one byte up to the first word boundary");
	CHECK(PushWord(&fake,12) == 0x5a5a5a5au && PushWord(&fake,14) == (Field(2u,0u,4u) | Field(17u,16u,3u)) && PushWord(&fake,17) == 0x1004u &&
		PushWord(&fake,19) == 3u,"three four-byte elements in the middle");
	CHECK(PushWord(&fake,23) == 0x5au && PushWord(&fake,25) == Field(2u,0u,4u) && PushWord(&fake,28) == 0x1010u && PushWord(&fake,30) == 1u,
		"and the last byte");
	CHECK(channel->push_used == 40u * 4u,"then the copy engine's fence release");
}

#define LINES_MAX 16u

typedef struct Lines
{
	uint64_t address[LINES_MAX];
	uint32_t length[LINES_MAX];
	uint32_t count;
} Lines;

static void LinesRead(const Fake *fake, uint32_t end_word, Lines *lines)
{
	uint32_t word = 0u;
	uint64_t address = 0u;
	memset(lines,0,sizeof(*lines));
	while ( word < end_word )
	{
		const uint32_t header = PushWord(fake,word);
		const uint32_t count = (header >> 16) & 0x1fffu,method = (header & 0xfffu) << 2;
		if ( method == 0x400u && count == 4u )
			address = ((uint64_t)PushWord(fake,word + 3u) << 32) | PushWord(fake,word + 4u);
		if ( method == 0x408u && count == 2u )
			address = ((uint64_t)PushWord(fake,word + 1u) << 32) | PushWord(fake,word + 2u);
		if ( method == 0x418u && count == 1u && lines->count < LINES_MAX )
		{
			lines->address[lines->count] = address;
			lines->length[lines->count++] = PushWord(fake,word + 1u);
		}
		word += 1u + count;
	}
}

static int LinesAre(const Lines *lines, uint64_t first, uint64_t bytes, uint32_t element_bytes)
{
	const uint64_t line_bytes = 1ull << 31;
	uint64_t done = 0u;
	uint32_t index;
	if ( lines->count != (uint32_t)((bytes + line_bytes - 1u) / line_bytes) )
		return(0);
	for (index = 0u; index < lines->count; index++)
	{
		const uint64_t expected = bytes - done < line_bytes ? bytes - done : line_bytes;
		if ( lines->address[index] != first + done || (uint64_t)lines->length[index] * element_bytes != expected )
			return(0);
		done += expected;
	}
	return(done == bytes);
}

static void TestLongLines(void)
{
	static const uint64_t sizes[] = {8ull << 30, (21ull << 30) / 2u, 16ull << 30};
	static Fake fake;
	Lines lines;
	uint64_t value = 0u;
	uint32_t index;
	for (index = 0u; index < sizeof(sizes) / sizeof(sizes[0]); index++)
	{
		char what[160];
		FakeOpen(&fake);
		CHECK(SparkNvrmFill(&fake.gpu,&fake.gpu.compute,0x2000000000ull,0u,sizes[index],&value,"fill") == SPARK_STATUS_OK,"push a fill of 8 GiB or more");
		LinesRead(&fake,(uint32_t)(fake.gpu.compute.push_used / 4u),&lines);
		snprintf(what,sizeof(what),"a %llu MiB fill goes in lines of at most 2^31 bytes, 2^29 four-byte elements each, back to back",
			(unsigned long long)(sizes[index] >> 20));
		CHECK(LinesAre(&lines,0x2000000000ull,sizes[index],4u),what);
		FakeOpen(&fake);
		CHECK(SparkNvrmCopy(&fake.gpu,&fake.gpu.compute,0x2000000000ull,0x6000000000ull,sizes[index],&value,"copy") == SPARK_STATUS_OK,
			"push a copy of 8 GiB or more");
		LinesRead(&fake,(uint32_t)(fake.gpu.compute.push_used / 4u),&lines);
		snprintf(what,sizeof(what),"a %llu MiB copy goes in lines of at most 2^31 bytes, back to back",(unsigned long long)(sizes[index] >> 20));
		CHECK(LinesAre(&lines,0x2000000000ull,sizes[index],1u),what);
	}
}

static void TestPageSizes(void)
{
	static Fake fake;
	uint32_t handle = 0u;
	FakeOpen(&fake);
	CHECK(SparkNvrmSystemPageBytes(4ull << 20) == SPARK_NVRM_PAGE_BYTES && SparkNvrmSystemPageBytes(8ull << 20) == SPARK_NVRM_HUGE_PAGE_BYTES &&
		SparkNvrmSystemPageBytes((4ull << 30) - (2ull << 20)) == SPARK_NVRM_HUGE_PAGE_BYTES && SparkNvrmSystemPageBytes(4ull << 30) == SPARK_NVRM_BIG_PAGE_BYTES &&
		SparkNvrmSystemPageBytes(16ull << 30) == SPARK_NVRM_BIG_PAGE_BYTES,
		"system memory takes 2 MiB pages from 8 MiB to below 4 GiB and 64 KiB pages from 4 GiB on");
	CHECK(SparkNvrmSystemAllocate(&fake.gpu,4ull << 30,SPARK_NVRM_HUGE_PAGE_BYTES,SPARK_NVRM_MEMORY_SYSTEM,&handle,"pages") == SPARK_STATUS_INVALID_ARGUMENT && handle == 0u &&
		strstr(SparkDeviceLastError()->text,"4 GiB") != 0,"an object of 2 MiB pages at 4 GiB or more is refused before the module sees it");
}

static void TestModuleZeroes(void)
{
	static const char *const contents[] = {"ResmanDebugLevel: 4294967295\nInitializeSystemMemoryAllocations: 1\nRmLogonRC: 1\n",
		"InitializeSystemMemoryAllocations: 0\n", "ResmanDebugLevel: 4294967295\n"};
	static const int expected[] = {1, 0, 0};
	char path[] = "/tmp/spark-nvrm-params-XXXXXX";
	const int fd = mkstemp(path);
	uint32_t index;
	CHECK(fd >= 0,"make a stand-in for the module's parameter file");
	if ( fd < 0 )
		return;
	close(fd);
	for (index = 0u; index < sizeof(expected) / sizeof(expected[0]); index++)
	{
		FILE *file = fopen(path,"w");
		if ( file != 0 )
		{
			fputs(contents[index],file);
			fclose(file);
		}
		CHECK(file != 0 && SparkNvrmModuleZeroesMemory(path) == expected[index],
			"the module zeroes system memory only when its parameters list InitializeSystemMemoryAllocations: 1");
	}
	unlink(path);
	CHECK(SparkNvrmModuleZeroesMemory(path) == 0 && SparkNvrmModuleZeroesMemory(0) == 0,"a missing parameter file means no zeroing");
}

static void TestMarks(void)
{
	static Fake fake;
	SparkNvrmChannel *channel;
	uint64_t value = 0u;
	const uint64_t mark = 0x1000a00040ull;
	const uint32_t header = (1u << 29) | (5u << 16) | (0x5cu >> 2);
	FakeOpen(&fake);
	channel = &fake.gpu.compute;
	CHECK(SparkNvrmMark(&fake.gpu,channel,mark,7u,&value,"mark") == SPARK_STATUS_OK && value == 1u && channel->push_used == 72u,
		"a mark is three five-word semaphore methods and takes the next fence value");
	CHECK(PushWord(&fake,0) == header && PushWord(&fake,1) == (uint32_t)(mark + 16u) && PushWord(&fake,3) == 7u &&
		PushWord(&fake,5) == (1u | (1u << 20) | (1u << 24) | (1u << 25)),"first a timestamped release of the payload 16 bytes into the mark");
	CHECK(PushWord(&fake,6) == header && PushWord(&fake,7) == (uint32_t)FENCE_ADDRESS && PushWord(&fake,9) == 1u,
		"then the channel's fence, so the queue has passed the mark before anyone sees it complete");
	CHECK(PushWord(&fake,12) == header && PushWord(&fake,13) == (uint32_t)mark && PushWord(&fake,15) == 7u &&
		PushWord(&fake,17) == (6u | (1u << 20) | (1u << 24) | (1u << 27) | (1u << 31)),
		"last an unsigned 64-bit maximum into the mark's first word, so a later mark that finishes first is never undone");
	CHECK(SparkNvrmAwait(&fake.gpu,channel,mark,7u,&value,"mark") == SPARK_STATUS_OK && value == 2u &&
		PushWord(&fake,18) == header && PushWord(&fake,19) == (uint32_t)mark && PushWord(&fake,21) == 7u && PushWord(&fake,23) == (3u | (1u << 24)),
		"a wait acquires the mark's first word at or past the payload, then releases the fence");
}

static void TestAddress(void)
{
	static SparkNvrmGpu gpu;
	const uint64_t bytes = 64ull << 20;
	void *probe = mmap(0,(size_t)bytes,PROT_NONE,MAP_PRIVATE | MAP_ANONYMOUS,-1,0);
	const uint64_t base = ((uint64_t)(uintptr_t)probe + (2ull << 20) - 1u) & ~((2ull << 20) - 1u);
	uint64_t first = 0u,huge = 0u,again = 0u,rest = 0u,none = 0u;
	munmap(probe,(size_t)bytes);
	memset(&gpu,0,sizeof(gpu));
	CHECK(SparkNvrmAddressOpen(&gpu,base,bytes - (2ull << 20),"address") == SPARK_STATUS_OK,"the device address range is reserved in the process");
	CHECK(SparkNvrmVirtual(&gpu,4096u,4096u,&first,"address") == SPARK_STATUS_OK && first == base,"the first span starts the range");
	CHECK(SparkNvrmVirtual(&gpu,2ull << 20,2ull << 20,&huge,"address") == SPARK_STATUS_OK && huge == base + (2ull << 20),"a 2 MiB span is 2 MiB aligned");
	SparkNvrmVirtualRelease(&gpu,first,4096u);
	CHECK(SparkNvrmVirtual(&gpu,4096u,4096u,&again,"address") == SPARK_STATUS_OK && again == first,"a released span is used again");
	CHECK(SparkNvrmVirtual(&gpu,bytes - (6ull << 20),4096u,&rest,"address") == SPARK_STATUS_OK,"the rest of the range is one span");
	CHECK(SparkNvrmVirtual(&gpu,2ull << 20,4096u,&none,"address") == SPARK_STATUS_CAPACITY_EXCEEDED,"an exhausted range refuses");
	SparkNvrmVirtualRelease(&gpu,huge,2ull << 20);
	SparkNvrmVirtualRelease(&gpu,again,4096u);
	SparkNvrmVirtualRelease(&gpu,rest,bytes - (6ull << 20));
	CHECK(gpu.free_count == 1u && gpu.free_ranges[0].base == base && gpu.free_ranges[0].bytes == bytes - (2ull << 20),
		"released spans merge with their free neighbours back into one range");
	{
		static SparkNvrmGpu other;
		memset(&other,0,sizeof(other));
		CHECK(SparkNvrmAddressOpen(&other,base,bytes - (2ull << 20),"address") == SPARK_STATUS_CAPACITY_EXCEEDED,
			"a range another mapping already holds is refused");
	}
	SparkNvrmAddressClose(&gpu);
}

static void TestSemaphore(void)
{
	static Fake fake;
	SparkNvrmChannel *channel;
	FakeOpen(&fake);
	channel = &fake.gpu.compute;
	CHECK(SparkNvrmSemaphoreRelease(channel,0x1000400010ull,0x1122334455667788ull,1u,"semaphore") == SPARK_STATUS_OK && channel->push_used == 24u,
		"a release is one five-word method");
	CHECK(PushWord(&fake,0) == ((1u << 29) | (5u << 16) | (0u << 13) | (0x5cu >> 2)) && PushWord(&fake,1) == 0x00400010u && PushWord(&fake,2) == 0x10u &&
		PushWord(&fake,3) == 0x55667788u && PushWord(&fake,4) == 0x11223344u,"the release starts at SEM_ADDR_LO on subchannel 0 with address and payload halves");
	CHECK(PushWord(&fake,5) == (1u | (1u << 20) | (1u << 24) | (1u << 25)),"the release waits for idle, writes 64 bits and a timestamp");
	CHECK(SparkNvrmSemaphoreRelease(channel,0x1000400018ull,1u,1u,"semaphore") == SPARK_STATUS_INVALID_ARGUMENT,"a timestamped release off 16 bytes is refused");
	CHECK(SparkNvrmSemaphoreRelease(channel,0x1000400018ull,1u,0u,"semaphore") == SPARK_STATUS_OK && PushWord(&fake,11) == (1u | (1u << 20) | (1u << 24)),
		"an untimed release needs 8 bytes and asks for no timestamp");
	CHECK(SparkNvrmSemaphoreRelease(channel,0x1000400014ull,1u,0u,"semaphore") == SPARK_STATUS_INVALID_ARGUMENT,"a release off 8 bytes is refused");
}

static uint8_t *ReadFile(const char *path, uint64_t *bytes)
{
	FILE *file = fopen(path,"rb");
	uint8_t *data;
	long length;
	if ( file == 0 )
		return(0);
	fseek(file,0,SEEK_END);
	length = ftell(file);
	fseek(file,0,SEEK_SET);
	data = (uint8_t *)malloc((size_t)length);
	if ( data != 0 && fread(data,1u,(size_t)length,file) != (size_t)length )
	{
		free(data);
		data = 0;
	}
	fclose(file);
	*bytes = (uint64_t)length;
	return(data);
}

static uint64_t QmdField(const uint8_t *slot, uint32_t high, uint32_t low)
{
	SparkNvrmQmd qmd;
	memcpy(qmd.word,slot,sizeof(qmd.word));
	return(SparkNvrmQmdRead(&qmd,high,low));
}

static void TestLaunch(void)
{
	static Fake fake;
	static uint8_t slot[8192];
	SparkNvrmProgram program;
	SparkNvrmLaunchRequest request;
	uint64_t bytes = 0u,output = 0x1000500000ull;
	uint8_t *blob = ReadFile(SPARK_NVRM_FIXTURE_CUBIN,&bytes),arguments[12];
	const uint32_t value = 0xfeedu;
	uint32_t word;
	FakeOpen(&fake);
	memset(&program,0,sizeof(program));
	CHECK(blob != 0 && SparkNvrmCubinParse(blob,bytes,"launch",&program.cubin) == SPARK_STATUS_OK,"the fixture parses");
	if ( blob == 0 )
		return;
	program.code.address = 0x1000600000ull;
	memcpy(arguments,&output,8u);
	memcpy(arguments + 8u,&value,4u);
	memset(&request,0,sizeof(request));
	request.kernel = SparkNvrmCubinKernelNamed(&program.cubin,"SparkNvrmFirstLight");
	request.grid[0] = 4u;
	request.grid[1] = request.grid[2] = 1u;
	request.block[0] = 128u;
	request.block[1] = request.block[2] = 1u;
	request.arguments = arguments;
	request.argument_bytes = sizeof(arguments);
	request.slot = slot;
	request.slot_address = 0x1000700000ull;
	request.slot_bytes = request.kernel != 0 ? SparkNvrmLaunchSlotBytes(request.kernel) : 0u;
	request.release_address = 0x1000800000ull;
	request.release_payload = 9u;
	CHECK(request.kernel != 0 && SparkNvrmLaunchQueue(&fake.gpu,&fake.gpu.compute,&program,&request,"launch") == SPARK_STATUS_OK,"the first-light kernel queues");
	if ( request.kernel != 0 )
	{
		const uint64_t program_address = program.code.address + request.kernel->code_offset;
		CHECK(fake.gpu.compute.push_used == 16u && PushWord(&fake,0) == ((1u << 29) | (1u << 16) | (1u << 13) | (0x02b4u >> 2)) &&
			PushWord(&fake,1) == (uint32_t)(0x1000700000ull >> 8) && PushWord(&fake,3) == 9u,
			"the launch sends the descriptor address >> 8 and then schedules it with prefetch");
		CHECK(QmdField(slot,1 ? NVCEC0_QMDV05_00_PROGRAM_ADDRESS_LOWER_SHIFTED4,0 ? NVCEC0_QMDV05_00_PROGRAM_ADDRESS_LOWER_SHIFTED4) == ((program_address >> 4) & 0xffffffffu) &&
			QmdField(slot,1 ? NVCEC0_QMDV05_00_REGISTER_COUNT,0 ? NVCEC0_QMDV05_00_REGISTER_COUNT) == request.kernel->registers &&
			QmdField(slot,1 ? NVCEC0_QMDV05_00_CONSTANT_BUFFER_ADDR_LOWER_SHIFTED6(0),0 ? NVCEC0_QMDV05_00_CONSTANT_BUFFER_ADDR_LOWER_SHIFTED6(0)) ==
			((request.slot_address + SPARK_NVRM_LAUNCH_QMD_BYTES) >> 6) &&
			QmdField(slot,1 ? NVCEC0_QMDV05_00_SHADER_LOCAL_MEMORY_HIGH_SIZE_SHIFTED4,0 ? NVCEC0_QMDV05_00_SHADER_LOCAL_MEMORY_HIGH_SIZE_SHIFTED4) == 0x24u &&
			QmdField(slot,1 ? NVCEC0_QMDV05_00_RELEASE_SEMAPHORE0_PAYLOAD_LOWER,0 ? NVCEC0_QMDV05_00_RELEASE_SEMAPHORE0_PAYLOAD_LOWER) == 9u,
			"the slot holds the descriptor for this kernel, its bank 0 after the descriptor and the release");
		memcpy(&word,slot + SPARK_NVRM_LAUNCH_QMD_BYTES + SPARK_NVRM_BANK0_DRIVER_BYTES + 8u,4u);
		CHECK(word == value,"the arguments land in bank 0 at the kernel's parameter offset");
		memcpy(&word,slot + SPARK_NVRM_LAUNCH_QMD_BYTES + 4u * SPARK_NVRM_BANK0_WORD_BLOCK_X,4u);
		CHECK(word == 128u,"the launch's group width lands in bank 0");
		memcpy(&word,slot + SPARK_NVRM_LAUNCH_QMD_BYTES + 4u * SPARK_NVRM_BANK0_WORD_GRID_X,4u);
		CHECK(word == 4u,"the launch's grid width lands in bank 0");
		fake.gpu.local_bytes_per_thread = 0x100u;
		CHECK(SparkNvrmLaunchQueue(&fake.gpu,&fake.gpu.compute,&program,&request,"launch") == SPARK_STATUS_INVALID_ARGUMENT,
			"a kernel needing more local memory than provisioned is refused");
		fake.gpu.local_bytes_per_thread = 0x240u;
		request.slot_address += 64u;
		CHECK(SparkNvrmLaunchQueue(&fake.gpu,&fake.gpu.compute,&program,&request,"launch") == SPARK_STATUS_INVALID_ARGUMENT,"a slot off 256 bytes is refused");
		request.slot_address -= 64u;
		request.slot_bytes -= 1u;
		CHECK(SparkNvrmLaunchQueue(&fake.gpu,&fake.gpu.compute,&program,&request,"launch") == SPARK_STATUS_INVALID_ARGUMENT,"a slot too small for bank 0 is refused");
	}
	SparkNvrmCubinRelease(&program.cubin);
	free(blob);
}

#define LATCH_FIELD(slot, field) QmdField((slot),1 ? field,0 ? field)

static SparkStatus DependentLaunch(Fake *fake, SparkNvrmProgram *program, SparkNvrmLaunchRequest *request, uint8_t *slot, uint64_t address,
	uint32_t dependent, uint64_t *start)
{
	uint64_t value = 0u;
	*start = fake->gpu.compute.push_used;
	request->slot = slot;
	request->slot_address = address;
	request->dependent = dependent;
	return(SparkNvrmLaunch(&fake->gpu,&fake->gpu.compute,program,request,&value,"dependent"));
}

static void TestDependentLaunch(void)
{
	static Fake fake;
	static uint8_t slots[4][4096];
	const uint32_t acquire = (1u << 29) | (5u << 16) | (0x5cu >> 2),pcas = (1u << 29) | (1u << 16) | (1u << 13) | (0x02b4u >> 2);
	SparkNvrmProgram program;
	SparkNvrmLaunchRequest request;
	uint64_t bytes = 0u,output = 0x1000500000ull,start = 0u,copied = 0u;
	uint8_t *blob = ReadFile(SPARK_NVRM_FIXTURE_CUBIN,&bytes),arguments[12];
	uint32_t first = 0u;
	FakeOpen(&fake);
	memset(&program,0,sizeof(program));
	CHECK(blob != 0 && SparkNvrmCubinParse(blob,bytes,"dependent",&program.cubin) == SPARK_STATUS_OK,"the fixture parses");
	if ( blob == 0 )
		return;
	program.code.address = 0x1000600000ull;
	memcpy(arguments,&output,8u);
	memset(arguments + 8u,0,4u);
	memset(&request,0,sizeof(request));
	request.kernel = SparkNvrmCubinKernelNamed(&program.cubin,"SparkNvrmFirstLight");
	request.grid[0] = 4u;
	request.grid[1] = request.grid[2] = 1u;
	request.block[0] = 128u;
	request.block[1] = request.block[2] = 1u;
	request.arguments = arguments;
	request.argument_bytes = sizeof(arguments);
	request.slot_bytes = request.kernel != 0 ? SparkNvrmLaunchSlotBytes(request.kernel) : 0u;
	if ( request.kernel == 0 )
	{
		CHECK(0,"the fixture holds the first-light kernel");
		return;
	}
	CHECK(DependentLaunch(&fake,&program,&request,slots[0],0x1000700000ull,1u,&start) == SPARK_STATUS_OK &&
		LATCH_FIELD(slots[0],NVCEC0_QMDV05_00_ARRIVE_AT_LATCH_VALID) == 1u && LATCH_FIELD(slots[0],NVCEC0_QMDV05_00_WAIT_ON_LATCH_VALID) == 0u &&
		(first = (uint32_t)LATCH_FIELD(slots[0],NVCEC0_QMDV05_00_ARRIVE_AT_LATCH_ID)) != 0u && fake.gpu.compute.latch == first,
		"the first launch on a channel arrives at a latch of its own and waits on none, dependent or not");
	CHECK(DependentLaunch(&fake,&program,&request,slots[1],0x1000701000ull,1u,&start) == SPARK_STATUS_OK && PushWord(&fake,(uint32_t)(start / 4u)) == pcas &&
		LATCH_FIELD(slots[1],NVCEC0_QMDV05_00_WAIT_ON_LATCH_VALID) == 1u && LATCH_FIELD(slots[1],NVCEC0_QMDV05_00_WAIT_ON_LATCH_ID) == first &&
		LATCH_FIELD(slots[1],NVCEC0_QMDV05_00_HOLD_CTA_LAUNCH_UNTIL_PARENT_LATCH_ACQUIRE_AND_CTA_COMPLETE) == 1u &&
		LATCH_FIELD(slots[1],NVCEC0_QMDV05_00_HOLD_MEMBAR_UNTIL_LATCH_ACQUIRE) == 1u &&
		LATCH_FIELD(slots[1],NVCEC0_QMDV05_00_LATCH_ACQUIRE_INVALIDATE_SHADER_DATA_CACHE) == 1u &&
		LATCH_FIELD(slots[1],NVCEC0_QMDV05_00_LATCH_ACQUIRE_INVALIDATE_TEXTURE_DATA_CACHE) == 1u && first < SPARK_NVRM_PROGRAM_LATCH_FIRST &&
		LATCH_FIELD(slots[1],NVCEC0_QMDV05_00_ARRIVE_AT_LATCH_ID) != first && fake.gpu.compute.latch == LATCH_FIELD(slots[1],NVCEC0_QMDV05_00_ARRIVE_AT_LATCH_ID),
		"a dependent launch after a launch pushes no acquire: its descriptor waits on the earlier latch, holds its blocks and membar, and invalidates the "
		"data caches when it acquires the latch; eager latch ids stay below the programs' half");
	CHECK(DependentLaunch(&fake,&program,&request,slots[2],0x1000702000ull,0u,&start) == SPARK_STATUS_OK && PushWord(&fake,(uint32_t)(start / 4u)) == acquire &&
		LATCH_FIELD(slots[2],NVCEC0_QMDV05_00_WAIT_ON_LATCH_VALID) == 0u,"a plain launch after a launch keeps the front-end acquire");
	CHECK(SparkNvrmCopy(&fake.gpu,&fake.gpu.compute,0x2000001000ull,0x3000002000ull,4096u,&copied,"dependent") == SPARK_STATUS_OK &&
		fake.gpu.compute.latch == 0u,"a copy ends the latch chain");
	CHECK(DependentLaunch(&fake,&program,&request,slots[3],0x1000703000ull,1u,&start) == SPARK_STATUS_OK && PushWord(&fake,(uint32_t)(start / 4u)) == acquire &&
		LATCH_FIELD(slots[3],NVCEC0_QMDV05_00_WAIT_ON_LATCH_VALID) == 0u,"a dependent launch after a copy waits with the acquire instead");
	CHECK(SparkNvrmOpBeginLatched(&fake.gpu.compute,"dependent") == SPARK_STATUS_OK,"a latched start is allowed right after a launch");
	fake.gpu.compute.after_value = 5u;
	CHECK(SparkNvrmOpBeginLatched(&fake.gpu.compute,"dependent") == SPARK_STATUS_INVALID_ARGUMENT,"but not with a wait on another channel pending");
	fake.gpu.compute.after_value = 0u;
	{
		uint32_t index,latched = 0u,acquired = 0u,pattern_holds = 1u;
		CHECK(DependentLaunch(&fake,&program,&request,slots[0],0x1000700000ull,0u,&start) == SPARK_STATUS_OK,"a plain launch starts a segment");
		for (index = 1u; index <= 40u; index++)
		{
			const uint32_t expect_latch = index != SPARK_NVRM_LATCH_SEGMENT ? 1u : 0u;
			fake.fence[0] = fake.gpu.compute.fence_value;
			if ( DependentLaunch(&fake,&program,&request,slots[index % 4u],0x1000700000ull + 0x1000ull * (index % 4u),1u,&start) != SPARK_STATUS_OK )
			{
				pattern_holds = 0u;
				break;
			}
			if ( LATCH_FIELD(slots[index % 4u],NVCEC0_QMDV05_00_WAIT_ON_LATCH_VALID) != expect_latch ||
				(expect_latch == 0u && PushWord(&fake,(uint32_t)(start / 4u)) != acquire) )
				pattern_holds = 0u;
			latched += expect_latch;
			acquired += 1u - expect_latch;
		}
		CHECK(pattern_holds != 0u && latched == 39u && acquired == 1u,
			"a chain of dependent launches goes in segments of 32: each starts with a front-end acquire and its other 31 launches wait on latches");
		SparkNvrmLatchSegmentSet(1u);
		fake.fence[0] = fake.gpu.compute.fence_value;
		CHECK(DependentLaunch(&fake,&program,&request,slots[0],0x1000700000ull,1u,&start) == SPARK_STATUS_OK &&
			LATCH_FIELD(slots[0],NVCEC0_QMDV05_00_WAIT_ON_LATCH_VALID) == 0u && PushWord(&fake,(uint32_t)(start / 4u)) == acquire,
			"with segments of one launch, every dependent launch takes the front-end acquire");
		SparkNvrmLatchSegmentSet(0u);
		CHECK(SparkNvrmLatchSegment() == SPARK_NVRM_LATCH_SEGMENT,"a segment of zero launches restores the default of 32");
	}
	SparkNvrmCubinRelease(&program.cubin);
	free(blob);
}

static void TestStatus(void)
{
	CHECK(strcmp(SparkNvrmRmStatusText(NV_ERR_NO_MEMORY),"NV_ERR_NO_MEMORY") == 0 && strcmp(SparkNvrmRmStatusText(NV_OK),"NV_OK") == 0,
		"module status codes are named");
	CHECK(strcmp(SparkNvrmRmStatusText(0x7fffffffu),"an unlisted status") == 0,"an unknown status is called unlisted");
	CHECK(SparkNvrmRmStatus(NV_OK,"status","test") == SPARK_STATUS_OK,"NV_OK is success");
	CHECK(SparkNvrmRmStatus(NV_ERR_NO_MEMORY,"status","test") == SPARK_STATUS_CAPACITY_EXCEEDED &&
		strstr(SparkDeviceLastError()->text,"NV_ERR_NO_MEMORY") != 0 && SparkDeviceLastError()->backend_code == (int32_t)NV_ERR_NO_MEMORY,
		"an out-of-memory answer is capacity exceeded, naming the status");
	CHECK(SparkNvrmRmStatus(NV_ERR_INVALID_CLASS,"status","test") == SPARK_STATUS_UNSUPPORTED,"an unknown class is unsupported");
	CHECK(SparkNvrmRmStatus(NV_ERR_INVALID_OBJECT_HANDLE,"status","test") == SPARK_STATUS_INVALID_ARGUMENT,"a bad handle is an invalid argument");
}

static void TestIntegrated(void)
{
	const uint32_t soc = DRF_DEF(0000_CTRL_GPU,_ID_INFO,_SOC,_TRUE);
	const uint32_t igpu = DRF_DEF(0000_CTRL_GPU,_ID_INFO,_SOC_TYPE,_IGPU);
	const uint32_t ats = DRF_DEF(0000_CTRL_GPU,_ID_INFO,_ATS_ENABLED,_TRUE);
	CHECK(SparkNvrmGpuIntegrated(soc) == 1u && SparkNvrmGpuIntegrated(soc | igpu | ats) == 1u,"a GPU the module calls an SoC is integrated");
	CHECK(SparkNvrmGpuIntegrated(0u) == 0u && SparkNvrmGpuIntegrated(ats) == 0u && SparkNvrmGpuIntegrated(igpu) == 0u,
		"a GPU the module does not call an SoC is not, whatever its other identity flags");
}

static int SemaphoreAt(const uint32_t *words, uint32_t at, uint64_t address, uint64_t value, SparkNvrmSemaphoreOperation operation)
{
	uint32_t expected[SPARK_NVRM_SEMAPHORE_METHOD_WORDS];
	SparkNvrmSemaphoreWords(expected,address,value,operation);
	return(memcmp(words + at,expected,sizeof(expected)) == 0);
}

static int LaunchAt(const uint32_t *words, uint32_t at, uint64_t slot)
{
	uint32_t expected[SPARK_NVRM_LAUNCH_WORDS];
	SparkNvrmLaunchWords(expected,slot);
	return(memcmp(words + at,expected,sizeof(expected)) == 0);
}

static void TestImage(void)
{
	enum { SEMAPHORE_WORDS = SPARK_NVRM_SEMAPHORE_METHOD_WORDS, LAUNCH_WORDS = SPARK_NVRM_LAUNCH_WORDS };
	static const uint64_t semaphore = 0x2000000000ull,slots = 0x2000000100ull,event = 0x3000000000ull,reached = 0x4000000000ull,done = 0x4000000040ull;
	static Fake fake;
	SparkNvrmStep steps[6];
	uint32_t words[128],count,at;
	memset(steps,0,sizeof(steps));
	memset(words,0xee,sizeof(words));
	steps[0].kind = SPARK_NVRM_STEP_LAUNCH;
	steps[1].kind = SPARK_NVRM_STEP_LAUNCH;
	steps[1].slot = 0x400u;
	steps[2].kind = SPARK_NVRM_STEP_RECORD;
	steps[2].address = event + SPARK_NVRM_MARK_TIME_OFFSET;
	steps[2].second_address = event;
	steps[3].kind = SPARK_NVRM_STEP_AWAIT;
	steps[3].address = event;
	steps[4].kind = SPARK_NVRM_STEP_HOST;
	steps[4].address = reached;
	steps[4].second_address = done;
	steps[5].kind = SPARK_NVRM_STEP_LAUNCH;
	steps[5].slot = 0x800u;
	count = SparkNvrmImageWords(steps,6u);
	SparkNvrmImageWrite(words,semaphore,slots,steps,6u);
	CHECK(count == 14u * SEMAPHORE_WORDS + 3u * LAUNCH_WORDS && words[count] == 0xeeeeeeeeu && words[count - 1u] != 0xeeeeeeeeu,
		"a segment of three launches, a record, a wait and a host step has the words it counts, and no more");
	CHECK(SemaphoreAt(words,0u,semaphore,0u,SPARK_NVRM_SEMAPHORE_RELEASE),"a run first sets the program's semaphore to zero");
	at = SEMAPHORE_WORDS;
	CHECK(LaunchAt(words,at,slots),"the first launch needs no acquire");
	at += LAUNCH_WORDS;
	CHECK(SemaphoreAt(words,at,semaphore,1u,SPARK_NVRM_SEMAPHORE_ACQUIRE) && LaunchAt(words,at + SEMAPHORE_WORDS,slots + 0x400u),
		"the second launch waits for the first to release 1, then sends its own descriptor");
	at += SEMAPHORE_WORDS + LAUNCH_WORDS;
	CHECK(SemaphoreAt(words,at,semaphore,2u,SPARK_NVRM_SEMAPHORE_ACQUIRE) && steps[2].payload == at + SEMAPHORE_WORDS &&
		SemaphoreAt(words,steps[2].payload,event + SPARK_NVRM_MARK_TIME_OFFSET,0u,SPARK_NVRM_SEMAPHORE_RELEASE_TIME) &&
		steps[2].second == steps[2].payload + SEMAPHORE_WORDS && SemaphoreAt(words,steps[2].second,event,0u,SPARK_NVRM_SEMAPHORE_MAX) &&
		SemaphoreAt(words,steps[2].second + SEMAPHORE_WORDS,semaphore,3u,SPARK_NVRM_SEMAPHORE_RELEASE),
		"a record waits for the launches before it, stamps the event's time, raises its generation and releases 3");
	at = steps[2].second + 2u * SEMAPHORE_WORDS;
	CHECK(SemaphoreAt(words,at,semaphore,3u,SPARK_NVRM_SEMAPHORE_ACQUIRE) && steps[3].payload == at + SEMAPHORE_WORDS &&
		SemaphoreAt(words,steps[3].payload,event,0u,SPARK_NVRM_SEMAPHORE_ACQUIRE) &&
		SemaphoreAt(words,steps[3].payload + SEMAPHORE_WORDS,semaphore,4u,SPARK_NVRM_SEMAPHORE_RELEASE),
		"a wait acquires the event's generation, then releases 4");
	at = steps[3].payload + 2u * SEMAPHORE_WORDS;
	CHECK(SemaphoreAt(words,at,semaphore,4u,SPARK_NVRM_SEMAPHORE_ACQUIRE) && steps[4].payload == at + SEMAPHORE_WORDS &&
		SemaphoreAt(words,steps[4].payload,reached,0u,SPARK_NVRM_SEMAPHORE_RELEASE) && steps[4].second == steps[4].payload + SEMAPHORE_WORDS &&
		SemaphoreAt(words,steps[4].second,done,0u,SPARK_NVRM_SEMAPHORE_ACQUIRE) &&
		SemaphoreAt(words,steps[4].second + SEMAPHORE_WORDS,semaphore,5u,SPARK_NVRM_SEMAPHORE_RELEASE),
		"a host step releases its ticket to the worker, acquires the worker's answer, then releases 5");
	at = steps[4].second + 2u * SEMAPHORE_WORDS;
	CHECK(SemaphoreAt(words,at,semaphore,5u,SPARK_NVRM_SEMAPHORE_ACQUIRE) && LaunchAt(words,at + SEMAPHORE_WORDS,slots + 0x800u) &&
		at + SEMAPHORE_WORDS + LAUNCH_WORDS == count,"the last launch waits for the host step and ends the segment");
	SparkNvrmImagePatch(words,steps[4].payload,0x123456789abcull);
	SparkNvrmImagePatchAddress(words,steps[4].payload,0x4100000000ull);
	CHECK(SparkNvrmImagePayload(words,steps[4].payload) == 0x123456789abcull &&
		SemaphoreAt(words,steps[4].payload,0x4100000000ull,0x123456789abcull,SPARK_NVRM_SEMAPHORE_RELEASE),
		"a run patches a step's payload and address in place");
	FakeOpen(&fake);
	CHECK(SparkNvrmSemaphoreAcquire(&fake.gpu.compute,semaphore,9u,"image") == SPARK_STATUS_OK &&
		SparkNvrmSemaphoreRelease(&fake.gpu.compute,event + SPARK_NVRM_MARK_TIME_OFFSET,9u,1u,"image") == SPARK_STATUS_OK,
		"push the same semaphores the eager path pushes");
	SparkNvrmSemaphoreWords(words,semaphore,9u,SPARK_NVRM_SEMAPHORE_ACQUIRE);
	SparkNvrmSemaphoreWords(words + SEMAPHORE_WORDS,event + SPARK_NVRM_MARK_TIME_OFFSET,9u,SPARK_NVRM_SEMAPHORE_RELEASE_TIME);
	CHECK(memcmp(fake.push,words,2u * SEMAPHORE_WORDS * sizeof(uint32_t)) == 0,"a segment's semaphores are word for word the ones the eager path pushes");
}

static void TestImageChain(void)
{
	enum { SEMAPHORE_WORDS = SPARK_NVRM_SEMAPHORE_METHOD_WORDS, LAUNCH_WORDS = SPARK_NVRM_LAUNCH_WORDS, STEPS = 74, HOST = 70, PLAIN = 10, LATER_PLAIN = 50 };
	static const uint64_t semaphore = 0x2000000000ull,slots = 0x2000000100ull;
	static SparkNvrmStep steps[STEPS];
	static uint32_t words[1024];
	uint32_t index,count,at,chained = 0u,marked = 1u,laid = 1u;
	memset(steps,0,sizeof(steps));
	memset(words,0xee,sizeof(words));
	for ( index = 0u; index < STEPS; index++ )
	{
		steps[index].kind = index == HOST ? SPARK_NVRM_STEP_HOST : SPARK_NVRM_STEP_LAUNCH;
		steps[index].dependent = index == PLAIN || index == LATER_PLAIN || index == HOST ? 0u : 1u;
		steps[index].slot = 0x400ull * index;
		steps[index].address = 0x4000000000ull;
		steps[index].second_address = 0x4000000040ull;
	}
	SparkNvrmImageChain(steps,STEPS,SPARK_NVRM_LATCH_SEGMENT);
	for ( index = 0u; index < STEPS; index++ )
	{
		const uint32_t head = index == 0u || index == PLAIN || index == 42u || index == LATER_PLAIN || index == HOST || index == HOST + 1u ? 1u : 0u;
		chained += steps[index].chained;
		if ( steps[index].chained != 1u - head )
			marked = 0u;
	}
	CHECK(marked != 0u && chained == 68u,
		"back-to-back dependent launches chain in segments of 32; a plain launch heads a chain, and a host step ends a chain and the launch after it starts one");
	count = SparkNvrmImageWords(steps,STEPS);
	SparkNvrmImageWrite(words,semaphore,slots,steps,STEPS);
	CHECK(count == 9u * SEMAPHORE_WORDS + 73u * LAUNCH_WORDS && words[count] == 0xeeeeeeeeu,
		"every launch is in the segment, and only the reset, four chain heads and the host step push semaphores");
	for ( index = 0u, at = SEMAPHORE_WORDS; index < STEPS && laid != 0u; index++ )
	{
		if ( index == HOST )
		{
			laid = SemaphoreAt(words,at,semaphore,HOST,SPARK_NVRM_SEMAPHORE_ACQUIRE) && steps[HOST].payload == at + SEMAPHORE_WORDS &&
				SemaphoreAt(words,steps[HOST].second + SEMAPHORE_WORDS,semaphore,HOST + 1u,SPARK_NVRM_SEMAPHORE_RELEASE);
			at = steps[HOST].second + 2u * SEMAPHORE_WORDS;
			continue;
		}
		if ( index != 0u && steps[index].chained == 0u )
		{
			laid = SemaphoreAt(words,at,semaphore,index,SPARK_NVRM_SEMAPHORE_ACQUIRE);
			at += SEMAPHORE_WORDS;
		}
		laid = laid && LaunchAt(words,at,slots + 0x400ull * index);
		at += LAUNCH_WORDS;
	}
	CHECK(laid != 0u && at == count,
		"a chained launch follows the launch before it with no acquire; a chain's head acquires the step before it, and the host step waits for the last chain");
	for ( index = 0u; index < STEPS; index++ )
		steps[index].chained = 0u;
	SparkNvrmImageChain(steps,STEPS,1u);
	for ( index = 0u, chained = 0u; index < STEPS; index++ )
		chained += steps[index].chained;
	CHECK(chained == 0u && SparkNvrmImageWords(steps,STEPS) == 77u * SEMAPHORE_WORDS + 73u * LAUNCH_WORDS,
		"in segments of one launch nothing chains, and every launch after the first takes the acquire");
	for ( index = 0u; index < STEPS; index++ )
		steps[index].dependent = 0u;
	SparkNvrmImageChain(steps,STEPS,SPARK_NVRM_LATCH_SEGMENT);
	for ( index = 0u, chained = 0u; index < STEPS; index++ )
		chained += steps[index].chained;
	CHECK(chained == 0u && SparkNvrmImageWords(steps,STEPS) == 77u * SEMAPHORE_WORDS + 73u * LAUNCH_WORDS,
		"plain launches never chain: each waits for the launch before it to end, as a plain kernel boundary does");
}

static void TestLatchRanges(void)
{
	uint32_t first = 0u,second = 0u,reused = 0u,later = 0u,huge = 1u;
	CHECK(SparkNvrmLatchRangeTake(10u,&first,"ranges") == SPARK_STATUS_OK && first >= SPARK_NVRM_PROGRAM_LATCH_FIRST &&
		SparkNvrmLatchRangeTake(5u,&second,"ranges") == SPARK_STATUS_OK && second == first + 10u,
		"programs take latch ids from the top half, one range after another");
	SparkNvrmLatchRangeGive(first,10u);
	CHECK(SparkNvrmLatchRangeTake(4u,&reused,"ranges") == SPARK_STATUS_OK && reused == first &&
		SparkNvrmLatchRangeTake(7u,&later,"ranges") == SPARK_STATUS_OK && later == second + 5u,
		"a range given back is taken again by a program it fits, and a larger one goes past the ranges in use");
	CHECK(SparkNvrmLatchRangeTake(SPARK_NVRM_PROGRAM_LATCH_FIRST,&huge,"ranges") == SPARK_STATUS_CAPACITY_EXCEEDED && huge == 0u,
		"a program larger than the free ids is refused");
	SparkNvrmLatchRangeGive(reused,4u);
	SparkNvrmLatchRangeGive(second,5u);
	SparkNvrmLatchRangeGive(later,7u);
	CHECK(SparkNvrmLatchRangeTake(3u,&first,"ranges") == SPARK_STATUS_OK && first == SPARK_NVRM_PROGRAM_LATCH_FIRST,"every range given back frees its ids");
	SparkNvrmLatchRangeGive(first,3u);
}

static void TestSegment(void)
{
	static Fake fake;
	const uint64_t segment = 0x5000000000ull,done = 0x2000000000ull;
	SparkNvrmChannel *channel;
	FakeOpen(&fake);
	channel = &fake.gpu.compute;
	CHECK(SparkNvrmSubmitSegment(&fake.gpu,channel,segment,0u,done,6u,"segment") == SPARK_STATUS_INVALID_ARGUMENT &&
		SparkNvrmSubmitSegment(&fake.gpu,channel,segment + 2u,100u,done,6u,"segment") == SPARK_STATUS_INVALID_ARGUMENT &&
		SparkNvrmSubmitSegment(&fake.gpu,channel,segment,SPARK_NVRM_SEGMENT_WORDS_MAX + 1u,done,6u,"segment") == SPARK_STATUS_INVALID_ARGUMENT &&
		SparkNvrmSubmitSegment(&fake.gpu,channel,(1ull << 40) - 64u,100u,done,6u,"segment") == SPARK_STATUS_INVALID_ARGUMENT && GpPut(&fake) == 0u,
		"a segment of no words, at an unaligned address, over 2^21 - 1 words or past 2^40 is refused before anything is queued");
	CHECK(SparkNvrmSemaphoreAcquire(channel,FENCE_ADDRESS,0u,"segment") == SPARK_STATUS_OK &&
		SparkNvrmSubmitSegment(&fake.gpu,channel,segment,100u,done,6u,"segment") == SPARK_STATUS_OK,"a run with a head submits");
	CHECK(GpPut(&fake) == 3u && ControlWord64(&fake,0) == (PUSH_ADDRESS | (6ull << 42) | (1ull << 41)) &&
		ControlWord64(&fake,1) == (segment | (100ull << 42) | (1ull << 41)) && ControlWord64(&fake,2) == ((PUSH_ADDRESS + 24u) | (12ull << 42) | (1ull << 41)),
		"three ring entries: the head from the push buffer, the prebuilt segment, then the tail after the head");
	CHECK(SemaphoreAt((const uint32_t *)(const void *)fake.push,6u,done,6u,SPARK_NVRM_SEMAPHORE_ACQUIRE) &&
		SemaphoreAt((const uint32_t *)(const void *)fake.push,12u,FENCE_ADDRESS,1u,SPARK_NVRM_SEMAPHORE_RELEASE),
		"the tail waits for the program's last step, then releases the channel's next fence value");
	CHECK(channel->fence_value == 1u && channel->last_engine == SPARK_NVRM_ENGINE_COMPUTE && channel->segment_count == 3u &&
		channel->segments[1].start == channel->segments[1].end && channel->segments[2].value == 1u && fake.usermode[DOORBELL_WORD] == 0x1234abcdu,
		"the run is one fenced operation whose three entries retire together, and the doorbell rings once for them");
	CHECK(SparkNvrmSubmitSegment(&fake.gpu,channel,segment,100u,done,6u,"segment") == SPARK_STATUS_OK && GpPut(&fake) == 5u &&
		ControlWord64(&fake,3) == (segment | (100ull << 42) | (1ull << 41)) && channel->fence_value == 2u,
		"a run with nothing to order first queues the segment and its tail alone");
}

int main(void)
{
	TestIntegrated();
	TestPush();
	TestSubmit();
	TestReclaim();
	TestOrdering();
	TestFillSplit();
	TestLongLines();
	TestPageSizes();
	TestModuleZeroes();
	TestMarks();
	TestAddress();
	TestSemaphore();
	TestLaunch();
	TestDependentLaunch();
	TestImage();
	TestImageChain();
	TestLatchRanges();
	TestSegment();
	TestStatus();
	if ( failures != 0 )
	{
		printf("\nFAIL (%d)\n",failures);
		return(1);
	}
	printf("PASS push methods, ring entries, the doorbell, push-buffer reuse, ordered copies and fills, semaphore releases, launch slots, dependent launches chained by latch, program segments, chains and runs, and the device address map\n");
	return(0);
}
