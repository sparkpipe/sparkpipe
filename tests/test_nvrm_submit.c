#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

#include "sparkpipe/spark_device.h"
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

int main(void)
{
	TestIntegrated();
	TestPush();
	TestSubmit();
	TestReclaim();
	TestOrdering();
	TestFillSplit();
	TestMarks();
	TestAddress();
	TestSemaphore();
	TestLaunch();
	TestStatus();
	if ( failures != 0 )
	{
		printf("\nFAIL (%d)\n",failures);
		return(1);
	}
	printf("PASS push methods, ring entries, the doorbell, push-buffer reuse, ordered copies and fills, semaphore releases, launch slots and the device address map\n");
	return(0);
}
