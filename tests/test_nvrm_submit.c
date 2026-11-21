#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_device.h"
#include "runtime/device/nvrm/nvrm_launch.h"
#include "runtime/device/nvrm/nvrm_rm.h"
#include "nvstatus.h"

#define MW(range) range
#include "classes/compute/clcec0qmd.h"

#ifndef SPARK_NVRM_FIXTURE_CUBIN
#define SPARK_NVRM_FIXTURE_CUBIN "tests/fixtures/nvrm/fixture_kernels.sm_121a.cubin"
#endif

#define PUSH_ADDRESS 0x1000200000ull
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
	channel->push_used = sizeof(fake.push) - 4u;
	CHECK(SparkNvrmPush(channel,1u,0x02b4u,values,1u,"push") == SPARK_STATUS_CAPACITY_EXCEEDED && channel->push_used == sizeof(fake.push) - 4u,
		"a method past the end of the push buffer is refused and nothing is written");
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
	CHECK(ControlWord64(&fake,0) == (PUSH_ADDRESS | (4ull << 42) | (1ull << 41)),"the ring entry holds the address, the word count at bit 42 and bit 41");
	CHECK(GpPut(&fake) == 1u && fake.usermode[DOORBELL_WORD] == 0x1234abcdu,"GPPut advances and the doorbell carries the channel's token");
	for (round = 1u; round < CONTROL_ENTRIES + 1u; round++)
	{
		(void)SparkNvrmPush(channel,1u,0x02b4u,&value,1u,"submit");
		(void)SparkNvrmSubmit(&fake.gpu,channel,"submit");
	}
	CHECK(GpPut(&fake) == 1u && ControlWord64(&fake,0) == ((PUSH_ADDRESS + 8u * (CONTROL_ENTRIES + 1u)) | (2ull << 42) | (1ull << 41)),
		"the ring wraps: GPPut returns to 1 and entry 0 holds the newest submission");
	CHECK(SparkNvrmSubmit(&fake.gpu,channel,"submit") == SPARK_STATUS_INVALID_ARGUMENT,"a submission with nothing new is refused");
	(void)SparkNvrmPush(channel,1u,0x02b4u,&value,1u,"submit");
	(void)SparkNvrmPush(channel,1u,0x02c0u,&value,1u,"submit");
	(void)SparkNvrmPush(channel,1u,0x02b4u,&value,1u,"submit");
	CHECK(SparkNvrmSubmit(&fake.gpu,channel,"submit") == SPARK_STATUS_OK && ControlWord64(&fake,1) == ((PUSH_ADDRESS + 80u) | (6ull << 42) | (1ull << 41)),
		"a submission sends every method pushed since the last one");
	CHECK(SparkNvrmPushRewind(channel,"submit") == SPARK_STATUS_OK && channel->push_used == 0u,"a fully submitted push buffer rewinds");
	(void)SparkNvrmPush(channel,1u,0x02b4u,&value,1u,"submit");
	CHECK(SparkNvrmPushRewind(channel,"submit") == SPARK_STATUS_BUSY && channel->push_used == 8u,"unsubmitted methods refuse a rewind");
	channel->push.address = 1ull << 40;
	CHECK(SparkNvrmSubmit(&fake.gpu,channel,"submit") == SPARK_STATUS_INVALID_ARGUMENT,"a push buffer at or above 2^40 is refused");
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
	CHECK(request.kernel != 0 && SparkNvrmLaunchQueue(&fake.gpu,&program,&request,"launch") == SPARK_STATUS_OK,"the first-light kernel queues");
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
		CHECK(SparkNvrmLaunchQueue(&fake.gpu,&program,&request,"launch") == SPARK_STATUS_INVALID_ARGUMENT,
			"a kernel needing more local memory than provisioned is refused");
		fake.gpu.local_bytes_per_thread = 0x240u;
		request.slot_address += 64u;
		CHECK(SparkNvrmLaunchQueue(&fake.gpu,&program,&request,"launch") == SPARK_STATUS_INVALID_ARGUMENT,"a slot off 256 bytes is refused");
		request.slot_address -= 64u;
		request.slot_bytes -= 1u;
		CHECK(SparkNvrmLaunchQueue(&fake.gpu,&program,&request,"launch") == SPARK_STATUS_INVALID_ARGUMENT,"a slot too small for bank 0 is refused");
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

int main(void)
{
	TestPush();
	TestSubmit();
	TestSemaphore();
	TestLaunch();
	TestStatus();
	if ( failures != 0 )
	{
		printf("\nFAIL (%d)\n",failures);
		return(1);
	}
	printf("PASS push methods, ring entries, the doorbell, semaphore releases and launch slots are encoded as the class headers say\n");
	return(0);
}
