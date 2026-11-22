#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "runtime/device/nvrm/nvrm_gpu.h"
#include "runtime/device/nvrm/nvrm_wait.h"
#include "nvgputypes.h"
#include "nvos.h"
#include "nverror.h"

static int failures;

#define CHECK(condition, what) \
	do { \
		if ( !(condition) ) \
		{ \
			printf("FAIL %s\n",what); \
			failures++; \
		} \
	} while (0)

static NvNotification notes[NV_CHANNELGPFIFO_NOTIFICATION_TYPE__SIZE_1];

static SparkNvrmChannel *Channel(void)
{
	static SparkNvrmChannel channel;
	memset(&channel,0,sizeof(channel));
	memset((void *)notes,0,sizeof(notes));
	channel.notifier.cpu = (uint8_t *)(void *)notes;
	channel.notifier.bytes = sizeof(notes);
	return(&channel);
}

static void TestNotifier(void)
{
	SparkNvrmChannel *channel = Channel();
	SparkNvrmFault fault;
	CHECK(SparkNvrmChannelFault(channel,&fault) == 0u && fault.code == 0u,"a fresh error notifier reports no fault");
	notes[NV_CHANNELGPFIFO_NOTIFICATION_TYPE_WORK_SUBMIT_TOKEN].info32 = 0x40000002u;
	notes[NV_CHANNELGPFIFO_NOTIFICATION_TYPE_WORK_SUBMIT_TOKEN].status = 0xffffu;
	CHECK(SparkNvrmChannelFault(channel,&fault) == 0u,"the work-submit token's notification is not a fault");
	notes[NV_CHANNELGPFIFO_NOTIFICATION_TYPE_ERROR].info32 = ROBUST_CHANNEL_GR_EXCEPTION;
	CHECK(SparkNvrmChannelFault(channel,&fault) == 0u,"an error code without the module's raised status is not a fault yet");
	notes[NV_CHANNELGPFIFO_NOTIFICATION_TYPE_ERROR].info16 = 1u;
	notes[NV_CHANNELGPFIFO_NOTIFICATION_TYPE_ERROR].status = 0xffffu;
	CHECK(SparkNvrmChannelFault(channel,&fault) == 1u && fault.code == ROBUST_CHANNEL_GR_EXCEPTION && fault.engine == 1u && fault.located == 0u,
		"a raised error notifier reports its code and engine");
	channel->notifier.cpu = 0;
	CHECK(SparkNvrmChannelFault(channel,&fault) == 0u,"a channel without a notifier reports no fault");
	CHECK(SparkNvrmChannelFault(0,&fault) == 0u,"no channel reports no fault");
}

static void TestText(void)
{
	SparkNvrmFault fault;
	char text[128];
	memset(&fault,0,sizeof(fault));
	fault.code = ROBUST_CHANNEL_GR_EXCEPTION;
	SparkNvrmFaultText(&fault,text,sizeof(text));
	CHECK(strcmp(text,"GPU fault 13 (graphics engine exception)") == 0,"a graphics exception is named with its code");
	fault.code = ROBUST_CHANNEL_FIFO_ERROR_MMU_ERR_FLT;
	fault.located = 1u;
	fault.address = 0x7f0000001000ull;
	snprintf(fault.kind,sizeof(fault.kind),"%s","FAULT_PDE");
	SparkNvrmFaultText(&fault,text,sizeof(text));
	CHECK(strcmp(text,"GPU fault 31 (MMU fault, FAULT_PDE at 0x7f0000001000)") == 0,"an MMU fault carries its type and address");
	fault.located = 0u;
	SparkNvrmFaultText(&fault,text,sizeof(text));
	CHECK(strcmp(text,"GPU fault 31 (MMU fault)") == 0,"an MMU fault the module could not locate carries no address");
	CHECK(strcmp(SparkNvrmFaultName(ROBUST_CHANNEL_CE1_ERROR),"copy engine error") == 0,"copy engine codes are named as such");
	CHECK(strcmp(SparkNvrmFaultName(200u),"channel error") == 0,"an unknown code is still a channel error");
}

static void TestLocateOnlyMmuFaults(void)
{
	static SparkNvrmGpu gpu;
	SparkNvrmChannel *channel = Channel();
	SparkNvrmFault fault;
	memset(&gpu,0,sizeof(gpu));
	memset(&fault,0,sizeof(fault));
	fault.code = ROBUST_CHANNEL_GR_EXCEPTION;
	channel->handle = 0x1234u;
	SparkNvrmFaultLocate(&gpu,channel,&fault);
	CHECK(fault.located == 0u && fault.address == 0u,"only an MMU fault asks the module where it happened");
	fault.code = ROBUST_CHANNEL_FIFO_ERROR_MMU_ERR_FLT;
	channel->handle = 0u;
	SparkNvrmFaultLocate(&gpu,channel,&fault);
	CHECK(fault.located == 0u,"a channel without a handle is not asked");
}

static void TestWatchCadence(void)
{
	SparkNvrmWait wait;
	SparkNvrmWaitBegin(&wait);
	wait.elapsed = 50000u;
	CHECK(SparkNvrmWaitDue(&wait,SPARK_NVRM_WAIT_WATCH_NS) == 0,"a wait inside its spin budget never looks for faults");
	wait.sleep_ns = SPARK_NVRM_WAIT_SLEEP_FIRST_NS;
	wait.elapsed = SPARK_NVRM_WAIT_SPIN_NS;
	CHECK(SparkNvrmWaitDue(&wait,SPARK_NVRM_WAIT_WATCH_NS) == 1,"the first look comes when the wait starts to sleep");
	wait.elapsed += SPARK_NVRM_WAIT_WATCH_NS - 1u;
	CHECK(SparkNvrmWaitDue(&wait,SPARK_NVRM_WAIT_WATCH_NS) == 0,"looks are at least the watch period apart");
	wait.elapsed += 1u;
	CHECK(SparkNvrmWaitDue(&wait,SPARK_NVRM_WAIT_WATCH_NS) == 1,"the next look comes one watch period later");
	SparkNvrmWaitEnd(&wait);
}

static void TestStall(void)
{
	SparkNvrmWait wait;
	SparkNvrmWaitBegin(&wait);
	wait.elapsed = 100000u;
	CHECK(SparkNvrmWaitStalled(&wait,5u,SPARK_NVRM_WAIT_STALL_NS) == 0,"the first progress reading starts the clock");
	wait.elapsed += SPARK_NVRM_WAIT_STALL_NS - 1u;
	CHECK(SparkNvrmWaitStalled(&wait,5u,SPARK_NVRM_WAIT_STALL_NS) == 0,"no stall before the deadline");
	wait.elapsed += 1u;
	CHECK(SparkNvrmWaitStalled(&wait,5u,SPARK_NVRM_WAIT_STALL_NS) == 1,"no progress for the whole deadline is a stall");
	wait.elapsed += 1000u;
	CHECK(SparkNvrmWaitStalled(&wait,6u,SPARK_NVRM_WAIT_STALL_NS) == 0,"progress restarts the clock");
	wait.elapsed += SPARK_NVRM_WAIT_STALL_NS - 1u;
	CHECK(SparkNvrmWaitStalled(&wait,6u,SPARK_NVRM_WAIT_STALL_NS) == 0,"the restarted clock runs a whole deadline");
	wait.elapsed += 1u;
	CHECK(SparkNvrmWaitStalled(&wait,6u,SPARK_NVRM_WAIT_STALL_NS) == 1,"and then reports the stall");
	SparkNvrmWaitEnd(&wait);
}

int main(void)
{
	TestNotifier();
	TestText();
	TestLocateOnlyMmuFaults();
	TestWatchCadence();
	TestStall();
	if ( failures != 0 )
	{
		printf("\nFAIL (%d)\n",failures);
		return(1);
	}
	printf("PASS the error notifier, fault names, MMU locations, watch cadence and stall deadline\n");
	return(0);
}
