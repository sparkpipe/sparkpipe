#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "sparkpipe/spark_gpu.h"
#include "runtime/launch.h"
#include "sparkpipe/spark_tp_mesh_kernels.cuh"

#define RECORDS_PREV_TAG 0x1234ull
#define RECORDS_SECOND_TAG 0x5678ull
#define RECORDS_SHIPPED 0x1200ull
#define RECORDS_DEADLINE_NS 2000000ull
#define RECORDS_SLOT_BYTES 4096ull
#define RECORDS_PAYLOAD_BYTES 64ull

static int failures;

#define CHECK(condition, what) \
	do { \
		if ( !(condition) ) \
		{ \
			fprintf(stderr,"FAIL %s: %s\n",what,SparkGpuGetErrorString(SparkGpuGetLastError())); \
			failures++; \
		} \
	} while (0)

static void RecordsArm(SparkTpMeshRoundControl *device, uint64_t tag)
{
	SparkTpMeshRoundControl control;
	memset(&control,0,sizeof(control));
	CHECK(SparkGpuMemcpy(&control,device,sizeof(control),SPARK_GPU_COPY_DEVICE_TO_HOST) == SPARK_GPU_SUCCESS,"read the round control");
	control.slot_cursor = 0u;
	control.rounds_total = 1u;
	control.rounds_done = 0u;
	control.round_seq = tag;
	control.cancel_expected = 0u;
	control.deadline_ns = RECORDS_DEADLINE_NS;
	control.error_word = 0u;
	control.diag_word = 0u;
	CHECK(SparkGpuMemcpy(device,&control,sizeof(control),SPARK_GPU_COPY_HOST_TO_DEVICE) == SPARK_GPU_SUCCESS,"arm the round control");
}

static void RecordsRead(SparkTpMeshRoundControl *device, SparkTpMeshRoundControl *control)
{
	CHECK(SparkGpuMemcpy(control,device,sizeof(*control),SPARK_GPU_COPY_DEVICE_TO_HOST) == SPARK_GPU_SUCCESS,"read back the round control");
}

int main(int argc, char **argv)
{
	const char *flavour = argc > 1 ? argv[1] : "this";
	SparkTpMeshRoundControl *control = 0,host;
	uint64_t *band = 0,*cells = 0,*local = 0,*full = 0,shipped = RECORDS_SHIPPED,output[2] = {17u, 29u};
	SparkGpuStream stream = 0;
	CHECK(SparkGpuSetDevice(0) == SPARK_GPU_SUCCESS && SparkGpuStreamCreateWithFlags(&stream,SPARK_GPU_STREAM_NON_BLOCKING) == SPARK_GPU_SUCCESS,"open the device");
	CHECK(SparkGpuMalloc((void **)&control,sizeof(*control)) == SPARK_GPU_SUCCESS && SparkGpuMemset(control,0,sizeof(*control)) == SPARK_GPU_SUCCESS,
		"allocate the round control");
	CHECK(SparkGpuMalloc((void **)&band,2u * 2u * RECORDS_SLOT_BYTES) == SPARK_GPU_SUCCESS && SparkGpuMemset(band,0,2u * 2u * RECORDS_SLOT_BYTES) == SPARK_GPU_SUCCESS,
		"allocate the band");
	CHECK(SparkGpuMalloc((void **)&cells,4u * sizeof(uint64_t)) == SPARK_GPU_SUCCESS && SparkGpuMemset(cells,0,4u * sizeof(uint64_t)) == SPARK_GPU_SUCCESS,
		"allocate the doorbell, shipped and cancel cells");
	CHECK(SparkGpuMalloc((void **)&local,RECORDS_PAYLOAD_BYTES) == SPARK_GPU_SUCCESS && SparkGpuMalloc((void **)&full,2u * RECORDS_PAYLOAD_BYTES) == SPARK_GPU_SUCCESS,
		"allocate the payloads");
	CHECK(SparkGpuMemcpy(cells + 1u,&shipped,sizeof(shipped),SPARK_GPU_COPY_HOST_TO_DEVICE) == SPARK_GPU_SUCCESS,"a shipped cell that never acknowledges the round");
	if ( failures != 0 )
		return(1);
	RecordsArm(control,RECORDS_PREV_TAG);
	CHECK(SparkTpLaunchMeshRoundLoop(stream,band,RECORDS_SLOT_BYTES,2u,cells,cells + 1u,cells + 2u,control,0u,2u,local,full,RECORDS_PAYLOAD_BYTES) ==
		SPARK_GPU_SUCCESS && SparkGpuStreamSynchronize(stream) == SPARK_GPU_SUCCESS,"run a round loop whose ship never acknowledges");
	RecordsRead(control,&host);
	CHECK(host.error_word == RECORDS_PREV_TAG && host.fault_phase == SPARK_TP_MESH_FAULT_SHIP_ACK && host.fault_peer == 0u &&
		host.fault_want == RECORDS_PREV_TAG && host.fault_seen == RECORDS_SHIPPED && host.fault_hits == 1u,
		"the timeout claims the record with the round's tag and the shipped cell it saw");
	RecordsArm(control,RECORDS_SECOND_TAG);
	CHECK(SparkTpLaunchMeshRoundLoop(stream,band,RECORDS_SLOT_BYTES,2u,cells,cells + 1u,cells + 2u,control,0u,2u,local,full,RECORDS_PAYLOAD_BYTES) ==
		SPARK_GPU_SUCCESS && SparkGpuStreamSynchronize(stream) == SPARK_GPU_SUCCESS,"time out a second round before the record is read");
	RecordsRead(control,&host);
	CHECK(host.error_word == RECORDS_SECOND_TAG && host.fault_want == RECORDS_PREV_TAG && host.fault_hits == 2u,
		"a later timeout counts a hit and leaves the first record");
	CHECK(SparkGpuMemcpy(local,output,sizeof(output),SPARK_GPU_COPY_HOST_TO_DEVICE) == SPARK_GPU_SUCCESS,"seed the guarded output");
	CHECK(SparkTpLaunchMeshGuard(stream,&control->error_word,local) == SPARK_GPU_SUCCESS &&
		SparkTpLaunchMeshGuard(stream,&control->error_word,local) == SPARK_GPU_SUCCESS && SparkGpuStreamSynchronize(stream) == SPARK_GPU_SUCCESS,
		"guard a failed round twice");
	CHECK(SparkGpuMemcpy(output,local,sizeof(output),SPARK_GPU_COPY_DEVICE_TO_HOST) == SPARK_GPU_SUCCESS && output[0] == UINT64_MAX && output[1] == 29u,
		"the guard poisons only the output word");
	RecordsRead(control,&host);
	CHECK(host.guard_poisons == 2u,"the round control counts each poison");
	(void)SparkTpMeshRecordPrint(stdout,0u,(const volatile uint64_t *)(const void *)&host);
	SparkGpuFree(control);
	SparkGpuFree(band);
	SparkGpuFree(cells);
	SparkGpuFree(local);
	SparkGpuFree(full);
	SparkGpuStreamDestroy(stream);
	if ( failures != 0 )
	{
		fprintf(stderr,"FAIL mesh records on the %s flavour: %d failures\n",flavour,failures);
		return(1);
	}
	fprintf(stderr,"PASS mesh records on the %s flavour\n",flavour);
	return(0);
}
