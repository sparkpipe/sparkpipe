#include <stdio.h>
#include <string.h>

#include "runtime/device/device_backend.h"
#include "runtime/device/nvrm/nvrm_gpu.h"
#include "nvgputypes.h"
#include "nvos.h"
#include "nverror.h"
#include "ctrl/ctrl906f.h"

#define FAULT_NOTIFIER_RAISED 0xffffu
#define FAULT_KIND_NONE "NONE"

uint32_t SparkNvrmChannelFault(const SparkNvrmChannel *channel, SparkNvrmFault *fault)
{
	const volatile NvNotification *note;
	memset(fault,0,sizeof(*fault));
	if ( channel == 0 || channel->notifier.cpu == 0 )
		return(0u);
	note = (const volatile NvNotification *)(const void *)channel->notifier.cpu + NV_CHANNELGPFIFO_NOTIFICATION_TYPE_ERROR;
	if ( note->status != FAULT_NOTIFIER_RAISED )
		return(0u);
	__atomic_thread_fence(__ATOMIC_ACQUIRE);
	fault->code = (uint32_t)note->info32;
	fault->engine = (uint32_t)note->info16;
	return(1u);
}

void SparkNvrmFaultLocate(SparkNvrmGpu *gpu, const SparkNvrmChannel *channel, SparkNvrmFault *fault)
{
	NV906F_CTRL_GET_MMU_FAULT_INFO_PARAMS info;
	if ( fault->code != ROBUST_CHANNEL_FIFO_ERROR_MMU_ERR_FLT || fault->located != 0u || channel == 0 || channel->handle == 0u )
		return;
	memset(&info,0,sizeof(info));
	if ( SparkNvrmRmControl(&gpu->rm,channel->handle,NV906F_CTRL_CMD_GET_MMU_FAULT_INFO,&info,(uint32_t)sizeof(info),"fault") != SPARK_STATUS_OK )
		return;
	info.faultString[sizeof(info.faultString) - 1u] = '\0';
	if ( info.faultString[0] == '\0' || strcmp(info.faultString,FAULT_KIND_NONE) == 0 )
		return;
	fault->located = 1u;
	fault->address = (uint64_t)info.addrHi << 32 | (uint64_t)info.addrLo;
	snprintf(fault->kind,sizeof(fault->kind),"%s",info.faultString);
}

const char *SparkNvrmFaultName(uint32_t code)
{
	if ( ROBUST_CHANNEL_IS_CE_ERROR(code) )
		return("copy engine error");
	switch ( code )
	{
	case ROBUST_CHANNEL_GR_EXCEPTION:
		return("graphics engine exception");
	case ROBUST_CHANNEL_FIFO_ERROR_MMU_ERR_FLT:
		return("MMU fault");
	case ROBUST_CHANNEL_PBDMA_ERROR:
		return("push buffer error");
	case ROBUST_CHANNEL_RESETCHANNEL_VERIF_ERROR:
		return("channel reset");
	case ROBUST_CHANNEL_GR_FAULT_DURING_CTXSW:
		return("fault during a context switch");
	case ROBUST_CHANNEL_PREEMPTIVE_REMOVAL:
		return("removed after an earlier error");
	case ROBUST_CHANNEL_GR_CLASS_ERROR:
		return("graphics class error");
	case ROBUST_CHANNEL_GPU_ECC_DBE:
		return("double-bit ECC error");
	case ROBUST_CHANNEL_CONTAINED_ERROR:
		return("contained error");
	case ROBUST_CHANNEL_UNCONTAINED_ERROR:
		return("uncontained error");
	case ROBUST_CHANNEL_GPU_HAS_FALLEN_OFF_THE_BUS:
		return("GPU fell off the bus");
	default:
		return("channel error");
	}
}

void SparkNvrmFaultText(const SparkNvrmFault *fault, char *text, uint32_t capacity)
{
	if ( fault->located != 0u )
		snprintf(text,capacity,"GPU fault %u (%s, %s at %#llx)",fault->code,SparkNvrmFaultName(fault->code),fault->kind,(unsigned long long)fault->address);
	else
		snprintf(text,capacity,"GPU fault %u (%s)",fault->code,SparkNvrmFaultName(fault->code));
}
