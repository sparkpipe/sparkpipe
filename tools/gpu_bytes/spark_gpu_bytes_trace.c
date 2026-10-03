#define _GNU_SOURCE
#include <dlfcn.h>
#include <execinfo.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <cuda.h>
#include <cupti.h>

#define GPUB_STACK_DEPTH 24
#define GPUB_LIVE_CAPACITY (1u << 20)

typedef struct GpubLive
{
	uint64_t key;
	uint64_t bytes;
	uint32_t kind;
} GpubLive;

typedef struct GpubNvmlProcess
{
	unsigned int pid;
	unsigned long long used;
	unsigned int gpu_instance;
	unsigned int compute_instance;
} GpubNvmlProcess;

typedef int (*GpubNvmlInit)(void);
typedef int (*GpubNvmlHandle)(unsigned int,void **);
typedef int (*GpubNvmlProcesses)(void *,unsigned int *,GpubNvmlProcess *);

static FILE *gpub_out;
static pthread_mutex_t gpub_lock = PTHREAD_MUTEX_INITIALIZER;
static GpubLive *gpub_live;
static uint64_t gpub_seq;
static uint64_t gpub_live_bytes[8];
static void *gpub_nvml_device;
static GpubNvmlProcesses gpub_nvml_processes;
static __thread unsigned long long gpub_enter_used;
static __thread uint64_t gpub_enter_seq;
static __thread CUgraph gpub_instantiate_graph;
static uint32_t gpub_sample_ms;
static void *gpub_seen_functions[65536];
static __thread uint32_t gpub_first_launch;

static const char *const gpub_kind_names[8] = {"device","host","register","managed","create","import","ipc","async"};

static uint64_t GpubNowUs(void)
{
	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC,&now);
	return((uint64_t)now.tv_sec * 1000000u + (uint64_t)now.tv_nsec / 1000u);
}

static unsigned long long GpubProcessUsed(void)
{
	GpubNvmlProcess processes[64];
	unsigned int count = 64u,index;
	pid_t self = getpid();
	if ( gpub_nvml_processes == 0 || gpub_nvml_processes(gpub_nvml_device,&count,processes) != 0 )
		return(0ull);
	for (index=0u; index<count; index++)
		if ( processes[index].pid == (unsigned int)self )
			return(processes[index].used);
	return(0ull);
}

static uint64_t GpubRssBytes(void)
{
	FILE *file = fopen("/proc/self/statm","r");
	unsigned long pages = 0ul,resident = 0ul;
	if ( file == 0 )
		return(0u);
	if ( fscanf(file,"%lu %lu",&pages,&resident) != 2 )
		resident = 0ul;
	fclose(file);
	return((uint64_t)resident * (uint64_t)sysconf(_SC_PAGESIZE));
}

static void GpubNvmlOpen(void)
{
	void *library = dlopen("libnvidia-ml.so.1",RTLD_NOW | RTLD_LOCAL);
	GpubNvmlInit init;
	GpubNvmlHandle handle;
	if ( library == 0 )
		return;
	init = (GpubNvmlInit)dlsym(library,"nvmlInit_v2");
	handle = (GpubNvmlHandle)dlsym(library,"nvmlDeviceGetHandleByIndex_v2");
	gpub_nvml_processes = (GpubNvmlProcesses)dlsym(library,"nvmlDeviceGetComputeRunningProcesses_v3");
	if ( init == 0 || handle == 0 || gpub_nvml_processes == 0 || init() != 0 || handle(0u,&gpub_nvml_device) != 0 )
		gpub_nvml_processes = 0;
}

static GpubLive *GpubSlot(uint64_t key,uint32_t insert)
{
	uint64_t index = (key * 0x9E3779B97F4A7C15ull) >> 44;
	uint32_t probe;
	for (probe=0u; probe<GPUB_LIVE_CAPACITY; probe++)
	{
		GpubLive *slot = &gpub_live[(index + probe) & (GPUB_LIVE_CAPACITY - 1u)];
		if ( slot->key == key )
			return(slot);
		if ( slot->key == 0u )
			return(insert != 0u ? slot : 0);
	}
	return(0);
}

static void GpubStack(void)
{
	void *frames[GPUB_STACK_DEPTH];
	Dl_info info;
	int count,index;
	count = backtrace(frames,GPUB_STACK_DEPTH);
	fputs(" stack=",gpub_out);
	for (index=2; index<count; index++)
	{
		if ( dladdr(frames[index],&info) != 0 && info.dli_fname != 0 )
			fprintf(gpub_out,"%s%s+0x%lx",index > 2 ? "," : "",info.dli_fname,(unsigned long)((uintptr_t)frames[index] - (uintptr_t)info.dli_fbase));
		else
			fprintf(gpub_out,"%s?+0x%lx",index > 2 ? "," : "",(unsigned long)(uintptr_t)frames[index]);
	}
}

static void GpubRecord(const char *api,uint32_t kind,uint64_t bytes,uint64_t key,int stack)
{
	GpubLive *slot;
	if ( kind < 8u && key != 0u )
	{
		slot = GpubSlot(key,1u);
		if ( slot != 0 )
		{
			slot->key = key;
			slot->bytes = bytes;
			slot->kind = kind;
		}
		gpub_live_bytes[kind] += bytes;
	}
	fprintf(gpub_out,"GPUB seq=%llu t_us=%llu api=%s kind=%s bytes=%llu key=0x%llx nvml=%llu rss=%llu",
		(unsigned long long)++gpub_seq,(unsigned long long)GpubNowUs(),api,kind < 8u ? gpub_kind_names[kind] : "event",
		(unsigned long long)bytes,(unsigned long long)key,GpubProcessUsed(),(unsigned long long)GpubRssBytes());
	if ( stack != 0 )
		GpubStack();
	fputc('\n',gpub_out);
}

static void GpubRelease(const char *api,uint64_t key)
{
	GpubLive *slot = GpubSlot(key,0u);
	if ( slot == 0 )
		return;
	gpub_live_bytes[slot->kind] -= slot->bytes;
	fprintf(gpub_out,"GPUB seq=%llu t_us=%llu api=%s kind=%s bytes=%llu key=0x%llx release=1\n",
		(unsigned long long)++gpub_seq,(unsigned long long)GpubNowUs(),api,gpub_kind_names[slot->kind],(unsigned long long)slot->bytes,(unsigned long long)key);
	slot->key = UINT64_MAX;
	slot->bytes = 0u;
}

static uint32_t GpubFirstLaunch(const void *function)
{
	uint64_t index = ((uint64_t)(uintptr_t)function * 0x9E3779B97F4A7C15ull) >> 48;
	uint32_t probe;
	for (probe=0u; probe<65536u; probe++)
	{
		void **slot = &gpub_seen_functions[(index + probe) & 65535u];
		if ( *slot == function )
			return(0u);
		if ( *slot == 0 )
		{
			*slot = (void *)function;
			return(1u);
		}
	}
	return(0u);
}

static void GpubEnter(CUpti_CallbackId id,const void *params)
{
	if ( id == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel || id == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx )
	{
		gpub_first_launch = GpubFirstLaunch(id == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel ? (const void *)((const cuLaunchKernel_params *)params)->f : (const void *)((const cuLaunchKernelEx_params *)params)->f);
		if ( gpub_first_launch == 0u )
			return;
	}
	gpub_enter_used = GpubProcessUsed();
	gpub_enter_seq = gpub_seq;
	if ( id == CUPTI_DRIVER_TRACE_CBID_cuGraphInstantiateWithFlags )
		gpub_instantiate_graph = ((const cuGraphInstantiateWithFlags_params *)params)->hGraph;
	else if ( id == CUPTI_DRIVER_TRACE_CBID_cuGraphInstantiateWithParams || id == CUPTI_DRIVER_TRACE_CBID_cuGraphInstantiateWithParams_ptsz )
		gpub_instantiate_graph = ((const cuGraphInstantiateWithParams_params *)params)->hGraph;
	else if ( id == CUPTI_DRIVER_TRACE_CBID_cuGraphInstantiate_v2 )
		gpub_instantiate_graph = ((const cuGraphInstantiate_v2_params *)params)->hGraph;
}

static void GpubDelta(const char *api,uint64_t key,uint64_t extra)
{
	unsigned long long after = GpubProcessUsed();
	fprintf(gpub_out,"GPUB seq=%llu t_us=%llu api=%s kind=delta bytes=%lld key=0x%llx nvml=%llu nodes=%llu rss=%llu",
		(unsigned long long)++gpub_seq,(unsigned long long)GpubNowUs(),api,(long long)after - (long long)gpub_enter_used,
		(unsigned long long)key,after,(unsigned long long)extra,(unsigned long long)GpubRssBytes());
	GpubStack();
	fputc('\n',gpub_out);
}

static void CUPTIAPI GpubCallback(void *userdata,CUpti_CallbackDomain domain,CUpti_CallbackId id,const void *data)
{
	const CUpti_CallbackData *callback = (const CUpti_CallbackData *)data;
	const void *params = callback->functionParams;
	CUresult result;
	size_t nodes = 0u;
	(void)userdata;
	if ( domain != CUPTI_CB_DOMAIN_DRIVER_API )
		return;
	pthread_mutex_lock(&gpub_lock);
	if ( callback->callbackSite == CUPTI_API_ENTER )
	{
		switch ( id )
		{
		case CUPTI_DRIVER_TRACE_CBID_cuMemFree_v2:
			GpubRelease(callback->functionName,(uint64_t)((const cuMemFree_v2_params *)params)->dptr);
			break;
		case CUPTI_DRIVER_TRACE_CBID_cuMemFreeAsync:
		case CUPTI_DRIVER_TRACE_CBID_cuMemFreeAsync_ptsz:
			GpubRelease(callback->functionName,(uint64_t)((const cuMemFreeAsync_params *)params)->dptr);
			break;
		case CUPTI_DRIVER_TRACE_CBID_cuMemFreeHost:
			GpubRelease(callback->functionName,(uint64_t)(uintptr_t)((const cuMemFreeHost_params *)params)->p);
			break;
		case CUPTI_DRIVER_TRACE_CBID_cuMemHostUnregister:
			GpubRelease(callback->functionName,(uint64_t)(uintptr_t)((const cuMemHostUnregister_params *)params)->p);
			break;
		case CUPTI_DRIVER_TRACE_CBID_cuMemRelease:
			GpubRelease(callback->functionName,(uint64_t)((const cuMemRelease_params *)params)->handle);
			break;
		default:
			GpubEnter(id,params);
			break;
		}
		pthread_mutex_unlock(&gpub_lock);
		return;
	}
	result = callback->functionReturnValue != 0 ? *(const CUresult *)callback->functionReturnValue : CUDA_SUCCESS;
	if ( result != CUDA_SUCCESS )
	{
		fprintf(gpub_out,"GPUB seq=%llu api=%s kind=error result=%d\n",(unsigned long long)++gpub_seq,callback->functionName,(int)result);
		pthread_mutex_unlock(&gpub_lock);
		return;
	}
	switch ( id )
	{
	case CUPTI_DRIVER_TRACE_CBID_cuMemAlloc_v2:
		GpubRecord(callback->functionName,0u,((const cuMemAlloc_v2_params *)params)->bytesize,(uint64_t)*((const cuMemAlloc_v2_params *)params)->dptr,1);
		break;
	case CUPTI_DRIVER_TRACE_CBID_cuMemAllocPitch_v2:
		GpubRecord(callback->functionName,0u,(uint64_t)*((const cuMemAllocPitch_v2_params *)params)->pPitch * ((const cuMemAllocPitch_v2_params *)params)->Height,(uint64_t)*((const cuMemAllocPitch_v2_params *)params)->dptr,1);
		break;
	case CUPTI_DRIVER_TRACE_CBID_cuMemAllocManaged:
		GpubRecord(callback->functionName,3u,((const cuMemAllocManaged_params *)params)->bytesize,(uint64_t)*((const cuMemAllocManaged_params *)params)->dptr,1);
		break;
	case CUPTI_DRIVER_TRACE_CBID_cuMemAllocAsync:
	case CUPTI_DRIVER_TRACE_CBID_cuMemAllocAsync_ptsz:
		GpubRecord(callback->functionName,7u,((const cuMemAllocAsync_params *)params)->bytesize,(uint64_t)*((const cuMemAllocAsync_params *)params)->dptr,1);
		break;
	case CUPTI_DRIVER_TRACE_CBID_cuMemAllocFromPoolAsync:
	case CUPTI_DRIVER_TRACE_CBID_cuMemAllocFromPoolAsync_ptsz:
		GpubRecord(callback->functionName,7u,((const cuMemAllocFromPoolAsync_params *)params)->bytesize,(uint64_t)*((const cuMemAllocFromPoolAsync_params *)params)->dptr,1);
		break;
	case CUPTI_DRIVER_TRACE_CBID_cuMemAllocHost_v2:
		GpubRecord(callback->functionName,1u,((const cuMemAllocHost_v2_params *)params)->bytesize,(uint64_t)(uintptr_t)*((const cuMemAllocHost_v2_params *)params)->pp,1);
		break;
	case CUPTI_DRIVER_TRACE_CBID_cuMemHostAlloc:
		GpubRecord(callback->functionName,1u,((const cuMemHostAlloc_params *)params)->bytesize,(uint64_t)(uintptr_t)*((const cuMemHostAlloc_params *)params)->pp,1);
		break;
	case CUPTI_DRIVER_TRACE_CBID_cuMemHostRegister_v2:
		GpubRecord(callback->functionName,2u,((const cuMemHostRegister_v2_params *)params)->bytesize,(uint64_t)(uintptr_t)((const cuMemHostRegister_v2_params *)params)->p,1);
		break;
	case CUPTI_DRIVER_TRACE_CBID_cuMemCreate:
		GpubRecord(callback->functionName,4u,((const cuMemCreate_params *)params)->size,(uint64_t)*((const cuMemCreate_params *)params)->handle,1);
		break;
	case CUPTI_DRIVER_TRACE_CBID_cuMemImportFromShareableHandle:
		GpubRecord(callback->functionName,5u,0u,(uint64_t)*((const cuMemImportFromShareableHandle_params *)params)->handle,1);
		break;
	case CUPTI_DRIVER_TRACE_CBID_cuMemMap:
		GpubRecord(callback->functionName,UINT32_MAX,((const cuMemMap_params *)params)->size,(uint64_t)((const cuMemMap_params *)params)->handle,0);
		break;
	case CUPTI_DRIVER_TRACE_CBID_cuIpcOpenMemHandle_v2:
		GpubRecord(callback->functionName,6u,0u,(uint64_t)*((const cuIpcOpenMemHandle_v2_params *)params)->pdptr,1);
		break;
	case CUPTI_DRIVER_TRACE_CBID_cuGraphInstantiateWithFlags:
	case CUPTI_DRIVER_TRACE_CBID_cuGraphInstantiateWithParams:
	case CUPTI_DRIVER_TRACE_CBID_cuGraphInstantiateWithParams_ptsz:
	case CUPTI_DRIVER_TRACE_CBID_cuGraphInstantiate_v2:
		if ( gpub_instantiate_graph != 0 )
			(void)cuGraphGetNodes(gpub_instantiate_graph,0,&nodes);
		GpubDelta(callback->functionName,(uint64_t)(uintptr_t)gpub_instantiate_graph,(uint64_t)nodes);
		gpub_instantiate_graph = 0;
		break;
	case CUPTI_DRIVER_TRACE_CBID_cuGraphUpload:
	case CUPTI_DRIVER_TRACE_CBID_cuGraphUpload_ptsz:
		GpubDelta(callback->functionName,(uint64_t)(uintptr_t)((const cuGraphUpload_params *)params)->hGraph,0u);
		break;
	case CUPTI_DRIVER_TRACE_CBID_cuGraphExecDestroy:
		GpubDelta(callback->functionName,(uint64_t)(uintptr_t)((const cuGraphExecDestroy_params *)params)->hGraphExec,0u);
		break;
	case CUPTI_DRIVER_TRACE_CBID_cuStreamEndCapture:
	case CUPTI_DRIVER_TRACE_CBID_cuStreamEndCapture_ptsz:
		GpubDelta(callback->functionName,0u,0u);
		break;
	case CUPTI_DRIVER_TRACE_CBID_cuDevicePrimaryCtxRetain:
	case CUPTI_DRIVER_TRACE_CBID_cuCtxCreate_v2:
	case CUPTI_DRIVER_TRACE_CBID_cuCtxCreate_v3:
	case CUPTI_DRIVER_TRACE_CBID_cuCtxCreate_v4:
	case CUPTI_DRIVER_TRACE_CBID_cuModuleLoadData:
	case CUPTI_DRIVER_TRACE_CBID_cuModuleLoadDataEx:
	case CUPTI_DRIVER_TRACE_CBID_cuModuleLoadFatBinary:
	case CUPTI_DRIVER_TRACE_CBID_cuLibraryLoadData:
	case CUPTI_DRIVER_TRACE_CBID_cuLibraryLoadFromFile:
	case CUPTI_DRIVER_TRACE_CBID_cuLibraryGetKernel:
	case CUPTI_DRIVER_TRACE_CBID_cuModuleGetFunction:
	case CUPTI_DRIVER_TRACE_CBID_cuMemPoolCreate:
	case CUPTI_DRIVER_TRACE_CBID_cuStreamCreate:
	case CUPTI_DRIVER_TRACE_CBID_cuStreamCreateWithPriority:
		if ( GpubProcessUsed() != gpub_enter_used )
			GpubDelta(callback->functionName,0u,0u);
		break;
	case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel:
	case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx:
		if ( gpub_first_launch != 0u && GpubProcessUsed() != gpub_enter_used )
			GpubDelta(callback->functionName,0u,0u);
		break;
	default:
		break;
	}
	fflush(gpub_out);
	pthread_mutex_unlock(&gpub_lock);
}

static void *GpubSampler(void *argument)
{
	uint32_t index;
	(void)argument;
	for (;;)
	{
		usleep(gpub_sample_ms * 1000u);
		pthread_mutex_lock(&gpub_lock);
		fprintf(gpub_out,"GPUB-SAMPLE t_us=%llu nvml=%llu rss=%llu",(unsigned long long)GpubNowUs(),GpubProcessUsed(),(unsigned long long)GpubRssBytes());
		for (index=0u; index<8u; index++)
			fprintf(gpub_out," live_%s=%llu",gpub_kind_names[index],(unsigned long long)gpub_live_bytes[index]);
		fputc('\n',gpub_out);
		fflush(gpub_out);
		pthread_mutex_unlock(&gpub_lock);
	}
	return(0);
}

static void GpubEnable(CUpti_SubscriberHandle subscriber,CUpti_CallbackId id)
{
	(void)cuptiEnableCallback(1u,subscriber,CUPTI_CB_DOMAIN_DRIVER_API,id);
}

int InitializeInjection(void)
{
	static const CUpti_CallbackId ids[] = {
		CUPTI_DRIVER_TRACE_CBID_cuMemAlloc_v2,CUPTI_DRIVER_TRACE_CBID_cuMemAllocPitch_v2,CUPTI_DRIVER_TRACE_CBID_cuMemAllocManaged,
		CUPTI_DRIVER_TRACE_CBID_cuMemAllocAsync,CUPTI_DRIVER_TRACE_CBID_cuMemAllocAsync_ptsz,
		CUPTI_DRIVER_TRACE_CBID_cuMemAllocFromPoolAsync,CUPTI_DRIVER_TRACE_CBID_cuMemAllocFromPoolAsync_ptsz,
		CUPTI_DRIVER_TRACE_CBID_cuMemAllocHost_v2,CUPTI_DRIVER_TRACE_CBID_cuMemHostAlloc,CUPTI_DRIVER_TRACE_CBID_cuMemHostRegister_v2,
		CUPTI_DRIVER_TRACE_CBID_cuMemCreate,CUPTI_DRIVER_TRACE_CBID_cuMemImportFromShareableHandle,CUPTI_DRIVER_TRACE_CBID_cuMemMap,
		CUPTI_DRIVER_TRACE_CBID_cuIpcOpenMemHandle_v2,
		CUPTI_DRIVER_TRACE_CBID_cuMemFree_v2,CUPTI_DRIVER_TRACE_CBID_cuMemFreeAsync,CUPTI_DRIVER_TRACE_CBID_cuMemFreeAsync_ptsz,
		CUPTI_DRIVER_TRACE_CBID_cuMemFreeHost,CUPTI_DRIVER_TRACE_CBID_cuMemHostUnregister,CUPTI_DRIVER_TRACE_CBID_cuMemRelease,
		CUPTI_DRIVER_TRACE_CBID_cuGraphInstantiateWithFlags,CUPTI_DRIVER_TRACE_CBID_cuGraphInstantiateWithParams,
		CUPTI_DRIVER_TRACE_CBID_cuGraphInstantiateWithParams_ptsz,CUPTI_DRIVER_TRACE_CBID_cuGraphInstantiate_v2,
		CUPTI_DRIVER_TRACE_CBID_cuGraphUpload,CUPTI_DRIVER_TRACE_CBID_cuGraphUpload_ptsz,CUPTI_DRIVER_TRACE_CBID_cuGraphExecDestroy,
		CUPTI_DRIVER_TRACE_CBID_cuStreamEndCapture,CUPTI_DRIVER_TRACE_CBID_cuStreamEndCapture_ptsz,
		CUPTI_DRIVER_TRACE_CBID_cuDevicePrimaryCtxRetain,CUPTI_DRIVER_TRACE_CBID_cuCtxCreate_v2,CUPTI_DRIVER_TRACE_CBID_cuCtxCreate_v3,
		CUPTI_DRIVER_TRACE_CBID_cuCtxCreate_v4,CUPTI_DRIVER_TRACE_CBID_cuModuleLoadData,CUPTI_DRIVER_TRACE_CBID_cuModuleLoadDataEx,
		CUPTI_DRIVER_TRACE_CBID_cuModuleLoadFatBinary,CUPTI_DRIVER_TRACE_CBID_cuLibraryLoadData,CUPTI_DRIVER_TRACE_CBID_cuLibraryLoadFromFile,
		CUPTI_DRIVER_TRACE_CBID_cuLibraryGetKernel,CUPTI_DRIVER_TRACE_CBID_cuModuleGetFunction,CUPTI_DRIVER_TRACE_CBID_cuMemPoolCreate,
		CUPTI_DRIVER_TRACE_CBID_cuStreamCreate,CUPTI_DRIVER_TRACE_CBID_cuStreamCreateWithPriority,
		CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel,CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx};
	CUpti_SubscriberHandle subscriber;
	const char *path = getenv("SPARK_GPU_BYTES_TRACE_PATH");
	const char *sample = getenv("SPARK_GPU_BYTES_TRACE_SAMPLE_MS");
	pthread_t thread;
	uint32_t index;
	char name[512];
	if ( path == 0 || path[0] == '\0' )
		return(0);
	snprintf(name,sizeof(name),"%s.%d",path,(int)getpid());
	gpub_out = fopen(name,"w");
	gpub_live = (GpubLive *)calloc(GPUB_LIVE_CAPACITY,sizeof(*gpub_live));
	if ( gpub_out == 0 || gpub_live == 0 )
		return(0);
	GpubNvmlOpen();
	fprintf(gpub_out,"GPUB-START pid=%d nvml=%s t_us=%llu rss=%llu\n",(int)getpid(),gpub_nvml_processes != 0 ? "ok" : "missing",(unsigned long long)GpubNowUs(),(unsigned long long)GpubRssBytes());
	if ( cuptiSubscribe(&subscriber,(CUpti_CallbackFunc)GpubCallback,0) != CUPTI_SUCCESS )
	{
		fprintf(gpub_out,"GPUB-ERROR cuptiSubscribe\n");
		fflush(gpub_out);
		return(0);
	}
	for (index=0u; index<sizeof(ids) / sizeof(ids[0]); index++)
		GpubEnable(subscriber,ids[index]);
	fflush(gpub_out);
	gpub_sample_ms = sample != 0 ? (uint32_t)strtoul(sample,0,10) : 0u;
	if ( gpub_sample_ms != 0u )
		(void)pthread_create(&thread,0,GpubSampler,0);
	return(1);
}
