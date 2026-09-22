#include "sparkpipe/spark_stage_module_common.h"
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
static void CUDART_CB delay(void *arg)
{
    (void)arg;
    usleep(100);
}
int main(int argc,char **argv)
{
    SparkStageModuleCudaWait wait;
    cudaStream_t stream;
    unsigned int i,busy=0u;
    if (argc != 2 || strcmp(argv[1],"--run") != 0)
    {
        fprintf(stderr,"usage: test_cuda_stream_receipt --run\n");
        return 2;
    }
    alarm(60);
    memset(&wait,0,sizeof(wait));
    if (cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking)!=cudaSuccess ||
        SparkStageModuleCudaWaitInitialize(&wait,stream)!=SPARK_STATUS_OK) return 1;
    for(i=0u;i<50000u;i++)
    {
        cudaError_t query;
        if(cudaLaunchHostFunc(stream,delay,0)!=cudaSuccess ||
           SparkStageModuleCudaWaitFor(&wait,UINT64_C(1000000000))!=SPARK_STATUS_OK) return 2;
        query=cudaStreamQuery(stream);
        if(query==cudaErrorNotReady) busy++;
        else if(query!=cudaSuccess) return 3;
    }
    if(cudaStreamSynchronize(stream)!=cudaSuccess ||
       SparkStageModuleCudaWaitDestroy(&wait)!=SPARK_STATUS_OK ||
       cudaStreamDestroy(stream)!=cudaSuccess) return 4;
    printf("iterations=%u success_followed_by_not_ready=%u\n",i,busy);
    return busy != 0u ? 5 : 0;
}
