#pragma once
#include "tests/host_cuda/lm_host_cuda.cuh"
#include <pthread.h>

#undef __shared__
#define __shared__ static
#undef __syncthreads
#define __syncthreads() LmHostThreadsBlockBarrier()
#undef LM_LAUNCH
#define LM_HOST_THREADS_MAX 1024u

typedef struct LmHostThreads
{
	pthread_barrier_t block_barrier;
	pthread_barrier_t warp_barrier[LM_HOST_THREADS_MAX / 32u];
	float shuffle[LM_HOST_THREADS_MAX];
}
LmHostThreads;

static LmHostThreads lm_host_threads;
static thread_local LmHostDim3 lm_host_thread_index,lm_host_block_index;
#define threadIdx lm_host_thread_index
#define blockIdx lm_host_block_index

static inline void LmHostThreadsBlockBarrier(void)
{
	pthread_barrier_wait(&lm_host_threads.block_barrier);
}

static inline float LmHostThreadsShuffleXor(unsigned, float value, int mask, int = 32)
{
	unsigned thread = threadIdx.x,warp = thread / 32u;
	float result;
	lm_host_threads.shuffle[thread] = value;
	pthread_barrier_wait(&lm_host_threads.warp_barrier[warp]);
	result = lm_host_threads.shuffle[(thread & ~31u) | ((thread % 32u) ^ (unsigned)mask)];
	pthread_barrier_wait(&lm_host_threads.warp_barrier[warp]);
	return(result);
}

#define __shfl_xor_sync LmHostThreadsShuffleXor

template<class Body> struct LmHostThreadsTask
{
	Body *body;
	unsigned thread;
};

template<class Body> static void *LmHostThreadsMain(void *argument)
{
	LmHostThreadsTask<Body> *task = (LmHostThreadsTask<Body> *)argument;
	unsigned x,y,z;
	lm_host_thread_index = dim3(task->thread,0u,0u);
	for (z=0u; z<gridDim.z; z++)
		for (y=0u; y<gridDim.y; y++)
			for (x=0u; x<gridDim.x; x++)
			{
				lm_host_block_index = dim3(x,y,z);
				(*task->body)();
				pthread_barrier_wait(&lm_host_threads.block_barrier);
			}
	return(0);
}

template<class Body> static void LmHostThreadsLaunch(dim3 grid,unsigned threads,Body body)
{
	static pthread_t handle[LM_HOST_THREADS_MAX];
	static LmHostThreadsTask<Body> task[LM_HOST_THREADS_MAX];
	unsigned thread;
	gridDim = grid;
	blockDim = dim3(threads,1u,1u);
	pthread_barrier_init(&lm_host_threads.block_barrier,0,threads);
	for (thread=0u; thread<threads/32u; thread++)
		pthread_barrier_init(&lm_host_threads.warp_barrier[thread],0,32u);
	for (thread=0u; thread<threads; thread++)
	{
		task[thread].body = &body;
		task[thread].thread = thread;
		pthread_create(&handle[thread],0,LmHostThreadsMain<Body>,&task[thread]);
	}
	for (thread=0u; thread<threads; thread++)
		pthread_join(handle[thread],0);
	pthread_barrier_destroy(&lm_host_threads.block_barrier);
	for (thread=0u; thread<threads/32u; thread++)
		pthread_barrier_destroy(&lm_host_threads.warp_barrier[thread]);
}

#define LM_LAUNCH(kernel, grid, block, shared, stream, ...) LmHostThreadsLaunch((grid),(block),[&]() { LM_UNPAREN kernel(__VA_ARGS__); })
