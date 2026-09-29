#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>
static double now(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }
int main(){
  uint64_t *d, *h, x; cudaMalloc(&d,144); cudaMemset(d,0,144); cudaHostAlloc((void**)&h,144,0);
  cudaStream_t s; cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking); cudaDeviceSynchronize();
  int n=2000; double t0,t1;
  for(int w=0;w<100;w++) cudaMemcpy(&x,d,8,cudaMemcpyDeviceToHost);
  t0=now(); for(int i=0;i<n;i++) cudaMemcpy(&x,d,8,cudaMemcpyDeviceToHost); t1=now();
  printf("cudaMemcpy 8B D2H pageable: %.2f us each\n",(t1-t0)/n);
  t0=now(); for(int i=0;i<n;i++){ cudaMemcpyAsync(h,d,144,cudaMemcpyDeviceToHost,s); cudaStreamSynchronize(s);} t1=now();
  printf("cudaMemcpyAsync 144B D2H pinned + stream sync: %.2f us each\n",(t1-t0)/n);
  t0=now(); for(int i=0;i<n;i++){ cudaMemcpy(&x,d,8,cudaMemcpyDeviceToHost); cudaMemcpy(&x,d+3,8,cudaMemcpyDeviceToHost); cudaMemcpy(&x,d+8,8,cudaMemcpyDeviceToHost);} t1=now();
  printf("3x cudaMemcpy 8B (disarm+error per band): %.2f us\n",(t1-t0)/n);
  return 0; }
