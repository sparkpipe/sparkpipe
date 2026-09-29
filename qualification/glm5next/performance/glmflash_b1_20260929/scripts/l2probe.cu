#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
static __global__ void readcs(const uint4 *p, size_t n, unsigned *out){ size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; unsigned acc=0; for(;i<n;i+=(size_t)gridDim.x*blockDim.x){ uint4 v=__ldcs(p+i); acc^=v.x^v.y^v.z^v.w;} if(acc==0x12345678u) out[0]=acc; }
static __global__ void readld(const uint4 *p, size_t n, unsigned *out){ size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; unsigned acc=0; for(;i<n;i+=(size_t)gridDim.x*blockDim.x){ uint4 v=__ldg(p+i); acc^=v.x^v.y^v.z^v.w;} if(acc==0x12345678u) out[0]=acc; }
static __global__ void pf(const uint8_t *p, uint32_t bytes, uint32_t chunk){ uint32_t first=(blockIdx.x*blockDim.x+threadIdx.x)*chunk, stride=gridDim.x*blockDim.x*chunk; for(uint32_t o=first;o<bytes;o+=stride){ uint32_t b=bytes-o; b=b<chunk?b:chunk; asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(p+o), "r"(b) : "memory"); } }
static __global__ void pfl(const uint8_t *p, uint32_t bytes){ uint32_t i=(blockIdx.x*blockDim.x+threadIdx.x)*128u; for(;i<bytes;i+=gridDim.x*blockDim.x*128u) asm volatile("prefetch.global.L2 [%0];" :: "l"(p+i)); }
static __global__ void spin(uint64_t ns){ uint64_t s,n; asm volatile("mov.u64 %0, %%globaltimer;":"=l"(s)); do asm volatile("mov.u64 %0, %%globaltimer;":"=l"(n)); while(n-s<ns); }
int main(){
  size_t W=8u<<20, F=256u<<20; uint8_t *w,*f; unsigned *o; cudaMalloc(&w,W); cudaMalloc(&f,F); cudaMalloc(&o,4); cudaMemset(w,1,W); cudaMemset(f,2,F);
  cudaStream_t s; cudaStreamCreate(&s); cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b); float ms;
  int grid=48*8, blk=256;
  for(int mode=0;mode<6;mode++){ float tot=0; float best=1e9;
    for(int it=0;it<20;it++){
      readcs<<<grid,blk,0,s>>>((const uint4*)f,F/16,o);
      if(mode==1){ pf<<<8,32,0,s>>>(w,(uint32_t)W,32768u); spin<<<1,32,0,s>>>(60000); }
      if(mode==2){ spin<<<1,32,0,s>>>(60000); }
      if(mode==3){ readld<<<grid,blk,0,s>>>((const uint4*)w,W/16,o); }
      if(mode==4){ pfl<<<48,256,0,s>>>(w,(uint32_t)W); spin<<<1,32,0,s>>>(60000); }
      if(mode==5){ pf<<<48,32,0,s>>>(w,(uint32_t)W,8192u); spin<<<1,32,0,s>>>(60000); }
      cudaEventRecord(a,s); readcs<<<grid,blk,0,s>>>((const uint4*)w,W/16,o); cudaEventRecord(b,s); cudaEventSynchronize(b); cudaEventElapsedTime(&ms,a,b); tot+=ms; if(ms<best)best=ms; }
    printf("mode=%d %s mean_us=%.1f best_us=%.1f GBps_best=%.0f\n",mode, mode==0?"cold":mode==1?"bulkpf32k+spin":mode==2?"spin-only":mode==3?"warm-by-ld":mode==4?"prefetch.L2-per-line+spin":"bulkpf8k-48cta+spin", tot/20*1000, best*1000, W/(best*1e-3)/1e9);
  }
  printf("err=%s\n",cudaGetErrorString(cudaGetLastError()));
  return 0; }
