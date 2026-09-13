#include <cuda.h>
#include <string.h>
static uint64_t seen_dim[3], seen_stride[3]; static uint32_t seen_box[3], seen_rank;
static CUtensorMapSwizzle seen_swizzle;
CUresult cuTensorMapEncodeTiled(CUtensorMap*m,CUtensorMapDataType,uint32_t r,void*,
  const uint64_t*d,const uint64_t*s,const uint32_t*b,const uint32_t*,
  CUtensorMapInterleave,CUtensorMapSwizzle sw,CUtensorMapL2promotion,CUtensorMapFloatOOBfill){
  seen_rank=r; seen_swizzle=sw;
  for(uint32_t i=0;i<r&&i<3;i++){seen_dim[i]=d[i];seen_box[i]=b[i];}
  for(uint32_t i=0;i+1<r&&i<3;i++)seen_stride[i]=s[i];
  memset(m,1,sizeof(*m)); return CUDA_SUCCESS;}
uint32_t stub_rank(void){return seen_rank;}
uint64_t stub_dim(int i){return seen_dim[i];}
uint32_t stub_box(int i){return seen_box[i];}
uint64_t stub_stride(int i){return seen_stride[i];}
int stub_swizzle(void){return (int)seen_swizzle;}
