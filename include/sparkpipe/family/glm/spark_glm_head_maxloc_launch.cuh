#pragma once

extern "C" cudaError_t SPARK_FAMILY(LaunchHeadMaxlocPack)(cudaStream_t stream,const float *scores,const uint32_t *token_ids,uint64_t *maxloc,uint32_t row_count,uint32_t rank_offset)
{
	if ( scores == 0 || token_ids == 0 || maxloc == 0 || row_count == 0u )
		return(cudaErrorInvalidValue);
	SPARK_FAMILY(HeadMaxlocPackKernel)<<<(row_count + 255u) / 256u,256u,0u,stream>>>(scores,token_ids,maxloc,row_count,rank_offset);
	return(cudaPeekAtLastError());
}

extern "C" cudaError_t SPARK_FAMILY(LaunchHeadMaxlocUnpack)(cudaStream_t stream,const uint64_t *maxloc,uint32_t *token_ids,uint32_t row_count)
{
	if ( maxloc == 0 || token_ids == 0 || row_count == 0u )
		return(cudaErrorInvalidValue);
	SPARK_FAMILY(HeadMaxlocUnpackKernel)<<<(row_count + 255u) / 256u,256u,0u,stream>>>(maxloc,token_ids,row_count);
	return(cudaPeekAtLastError());
}
