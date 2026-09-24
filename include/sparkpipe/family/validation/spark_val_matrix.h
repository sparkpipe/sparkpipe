#pragma once

static int SPARK_FAMILY(ValAllocMatrix)(SPARK_FAMILY(ValMatrix) *matrix,uint32_t rows,uint32_t columns,int mode,float scale)
{
	uint16_t *packed;
	uint64_t count = (uint64_t)rows * columns;
	matrix->rows = rows;
	matrix->columns = columns;
	matrix->host = (float *)malloc(count * sizeof(float));
	packed = (uint16_t *)malloc(count * sizeof(uint16_t));
	if (matrix->host == 0 || packed == 0)
		return(SPARK_FAMILY(ValFail)("fixture","host_alloc"));
	if (cudaMalloc((void **)&matrix->device,count * sizeof(uint16_t)) != cudaSuccess)
		return(SPARK_FAMILY(ValFail)("fixture","device_alloc"));
	SPARK_FAMILY(ValRandomState) += 101u;
	if (mode == 1)
		SPARK_FAMILY(ValFillNorm)(packed,matrix->host,count);
	else
		SPARK_FAMILY(ValFill)(packed,matrix->host,count,scale);
	if (cudaMemcpy(matrix->device,packed,count * sizeof(uint16_t),cudaMemcpyHostToDevice) != cudaSuccess)
		return(SPARK_FAMILY(ValFail)("fixture","weight_upload"));
	free(packed);
	return(0);
}

static void SPARK_FAMILY(ValFreeMatrix)(SPARK_FAMILY(ValMatrix) *matrix)
{
	free(matrix->host);
	cudaFree(matrix->device);
	memset(matrix,0,sizeof(*matrix));
}
