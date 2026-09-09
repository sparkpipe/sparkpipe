#include "tests/host_cuda/lm_host_cuda.cuh"
#include "inference/kernels/tp_reduce.cuh"
#include <stdio.h>

LmHostDim3 blockIdx,threadIdx,blockDim,gridDim;

static int32_t run(uint32_t degree,uint32_t rows,uint32_t local)
{
	uint16_t contributions[16][119],destination[119],expected;
	LmTpBf16Contributions<16> inputs = {};
	uint32_t rank,element,row,thread;
	float value;
	double total;
	for (rank=0u; rank<degree; rank++)
	{
		for (element=0u; element<(rows * 7u); element++)
		{
			value = rank == 0u ? 256.0f : (rank == 1u ? 1.0f : (rank == 2u ? -256.0f : 0.0f));
			contributions[rank][element] = LmFloatToBf16(value);
		}
		inputs.rank[rank] = contributions[rank];
	}
	memcpy(destination,contributions[local],rows * 7u * sizeof(uint16_t));
	inputs.rank[local] = destination;
	blockDim.x = 4u;
	for (row=0u; row<rows; row++)
		for (thread=0u; thread<4u; thread++)
		{
			blockIdx.x = row;
			threadIdx.x = thread;
			LmTpBf16SumKernel(destination,inputs,rows,7u);
		}
	for (element=0u; element<(rows * 7u); element++)
	{
		total = 0.0;
		for (rank=0u; rank<degree; rank++)
			total += LmBf16ToFloat(contributions[rank][element]);
		expected = LmFloatToBf16((float)total);
		if ( destination[element] != expected )
			return(-1);
	}
	return(0);
}

int main(void)
{
	uint32_t degrees[3] = {1u,4u,16u},rows[3] = {1u,3u,17u},i,j,rank;
	for (i=0u; i<3u; i++)
		for (j=0u; j<3u; j++)
			for (rank=0u; rank<degrees[i]; rank++)
				if ( run(degrees[i],rows[j],rank) != 0 )
					return(1);
	puts("PASS fixed-rank BF16 sum, cancellation, local aliasing and arbitrary rows");
	return(0);
}
