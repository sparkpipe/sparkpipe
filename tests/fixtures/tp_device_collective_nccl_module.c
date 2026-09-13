#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define TEST_NCCL_SUCCESS 0
#define TEST_NCCL_VERSION 23007
#define TEST_NCCL_UNIQUE_ID_BYTES 128u
#define TEST_NCCL_U64 5
#define TEST_NCCL_BF16 9
#define TEST_NCCL_SUM 0
#define TEST_NCCL_MAX 2

typedef struct TestNcclComm
{
	int32_t rank_count;
	int32_t rank;
} *ncclComm_t;

typedef struct TestNcclUniqueId
{
	uint8_t bytes[TEST_NCCL_UNIQUE_ID_BYTES];
} ncclUniqueId;

int32_t ncclGetVersion(int32_t *version)
{
	if ( version == 0 )
		return(4);
	*version = TEST_NCCL_VERSION;
	return(TEST_NCCL_SUCCESS);
}

int32_t ncclGetUniqueId(ncclUniqueId *unique_id)
{
	uint32_t index;
	if ( unique_id == 0 )
		return(4);
	for (index=0u; index<TEST_NCCL_UNIQUE_ID_BYTES; index++)
		unique_id->bytes[index] = (uint8_t)(index ^ 0x5au);
	return(TEST_NCCL_SUCCESS);
}

int32_t ncclCommInitRank(ncclComm_t *communicator,int32_t rank_count,
	ncclUniqueId unique_id,int32_t rank)
{
	ncclComm_t value;
	if ( communicator == 0 || rank_count <= 1 || rank < 0 ||
		rank >= rank_count || unique_id.bytes[0] != 0x5au )
		return(4);
	value = (ncclComm_t)calloc(1u,sizeof(*value));
	if ( value == 0 )
		return(2);
	value->rank_count = rank_count;
	value->rank = rank;
	*communicator = value;
	return(TEST_NCCL_SUCCESS);
}

int32_t ncclAllReduce(const void *send_device,void *receive_device,
	size_t element_count,int32_t data_type,int32_t operation,
	ncclComm_t communicator,void *cuda_stream)
{
	size_t element_bytes;
	if ( send_device == 0 || receive_device == 0 || element_count == 0u ||
		!((data_type == TEST_NCCL_BF16 && operation == TEST_NCCL_SUM) ||
		  (data_type == TEST_NCCL_U64 && operation == TEST_NCCL_MAX)) ||
		communicator == 0 || cuda_stream == 0 )
		return(4);
	element_bytes = data_type == TEST_NCCL_U64 ? 8u : 2u;
	if ( send_device != receive_device )
		memmove(receive_device,send_device,element_count * element_bytes);
	return(TEST_NCCL_SUCCESS);
}

int32_t ncclCommGetAsyncError(ncclComm_t communicator,int32_t *async_error)
{
	if ( communicator == 0 || async_error == 0 )
		return(4);
	*async_error = TEST_NCCL_SUCCESS;
	return(TEST_NCCL_SUCCESS);
}

int32_t ncclCommDestroy(ncclComm_t communicator)
{
	free(communicator);
	return(TEST_NCCL_SUCCESS);
}

int32_t ncclCommAbort(ncclComm_t communicator)
{
	free(communicator);
	return(TEST_NCCL_SUCCESS);
}

const char *ncclGetErrorString(int32_t result)
{
	return(result == TEST_NCCL_SUCCESS ? "success" : "fixture error");
}
