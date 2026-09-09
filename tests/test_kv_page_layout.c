#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_kv_page_store.h"

typedef struct TestCopyState
{
	uint32_t calls;
	uint32_t fail_at;
} TestCopyState;

static SparkStatus TestCopy(void *context,uint32_t direction,uintptr_t device,void *host,uint64_t bytes)
{
	TestCopyState *state;
	state = (TestCopyState *)context;
	state->calls++;
	if ( state->calls == state->fail_at )
		return(SPARK_STATUS_IO_ERROR);
	if ( direction == SPARK_KV_PAGE_STORE_COPY_DEVICE_TO_HOST )
		memcpy(host,(const void *)device,(size_t)bytes);
	else
		memcpy((void *)device,host,(size_t)bytes);
	return(SPARK_STATUS_OK);
}

static int32_t TestRoundTrip(void)
{
	uint8_t device[3u * 41u],original[sizeof(device)],packed[3u * 7u];
	SparkKvLayeredPageLayout layout = {(uintptr_t)device,sizeof(device),41u,7u,3u,5u};
	TestCopyState copy = {0u,0u};
	uint32_t page,layer,index;
	for (index=0u; index<sizeof(device); index++)
		device[index] = (uint8_t)index;
	memcpy(original,device,sizeof(device));
	for (page=0u; page<5u; page++)
	{
		if ( SparkKvPageStoreCopyLayered(&layout,SPARK_KV_PAGE_STORE_COPY_DEVICE_TO_HOST,page,packed,sizeof(packed),TestCopy,&copy) != SPARK_STATUS_OK )
			return(-1);
		for (layer=0u; layer<3u; layer++)
		{
			if ( memcmp(packed + layer * 7u,original + layer * 41u + page * 7u,7u) != 0 )
				return(-2);
			memset(device + layer * 41u + page * 7u,0xff,7u);
		}
		if ( SparkKvPageStoreCopyLayered(&layout,SPARK_KV_PAGE_STORE_COPY_HOST_TO_DEVICE,page,packed,sizeof(packed),TestCopy,&copy) != SPARK_STATUS_OK )
			return(-3);
		if ( memcmp(device,original,sizeof(device)) != 0 )
			return(-4);
	}
	return(copy.calls == 30u ? 0 : -5);
}

static int32_t TestInvalidLayouts(void)
{
	uint8_t device[48],packed[16];
	SparkKvLayeredPageLayout layout = {(uintptr_t)device,sizeof(device),24u,8u,2u,3u},bad;
	TestCopyState copy = {0u,0u};
	uint32_t index;
	for (index=0u; index<7u; index++)
	{
		bad = layout;
		if ( index == 0u ) bad.device_bytes--;
		if ( index == 1u ) bad.layer_stride_bytes = 23u;
		if ( index == 2u ) bad.layer_page_bytes = UINT64_MAX;
		if ( index == 3u ) bad.layer_stride_bytes = UINT64_MAX;
		if ( index == 4u ) bad.device_base = UINTPTR_MAX - 2u;
		if ( index == 5u ) bad.layer_count = 0u;
		if ( index == 6u ) bad.page_count = 0u;
		if ( SparkKvPageStoreCopyLayered(&bad,SPARK_KV_PAGE_STORE_COPY_DEVICE_TO_HOST,0u,packed,sizeof(packed),TestCopy,&copy) == SPARK_STATUS_OK || copy.calls != 0u )
			return(-1);
	}
	if ( SparkKvPageStoreCopyLayered(&layout,SPARK_KV_PAGE_STORE_COPY_DEVICE_TO_HOST,3u,packed,sizeof(packed),TestCopy,&copy) == SPARK_STATUS_OK || copy.calls != 0u )
		return(-2);
	if ( SparkKvPageStoreCopyLayered(&layout,0u,0u,packed,sizeof(packed),TestCopy,&copy) == SPARK_STATUS_OK || copy.calls != 0u )
		return(-3);
	copy.fail_at = 1u;
	if ( SparkKvPageStoreCopyLayered(&layout,SPARK_KV_PAGE_STORE_COPY_DEVICE_TO_HOST,0u,packed,sizeof(packed),TestCopy,&copy) != SPARK_STATUS_IO_ERROR || copy.calls != 1u )
		return(-4);
	return(0);
}

int main(void)
{
	return(TestRoundTrip() != 0 || TestInvalidLayouts() != 0 ? 1 : 0);
}
