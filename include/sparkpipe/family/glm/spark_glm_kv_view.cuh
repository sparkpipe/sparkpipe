#pragma once

static void SPARK_FAMILY(BuildKvView)(
	LmKvView *view,
	uint8_t *pool,
	const SPARK_FAMILY(CudaWave) *wave)
{
	view->pool = pool;
	view->page_table = wave->page_table;
	view->page_table_stride = wave->pages_per_sequence;
	view->sequence_count = wave->resident_sequence_capacity;
	view->pool_page_count = wave->resident_sequence_capacity * wave->pages_per_sequence;
	view->access_error = (LmKvAccessError *)wave->slot->kv_access_error;
}
