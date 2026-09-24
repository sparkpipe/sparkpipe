#pragma once

static SparkStatus SPARK_FAMILY(ValidateFrameBuffers)(
	const SPARK_FAMILY(ModuleState) *state,
	const SparkModelDriverFrame *frame,
	uint32_t row_count)
{
	const SparkModelDriverBuffer *buffer;
	if ( state->owns_final_head == 0u )
		return(frame->buffer_count == 0u ? SPARK_STATUS_OK : SPARK_STATUS_INVALID_ARGUMENT);
	if ( frame->buffer_count != 1u || frame->buffers == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	buffer = &frame->buffers[0];
	if ( buffer->flags != SPARK_MODEL_DRIVER_BUFFER_FLAG_WRITE || buffer->address == 0 || buffer->bytes < (uint64_t)row_count * sizeof(uint32_t) )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	return(SPARK_STATUS_OK);
}
