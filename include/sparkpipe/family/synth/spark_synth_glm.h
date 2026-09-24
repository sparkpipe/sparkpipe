#pragma once

static int32_t SPARK_FAMILY(SynthesizeAppend)(SPARK_FAMILY(SynthesizeContext) *context, uint32_t tensor_kind, uint32_t layer_index)
{
	SPARK_FAMILY(StagePackTensorShape) shape;
	SPARK_FAMILY(StagePackEntry) *entry;
	if ( context->entry_count >= SPARK_FAMILY_CONST(SYNTHESIZE_MAX_TENSORS) )
		return(-1);
	if ( SPARK_FAMILY(StagePackExpectedShape)(tensor_kind,layer_index,context->expert_codec,context->tp_degree,&shape) != 0 )
		return(-2);
	entry = &context->entries[context->entry_count];
	memset(entry,0,sizeof(*entry));
	entry->tensor_kind = tensor_kind;
	entry->layer_index = layer_index;
	entry->payload_type = shape.payload_type;
	entry->weight_codec = shape.weight_codec;
	entry->scale_encoding = shape.scale_encoding;
	entry->group_count = shape.group_count;
	entry->rows = shape.rows;
	entry->columns = shape.columns;
	entry->payload_bytes = SPARK_FAMILY(StagePackExpectedPayloadBytes)(&shape);
	entry->scale_bytes = SPARK_FAMILY(StagePackExpectedScaleBytes)(&shape);
	entry->payload_offset = SparkSynthAlign(context->payload_cursor);
	entry->scale_offset = entry->scale_bytes != 0u ? SparkSynthAlign(entry->payload_offset + entry->payload_bytes) : 0u;
	context->payload_cursor = entry->scale_bytes != 0u ? (entry->scale_offset + entry->scale_bytes) : (entry->payload_offset + entry->payload_bytes);
	if ( entry->payload_bytes == 0u )
		return(-3);
	context->entry_count++;
	return(0);
}

static void SPARK_FAMILY(SynthesizeHexParse)(const char *text, uint8_t *out, uint32_t bytes)
{
	uint32_t index;
	for (index = 0; index < bytes; index++)
		out[index] = 0u;
	if ( text == 0 )
		return;
	for (index = 0; index < bytes * 2u; index++)
	{
		char c = text[index];
		uint8_t value;
		if ( c >= '0' && c <= '9' )
			value = (uint8_t)(c - '0');
		else if ( c >= 'a' && c <= 'f' )
			value = (uint8_t)(c - 'a' + 10);
		else if ( c >= 'A' && c <= 'F' )
			value = (uint8_t)(c - 'A' + 10);
		else
			break;
		out[index / 2u] = (uint8_t)((out[index / 2u] << 4) | value);
	}
}
