#pragma once

SparkStatus SPARK_FAMILY(ResidentDecodeStageInitialize)(
    const SparkFirmwareModuleConfiguration *configuration,
    const SparkFirmwareModuleHostServices *host_services,
    void **module_state)
{
	return(SparkStageModuleLifecycleInitialize(configuration,host_services,module_state,&SPARK_FAMILY(ModuleLifecycle)));
}

SparkStatus SPARK_FAMILY(ResidentDecodeStageAdmit)(
    void *module_state,
    const SparkModelDriverAdmissionRequest *request,
    SparkModelDriverAdmissionDecision *decision)
{
	return(SparkStageModuleLifecycleAdmit(module_state,request,decision,&SPARK_FAMILY(ModuleLifecycle)));
}

SparkStatus SPARK_FAMILY(ResidentDecodeStageExecute)(
    void *module_state,
    SparkModelDriverFrame *frame)
{
	return(SparkStageModuleLifecycleExecute(module_state,frame,&SPARK_FAMILY(ModuleLifecycle)));
}

void SPARK_FAMILY(ResidentDecodeStageDestroy)(void *module_state)
{
	SparkStageModuleLifecycleDestroy(module_state,&SPARK_FAMILY(ModuleLifecycle));
}

static SparkStatus SPARK_FAMILY(ModuleEmitHiddenOutput)(SPARK_FAMILY(ModuleSlot) *slot, SPARK_FAMILY(ResidentDecodeStageFrameContext) *context, uint32_t rows)
{
	SparkHiddenTransportPacket *packet = &context->hidden_output_packet;
	memset(packet,0,sizeof(*packet));
	packet->abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
	packet->descriptor_bytes = SPARK_HIDDEN_TRANSPORT_PACKET_BYTES;
	packet->flags = SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_BF16 | SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_DEVICE_POINTER;
	packet->active_sequence_count = rows;
	packet->hidden_dimension = SPARK_FAMILY_CONST(MODEL_HIDDEN_DIMENSION);
	packet->bytes_per_sequence = SPARK_FAMILY_CONST(MODEL_HIDDEN_BF16_BYTES);
	packet->hidden_bf16 = slot->hidden_bf16;
	packet->cuda_stream = slot->cuda_stream;
	return(context->hidden_output_send_function(context->hidden_output_transport_session,packet));
}
