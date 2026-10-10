#pragma once

#define TEST_K3_RECURRENT_BYTES 48u
#define TEST_K3_STUB_SLOTS 8u

static uint32_t test_dispatch_rows;
static uint32_t test_dispatch_sequences;
static uint32_t test_dispatch_calls;
static SparkStatus test_dispatch_status = SPARK_STATUS_OK;
static uint32_t test_reset_slots[TEST_K3_STUB_SLOTS];
static uint32_t test_reset_count;
static uint32_t test_reset_calls;
static SparkStatus test_reset_status = SPARK_STATUS_OK;
static SparkStageRunnerKv test_attached_kv;
static uint32_t test_attach_calls;
static uint32_t test_kv_layers = 2u;
static uint64_t test_recurrent_bytes = TEST_K3_RECURRENT_BYTES;
static uint8_t test_recurrent_lanes[TEST_K3_STUB_SLOTS][TEST_K3_RECURRENT_BYTES];
static uint32_t test_recurrent_to_buffer;
static uint32_t test_recurrent_from_buffer;
static SparkStatus test_recurrent_status = SPARK_STATUS_OK;
static SparkStatus test_pack_status = SPARK_STATUS_OK;

const SparkStageRunnerModelInterface *SparkK3StageModel(void)
{
	return(0);
}

SparkStatus SparkStageRunnerInitialize(SparkStageRunner *runner,
	const SparkStageRunnerConfiguration *configuration, const SparkStageRunnerModelInterface *model)
{
	(void)runner;
	(void)configuration;
	(void)model;
	return(SPARK_STATUS_UNSUPPORTED);
}

SparkStatus SparkStageRunnerSubmit(SparkStageRunner *runner,
	const SparkStageRunnerDispatch *dispatch)
{
	(void)runner;
	test_dispatch_calls++;
	test_dispatch_rows = dispatch->row_count;
	test_dispatch_sequences = dispatch->active_sequence_count;
	return(test_dispatch_status);
}

SparkStatus SparkStageRunnerGetStats(const SparkStageRunner *runner,
	SparkStageRunnerStats *stats_out)
{
	(void)runner;
	if ( stats_out != 0 )
		memset(stats_out, 0, sizeof(*stats_out));
	return(SPARK_STATUS_OK);
}

void SparkStageRunnerDestroy(SparkStageRunner *runner)
{
	(void)runner;
}

SparkStatus SparkStageRunnerResetSlots(SparkStageRunner *runner,
	const uint32_t *slots, uint32_t count)
{
	(void)runner;
	test_reset_calls++;
	test_reset_count = count;
	for ( uint32_t index = 0u; index < count && index < TEST_K3_STUB_SLOTS; index++ )
		test_reset_slots[index] = slots[index];
	return(test_reset_status);
}

uint32_t SparkStageRunnerKvLayerCount(const SparkStageRunner *runner)
{
	(void)runner;
	return(test_kv_layers);
}

SparkStatus SparkStageRunnerAttachKv(SparkStageRunner *runner, const SparkStageRunnerKv *kv)
{
	(void)runner;
	test_attach_calls++;
	test_attached_kv = *kv;
	return(SPARK_STATUS_OK);
}

uint64_t SparkStageRunnerRecurrentBytes(const SparkStageRunner *runner)
{
	(void)runner;
	return(test_recurrent_bytes);
}

SparkStatus SparkStageRunnerRecurrentCopy(SparkStageRunner *runner, uint32_t to_buffer,
	uint32_t slot, void *buffer, uint64_t bytes, void *stream)
{
	(void)runner;
	(void)stream;
	if ( slot >= TEST_K3_STUB_SLOTS || bytes != test_recurrent_bytes || buffer == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( test_recurrent_status != SPARK_STATUS_OK )
		return(test_recurrent_status);
	if ( to_buffer != 0u )
	{
		test_recurrent_to_buffer++;
		memcpy(buffer, test_recurrent_lanes[slot], (size_t)bytes);
	}
	else
	{
		test_recurrent_from_buffer++;
		memcpy(test_recurrent_lanes[slot], buffer, (size_t)bytes);
	}
	return(SPARK_STATUS_OK);
}

SparkStatus SparkStageRunnerPackIdentity(const SparkStageRunner *runner, uint8_t *digest, uint32_t digest_bytes)
{
	(void)runner;
	if ( test_pack_status != SPARK_STATUS_OK )
		return(test_pack_status);
	memset(digest, 0x5a, digest_bytes);
	return(SPARK_STATUS_OK);
}
