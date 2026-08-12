#pragma once

#include "sparkpipe/spark_tp_device_collective.h"

SparkStatus SparkTpDeviceCollectiveNcclCreate(
	const SparkTpDeviceCollectiveConfig *config,
	SparkTpDeviceCollective *collective_out);

SparkStatus SparkTpDeviceCollectiveNcclSubmitBf16(
	SparkTpDeviceCollective *collective,
	const SparkTpDeviceCollectiveSubmission *submission);

SparkStatus SparkTpDeviceCollectiveNcclRequestFailure(
	SparkTpDeviceCollective *collective,
	SparkStatus failure_status);

SparkStatus SparkTpDeviceCollectiveNcclRequestOperationFailure(
	SparkTpDeviceCollective *collective,
	uint64_t ordinal,
	SparkStatus failure_status);

SparkStatus SparkTpDeviceCollectiveNcclOperationPhase(
	const SparkTpDeviceCollective *collective,
	uint64_t ordinal,
	uint32_t *phase_out,
	uint32_t *failure_requested_out);

void SparkTpDeviceCollectiveNcclDestroy(
	SparkTpDeviceCollective *collective);
