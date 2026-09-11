#include "spark_hy4_fp8_scale_contract.h"
#include "spark_hy4_stagepack_format.h"

const SparkStagePackGeometryTable *SparkHy4FormatCheckGeometry(void)
{
	return &SparkHy4StagePackGeometry;
}

int32_t SparkHy4FormatCheckShape(uint32_t kind,
	SparkHy4StagePackTensorShape *shape)
{
	return SparkHy4StagePackTensorShapeOf(kind, shape);
}

uint64_t SparkHy4FormatCheckPayload(uint32_t format, uint32_t rows,
	uint32_t columns)
{
	return SparkHy4StagePackPayloadBytes(format, rows, columns);
}

uint64_t SparkHy4FormatCheckScale(uint32_t format, uint32_t rows,
	uint32_t columns)
{
	return SparkHy4StagePackScaleBytes(format, rows, columns);
}

int32_t SparkHy4FormatCheckFp8Scale(uint32_t kind, uint32_t rank,
	uint32_t ranks, uint32_t payload_rows, uint32_t payload_columns,
	uint32_t scale_rows, uint32_t scale_groups,
	SparkHy4Fp8ScaleContract *contract)
{
	return SparkHy4Fp8ScaleContractValidate(kind, rank, ranks,
	    payload_rows, payload_columns, scale_rows, scale_groups,
	    contract);
}

uint64_t SparkHy4FormatCheckFp8ScaleIndex(
	const SparkHy4Fp8ScaleContract *contract, uint32_t local_row,
	uint32_t local_group)
{
	return SparkHy4Fp8ScaleByteIndex(contract, local_row, local_group);
}

uint32_t SparkHy4FormatCheckFp8ScaleZeroCopy(
	const SparkHy4Fp8ScaleContract *contract)
{
	return SparkHy4Fp8ScaleZeroCopy(contract);
}
