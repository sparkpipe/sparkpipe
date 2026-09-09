#include "spark_hy4_stagepack_format.h"

/* Compile-only proof for the hy4 stagepack format header: the layout
 * static asserts fire here, and the geometry table and shape helpers
 * are referenced so optimization cannot hide them. */
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
