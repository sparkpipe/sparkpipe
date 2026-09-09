#include <float.h>
#include "sparkpipe/spark_numerical_metrics.h"

static int32_t TestFiniteMetrics(void)
{
	float reference[3] = {3.0f,4.0f,0.0f},actual[3] = {3.0f,0.0f,0.0f};
	SparkNumericalMetrics metrics;
	metrics = SparkNumericalMeasureF32(actual,reference,3u);
	if ( fabs(metrics.relative_l2 - 0.8) > 1e-12 || fabs(metrics.cosine - 0.6) > 1e-12 || metrics.max_absolute != 4.0 )
		return(-1);
	if ( SparkNumericalMetricsWithin(&metrics,0.02,0.999) != 0u )
		return(-2);
	metrics = SparkNumericalMeasureF32(reference,reference,3u);
	if ( SparkNumericalMetricsWithin(&metrics,0.0,0.999) == 0u || metrics.relative_l2 != 0.0 )
		return(-3);
	actual[0] = reference[0] = FLT_MAX;
	actual[1] = 1.0f;
	reference[1] = 2.0f;
	metrics = SparkNumericalMeasureF32(actual,reference,3u);
	if ( metrics.relative_l2 <= 0.0 || metrics.max_absolute != 1.0 || SparkNumericalMetricsWithin(&metrics,0.0,0.999) != 0u )
		return(-4);
	return(0);
}

static int32_t TestInvalidMetrics(void)
{
	float zero = 0.0f,one = 1.0f,invalid[3] = {NAN,INFINITY,-INFINITY};
	SparkNumericalMetrics metrics;
	uint32_t index;
	for (index=0u; index<3u; index++)
	{
		metrics = SparkNumericalMeasureF32(&invalid[index],&zero,1u);
		if ( SparkNumericalMetricsWithin(&metrics,1.0,-1.0) != 0u )
			return(-1);
		metrics = SparkNumericalMeasureF32(&zero,&invalid[index],1u);
		if ( SparkNumericalMetricsWithin(&metrics,1.0,-1.0) != 0u )
			return(-2);
	}
	metrics = SparkNumericalMeasureF32(&zero,&zero,1u);
	if ( SparkNumericalMetricsWithin(&metrics,0.0,1.0) == 0u )
		return(-3);
	metrics = SparkNumericalMeasureF32(&zero,&one,1u);
	if ( metrics.relative_l2 != 1.0 || metrics.cosine != 0.0 )
		return(-4);
	metrics = SparkNumericalMeasureF32(&one,&zero,1u);
	if ( metrics.relative_l2 != 1.0 || metrics.cosine != 0.0 )
		return(-5);
	metrics = SparkNumericalMeasureF32(&zero,&zero,0u);
	if ( SparkNumericalMetricsWithin(&metrics,1.0,-1.0) != 0u )
		return(-6);
	return(0);
}

int main(void)
{
	return(TestFiniteMetrics() != 0 || TestInvalidMetrics() != 0 ? 1 : 0);
}
