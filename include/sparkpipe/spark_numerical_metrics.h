#pragma once

#include <math.h>
#include <stdint.h>

typedef struct SparkNumericalMetrics
{
	double relative_l2;
	double cosine;
	double max_absolute;
} SparkNumericalMetrics;

static inline SparkNumericalMetrics SparkNumericalMeasureF32(const float *actual,const float *reference,uint64_t count)
{
	SparkNumericalMetrics result = {INFINITY,-1.0,INFINITY};
	double dot = 0.0,actual_norm = 0.0,reference_norm = 0.0,error_norm = 0.0,maximum = 0.0;
	double a,r,error;
	uint64_t index;
	if ( actual == 0 || reference == 0 || count == 0u )
		return(result);
	for (index=0u; index<count; index++)
	{
		a = actual[index];
		r = reference[index];
		if ( isfinite(a) == 0 || isfinite(r) == 0 )
			return(result);
		error = a - r;
		dot += a * r;
		actual_norm += a * a;
		reference_norm += r * r;
		error_norm += error * error;
		if ( fabs(error) > maximum )
			maximum = fabs(error);
	}
	result.relative_l2 = reference_norm > 0.0 ? sqrt(error_norm / reference_norm) : sqrt(error_norm);
	result.cosine = actual_norm > 0.0 && reference_norm > 0.0 ? dot / (sqrt(actual_norm) * sqrt(reference_norm)) : (actual_norm == reference_norm ? 1.0 : 0.0);
	result.max_absolute = maximum;
	return(result);
}

static inline uint32_t SparkNumericalMetricsWithin(const SparkNumericalMetrics *metrics,double maximum_relative_l2,double minimum_cosine)
{
	if ( metrics == 0 || isfinite(maximum_relative_l2) == 0 || isfinite(minimum_cosine) == 0 || maximum_relative_l2 < 0.0 || minimum_cosine < -1.0 || minimum_cosine > 1.0 )
		return(0u);
	return(isfinite(metrics->relative_l2) != 0 && isfinite(metrics->cosine) != 0 && isfinite(metrics->max_absolute) != 0 && metrics->relative_l2 <= maximum_relative_l2 && metrics->cosine >= minimum_cosine);
}
