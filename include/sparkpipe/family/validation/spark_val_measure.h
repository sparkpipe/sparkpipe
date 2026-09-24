#pragma once

static void SPARK_FAMILY(ValMeasure)(SPARK_FAMILY(ValMetrics) *metrics, const float *actual, const float *reference, uint64_t count)
{
	uint64_t index;
	double difference;
	memset(metrics,0,sizeof(*metrics));
	metrics->count = count;
	for (index = 0u; index < count; index++)
	{
		difference = (double)actual[index] - (double)reference[index];
		metrics->difference_l2 += difference * difference;
		metrics->reference_l2 += (double)reference[index] * (double)reference[index];
		metrics->actual_l2 += (double)actual[index] * (double)actual[index];
		metrics->dot += (double)actual[index] * (double)reference[index];
		if (fabs((double)actual[index] - (double)reference[index]) > metrics->maximum_absolute)
			metrics->maximum_absolute = fabs((double)actual[index] - (double)reference[index]);
	}
}

static int SPARK_FAMILY(ValReport)(const char *check,const SPARK_FAMILY(ValMetrics) *metrics,double max_relative_l2,double minimum_cosine)
{
	double relative_l2 = metrics->reference_l2 > 0.0
		? sqrt(metrics->difference_l2 / metrics->reference_l2) : INFINITY;
	double cosine = metrics->actual_l2 > 0.0 && metrics->reference_l2 > 0.0
		? metrics->dot / sqrt(metrics->actual_l2 * metrics->reference_l2) : 0.0;
	printf(SPARK_FAMILY_STRING(SPARK_FAMILY_LOWER) "_validation check=%s elements=%llu relative_l2=%.9g cosine=%.9g max_abs=%.9g\n",
		check,(unsigned long long)metrics->count,relative_l2,cosine,metrics->maximum_absolute);
	if ( isfinite(relative_l2) == 0 || relative_l2 > max_relative_l2 )
		return(SPARK_FAMILY(ValFail)(check,"relative_l2"));
	if ( cosine < minimum_cosine )
		return(SPARK_FAMILY(ValFail)(check,"cosine"));
	return(0);
}
