#include <string.h>

#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_minimax_h3_scheduler.h"

int32_t SparkMinimaxH3SchedulerBuildSigmas(float shift, uint32_t sigma_point_count,
	float *sigmas_out, uint32_t *sigma_count_out)
{
	uint32_t index,out_count;
	float previous;
	if ( sigmas_out == 0 || sigma_count_out == 0 )
		SPARK_FAIL(-2101);
	if ( sigma_point_count < 2u )
		SPARK_FAIL(-2102);
	if ( !(shift > 0.0f) )
		SPARK_FAIL(-2103);
	out_count = 0u;
	previous = 0.0f;
	for (index=0u; index<sigma_point_count; index++)
	{
		float base = 1.0f - (float)index / (float)(sigma_point_count - 1u);
		float shifted = shift * base / (1.0f + (shift - 1.0f) * base);
		if ( out_count != 0u )
		{
			float canonical = shifted;
			(void)canonical;
			if ( memcmp(&shifted,&previous,sizeof(float)) == 0 )
				continue;
		}
		sigmas_out[out_count++] = shifted;
		previous = shifted;
	}
	if ( out_count < 2u )
		SPARK_FAIL(-2104);
	*sigma_count_out = out_count;
	return(0);
}

void SparkMinimaxH3SchedulerTimestepsFromSigmas(const float *sigmas, uint32_t sigma_count,
	float *timesteps_out, uint32_t *timestep_count_out)
{
	uint32_t index;
	if ( sigmas == 0 || timesteps_out == 0 || timestep_count_out == 0 || sigma_count < 1u )
		return;
	for (index=0u; index + 1u<sigma_count; index++)
		timesteps_out[index] = 1.0f - sigmas[index];
	*timestep_count_out = sigma_count - 1u;
}

void SparkMinimaxH3SchedulerStepElement(float timestep, float sigma, float sigma_next,
	float sample, float velocity, float *sample_next_out)
{
	float sigma_from_timestep = 1.0f - timestep;
	float denoised = sample + sigma_from_timestep * velocity;
	float ratio = sigma_next / sigma;
	*sample_next_out = ratio * sample + (1.0f - ratio) * denoised;
}
