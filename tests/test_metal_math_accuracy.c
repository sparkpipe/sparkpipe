#include <math.h>
#include <stdio.h>
#include <string.h>

#include "sparkpipe/spark_device.h"

#define EXP_ENVELOPE_ULP 16.0

typedef struct ExpStats
{
	double worst;
	double mean_abs;
	double bias;
	double exact;
	uint32_t worst_at;
} ExpStats;

static ExpStats MeasureExp(const float *values, uint32_t count, float step, uint32_t nudged)
{
	ExpStats stats;
	uint32_t index, exact = 0u;
	memset(&stats,0,sizeof(stats));
	for ( index = 0u; index < count; index++ )
	{
		const float x = -(float)index * step;
		const double truth = exp((double)x), ulp = (double)(nextafterf((float)truth,INFINITY) - (float)truth);
		const double got = (double)values[index] + (index == nudged ? (EXP_ENVELOPE_ULP + 1.0) * ulp : 0.0);
		const double error = (got - truth) / ulp;
		stats.bias += error;
		stats.mean_abs += fabs(error);
		exact += got == (double)(float)truth ? 1u : 0u;
		if ( fabs(error) > stats.worst )
		{
			stats.worst = fabs(error);
			stats.worst_at = index;
		}
	}
	stats.bias /= count;
	stats.mean_abs /= count;
	stats.exact = (double)exact / count;
	return(stats);
}

int main(void)
{
	enum { COUNT = 1u << 22 };
	const float step = 80.0f / (float)COUNT;
	SparkDeviceQueue *queue = 0;
	SparkDeviceLibrary *library = 0;
	SparkDeviceKernel *kernel = 0;
	SparkDeviceBuffer out;
	SparkDeviceBinding binding;
	SparkDeviceLaunch launch;
	if ( SparkDeviceInitialize(0) != SPARK_STATUS_OK || strcmp(SparkDeviceGetProperties()->target,"metal.apple10") != 0 )
	{
		printf("SKIP test_metal_math_accuracy (needs an Apple family 10 GPU)\n");
		return(0);
	}
	if ( SparkDeviceQueueCreate(&queue,"math accuracy") != SPARK_STATUS_OK || SparkDeviceLibraryLoad(&library,SPARK_MATH_TEST_LIBRARY,"math accuracy") != SPARK_STATUS_OK ||
		SparkDeviceKernelCreate(&kernel,library,"spark_test_exp") != SPARK_STATUS_OK ||
		SparkDeviceAllocate(&out,SPARK_DEVICE_SPACE_SHARED_COHERENT,COUNT * sizeof(float),"exp out") != SPARK_STATUS_OK )
	{
		printf("FAIL math accuracy setup: %s\n",SparkDeviceLastError()->text);
		return(1);
	}
	binding.range.buffer = &out;
	binding.range.offset = 0u;
	binding.bytes = out.bytes;
	binding.access = SPARK_DEVICE_BIND_WRITE;
	memset(&launch,0,sizeof(launch));
	launch.groups.x = COUNT / 256u;
	launch.groups.y = launch.groups.z = 1u;
	launch.threads.x = 256u;
	launch.threads.y = launch.threads.z = 1u;
	launch.constants = &step;
	launch.constant_bytes = sizeof(step);
	launch.bindings = &binding;
	launch.binding_count = 1u;
	if ( SparkDeviceKernelLaunch(queue,kernel,&launch) != SPARK_STATUS_OK || SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) != SPARK_STATUS_OK )
	{
		printf("FAIL math accuracy launch: %s\n",SparkDeviceLastError()->text);
		return(1);
	}
	{
		const float *values = (const float *)out.host;
		const uint32_t nudged = COUNT / 3u;
		ExpStats clean = MeasureExp(values,COUNT,step,COUNT), control = MeasureExp(values,COUNT,step,nudged);
		const int pass = clean.worst <= EXP_ENVELOPE_ULP && control.worst > EXP_ENVELOPE_ULP;
		printf("%s exp                    [-80, 0], %u inputs: max %.2f ulp (at %.6f), mean |error| %.3f ulp, mean signed error %+.4f ulp, correctly rounded %.1f%%; "
			"envelope %.0f ulp; one output moved by %.0f ulp: max %.2f ulp (control fails as required: %s)\n",pass ? "PASS" : "FAIL",(unsigned)COUNT,clean.worst,
			-(double)clean.worst_at * step,clean.mean_abs,clean.bias,100.0 * clean.exact,EXP_ENVELOPE_ULP,EXP_ENVELOPE_ULP + 1.0,control.worst,
			control.worst > EXP_ENVELOPE_ULP ? "yes" : "no");
		return(pass ? 0 : 1);
	}
}
