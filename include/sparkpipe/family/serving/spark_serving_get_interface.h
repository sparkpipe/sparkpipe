#pragma once

__attribute__((visibility("default")))
const SparkModelServingAdapterInterface *SparkModelServingAdapterGetInterface(void)
{
	return(&SPARK_FAMILY(ServingInterface));
}
