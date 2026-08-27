/* K3 pack-lane build shim: the residentd resolves the serving-adapter
 * interface by the FIXED symbol SparkModelServingAdapterGetInterface
 * (include/sparkpipe/spark_model_serving_adapter.h), which every other
 * family's adapter exports. spark_k3_serving_adapter.c exports only the
 * K3-named getter, so no K3 deployment can load until the module adds the
 * generic export. This translation unit supplies it as build glue for the
 * lane's deployment copy; INTEGRATION REQUEST: move the export (or an
 * alias) into modules/k3_resident_decode_stage. */
#include "sparkpipe/spark_k3_serving_adapter.h"
#include "sparkpipe/spark_model_serving_adapter.h"

const SparkModelServingAdapterInterface *SparkModelServingAdapterGetInterface(void);

const SparkModelServingAdapterInterface *SparkModelServingAdapterGetInterface(void)
{
	return SparkK3ServingAdapterGetInterface();
}
