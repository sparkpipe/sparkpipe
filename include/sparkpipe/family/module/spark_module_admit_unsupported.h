#pragma once

static SparkStatus SPARK_FAMILY(ModuleAdmit)(void *module_state,const SparkModelDriverAdmissionRequest *request,SparkModelDriverAdmissionDecision *decision)
{
	(void)module_state;
	(void)request;
	(void)decision;
	return(SPARK_STATUS_UNSUPPORTED);
}
