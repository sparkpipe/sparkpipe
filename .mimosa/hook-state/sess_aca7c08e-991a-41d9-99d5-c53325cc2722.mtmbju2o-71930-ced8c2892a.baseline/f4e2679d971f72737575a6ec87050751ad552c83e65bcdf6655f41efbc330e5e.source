/* Compile check: the speculation-provider slot header and the speculation
 * dispatch-policy header must coexist in one translation unit, in either
 * include order (the SparkSpeculationDraftRequest collision fix). The
 * Makefile builds this TU once per order. */
#ifdef SPARK_COEXIST_POLICY_FIRST
#include "sparkpipe/spark_speculation_policy.h"
#include "sparkpipe/spark_speculation_provider.h"
#else
#include "sparkpipe/spark_speculation_provider.h"
#include "sparkpipe/spark_speculation_policy.h"
#endif

int main(void)
{
	return 0;
}
