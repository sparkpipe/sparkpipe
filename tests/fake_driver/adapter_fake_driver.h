/* Shared layout between the parameterized fake model-driver archive
 * (adapter_fake_driver.c, compiled once per family profile) and
 * tests/test_adapter_common_integration.c. Exposing the instance lets the
 * harness drive the EXACT callbacks SparkAdapterLoadDriver wired - the
 * shared orphan-completion route and the wake trampoline - plus mutate the
 * runtime-snapshot counters Quiesce/Snapshot/DestroyReady consume.
 *
 * Test infrastructure only; never linked into production units. */
#ifndef SPARK_TEST_ADAPTER_FAKE_DRIVER_H
#define SPARK_TEST_ADAPTER_FAKE_DRIVER_H

#include <stdint.h>

#include "sparkpipe/spark_model_driver.h"

typedef struct SparkFakeDriverInstance
{
	/* Verbatim copy of the create request: holds the adapter-wired
	 * completion/wake routes exactly as production drivers retain them. */
	SparkModelDriverCreateRequest request;
	/* Runtime-snapshot counters the harness mutates between phases. */
	uint64_t submitted_count;
	uint64_t completed_count;
	uint64_t rejected_count;
	uint32_t active_submission_count;
	uint32_t available_dispatch_slot_count;
} SparkFakeDriverInstance;

#endif
