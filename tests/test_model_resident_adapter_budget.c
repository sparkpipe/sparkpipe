#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define main SparkModelResidentdProgramMain
#include "../node/model_residentd.c"
#undef main

#define TEST_BUDGET_ROUTES 2u
#define TEST_BUDGET_CALLS 8u

typedef struct TestBudgetAdapter
{
	uint32_t calls;
	uint64_t submitted[TEST_BUDGET_CALLS];
	SparkStatus result;
} TestBudgetAdapter;

typedef struct TestBudgetFixture
{
	SparkModelResidentdRuntime runtime;
	SparkModelResidentdRoute routes[TEST_BUDGET_ROUTES];
	SparkModelServingAdapterDescriptor descriptor;
} TestBudgetFixture;

static TestBudgetAdapter TestBudgetState;
static uint32_t TestBudgetFailures;

static void TestBudgetCheck(uint32_t condition,const char *message)
{
	if ( condition == 0u )
	{
		fprintf(stderr,"FAIL %s\n",message);
		TestBudgetFailures++;
	}
}

static SparkStatus TestBudgetSubmit(void *adapter_state,const SparkModelServingSubmission *submission)
{
	(void)adapter_state;
	if ( TestBudgetState.calls < TEST_BUDGET_CALLS )
		TestBudgetState.submitted[TestBudgetState.calls] = submission->submission_id;
	TestBudgetState.calls++;
	return(TestBudgetState.result);
}

static uint32_t TestBudgetWakeBytes(const SparkModelResidentdRuntime *runtime)
{
	uint8_t values[64];
	uint32_t total = 0u;
	ssize_t bytes;
	while ( (bytes = read(runtime->wake_read_fd,values,sizeof(values))) > 0 )
		total += (uint32_t)bytes;
	return(total);
}

static void TestBudgetRoute(TestBudgetFixture *fixture,uint32_t slot,uint64_t submission_id)
{
	SparkModelResidentdRoute *route = &fixture->routes[slot];
	route->active = 1u;
	route->slot_index = slot;
	route->state = SPARK_MODEL_RESIDENTD_ROUTE_READY_ADAPTER;
	route->submission_id = submission_id;
	route->submission.submission_id = submission_id;
	TestBudgetCheck(SparkModelResidentdEnqueueCommittedLocked(&fixture->runtime,route) == SPARK_STATUS_OK,"the committed FIFO takes the route");
}

static void TestBudgetSetup(TestBudgetFixture *fixture,uint32_t capability_flags,SparkStatus result)
{
	memset(fixture,0,sizeof(*fixture));
	memset(&TestBudgetState,0,sizeof(TestBudgetState));
	TestBudgetState.result = result;
	fixture->descriptor.capability_flags = capability_flags;
	fixture->runtime.routes = fixture->routes;
	fixture->runtime.route_capacity = TEST_BUDGET_ROUTES;
	fixture->runtime.adapter_library.adapter_interface.descriptor = &fixture->descriptor;
	fixture->runtime.adapter_library.adapter_interface.submit = TestBudgetSubmit;
	TestBudgetCheck(pthread_mutex_init(&fixture->runtime.mutex,0) == 0,"the runtime mutex initializes");
	TestBudgetCheck(SparkModelResidentdOpenWakePipe(&fixture->runtime) == SPARK_STATUS_OK,"the wake pipe opens");
	TestBudgetRoute(fixture,1u,1u);
	TestBudgetRoute(fixture,0u,2u);
}

static void TestBudgetTeardown(TestBudgetFixture *fixture)
{
	close(fixture->runtime.wake_read_fd);
	close(fixture->runtime.wake_write_fd);
	pthread_mutex_destroy(&fixture->runtime.mutex);
}

static void TestBusyHeadDoesNotSpin(void)
{
	TestBudgetFixture fixture;
	SparkStatus status;
	TestBudgetSetup(&fixture,SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_ASYNC_COMPLETION,SPARK_STATUS_BUSY);
	status = SparkModelResidentdProgressRoutes(&fixture.runtime,1u);
	TestBudgetCheck(status == SPARK_STATUS_OK,"a busy head is not an error");
	TestBudgetCheck(TestBudgetState.calls == 1u && TestBudgetState.submitted[0] == 1u,"only the FIFO head reaches the adapter");
	TestBudgetCheck(fixture.runtime.adapter_op_count == 0u,"a route behind the head is not counted as an adapter op");
	TestBudgetCheck(TestBudgetWakeBytes(&fixture.runtime) == 0u,"a busy head leaves the wake pipe empty, so poll sleeps until a completion");
	TestBudgetCheck(fixture.routes[1].state == SPARK_MODEL_RESIDENTD_ROUTE_READY_ADAPTER && fixture.routes[0].state == SPARK_MODEL_RESIDENTD_ROUTE_READY_ADAPTER,"both routes stay ready for the next pass");
	TestBudgetCheck(fixture.runtime.committed_fifo_head == 2u,"the head keeps its FIFO place");
	TestBudgetTeardown(&fixture);
}

static void TestSerialHeadIsSubmitted(void)
{
	TestBudgetFixture fixture;
	SparkStatus status;
	TestBudgetSetup(&fixture,0u,SPARK_STATUS_OK);
	status = SparkModelResidentdProgressRoutes(&fixture.runtime,1u);
	TestBudgetCheck(status == SPARK_STATUS_OK,"the serial pass succeeds");
	TestBudgetCheck(TestBudgetState.calls == 1u && TestBudgetState.submitted[0] == 1u,"a serial adapter gets the FIFO head even when a later route sits in a lower slot");
	TestBudgetCheck(fixture.routes[1].state == SPARK_MODEL_RESIDENTD_ROUTE_WAIT_ADAPTER,"the head waits on the adapter");
	TestBudgetCheck(fixture.runtime.adapter_op_count == 1u && TestBudgetWakeBytes(&fixture.runtime) == 1u,"one real op, one wake");
	TestBudgetCheck(fixture.runtime.committed_fifo_head == 1u,"the next route becomes the head");
	status = SparkModelResidentdProgressRoutes(&fixture.runtime,1u);
	TestBudgetCheck(status == SPARK_STATUS_OK && TestBudgetState.calls == 2u && TestBudgetState.submitted[1] == 2u,"the next pass submits the new head");
	TestBudgetTeardown(&fixture);
}

int main(void)
{
	TestBusyHeadDoesNotSpin();
	TestSerialHeadIsSubmitted();
	if ( TestBudgetFailures != 0u )
		return(1);
	printf("PASS model_residentd adapter budget: routes behind the FIFO head neither spin the loop nor starve the head\n");
	return(0);
}
