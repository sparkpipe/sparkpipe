#pragma once

#include <stdint.h>
#include <sys/types.h>

#define TEST_CHILD_GUARD_OWNER_VARIABLE "SPARK_TEST_FIXTURE_OWNER"
#define TEST_CHILD_GUARD_CAPACITY 256u
#define TEST_CHILD_GUARD_REAP_WAIT_MS 3000u

pid_t TestChildGuardFork(void);
void TestChildGuardKillAll(void);
uint32_t TestChildGuardTrackedCount(void);
