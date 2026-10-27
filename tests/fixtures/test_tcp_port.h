#pragma once

#include <stdint.h>

#define TEST_TCP_PORT_FIRST 20000u
#define TEST_TCP_PORT_END 32768u
#define TEST_TCP_PORT_ATTEMPTS 512u
#define TEST_TCP_PORT_HANDED_MAX 256u

uint32_t TestTcpPortRange(uint32_t count);
