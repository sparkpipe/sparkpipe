#pragma once

#include <stdint.h>

#define EFD_CLOEXEC 1
#define EFD_NONBLOCK 2

int eventfd(uint32_t initial_value,int flags);
