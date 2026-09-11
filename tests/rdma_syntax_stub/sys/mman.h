#pragma once

#include_next <sys/mman.h>

#ifndef MFD_CLOEXEC
#define MFD_CLOEXEC 1
#endif

int memfd_create(const char *name, unsigned int flags);
