#pragma once

#include_next <unistd.h>

#if defined(__APPLE__)
#ifndef _SC_NPROCESSORS_CONF
#define _SC_NPROCESSORS_CONF 57
#endif
#endif
