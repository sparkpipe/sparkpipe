#include <stddef.h>
#include <stdint.h>

#include "runtime/device/nvrm/nvrm_rm.h"

typedef struct RmStatusName
{
	uint32_t code;
	const char *name;
} RmStatusName;

#define NV_STATUS_CODE(name, code, string) {(code), #name},
static const RmStatusName rm_status_names[] = {
#include "nvstatuscodes.h"
};
#undef NV_STATUS_CODE

const char *SparkNvrmRmStatusText(uint32_t rm_status)
{
	size_t index;
	for (index = 0u; index < sizeof(rm_status_names) / sizeof(rm_status_names[0]); index++)
		if ( rm_status_names[index].code == rm_status )
			return(rm_status_names[index].name);
	return("an unlisted status");
}
