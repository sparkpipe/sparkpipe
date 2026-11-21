#ifndef _NV_CPU_UUID_H_
#define _NV_CPU_UUID_H_

#define NV_UUID_LEN 16

typedef struct nv_uuid
{
    NvU8 uuid[NV_UUID_LEN];

} NvUuid;

#define NV_UUID_HI(pUuid) (*((NvU64*)((pUuid)->uuid + (NV_UUID_LEN >> 1))))
#define NV_UUID_LO(pUuid) (*((NvU64*)((pUuid)->uuid + 0)))

typedef NvUuid NvSystemUuid;

typedef NvUuid NvProcessorUuid;

extern const NvProcessorUuid NV_PROCESSOR_UUID_CPU_DEFAULT;

#endif
