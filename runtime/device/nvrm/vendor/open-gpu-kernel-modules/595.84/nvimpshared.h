#pragma once

#include <nvtypes.h>

typedef struct DRAM_CLK_INSTANCE {
    NvU32 dram_clk_freq_khz;

    NvU32 mchub_clk_khz;

    NvU32 mc_clk_khz;

    NvU32 max_iso_bw_kbps;

    NvU32 switch_latency_ns;
} DRAM_CLK_INSTANCE;

typedef struct TEGRA_IMP_IMPORT_DATA {

    NvU32             max_iso_bw_kbps;

    NvU32             num_dram_channels;

    NvU32             num_dram_clk_entries;
    DRAM_CLK_INSTANCE dram_clk_instance[24];
} TEGRA_IMP_IMPORT_DATA;
