#include <stdlib.h>
#include "../validation/l4_route_synthesis.h"
int main(void)
{
    Slr4CaseShape shape = { 3u, 4u, 16u, 64u, 32u, SLR4_FORMAT_BF16, 0u,
                            0.0f, 7u };
    uint32_t indices[12];
    float weights[12];
    Slr4SynthesizeRoute(&shape, indices, weights);
    return indices[0] < shape.expert_count ? 0 : 1;
}
