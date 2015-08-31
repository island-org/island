#include "shadertoy.cuh"

extern "C" __global__ void mainImage()
{
    float2 fragCoord = calcFragCoord();
    float u = fragCoord.x / iResolution.x;
    float v = fragCoord.y / iResolution.y;

    if ((fragCoord.x < iResolution.x) && (fragCoord.y < iResolution.y))
    {
        uchar4* fragColor = calcFragColor(fragCoord);
        *fragColor = make_uchar4(u * 255, v * 255, 122 + 122 * sin(iGlobalTime), 255);
    }
}
