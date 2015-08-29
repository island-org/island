#include "shadertoy.cu"

extern "C" __global__ void mainImage(unsigned char* fragColor)
{
    float2 fragCoord = calcFragCoord();
    float u = fragCoord.x / iResolution.x;
    float v = fragCoord.y / iResolution.y;

    if ((fragCoord.x < iResolution.x) && (fragCoord.y < iResolution.y))
    {
        int idx = (fragCoord.y * iResolution.x + fragCoord.x) * 4;
        fragColor[idx+0] = u * 255;
        fragColor[idx+1] = v * 255;
        fragColor[idx+2] = 122 + 122*sin(iGlobalTime);
        fragColor[idx+3] = 255;
    }
}
