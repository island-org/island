#include "shadertoy.cuh"

extern "C" __global__ void
cuda_main(unsigned char *outColor)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    float u = x / (float) iResolution.x;
    float v = y / (float) iResolution.y;

    if ((x < iResolution.x) && (y < iResolution.y))
    {
        int idx = (y * iResolution.x + x) * 4;
        outColor[idx+0] = u * 255;
        outColor[idx+1] = v * 255;
        outColor[idx+2] = 122 + 122*sin(iGlobalTime);
        outColor[idx+3] = 255;
    }
}
