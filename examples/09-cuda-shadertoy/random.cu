extern "C" __global__ void
kernel(unsigned char *img, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    float u = x / (float) width;
    float v = y / (float) height;

    if ((x < width) && (y < height))
    {
        // write output color
        int idx = (y * width + x) * 4;
        img[idx+0] = u * 255;
        img[idx+1] = v * 255;
        img[idx+2] = 0;
        img[idx+3] = 255;
    }
}
