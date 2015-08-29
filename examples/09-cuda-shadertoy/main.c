#define SKETCH_2D_IMPLEMENTATION
#include "sketch2d.h"

#include "CudaApi.h"

CUdeviceptr d_img_content;
CUfunction kernel_addr;
PImage img;
unsigned char* img_content;
size_t item_size;

void setup()
{
    checkCudaErrors(cuInit(0));

    char* kernel_file = "../examples/09-cuda-shadertoy/random.cu";
    kernel_addr = getCompiledKernel(kernel_file, "kernel");

    item_size = width * height * 4;
    checkCudaErrors(cuMemAlloc(&d_img_content, item_size));

    img = createImage(width, height);
    img_content = (unsigned char*)malloc(item_size);
}

void draw()
{
    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (img.width * img.height + threadsPerBlock - 1) / threadsPerBlock;
    //printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    dim3 blockDim = { 16, 16, 1 };
    dim3 gridDim = { width / blockDim.x, height / blockDim.y, 1 };

    void *arr[] = { (void *)&d_img_content, (void *)&img.width, (void *)&img.height };
    checkCudaErrors(cuLaunchKernel(kernel_addr,
        gridDim.x, gridDim.y, gridDim.z, /* grid dim */
        blockDim.x, blockDim.y, blockDim.z, /* block dim */
        0, 0, /* shared mem, stream */
        &arr[0], /* arguments */
        0));
    checkCudaErrors(cuCtxSynchronize());

    checkCudaErrors(cuMemcpyDtoH(img_content, d_img_content, item_size));

    updateImage(img, img_content);
    image(img, 0, 0, width, height);
}

void shutdown()
{
    // Free device global memory
    checkCudaErrors(cuMemFree(d_img_content));
    cuProfilerStop();

    free(img_content);

    printf("Done\n");
}
