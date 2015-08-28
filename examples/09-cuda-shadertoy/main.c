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
    char* ptx;
    size_t ptxSize;
    compileFileToPTX(kernel_file, 0, NULL, &ptx, &ptxSize);
    CUmodule module = loadPTX(ptx);
    checkCudaErrors(cuModuleGetFunction(&kernel_addr, module, "main"));

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
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    dim3 cudaBlockSize = { threadsPerBlock, 1, 1 };
    dim3 cudaGridSize = { blocksPerGrid, 1, 1 };

    void *arr[] = { (void *)&d_img_content, (void *)&img.width, (void *)&img.height };
    checkCudaErrors(cuLaunchKernel(kernel_addr,
        cudaGridSize.x, cudaGridSize.y, cudaGridSize.z, /* grid dim */
        cudaBlockSize.x, cudaBlockSize.y, cudaBlockSize.z, /* block dim */
        0, 0, /* shared mem, stream */
        &arr[0], /* arguments */
        0));
    checkCudaErrors(cuCtxSynchronize());

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    checkCudaErrors(cuMemcpyDtoH(img_content, d_img_content, item_size));

    updateImage(img, img_content);
    image(img, 0, 0, width, height);

    ellipse(mouseX, mouseY, 10, 10);
}

void shutdown()
{
    // Free device global memory
    checkCudaErrors(cuMemFree(d_img_content));
    cuProfilerStop();

    free(img_content);

    printf("Done\n");
}
