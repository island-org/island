#define SKETCH_2D_IMPLEMENTATION
#include "sketch2d.h"

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cuda.h>
#include <cudaProfiler.h>
#include <nvrtc.h>

#include "drvapi_error_string.h"
// helper functions and utilities to work with CUDA
//#include "nvrtc_helper.h"

#define checkCudaRtcErrors(err)  __checkCudaRtcErrors (err, __FILE__, __LINE__)

// These are the inline versions for all of the SDK helper functions
static void __checkCudaRtcErrors(nvrtcResult err, const char *file, const int line)
{
    if (NVRTC_SUCCESS != err)
    {
        fprintf(stderr, "checkCudaRtcErrors() error = %04d \"%s\" from file <%s>, line %i.\n",
            err, nvrtcGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

// These are the inline versions for all of the SDK helper functions
static void __checkCudaErrors(CUresult err, const char *file, const int line)
{
    if (CUDA_SUCCESS != err)
    {
        fprintf(stderr, "checkCudaErrors() Driver API error = %04d \"%s\" from file <%s>, line %i.\n",
            err, getCudaDrvErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

void compileFileToPTX(char *filename, int argc, const char **argv,
    char **ptxResult, size_t *ptxResultSize)
{
    FILE* fp = fopen(filename, "rb");
    if (!fp)
    {
        printf("\nerror: unable to open %s for reading!\n", filename);
        exit(1);
    }

    fseek(fp, 0L, SEEK_END);
    size_t inputSize = ftell(fp);
    char* memBlock = (char*)malloc(inputSize+1);
    fseek(fp, 0, SEEK_SET);
    fread(memBlock, sizeof(char), inputSize, fp);
    memBlock[inputSize] = '\x0';

    // compile
    nvrtcProgram prog;
    checkCudaRtcErrors(nvrtcCreateProgram(&prog, memBlock,
        filename, 0, NULL, NULL));
    checkCudaRtcErrors(nvrtcCompileProgram(prog, argc, argv));

    // dump log
    size_t logSize;
    checkCudaRtcErrors(nvrtcGetProgramLogSize(prog, &logSize));
    char *log = (char *)malloc(sizeof(char) * logSize + 1);
    checkCudaRtcErrors(nvrtcGetProgramLog(prog, log));
    log[logSize] = '\x0';

    printf("\n compilation log-- - \n %s \n end log ---\n", log);
    free(log);

    // fetch PTX
    size_t ptxSize;
    checkCudaRtcErrors(nvrtcGetPTXSize(prog, &ptxSize));
    char *ptx = (char *)malloc(sizeof(char) * ptxSize);
    checkCudaRtcErrors(nvrtcGetPTX(prog, ptx));
    checkCudaRtcErrors(nvrtcDestroyProgram(&prog));
    *ptxResult = ptx;
    *ptxResultSize = ptxSize;
}

CUmodule loadPTX(char *ptx)
{
    CUmodule module;
    CUcontext context;
    int major = 0, minor = 0;
    char deviceName[256];

    // Picks the best CUDA device available
    int cuDevice = 0;

    // get compute capabilities and the devicename
    checkCudaErrors(cuDeviceComputeCapability(&major, &minor, cuDevice));
    checkCudaErrors(cuDeviceGetName(deviceName, 256, cuDevice));
    printf("> GPU Device has SM %d.%d compute capability\n", major, minor);

    checkCudaErrors(cuDeviceGet(&cuDevice, 0));
    checkCudaErrors(cuCtxCreate(&context, 0, cuDevice));

    checkCudaErrors(cuModuleLoadDataEx(&module, ptx, 0, 0, 0));

    return module;
}

float* h_A;
float* h_B;
float* h_C;
CUdeviceptr d_A;
CUdeviceptr d_B;
CUdeviceptr d_C;
const int numElements = 50000;
CUfunction kernel_addr;
size_t item_size;
PImage img;
unsigned char* img_content;

void setup()
{
#if 0
    checkCudaErrors(cuInit(0));

    char* kernel_file = "../examples/09-cuda-shadertoy/vectorAdd_kernel.cu";
    char* ptx;
    size_t ptxSize;
    compileFileToPTX(kernel_file, 0, NULL, &ptx, &ptxSize);
    CUmodule module = loadPTX(ptx);
    item_size = numElements * sizeof(float);

    checkCudaErrors(cuModuleGetFunction(&kernel_addr, module, "vectorAdd"));

    // Print the vector length to be used, and compute its size
    printf("[Vector addition of %d elements]\n", numElements);

    h_A = (float *)malloc(item_size);
    h_B = (float *)malloc(item_size);
    h_C = (float *)malloc(item_size);

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // Allocate the device input vector A
    checkCudaErrors(cuMemAlloc(&d_A, item_size));
    checkCudaErrors(cuMemAlloc(&d_B, item_size));
    checkCudaErrors(cuMemAlloc(&d_C, item_size));

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    checkCudaErrors(cuMemcpyHtoD(d_A, h_A, item_size));
    checkCudaErrors(cuMemcpyHtoD(d_B, h_B, item_size));

#endif

    img = createImage(width, height);
    img_content = (unsigned char*)malloc(width * height * 4);
}

void draw()
{
#if 0
    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    dim3 cudaBlockSize = { threadsPerBlock, 1, 1 };
    dim3 cudaGridSize = { blocksPerGrid, 1, 1 };

    void *arr[] = { (void *)&d_A, (void *)&d_B, (void *)&d_C, (void *)&numElements };
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
    checkCudaErrors(cuMemcpyDtoH(h_C, d_C, item_size));

    // Verify that the result vector is correct
    for (int i = 0; i < numElements; ++i)
    {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
#endif
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            unsigned char* rgba = img_content + (y * width + x) * 4;
            rgba[0] = rand() % 255;
            rgba[1] = rand() % 255;
            rgba[2] = rand() % 255;
            rgba[3] = 255;
        }
    }
    updateImage(img, img_content);
    image(img, 0, 0, width, height);

    ellipse(mouseX, mouseY, 10, 10);
}

void shutdown()
{
#if 0
    // Free device global memory
    checkCudaErrors(cuMemFree(d_A));
    checkCudaErrors(cuMemFree(d_B));
    checkCudaErrors(cuMemFree(d_C));
    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    cuProfilerStop();
#endif

    free(img_content);

    printf("Done\n");
}
