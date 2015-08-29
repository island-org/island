#define SKETCH_2D_IMPLEMENTATION
#include "sketch2d.h"

#include "CudaApi.h"
#include "Remotery.h"

CUdeviceptr d_img_content;
CUfunction kernel_addr;
PImage img;
unsigned char* img_content;
size_t item_size;
CUdeviceptr d_iResolution, d_iGlobalTime, d_iMouse;

Remotery *rmt;

void setupResource()
{
    item_size = width * height * 4;

    img = createImage(width, height);
    img_content = (unsigned char*)malloc(item_size);
    checkCudaErrors(cuMemAlloc(&d_img_content, item_size));
}

void setup()
{
    if (RMT_ERROR_NONE != rmt_CreateGlobalInstance(&rmt)) {
        //return -1;
    }

    glfwSetWindowTitle(window, "CUDA ShaderToy");

    checkCudaErrors(cuInit(0));

    CUmodule module = createModuleFromFile("../examples/09-cuda-shadertoy/random.cu");
    checkCudaErrors(cuModuleGetFunction(&kernel_addr, module, "mainImage"));

    // TODO: take care of bytes
    size_t bytes;
    checkCudaErrors(cuModuleGetGlobal(&d_iResolution, &bytes, module, "iResolution"));
    checkCudaErrors(cuModuleGetGlobal(&d_iGlobalTime, &bytes, module, "iGlobalTime"));
    checkCudaErrors(cuModuleGetGlobal(&d_iMouse, &bytes, module, "iMouse"));

    rmtCUDABind bind;
    bind.context = 0;
    bind.CtxSetCurrent = &cuCtxSetCurrent;
    bind.CtxGetCurrent = &cuCtxGetCurrent;
    bind.EventCreate = &cuEventCreate;
    bind.EventDestroy = &cuEventDestroy;
    bind.EventRecord = &cuEventRecord;
    bind.EventQuery = &cuEventQuery;
    bind.EventElapsedTime = &cuEventElapsedTime;
    rmt_BindCUDA(&bind);

    rmt_BindOpenGL();

    setupResource();
}

void draw()
{
    rmt_LogText("start profiling");

    rmt_BeginCUDASample(main, 0);
    {
        if (isResized())
        {
            deleteImage(img);
            free(img_content);
            checkCudaErrors(cuMemFree(d_img_content));
            setupResource();
        }
        // Launch the Vector Add CUDA Kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (img.width * img.height + threadsPerBlock - 1) / threadsPerBlock;
        //printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
        dim3 blockDim = { 32, 32, 1 };
        dim3 gridDim = { width / blockDim.x, height / blockDim.y, 1 };

        float3 iResolution = { width, height, 1 };
        float iGlobalTime = glfwGetTime();
        float4 iMouse = { mouseX, mouseY, mouseX, mouseY };
        checkCudaErrors(cuMemcpyHtoD(d_iResolution, &iResolution, sizeof iResolution));
        checkCudaErrors(cuMemcpyHtoD(d_iGlobalTime, &iGlobalTime, sizeof iGlobalTime));
        checkCudaErrors(cuMemcpyHtoD(d_iMouse, &iMouse, sizeof iMouse));

        void *arr[] = { (void *)&d_img_content };
        checkCudaErrors(cuLaunchKernel(kernel_addr,
            gridDim.x, gridDim.y, gridDim.z, /* grid dim */
            blockDim.x, blockDim.y, blockDim.z, /* block dim */
            0, 0, /* shared mem, stream */
            &arr[0], /* arguments */
            0));
        checkCudaErrors(cuCtxSynchronize());

        checkCudaErrors(cuMemcpyDtoH(img_content, d_img_content, item_size));
    }
    rmt_EndCUDASample(0);

    rmt_BeginOpenGLSample(main);
    {
        updateImage(img, img_content);
        image(img, 0, 0, width, height);
    }
    rmt_EndOpenGLSample();

    rmt_UpdateOpenGLFrame();

    rmt_LogText("end profiling");
}

void teardown()
{
    // Free device global memory
    checkCudaErrors(cuMemFree(d_img_content));
    cuProfilerStop();

    free(img_content);

    rmt_UnbindOpenGL();
    rmt_DestroyGlobalInstance(rmt);
}
