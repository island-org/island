#define SKETCH_2D_IMPLEMENTATION
#include "sketch2d.h"

#include <AntTweakBar.h>

#include "CudaApi.h"
#include "Remotery.h"

#include "uv.h"
uv_fs_event_t fs_event;

CUfunction kernel_addr;
PImage img;
unsigned char* img_content;
size_t item_size;
CUdeviceptr d_iResolution, d_iGlobalTime, d_iMouse;
CUdeviceptr d_img_content;
CUdeviceptr d_fragColor;
size_t d_fragColor_bytes;

CUmodule module;
Remotery *rmt;
CUdevice cuDevice = 0;

GLFWwindow* barWindow;
TwBar* bar;

void WindowSizeCB(GLFWwindow* window, int width, int height)
{
    TwWindowSize(width, height);
}

void setupModuleResource(const char* kernelFileName)
{
    CUmodule newModule = createModuleFromFile(kernelFileName);
    if (newModule != NULL)
    {
        if (module != NULL) cuModuleUnload(module);
        module = newModule;
    }
    checkCudaErrors(cuModuleGetFunction(&kernel_addr, module, "mainImage"));

    // TODO: take care of bytes
    size_t bytes;
    checkCudaErrors(cuModuleGetGlobal(&d_iResolution, &bytes, module, "iResolution"));
    checkCudaErrors(cuModuleGetGlobal(&d_iGlobalTime, &bytes, module, "iGlobalTime"));
    checkCudaErrors(cuModuleGetGlobal(&d_iMouse, &bytes, module, "iMouse"));
    checkCudaErrors(cuModuleGetGlobal(&d_fragColor, &d_fragColor_bytes, module, "fragColor"));
}

void on_fs_event_cb(uv_fs_event_t *handle, const char *filename, int events, int status)
{
    char path[1024];
    size_t size = 1023;
    // Does not handle error if path is longer than 1023.
    uv_fs_event_getpath(handle, path, &size);
    path[size] = '\0';

    printf("%s is ", filename ? filename : "");
    if (events & UV_RENAME)
        printf("renamed\n");
    if (events & UV_CHANGE)
        printf("changed\n");

    setupModuleResource(path);

    // TODO: so ugly
    checkCudaErrors(cuMemcpyHtoD(d_fragColor, &d_img_content, d_fragColor_bytes));
}

void setupSizeResource()
{
    deleteImage(img);
    free(img_content);
    checkCudaErrors(cuMemFree(d_img_content));

    item_size = width * height * 4;

    img = createImage(width, height);
    img_content = (unsigned char*)malloc(item_size);
    checkCudaErrors(cuMemAlloc(&d_img_content, item_size));
    checkCudaErrors(cuMemcpyHtoD(d_fragColor, &d_img_content, d_fragColor_bytes));
}

void setupAntTweakBar()
{
    TwInit(TW_OPENGL_CORE, NULL);
#if 0
    barWindow = glfwCreateWindow(400, 400, "Param", NULL, NULL);
    TwSetCurrentWindow(barWindow);
#endif
    bar = TwNewBar("Param");
    WindowSizeCB(window, width, height);

    glfwSetWindowSizeCallback(window, WindowSizeCB);
    glfwSetMouseButtonCallback(window, TwEventMouseButtonGLFW);
    glfwSetCursorPosCallback(window, TwEvenCursorPosGLFW);
    glfwSetScrollCallback(window, TwEventScrollGLFW);
    glfwSetKeyCallback(window, TwEventKeyGLFW);
    glfwSetCharCallback(window, TwEventCharGLFW);

    TwDefine(" GLOBAL help='This example shows how to integrate AntTweakBar with GLFW and OpenGL.' "); // Message added to the help bar.

}

void setup()
{
    setupAntTweakBar();

    checkCudaErrors(cuInit(0));

    if (sketchArgc != 2)
    {
        printf("Usage: %s <cuda_toy.cu>\n", sketchArgv[0]);
        quit();
        return;
    }
    if (RMT_ERROR_NONE != rmt_CreateGlobalInstance(&rmt)) {
        //return -1;
    }
    
    int r = uv_fs_event_init(uv_default_loop(), &fs_event);
    r = uv_fs_event_start(&fs_event, on_fs_event_cb, sketchArgv[1], 0);

    char title[256];
    sprintf(title, "CUDA ShaderToy - %s", sketchArgv[1]);
    glfwSetWindowTitle(window, title);

    setupModuleResource(sketchArgv[1]);
    setupSizeResource();

    rmtCUDABind bind;
    bind.context = cuContext;
    bind.CtxSetCurrent = &cuCtxSetCurrent;
    bind.CtxGetCurrent = &cuCtxGetCurrent;
    bind.EventCreate = &cuEventCreate;
    bind.EventDestroy = &cuEventDestroy;
    bind.EventRecord = &cuEventRecord;
    bind.EventQuery = &cuEventQuery;
    bind.EventElapsedTime = &cuEventElapsedTime;
    rmt_BindCUDA(&bind);

    rmt_BindOpenGL();
}

void draw()
{
    rmt_LogText("start profiling");

    //rmt_BeginCPUSample(uv_run);
    uv_run(uv_default_loop(), UV_RUN_NOWAIT);
    //rmt_EndCPUSample();

    CUstream stream0 = 0;
    rmt_BeginCUDASample(main, stream0);
    {
        if (isResized())
        {
            setupSizeResource();
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
        rmt_BeginCUDASample(cuMemcpyHtoD, stream0);
        checkCudaErrors(cuMemcpyHtoD(d_iResolution, &iResolution, sizeof iResolution));
        checkCudaErrors(cuMemcpyHtoD(d_iGlobalTime, &iGlobalTime, sizeof iGlobalTime));
        checkCudaErrors(cuMemcpyHtoD(d_iMouse, &iMouse, sizeof iMouse));
        rmt_EndCUDASample(stream0);

        rmt_BeginCUDASample(cuLaunchKernel, stream0);
        checkCudaErrors(cuLaunchKernel(kernel_addr,
            gridDim.x, gridDim.y, gridDim.z, /* grid dim */
            blockDim.x, blockDim.y, blockDim.z, /* block dim */
            0, 0, /* shared mem, stream */
            0, /* arguments */
            0));
        rmt_EndCUDASample(stream0);
        rmt_BeginCUDASample(cuCtxSynchronize, stream0);
        checkCudaErrors(cuCtxSynchronize());
        rmt_EndCUDASample(stream0);
        
        rmt_BeginCUDASample(cuMemcpyDtoH, stream0);
        checkCudaErrors(cuMemcpyDtoH(img_content, d_img_content, item_size));
        rmt_EndCUDASample(stream0);
    }
    rmt_EndCUDASample(stream0);

    rmt_BeginOpenGLSample(main);
    {
        background(color(0,0,0));
        updateImage(img, img_content);
        image(img, 0, 0, width, height);

        TwDraw();
    }
    rmt_EndOpenGLSample();

    rmt_LogText("end profiling");
}

void teardown()
{
    TwTerminate();

    // Free device global memory
    checkCudaErrors(cuMemFree(d_img_content));
    //cuProfilerStop();

    free(img_content);

    rmt_UnbindOpenGL();
    rmt_DestroyGlobalInstance(rmt);

    cuDevicePrimaryCtxRelease(cuDevice);

    cudaDeviceReset();
}
