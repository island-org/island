/*
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#ifndef _DRVAPI_ERROR_STRING_H_
#define _DRVAPI_ERROR_STRING_H_

// TODO: add CUDA_API_IMPLEMENTATION macro

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cuda.h>
#include <cudaProfiler.h>
#include <nvrtc.h>

const char *getCudaDrvErrorString(CUresult error_id);

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

void compileFileToPTX(const char *filename, int argc, const char **argv,
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
    char* memBlock = (char*)malloc(inputSize + 1);
    fseek(fp, 0, SEEK_SET);
    fread(memBlock, sizeof(char), inputSize, fp);
    memBlock[inputSize] = '\x0';
    fclose(fp);

    // compile
    nvrtcProgram prog;
    checkCudaRtcErrors(nvrtcCreateProgram(&prog, memBlock,
        filename, 0, NULL, NULL));
    nvrtcResult result = nvrtcCompileProgram(prog, argc, argv);

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

CUmodule createModuleFromFile(const char* kernel_file)
{
    char* ptx;
    size_t ptxSize;
    // TODO: cache the PTX
    compileFileToPTX(kernel_file, 0, NULL, &ptx, &ptxSize);
    CUmodule module = loadPTX(ptx);

    return module;
}

const char *getCudaDrvErrorString(CUresult error_id)
{
    int index = 0;
    static char* errorString;
    cuGetErrorName(error_id, &errorString);
    return errorString;
}

#endif
