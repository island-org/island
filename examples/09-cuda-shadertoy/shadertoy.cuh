#ifndef _SHADER_TOY_CUH_

__constant__ float3	iResolution;	// image	The viewport resolution (z is pixel aspect ratio, usually 1.0)
__constant__ float	iGlobalTime;	// image/sound	Current time in seconds
// __constant__ float	iChannelTime[4];	// image	Time for channel (if video or sound), in seconds
// __constant__ float3	iChannelResolution0..3	image/sound	Input texture resolution for each channel
__constant__ float4	iMouse;	// image	xy = current pixel coords (if LMB is down). zw = click pixel
// __constant__ sampler2D	iChannel{i}	// image/sound	Sampler for input textures i
// __constant__ float4	iDate;	// image/sound	Year, month, day, time in seconds in .xyzw
// __constant__ float	iSampleRate;	// image/sound	The sound sample rate (typically 44100)

__device__ uchar4* fragColor;

__device__ float2 calcFragCoord()
{
	float2 coord = { blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y};
	return coord;
}

__device__ uchar4* calcFragColor(float2 fragCoord)
{
    int idx = fragCoord.y * iResolution.x + fragCoord.x;
    return fragColor + idx;
}

#endif
