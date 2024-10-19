#ifndef SRC_LAYER_CUSTOM_GPU_UTILS_H
#define SRC_LAYER_CUSTOM_GPU_UTILS_H
#pragma once

#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#define CHECK(call)\
{\
	const cudaError_t error = call;\
	if (error != cudaSuccess)\
	{\
		fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
		fprintf(stderr, "code: %d, reason: %s\n", error,\
				cudaGetErrorString(error));\
		exit(EXIT_FAILURE);\
	}\
}

struct GpuTimer
{
	cudaEvent_t start;
	cudaEvent_t stop;

	GpuTimer()
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}

	~GpuTimer()
	{
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	void Start()
	{
		cudaEventRecord(start, 0);
		cudaEventSynchronize(start);
	}

	void Stop()
	{
		cudaEventRecord(stop, 0);
	}

	float Elapsed()
	{
		float elapsed;
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed, start, stop);
		return elapsed;
	}
};

class GPU_Utils
{
public:
	char *concatStr(const char *s1, const char *s2);
	void printDeviceInfo();
  void insert_post_barrier_kernel();
  void insert_pre_barrier_kernel();
};

#endif