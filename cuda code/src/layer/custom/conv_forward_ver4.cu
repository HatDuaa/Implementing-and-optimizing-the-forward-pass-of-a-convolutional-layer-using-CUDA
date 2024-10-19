#include "./conv_forward.h"
#include <cmath>
#include <iostream>

#define TILE_WIDTH_C1 16
#define TILE_WIDTH_C3 12

__constant__ float dc_filter[2400];


__global__ void conv_forward_kernel(float *out, const float *in, const int num_samples,
                                    const int num_output_channels, const int num_input_channels,
                                    const int h_in, const int w_in, const int filter_size)
{
    extern __shared__ float shared_input[];

    int TILE_WIDTH;    
    if (num_input_channels == 1){
        TILE_WIDTH = TILE_WIDTH_C1;
    }
    else{
        TILE_WIDTH = TILE_WIDTH_C3;
    }
    
    const int INPUT_TILE = TILE_WIDTH + filter_size - 1;
    
    const int h_out = h_in - filter_size + 1;
    const int w_out = w_in - filter_size + 1;


    #define o4d(i3, i2, i1, i0) out[(i3) * (num_output_channels * h_out * w_out) + (i2) * (h_out * w_out) + (i1) * (w_out) + i0]
    #define i4d(i3, i2, i1, i0) in[(i3) * (num_input_channels * h_in * w_in) + (i2) * (h_in * w_in) + (i1) * (w_in) + i0]
    #define f4d(i3, i2, i1, i0) dc_filter[(i3) * (num_input_channels * filter_size * filter_size) + (i2) * (filter_size * filter_size) + (i1) * (filter_size) + i0]
    #define sm3d(i2, i1, i0) shared_input[(i2) * (INPUT_TILE * INPUT_TILE) + (i1) * INPUT_TILE + i0]

    int w_grid = ceil(1.0*w_out / TILE_WIDTH); 
    int i_batch = blockIdx.x;                 // batch number
    int i_out_channel = blockIdx.y;           // output feature
    
    int ty = threadIdx.y;              // thread ID in the current TILE  
    int tx = threadIdx.x;
    
    int row_in = (blockIdx.z / w_grid) * TILE_WIDTH; // row of the input image matrix
    int col_in = (blockIdx.z % w_grid) * TILE_WIDTH; // col of the input image matrix
    
    int row_out = (blockIdx.z / w_grid) * TILE_WIDTH + ty; // row of the output image matrix
    int col_out = (blockIdx.z % w_grid) * TILE_WIDTH + tx; // col of the ouput image matrix    

    #pragma unroll
    for (int input_channel_idx = 0; input_channel_idx < num_input_channels; input_channel_idx++)
    {
        #pragma unroll
        for(int i = ty; i < INPUT_TILE; i += TILE_WIDTH)
        {
            #pragma unroll
            for(int j = tx; j < INPUT_TILE; j += TILE_WIDTH)
            {
                if (row_in + i < h_in && col_in + j < w_in)
                {
                    sm3d(input_channel_idx, i, j) = i4d(i_batch, input_channel_idx, row_in + i, col_in + j);
                }
            }
        }
    }

    // Make sure all threads loaded data into shared memory
    __syncthreads();

    // compute only within bounds
    if ((row_out < h_out) && (col_out < w_out)) 
    {
        float sum = 0.0f;
        #pragma unroll
        for(int input_channel_idx = 0; input_channel_idx < num_input_channels; input_channel_idx++)             // sum over all input features
        {
            #pragma unroll
            for(int p=0; p< filter_size; p++)         // KxK filter 
                #pragma unroll
                for(int q=0; q< filter_size; q++)
                    sum += sm3d(input_channel_idx, p + ty, q+tx) * f4d(i_out_channel, input_channel_idx, p, q); 
        }
        o4d(i_batch,i_out_channel,row_out,col_out) = sum;
    } 
    
    #undef sm3d
    #undef o4d
    #undef i4d
    #undef f4d
}

__host__ void GPUInterface::conv_forward_gpu(float *output_data, const float *input_data, const float *weight_data,
                                                  const int num_samples, const int output_channel, const int num_input_channels,
                                                  const int height_in, const int width_in, const int kernel_height)
{
    // Set the tile width
    int TILE_WIDTH;    
    if (num_input_channels == 1){
        TILE_WIDTH = TILE_WIDTH_C1;
    }
    else{
        TILE_WIDTH = TILE_WIDTH_C3;
    }
    

    // Allocate memory and copy over the relevant data structures to the GPU
    const int h_out = height_in - kernel_height + 1;
    const int w_out = width_in - kernel_height + 1;

    int inputSize = num_samples * num_input_channels * height_in * width_in * sizeof(float);
    int outputSize = num_samples * output_channel * h_out * w_out * sizeof(float);
    int filter_size = output_channel * num_input_channels * kernel_height * kernel_height * sizeof(float);

    float *device_input, *device_output, *device_kernel;

    cudaMalloc((void **)&device_input, inputSize);
    cudaMalloc((void **)&device_output, outputSize);
    cudaMalloc((void **)&device_kernel, filter_size);

    cudaMemcpy(device_input, input_data, inputSize, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(dc_filter, weight_data, filter_size);

    dim3 numThreadsPerBlock, numBlocksInGrid;

    int Z = ceil(1.0 * h_out / TILE_WIDTH) * ceil(1.0 * w_out / TILE_WIDTH);
    numThreadsPerBlock = dim3(TILE_WIDTH, TILE_WIDTH, 1);
    int shmem_size = num_input_channels * (TILE_WIDTH + kernel_height - 1) * (TILE_WIDTH + kernel_height - 1) * sizeof(float);
    numBlocksInGrid = dim3(num_samples, output_channel, Z);
    
    std::cout<<"\nGPU custom version 4:\n";
    // Launch the kernel
    conv_forward_kernel<<<numBlocksInGrid, numThreadsPerBlock, shmem_size>>>(device_output, device_input, num_samples, output_channel, num_input_channels, height_in, width_in, kernel_height);

    // Copy the output back to host
    cudaMemcpy(output_data, device_output, outputSize, cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_kernel);

}
