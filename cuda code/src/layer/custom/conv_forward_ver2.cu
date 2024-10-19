#include "./conv_forward.h"
#include <cmath>
#include <iostream>

#define TILE_WIDTH 16

__constant__ float dc_filter[2400];

__global__ void conv_forward_kernel(float *out, const float *in, const int num_samples,
                                        const int num_output_channels, const int num_input_channels,
                                        const int h_in, const int w_in, const int filter_size)
{
    const int w_out = w_in - filter_size + 1;
    const int h_out = h_in - filter_size + 1;

    int width_grid = ceil(1.0 * w_out / TILE_WIDTH);
    int height_grid = ceil(1.0 * h_out / TILE_WIDTH);

    int i_batch = blockIdx.x;        // batch number
    int i_out_channel = blockIdx.y; // output channel
    int row = (blockIdx.z / width_grid) * TILE_WIDTH + threadIdx.y; // row of the output matrix
    int col = (blockIdx.z % width_grid) * TILE_WIDTH + threadIdx.x; // column of the output matrix

    #define o4d(i3, i2, i1, i0) out[(i3) * (num_output_channels * h_out * w_out) + (i2) * (h_out * w_out) + (i1) * (w_out) + i0]
    #define i4d(i3, i2, i1, i0) in[(i3) * (num_input_channels * h_in * w_in) + (i2) * (h_in * w_in) + (i1) * (w_in) + i0]
    #define f4d(i3, i2, i1, i0) dc_filter[(i3) * (num_input_channels * filter_size * filter_size) + (i2) * (filter_size * filter_size) + (i1) * (filter_size) + i0]

    if (row < h_out && col < w_out) 
    {
        float sum = 0.0f;
        for(int input_channel_idx = 0; input_channel_idx < num_input_channels; input_channel_idx++)   // sum over all input channels
        {
            for(int filter_row = 0; filter_row < filter_size; filter_row++) // filter_size x filter_size filter 
            {
                for(int filter_col = 0; filter_col < filter_size; filter_col++)
                {
                    int i_row = row + filter_row;
                    int i_col = col + filter_col;
                    sum += i4d(i_batch, input_channel_idx, i_row, i_col) * 
                                f4d(i_out_channel, input_channel_idx, filter_row, filter_col);
                }
            }
        }
        o4d(i_batch, i_out_channel, row, col) = sum;
    }
    #undef o4d
    #undef i4d
    #undef f4d
}


__host__ void GPUInterface::conv_forward_gpu(float *output, const float *input, const float *weight_data,
                                                  const int num_samples, const int output_channel, const int input_channel,
                                                  const int height_in, const int width_in, const int kernel_height)
{
    const int height_out = height_in - kernel_height + 1;
    const int width_out = width_in - kernel_height + 1;

    // Allocate device memory
    float *device_input, *device_output;
    cudaMalloc((void **)&device_input, num_samples * input_channel * height_in * width_in * sizeof(float));     // input features map is input_channel
    cudaMalloc((void **)&device_output, num_samples * output_channel * height_out * width_out * sizeof(float)); // output feature map is output_channel

    // Copy input and mask data to device
    cudaMemcpy(device_input, input, num_samples * input_channel * height_in * width_in * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(dc_filter, weight_data, output_channel * input_channel * kernel_height * kernel_height * sizeof(float));

    // Set the kernel dimensions and call the kernel
    int Z = ceil(1.0 * height_out / TILE_WIDTH) * ceil(1.0 * width_out / TILE_WIDTH);
    dim3 num_threads_per_block(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 num_blocks_in_grid(num_samples, output_channel, Z);

    std::cout<<"\nGPU custom version 2:\n";
    // Launch the kernel
    conv_forward_kernel<<<num_blocks_in_grid, num_threads_per_block>>>(device_output, device_input, num_samples, output_channel, input_channel, height_in, width_in, kernel_height);

    // Copy the output back to host
    cudaMemcpy(output, device_output, num_samples * output_channel * height_out * width_out * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_input);
    cudaFree(device_output);
}
