#include "./conv_gpu.h"


void Conv_GPU::init()
{
    height_out = (1 + (height_in - height_kernel + 2 * pad_h) / stride);
    width_out = (1 + (width_in - width_kernel + 2 * pad_w) / stride);
    dim_out = height_out * width_out * channel_out;

    weight.resize(channel_in * height_kernel * width_kernel, channel_out);
    bias.resize(channel_out);
    grad_weight.resize(channel_in * height_kernel * width_kernel, channel_out);
    grad_bias.resize(channel_out);
    set_normal_random(weight.data(), weight.size(), 0, 0.01);
    set_normal_random(bias.data(), bias.size(), 0, 0.01);
}

void Conv_GPU::forward(const Matrix &bottom)
{
    int n_sample = bottom.cols();
    top.resize(height_out * width_out * channel_out, n_sample);
    float *input_data = (float *)bottom.data();
    float *output_data = (float *)top.data();
    float *weight_data = (float *)weight.data();

    const int num_samples = n_sample;
    const int input_channel = channel_in;
    const int output_channel = channel_out;
    const int kernel_height = height_kernel; // Assuming width_kernel is also K

    if(input_channel == 1)
        std::cout << "Convolution c1 - GPU";
    else
        std::cout << "Convolution c3 - GPU";

    // Start layer timer
    GpuTimer timer;
	  timer.Start();
    gpuintergace.conv_forward_gpu(output_data, input_data, weight_data,
                                    num_samples, output_channel, input_channel,
                                    height_in, width_in, kernel_height);

    // Stop layer timer
    timer.Stop();
	  float duration_layer = timer.Elapsed();
    
    std::cout << "\t - Layer Time: " << duration_layer << " ms" << std::endl;
 
}

void Conv_GPU::backward(const Matrix &bottom, const Matrix &grad_top)
{
    
}

void Conv_GPU::update(Optimizer &opt)
{
    
}

std::vector<float> Conv_GPU::get_parameters() const
{
    std::vector<float> res(weight.size() + bias.size());
    // Copy the data of weights and bias to a long vector
    std::copy(weight.data(), weight.data() + weight.size(), res.begin());
    std::copy(bias.data(), bias.data() + bias.size(), res.begin() + weight.size());
    return res;
}

void Conv_GPU::set_parameters(const std::vector<float> &param)
{
    if (static_cast<int>(param.size()) != weight.size() + bias.size())
        throw std::invalid_argument("Parameter size does not match");
    std::copy(param.begin(), param.begin() + weight.size(), weight.data());
    std::copy(param.begin() + weight.size(), param.end(), bias.data());
}

std::vector<float> Conv_GPU::get_derivatives() const
{
    std::vector<float> res(grad_weight.size() + grad_bias.size());
    // Copy the data of weights and bias to a long vector
    std::copy(grad_weight.data(), grad_weight.data() + grad_weight.size(), res.begin());
    std::copy(grad_bias.data(), grad_bias.data() + grad_bias.size(),
              res.begin() + grad_weight.size());
    return res;
}
