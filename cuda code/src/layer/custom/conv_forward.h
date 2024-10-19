#ifndef SRC_LAYER_CUSTOM_CONV_FORWARD_H
#define SRC_LAYER_CUSTOM_CONV_FORWARD_H

#include <cmath>
#include <iostream>

#include "./gpu_utils.h"

class GPUInterface
{
    public:
    void conv_forward_gpu(float *host_y, const float *host_x, const float *host_k,
                                                  const int B, const int M, const int C,
                                                  const int H, const int W, const int K);
};

#endif