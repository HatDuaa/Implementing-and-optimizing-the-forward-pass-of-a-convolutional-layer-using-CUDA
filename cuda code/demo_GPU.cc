#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <cstdlib>

#include "src/layer.h"
#include "src/layer/conv.h"
#include "src/layer/conv_gpu.h"
#include "src/layer/fully_connected.h"
#include "src/layer/ave_pooling.h"
#include "src/layer/max_pooling.h"
#include "src/layer/relu.h"
#include "src/layer/sigmoid.h"
#include "src/layer/softmax.h"
#include "src/loss.h"
#include "src/loss/mse_loss.h"
#include "src/loss/cross_entropy_loss.h"
#include "src/mnist.h"
#include "src/network.h"
#include "src/optimizer.h"
#include "src/optimizer/sgd.h"
#include "src/layer/custom/gpu_utils.h"


// Design lenet-5 with Conv layer on host
Network Network_lenet5_host()
{
    Network Lenet5_host;
    Layer *conv1 = new Conv(1, 28, 28, 6, 5, 5);
    Layer *pool1 = new MaxPooling(6, 24, 24, 2, 2, 2);
    Layer *conv2 = new Conv(6, 12, 12, 16, 5, 5);
    Layer *pool2 = new MaxPooling(16, 8, 8, 2, 2, 2);
    Layer *fc1 = new FullyConnected(pool2->output_dim(), 120);
    Layer *fc2 = new FullyConnected(120, 84);
    Layer *fc3 = new FullyConnected(84, 10);
    Layer *relu_conv1 = new ReLU;
    Layer *relu_conv2 = new ReLU;
    Layer *relu_fc1 = new ReLU;
    Layer *relu_fc2 = new ReLU;
    Layer *softmax = new Softmax;
    Lenet5_host.add_layer(conv1);
    Lenet5_host.add_layer(relu_conv1);
    Lenet5_host.add_layer(pool1);
    Lenet5_host.add_layer(conv2);
    Lenet5_host.add_layer(relu_conv2);
    Lenet5_host.add_layer(pool2);
    Lenet5_host.add_layer(fc1);
    Lenet5_host.add_layer(relu_fc1);
    Lenet5_host.add_layer(fc2);
    Lenet5_host.add_layer(relu_fc2);
    Lenet5_host.add_layer(fc3);
    Lenet5_host.add_layer(softmax);
    // loss
    Loss *loss = new CrossEntropy;
    Lenet5_host.add_loss(loss);
    return Lenet5_host;
}

// Design lenet-5 with Conv layer on device
Network Network_lenet5_device()
{
    Network Lenet5_device;
    Layer *conv1 = new Conv_GPU(1, 28, 28, 6, 5, 5);
    Layer *pool1 = new MaxPooling(6, 24, 24, 2, 2, 2);
    Layer *conv2 = new Conv_GPU(6, 12, 12, 16, 5, 5);
    Layer *pool2 = new MaxPooling(16, 8, 8, 2, 2, 2);
    Layer *fc1 = new FullyConnected(pool2->output_dim(), 120);
    Layer *fc2 = new FullyConnected(120, 84);
    Layer *fc3 = new FullyConnected(84, 10);
    Layer *relu_conv1 = new ReLU;
    Layer *relu_conv2 = new ReLU;
    Layer *relu_fc1 = new ReLU;
    Layer *relu_fc2 = new ReLU;
    Layer *softmax = new Softmax;
    Lenet5_device.add_layer(conv1);
    Lenet5_device.add_layer(relu_conv1);
    Lenet5_device.add_layer(pool1);
    Lenet5_device.add_layer(conv2);
    Lenet5_device.add_layer(relu_conv2);
    Lenet5_device.add_layer(pool2);
    Lenet5_device.add_layer(fc1);
    Lenet5_device.add_layer(relu_fc1);
    Lenet5_device.add_layer(fc2);
    Lenet5_device.add_layer(relu_fc2);
    Lenet5_device.add_layer(fc3);
    Lenet5_device.add_layer(softmax);
    // loss
    Loss *loss = new CrossEntropy;
    Lenet5_device.add_loss(loss);

    return Lenet5_device;
}

// Train data using lenet-5 model and save weights on build folder
void train()
{
  // data
  MNIST dataset("./data/fashion/");
  dataset.read();
  int n_train = dataset.train_data.cols();
  int dim_in = dataset.train_data.rows();
  std::cout << "mnist train number: " << n_train << std::endl;
  std::cout << "mnist test number: " << dataset.test_labels.cols() << std::endl;
  // Lenet-5
  Network Lenet5 = Network_lenet5_host();
  // train & test
  SGD opt(0.001, 5e-4, 0.9, true);
  // SGD opt(0.001);
  const int n_epoch = 5;
  const int batch_size = 128;
  for (int epoch = 0; epoch < n_epoch; epoch++)
  {
    Lenet5.save_parameters("./build/save-weights.bin");
    shuffle_data(dataset.train_data, dataset.train_labels);
    for (int start_idx = 0; start_idx < n_train; start_idx += batch_size)
    {
      int ith_batch = start_idx / batch_size;
      Matrix x_batch = dataset.train_data.block(0, start_idx, dim_in,
                                                std::min(batch_size, n_train - start_idx));
      Matrix label_batch = dataset.train_labels.block(0, start_idx, 1,
                                                      std::min(batch_size, n_train - start_idx));
      Matrix target_batch = one_hot_encode(label_batch, 10);
      if (false && ith_batch % 10 == 1)
      {
        std::cout << ith_batch << "-th grad: " << std::endl;
        Lenet5.check_gradient(x_batch, target_batch, 10);
      }
      Lenet5.forward(x_batch);
      Lenet5.backward(x_batch, target_batch);
      // display
      if (ith_batch % 50 == 0)
      {
        std::cout << ith_batch << "-th batch, loss: " << Lenet5.get_loss()
                  << std::endl;
      }
      // optimize
      Lenet5.update(opt);
    }
    // test
    Lenet5.forward(dataset.test_data);
    float acc = compute_accuracy(Lenet5.output(), dataset.test_labels);
    std::cout << std::endl;
    std::cout << epoch + 1 << "-th epoch, test acc: " << acc << std::endl;
    std::cout << std::endl;
  }
  Lenet5.save_parameters("./build/save-weights.bin");
}

// Test model lenet-5 model on host and device
void test()
{
    // Print device info
    GPU_Utils gpu_utils;
    gpu_utils.printDeviceInfo();    
    
    // data
    MNIST dataset("./data/fashion/");
    dataset.read();
    int n_train = dataset.train_data.cols();
    int dim_in = dataset.train_data.rows();
    std::cout << "mnist test number: " << dataset.test_labels.cols() << std::endl;
    std::cout << "==============================" << std::endl;


    // device
    std::cout << "Test using Lenet-5 on device:" << std::endl;
    Network Lenet5_device = Network_lenet5_device();
    Lenet5_device.load_parameters("./build/save-weights.bin");

    Lenet5_device.forward(dataset.test_data);
    float acc_device = compute_accuracy(Lenet5_device.output(), dataset.test_labels);
    std::cout << "test acc: " << acc_device << std::endl;
    std::cout << "==============================" << std::endl;
}

//--------------------------------------MAIN------------------------------------
int main() {
    test();

    return 0;
}