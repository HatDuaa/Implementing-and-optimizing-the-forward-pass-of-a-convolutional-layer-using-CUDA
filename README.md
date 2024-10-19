# Implementing and optimizing the forward pass of a convolutional layer using CUDA
Modify the project mini-dnn-cpp to change the network architecture to LeNet-5 and run on Fashion MNIST. Optimize GPU Convolution kernel such as:
- Tiled shared memory convolution.
- Shared memory matrix multiplication and input matrix unrolling.
- Weight matrix (kernel values) in constant memory.
- ...

To run this project, you just run "main.ipynb" in [google colab GPU]([url](https://colab.research.google.com/drive/16B4blnAH9ewjb7Ejxnxi542fA1YgHwnY?usp=sharing)).
