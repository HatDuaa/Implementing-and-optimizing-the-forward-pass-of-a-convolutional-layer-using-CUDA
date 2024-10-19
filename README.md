# Implementing and Optimizing the Forward Pass of a Convolutional Layer using CUDA

This project modifies the mini-dnn-cpp framework to implement the LeNet-5 architecture and run it on the Fashion MNIST dataset. The focus is on optimizing the GPU convolution kernel with techniques such as:

- Tiled shared memory convolution.
- Shared memory matrix multiplication and input matrix unrolling.
- Weight matrix (kernel values) stored in constant memory.
- Additional optimizations as needed.

## Prerequisites
- [Google Colab](https://colab.research.google.com/) (with GPU enabled)

## Usage
To run this project, simply execute the "main.ipynb" notebook in [Google Colab GPU](https://colab.research.google.com/drive/16B4blnAH9ewjb7Ejxnxi542fA1YgHwnY?usp=sharing). You will be able to visualize the results of the LeNet-5 architecture on the Fashion MNIST dataset.

## Contributing
If you would like to contribute to this project, please fork the repository and submit a pull request.
