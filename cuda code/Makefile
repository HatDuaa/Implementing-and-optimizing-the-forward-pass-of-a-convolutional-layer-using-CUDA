demo: demo.o 
	nvcc -arch=sm_75 -o demo -lm -lcuda -lrt demo.o src/network.o src/mnist.o src/layer/*.o src/loss/*.o src/optimizer/*.o src/layer/custom/*.o -I./ -L/usr/local/cuda/lib64 -lcudart

demo.o: demo.cc
	nvcc -arch=sm_75 --compile demo.cc -I./ -L/usr/local/cuda/lib64 -lcudart

demo_CPU: demo_CPU.o 
	nvcc -arch=sm_75 -o demo_CPU -lm -lcuda -lrt demo_CPU.o src/network.o src/mnist.o src/layer/*.o src/loss/*.o src/optimizer/*.o src/layer/custom/*.o -I./ -L/usr/local/cuda/lib64 -lcudart

demo_CPU.o: demo_CPU.cc
	nvcc -arch=sm_75 --compile demo_CPU.cc -I./ -L/usr/local/cuda/lib64 -lcudart

demo_GPU: demo_GPU.o
	nvcc -arch=sm_75 -o demo_GPU -lm -lcuda -lrt demo_GPU.o src/network.o src/mnist.o src/layer/*.o src/loss/*.o src/optimizer/*.o src/layer/custom/*.o -I./ -L/usr/local/cuda/lib64 -lcudart

demo_GPU.o: demo_GPU.cc
	nvcc -arch=sm_75 --compile demo_GPU.cc -I./ -L/usr/local/cuda/lib64 -lcudart

network.o: src/network.cc
	nvcc -arch=sm_75 --compile src/network.cc -o src/network.o -I./ -L/usr/local/cuda/lib64 -lcudart

mnist.o: src/mnist.cc
	nvcc -arch=sm_75 --compile src/mnist.cc -o src/mnist.o  -I./ -L/usr/local/cuda/lib64 -lcudart

layer: src/layer/conv.cc src/layer/ave_pooling.cc src/layer/fully_connected.cc src/layer/max_pooling.cc src/layer/relu.cc src/layer/sigmoid.cc src/layer/softmax.cc 
	nvcc -arch=sm_75 --compile src/layer/ave_pooling.cc -o src/layer/ave_pooling.o -I./ -L/usr/local/cuda/lib64 -lcudart
	nvcc -arch=sm_75 --compile src/layer/conv.cc -o src/layer/conv.o -I./ -L/usr/local/cuda/lib64 -lcudart
	nvcc -arch=sm_75 --compile src/layer/conv_gpu.cc -o src/layer/conv_gpu.o -I./ -L/usr/local/cuda/lib64 -lcudart
	nvcc -arch=sm_75 --compile src/layer/fully_connected.cc -o src/layer/fully_connected.o -I./ -L/usr/local/cuda/lib64 -lcudart
	nvcc -arch=sm_75 --compile src/layer/max_pooling.cc -o src/layer/max_pooling.o -I./ -L/usr/local/cuda/lib64 -lcudart
	nvcc -arch=sm_75 --compile src/layer/relu.cc -o src/layer/relu.o -I./ -L/usr/local/cuda/lib64 -lcudart
	nvcc -arch=sm_75 --compile src/layer/sigmoid.cc -o src/layer/sigmoid.o -I./ -L/usr/local/cuda/lib64 -lcudart
	nvcc -arch=sm_75 --compile src/layer/softmax.cc -o src/layer/softmax.o -I./ -L/usr/local/cuda/lib64 -lcudart

version1:
	rm -f src/layer/custom/*.o demo demo_CPU demo_GPU
	nvcc -arch=sm_75 --compile src/layer/custom/gpu_utils.cu -o src/layer/custom/gpu_utils.o -I./ -L/usr/local/cuda/lib64 -lcudart 
	nvcc -arch=sm_75 --compile src/layer/custom/conv_forward_ver1.cu -o src/layer/custom/conv_forward_ver1.o -I./ -L/usr/local/cuda/lib64 -lcudart

version2: 
	rm -f src/layer/custom/*.o demo demo_CPU demo_GPU
	nvcc -arch=sm_75 --compile src/layer/custom/gpu_utils.cu -o src/layer/custom/gpu_utils.o -I./ -L/usr/local/cuda/lib64 -lcudart 
	nvcc -arch=sm_75 --compile src/layer/custom/conv_forward_ver2.cu -o src/layer/custom/conv_forward_ver2.o -I./ -L/usr/local/cuda/lib64 -lcudart

version3: 
	rm -f src/layer/custom/*.o demo demo_CPU demo_GPU
	nvcc -arch=sm_75 --compile src/layer/custom/gpu_utils.cu -o src/layer/custom/gpu_utils.o -I./ -L/usr/local/cuda/lib64 -lcudart 
	nvcc -arch=sm_75 --compile src/layer/custom/conv_forward_ver3.cu -o src/layer/custom/conv_forward_ver3.o -I./ -L/usr/local/cuda/lib64 -lcudart

version4: 
	rm -f src/layer/custom/*.o demo demo_CPU demo_GPU
	nvcc -arch=sm_75 --compile src/layer/custom/gpu_utils.cu -o src/layer/custom/gpu_utils.o -I./ -L/usr/local/cuda/lib64 -lcudart 
	nvcc -arch=sm_75 --compile src/layer/custom/conv_forward_ver4.cu -o src/layer/custom/conv_forward_ver4.o -I./ -L/usr/local/cuda/lib64 -lcudart

loss: src/loss/cross_entropy_loss.cc src/loss/mse_loss.cc
	nvcc -arch=sm_75 --compile src/loss/cross_entropy_loss.cc -o src/loss/cross_entropy_loss.o -I./ -L/usr/local/cuda/lib64 -lcudart
	nvcc -arch=sm_75 --compile src/loss/mse_loss.cc -o src/loss/mse_loss.o -I./ -L/usr/local/cuda/lib64 -lcudart

optimizer: src/optimizer/sgd.cc
	nvcc -arch=sm_75 --compile src/optimizer/sgd.cc -o src/optimizer/sgd.o -I./ -L/usr/local/cuda/lib64 -lcudart

clean:
	rm -f demo
	rm -f demo_CPU
	rm -f demo_GPU
	rm -f *.o src/*.o src/layer/*.o src/loss/*.o src/optimizer/*.o src/layer/custom/*.o

run:
	make network.o
	make mnist.o
	make layer
	make loss
	make optimizer