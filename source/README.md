# Source Files

* ```AvgPoolLayer.cu```,```ConvLayer.cu```,```LinearLayer.cu```,```MaxPoolLayer.cu```: Implement network layer classes in ResNet18.
* ```Gemm.cu```: Implement CUDA matrix multiplication kernel.
* ```Tensor.cpp```: Implement of tensor class, which serve as base data type for network layer input and output.
* ```npy.cpp```: Numpy source file. For loading network weights saved as ```.npy``` files.
* ```compute_util.cu```: Implement various CUDA kernels for addition, multiplication, relu and so on.
* ```Inference.cpp```: Implement main test functions, initModel() and inference(), as assignment required. The default GPU is '0', which is defined in InitializeCUDA(0); in initModel().
* ```resnet18_main.cc```: The provided main file. initModel() and inference() are instead implemented in ```Inference.cpp``` file.