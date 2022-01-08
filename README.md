# CudaInference
Cuda NN inference. Example: ResNet18 in source/resnet18_main.cpp.


## Functionality implemented:
* Convolution - with/without bias, arbitrary padding, arbitrary stride.
* Linear - with/without bias.
* BatchNorm.
* ReLU.
* MaxPool - arbitrary padding, arbitrary stride.
* AvgPool - arbitrary padding, arbitrary stride.
* Tensor operations:
    * common operations (+, -, *, \/).
    * transpose - arbitrary number of dimentions, arbitrary axes permutation.
    * reshape.
    * Relu, add_relu.


# Usage

## Install
```
git clone https://github.com/shiyegao/CudaResnet18.git
cd CudaResnet18
```

## Prepare weights
This step is already finished in the folder 'weights'. We use 'CudaResnet18/utils/check_npy.py' to change a onnx file into npy files as for model weights.

If you want to use another onnx file as inputs, you should change the corresponding ```dic``` and ```root``` in 'utils/check_npy.py' and run
```
python ./utils/check_npy.py  # YOU DO NOT NEED TO RUN THIS CODE!
```

## Build
After installation and preparation, we need to build the codes. If you are already under folder 'CudaResnet18', just run
```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release .. 
make -j
```

## Run
After building, we can run the resnet_cuda codes if we are under 'CudaResnet18/build'.
```
./Release/cuda_proj --weights_dir ../weights/
```

# File structure
There are 4 folders which contain our main codes. We will introduce the function of each part.
```
CudaResnet18
    |----include
            |----*.hpp
            |----*.cuh
            |----*.h
    |----source
            |----*.cu
            |----*.cc
            |----*.cpp
    |----utils
            |----check_npy.py
    |----weights
            |----conv1.bias.npy
            |----conv1.weight.npy
            |----......
    |----CMakeLists.txt
    |----README.md
    |----resnet18.onnx
    |----resnet18Input.txt
    |----resnet18Output.txt
```
## include
Head files of ```cu``` and ```cpp``` are saved here.

## source
Files of ```cu``` and ```cpp``` are saved here.
The ```Inference.cpp``` contains the main test function. The ```initModel()``` and ```inference()``` functions are finished in this file which are required in the assignment. The default GPU is '0', which is defined in ```InitializeCUDA(0);``` in ```initModel()```.

## utils
The file to change ```.onnx``` file into ```.npy``` weights is saved here.

## weights
Weights files are saved here for model weights loading.

## other files
1. ```CMakeLists.txt``` is for building the project before running Resnet18.
2. ```README.md``` is for code reading. 
3. ```resnet18.onnx``` is the file saving the Resnet18 model and its weights.
4. ```resnet18Input.txt``` is the input file for test.
5. ```resnet18Output.txt``` is the standard output file for comparison with our output.
