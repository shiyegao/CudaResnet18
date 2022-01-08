#ifndef CUDA_PROJ_HELPER_FUNCTIONS_CUH
#define CUDA_PROJ_HELPER_FUNCTIONS_CUH

#include <cuda_runtime.h>

#include <sstream>
#include <iostream>
#include <cstdlib>
#include <stdexcept>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/execution_policy.h>

#define DSIZEW 10000
#define DSIZEH 2000

template <typename T>
struct isnan_test { 
    __host__ __device__ bool operator()(const T a) const {
        //if (isnan(a) | isinf(a) | a>100)
        //    printf("a: %f\n", a);
        return isnan(a) | isinf(a);
    }
};


template<typename T>
inline bool cudaCheckNan(const T* d_ptr, size_t size){
    return thrust::transform_reduce(thrust::device, d_ptr, d_ptr + size, isnan_test<T>(), 0, thrust::plus<bool>());
}


inline void _checkCudaErrors(const cudaError_t& status, const char* file, int line) {
    if (status != cudaSuccess) {
        std::stringstream _error;
        _error  << "Cuda failure: "
                << cudaGetErrorString(status)
                << std::endl;
        _error << file << ':' << line << std::endl;
        throw std::runtime_error(_error.str().c_str());
    }
}

#define checkCudaErrors(status) _checkCudaErrors(status, __FILE__, __LINE__);





inline void InitializeCUDA(int dev, bool verbose=false){
    int nDevices;

    cudaGetDeviceCount(&nDevices);

    if (nDevices < 1)
        throw std::runtime_error("Could not find GPU device");

    if (verbose){
        printf("Found %d devices:\n", nDevices);
        for (int i = 0; i < nDevices; i++) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            printf("Device Number: %d\n", i);
            printf("  Device name: %s\n", prop.name);
            printf("  Number of concurrent kernels: %d\n",
                   prop.concurrentKernels);
            printf("  Multi Processor Count: %d\n",
                   prop.multiProcessorCount);
            printf("  Clock Frequency (KHz): %d\n",
                   prop.clockRate);
            printf("  Memory Clock Rate (KHz): %d\n",
                   prop.memoryClockRate);
            printf("  Memory Bus Width (bits): %d\n",
                   prop.memoryBusWidth);
            printf("  Peak Memory Bandwidth (GB/s): %f\n",
                   2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
            printf("  Total Global Memory (MB): %lu\n\n",
                   prop.totalGlobalMem/1024/1024);
        }
        printf("Using GPU number %d\n", dev);
    }

    checkCudaErrors(cudaSetDevice(dev));
}

#endif //CUDA_PROJ_HELPER_FUNCTIONS_CUH
