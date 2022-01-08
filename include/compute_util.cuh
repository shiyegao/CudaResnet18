#ifndef CUDA_PROJ_COMPUTE_UTIL_CUH
#define CUDA_PROJ_COMPUTE_UTIL_CUH

#include <cublas.h>
#include <cublas_v2.h>

const int TILE_WIDTHn = 32;

__global__ void debug_ker(float* ptr, int addr);
void debug_array(float* arr, int N, int M);
__global__ void debug_ker2(float* ptr, float* ptr2, int addr, int *a);
void debug_array2(float* arr,float* arr2, int M);

void rmgemm(int m, int n, int k, float* A, float* B, float* C);

template<typename T>
void cuda_relu(T* src1, T* res, int N);
template<typename T>
void cuda_add_relu(T* src1, T* src2, T* res, int N);
template<typename T>
void cuda_add(T* src1, T* src2, T* res, int N);
template<typename T>
void cuda_sub(T* src1, T* src2, T* res, int N);
template<typename T>
void cuda_mul(T* src1, T* src2, T* res, int N);
template<typename T>
void cuda_div(T* src1, T* src2, T* res, int N);

template<typename T>
__global__ void transpose_ker(T* src_ptr, T* dst_ptr, int* src_dims, int* strides, int* reorder, int* new_strides, int Ndims);
template<typename T>
void cuda_transpose(T* src_ptr, T* dst_ptr, int* src_dims, int* strides, int* reorder, int* new_strides, int Ndims, int N);

#endif //CUDA_PROJ_TENSOR_CUH
