#ifndef CUDA_PROJ_GEMM_CUH
#define CUDA_PROJ_GEMM_CUH

#include "cuda_runtime.h"

__global__ void MatrixMulKernel(float* Md, float* Nd, float* Pd, int m, int n, int k);

#endif //CUDA_PROJ_GEMM_CUH
