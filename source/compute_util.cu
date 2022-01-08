// #include "helper_functions.cuh"
#include "gemm.cuh"
#include "compute_util.cuh"


void rmgemm(int m, int n, int k, float* A, float* B, float* C){
    
    int num_blocks_x = n / TILE_WIDTHn + (n % TILE_WIDTHn != 0);
    int num_blocks_y = m / TILE_WIDTHn + (m % TILE_WIDTHn != 0);
    dim3 dimGrid(num_blocks_x, num_blocks_y);
    dim3 dimBlock(TILE_WIDTHn, TILE_WIDTHn/8);
    MatrixMulKernel<<<dimGrid, dimBlock>>>(A, B, C, m, n, k);
}

template<typename T>
__global__ void relu_ker(T* src, T* dst, int N){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N){
        return;
    }
    dst[i] = fmaxf(0.0, src[i]);
}

template<typename T>
void cuda_relu(T* src1, T* res, int N){
    int cell_size = 32;
    dim3 block_size;
    dim3 grid_size;
    int num_blocks_x;

    num_blocks_x = N/cell_size + (N % cell_size != 0);
    block_size = dim3(cell_size);
    grid_size = dim3(num_blocks_x);

    relu_ker<<<grid_size, block_size>>>(src1, res, N);
}

template<typename T>
__global__ void add_relu_ker(T* src1, T* src2, T* dst, int N){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N){
        return;
    }
    dst[i] = fmaxf(0.0, src1[i] + src2[i]);
}

template<typename T>
void cuda_add_relu(T* src1, T* src2, T* res, int N){
    int cell_size = 32;
    dim3 block_size;
    dim3 grid_size;
    int num_blocks_x;

    num_blocks_x = N/cell_size + (N % cell_size != 0);
    block_size = dim3(cell_size);
    grid_size = dim3(num_blocks_x);
    add_relu_ker<T><<<grid_size, block_size>>>(src1, src2, res, N);
}

template<typename T>
__global__ void add_ker(T* src1, T* src2, T* dst, int N){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N){
        return;
    }
    dst[i] = src1[i] + src2[i];
}

template<typename T>
void cuda_add(T* src1, T* src2, T* res, int N){
    int cell_size = 32;
    dim3 block_size;
    dim3 grid_size;
    int num_blocks_x;

    num_blocks_x = N/cell_size + (N % cell_size != 0);
    block_size = dim3(cell_size);
    grid_size = dim3(num_blocks_x);
    add_ker<T><<<grid_size, block_size>>>(src1, src2, res, N);
}

template<typename T>
__global__ void sub_ker(T* src1, T* src2, T* dst, int N){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N){
        return;
    }
    dst[i] = src1[i] - src2[i];
}

template<typename T>
void cuda_sub(T* src1, T* src2, T* res, int N){
    int cell_size = 32;
    dim3 block_size;
    dim3 grid_size;
    int num_blocks_x;

    num_blocks_x = N/cell_size + (N % cell_size != 0);
    block_size = dim3(cell_size);
    grid_size = dim3(num_blocks_x);
    sub_ker<T><<<grid_size, block_size>>>(src1, src2, res, N);
}


template<typename T>
__global__ void mul_ker(T* src1, T* src2, T* dst, int N){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N){
        return;
    }
    dst[i] = src1[i] * src2[i];
}

template<typename T>
void cuda_mul(T* src1, T* src2, T* res, int N){
    int cell_size = 32;
    dim3 block_size;
    dim3 grid_size;
    int num_blocks_x;

    num_blocks_x = N/cell_size + (N % cell_size != 0);
    block_size = dim3(cell_size);
    grid_size = dim3(num_blocks_x);
    mul_ker<T><<<grid_size, block_size>>>(src1, src2, res, N);
}

template<typename T>
__global__ void div_ker(T* src1, T* src2, T* dst, int N){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N){
        return;
    }
    dst[i] = src1[i] / src2[i];
}

template<typename T>
void cuda_div(T* src1, T* src2, T* res, int N){
    int cell_size = 32;
    dim3 block_size;
    dim3 grid_size;
    int num_blocks_x;

    num_blocks_x = N/cell_size + (N % cell_size != 0);
    block_size = dim3(cell_size);
    grid_size = dim3(num_blocks_x);
    div_ker<T><<<grid_size, block_size>>>(src1, src2, res, N);
}


template<typename T>
__global__ void transpose_ker(T* src_ptr, T* dst_ptr, int* src_dims, int* strides, int* reorder, int* new_strides, int Ndims, int N){
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    if (i >= N){
        return;
    }

    int new_idx[10];
    int acc = 0;
    for (int k = 0; k < Ndims; ++k) {
        int cur_i = (i - acc) / strides[k];
        acc += cur_i*strides[k];

        new_idx[reorder[k]] = cur_i;
    }

    int new_i = 0;
    for (int k = 0; k < Ndims; ++k) {
        new_i += new_strides[k]*new_idx[k];
    }

    dst_ptr[new_i] = src_ptr[i];
}

template<typename T>
void cuda_transpose(T* src_ptr, T* dst_ptr, int* src_dims, int* strides, int* reorder, int* new_strides, int Ndims, int N){
    int cell_size = 32;
    dim3 block_size;
    dim3 grid_size;
    int num_blocks_x;
    num_blocks_x = (N)/cell_size + ((N) % cell_size != 0);
    block_size = dim3(cell_size);
    grid_size = dim3(num_blocks_x);

    transpose_ker<<<grid_size, block_size>>>(src_ptr, dst_ptr, src_dims, strides, reorder, new_strides, Ndims, N);
}

template void cuda_relu<float>(float*, float*, int);
template void cuda_add_relu<float>(float*, float*, float*, int);
template void cuda_add<float>(float*, float*, float*, int);
template void cuda_sub<float>(float*, float*, float*, int);
template void cuda_mul<float>(float*, float*, float*, int);
template void cuda_div<float>(float*, float*, float*, int);

template void cuda_relu<int>(int*, int*, int);
template void cuda_add_relu<int>(int*, int*, int*, int);
template void cuda_add<int>(int*, int*, int*, int);
template void cuda_sub<int>(int*, int*, int*, int);
template void cuda_mul<int>(int*, int*, int*, int);
template void cuda_div<int>(int*, int*, int*, int);

template void cuda_transpose<float>(float*, float*, int*, int*, int*, int*, int, int);
template void cuda_transpose<int>(int*, int*, int*, int*, int*, int*, int, int);
