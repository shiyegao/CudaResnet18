#include "gemm.cuh"
#include "cuda_runtime.h"
#include "compute_util.cuh"
#include <cstdio>

//Eight Opt
__global__ void MatrixMulKernel(float* Md, float* Nd, float* Pd, int m, int n, int k){ // mk , kn -> mn 
    __shared__ float Mds[TILE_WIDTHn][TILE_WIDTHn];
    __shared__ float Nds[TILE_WIDTHn][TILE_WIDTHn];

    int TILE_WIDTH = TILE_WIDTHn;

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int Row[8];
    int hT[8];
    for(int i = 0;i < 8;++i){
        hT[i] = TILE_WIDTH / 8 * i;
        Row[i] = by * TILE_WIDTH + ty + hT[i];
    }

    int Col = bx * TILE_WIDTH + tx;

    float Pvalue[8] = {0.};

    int Mx = tx;
    int Ny = ty;
    for (int i = 0; i < k / TILE_WIDTH; ++i){
        Mds[ty][tx] = Md[Row[0]*k + Mx];
        Nds[ty][tx] = Nd[Col*k + Ny]; 
        Mds[ty + hT[1]][tx] = Md[Row[1]*k + Mx];
        Nds[ty + hT[1]][tx] = Nd[Col*k + Ny + hT[1]];      
        Mds[ty + hT[2]][tx] = Md[Row[2]*k + Mx];
        Nds[ty + hT[2]][tx] = Nd[Col*k + Ny + hT[2]];
        Mds[ty + hT[3]][tx] = Md[Row[3]*k + Mx];
        Nds[ty + hT[3]][tx] = Nd[Col*k + Ny + hT[3]];
        Mds[ty + hT[4]][tx] = Md[Row[4]*k + Mx];
        Nds[ty + hT[4]][tx] = Nd[Col*k + Ny + hT[4]];      
        Mds[ty + hT[5]][tx] = Md[Row[5]*k + Mx];
        Nds[ty + hT[5]][tx] = Nd[Col*k + Ny + hT[5]];
        Mds[ty + hT[6]][tx] = Md[Row[6]*k + Mx];
        Nds[ty + hT[6]][tx] = Nd[Col*k + Ny + hT[6]];        
        Mds[ty + hT[7]][tx] = Md[Row[7]*k + Mx];
        Nds[ty + hT[7]][tx] = Nd[Col*k + Ny + hT[7]];


        Mx += TILE_WIDTH;
        Ny += TILE_WIDTH;

        __syncthreads();        
        for(int j = 0; j < TILE_WIDTH;++j){
            Pvalue[0] += Mds[ty][j] * Nds[j][tx];
            Pvalue[1] += Mds[ty + hT[1]][j] * Nds[j][tx];
            Pvalue[2] += Mds[ty + hT[2]][j] * Nds[j][tx];
            Pvalue[3] += Mds[ty + hT[3]][j] * Nds[j][tx];            
            Pvalue[4] += Mds[ty + hT[4]][j] * Nds[j][tx];
            Pvalue[5] += Mds[ty + hT[5]][j] * Nds[j][tx];
            Pvalue[6] += Mds[ty + hT[6]][j] * Nds[j][tx];
            Pvalue[7] += Mds[ty + hT[7]][j] * Nds[j][tx];
        }
        __syncthreads();
    }
    if(k % TILE_WIDTH != 0){
        Mds[ty][tx] = Md[Row[0]*k + Mx];
        Nds[ty][tx] = Nd[Col*k + Ny]; 
        Mds[ty + hT[1]][tx] = Md[Row[1]*k + Mx];
        Nds[ty + hT[1]][tx] = Nd[Col*k + Ny + hT[1]];      
        Mds[ty + hT[2]][tx] = Md[Row[2]*k + Mx];
        Nds[ty + hT[2]][tx] = Nd[Col*k + Ny + hT[2]];
        Mds[ty + hT[3]][tx] = Md[Row[3]*k + Mx];
        Nds[ty + hT[3]][tx] = Nd[Col*k + Ny + hT[3]];
        Mds[ty + hT[4]][tx] = Md[Row[4]*k + Mx];
        Nds[ty + hT[4]][tx] = Nd[Col*k + Ny + hT[4]];      
        Mds[ty + hT[5]][tx] = Md[Row[5]*k + Mx];
        Nds[ty + hT[5]][tx] = Nd[Col*k + Ny + hT[5]];
        Mds[ty + hT[6]][tx] = Md[Row[6]*k + Mx];
        Nds[ty + hT[6]][tx] = Nd[Col*k + Ny + hT[6]];        
        Mds[ty + hT[7]][tx] = Md[Row[7]*k + Mx];
        Nds[ty + hT[7]][tx] = Nd[Col*k + Ny + hT[7]];


        __syncthreads();
        for(int j = 0; j < k % TILE_WIDTH ;++j){
            Pvalue[0] += Mds[ty][j] * Nds[j][tx];
            Pvalue[1] += Mds[ty + hT[1]][j] * Nds[j][tx];
            Pvalue[2] += Mds[ty + hT[2]][j] * Nds[j][tx];
            Pvalue[3] += Mds[ty + hT[3]][j] * Nds[j][tx];            
            Pvalue[4] += Mds[ty + hT[4]][j] * Nds[j][tx];
            Pvalue[5] += Mds[ty + hT[5]][j] * Nds[j][tx];
            Pvalue[6] += Mds[ty + hT[6]][j] * Nds[j][tx];
            Pvalue[7] += Mds[ty + hT[7]][j] * Nds[j][tx];
        }
        __syncthreads();
    }
    if(Col < n){
        for(int i = 0;i < 8;++i){
            if(Row[i] < m) Pd[Row[i]*n +Col] = Pvalue[i];
        }
    }
}
