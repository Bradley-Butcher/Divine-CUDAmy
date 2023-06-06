#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

extern "C"
{
#include "matrix.h"
}

// Kernel definition
__global__ void addMatrices(const int *a, const int *b, int *c, const int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        c[row*N + col] = a[row*N + col] + b[row*N + col];
    }
}

void gpu_add_matrices(int *a, int *b, int *c, const int N) {
    const int THREADS_PER_BLOCK = 16;
    const int BLOCKS = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    int *d_a, *d_b, *d_c; // device copies of a, b, c
    int size = N * N * sizeof(int);

    // Allocate space for device copies of a, b, c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Call the kernel
    dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 numBlocks(BLOCKS, BLOCKS);
    addMatrices<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, N);

    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}