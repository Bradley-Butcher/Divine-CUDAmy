#include <iostream>
#include <cassert>

// Error checking macro
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Kernel definition
__global__ void addMatrices(const int *a, const int *b, int *c, const int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        c[row*N + col] = a[row*N + col] + b[row*N + col];
    }
}

void fill_matrix(int *data, const int size) {
    for (int i = 0; i < size*size; ++i)
        data[i] = rand() % 10;
}

int main() {
    const int N = 1<<10; // size of matrices
    const int THREADS_PER_BLOCK = 16;
    const int BLOCKS = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    int *a, *b, *c; // host copies of a, b, c
    int *d_a, *d_b, *d_c; // device copies of a, b, c
    int size = N * N * sizeof(int);

    // Allocate space for device copies of a, b, c
    gpuErrchk(cudaMalloc((void **)&d_a, size));
    gpuErrchk(cudaMalloc((void **)&d_b, size));
    gpuErrchk(cudaMalloc((void **)&d_c, size));

    // Setup input values
    a = (int*)malloc(size); assert(a != NULL); fill_matrix(a, N);
    b = (int*)malloc(size); assert(b != NULL); fill_matrix(b, N);
    c = (int*)malloc(size); assert(c != NULL);

    // Copy inputs to device
    gpuErrchk(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice));

    // Call the kernel
    dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 numBlocks(BLOCKS, BLOCKS);
    addMatrices<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, N);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // Copy result back to host
    gpuErrchk(cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost));

    std::cout << "a[4][7] = " << a[4*N + 7] << std::endl;
    std::cout << "b[4][7] = " << b[4*N + 7] << std::endl;
    std::cout << "c[4][7] = " << c[4*N + 7] << std::endl;

    // Cleanup
    free(a); free(b); free(c);
    gpuErrchk(cudaFree(d_a)); gpuErrchk(cudaFree(d_b)); gpuErrchk(cudaFree(d_c));

    return 0;
}
