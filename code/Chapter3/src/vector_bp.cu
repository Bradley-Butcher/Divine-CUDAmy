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
__global__ void addVectors(const int *a, const int *b, int *c, const int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {
        c[index] = a[index] + b[index];
    }
}

void fill_array(int *data, int size) {
    for (int i = 0; i < size; ++i)
        data[i] = rand() % 100;
}

int main() {
    const int N = 1<<20; // size of vectors
    const int THREADS_PER_BLOCK = 256;
    const int BLOCKS = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    int *a, *b, *c; // host copies of a, b, c
    int *d_a, *d_b, *d_c; // device copies of a, b, c
    int size = N * sizeof(int);

    // Allocate space for device copies of a, b, c
    gpuErrchk(cudaMalloc((void **)&d_a, size));
    gpuErrchk(cudaMalloc((void **)&d_b, size));
    gpuErrchk(cudaMalloc((void **)&d_c, size));

    // Setup input values
    a = (int*)malloc(size); assert(a != NULL); fill_array(a, N);
    b = (int*)malloc(size); assert(b != NULL); fill_array(b, N);
    c = (int*)malloc(size); assert(c != NULL);

    // Copy inputs to device
    gpuErrchk(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice));

    // Call the kernel
    addVectors<<<BLOCKS, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // Copy result back to host
    gpuErrchk(cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost));

    // Now let's inspect the results
    std::cout << "a[0]=" << a[0] << std::endl;
    std::cout << "b[0]=" << b[0] << std::endl;
    std::cout << "c[0]=" << c[0] << std::endl;

    // Cleanup
    free(a); free(b); free(c);
    gpuErrchk(cudaFree(d_a)); gpuErrchk(cudaFree(d_b)); gpuErrchk(cudaFree(d_c));

    return 0;
}