#include <iostream>
#include <cstdlib>

// Kernel definition
__global__ void addVectors(int *a, int *b, int *c, int N) {
    int index = threadIdx.x;
    if (index < N) {
        c[index] = a[index] + b[index];
    }
}

void fill_array(int *data, int size) {
    for (int i = 0; i < size; ++i)
        data[i] = rand() % 100;
}

int main() {
    int N = 1<<20; // size of vectors
    int *a, *b, *c; // host copies of a, b, c
    int *d_a, *d_b, *d_c; // device copies of a, b, c
    int size = N * sizeof(int);

    // Allocate space for device copies of a, b, c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Setup input values
    a = (int*)malloc(size); fill_array(a, N);
    b = (int*)malloc(size); fill_array(b, N);
    c = (int*)malloc(size);

    // Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Call the kernel - each block has 1 thread
    addVectors<<<N,1>>>(d_a, d_b, d_c, N);

    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    std::cout << "a[0]=" << a[0] << std::endl;
    std::cout << "b[0]=" << b[0] << std::endl;
    std::cout << "c[0]=" << c[0] << std::endl;

    // Cleanup
    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return 0;
}