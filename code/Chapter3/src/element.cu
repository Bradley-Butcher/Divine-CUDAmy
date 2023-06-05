#include <iostream>

// Kernel definition
__global__ void add(int a, int b, int *c) {
    *c = a + b;
}

int main() {
    int c; // host copy of c
    int *d_c; // device copy of c
    int size = sizeof(int);

    // Allocate space for device copy of c
    cudaMalloc((void **)&d_c, size);

    // Call the kernel
    add<<<1,1>>>(2, 7, d_c);

    // Copy result back to host
    cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);

    std::cout << "2 + 7 = " << c << std::endl;

    // Cleanup
    cudaFree(d_c);

    return 0;
}