# *The Second Circle*: Adding (to the Pain)

Welcome, dear reader, to the Second Circle of our CUDA Inferno. If you've made it this far, congratulations! You've either successfully installed CUDA and set up your environment, or you've decided to skip the first chapter entirely and dive headfirst into the deep end. Either way, I salute your bravery (or foolhardiness).

In this chapter, we're going to tackle the "Hello, World!" of CUDA programming: a simple addition program. Yes, you heard that right. We're going to use your GPU, a device so computationally intensive you could grill a steak on it, to add numbers. It's like using a supercomputer to play tic-tac-toe, but bear with me. This simple program will serve as a gentle introduction to the basics of CUDA programming, and it will set the stage for the more complex and exciting programs we'll tackle in the later chapters.

### **2.1 The Kernel**

The heart of any CUDA program is the kernel. This is the function that gets executed on the GPU. In CUDA, we define a kernel using the `__global__` keyword. Here's what our addition kernel looks like:

```cpp
__global__ void add(int a, int b, int *c) {
    *c = a + b;
}
```

This kernel takes two integers, `a` and `b`, and a pointer to an integer `c`. It adds `a` and `b` together and stores the result in `c`.

### **2.2 Calling the Kernel**

To call our kernel, we use the triple angle bracket syntax `<<< >>>`. This syntax allows us to specify the number of blocks and threads we want to use. For our simple addition program, we only need one block and one thread:

```cpp
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

    // Cleanup
    cudaFree(d_c);

    return 0;
}
```

In this code, we first allocate space on the GPU for our result using `cudaMalloc`. We then call our kernel. After the kernel has finished executing, we copy the result back to the host using `cudaMemcpy`. Finally, we clean up by freeing the memory we allocated on the GPU using `cudaFree`.

### **2.3 Extending to Vectors**

Now that we've mastered the art of adding two numbers together, let's raise the stakes a bit. Let's add two vectors together. Here's what our vector addition kernel looks like:

```cpp
__global__ void addVectors(int *a, int *b, int *c, int N) {
    int index = threadIdx.x;
    if (index < N) {
        c[index] = a[index] + b[index];
    }
}
```

In this kernel, `a`, `b`, and `c` are pointers to our vectors, and `N` is the size of the vectors. We calculate the

...index of the current thread using `threadIdx.x`, and we use this index to access the corresponding elements in `a`, `b`, and `c`.

The `threadIdx.x` is a built-in variable in CUDA that gives the index of the current thread within a block. Since we're dealing with vectors, we only need one dimension, hence the `.x`.

The beauty of CUDA is that it allows us to perform operations on multiple elements simultaneously. Each thread operates independently and performs its operation on a different element of the vector. This is why we don't need to write any loops in our kernel. The loop is implicit: one iteration per thread.

### **2.4 Calling the Vector Addition Kernel**

Calling our vector addition kernel is a bit more involved than calling our simple addition kernel. We need to allocate space on the GPU for our vectors, copy our vectors from the host to the GPU, call our kernel, and then copy the result back to the host. Here's what that looks like:

```cpp
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

    // Call the kernel
    addVectors<<<N,1>>>(d_a, d_b, d_c, N);

    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Cleanup
    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return 0;
}
```

### **2.5 Extending to Matrices**

Adding two matrices together is a natural extension of adding two vectors together. Instead of one index, we now have two indices: one for the row and one for the column. Here's what our matrix addition kernel looks like:

```cpp
__global__ void addMatrices(int *a, int *b, int *c, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        c[row*N + col] = a[row*N + col] + b[row*N + col];
    }
}
```

In this kernel, `a`, `b`, and `c` are pointers to our matrices, and `N` is the size of the matrices. We calculate the row and column indices using `blockIdx`, `blockDim`, and `threadIdx`, and we use these indices to access the corresponding elements in `a`, `b`, and `c`.

The `blockIdx` and `blockDim` are built-in variables in CUDA that give the index of the current block within the grid and the dimensions of the block, respectively. By multiplying the block index by the block dimension and adding the thread index, we can calculate a unique index for each

...thread across all blocks. This is how we can use `blockIdx`, `blockDim`, and `threadIdx` to index into our matrices.

Let's break it down with an example. Suppose we have a grid of blocks, where each block is of size 16x16 (so `blockDim.x = blockDim.y = 16`), and we have 4 blocks in the x-direction and 3 blocks in the y-direction. If we're in the thread of `blockIdx.x = 2`, `blockIdx.y = 1`, `threadIdx.x = 5`, and `threadIdx.y = 3`, the unique row and column indices for this thread would be:

```cpp
int row = blockIdx.y * blockDim.y + threadIdx.y; // 1 * 16 + 3 = 19
int col = blockIdx.x * blockDim.x + threadIdx.x; // 2 * 16 + 5 = 37
```

So, this thread would be responsible for the element at the 20th row and 38th column of the matrix (remember, indices are 0-based).

Just like with vector addition, each thread operates independently and performs its operation on a different element of the matrix. This is why we don't need to write any loops in our kernel. The loop is implicit: one iteration per thread.

### **2.6 Calling the Matrix Addition Kernel**

Calling our matrix addition kernel is similar to calling our vector addition kernel. We need to allocate space on the GPU for our matrices, copy our matrices from the host to the GPU, call our kernel, and then copy the result back to the host. Here's what that looks like:

```cpp
int main() {
    int N = 1<<10; // size of matrices
    int *a, *b, *c; // host copies of a, b, c
    int *d_a, *d_b, *d_c; // device copies of a, b, c
    int size = N * N * sizeof(int);

    // Allocate space for device copies of a, b, c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Setup input values
    a = (int*)malloc(size); fill_matrix(a, N);
    b = (int*)malloc(size); fill_matrix(b, N);
    c = (int*)malloc(size);

    // Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Call the kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    addMatrices<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, N);

    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Cleanup
    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return 0;
}
```

And there you have it, dear reader. You've successfully navigated the Second Circle of our CUDA Inferno. You've learned how to write and call CUDA kernels, and you've even learned how to add vectors and matrices together. I hope you're feeling proud of yourself, because you should be. In the next chapter, we'll learn how to interface our CUDA programs with Python, and things will start to get a lot more interesting. So hang in there. The best is yet to come

! And remember, if you're feeling overwhelmed, just take a deep breath and remember that every expert was once a beginner. Even me. Especially me.

And if you're feeling like you're in over your head, just remember: it's not rocket science. Well, unless you're using CUDA for rocket science. In which case, it is. But don't worry, you're smart. You wouldn't be here if you weren't. Or maybe you're just masochistic. Either way, I like your style.

So, stick around, won't you? Let's continue this dance of ours. I promise, it'll be worth your while.

## Full Code

### Element addition

```cpp
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
```

### Vector Addition

```cpp
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
        data[i] = i;
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

    // Call the kernel
    addVectors<<<N,1>>>(d_a, d_b, d_c, N);

    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Cleanup
    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return 0;
}
```

## Matrix Addition

```cpp
#include <iostream>
#include <cstdlib>

// Kernel definition
__global__ void addMatrices(int *a, int *b, int *c, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        c[row*N + col] = a[row*N + col] + b[row*N + col];
    }
}

void fill_matrix(int *data, int size) {
    for (int i = 0; i < size*size; ++i)
        data[i] = i;
}

int main() {
    int N = 1<<10; // size of matrices
    int *a, *b, *c; // host copies of...a, b, c
    int *d_a, *d_b, *d_c; // device copies of a, b, c
    int size = N * N * sizeof(int);

    // Allocate space for device copies of a, b, c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Setup input values
    a = (int*)malloc(size); fill_matrix(a, N);
    b = (int*)malloc(size); fill_matrix(b, N);
    c = (int*)malloc(size);

    // Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Call the kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    addMatrices<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, N);

    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Cleanup
    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return 0;
}
```