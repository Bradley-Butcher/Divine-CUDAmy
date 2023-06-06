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

# Avoiding sweet but unneccesary pain with CUDA best practices

Isn't adding great? We've created something truly special, flawless. Truly cast-iron in its design. Or so we'd like to think. In the blissful ignorance of missing best practices, our program can become akin to a sinking ship, leaking memory from small unseen cracks while mismanaging thread blocks like a lost conductor in a grand symphony. But fear not, for I have compiled a list of best practices, which I have just googled (remember, I'm new to this too) to help you avoid the same mistakes I made. You may skip this section, as I probably would, but do so at your own peril. I salute you, you brave, idiotic soul.
## Error Checking: It's Not Paranoia If They're Really Out to Get You

First and foremost, CUDA doesn't like to tell you when things go wrong. It'll sit in silence, maybe smirk a bit, and watch you pull your hair out in confusion. That's why it's paramount to check for errors for CUDA API calls and kernel launches. I've done you a favor and included a handy macro and function to assist with this. It's a like a smoke detector. When it's silent, you can enjoy the soothing hum of your GPU. When it goes off, it's time to evacuate your code.

```cpp
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
```

So, what is a macro, you ask? In C and C++, a macro is a way of defining shorthand for other code. It's like when your girlfriend whispers in your ear to remind you of her parents names. You just have to say gpuErrchk and the compiler understands it as gpuAssert.

The #define directive defines the gpuErrchk(ans) macro. When this macro is used, it's as if the code inside the curly brackets { gpuAssert((ans), __FILE__, __LINE__); } is copied and pasted in its place. This bit of magic invokes the gpuAssert function with three arguments: a CUDA operation that returns a cudaError_t, the current file name __FILE__, and the current line number __LINE__.

The gpuAssert fucntion takes a cudaError_t code, file name, line number, and whether to abort if an error is found. If the cudaError_t code indicates an error, the function prints an error message with the name of the file and the line number where the error happened, adding a personal touch to your debugging experience. If the abort parameter is true, the program stops there. This is the equivalent of someone yelling "Stop the presses!" when a typo is found in an old-timey newspaper.

So let's say you've called a CUDA function like so: `gpuErrchk(cudaMalloc((void **)&d_a, size));`. If cudaMalloc encounters an error and doesn't return cudaSuccess, gpuErrchk will call gpuAssert which will print out an error message telling you exactly where the error occurred. Without this careful error checking, CUDA might lead you on a wild goose chase around your code, cackling in the shadows as you wonder why your program is crashing or producing incorrect results. Error checking can save you from such torment. So, take a moment to appreciate these lines of code, for they will be your steadfast allies in the turbulent seas of CUDA programming.

## Choosing Optimal Block and Grid Sizes: It's Not About Size, It's How You Use It

In our example, we've been launching an excessive number of blocks, each with a single thread. You may find this satisfying in its simplicity, but unfortunately, it's akin to buying a sports car and never shifting out of first gear.

You see, CUDA cores enjoy company. They perform their best when surrounded by threads, not blocks. Instead, aim for fewer blocks, but pack them full of threads. Like stuffing clowns into a car, the more threads you fit in a block, the more efficiently you're using your resources. Common choices range from 256 to 512 threads per block, though you'll have to gauge the clown-car capacity of your specific GPU.

```cpp
    const int THREADS_PER_BLOCK = 16;
    const int BLOCKS = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    ...
    dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 numBlocks(BLOCKS, BLOCKS);
    addMatrices<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, N);
```

In this piece of code, that I, your humble guide, have painstakingly assembled, we've filled each block with threads until they're practically bulging at the seams. This is courtesy of the variable `THREADS_PER_BLOCK` set to 16 – a decision driven by the undeniable charm of powers of 2, and maybe a dash of mathematical superstition.

These threads are then herded into blocks like overzealous party-goers crammed into a small room. When we set the kernel in motion with addMatrices<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, N);, we're essentially opening the door to the party and hoping nobody notices the fire code violation. Here, numBlocks and threadsPerBlock are dim3 variables, a dim3 type being the CUDA equivalent of an overzealous party planner, making sure even one-dimensional parties are technically ready to go 3D at a moment's notice. In our case, numBlocks takes care of the number of blocks per dimension, and threadsPerBlock is the bouncer, keeping a check on the number of threads in each block.
## Memory Allocation and De-allocation: Give and Take, But Mostly Take

When it comes to memory, CUDA is a bit of a hoarder. It wants its own separate memory, separate from what the CPU uses. This memory needs to be allocated and then freed once we're done with it. If malloc or cudaMalloc fails, it returns a null pointer. That's like the bank declining your credit card. It's embarrassing, it's problematic, and it's something you need to check for. Always verify your allocations, or you'll be dining on a fine plate of segmentation faults for dinner.

## Const Correctness: Some Things Never Change

As a final note, let's talk about 'const'. In our kernel, there's a parameter 'N' that doesn't get modified. It's like the grumpy old man of parameters, stuck in his ways. When you come across such parameters, declare them as 'const'. It helps the compiler, and it signals your intentions to anyone reading your code.


And there you have it, dear reader. You've successfully navigated the Second Circle of our CUDA Inferno. You've learned how to write and call CUDA kernels, and you've even learned how to add vectors and matrices together. I hope you're feeling proud of yourself, because you should be. In the next chapter, we'll learn how to interface our CUDA programs with Python, and things will start to get a lot more interesting. So hang in there. The best is yet to come.

And remember, if you're feeling overwhelmed, just take a deep breath and remember that every expert was once a beginner. Even me. Especially me. If you're feeling like you're in over your head, just remember: it's not rocket science. Well, unless you're using CUDA for rocket science. In which case, it is. But don't worry, you're smart. You wouldn't be here if you weren't. Or maybe you're just masochistic. Either way, I like your style.

As always, the full code can be found in the /code folder. Except when it can't, then it won't be there. 