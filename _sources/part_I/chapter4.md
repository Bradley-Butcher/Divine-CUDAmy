# The Third Circle: Speaking with the Python

Ah, dear reader, welcome to the Third Circle of our shared Inferno, where we begin the journey of touching our CUDA to the Python. 

## Step 1: Making our CUDA code play nice

First things first, we must modify our matrix addition code from the previous chapter.

```cpp
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

extern "C"
{
#include "add_matrices.h"
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
```

The keen eyed among you, of which I'm sure there must be at least one, will notice we've changed the name of the `main` function to `gpu_add_matrices`. This is because we're going to be calling this function from Python, and we don't want to have to deal with the `main` function.
In addition, it now takes in parameters, how nice. Note I've also immediately thrown away most of our best practices from the previous chapter. I'm trying to keep it short, who needs error checking anyway?

We'll also need to import `<cuda.h>` and `<cuda_runtime.h>` to make sure we have access to CUDA's functions and types when using Cython. In previous chapters these were automatically added by the NVCC compiler, but now we're using Cython, we need to do it ourselves. Boo. 

Lastly, we'll need to add the `extern "C"` block. This tells the compiler to treat the code like C code to avoid name mangling, allowing Cython to correctly call our CUDA function. It's totally a thing, Google it.

## Step 2: Head over heels for headers

Next on the devils list is to create a header file for our new function. This allows Cython (discussed imminently) to know what functions exist, and how to call them. 

```cpp
#ifndef ADD_MATRICES_H
#define ADD_MATRICES_H

void gpu_add_matrices(int *a, int *b, int *c, const int N);

#endif
```

In the nightclub of our code, the `#ifndef` serves as the bouncer. It checks if the guest `ADD_MATRICES_H` is already at the party. If it is, the bouncer firmly says, "You're not coming in twice, mate!"
If `ADD_MATRICES_H` isn't at the party yet, `#define` puts its name on the guest list. This way, if it tries to sneak in again, the bouncer will be ready.

The line `void gpu_add_matrices(int *a, int *b, int *c, const int N);` is like the DJ announcing the next track: "Next up, we've got `gpu_add_matrices`, spinning some beats with `a`, `b`, `c`, and `N`!". It's declaring our CUDA function and what it expects to do its magic.

And finally, `#endif` is our bouncer giving a sigh of relief and shutting the door, signaling the end of the include guard shift.

## Step 3: Wrapping it up with Cython

Now, we will bridge the chasm between CUDA and Python, the high-level convenience and the low-level power. It's akin to merging water and oil, but fortunately for us, Cython exists to make this seemingly impossible task possible. 
We'll need to create a Cython file, for example: `matrix_adder_wrapper.pyx`.

Let's start by adding these enigmatic directives:

```python
# distutils: language = c++
# distutils: sources = add_matrices.cu
```

These lines are specifically for distutils, a module in Python responsible for building and installing additional modules. We inform it that the language we're going to deal with is C++ (although it's really CUDA, but for the build process, it's nearly the same). The second line specifies the source file that will be compiled, our CUDA program, `add_matrices.cu`. This information is vital for distutils to correctly build our extension.

Then, we have this exotic block:

```python
cdef extern from "add_matrices.h":
    void gpu_add_matrices(int *a, int *b, int *c, const int N)
```

The `cdef extern` here is Cython's way of declaring that there's a C function we'd like to use. By doing so, Cython can generate the correct C code that calls our CUDA function from Python. This block is our diplomatic envoy, effectively stating, "We've got an outside function named `gpu_add_matrices`. It looks like this, and it's from `add_matrices.h`."

Now, onto the pièce de résistance: the Python wrapper function:

```python
def add_matrices(a, b):
    cdef int N = len(a)
    cdef int c[N*N]
    gpu_add_matrices(&a[0], &b[0], &c[0], N)
    return c
```

This Python function uses Cython's ability to declare C-style variables with `cdef`, and it calls our CUDA function using pointers (i.e., `&a[0]`) to the data stored in Python lists. The result of the CUDA function is then returned back to Python land.

This leaves us with the full code:

```python
# distutils: language = c++
# distutils: sources = add_matrices.cu

cdef extern from "add_matrices.h":
    void gpu_add_matrices(int *a, int *b, int *c, const int N)

def add_matrices(a, b):
    cdef int N = len(a)
    cdef int c[N*N]
    gpu_add_matrices(&a[0], &b[0], &c[0], N)
    return c
```

## Step 4: Building with Distutils

Oh, my favorite part of coding: the build process. It's like waiting for paint to dry, but the paint is code. Let's turn to our trusty sidekick, distutils, to help us out.

First, we need to write a `setup.py` file, which is about as fun as it sounds:

```python
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        "add_matrices",
        sources=["add_matrices.pyx", "add_matrices.cu"],
        include_dirs=[numpy.get_include()],
        language='c++',
        extra_compile_args=['-O2', '--ptxas-options=-v', '-arch=sm_60', '--compiler-options', "'-fPIC'"],
        extra_link_args=['-lcudart'],
        library_dirs=['/usr/local/cuda/lib64']
    )
]

setup(
    name='Add Matrices',
    ext_modules=cythonize(ext_modules),
)
```

This code tells distutils what it should be doing, in a way that is clear to even the most stubborn of machines. There are various compile and link arguments that are specific to your system, so you may need to change them to match your setup.

Our `ext_modules` list contains a single `Extension` object, which is essentially a glorified sticky note saying: "Compile these files with these settings, and please don't mess it up". Then `cythonize(ext_modules)` takes this sticky note and translates it into a format that distutils can understand, sort of like a parent interpreting their toddler's scribbles.  Finally, `setup` ties this all together and kicks off the build process.

With any luck, we'll be able to run `python setup.py build_ext --inplace`, and be able to import our new module into Python with `import add_matrices`. If not, well, I'm sure you'll figure it out. 
The full code is available in the `code` folder once more.