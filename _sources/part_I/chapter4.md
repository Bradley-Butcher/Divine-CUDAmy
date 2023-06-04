# The Third Circle: Speaking with the Python

*In the Third Circle, we see the allure of Python and the unyielding fortitude of CUDA form a tumultuous bond. A bit like those celebrity relationships in the tabloids—eccentric and always on the edge of implosion. Yet, we dare to tread this path.*

Before we delve into the deep, dark abyss of mixing Python and CUDA, let's comprehend why we're subjecting ourselves to this ordeal in the first place.

## Why Are We Doing This Again?

We're looking to have our Python and CUDA cake and eat it too. Python, being an interpreted language, comes with the gift of simplicity but pays the price in speed. CUDA, on the other hand, allows us to perform complex calculations rapidly by using the power of the GPU. The idea is to exploit the easy-to-use nature of Python for tasks such as data preprocessing, then pass the data to CUDA for the heavy lifting. A harmonious duet of simplicity and power.

But how do we create this harmony? Enter Cython.

## Cython: Your Translator in This Hellish Journey

Cython is our bridge, an interpreter between Python and CUDA. It's a programming language that is a superset of Python, which makes it possible for Python to play well with C/C++. Since CUDA C is essentially C/C++ with a few extra bits, Cython can help us interface with CUDA code as well.

To put it in layman's terms, Cython is like a skilled diplomat, proficient in the languages of both Python and CUDA, and capable of facilitating productive dialogue between the two.

Now, let's roll up our sleeves and get dirty.

## Setting Up Cython

First things first, we need to install Cython. Stay calm and command your terminal:

```bash
pip install cython
```

Well done, survivor. You've achieved another monumental task in this journey through the inferno.

## Creating The Bridge: Building a Cython Wrapper

The first step in calling CUDA from Python is creating a Cython wrapper for your CUDA code. I know this sounds confusing, but it's really not.

You see, Cython works by translating Python code into C code and then compiling this C code into a Python extension module, a shared library that Python can import just like a normal Python module.

First, let's create a file named '[add.cu](http://add.cu/)' and write our CUDA kernel function for adding two integers. Remember that from Chapter 3? This time, we'll also include a wrapper function to call our kernel. The 'extern "C"' bit tells the compiler that the code inside the braces should be compiled as C code.

```cpp
#include <cuda.h>

// The CUDA kernel
__global__ void add_kernel(int a, int b, int *c)
{
    *c = a + b;
}

// The C++ wrapper function
extern "C" void add(int a, int b, int *c)
{
    int *dev_c;

    // Allocate memory on the GPU
    cudaMalloc((void**)&dev_c, sizeof(int));

    // Run the kernel
    add_kernel<<<1,1>>>(a, b, dev_c);

    // Copy the result back to the host
    cudaMemcpy(c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);

    // Free the memory on the GPU
    cudaFree(dev_c);
}
```

The next step is to compile this CUDA code into a shared library that Python can link against. The compilation process involves both the CUDA compiler (nvcc) and the system's C++ compiler (g++). Here's a simple Makefile that will get this job done:

```makefile
all:
    nvcc -c -o add.o add.cu
    g++ -shared -o libadd.so add.o
```

After running 'make', we'll get a shared library file named '[libadd.so](http://libadd.so/)'. This library contains our CUDA function that we're going to call from Python.

## Interfacing With Python

Now it's time to use Cython to create a Python module for our CUDA function. First, create a file called 'add.pyx' with the following contents:

```python
cdef extern from "add.cu":
    void add(int a, int b, int *c)

def add_integers(int a, int b):
    cdef int c
    add(a, b, &c)
    return c
```

This Cython file does two things:

1. It declares that there's a function named 'add' (from our CUDA code) that we want to call.
2. It defines a Python function named 'add_integers' that Python code can call. This function just wraps the CUDA 'add' function, passing its arguments along and returning its result.

Finally, we need to compile this Cython code into a Python extension module. We'll use a setup script named '[setup.py](http://setup.py/)':

```python
from setuptools import setup
from Cython.Build import cythonize

setup(
    name='CUDA addition',
    ext_modules=cythonize("add.pyx"),
    zip_safe=False,
)
```

You can then build your Cython module using the command 'python [setup.py](http://setup.py/) build_ext --inplace'. This command will create a file named 'add.cpython-XXm-x86_64-linux-gnu.so', which is a Python extension module that you can import in Python using the name 'add'.

If you've made it this far without pulling out your hair, well done! You've traversed the Third Circle, linking Python and CUDA using Cython. I'm afraid to say, dear reader, that the journey is only going to get tougher from here on out. But worry not! After all, aren't we here to endure a little punishment?