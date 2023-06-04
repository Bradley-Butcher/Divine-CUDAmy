# *The Dark Wood of Errors*: Installing CUDA

Welcome, intrepid wayfarer, to the murky depths of the CUDA forest. Here be dragons, and segmentation faults, and don't even get me started on the debugging trolls. But fear not! I shall be your unworthy guide, leading you astray... err... I mean along the winding path of CUDA setup.

Let's venture forth, shall we?

## Installing CUDA

First things first, we have to tame the beast that is CUDA. For those of you sporting a Windows crest, take yourself to the green-themed NVIDIA website, locate the CUDA Toolkit, and download the latest version. Execute the downloaded file and watch as CUDA bends to your will (or crashes spectacularly, whichever comes first).

Now, my dear Linux acolytes, your path is slightly less traveled. You're required to recite these sacred lines to your command-line deity:

```bash
sudo apt-get install nvidia-cuda-toolkit
```

And to those proudly brandishing the Apple sigil... Well, there's no easy way to say this but NVIDIA stopped supporting CUDA for MacOS after version 10.2. You might want to reconsider your allegiance.

## Compiling C++/CUDA Programs

With CUDA installed, we must now prepare for battle by forging our weapon, the compiler. NVCC is the CUDA Compiler from NVIDIA, capable of transforming your innocent C++ code into CUDA executables.

CMake is our trusty assistant here, a build system that kindly takes care of all the mundane chores. Your CMakeLists.txt file is the instruction manual that CMake follows to build your project. Let's write a simple one:

```makefile
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cu_hello_world LANGUAGES CXX CUDA)
add_executable(cu_hello_world src/cuda/hello.cu)
```

Here, we're telling CMake a few important things. `cmake_minimum_required(VERSION 3.8 FATAL_ERROR)` states the minimum version of CMake required to build our project. If a user tries to build your project with a CMake version less than 3.8, the build process will fail with a fatal error.

With `project(cu_hello_world LANGUAGES CXX CUDA)`, we're declaring a new project named "cu_hello_world" and specifying that it will contain both C++ (CXX) and CUDA code.

Finally, `add_executable(cu_hello_world src/cuda/hello.cu)` is the command that adds an executable target to our project. This executable is called "cu_hello_world" and will be built from the `src/cuda/hello.cu` source file.

## Setting Up Your Project

Lastly, we need to form some sort of project structure. I'm used to automated Python tooling doing this all for me. Alas, I know so little about C++ that I don't know what tools exist. Anyway, here’s a suggestion:

```bash
/my_cuda_project
    /src
        /cuda
            hello.cu
```

And for the content of `hello.cu`, let's keep things simple:

```c++
#include <stdio.h>

__global__ void helloFromGPU(void) {
    printf("Hello World from GPU!\\n");
}

int main(void) {
    printf("Hello World from CPU!\\n");

    helloFromGPU<<<1, 10>>>();
    cudaDeviceReset();

    return 0;
}

```

To get this all running either type the following lines in one-by-one, or be smart and add them to a bash script called `build.sh` or something similar:

```bash
mkdir build
cd build
cmake ..
make
./cu_hello_world
```

This program will print "Hello World from CPU!" from the CPU, and "Hello World from GPU!" from the GPU. `<<<1, 10>>>` is CUDA syntax for launching a kernel on the GPU. This specific configuration means we are launching 1 block of 10 threads. 
The details aren't important - just make sure it runs to test your install for now. Don't worry, we'll dive deeper into CUDA programming, and what all these weird makes and builds are in the upcoming chapters. A full compilable (maybe) repository of each chapter can be found on github - unless I get lazy that is! 

The one for this chapter should be [here](../code/Chapter1/).

Onward, my brave, foolhardy friend! The journey has only just begun.