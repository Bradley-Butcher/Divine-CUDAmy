cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cu_hello_world LANGUAGES CXX CUDA)
add_executable(cu_hello_world src/cuda/hello.cu)