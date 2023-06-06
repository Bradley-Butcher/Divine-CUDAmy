# distutils: language = c++
# distutils: sources = matrix.cu

cdef extern from "add_matrices.h":
    void gpu_add_matrices(int *a, int *b, int *c, const int N)

def add_matrices(a, b):
    cdef int N = len(a)
    cdef int c[N*N]
    gpu_add_matrices(&a[0], &b[0], &c[0], N)
    return c