# distutils: language = c++
# distutils: sources = quadtree.cpp tsne.cpp
# distutils: libraries = cblas
# distutils: include_dirs = /System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/Headers

from libc.stdlib cimport rand, srand

cdef extern from "tsne.h":
    cdef cppclass TSNE:
        TSNE(double* X, int N, int D, double* Y, int no_dims, double perplexity, double theta)
        void run()
        bool step()


def tsne(double[:, ::1] X not None, double[:, ::1] Y not None, double perplexity=30.0, double theta=0.5):
    cdef int N = X.shape[0]
    cdef int D = X.shape[1]
    cdef int no_dims = Y.shape[1]

    cdef TSNE* runner = new TSNE(&X[0,0], N, D, &Y[0,0], no_dims, perplexity, theta)
    try:
        runner.run()
    finally:
        del runner

def crand():
    return rand()
