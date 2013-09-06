# distutils: language = c++
# distutils: sources = quadtree.cpp tsne.cpp
# distutils: libraries = cblas
# distutils: include_dirs = /System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/Headers

from libcpp cimport bool

cdef extern from "tsne.h":
    cdef cppclass TSNE:
        TSNE(double* X, int N, int D, double* Y, int no_dims, double perplexity, bool exact)
        void run(double theta)
        bool step(double theta)


cdef class Processor(object):
    cdef TSNE* runner
    cdef double[:, ::1] Y
    def __init__(self, double[:, ::1] X not None, double[:, ::1] Y not None, double perplexity=30.0):
        cdef int N = X.shape[0]
        cdef int D = X.shape[1]
        cdef int no_dims = Y.shape[1]

        self.Y = Y
        self.runner = new TSNE(&X[0,0], N, D, &Y[0,0], no_dims, perplexity, False)

    def run(self, theta=0.5):
        while self.runner.step(theta):
            pass

    def step(self, theta=0.5):
        self.runner.step(theta)
