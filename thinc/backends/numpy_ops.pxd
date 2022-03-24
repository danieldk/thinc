ctypedef double[:, :, ::1] double3d_t
ctypedef float[:, :, ::1] float3d_t


cdef fused reals3d_ft:
    float3d_t
    double3d_t


cdef extern from "cpu_kernels.hh":
    void cpu_maxout[A, L](A* best__bo, L* which__bo, const A* cands_bop,
        L B, L O, L P)
    void cpu_mish[A, L](A* Y, L N, A threshold)
    void cpu_relu[A, L](A* X, L N)


cdef void seq2col(float* output, const float* X, const int* L, int nW, int B, int I, int nL) nogil

cdef void backprop_seq2col(float* d_seqs,
        const float* d_cols, const int* L, int B, int I, int nW, int nL) nogil

cdef int cpu_backprop_maxout(float* dX__bop,
        const float* dX__bo, const int* which__bo, int B, int O, int P) nogil except -1

cdef int cpu_reduce_mean(float* means__bo,
        const float* X__to, const int* lengths__b,
        int B, int T, int O) nogil except -1

cdef void cpu_backprop_reduce_mean(float* dX__to,
        const float* d_means__bo, const int* lengths__b,
        int B, int T, int O) nogil

cdef int cpu_reduce_max(float* maxes__bo, int* which__bo,
        const float* X__to, const int* lengths__b,
        int B, int T, int O) nogil except -1

cdef int cpu_backprop_reduce_max(float* dX__to,
        const float* d_maxes__bo, const int* which__bo, const int* lengths__b,
        int B, int T, int O) nogil except -1
