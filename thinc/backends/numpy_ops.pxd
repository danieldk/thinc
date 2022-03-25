ctypedef double[:, ::1] double2d_t
ctypedef double[:, :, ::1] double3d_t
ctypedef float[:, ::1] float2d_t
ctypedef float[:, :, ::1] float3d_t

cdef fused reals2d_ft:
    float2d_t
    double2d_t

cdef fused reals3d_ft:
    float3d_t
    double3d_t


cdef extern from "cpu_kernels.hh":
    void cpu_maxout[A, L](A* best__bo, L* which__bo, const A* cands_bop,
        L B, L O, L P)
    void cpu_reduce_max[A, L](A* maxes__bo, L* which_bo, const A* X__to,
        const L* lengths__b, L B, L T, L O) except +
    void cpu_reduce_mean[A, L](A* means__bo, const A* X__to, const L* lengths__b,
        L B, L T, L O) except +
    void cpu_mish[A, L](A* Y, L N, A threshold)
    void cpu_reduce_sum[A, L](A* sums__bo, const A* X__to, const L* lengths__b,
        L B, L T, L O) except +
    void cpu_relu[A, L](A* X, L N)
    void backprop_seq2col[A, L](A* d_seqs, const A* d_cols, const L* lengths, L B, L I, L nW, L nL)
    void seq2col[A, L](A* output, const A* X, const L* lengths, L nW, L B, L I, L nL)


cdef int cpu_backprop_maxout(float* dX__bop,
        const float* dX__bo, const int* which__bo, int B, int O, int P) nogil except -1

cdef void cpu_backprop_reduce_mean(float* dX__to,
        const float* d_means__bo, const int* lengths__b,
        int B, int T, int O) nogil

cdef int cpu_backprop_reduce_max(float* dX__to,
        const float* d_maxes__bo, const int* which__bo, const int* lengths__b,
        int B, int T, int O) nogil except -1
