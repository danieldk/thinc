ctypedef void (*sgemm_ptr)(bint TransA, bint TransB, int M, int N, int K,
                           float alpha, const float* A, int lda, const float *B,
                           int ldb, float beta, float* C, int ldc) nogil


ctypedef void (*saxpy_ptr)(int N, float alpha, const float* X, int incX,
                           float *Y, int incY) nogil


cdef class CBlas:
    cdef saxpy_ptr saxpy
    cdef sgemm_ptr sgemm
