cimport blis.cy

cdef class CBlas:
    __slots__ = []

    def __init__(self):
        self.saxpy = blis.cy.saxpy
        self.sgemm = blis.cy.sgemm
