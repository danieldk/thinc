from typing import Optional
import numpy

from . import apple_blas
from .apple_blas cimport saxpy, sgemm
from .cblas cimport CBlas, set_saxpy, set_sgemm
from .numpy_ops import NumpyOps
from ..types import Floats2d
from .. import registry


@registry.ops("AppleOps")
class AppleOps(NumpyOps):
    """Thinc Ops class that calls into Apple's native libraries for some
    operations. Other operations fall back to numpy."""
    name = "apple"
    xp = numpy

    def cblas(self) -> CBlas:
        cdef CBlas cblas = CBlas()
        set_saxpy(cblas, saxpy)
        set_sgemm(cblas, sgemm)
        return cblas

    def gemm(
        self,
        x: Floats2d,
        y: Floats2d,
        out: Optional[Floats2d] = None,
        trans1: bool = False,
        trans2: bool = False,
    ) -> Floats2d:
        """Perform General Matrix Multiplication (GeMM) and optionally store
        the result in the specified output variable.
        """
        return apple_blas.gemm(x, y, out=out, trans1=trans1, trans2=trans2)
