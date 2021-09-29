from typing import Optional, Callable, Any, Tuple

from ..model import BaseModel, DelegateModel, Model, wrap_with_callbacks
from ..util import use_nvtx_range


def with_nvtx_range(
    layer: Model,
    name: Optional[str] = None,
    *,
    forward_color: int = -1,
    backprop_color: int = -1,
):
    """Layer that wraps any layer and marks the forward and backprop
    phases as NVTX ranges for CUDA profiling.

    By default, the name of the layer is used as the name of the range,
    followed by the name of the pass.
    """
    name = layer.name if name is None else name

    def forward(_model: DelegateModel, X: Any, is_train: bool) -> Tuple[Any, Callable]:
        with use_nvtx_range(f"{name} forward", forward_color):
            layer_Y, layer_callback = layer(X, is_train=is_train)

        def backprop(dY: Any) -> Any:
            with use_nvtx_range(f"{name} backprop", backprop_color):
                return layer_callback(dY)

        return layer_Y, backprop

    def init(_model: DelegateModel, X: Any, Y: Any) -> BaseModel:
        return layer.initialize(X, Y)

    return wrap_with_callbacks(layer, f"debug({name})", forward, init=init)
