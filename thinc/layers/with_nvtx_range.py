from typing import Optional, Callable, Any, Tuple

from ..model import Model
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

    def forward(model: Model, X: Any, is_train: bool) -> Tuple[Any, Callable]:
        with use_nvtx_range(f"{name} forward", forward_color):
            layer_Y, layer_callback = layer(X, is_train=is_train)

        def backprop(dY: Any) -> Any:
            with use_nvtx_range(f"{name} backprop", backprop_color):
                return layer_callback(dY)

        return layer_Y, backprop

    def init(_model: Model, X: Any, Y: Any) -> Model:
        return layer.initialize(X, Y)

    m = Model(
        f"nvtx_range({name})",
        forward,
        init=init,
        layers=[layer],
        shims=layer.shims,
        dims=layer._dims,
    )

    return NVTXWrapper(m)


class NVTXWrapper:
    def __init__(self, model):
        self._model = model

    def __getattr__(self, attr):
        if (
            attr == "begin_update"
            or attr == "initialize"
            or attr == "predict"
            or attr == "finish_update"
        ):
            return getattr(self._model, attr)
        return getattr(self._model.layers[0], attr)

    def __call__(self, X, is_train):
        return self._model(X, is_train)
