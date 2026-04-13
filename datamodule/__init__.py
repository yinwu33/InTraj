from __future__ import annotations

from importlib import import_module


_DATAMODULE_IMPORTS = {
    "AV2VectorNetDatamodule": ".av2_vectornet",
    "AV2SimplDatamodule": ".av2_simpl",
    "SimplDatamodule": ".simpl_datamodule",
    "AV2QCNetDatamodule": ".av2_qcnet",
    "AV2SMARTDatamodule": ".av2_smart",
}


def __getattr__(name: str):
    if name not in _DATAMODULE_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_DATAMODULE_IMPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


__all__ = list(_DATAMODULE_IMPORTS)
