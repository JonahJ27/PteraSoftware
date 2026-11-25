"""Contains the functions that find and prepare the optional GPU libraries.

CuPy and numba-cuda aren't dependencies of Ptera Software (see requirements_gpu.txt).
They are imported only the first time GPU support is asked for, so a run that doesn't
use the GPU never imports them, and Ptera Software works on machines without them or
without an NVIDIA GPU.
"""

from __future__ import annotations

import importlib
from types import ModuleType

import numpy as np

# The command that installs the GPU libraries, for the error raised when they're
# missing.
_INSTALL_COMMAND = "pip install -r requirements_gpu.txt"

# CuPy and numba.cuda, once they have been imported and checked, and None until then.
_gpu_modules: tuple[ModuleType, ModuleType] | None = None


def require() -> tuple[ModuleType, ModuleType]:
    """Returns the cupy and numba.cuda modules, importing and checking them the first
    time.

    It exists so that GPU support has one place that imports the optional libraries,
    gives a clear error when they or a GPU are missing, and prepares them, and so that
    nothing imports them unless GPU support is asked for.

    numba-cuda 0.30.4 still registers an overload of np.row_stack, an alias of np.vstack
    that NumPy 2.5 removed, so importing numba.cuda's compiler fails without it. Before
    the first import, this puts the alias back on NumPy when it is missing. The alias is
    what NumPy itself had, so code that doesn't use it is unaffected.

    :return: A tuple of the cupy module and the numba.cuda module.
    :raises ImportError: If CuPy or numba-cuda isn't installed.
    :raises RuntimeError: If they are installed but no CUDA GPU is available.
    """
    global _gpu_modules
    if _gpu_modules is not None:
        return _gpu_modules

    if not hasattr(np, "row_stack"):
        setattr(np, "row_stack", np.vstack)

    try:
        cupy = importlib.import_module("cupy")
        numba_cuda = importlib.import_module("numba.cuda")
    except ImportError as error:
        raise ImportError(
            "GPU support needs CuPy and numba-cuda, which aren't installed. Install "
            f"them with: {_INSTALL_COMMAND}"
        ) from error

    if not numba_cuda.is_available():
        raise RuntimeError(
            "GPU support needs a CUDA GPU, but numba-cuda found none. Check that an "
            "NVIDIA GPU and its drivers are installed."
        )

    _gpu_modules = (cupy, numba_cuda)
    return _gpu_modules


def available() -> bool:
    """Returns whether GPU support can be used: CuPy and numba-cuda are installed and a
    CUDA GPU is available.

    It exists so that callers (and the tests) can check without handling require's
    errors.

    :return: True if require succeeds, and False if it raises.
    """
    try:
        require()
    except (ImportError, RuntimeError):
        return False
    return True
