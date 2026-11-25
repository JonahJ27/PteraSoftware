"""This module contains classes to test the functions that find and prepare the optional
GPU libraries."""

import importlib
import unittest
from typing import Any
from unittest.mock import patch

import numpy as np

from pterasoftware import _gpu


class TestRequireWithoutTheLibraries(unittest.TestCase):
    """This is a class with functions to test require and available when the GPU
    libraries or a GPU are missing, which run on any machine."""

    def setUp(self) -> None:
        """Forget any modules an earlier test found, so require imports again."""
        self._saved = _gpu._gpu_modules
        _gpu._gpu_modules = None
        self.addCleanup(setattr, _gpu, "_gpu_modules", self._saved)

    def test_missing_libraries_raise_a_clear_import_error(self) -> None:
        """Test that a missing CuPy or numba-cuda raises an ImportError naming the
        install command, and that available is False."""
        original = importlib.import_module
        for missing in ("cupy", "numba.cuda"):
            with self.subTest(missing=missing):

                def import_module(name: str, *args: Any, **kwargs: Any) -> Any:
                    if name == missing:
                        raise ImportError(f"No module named {name!r}")
                    return original(name, *args, **kwargs)

                with patch.object(importlib, "import_module", import_module):
                    with self.assertRaisesRegex(
                        ImportError, "pip install -r requirements_gpu.txt"
                    ):
                        _gpu.require()
                    self.assertFalse(_gpu.available())
                self.assertIsNone(_gpu._gpu_modules)

    def test_no_gpu_raises_a_runtime_error(self) -> None:
        """Test that installed libraries with no CUDA GPU raise a RuntimeError, and that
        available is False."""

        class FakeCuda:
            @staticmethod
            def is_available() -> bool:
                return False

        def import_module(name: str, *args: Any, **kwargs: Any) -> Any:
            return FakeCuda if name == "numba.cuda" else object()

        with patch.object(importlib, "import_module", import_module):
            with self.assertRaisesRegex(RuntimeError, "CUDA GPU"):
                _gpu.require()
            self.assertFalse(_gpu.available())
        self.assertIsNone(_gpu._gpu_modules)


@unittest.skipUnless(_gpu.available(), "needs CuPy, numba-cuda, and a CUDA GPU")
class TestRequireWithAGpu(unittest.TestCase):
    """This is a class with functions to test require on a machine with the GPU
    libraries and a CUDA GPU."""

    def test_returns_working_modules_once(self) -> None:
        """Test that require returns the cupy and numba.cuda modules, the same ones each
        time, and that a numba-cuda kernel runs on CuPy arrays."""
        cupy, numba_cuda = _gpu.require()
        self.assertIs(_gpu.require()[0], cupy)
        self.assertEqual(cupy.__name__, "cupy")
        self.assertEqual(numba_cuda.__name__, "numba.cuda")

        @numba_cuda.jit
        def add_one(values: Any, out: Any) -> None:
            index = numba_cuda.grid(1)
            if index < values.size:
                out[index] = values[index] + 1.0

        values = cupy.arange(1000, dtype=cupy.float64)
        out = cupy.zeros_like(values)
        add_one[4, 256](values, out)
        np.testing.assert_array_equal(cupy.asnumpy(out), np.arange(1000.0) + 1.0)

    def test_numpy_has_its_row_stack_alias(self) -> None:
        """Test that after require, np.row_stack is np.vstack (the alias numba-cuda
        needs), and that it behaves as np.vstack."""
        _gpu.require()
        self.assertIs(getattr(np, "row_stack"), np.vstack)


if __name__ == "__main__":
    unittest.main()
