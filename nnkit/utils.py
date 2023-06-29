from collections.abc import Callable, Iterable
import numpy as np


def numpy_deep_zeros_like(complex_ndarray: np.ndarray, dtype=object) -> np.ndarray:
    return numpy_deep_apply_func(np.zeros_like, complex_ndarray, dtype=dtype)


def numpy_deep_full_like(
    complex_ndarray: np.ndarray,
    fill_value: np.ndarray | Iterable | int | float,
    dtype=object
) -> np.ndarray:
    return numpy_deep_apply_func(lambda ndarray: np.full_like(ndarray, fill_value), complex_ndarray, dtype=dtype)


def numpy_deep_apply_func(
    numpy_func: Callable[[np.ndarray], np.ndarray],
    complex_ndarray: np.ndarray,
    dtype=object
) -> np.ndarray:
    return np.array([
        numpy_func(array)
        for array in complex_ndarray
    ], dtype=dtype)
