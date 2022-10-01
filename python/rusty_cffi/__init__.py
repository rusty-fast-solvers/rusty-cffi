"""
Rusty-cffi - Tools for Python/Rust interaction.abs
"""

from .rusty_cffi import ffi, lib

import numpy as np

_RUST_TO_PYTHON = {
    lib.Float32: ("float32", "float *"),
    lib.Float64: ("float64", "double *"),
    lib.Int8: ("int8", "int8_t *"),
    lib.Int32: ("int32", "int32_t *"),
    lib.Int64: ("int64", "int64_t *"),
    lib.Unsigned8: ("uint8", "uint8_t *"),
    lib.Unsigned32: ("uint32", "uint32_t *"),
    lib.Unsigned64: ("uint64", "uint64_t *"),
}

_PYTHON_TO_RUST = {
    "float32": (lib.Float32, lib.data_container_new_f32),
    "float64": (lib.Float64, lib.data_container_new_f64),
    "int8": (lib.Int8, lib.data_container_new_i8),
    "int32": (lib.Int32, lib.data_container_new_i32),
    "int64": (lib.Int64, lib.data_container_new_i64),
    "uint8": (lib.Unsigned8, lib.data_container_new_u8),
    "uint32": (lib.Unsigned32, lib.data_container_new_u32),
    "uint64": (lib.Unsigned64, lib.data_container_new_u64),
}

import numpy as np


class DataContainer:
    """A data container interface with Rust."""

    def __init__(self, ptr):
        """Initialize new object from pointer."""
        self._ptr = ptr

        self._data = np.frombuffer(
            ffi.buffer(
                ffi.cast(_RUST_TO_PYTHON[ptr.dtype][1], ptr.data),
                ptr.itemsize * ptr.nitems,
            ),
            dtype = _RUST_TO_PYTHON[ptr.dtype][0]
        )

        if not ptr.is_mutable:
            self._data.setflags(write=False)

    def __del__(self):
        """Destructor."""
        lib.data_container_destroy(self._ptr)

    @property
    def nitems(self):
        """Return the number of items."""
        return self._ptr.nitems

    @property
    def capacity(self):
        """Return the capacity."""
        return self._ptr.capacity

    @property
    def itemsize(self):
        """Return the itemsize in bytes."""
        return self._ptr.itemsize

    @property
    def dtype(self):
        """Return the type."""
        return np.dtype(_RUST_TO_PYTHON[self._ptr.dtype][0])

    @property
    def data(self):
        """Data as numpy object."""
        return self._data

    @classmethod
    def from_array(cls, arr):
        """New data container from numpy array."""
        arr_ptr = ffi.cast("void *", arr.ctypes.data)
        is_mutable = 1
        container_ptr = lib.new_from_pointer(
            arr_ptr, arr.size, arr.size, _PYTHON_TO_RUST[arr.dtype.name][0], is_mutable
        )
        return cls(container_ptr)

    @classmethod
    def new(cls, nitems, dtype):
        """New data container by specifying number of items and dtype."""

        new_fun = _PYTHON_TO_RUST[dtype][1]
        return cls(new_fun(nitems))

