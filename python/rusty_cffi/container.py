"""Definition of the container interface."""

import numpy as np


def _rust_to_python(symbol):
    """Convert Rust symbol to corresponding Rust symbols."""
    from .config import get_lib

    lib = get_lib()

    return {
        lib.Float32: ("float32", "float *"),
        lib.Float64: ("float64", "double *"),
        lib.Int8: ("int8", "int8_t *"),
        lib.Int32: ("int32", "int32_t *"),
        lib.Int64: ("int64", "int64_t *"),
        lib.Unsigned8: ("uint8", "uint8_t *"),
        lib.Unsigned32: ("uint32", "uint32_t *"),
        lib.Unsigned64: ("uint64", "uint64_t *"),
        lib.Usize: ("uintp", "uintptr_t *"),
    }[symbol]


def _python_to_rust(symbol):
    """Convert Python symbol to corresponding Rust symbols."""
    from .config import get_lib

    lib = get_lib()

    return {
        "float32": (lib.Float32, lib.rusty_data_container_new_f32),
        "float64": (lib.Float64, lib.rusty_data_container_new_f64),
        "int8": (lib.Int8, lib.rusty_data_container_new_i8),
        "int32": (lib.Int32, lib.rusty_data_container_new_i32),
        "int64": (lib.Int64, lib.rusty_data_container_new_i64),
        "uint8": (lib.Unsigned8, lib.rusty_data_container_new_u8),
        "uint32": (lib.Unsigned32, lib.rusty_data_container_new_u32),
        "uint64": (lib.Unsigned64, lib.rusty_data_container_new_u64),
        "uintp": (lib.Usize, lib.rusty_data_container_new_usize),
    }[symbol]


class RustyDataContainer:
    """A data container interface with Rust."""

    def __init__(self, ptr):
        """Initialize new object from pointer."""
        from .config import get_lib, get_ffi

        lib = get_lib()
        ffi = get_ffi()

        self._ptr = ptr
        self._nitems = lib.rusty_data_container_get_nitems(ptr)
        self._itemsize = lib.rusty_data_container_get_itemsize(ptr)
        self._is_mutable = lib.rusty_data_container_get_is_mutable(ptr)
        self._is_owner = lib.rusty_data_container_get_is_owner(ptr)
        self._dtype = _rust_to_python(lib.rusty_data_container_get_dtype(ptr))[0]

        self._data = np.frombuffer(
            ffi.buffer(
                ffi.cast(
                    _rust_to_python(lib.rusty_data_container_get_dtype(ptr))[1],
                    lib.rusty_data_container_get_data(ptr),
                ),
                self._itemsize * self._nitems,
            ),
            dtype=self._dtype,
        )

        if not self._is_mutable:
            self._data.setflags(write=False)

    def __del__(self):
        """Destructor."""
        from .config import get_lib

        get_lib().rusty_data_container_destroy(self._ptr)

    @property
    def nitems(self):
        """Return the number of items."""
        return self._nitems

    @property
    def itemsize(self):
        """Return the itemsize in bytes."""
        return self._itemsize

    @property
    def dtype(self):
        """Return the type."""
        return np.dtype(self._dtype)

    @property
    def data(self):
        """Data as numpy object."""
        return self._data

    @property
    def c_ptr(self):
        """Return the pointer to the underlying C data structure."""
        return self._ptr

    @classmethod
    def from_array(cls, arr, dtype=None):
        """New data container from numpy array."""
        from .config import get_lib, get_ffi

        if dtype is None:
            dtype = arr.dtype.name

        arr_ptr = get_ffi().cast("void *", arr.ctypes.data)
        is_mutable = 1
        container_ptr = get_lib().new_from_pointer(
            arr_ptr, arr.size, _python_to_rust(dtype)[0], is_mutable
        )
        return cls(container_ptr)

    @classmethod
    def new(cls, nitems, dtype):
        """New data container by specifying number of items and dtype."""
        new_fun = _python_to_rust(dtype)[1]
        return cls(new_fun(nitems))
