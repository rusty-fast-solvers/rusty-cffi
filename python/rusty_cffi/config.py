"""Configure FFI"""

from .rusty_cffi import lib as _local_lib
from .rusty_cffi import ffi as _local_ffi

_lib = None
_ffi = None

def get_lib():
    """Get the lib module."""
    return _lib

def get_ffi():
    """Get the ffi module."""
    return _ffi

def config(lib, ffi):
    """Set the lib and ffi libraries."""
    global _lib, _ffi

    _lib = lib
    _ffi = ffi

config(_local_lib, _local_ffi)