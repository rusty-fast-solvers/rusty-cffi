//! Container to make working with arrays easier.
//!
//! This module defines a type [DataContainer], which
//! is a simple C data structure that can be flexibly converted to
//! various Rust types.

use crate::{assert_dtype, ConversionType, DTYPE, MUTABILITY, OWNERSHIP};
use libc::{c_void, size_t};

/// A data container for communication with a C ABI.
///
/// The basic task of this container is to provide a C ABI
/// compatible type to store arbitrary data arrays, and
/// to convert them back and forth into corresponding Rust types.
#[repr(C)]
pub struct DataContainer {
    /// The number of elements in the Array.
    nitems: size_t,
    /// The size in bytes of each element.
    itemsize: size_t,
    /// The capacity of the underlying array.
    /// This can be larger than the number of items and
    /// denotes the actual reserved memory.
    capacity: size_t,
    /// The type of the data.
    dtype: DTYPE,
    /// The ownership of the data. It is either
    /// [OWNERSHIP::Owner] or [OWNERSHIP::NotOwner].
    /// The underlying data can only be destroyed if
    /// [DataContainer] is owner.
    is_owner: OWNERSHIP,
    /// Mutability of the underlying data. It is either
    /// [MUTABILITY::Mutable] or [MUTABILITY::NotMutable].
    is_mutable: MUTABILITY,
    /// A pointer to the underlying data.
    data: *mut c_void,
}

impl DataContainer {
    /// Create a new non-owning and non-mutable container from a given slice.
    pub fn from_slice<T: ConversionType>(slice: &[T]) -> Self {
        Self {
            nitems: slice.len(),
            capacity: slice.len(),
            itemsize: crate::get_size::<T>(),
            dtype: crate::get_dtype::<T>(),
            is_owner: OWNERSHIP::NotOwner,
            is_mutable: MUTABILITY::NotMutable,
            data: slice.as_ptr() as *mut c_void,
        }
    }
    /// Create a new non-owning but mutable container from a given slice.
    pub fn from_slice_mut<T: ConversionType>(slice: &mut [T]) -> Self {
        Self {
            nitems: slice.len(),
            capacity: slice.len(),
            itemsize: crate::get_size::<T>(),
            dtype: crate::get_dtype::<T>(),
            is_owner: OWNERSHIP::NotOwner,
            is_mutable: MUTABILITY::Mutable,
            data: slice.as_ptr() as *mut c_void,
        }
    }

    /// Create a new owning and mutable container from a vector.
    /// The vector is consumed by this method.
    pub fn from_vec<T: ConversionType>(vec: Vec<T>) -> Self {
        let nitems = vec.len();
        let capacity = vec.capacity();
        let data = vec.as_ptr() as *mut c_void;
        std::mem::forget(vec);
        Self {
            nitems,
            capacity,
            itemsize: crate::get_size::<T>(),
            dtype: crate::get_dtype::<T>(),
            is_owner: OWNERSHIP::Owner,
            is_mutable: MUTABILITY::Mutable,
            data,
        }
    }

    ///
    pub unsafe fn to_vec<T: ConversionType>(self) -> Vec<T> {
        assert_eq!(self.is_owner, OWNERSHIP::Owner);
        assert_dtype::<T>(self.dtype);
        Vec::<T>::from_raw_parts(self.data as *mut T, self.nitems, self.capacity)
    }

    pub unsafe fn to_slice<T: ConversionType>(&self) -> &[T] {
        assert_dtype::<T>(self.dtype);
        std::slice::from_raw_parts::<'static, T>(self.data as *const T, self.nitems)
    }

    pub unsafe fn to_slice_mut<T: ConversionType>(&mut self) -> &'static mut [T] {
        assert_eq!(self.is_mutable, MUTABILITY::Mutable);
        assert_dtype::<T>(self.dtype);
        std::slice::from_raw_parts_mut::<'static, T>(self.data as *mut T, self.nitems)
    }
}

impl Drop for DataContainer {
    /// Destroy a data container. If the container owns the
    /// data the corresponding memory is also deallocated.
    fn drop(&mut self) {
        if let OWNERSHIP::Owner = self.is_owner {
            let len = self.nitems * self.itemsize;
            let cap = self.capacity * self.itemsize;
            let vec = unsafe { Vec::<u8>::from_raw_parts(self.data as *mut u8, len, cap) };
            drop(vec);
        }
    }
}

/// Destroy a data container.
#[no_mangle]
pub extern "C" fn data_container_destroy(_: Option<Box<DataContainer>>) {}

/// Create a new f32 data container.
#[no_mangle]
pub extern "C" fn data_container_new_f32(nitems: size_t) -> Box<DataContainer> {
    Box::new(DataContainer::from_vec(vec![0 as f32; nitems]))
}

/// Create a new f64 data container.
#[no_mangle]
pub extern "C" fn data_container_new_f64(nitems: size_t) -> Box<DataContainer> {
    Box::new(DataContainer::from_vec(vec![0 as f64; nitems]))
}

/// Create a new u8 data container.
#[no_mangle]
pub extern "C" fn data_container_new_u8(nitems: size_t) -> Box<DataContainer> {
    Box::new(DataContainer::from_vec(vec![0 as u8; nitems]))
}

/// Create a new u32 data container.
#[no_mangle]
pub extern "C" fn data_container_new_u32(nitems: size_t) -> Box<DataContainer> {
    Box::new(DataContainer::from_vec(vec![0 as u32; nitems]))
}

/// Create a new u64 data container.
#[no_mangle]
pub extern "C" fn data_container_new_u64(nitems: size_t) -> Box<DataContainer> {
    Box::new(DataContainer::from_vec(vec![0 as u64; nitems]))
}

/// Create a new i8 data container.
#[no_mangle]
pub extern "C" fn data_container_new_i8(nitems: size_t) -> Box<DataContainer> {
    Box::new(DataContainer::from_vec(vec![0 as i8; nitems]))
}

/// Create a new i32 data container.
#[no_mangle]
pub extern "C" fn data_container_new_i32(nitems: size_t) -> Box<DataContainer> {
    Box::new(DataContainer::from_vec(vec![0 as i32; nitems]))
}

/// Create a new i64 data container.
#[no_mangle]
pub extern "C" fn data_container_new_i64(nitems: size_t) -> Box<DataContainer> {
    Box::new(DataContainer::from_vec(vec![0 as i64; nitems]))
}

#[no_mangle]
pub extern "C" fn get_itemsize(dtype: DTYPE) -> size_t {
    match dtype {
        DTYPE::Float32 => crate::get_size::<f32>(),
        DTYPE::Float64 => crate::get_size::<f64>(),
        DTYPE::Unsigned8 => crate::get_size::<u8>(),
        DTYPE::Unsigned32 => crate::get_size::<u32>(),
        DTYPE::Unsigned64 => crate::get_size::<u64>(),
        DTYPE::Int8 => crate::get_size::<i8>(),
        DTYPE::Int32 => crate::get_size::<i32>(),
        DTYPE::Int64 => crate::get_size::<i64>(),
    }
}

#[no_mangle]
pub extern "C" fn new_from_pointer(
    ptr: *mut c_void,
    nitems: size_t,
    capacity: size_t,
    dtype: DTYPE,
    is_mutable: MUTABILITY,
) -> Box<DataContainer> {
    Box::new(DataContainer {
        nitems,
        capacity,
        itemsize: get_itemsize(dtype),
        dtype,
        is_owner: OWNERSHIP::NotOwner,
        is_mutable,
        data: ptr,
    })
}

// The following code is not recognized by maturin without macro expansion.
// However, macro expansion uses the `cargo expand` command, which depends
// on a nightly toolchain.

// macro_rules! c_new_container {
//     ($dtype:ident) => {
//         paste! {
//             #[doc = "Create a new `" $dtype "` data container."]
//             #[no_mangle]
//             pub extern "C" fn [<data_container_new_ $dtype>](nitems: size_t) -> Box<DataContainer> {
//                 Box::new(DataContainer::from_vec(vec![0 as $dtype; nitems]))
//             }

//         }
//     };
// }

// iterate_over_type!(c_new_container);
