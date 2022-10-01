//! Container to make working with arrays easier.
//!
//! This module defines a type [DataContainer], which
//! is a simple C data structure that can be flexibly converted to
//! various Rust types.

use crate::iterate_over_type;
use crate::{assert_dtype, ConversionType, DTYPE, MUTABILITY, OWNERSHIP};
use libc::{c_void, size_t};
use paste::paste;

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
    owner: OWNERSHIP,
    /// Mutability of the underlying data. It is either
    /// [MUTABILITY::Mutable] or [MUTABILITY::NotMutable].
    mutable: MUTABILITY,
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
            owner: OWNERSHIP::NotOwner,
            mutable: MUTABILITY::NotMutable,
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
            owner: OWNERSHIP::NotOwner,
            mutable: MUTABILITY::Mutable,
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
            owner: OWNERSHIP::Owner,
            mutable: MUTABILITY::Mutable,
            data,
        }
    }

    ///
    pub unsafe fn to_vec<T: ConversionType>(self) -> Vec<T> {
        assert_eq!(self.owner, OWNERSHIP::Owner);
        assert_dtype::<T>(self.dtype);
        Vec::<T>::from_raw_parts(self.data as *mut T, self.nitems, self.capacity)
    }

    pub unsafe fn to_slice<T: ConversionType>(&self) -> &[T] {
        assert_dtype::<T>(self.dtype);
        std::slice::from_raw_parts::<'static, T>(self.data as *const T, self.nitems)
    }

    pub unsafe fn to_slice_mut<T: ConversionType>(&mut self) -> &'static mut [T] {
        assert_eq!(self.mutable, MUTABILITY::Mutable);
        assert_dtype::<T>(self.dtype);
        std::slice::from_raw_parts_mut::<'static, T>(self.data as *mut T, self.nitems)
    }
}

impl Drop for DataContainer {
    /// Destroy a data container. If the container owns the
    /// data the corresponding memory is also deallocated.
    fn drop(&mut self) {
        if let OWNERSHIP::Owner = self.owner {
            let len = self.nitems * self.itemsize;
            let cap = self.capacity * self.itemsize;
            let vec = unsafe { Vec::<u8>::from_raw_parts(self.data as *mut u8, len, cap) };
            drop(vec)
        }
    }
}

/// Destroy a data container.
#[no_mangle]
pub extern "C" fn data_container_destroy(_: Option<Box<DataContainer>>) {}

macro_rules! c_new_container {
    ($dtype:ident) => {
        paste! {
            #[doc = "Create a new `" $dtype "` data container."]
            #[no_mangle]
            pub extern "C" fn [<data_container_new_ $dtype>](nitems: size_t) -> Box<DataContainer> {
                Box::new(DataContainer::from_vec(vec![0 as $dtype; nitems]))
            }

        }
    };
}

iterate_over_type!(c_new_container);
