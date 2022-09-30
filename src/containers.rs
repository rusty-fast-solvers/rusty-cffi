//! Containers to make working with arrays easier.

use crate::{assert_dtype, ConversionType, DTYPE, MUTABILITY, OWNERSHIP};
use libc::{c_void, size_t};

#[repr(C)]
pub struct DataContainer {
    nelems: size_t,
    itemsize: size_t,
    capacity: size_t,
    dtype: DTYPE,
    owner: OWNERSHIP,
    mutable: MUTABILITY,
    data: *mut c_void,
}

impl DataContainer {
    pub fn from_slice<T: ConversionType>(slice: &[T]) -> Self {
        Self {
            nelems: slice.len(),
            capacity: slice.len(),
            itemsize: crate::get_size::<T>(),
            dtype: crate::get_dtype::<T>(),
            owner: OWNERSHIP::NotOwner,
            mutable: MUTABILITY::NotMutable,
            data: slice.as_ptr() as *mut c_void,
        }
    }
    pub fn from_slice_mut<T: ConversionType>(slice: &mut [T]) -> Self {
        Self {
            nelems: slice.len(),
            capacity: slice.len(),
            itemsize: crate::get_size::<T>(),
            dtype: crate::get_dtype::<T>(),
            owner: OWNERSHIP::NotOwner,
            mutable: MUTABILITY::Mutable,
            data: slice.as_ptr() as *mut c_void,
        }
    }

    pub fn from_vec<T: ConversionType>(mut vec: Vec<T>) -> Self {
        vec.shrink_to_fit();
        let nelems = vec.len();
        let capacity = vec.capacity();
        let data = vec.as_ptr() as *mut c_void;
        std::mem::forget(vec);
        Self {
            nelems,
            capacity,
            itemsize: crate::get_size::<T>(),
            dtype: crate::get_dtype::<T>(),
            owner: OWNERSHIP::NotOwner,
            mutable: MUTABILITY::Mutable,
            data,
        }
    }

    pub unsafe fn destroy<T: ConversionType>(self) {
        assert_dtype::<T>(self.dtype);
        assert_eq!(self.owner, OWNERSHIP::Owner);
        let vec = Vec::<T>::from_raw_parts(self.data as *mut T, self.nelems, self.capacity);
        drop(vec)
    }

    pub unsafe fn to_vec<T: ConversionType>(self) -> Vec<T> {
        assert_eq!(self.owner, OWNERSHIP::Owner);
        assert_dtype::<T>(self.dtype);
        Vec::<T>::from_raw_parts(self.data as *mut T, self.nelems, self.nelems)
    }

    pub unsafe fn to_slice<T: ConversionType>(&self) -> &[T] {
        assert_dtype::<T>(self.dtype);
        std::slice::from_raw_parts::<'static, T>(self.data as *const T, self.nelems)
    }

    pub unsafe fn to_slice_mut<T: ConversionType>(&mut self) -> &'static mut [T] {
        assert_eq!(self.mutable, MUTABILITY::Mutable);
        assert_dtype::<T>(self.dtype);
        std::slice::from_raw_parts_mut::<'static, T>(self.data as *mut T, self.nelems)
    }
}
