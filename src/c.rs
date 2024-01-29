use core::{
    alloc::{Allocator, Layout},
    ffi::{c_int, c_void},
    ptr::{self, NonNull},
};

use crate::Ferroc;

#[no_mangle]
extern "C" fn malloc(size: usize) -> *mut c_void {
    let Ok(layout) = Layout::array::<u8>(size) else {
        return ptr::null_mut();
    };
    Ferroc
        .allocate(layout)
        .map_or(ptr::null_mut(), |ptr| ptr.as_ptr().cast())
}

#[no_mangle]
unsafe extern "C" fn posix_memalign(slot: *mut *mut c_void, align: usize, size: usize) -> c_int {
    let Ok(layout) = Layout::from_size_align(size, align) else {
        return libc::EINVAL;
    };
    if slot.is_null() {
        return libc::EINVAL;
    }
    if let Ok(ptr) = Ferroc.allocate(layout) {
        slot.write(ptr.as_ptr().cast());
        return 0;
    }
    libc::ENOMEM
}

#[no_mangle]
extern "C" fn aligned_alloc(align: usize, size: usize) -> *mut c_void {
    let Ok(layout) = Layout::from_size_align(size, align) else {
        return ptr::null_mut();
    };
    Ferroc
        .allocate(layout)
        .map_or(ptr::null_mut(), |ptr| ptr.as_ptr().cast())
}

#[no_mangle]
unsafe extern "C" fn free(ptr: *mut c_void) {
    if let Some(ptr) = NonNull::new(ptr) {
        unsafe { Ferroc.free(ptr.cast()) }
    }
}

#[no_mangle]
extern "C" fn calloc(nmemb: usize, size: usize) -> *mut c_void {
    let Some(size) = nmemb.checked_mul(size) else {
        return ptr::null_mut();
    };
    let Ok(layout) = Layout::array::<u8>(size) else {
        return ptr::null_mut();
    };
    Ferroc
        .allocate_zeroed(layout)
        .map_or(ptr::null_mut(), |ptr| ptr.as_ptr().cast())
}
