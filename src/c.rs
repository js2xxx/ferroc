use core::{
    alloc::{Allocator, Layout},
    ffi::{c_int, c_void},
    ptr::{self, NonNull},
};

use crate::Ferroc;

#[no_mangle]
extern "C" fn fe_malloc(size: usize) -> *mut c_void {
    let Ok(layout) = Layout::array::<u8>(size) else {
        return ptr::null_mut();
    };
    Ferroc
        .allocate(layout)
        .map_or(ptr::null_mut(), |ptr| ptr.as_ptr().cast())
}

#[no_mangle]
unsafe extern "C" fn fe_posix_memalign(slot: *mut *mut c_void, align: usize, size: usize) -> c_int {
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
extern "C" fn fe_aligned_alloc(align: usize, size: usize) -> *mut c_void {
    let Ok(layout) = Layout::from_size_align(size, align) else {
        return ptr::null_mut();
    };
    Ferroc
        .allocate(layout)
        .map_or(ptr::null_mut(), |ptr| ptr.as_ptr().cast())
}

#[no_mangle]
unsafe extern "C" fn fe_free(ptr: *mut c_void) {
    if let Some(ptr) = NonNull::new(ptr) {
        unsafe { Ferroc.free(ptr.cast()) }
    }
}

#[no_mangle]
extern "C" fn fe_calloc(nmemb: usize, size: usize) -> *mut c_void {
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

#[no_mangle]
unsafe extern "C" fn fe_realloc(ptr: *mut c_void, new_size: usize) -> *mut c_void {
    let Some(ptr) = NonNull::new(ptr) else {
        return fe_malloc(new_size);
    };
    if new_size == 0 {
        fe_free(ptr.as_ptr().cast());
        return fe_malloc(new_size);
    }
    let Some(layout) = Ferroc.layout_of(ptr.cast()) else {
        return ptr::null_mut();
    };
    if (layout.size() / 2..layout.size()).contains(&new_size) {
        return ptr.as_ptr();
    }

    let new = fe_malloc(new_size);
    if !new.is_null() {
        new.copy_from_nonoverlapping(ptr.as_ptr(), layout.size());
        fe_free(ptr.as_ptr().cast());
    }
    new
}
