use core::{
    alloc::{Allocator, Layout},
    ffi::{c_char, c_int, c_void},
    ptr::{self, NonNull},
};

use crate::Ferroc;

#[no_mangle]
#[cfg_attr(not(sys_alloc), export_name = "fe_malloc")]
pub extern "C" fn malloc(size: usize) -> *mut c_void {
    let Ok(layout) = Layout::array::<u8>(size) else {
        return ptr::null_mut();
    };
    Ferroc
        .allocate(layout)
        .map_or(ptr::null_mut(), |ptr| ptr.as_ptr().cast())
}

#[no_mangle]
#[cfg_attr(not(sys_alloc), export_name = "fe_posix_memalign")]
pub unsafe extern "C" fn posix_memalign(
    slot: *mut *mut c_void,
    align: usize,
    size: usize,
) -> c_int {
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
#[cfg_attr(not(sys_alloc), export_name = "fe_aligned_alloc")]
pub extern "C" fn aligned_alloc(align: usize, size: usize) -> *mut c_void {
    let Ok(layout) = Layout::from_size_align(size, align) else {
        return ptr::null_mut();
    };
    Ferroc
        .allocate(layout)
        .map_or(ptr::null_mut(), |ptr| ptr.as_ptr().cast())
}

#[no_mangle]
#[cfg_attr(not(sys_alloc), export_name = "fe_malloc_usable_size")]
pub unsafe extern "C" fn malloc_usable_size(ptr: *mut c_void) -> usize {
    match NonNull::new(ptr) {
        Some(ptr) => Ferroc.layout_of(ptr.cast()).map_or(0, |l| l.size()),
        None => 0,
    }
}

#[no_mangle]
#[cfg_attr(not(sys_alloc), export_name = "fe_free")]
pub unsafe extern "C" fn free(ptr: *mut c_void) {
    if let Some(ptr) = NonNull::new(ptr) {
        unsafe { Ferroc.free(ptr.cast()) }
    }
}

#[no_mangle]
#[cfg_attr(not(sys_alloc), export_name = "fe_calloc")]
pub extern "C" fn calloc(nmemb: usize, size: usize) -> *mut c_void {
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
#[cfg_attr(not(sys_alloc), export_name = "fe_realloc")]
pub unsafe extern "C" fn realloc(ptr: *mut c_void, new_size: usize) -> *mut c_void {
    let Some(ptr) = NonNull::new(ptr) else {
        return malloc(new_size);
    };
    if new_size == 0 {
        free(ptr.as_ptr().cast());
        return malloc(new_size);
    }
    let Some(layout) = Ferroc.layout_of(ptr.cast()) else {
        return ptr::null_mut();
    };
    if (layout.size() / 2..layout.size()).contains(&new_size) {
        return ptr.as_ptr();
    }

    let new = malloc(new_size);
    if !new.is_null() {
        new.copy_from_nonoverlapping(ptr.as_ptr(), layout.size());
        free(ptr.as_ptr().cast());
    }
    new
}

#[no_mangle]
#[cfg_attr(not(sys_alloc), export_name = "fe_strdup")]
pub unsafe extern "C" fn strdup(s: *const c_char) -> *mut c_char {
    let len = libc::strlen(s);
    let ptr = malloc(len + 1).cast::<c_char>();
    if !ptr.is_null() {
        ptr.copy_from_nonoverlapping(s, len);
        ptr.add(len).write(0);
    }
    ptr
}

#[no_mangle]
#[cfg_attr(not(sys_alloc), export_name = "fe_strndup")]
pub unsafe extern "C" fn strndup(s: *const c_char, n: usize) -> *mut c_char {
    let len = libc::strnlen(s, n);
    let ptr = malloc(len + 1).cast::<c_char>();
    if !ptr.is_null() {
        ptr.copy_from_nonoverlapping(s, len);
        ptr.add(len).write(0);
    }
    ptr
}

#[no_mangle]
#[cfg_attr(not(sys_alloc), export_name = "fe_realpath")]
pub unsafe extern "C" fn realpath(name: *const c_char, resolved: *mut c_char) -> *mut c_char {
    if !resolved.is_null() {
        return libc::realpath(name, resolved);
    }
    let r = libc::realpath(name, ptr::null_mut());
    if r.is_null() {
        return ptr::null_mut();
    }
    let dupped = strdup(r);
    libc::free(r as *mut c_void);
    dupped
}
