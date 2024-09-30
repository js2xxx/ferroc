use core::{
    alloc::{Allocator, Layout},
    ffi::{c_char, c_int, c_void},
    ptr::{self, NonNull},
};

use crate::{Ferroc, base::Mmap};

#[unsafe(no_mangle)]
pub extern "C" fn fe_malloc(size: usize) -> *mut c_void {
    Ferroc
        .malloc(size, false)
        .map_or(ptr::null_mut(), |ptr| ptr.as_ptr().cast())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn fe_posix_memalign(
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
    if let Ok(ptr) = Allocator::allocate(&Ferroc, layout) {
        unsafe { slot.write(ptr.as_ptr().cast()) };
        return 0;
    }
    libc::ENOMEM
}

#[unsafe(no_mangle)]
pub extern "C" fn fe_aligned_alloc(align: usize, size: usize) -> *mut c_void {
    let Ok(layout) = Layout::from_size_align(size, align) else {
        return ptr::null_mut();
    };
    Allocator::allocate(&Ferroc, layout).map_or(ptr::null_mut(), |ptr| ptr.as_ptr().cast())
}

#[unsafe(no_mangle)]
pub extern "C" fn fe_memalign(align: usize, size: usize) -> *mut c_void {
    let Ok(layout) = Layout::from_size_align(size, align) else {
        return ptr::null_mut();
    };
    Allocator::allocate(&Ferroc, layout).map_or(ptr::null_mut(), |ptr| ptr.as_ptr().cast())
}

#[unsafe(no_mangle)]
pub extern "C" fn fe_valloc(size: usize) -> *mut c_void {
    fe_aligned_alloc(Mmap.page_size(), size)
}

#[unsafe(no_mangle)]
pub extern "C" fn fe_pvalloc(size: usize) -> *mut c_void {
    let page_size = Mmap.page_size();
    let rounded_size = (size + page_size - 1) & !(page_size - 1);
    fe_aligned_alloc(page_size, rounded_size)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn fe_malloc_size(ptr: *mut c_void) -> usize {
    match NonNull::new(ptr) {
        Some(ptr) => unsafe { Ferroc.layout_of(ptr.cast()).size() },
        None => 0,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn fe_free(ptr: *mut c_void) {
    if let Some(ptr) = NonNull::new(ptr) {
        unsafe { Ferroc.free(ptr.cast()) }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn fe_calloc(nmemb: usize, size: usize) -> *mut c_void {
    let Some(size) = nmemb.checked_mul(size) else {
        return ptr::null_mut();
    };
    Ferroc
        .malloc(size, true)
        .map_or(ptr::null_mut(), |ptr| ptr.as_ptr().cast())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn fe_realloc(ptr: *mut c_void, new_size: usize) -> *mut c_void {
    let Some(ptr) = NonNull::new(ptr) else {
        return fe_malloc(new_size);
    };
    if new_size == 0 {
        unsafe { fe_free(ptr.as_ptr().cast()) };
        return fe_malloc(new_size);
    }
    let layout = unsafe { Ferroc.layout_of(ptr.cast()) };
    let old_size = layout.size();
    if (old_size / 2..old_size).contains(&new_size) {
        return ptr.as_ptr();
    }

    let new = fe_malloc(new_size);
    if !new.is_null() {
        let copied = old_size.min(new_size);
        unsafe { new.copy_from_nonoverlapping(ptr.as_ptr(), copied) };
        if let Some(zeroed) = new_size.checked_sub(old_size) {
            unsafe { new.add(copied).write_bytes(0, zeroed) };
        }
        unsafe { fe_free(ptr.as_ptr().cast()) };
    }
    new
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn fe_strdup(s: *const c_char) -> *mut c_char {
    let len = unsafe { libc::strlen(s) };
    let ptr = fe_malloc(len + 1).cast::<c_char>();
    if !ptr.is_null() {
        unsafe {
            ptr.copy_from_nonoverlapping(s, len);
            ptr.add(len).write(0);
        }
    }
    ptr
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn fe_strndup(s: *const c_char, n: usize) -> *mut c_char {
    let len = unsafe { libc::strnlen(s, n) };
    let ptr = fe_malloc(len + 1).cast::<c_char>();
    if !ptr.is_null() {
        unsafe {
            ptr.copy_from_nonoverlapping(s, len);
            ptr.add(len).write(0);
        }
    }
    ptr
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn fe_realpath(name: *const c_char, resolved: *mut c_char) -> *mut c_char {
    if !resolved.is_null() {
        return unsafe { libc::realpath(name, resolved) };
    }
    let r = unsafe { libc::realpath(name, ptr::null_mut()) };
    if r.is_null() {
        return ptr::null_mut();
    }
    let dupped = unsafe { fe_strdup(r) };
    unsafe { libc::free(r as *mut c_void) };
    dupped
}

forward! {
    malloc(size: usize) -> *mut c_void => fe_malloc;
    posix_memalign(memptr: *mut *mut c_void, alignment: usize, size: usize) -> c_int
        => fe_posix_memalign;
    aligned_alloc(alignment: usize, size: usize) -> *mut c_void => fe_aligned_alloc;
    memalign(alignment: usize, size: usize) -> *mut c_void => fe_memalign;
    valloc(size: usize) -> *mut c_void => fe_valloc;
    pvalloc(size: usize) -> *mut c_void => fe_pvalloc;
    calloc(count: usize, size: usize) -> *mut c_void => fe_calloc;
    realloc(p: *mut c_void, size: usize) -> *mut c_void => fe_realloc;
    malloc_usable_size(p: *mut c_void) -> usize => fe_malloc_size;
    free(p: *mut c_void) => fe_free;

    __libc_malloc(size: usize) -> *mut c_void => fe_malloc;
    __libc_calloc(count: usize, size: usize) -> *mut c_void => fe_calloc;
    __libc_realloc(p: *mut c_void, size: usize) -> *mut c_void => fe_realloc;
    __libc_free(p: *mut c_void) => fe_free;
    __libc_cfree(p: *mut c_void) => fe_free;
    __libc_valloc(size: usize) -> *mut c_void => fe_valloc;
    __libc_pvalloc(size: usize) -> *mut c_void => fe_pvalloc;
    __libc_memalign(align: usize, size: usize) -> *mut c_void => fe_memalign;
    __posix_memalign(memptr: *mut *mut c_void, alignment: usize, size: usize) -> c_int
        => fe_posix_memalign;
}
