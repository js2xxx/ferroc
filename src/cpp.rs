use core::{
    alloc::{Allocator, Layout},
    ffi::{c_int, c_void},
    ptr::{self, NonNull},
};

use crate::Ferroc;
#[cfg(feature = "c-override")]
use crate::c::fe_free;

#[repr(C)]
pub struct NoThrow {
    _tag: c_int,
}

unsafe fn new_handler(nothrow: bool) -> bool {
    type CppNewHandler = unsafe extern "C" fn();

    #[linkage = "weak"]
    #[cfg_attr(unix, unsafe(export_name = "_ZSt15get_new_handlerv"))]
    #[cfg_attr(windows, unsafe(export_name = "?get_new_handler@std@@YAP6AXXZXZ"))]
    extern "C" fn get_new_handler() -> Option<CppNewHandler> {
        None
    }

    let Some(f) = get_new_handler() else {
        return if nothrow { false } else { unreachable!() };
    };
    unsafe { f() };
    true
}

unsafe fn try_new(size: usize, nothrow: bool) -> *mut c_void {
    loop {
        if let Some(p) = Ferroc.malloc(size, false) {
            break p.as_ptr().cast();
        }
        if !unsafe { new_handler(nothrow) } {
            break ptr::null_mut();
        }
    }
}

unsafe fn try_allocate(layout: Layout, nothrow: bool) -> *mut c_void {
    loop {
        if let Ok(p) = Allocator::allocate(&Ferroc, layout) {
            break p.as_ptr().cast();
        }
        if !unsafe { new_handler(nothrow) } {
            break ptr::null_mut();
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn fe_new(size: usize) -> *mut c_void {
    match Ferroc.malloc(size, false) {
        Some(ptr) => ptr.as_ptr().cast(),
        None => unsafe { try_new(size, false) },
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn fe_new_nothrow(size: usize, _: NoThrow) -> *mut c_void {
    match Ferroc.malloc(size, false) {
        Some(ptr) => ptr.as_ptr().cast(),
        None => unsafe { try_new(size, true) },
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn fe_alloc(size: usize, align: usize) -> *mut c_void {
    let Ok(layout) = Layout::from_size_align(size, align) else {
        unsafe { new_handler(false) };
        return ptr::null_mut();
    };
    match Allocator::allocate(&Ferroc, layout) {
        Ok(ptr) => ptr.as_ptr().cast(),
        Err(_) => unsafe { try_allocate(layout, false) },
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn fe_alloc_nothrow(size: usize, align: usize, _: NoThrow) -> *mut c_void {
    let Ok(layout) = Layout::from_size_align(size, align) else {
        unsafe { new_handler(true) };
        return ptr::null_mut();
    };
    match Allocator::allocate(&Ferroc, layout) {
        Ok(ptr) => ptr.as_ptr().cast(),
        Err(_) => unsafe { try_allocate(layout, true) },
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn fe_free_sized(p: *mut c_void, size: usize) {
    if let Some(ptr) = NonNull::new(p)
        && let Ok(layout) = Layout::from_size_align(size, 1)
    {
        unsafe { Ferroc.deallocate(ptr.cast(), layout) }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn fe_free_aligned(p: *mut c_void, align: usize) {
    if let Some(ptr) = NonNull::new(p)
        && let Ok(layout) = Layout::from_size_align(1, align)
    {
        unsafe { Ferroc.deallocate(ptr.cast(), layout) }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn fe_dealloc(p: *mut c_void, size: usize, align: usize) {
    if let Some(ptr) = NonNull::new(p)
        && let Ok(layout) = Layout::from_size_align(size, align)
    {
        unsafe { Ferroc.deallocate(ptr.cast(), layout) }
    }
}

forward! {
    _ZdlPv(p: *mut c_void) => fe_free;
    _ZdaPv(p: *mut c_void) => fe_free;
    _ZdlPvm(p: *mut c_void, size: usize) => fe_free_sized;
    _ZdaPvm(p: *mut c_void, size: usize) => fe_free_sized;
    _ZdlPvSt11align_val_t(p: *mut c_void, align: usize) => fe_free_aligned;
    _ZdaPvSt11align_val_t(p: *mut c_void, align: usize) => fe_free_aligned;
    _ZdlPvmSt11align_val_t(p: *mut c_void, size: usize, align: usize) => fe_dealloc;
    _ZdaPvmSt11align_val_t(p: *mut c_void, size: usize, align: usize) => fe_dealloc;
}

#[cfg(target_pointer_width = "64")]
forward! {
    _Znwm(size: usize) -> *mut c_void => fe_new;
    _Znam(size: usize) -> *mut c_void => fe_new;
    _ZnwmRKSt9nothrow_t(size: usize, nothrow: NoThrow) -> *mut c_void => fe_new_nothrow;
    _ZnamRKSt9nothrow_t(size: usize, nothrow: NoThrow) -> *mut c_void => fe_new_nothrow;
    _ZnwmSt11align_val_t(size: usize, align: usize) -> *mut c_void => fe_alloc;
    _ZnamSt11align_val_t(size: usize, align: usize) -> *mut c_void => fe_alloc;
    _ZnwmRKSt11align_val_tS1_St9nothrow_t(size: usize, align: usize, nothrow: NoThrow)
        -> *mut c_void => fe_alloc_nothrow;
    _ZnamRKSt11align_val_tS1_St9nothrow_t(size: usize, align: usize, nothrow: NoThrow)
        -> *mut c_void => fe_alloc_nothrow;
}

#[cfg(target_pointer_width = "32")]
forward! {
    _Znwj(size: usize) -> *mut c_void => fe_new;
    _Znaj(size: usize) -> *mut c_void => fe_new;
    _ZnwjRKSt9nothrow_t(size: usize, nothrow: NoThrow) -> *mut c_void => fe_new_nothrow;
    _ZnajRKSt9nothrow_t(size: usize, nothrow: NoThrow) -> *mut c_void => fe_new_nothrow;
    _ZnwjSt11align_val_t(size: usize, align: usize) -> *mut c_void => fe_alloc;
    _ZnajSt11align_val_t(size: usize, align: usize) -> *mut c_void => fe_alloc;
    _ZnwjRKSt11align_val_tS1_St9nothrow_t(size: usize, align: usize, nothrow: NoThrow)
        -> *mut c_void => fe_alloc_nothrow;
    _ZnajRKSt11align_val_tS1_St9nothrow_t(size: usize, align: usize, nothrow: NoThrow)
        -> *mut c_void => fe_alloc_nothrow;
}
