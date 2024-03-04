#[cfg(not(sys_alloc))]
use core::ffi::c_char;
use core::{
    alloc::Layout,
    ffi::{c_int, c_void},
    ptr::{self, NonNull},
};

use crate::{base::Mmap, Ferroc};

#[no_mangle]
#[cfg_attr(not(sys_alloc), export_name = "fe_malloc")]
pub extern "C" fn malloc(size: usize) -> *mut c_void {
    Ferroc
        .malloc(size, false)
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
#[cfg_attr(not(sys_alloc), export_name = "fe_memalign")]
pub extern "C" fn memalign(align: usize, size: usize) -> *mut c_void {
    let Ok(layout) = Layout::from_size_align(size, align) else {
        return ptr::null_mut();
    };
    Ferroc
        .allocate(layout)
        .map_or(ptr::null_mut(), |ptr| ptr.as_ptr().cast())
}

#[no_mangle]
#[cfg_attr(not(sys_alloc), export_name = "fe_valloc")]
pub extern "C" fn valloc(size: usize) -> *mut c_void {
    aligned_alloc(Mmap.page_size(), size)
}

#[no_mangle]
#[cfg_attr(not(sys_alloc), export_name = "fe_pvalloc")]
pub extern "C" fn pvalloc(size: usize) -> *mut c_void {
    let page_size = Mmap.page_size();
    let rounded_size = (size + page_size - 1) & !(page_size - 1);
    aligned_alloc(page_size, rounded_size)
}

#[no_mangle]
#[cfg_attr(not(sys_alloc), export_name = "fe_malloc_usable_size")]
pub unsafe extern "C" fn malloc_usable_size(ptr: *mut c_void) -> usize {
    match NonNull::new(ptr) {
        Some(ptr) => Ferroc.layout_of(ptr.cast()).size(),
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
    Ferroc
        .malloc(size, true)
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
    let layout = Ferroc.layout_of(ptr.cast());
    let old_size = layout.size();
    if (old_size / 2..old_size).contains(&new_size) {
        return ptr.as_ptr();
    }

    let new = malloc(new_size);
    if !new.is_null() {
        let copied = old_size.min(new_size);
        new.copy_from_nonoverlapping(ptr.as_ptr(), copied);
        if let Some(zeroed) = new_size.checked_sub(old_size) {
            new.add(copied).write_bytes(0, zeroed);
        }
        free(ptr.as_ptr().cast());
    }
    new
}

#[no_mangle]
#[cfg(not(sys_alloc))]
#[export_name = "fe_strdup"]
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
#[cfg(not(sys_alloc))]
#[export_name = "fe_strndup"]
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
#[cfg(not(sys_alloc))]
#[export_name = "fe_realpath"]
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

#[cfg(sys_alloc)]
mod forward {
    use super::*;

    macro_rules! forward {
        (@IMPL $name:ident($($aname:ident, $atype:ty),*) $(-> $ret:ty)? => $target:ident) => {
            #[no_mangle]
            pub unsafe extern "C" fn $name($($aname: $atype),*) $(-> $ret)? {
                $target($($aname),*)
            }
        };
        ($($name:ident($($aname:ident: $atype:ty),*) $(-> $ret:ty)? => $target:ident;)*) => {
            $(forward!(@IMPL $name($($aname, $atype),*) $(-> $ret)? => $target);)*
        };
    }

    forward! {
        __libc_malloc(size: usize) -> *mut c_void => malloc;
        __libc_calloc(count: usize, size: usize) -> *mut c_void => calloc;
        __libc_realloc(p: *mut c_void, size: usize) -> *mut c_void => realloc;
        __libc_free(p: *mut c_void) => free;
        __libc_cfree(p: *mut c_void) => free;
        __libc_valloc(size: usize) -> *mut c_void => valloc;
        __libc_pvalloc(size: usize) -> *mut c_void => pvalloc;
        __libc_memalign(align: usize, size: usize) -> *mut c_void => memalign;
        __posix_memalign(memptr: *mut *mut c_void, alignment: usize, size: usize) -> c_int
            => posix_memalign;
    }
}
