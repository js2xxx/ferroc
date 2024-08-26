#[cfg(feature = "track-valgrind")]
mod imp {
    use core::ptr::NonNull;

    use crabgrind::memcheck::*;

    pub fn allocate(ptr: NonNull<[u8]>, red_zone: usize, is_zeroed: bool) {
        alloc::malloc(ptr.as_ptr().cast(), ptr.len(), red_zone, is_zeroed)
    }

    pub fn deallocate(ptr: NonNull<u8>, red_zone: usize) {
        alloc::free(ptr.as_ptr().cast(), red_zone)
    }

    pub fn undefined(ptr: NonNull<u8>, size: usize) {
        let _ = mark_mem(ptr.as_ptr().cast(), size, MemState::Undefined);
    }

    pub fn defined(ptr: NonNull<u8>, size: usize) {
        let _ = mark_mem(ptr.as_ptr().cast(), size, MemState::Defined);
    }

    pub fn no_access(ptr: NonNull<u8>, size: usize) {
        let _ = mark_mem(ptr.as_ptr().cast(), size, MemState::NoAccess);
    }
}
#[cfg(not(feature = "track-valgrind"))]
mod imp {
    use core::ptr::NonNull;

    pub fn allocate(ptr: NonNull<[u8]>, red_zone: usize, is_zeroed: bool) {
        let _ = (ptr, red_zone, is_zeroed);
    }

    pub fn deallocate(ptr: NonNull<u8>, red_zone: usize) {
        let _ = (ptr, red_zone);
    }

    pub fn undefined(ptr: NonNull<u8>, size: usize) {
        let _ = (ptr, size);
    }

    pub fn defined(ptr: NonNull<u8>, size: usize) {
        let _ = (ptr, size);
    }

    pub fn no_access(ptr: NonNull<u8>, size: usize) {
        let _ = (ptr, size);
    }
}
pub use self::imp::*;
