//! The module of base allocators.
//!
//! See [`BaseAlloc`] for more information.

#[cfg(feature = "base-mmap")]
mod mmap;
#[cfg(feature = "base-static")]
mod static_;

use core::{alloc::Layout, ptr::NonNull};

#[cfg(feature = "base-mmap")]
pub use self::mmap::Mmap;
#[cfg(feature = "base-static")]
pub use self::static_::Static;
use crate::arena::SLAB_SIZE;

/// A static memory handle, unable to be deallocated any longer.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StaticHandle;

/// The trait of base allocators.
///
/// To support `no_std` features and various application scenarios, ferroc
/// serves itself and a middleware between the user and its base allocators.
///
/// Base allocators are generally allocating memory at a coarser granularity,
/// usually page-aligned.
///
/// The default implementations of `BaseAlloc` in this crate are [`Mmap`] backed
/// by [a Rust mmap interface](https://github.com/darfink/region-rs), and
/// [`Static`] backed by manual & static memory allocations.
///
/// # Safety
///
/// `allocate` must return a valid & free memory block containing `layout`,
/// zeroed if `IS_ZEROED`, if possible.
pub unsafe trait BaseAlloc: Sized {
    /// Indicates if the base allocator are returning zeroed allocations by
    /// default.
    const IS_ZEROED: bool;

    /// The opaque handle of this allocator, usually its metadata or for RAII
    /// purposes.
    type Handle;
    /// The errors of the base allocator.
    type Error: BaseError;

    /// Allocate a memory [`Chunk`] of `layout`.
    ///
    /// `commit` indicates whether the chunk should be committed right after the
    /// allocation and don't need to be [`commit`](BaseAlloc::commit)ted again.
    fn allocate(&self, layout: Layout, commit: bool) -> Result<Chunk<Self>, Self::Error>;

    /// Deallocate a memory [`Chunk`].
    ///
    /// Note that this function doesn't contain a receiver argument, since its
    /// additional information should be contained in the
    /// [`handle`](Chunk::handle) of the chunk.
    ///
    /// # Safety
    ///
    /// - `chunk` must point to a valid & owned memory block containing
    ///   `layout`, previously allocated by this allocator.
    /// - `chunk` must not be used any longer after the deallocation.
    unsafe fn deallocate(chunk: &mut Chunk<Self>);

    /// Commit a block of memory in a memory chunk previously allocated by this
    /// allocator.
    ///
    /// # Safety
    ///
    /// `ptr` must point to a block of memory in a memory chunk previously
    /// allocated by this allocator.
    unsafe fn commit(&self, ptr: NonNull<[u8]>) -> Result<(), Self::Error> {
        let _ = ptr;
        Ok(())
    }

    /// Decommit a block of memory in a memory chunk previously allocated by
    /// this allocator.
    ///
    /// # Errors
    ///
    /// This function will return an error if the decommission failed.
    ///
    /// # Safety
    ///
    /// `ptr` must point to a block of memory whose content is no longer used.
    unsafe fn decommit(&self, ptr: NonNull<[u8]>) {
        let _ = ptr;
    }
}

/// A marker trait depends on what features enabled:
///
/// - `error-log`: [`core::fmt::Display`];
pub trait BaseError {}

#[cfg(not(feature = "error-log"))]
impl<T> BaseError for T {}

#[cfg(feature = "error-log")]
impl<T> BaseError for T where T: core::fmt::Display {}

/// An owned representation of a valid memory block. Implementations like
/// `Clone` and `Copy` are banned for its unique ownership.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct Chunk<B: BaseAlloc> {
    ptr: NonNull<u8>,
    layout: Layout,
    /// The handle/metadata given from the base allocator that owns this chunk.
    /// Used for base allocators to deallocate or modify information of it.
    pub handle: B::Handle,
}

unsafe impl<B: BaseAlloc> Send for Chunk<B> where B::Handle: Send {}
unsafe impl<B: BaseAlloc> Sync for Chunk<B> where B::Handle: Sync {}

impl<B: BaseAlloc> Chunk<B> {
    /// Creates a memory chunk manually. This function should only be used by an
    /// implementation of a base allocator.
    ///
    /// # Panics
    ///
    /// This function panics if the alignment of the layout is less then
    /// [`SLAB_SIZE`].
    ///
    /// # Safety
    ///
    /// `ptr` must point to a valid & owned block of memory of `layout`, and
    /// must be allocated from `base`.
    pub const unsafe fn new(ptr: NonNull<u8>, layout: Layout, handle: B::Handle) -> Self {
        assert!(
            layout.align() <= SLAB_SIZE,
            "the alignment must be greater than ferroc::arena::SLAB_SIZE",
        );
        Chunk { ptr, layout, handle }
    }

    /// Creates a static memory chunk.
    ///
    /// # Safety
    ///
    /// `ptr` must point to a valid, owned & static block of memory of
    /// `layout`.
    pub const unsafe fn from_static(ptr: NonNull<u8>, layout: Layout) -> Self
    where
        B: BaseAlloc<Handle = StaticHandle>,
    {
        // SAFETY: `ptr` points to a valid, owned block of memory of `layout`.
        unsafe { Self::new(ptr, layout, StaticHandle) }
    }

    /// Retrieves the layout information of this chunk.
    pub fn layout(&self) -> Layout {
        self.layout
    }

    /// Retrieves the pointer of this chunk.
    pub fn pointer(&self) -> NonNull<[u8]> {
        NonNull::from_raw_parts(self.ptr.cast(), self.layout.size())
    }
}

impl<B: BaseAlloc> Drop for Chunk<B> {
    fn drop(&mut self) {
        // SAFETY: `chunk` points to a valid & owned memory block containing `layout`,
        // previously allocated by this allocator.
        unsafe { B::deallocate(self) }
    }
}
