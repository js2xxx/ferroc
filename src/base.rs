//! The module of base allocators.
//!
//! See [`BaseAlloc`] for more information.

#[cfg(feature = "base-mmap")]
mod mmap;
#[cfg(feature = "base-static")]
mod static_;

use core::{
    alloc::{AllocError, Allocator, Layout},
    mem::ManuallyDrop,
    ptr::NonNull,
};

#[cfg(feature = "base-mmap")]
pub use self::mmap::Mmap;
#[cfg(feature = "base-static")]
pub use self::static_::Static;

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
    type Error;

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

// SAFETY: Any `Allocator` is a valid `BaseAlloc`.
unsafe impl<A: Allocator + Clone> BaseAlloc for A {
    /// Regardless of whether the allocator is zeroed by default.
    const IS_ZEROED: bool = false;

    type Handle = ManuallyDrop<Self>;

    type Error = AllocError;

    fn allocate(&self, layout: Layout, _commit: bool) -> Result<Chunk<Self>, Self::Error> {
        let ptr = Allocator::allocate(self, layout)?;
        Ok(unsafe { Chunk::new(ptr.cast(), layout, ManuallyDrop::new(self.clone())) })
    }

    unsafe fn deallocate(chunk: &mut Chunk<Self>) {
        let ptr = chunk.pointer().cast();
        chunk.handle.deallocate(ptr, chunk.layout());
        ManuallyDrop::drop(&mut chunk.handle);
    }
}

/// A zeroed base allocator wrapper of an [`Allocator`].
///
/// The allocated memory chunk of this structure is always zeroed.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Zeroed<A: Allocator>(pub A);

unsafe impl<A: Allocator + Clone> BaseAlloc for Zeroed<A> {
    const IS_ZEROED: bool = true;

    type Handle = ManuallyDrop<A>;

    type Error = AllocError;

    fn allocate(&self, layout: Layout, _commit: bool) -> Result<Chunk<Self>, Self::Error> {
        let ptr = Allocator::allocate_zeroed(&self.0, layout)?;
        Ok(unsafe { Chunk::new(ptr.cast(), layout, ManuallyDrop::new(self.0.clone())) })
    }

    unsafe fn deallocate(chunk: &mut Chunk<Self>) {
        let ptr = chunk.pointer().cast();
        chunk.handle.deallocate(ptr, chunk.layout());
        ManuallyDrop::drop(&mut chunk.handle);
    }
}

/// An owned representation of a valid memory block. Implementations like
/// `Clone` and `Copy` are banned for its unique ownership.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct Chunk<B: BaseAlloc> {
    ptr: NonNull<u8>,
    layout: Layout,
    pub handle: B::Handle,
}

unsafe impl<B: BaseAlloc> Send for Chunk<B> where B::Handle: Send {}
unsafe impl<B: BaseAlloc> Sync for Chunk<B> where B::Handle: Sync {}

impl<B: BaseAlloc> Chunk<B> {
    /// Creates a memory chunk manually. This function should only be used by an
    /// implementation of a base allocator.
    ///
    /// # Safety
    ///
    /// `ptr` must points to a valid & owned block of memory of `layout`, and
    /// must be allocated from `base`.
    pub unsafe fn new(ptr: NonNull<u8>, layout: Layout, handle: B::Handle) -> Self {
        Chunk { ptr, layout, handle }
    }

    /// Creates a static memory chunk.
    ///
    /// # Safety
    ///
    /// `ptr` must points to a valid, owned & static block of memory of
    /// `layout`.
    pub unsafe fn from_static(ptr: NonNull<u8>, layout: Layout) -> Self
    where
        B: BaseAlloc<Handle = StaticHandle>,
    {
        Self::new(ptr, layout, StaticHandle)
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
