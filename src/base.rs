#[cfg(feature = "base-baremetal")]
mod bare;
#[cfg(feature = "base-mmap")]
mod mmap;

use core::{
    alloc::{AllocError, Allocator, Layout},
    mem::ManuallyDrop,
    ptr::NonNull,
};

#[cfg(feature = "base-baremetal")]
pub use self::bare::BareMetal;
#[cfg(feature = "base-mmap")]
pub use self::mmap::MmapAlloc;

/// A static memory handle, unable to be deallocated any longer.
pub struct Static;

/// # Safety
///
/// `allocate` must return a valid & free memory block containing `layout`,
/// zeroed if `IS_ZEROED`, if possible.
pub unsafe trait BaseAlloc: Clone {
    const IS_ZEROED: bool;

    type Handle;
    type Error;

    fn allocate(self, layout: Layout) -> Result<Chunk<Self>, Self::Error>;

    /// # Safety
    ///
    /// - `chunk` must point to a valid & owned memory block containing
    ///   `layout`, previously allocated by this allocator.
    /// - `chunk` must not be used any longer after the deallocation.
    unsafe fn deallocate(chunk: &mut Chunk<Self>);
}

// SAFETY: Any `Allocator` is a valid `BaseAlloc`.
unsafe impl<A: Allocator + Clone> BaseAlloc for A {
    /// Regardless of whether the allocator is zeroed by default.
    const IS_ZEROED: bool = false;

    type Handle = ManuallyDrop<Self>;

    type Error = AllocError;

    fn allocate(self, layout: Layout) -> Result<Chunk<Self>, Self::Error> {
        let ptr = Allocator::allocate(&self, layout)?;
        Ok(unsafe { Chunk::new(ptr.cast(), layout, ManuallyDrop::new(self)) })
    }

    unsafe fn deallocate(chunk: &mut Chunk<Self>) {
        let ptr = chunk.pointer().cast();
        chunk.handle.deallocate(ptr, chunk.layout());
        ManuallyDrop::drop(&mut chunk.handle);
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Zeroed<A: Allocator>(pub A);

unsafe impl<A: Allocator + Clone> BaseAlloc for Zeroed<A> {
    const IS_ZEROED: bool = true;

    type Handle = ManuallyDrop<A>;

    type Error = AllocError;

    fn allocate(self, layout: Layout) -> Result<Chunk<Self>, Self::Error> {
        let ptr = Allocator::allocate_zeroed(&self.0, layout)?;
        Ok(unsafe { Chunk::new(ptr.cast(), layout, ManuallyDrop::new(self.0)) })
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
    /// # Safety
    ///
    /// `ptr` must points to a valid & owned block of memory of `layout`, and
    /// must be allocated from `base`.
    pub unsafe fn new(ptr: NonNull<u8>, layout: Layout, handle: B::Handle) -> Self {
        Chunk { ptr, layout, handle }
    }

    /// # Safety
    ///
    /// `ptr` must points to a valid, owned & static block of memory of
    /// `layout`.
    pub unsafe fn from_static(ptr: NonNull<u8>, layout: Layout) -> Self
    where
        B: BaseAlloc<Handle = Static>,
    {
        Self::new(ptr, layout, Static)
    }

    pub fn layout(&self) -> Layout {
        self.layout
    }

    pub fn pointer(&self) -> NonNull<[u8]> {
        NonNull::slice_from_raw_parts(self.ptr, self.layout.size())
    }

    // pub fn into_static(self)
}

impl<B: BaseAlloc> Drop for Chunk<B> {
    fn drop(&mut self) {
        // SAFETY: `chunk` points to a valid & owned memory block containing `layout`,
        // previously allocated by this allocator.
        unsafe { B::deallocate(self) }
    }
}
