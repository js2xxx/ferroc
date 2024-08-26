#[cfg(feature = "base-baremetal")]
mod bare;
#[cfg(feature = "base-mmap")]
mod mmap;

use core::{alloc::Layout, ptr::NonNull};

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
    /// `chunk` must point to a valid & owned memory block containing `layout`,
    /// previously allocated by this allocator.
    unsafe fn deallocate(chunk: &mut Chunk<Self>);
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
