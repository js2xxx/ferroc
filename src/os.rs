#[cfg(feature = "std")]
pub mod mmap;

use core::{alloc::Layout, fmt, ptr::NonNull};

/// # Safety
///
/// `allocate` must return a valid & free memory block containing `layout`,
/// zeroed if `IS_ZEROED`, if possible.
pub unsafe trait OsAlloc: Clone {
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
pub struct Chunk<Os: OsAlloc> {
    ptr: NonNull<u8>,
    layout: Layout,
    pub handle: Os::Handle,
}

unsafe impl<Os: OsAlloc> Send for Chunk<Os> where Os::Handle: Send {}
unsafe impl<Os: OsAlloc> Sync for Chunk<Os> where Os::Handle: Sync {}

impl<Os: OsAlloc> Chunk<Os> {
    /// # Safety
    ///
    /// `ptr` must points to a valid & owned block of memory of `layout`, and
    /// must be allocated from `os`.
    pub unsafe fn new(ptr: NonNull<u8>, layout: Layout, handle: Os::Handle) -> Self {
        Chunk { ptr, layout, handle }
    }

    pub fn layout(&self) -> Layout {
        self.layout
    }

    pub fn pointer(&self) -> NonNull<[u8]> {
        NonNull::slice_from_raw_parts(self.ptr, self.layout.size())
    }
}

impl<Os: OsAlloc> Drop for Chunk<Os> {
    fn drop(&mut self) {
        // SAFETY: `chunk` points to a valid & owned memory block containing `layout`,
        // previously allocated by this allocator.
        unsafe { Os::deallocate(self) }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ChunkAddr(u32);

impl fmt::Debug for ChunkAddr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ChunkAddr({:#x})", self.0)
    }
}
