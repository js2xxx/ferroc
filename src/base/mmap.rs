use core::{alloc::Layout, mem::ManuallyDrop, ptr::NonNull};

use region::{Allocation, Protection};

use super::{BaseAlloc, Chunk};

/// A base allocator backed by `mmap` function series.
#[derive(Debug, Clone, Copy, Default, Hash)]
pub struct Mmap;

impl Mmap {
    /// Creates a new `mmap` base allocator.
    pub const fn new() -> Self {
        Mmap
    }
}

unsafe impl BaseAlloc for Mmap {
    const IS_ZEROED: bool = true;

    type Error = region::Error;
    type Handle = ManuallyDrop<Allocation>;

    fn allocate(&self, layout: Layout, commit: bool) -> Result<Chunk<Self>, Self::Error> {
        fn round_up(addr: usize, layout: Layout) -> usize {
            (addr + layout.align() - 1) & !(layout.align() - 1)
        }
        let prot = if commit {
            Protection::READ_WRITE
        } else {
            Protection::NONE
        };
        // let prot = Protection::READ_WRITE;

        let mut trial = region::alloc(layout.size(), prot)?;
        if trial.as_ptr::<()>().is_aligned_to(layout.align()) {
            let ptr = NonNull::new(trial.as_mut_ptr()).unwrap();
            // SAFETY: `Chunk` is allocated from self.
            return Ok(unsafe { Chunk::new(ptr, layout, ManuallyDrop::new(trial)) });
        }

        drop(trial);
        let mut a = region::alloc(layout.size() + layout.align(), prot)?;
        let ptr = NonNull::new(a.as_mut_ptr::<u8>().map_addr(|addr| round_up(addr, layout)));

        // SAFETY: `Chunk` is allocated from self.
        Ok(unsafe { Chunk::new(ptr.unwrap(), layout, ManuallyDrop::new(a)) })
    }

    unsafe fn deallocate(chunk: &mut Chunk<Self>) {
        unsafe { ManuallyDrop::drop(&mut chunk.handle) }
    }

    unsafe fn commit(&self, ptr: NonNull<[u8]>) -> Result<(), Self::Error> {
        let (ptr, len) = ptr.to_raw_parts();
        // SAFETY: The corresponding memory area is going to be used.
        unsafe { region::protect(ptr.as_ptr(), len, Protection::READ_WRITE) }
    }

    unsafe fn decommit(&self, ptr: NonNull<[u8]>) {
        let (ptr, len) = ptr.to_raw_parts();
        #[cfg(all(unix, feature = "libc"))]
        {
            libc::madvise(ptr.as_ptr().cast(), len, libc::MADV_DONTNEED);
        }
        // SAFETY: The corresponding memory area is going to be disposed.
        let _ = unsafe { region::protect(ptr.as_ptr(), len, Protection::NONE) };
    }
}
