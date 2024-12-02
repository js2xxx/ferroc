use core::{alloc::Layout, mem::ManuallyDrop, ptr::NonNull};

use memmap2::MmapMut;

use super::{BaseAlloc, Chunk};

/// A base allocator backed by [a Rust mmap interface](https://crates.io/crates/memmap2).
#[derive(Debug, Clone, Copy, Default, Hash)]
pub struct Mmap;

impl Mmap {
    /// Creates a new `mmap` base allocator.
    pub const fn new() -> Self {
        Mmap
    }

    /// Returns the memory page size of the current platform.
    pub fn page_size(&self) -> usize {
        unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
    }
}

unsafe impl BaseAlloc for Mmap {
    const IS_ZEROED: bool = true;

    // The inner type of `Errno` is `i32`, which is incompatible for UEFI targets.
    // However, the `mmap` functionalities aren't available on UEFI either, so this
    // should be fine.
    type Error = errno::Errno;

    type Handle = ManuallyDrop<MmapMut>;

    fn allocate(&self, layout: Layout, commit: bool) -> Result<Chunk<Self>, Self::Error> {
        fn round_up(addr: usize, layout: Layout) -> usize {
            (addr + layout.align() - 1) & !(layout.align() - 1)
        }

        let layout = layout.pad_to_align();
        let mut options = memmap2::MmapOptions::new();
        if cfg!(not(miri)) && commit {
            options.populate();
        }

        let mut mmap = |len| {
            let res = options.len(len).map_anon();
            res.map_err(|err| errno::Errno(err.raw_os_error().unwrap()))
        };

        let mut trial = mmap(layout.size())?;
        if trial.as_ptr().is_aligned_to(layout.align()) {
            let ptr = NonNull::new(trial.as_mut_ptr()).unwrap();
            // SAFETY: `Chunk` is allocated from self.
            return Ok(unsafe { Chunk::new(ptr, layout, ManuallyDrop::new(trial)) });
        }
        drop(trial);

        let mut a = mmap(layout.size() + layout.align())?;
        let ptr = NonNull::new(a.as_mut_ptr().map_addr(|addr| round_up(addr, layout)));

        // SAFETY: `Chunk` is allocated from self.
        Ok(unsafe { Chunk::new(ptr.unwrap(), layout, ManuallyDrop::new(a)) })
    }

    unsafe fn deallocate(chunk: &mut Chunk<Self>) {
        unsafe { ManuallyDrop::drop(&mut chunk.handle) }
    }

    #[cfg(all(unix, not(miri)))]
    unsafe fn commit(&self, ptr: NonNull<[u8]>) -> Result<(), Self::Error> {
        let (ptr, len) = ptr.to_raw_parts();
        #[cfg(not(target_os = "macos"))]
        let flags = libc::MADV_WILLNEED | libc::MADV_HUGEPAGE;
        #[cfg(target_os = "macos")]
        let flags = libc::MADV_WILLNEED;
        // SAFETY: The corresponding memory area is going to be used.
        match unsafe { libc::madvise(ptr.as_ptr().cast(), len, flags) } {
            0 => Ok(()),
            _ => Err(errno::errno()),
        }
    }

    #[cfg(all(unix, not(miri)))]
    unsafe fn decommit(&self, ptr: NonNull<[u8]>) {
        let (ptr, len) = ptr.to_raw_parts();
        // SAFETY: The corresponding memory area is going to be disposed.
        unsafe { libc::madvise(ptr.as_ptr().cast(), len, libc::MADV_DONTNEED) };
    }
}
