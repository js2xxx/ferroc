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

// Set as `std::io::RawOsError` so as to get rid of direct `std` dependency.
#[cfg(not(target_os = "uefi"))]
type RawOsError = i32;
#[cfg(target_os = "uefi")]
type RawOsError = usize;

unsafe impl BaseAlloc for Mmap {
    const IS_ZEROED: bool = true;

    type Error = RawOsError;

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
        let mut trial = options
            .len(layout.size())
            .map_anon()
            .map_err(|err| err.raw_os_error().unwrap())?;
        if trial.as_ptr().is_aligned_to(layout.align()) {
            let ptr = NonNull::new(trial.as_mut_ptr()).unwrap();
            // SAFETY: `Chunk` is allocated from self.
            return Ok(unsafe { Chunk::new(ptr, layout, ManuallyDrop::new(trial)) });
        }

        drop(trial);
        let mut a = options
            .len(layout.size() + layout.align())
            .map_anon()
            .map_err(|err| err.raw_os_error().unwrap())?;
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
            _ => Err(errno()),
        }
    }

    #[cfg(all(unix, not(miri)))]
    unsafe fn decommit(&self, ptr: NonNull<[u8]>) {
        let (ptr, len) = ptr.to_raw_parts();
        // SAFETY: The corresponding memory area is going to be disposed.
        unsafe { libc::madvise(ptr.as_ptr().cast(), len, libc::MADV_DONTNEED) };
    }
}

#[cfg(all(unix, not(miri)))]
/// Returns the platform-specific value of errno
pub fn errno() -> RawOsError {
    unsafe extern "C" {
        #[cfg_attr(
            any(
                target_os = "linux",
                target_os = "emscripten",
                target_os = "fuchsia",
                target_os = "l4re",
                target_os = "hurd",
            ),
            link_name = "__errno_location"
        )]
        #[cfg_attr(
            any(
                target_os = "netbsd",
                target_os = "openbsd",
                target_os = "android",
                target_os = "redox",
                target_env = "newlib"
            ),
            link_name = "__errno"
        )]
        #[cfg_attr(
            any(target_os = "solaris", target_os = "illumos"),
            link_name = "___errno"
        )]
        #[cfg_attr(target_os = "nto", link_name = "__get_errno_ptr")]
        #[cfg_attr(
            any(
                target_os = "macos",
                target_os = "ios",
                target_os = "tvos",
                target_os = "freebsd",
                target_os = "watchos"
            ),
            link_name = "__error"
        )]
        #[cfg_attr(target_os = "haiku", link_name = "_errnop")]
        #[cfg_attr(target_os = "aix", link_name = "_Errno")]
        fn errno_location() -> *mut RawOsError;
    }
    unsafe { *errno_location() }
}
