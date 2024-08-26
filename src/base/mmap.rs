use core::{alloc::Layout, mem::ManuallyDrop, ptr::NonNull};

use region::{Allocation, Protection};

use super::{BaseAlloc, Chunk};

#[derive(Debug, Clone, Copy, Default, Hash)]
pub struct MmapAlloc;

impl MmapAlloc {
    pub const fn new() -> Self {
        MmapAlloc
    }
}

unsafe impl BaseAlloc for MmapAlloc {
    const IS_ZEROED: bool = true;

    type Error = region::Error;
    type Handle = ManuallyDrop<Allocation>;

    fn allocate(self, layout: Layout) -> Result<Chunk<Self>, Self::Error> {
        fn round_up(addr: usize, layout: Layout) -> usize {
            (addr + layout.align() - 1) & !(layout.align() - 1)
        }

        let mut trial = region::alloc(layout.size(), Protection::READ_WRITE)?;
        if trial.as_ptr::<()>().is_aligned_to(layout.align()) {
            let ptr = NonNull::new(trial.as_mut_ptr()).unwrap();
            // SAFETY: `Chunk` is allocated from self.
            return Ok(unsafe { Chunk::new(ptr, layout, ManuallyDrop::new(trial)) });
        }

        drop(trial);
        let mut a = region::alloc(layout.size() + layout.align(), Protection::READ_WRITE)?;
        let ptr = NonNull::new(a.as_mut_ptr::<u8>().map_addr(|addr| round_up(addr, layout)));

        // SAFETY: `Chunk` is allocated from self.
        Ok(unsafe { Chunk::new(ptr.unwrap(), layout, ManuallyDrop::new(a)) })
    }

    unsafe fn deallocate(chunk: &mut Chunk<Self>) {
        unsafe { ManuallyDrop::drop(&mut chunk.handle) }
    }
}
